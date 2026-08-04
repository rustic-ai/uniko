//! P4 Consolidation — derive Facts from Observations.
//!
//! A consolidation cycle:
//! 1. Queries unprocessed Observations carrying a structured triple
//!    (`subject` and `predicate` non-null).
//! 2. Groups by `(subject, predicate)`; picks a canonical object per
//!    group (cosine-clustered mode, tie-broken by recency).
//! 3. Upserts a Fact per group via
//!    [`KnowledgeBase::upsert_fact_by_triple`]; first observation in a
//!    cluster becomes the embedding source.
//! 4. Wires `SUPPORTED_BY` edges from every contributing Observation
//!    to its Fact.
//! 5. Writes a `ConsolidationCycle` audit node with `PROCESSED`,
//!    `CREATED`, and `INVOLVED` edges — the `PROCESSED` edges are the
//!    idempotency anchor so future cycles skip the same Observations.
//!
//! Object surface forms within a group are clustered by cosine
//! similarity over their document embeddings before mode-voting.  This
//! keeps near-duplicate phrasings ("adoption agencies" / "adoption
//! agency") from splitting the vote between three exact buckets and
//! letting an off-topic recency-winner take canonical.
//!
//! F38 contradiction detection: when contradicting observations within
//! a `(subject, predicate)` group exceed [`CONTRADICTION_THRESHOLD`] of
//! the total, any prior open-BTIC Fact for that pair with a different
//! object is invalidated (BTIC `hi` closed; `INVALIDATES` edge wired
//! from the new Fact).  "Different" here means *outside* the cluster
//! that contains the prior Fact's object, so paraphrase-only changes
//! don't trigger spurious invalidations.
//!
//! F39 entity drift: each invalidation records against the subject
//! Entity's `invalidation_count`; once cumulative invalidations exceed
//! [`DRIFT_THRESHOLD`], `Entity.unstable = true` so the recall cascade
//! can force Phase 2+ for queries that reference it.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};

use uniko_extract::embedding::embed_batch_chunked;
use uniko_store::Value;
use uniko_store::operations::facts::FactUpsertInput;
use uniko_store::schema::constants::{edges, labels};
use uniko_store::temporal::Btic;
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use crate::llm_triples::{ObservationInput, refine_triples};

/// Number of Facts created, reinforced, and invalidated in one cycle.
#[derive(Debug, Default, Clone, Copy)]
pub struct CycleStats {
    /// Observations processed (PROCESSED edges emitted).
    pub observations_processed: usize,
    /// Facts newly created.
    pub facts_created: usize,
    /// Facts reinforced (existing Fact whose count was incremented).
    pub facts_reinforced: usize,
    /// Facts whose BTIC interval was closed by F38 contradiction
    /// detection in this cycle.
    pub facts_invalidated: usize,
    /// Number of Entities that transitioned to `unstable = true` in
    /// this cycle (F39 drift alerts).
    pub drift_alerts: usize,
}

/// Fraction of observations within a `(subject, predicate)` group that
/// must disagree with the canonical object to trigger F38 invalidation
/// of any prior open-BTIC Facts for the same pair.  Spec §5 F38.
const CONTRADICTION_THRESHOLD: f64 = 0.40;

/// Cumulative invalidation count at or above which an Entity is flagged
/// `unstable = true` (F39).  Spec calls for "> 4 invalidations within
/// 30 days" — the windowing belongs in the store helper, this constant
/// captures only the count.
const DRIFT_THRESHOLD: i64 = 4;

/// Maximum observations to process in a single cycle.
///
/// Caps work per cycle so a long-running ingest doesn't starve other
/// agents.  Spillover is picked up on the next sweep.
const DEFAULT_BATCH_SIZE: i64 = 500;

/// Cosine similarity at or above which two object surface forms are
/// treated as paraphrases of the same canonical claim.
///
/// Tuned for BGE-small-en (the bench default).  BGE embeds short noun
/// phrases tightly — exact paraphrases land in the 0.93+ range and
/// genuine different objects ("Rust" vs "Go") sit well below 0.7, so
/// 0.88 is a safe split that collapses inflection / article variation
/// without merging distinct entities.
const COSINE_THRESHOLD: f32 = 0.88;

/// Maximum input texts per single ONNX embedding forward.
///
/// Caps activation memory per call. The unchunked path crashed the
/// ORT BFC arena at ~6k inputs requesting 1.3 GB — chunks of 64 keep
/// each forward in tens of megabytes of activations for the BGE-small
/// model used in the bench, with negligible per-call overhead at this
/// chunk size on GPU.
const EMBED_BATCH_CHUNK_SIZE: usize = 64;

/// Source of `(subject, predicate, object)` triples for grouping.
///
/// Selects whether to trust the cheap rule-based triples already on
/// the Observation node or to call an LLM at consolidation time to
/// refine them.  The LLM path costs one model call per Observation
/// per cycle but yields markedly cleaner predicates and objects.
#[derive(Debug, Clone, Default)]
pub enum TripleSource {
    /// Use the `predicate` and `object` columns P3 stored on the
    /// Observation node (SRL/DEP matcher output).  Default — no LLM
    /// dependency.
    #[default]
    SrlDep,
    /// Refine each Observation's triple via the LLM at the given
    /// alias before grouping.  Falls back to the SRL/DEP triple when
    /// the LLM declines or the response fails to parse.
    Llm {
        /// Xervo alias of the model to use for triple extraction.
        alias: String,
    },
}

/// Run one consolidation cycle for `agent_id` with the default
/// (SRL/DEP) triple source.
///
/// Idempotent across runs: Observations already wired via `PROCESSED`
/// from any prior `ConsolidationCycle` are excluded by the query.
///
/// Returns counts useful for metrics and tests.  Caller is responsible
/// for emitting `uniko.consolidation.*` metrics from the returned
/// stats; this function only writes to the graph.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on any database failure.
pub async fn run_cycle(
    kb: &KnowledgeBase,
    agent_id: &str,
    batch_size: Option<i64>,
) -> Result<CycleStats, UnikoError> {
    run_cycle_with(kb, agent_id, batch_size, &TripleSource::SrlDep).await
}

/// Run one consolidation cycle, choosing the triple source explicitly.
///
/// See [`run_cycle`] for the default-source convenience wrapper.  Use
/// this entry point when callers want LLM-refined triples (typically
/// the bench harness behind a `--extract-triples-llm-alias` flag).
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on any database failure.
pub async fn run_cycle_with(
    kb: &KnowledgeBase,
    agent_id: &str,
    batch_size: Option<i64>,
    triple_source: &TripleSource,
) -> Result<CycleStats, UnikoError> {
    let started_at = Utc::now();
    let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);

    // P4 phase profiling: cumulative elapsed since just before the first read,
    // logged at each phase boundary under target "p4_prof". Diff consecutive
    // marks to get per-phase time.
    let _prof = std::time::Instant::now();
    let phase = |name: &str| {
        tracing::debug!(
            target: "p4_prof",
            phase = name,
            total_ms = _prof.elapsed().as_millis() as u64,
            "P4 phase mark"
        );
    };

    let mut observations = kb.fetch_unprocessed_observations(batch_size).await?;
    phase("fetch_observations");

    // Optionally refine each Observation's triple via the LLM.  Done
    // before grouping so the LLM's cleaner predicates collapse near-
    // duplicates the SRL/DEP path leaves spread across distinct keys
    // (e.g. "got" vs "got_a" vs "received" all collapsing to "received").
    if let TripleSource::Llm { alias } = triple_source
        && !observations.is_empty()
    {
        let (inputs, requested) = {
            let inputs: Vec<ObservationInput> = observations
                .iter()
                .map(|o| ObservationInput {
                    node_id: o.node_id,
                    content: o.content.clone(),
                    speaker_hint: Some(o.subject.clone()),
                })
                .collect();
            let len = inputs.len();
            (inputs, len)
        };
        let refined = refine_triples(kb, alias, &inputs).await?;
        drop(inputs);
        let mut by_id: HashMap<NodeId, crate::llm_triples::LlmTriple> =
            HashMap::with_capacity(refined.len());
        for t in refined {
            by_id.insert(t.node_id, t);
        }
        let mut overwritten = 0usize;
        for obs in &mut observations {
            if let Some(t) = by_id.remove(&obs.node_id) {
                obs.subject = t.subject;
                obs.predicate = t.predicate;
                obs.object = t.object;
                overwritten += 1;
            }
        }
        tracing::info!(
            requested,
            refined = overwritten,
            dropped = requested - overwritten,
            "llm triple refinement complete",
        );
    }
    if observations.is_empty() {
        let completed_at = Utc::now();
        kb.write_consolidation_cycle(
            agent_id,
            started_at,
            completed_at,
            &[],
            &[],
            &[],
            &[],
            0,
            &[],
        )
        .await?;
        return Ok(CycleStats::default());
    }

    // Group by (subject, predicate); collect contributing observations
    // and object surface forms for canonical selection.
    let mut groups: HashMap<(String, String), GroupBuilder> = HashMap::new();
    for obs in &observations {
        // Normalize the subject in the grouping key so case/punctuation
        // variants ("Melanie" / "melanie" / "Melanie.") consolidate into a
        // single Fact instead of fragmenting. Mirrors `normalize_object`
        // for objects; idempotent on already-normalized stored subjects.
        let entry = groups
            .entry((
                uniko_store::text::normalize_canonical(&obs.subject),
                obs.predicate.clone(),
            ))
            .or_default();
        entry.contributing.push(obs.node_id);
        entry.first_observed_at = Some(min_or(entry.first_observed_at, obs.observed_at));
        if let Some(anchor) = obs.temporal_anchor {
            entry.first_temporal_anchor = Some(min_or(entry.first_temporal_anchor, anchor));
        }
        entry.object_votes.push(ObjectVote {
            text: obs.object.clone(),
            observed_at: obs.observed_at,
            content: obs.content.clone(),
        });
    }

    // Pre-pass: look up prior open Facts per group and read their
    // stored object text. Single batched query — replaces the
    // previous loop of one `find_stale_open_facts` call per group
    // (~2,800 round-trips per cycle) plus a follow-up per-Fact
    // `MATCH ... RETURN f.object` (~10-50 extra round-trips). The
    // batched helper returns the object surface form alongside the
    // BTIC so no second query is needed.
    //
    // We do this before clustering so prior objects participate in
    // the cluster assignment alongside the votes — F38 must compare
    // in cluster-space, not raw-string space.
    phase("group_observations");
    let group_keys: Vec<(String, String)> = groups.keys().cloned().collect();
    let mut group_priors: HashMap<(String, String), Vec<PriorFact>> =
        match kb.find_stale_open_facts_batched(&group_keys).await {
            Ok(map) => map
                .into_iter()
                .map(|(key, rows)| {
                    let priors: Vec<PriorFact> = rows
                        .into_iter()
                        .map(|(nid, btic, object)| PriorFact {
                            node_id: nid,
                            btic,
                            object,
                        })
                        .collect();
                    (key, priors)
                })
                .collect(),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    group_count = group_keys.len(),
                    "batched open-Fact lookup failed (continuing without F38)",
                );
                HashMap::new()
            }
        };

    // Collect every unique non-empty object surface form (votes +
    // prior Fact objects), single batch embed call.  Per-group
    // clusters are computed from a slice of this shared map below.
    let mut unique_objects: HashSet<String> = HashSet::new();
    for group in groups.values() {
        for v in &group.object_votes {
            if let Some(text) = &v.text {
                let k = normalize_object(text);
                if !k.is_empty() {
                    unique_objects.insert(k);
                }
            }
        }
    }
    for priors in group_priors.values() {
        for p in priors {
            let k = normalize_object(&p.object);
            if !k.is_empty() {
                unique_objects.insert(k);
            }
        }
    }
    let unique_vec: Vec<String> = unique_objects.into_iter().collect();
    let cluster_objects = kb.config().consolidation_cluster_objects;
    phase("find_priors_and_collect");
    let object_embeddings: HashMap<String, Vec<f32>> = if unique_vec.is_empty() || !cluster_objects
    {
        // Disabled clustering: skip the batch embed entirely.
        // `build_clusters` falls back to singleton-per-key when an
        // embedding is missing, which reproduces the legacy exact-
        // string mode/vote behavior.
        HashMap::new()
    } else {
        let refs: Vec<&str> = unique_vec.iter().map(String::as_str).collect();
        embed_or_warn(kb, &refs, "object")
            .await
            .map(|embs| unique_vec.iter().cloned().zip(embs).collect())
            .unwrap_or_default()
    };
    phase("object_embed");

    let mut created_facts: Vec<NodeId> = Vec::new();
    let mut reinforced_facts: Vec<NodeId> = Vec::new();
    let mut invalidated_facts: Vec<NodeId> = Vec::new();
    let mut drift_alerts: usize = 0;

    // Per-group preparation pass — no I/O. Collects every input the
    // batched embed and upsert phases need so they each run as a
    // single batched op, regardless of group count.
    struct FactPlan {
        subject: String,
        predicate: String,
        canonical: Option<String>,
        contributing: Vec<NodeId>,
        prior_stale: Vec<(NodeId, Btic, String)>,
        first_observed: DateTime<Utc>,
        embed_text: String,
    }

    let mut plans: Vec<FactPlan> = Vec::with_capacity(groups.len());
    for ((subject, predicate), group) in groups {
        let priors = group_priors
            .remove(&(subject.clone(), predicate.clone()))
            .unwrap_or_default();

        // Build the cluster map for this group from the shared
        // embedding cache. Includes vote surface forms AND prior Fact
        // object so F38 can compare in cluster-space.
        let mut group_keys: HashSet<String> = HashSet::new();
        for v in &group.object_votes {
            if let Some(text) = &v.text {
                let k = normalize_object(text);
                if !k.is_empty() {
                    group_keys.insert(k);
                }
            }
        }
        for p in &priors {
            let k = normalize_object(&p.object);
            if !k.is_empty() {
                group_keys.insert(k);
            }
        }
        let clusters = build_clusters(&group_keys, &object_embeddings, COSINE_THRESHOLD);

        let canonical = canonical_object(&group.object_votes, &clusters);

        // F38: identify prior open Facts whose object's cluster gets
        // outvoted in this cycle.
        let mut prior_stale: Vec<(NodeId, Btic, String)> = Vec::new();
        for prior in &priors {
            let prior_key = normalize_object(&prior.object);
            let prior_cluster = clusters.get(&prior_key).copied();
            let (total, agree) = vote_tallies(&group.object_votes, prior_cluster, &clusters);
            if total == 0 {
                continue;
            }
            let contradicting = total - agree;
            if (contradicting as f64) / (total as f64) > CONTRADICTION_THRESHOLD {
                prior_stale.push((prior.node_id, prior.btic, prior.object.clone()));
            }
        }

        let first_observed = group
            .first_temporal_anchor
            .or(group.first_observed_at)
            .unwrap_or(started_at);

        // Compose the embedding text from the canonical triple. When
        // the object slot is empty, embed the freshest contributor's
        // content as a paraphrase fallback. Optionally prepend a
        // `"%B %Y"` date prefix (e.g. `"[January 2024] "`) so
        // temporally-near Facts co-locate in the embedding space.
        let embed_text = compose_embed_text(
            first_observed,
            &subject,
            &predicate,
            canonical.as_deref(),
            freshest_content(&group.object_votes).as_deref(),
            kb.config().consolidation_date_augment_embedding,
        );

        plans.push(FactPlan {
            subject,
            predicate,
            canonical,
            contributing: group.contributing,
            prior_stale,
            first_observed,
            embed_text,
        });
    }

    // Single chunked-batched embedding call for every Fact in the
    // cycle (replaces 2 874+ sequential `embed_document` calls).  A
    // partial / failed batch degrades to storing Facts without an
    // embedding — same fallback the per-group path used.
    phase("build_plans_clusters");
    let embed_texts: Vec<&str> = plans.iter().map(|p| p.embed_text.as_str()).collect();
    let fact_embeddings: Vec<Option<Vec<f32>>> = embed_or_warn(kb, &embed_texts, "fact")
        .await
        .map(|embs| embs.into_iter().map(Some).collect())
        .unwrap_or_else(|| vec![None; plans.len()]);
    phase("fact_embed");

    // Batched fact upsert: one batched MATCH, one batched CREATE for
    // new facts, one batched UPDATE for reinforced facts. Replaces
    // the per-fact `upsert_fact_by_triple` loop.
    let upsert_inputs: Vec<FactUpsertInput> = plans
        .iter()
        .zip(fact_embeddings.iter().cloned())
        .map(|(plan, embedding)| FactUpsertInput {
            subject: plan.subject.clone(),
            predicate: plan.predicate.clone(),
            object: plan.canonical.clone(),
            observation_count: plan.contributing.len() as i64,
            observed_at: plan.first_observed,
            embedding,
        })
        .collect();
    let upserts = kb.batch_upsert_facts(upsert_inputs).await?;
    phase("fact_upsert");

    // Collect every SUPPORTED_BY edge across the entire cycle into one
    // batched call. Idempotency invariant from `attach_supported_by`
    // is preserved: cycles only process observations with no inbound
    // PROCESSED edge from prior cycles, so each (fact, obs) pair is
    // wired exactly once.
    // 1c: weight each SUPPORTED_BY edge by cosine(Fact embedding,
    // Observation content embedding) so downstream PPR (and any consumer
    // of per-edge support strength) can favour strongly-supporting
    // evidence over weak paraphrase matches. Falls back to 1.0 — the prior
    // uniform behaviour — whenever either embedding is absent (e.g. the
    // embedding model was unavailable this cycle), so the edge is never
    // dropped or zero-weighted by accident.
    let contributing_ids: Vec<NodeId> = plans
        .iter()
        .flat_map(|p| p.contributing.iter().copied())
        .collect();
    let obs_embeddings = fetch_observation_embeddings(kb, &contributing_ids).await;
    phase("obs_embed_read");

    let mut all_supported_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = Vec::new();
    for (idx, (plan, up)) in plans.iter().zip(upserts.iter()).enumerate() {
        let fact_emb = fact_embeddings.get(idx).and_then(Option::as_ref);
        for &obs_nid in &plan.contributing {
            let weight = supported_by_weight(
                fact_emb.map(Vec::as_slice),
                obs_embeddings.get(&obs_nid).map(Vec::as_slice),
            );
            let mut props = HashMap::with_capacity(1);
            props.insert("weight".into(), Value::Float(weight));
            all_supported_edges.push((up.node_id, obs_nid, props));
        }
    }
    if !all_supported_edges.is_empty() {
        kb.batch_create_edges_fast(
            edges::SUPPORTED_BY,
            Some(labels::FACT),
            Some(labels::OBSERVATION),
            &all_supported_edges,
        )
        .await?;
    }
    phase("supported_by_edges");

    // Classify upserts and drive invalidations. Invalidations are
    // still per-stale (the contradiction path is rare and each one
    // also bumps the Entity's drift counter, which has its own
    // read-modify-write semantics inside the helper).
    for (plan, up) in plans.iter().zip(upserts.iter()) {
        if up.was_created {
            created_facts.push(up.node_id);
        } else {
            reinforced_facts.push(up.node_id);
        }
        if plan.prior_stale.is_empty() {
            continue;
        }
        let now = Utc::now();
        for (stale_nid, stale_btic, _stale_obj) in &plan.prior_stale {
            if *stale_nid == up.node_id {
                continue;
            }
            if let Err(e) = kb
                .invalidate_fact(
                    *stale_nid,
                    stale_btic,
                    now,
                    Some(up.node_id),
                    Some("consolidation contradiction"),
                )
                .await
            {
                tracing::warn!(
                    error = %e,
                    stale_fact = stale_nid,
                    subject = %plan.subject,
                    predicate = %plan.predicate,
                    "fact invalidation failed (continuing)",
                );
                continue;
            }
            invalidated_facts.push(*stale_nid);

            match kb
                .record_entity_invalidation(&plan.subject, now, DRIFT_THRESHOLD)
                .await
            {
                Ok(true) => {
                    drift_alerts += 1;
                    tracing::info!(
                        subject = %plan.subject,
                        "entity transitioned to unstable",
                    );
                }
                Ok(false) => {}
                Err(e) => tracing::warn!(
                    error = %e,
                    subject = %plan.subject,
                    "drift accounting failed (continuing)",
                ),
            }
        }
    }

    phase("invalidations");
    let processed_ids: Vec<NodeId> = observations.iter().map(|o| o.node_id).collect();
    let completed_at = Utc::now();
    kb.write_consolidation_cycle(
        agent_id,
        started_at,
        completed_at,
        &processed_ids,
        &created_facts,
        &reinforced_facts,
        &invalidated_facts,
        drift_alerts as i64,
        &[],
    )
    .await?;
    phase("write_cycle_audit");

    Ok(CycleStats {
        observations_processed: processed_ids.len(),
        facts_created: created_facts.len(),
        facts_reinforced: reinforced_facts.len(),
        facts_invalidated: invalidated_facts.len(),
        drift_alerts,
    })
}

/// Chunked-batched embed with uniform "warn and fall back" behaviour.
///
/// Returns `Some(embeddings)` only when the batch yields exactly one
/// vector per input.  Length mismatches and call failures both log at
/// `warn` and return `None`, letting callers degrade gracefully (the
/// object-clustering path falls back to string-exact dedup; the fact
/// path stores Facts without an embedding).
///
/// `purpose` is interpolated into the warning so the log identifies
/// which call site degraded (e.g. `"object"` vs `"fact"`).
async fn embed_or_warn(
    kb: &KnowledgeBase,
    inputs: &[&str],
    purpose: &str,
) -> Option<Vec<Vec<f32>>> {
    let doc_prefix = kb.config().embedding.document_prefix.clone();
    // Chunked to avoid the BFC arena OOM seen with single-call batches
    // at ~6k inputs (~1.3 GB activation buffer for BGE).
    match embed_batch_chunked(kb, inputs, doc_prefix.as_deref(), EMBED_BATCH_CHUNK_SIZE).await {
        Ok(embs) if embs.len() == inputs.len() => Some(embs),
        Ok(embs) => {
            tracing::warn!(
                purpose,
                requested = inputs.len(),
                returned = embs.len(),
                "embedding batch length mismatch; falling back",
            );
            None
        }
        Err(e) => {
            tracing::warn!(purpose, error = %e, "embedding batch failed; falling back");
            None
        }
    }
}

/// `min(prev, new)` when `prev` is set, otherwise `new`.  Tiny helper
/// for the per-group "earliest seen" reductions in `run_cycle`.
fn min_or<T: Ord>(prev: Option<T>, new: T) -> T {
    match prev {
        Some(p) => std::cmp::min(p, new),
        None => new,
    }
}

/// Canonical normalization for clustering keys: trim + lowercase.
///
/// Keeps surface-form variations (e.g. casing, surrounding whitespace)
/// from producing distinct cluster keys before semantic clustering even
/// has a chance to dedupe them.
fn normalize_object(text: &str) -> String {
    text.trim().to_ascii_lowercase()
}

/// Cosine similarity over two equal-length float vectors.
///
/// Returns 0.0 when either input is zero-norm; mismatched lengths are
/// truncated to the shorter prefix (callers should pass vectors from
/// the same embedding model so lengths always match in practice).
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Support-strength weight for a `SUPPORTED_BY` edge (1c): the cosine of
/// the Fact and Observation embeddings, clamped to `[0, 1]`. Falls back to
/// `1.0` — the prior uniform behaviour — when either embedding is absent,
/// so an edge is never dropped or zero-weighted by accident.
fn supported_by_weight(fact_emb: Option<&[f32]>, obs_emb: Option<&[f32]>) -> f64 {
    match (fact_emb, obs_emb) {
        (Some(f), Some(o)) => f64::from(cosine_sim(f, o)).clamp(0.0, 1.0),
        _ => 1.0,
    }
}

/// Batched read of `Observation.embedding` for the given node ids
/// (auto-embedded on `content` at ingest). Ids absent from the result —
/// because the column is null or the read fails — simply don't appear in
/// the map, so [`run_cycle_with`]'s SUPPORTED_BY weighting falls back to
/// `1.0` for them. Read-only, runs on a plain session (no tx needed).
async fn fetch_observation_embeddings(
    kb: &KnowledgeBase,
    ids: &[NodeId],
) -> HashMap<NodeId, Vec<f32>> {
    // The query lives in `uniko_store` (the sole uni-db boundary); the
    // degrade-on-failure policy stays here, where the fallback weight is
    // defined.
    kb.fetch_observation_embeddings(ids)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(
                error = %e,
                "fetch_observation_embeddings failed; SUPPORTED_BY weights fall back to 1.0"
            );
            HashMap::new()
        })
}

/// Assign each normalized key to a cluster id via single-pass greedy
/// agglomeration over cosine similarity.
///
/// Iterates keys in deterministic sorted order so the cluster ids are
/// stable across runs with the same input set.  Centroids are running
/// means updated each time a key joins.  When a key has no embedding
/// in the cache (or the cache itself is empty), it's placed in its
/// own singleton cluster — that gives string-exact behavior as a
/// natural fallback when embedding is unavailable.
fn build_clusters(
    keys: &HashSet<String>,
    embeddings: &HashMap<String, Vec<f32>>,
    threshold: f32,
) -> HashMap<String, usize> {
    let mut sorted: Vec<&String> = keys.iter().collect();
    sorted.sort();

    let mut centroids: Vec<Vec<f32>> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut out: HashMap<String, usize> = HashMap::with_capacity(keys.len());

    for key in sorted {
        let Some(emb) = embeddings.get(key) else {
            // No embedding available — singleton cluster.
            let id = centroids.len();
            centroids.push(Vec::new());
            counts.push(1);
            out.insert(key.clone(), id);
            continue;
        };

        let mut best: Option<(usize, f32)> = None;
        for (i, c) in centroids.iter().enumerate() {
            if c.is_empty() {
                continue;
            }
            let s = cosine_sim(emb, c);
            if s > best.map(|(_, x)| x).unwrap_or(f32::MIN) {
                best = Some((i, s));
            }
        }

        match best {
            Some((i, s)) if s >= threshold => {
                let n = counts[i] as f32;
                let centroid = &mut centroids[i];
                for (j, x) in emb.iter().enumerate() {
                    if j < centroid.len() {
                        centroid[j] = (centroid[j] * n + x) / (n + 1.0);
                    }
                }
                counts[i] += 1;
                out.insert(key.clone(), i);
            }
            _ => {
                let id = centroids.len();
                centroids.push(emb.clone());
                counts.push(1);
                out.insert(key.clone(), id);
            }
        }
    }
    out
}

/// Tally `(total, agree)` votes for F38, in cluster space.
///
/// `total` counts every contributing observation with a non-empty
/// object string.  `agree` counts those whose object falls in the same
/// cluster as the prior Fact's object (`canonical_cluster`).  When no
/// prior cluster is supplied (e.g. prior object was empty), `agree`
/// is always 0 — caller will see contradicting = total and apply the
/// threshold accordingly.
fn vote_tallies(
    votes: &[ObjectVote],
    canonical_cluster: Option<usize>,
    clusters: &HashMap<String, usize>,
) -> (usize, usize) {
    let mut total = 0usize;
    let mut agree = 0usize;
    for vote in votes {
        let Some(text) = vote.text.as_ref() else {
            continue;
        };
        let key = normalize_object(text);
        if key.is_empty() {
            continue;
        }
        total += 1;
        if let (Some(target), Some(&cid)) = (canonical_cluster, clusters.get(&key))
            && cid == target
        {
            agree += 1;
        }
    }
    (total, agree)
}

/// Pick the canonical object text for a `(subject, predicate)` cluster.
///
/// Mode over **clusters** (not raw strings); within the winning cluster,
/// mode over the surface forms; ties broken by the most recent
/// `observed_at`.  Returns `None` when no contributing Observation had
/// an object slot.
fn canonical_object(votes: &[ObjectVote], clusters: &HashMap<String, usize>) -> Option<String> {
    let mut cluster_count: HashMap<usize, (usize, DateTime<Utc>)> = HashMap::new();
    let mut surface_count: HashMap<(usize, String), (usize, DateTime<Utc>)> = HashMap::new();

    for vote in votes {
        let Some(text) = vote.text.as_ref() else {
            continue;
        };
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        let key = trimmed.to_ascii_lowercase();
        let Some(&cid) = clusters.get(&key) else {
            continue;
        };

        let ce = cluster_count.entry(cid).or_insert((0, vote.observed_at));
        ce.0 += 1;
        if vote.observed_at > ce.1 {
            ce.1 = vote.observed_at;
        }

        let se = surface_count
            .entry((cid, trimmed.to_string()))
            .or_insert((0, vote.observed_at));
        se.0 += 1;
        if vote.observed_at > se.1 {
            se.1 = vote.observed_at;
        }
    }

    let (winning_cluster, _) = cluster_count
        .into_iter()
        .max_by(|a, b| a.1.0.cmp(&b.1.0).then_with(|| a.1.1.cmp(&b.1.1)))?;

    surface_count
        .into_iter()
        .filter(|((cid, _), _)| *cid == winning_cluster)
        .max_by(|a, b| a.1.0.cmp(&b.1.0).then_with(|| a.1.1.cmp(&b.1.1)))
        .map(|((_, text), _)| text)
}

/// Compose the text that gets embedded for a Fact.
///
/// Canonical path: `"{subject} {predicate} {object}"`.  Fallback path
/// (no canonical object): the freshest contributor's `content`, else
/// `"{subject} {predicate}"`.  When `date_augment` is `true`, a
/// `"[Month Year] "` prefix derived from `first_observed` is prepended
/// to the result — pulls temporally-near Facts together in the
/// embedding space at the cost of one chrono format call.
///
/// Asymmetric on purpose: queries are not augmented this way.  Uniko's
/// Phase-2 temporal channel already handles query-side temporal cues
/// via BTIC overlap, so the prefix only needs to bias the document
/// side of the encoder.
fn compose_embed_text(
    first_observed: DateTime<Utc>,
    subject: &str,
    predicate: &str,
    canonical: Option<&str>,
    fallback_content: Option<&str>,
    date_augment: bool,
) -> String {
    let body = match canonical {
        Some(obj) => format!("{subject} {predicate} {obj}"),
        None => fallback_content
            .map(str::to_string)
            .unwrap_or_else(|| format!("{subject} {predicate}")),
    };
    if date_augment {
        let prefix = first_observed.format("%B %Y").to_string();
        format!("[{prefix}] {body}")
    } else {
        body
    }
}

/// Freshest contributing observation's `content`, used as embedding
/// fallback when no contributor produced an object slot.
fn freshest_content(votes: &[ObjectVote]) -> Option<String> {
    votes
        .iter()
        .max_by_key(|v| v.observed_at)
        .map(|v| v.content.clone())
}

#[derive(Debug, Default)]
struct GroupBuilder {
    contributing: Vec<NodeId>,
    object_votes: Vec<ObjectVote>,
    first_observed_at: Option<DateTime<Utc>>,
    first_temporal_anchor: Option<DateTime<Utc>>,
}

#[derive(Debug)]
struct ObjectVote {
    text: Option<String>,
    observed_at: DateTime<Utc>,
    content: String,
}

#[derive(Debug, Clone)]
struct PriorFact {
    node_id: NodeId,
    btic: Btic,
    object: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    /// Trivial cluster map: each distinct normalized string is its own
    /// cluster id.  Reproduces the pre-clustering string-exact behavior
    /// so the legacy mode/recency tests still pin meaningful invariants.
    fn singleton_clusters(votes: &[ObjectVote]) -> HashMap<String, usize> {
        let mut out = HashMap::new();
        for v in votes {
            if let Some(t) = &v.text {
                let k = normalize_object(t);
                if !k.is_empty() && !out.contains_key(&k) {
                    let id = out.len();
                    out.insert(k, id);
                }
            }
        }
        out
    }

    #[test]
    fn canonical_object_picks_mode() {
        let votes = vec![
            ObjectVote {
                text: Some("adoption agencies".into()),
                observed_at: ts(2024, 1, 1),
                content: "Caroline researches adoption agencies".into(),
            },
            ObjectVote {
                text: Some("adoption agencies".into()),
                observed_at: ts(2024, 1, 2),
                content: "Caroline is researching adoption agencies".into(),
            },
            ObjectVote {
                text: Some("foster care".into()),
                observed_at: ts(2024, 1, 3),
                content: "Caroline also looking at foster care".into(),
            },
        ];
        let clusters = singleton_clusters(&votes);
        assert_eq!(
            canonical_object(&votes, &clusters).as_deref(),
            Some("adoption agencies")
        );
    }

    #[test]
    fn canonical_object_breaks_tie_by_recency() {
        let votes = vec![
            ObjectVote {
                text: Some("Rust".into()),
                observed_at: ts(2024, 1, 1),
                content: "".into(),
            },
            ObjectVote {
                text: Some("Go".into()),
                observed_at: ts(2024, 3, 1),
                content: "".into(),
            },
        ];
        let clusters = singleton_clusters(&votes);
        assert_eq!(canonical_object(&votes, &clusters).as_deref(), Some("Go"));
    }

    #[test]
    fn canonical_object_none_when_all_empty() {
        let votes = vec![ObjectVote {
            text: None,
            observed_at: ts(2024, 1, 1),
            content: "Caroline is happy".into(),
        }];
        let clusters = singleton_clusters(&votes);
        assert!(canonical_object(&votes, &clusters).is_none());
    }

    #[test]
    fn canonical_object_clustered_paraphrase_wins() {
        // "adoption agencies" + "adoption agency" cluster together
        // (3 votes), out-vote the single "foster care" (1 vote).  Pre-
        // clustering these would split 2/1/1 and let recency winner
        // "foster care" take canonical.
        let votes = vec![
            ObjectVote {
                text: Some("adoption agencies".into()),
                observed_at: ts(2024, 1, 1),
                content: "".into(),
            },
            ObjectVote {
                text: Some("adoption agencies".into()),
                observed_at: ts(2024, 1, 2),
                content: "".into(),
            },
            ObjectVote {
                text: Some("adoption agency".into()),
                observed_at: ts(2024, 1, 3),
                content: "".into(),
            },
            ObjectVote {
                text: Some("foster care".into()),
                observed_at: ts(2024, 1, 4),
                content: "".into(),
            },
        ];
        // Synthetic clusters: agencies/agency share id 0, foster id 1.
        let mut clusters = HashMap::new();
        clusters.insert("adoption agencies".to_string(), 0);
        clusters.insert("adoption agency".to_string(), 0);
        clusters.insert("foster care".to_string(), 1);
        // Winning cluster is 0 (3 votes); within it the agencies
        // surface form has 2 votes vs agency's 1, so agencies wins.
        assert_eq!(
            canonical_object(&votes, &clusters).as_deref(),
            Some("adoption agencies"),
        );
    }

    #[test]
    fn vote_tallies_clustered_counts_in_cluster_space() {
        // Prior Fact: object = "agencies" (cluster 0).
        // Votes: 3 in cluster 0, 1 in cluster 1.  Without clustering,
        // string-exact would see only 1 "agencies" vote agreeing.
        let votes = vec![
            ObjectVote {
                text: Some("adoption agencies".into()),
                observed_at: ts(2024, 1, 1),
                content: "".into(),
            },
            ObjectVote {
                text: Some("adoption agency".into()),
                observed_at: ts(2024, 1, 2),
                content: "".into(),
            },
            ObjectVote {
                text: Some("adoption agencies".into()),
                observed_at: ts(2024, 1, 3),
                content: "".into(),
            },
            ObjectVote {
                text: Some("foster care".into()),
                observed_at: ts(2024, 1, 4),
                content: "".into(),
            },
        ];
        let mut clusters = HashMap::new();
        clusters.insert("adoption agencies".to_string(), 0);
        clusters.insert("adoption agency".to_string(), 0);
        clusters.insert("foster care".to_string(), 1);
        let (total, agree) = vote_tallies(&votes, Some(0), &clusters);
        assert_eq!(total, 4);
        assert_eq!(agree, 3);
    }

    #[test]
    fn build_clusters_collapses_near_duplicates() {
        // Two vectors at cos ~= 0.99, one at cos ~= 0.1 — the first
        // two should land in the same cluster, the third in its own.
        let mut embs: HashMap<String, Vec<f32>> = HashMap::new();
        embs.insert("agencies".into(), vec![1.0, 0.0, 0.0, 0.0]);
        embs.insert("agency".into(), vec![0.99, 0.14, 0.0, 0.0]);
        embs.insert("foster care".into(), vec![0.0, 1.0, 0.0, 0.0]);
        let mut keys = HashSet::new();
        keys.insert("agencies".to_string());
        keys.insert("agency".to_string());
        keys.insert("foster care".to_string());

        let clusters = build_clusters(&keys, &embs, 0.88);
        assert_eq!(clusters["agencies"], clusters["agency"]);
        assert_ne!(clusters["agencies"], clusters["foster care"]);
    }

    #[test]
    fn build_clusters_missing_embedding_is_singleton() {
        // No embeddings — every key its own cluster.
        let keys: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let clusters = build_clusters(&keys, &HashMap::new(), 0.88);
        let unique_ids: HashSet<usize> = clusters.values().copied().collect();
        assert_eq!(unique_ids.len(), 3);
    }

    #[test]
    fn date_prefix_appears_in_embed_text_for_canonical_path() {
        let t = ts(2024, 1, 15);
        let s = compose_embed_text(t, "Caroline", "has_pet", Some("cockatiel"), None, true);
        assert_eq!(s, "[January 2024] Caroline has_pet cockatiel");
    }

    #[test]
    fn date_prefix_appears_in_embed_text_for_fallback_path() {
        let t = ts(2024, 3, 1);
        let s = compose_embed_text(
            t,
            "Caroline",
            "feels",
            None,
            Some("Caroline is feeling happy today"),
            true,
        );
        assert_eq!(
            s, "[March 2024] Caroline is feeling happy today",
            "fallback path should embed the freshest contributor's content prefixed with the date"
        );
    }

    #[test]
    fn supported_by_weight_reflects_cosine_and_falls_back() {
        let v = [1.0_f32, 0.0, 0.0];
        // Identical embeddings → weight 1.0.
        assert!((supported_by_weight(Some(&v), Some(&v)) - 1.0).abs() < 1e-6);
        // Orthogonal embeddings → weight 0.0.
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!(supported_by_weight(Some(&a), Some(&b)).abs() < 1e-6);
        // Partial overlap (45°) → ~0.707, strictly between 0 and 1.
        let c = [1.0_f32, 1.0];
        let d = [1.0_f32, 0.0];
        let w = supported_by_weight(Some(&c), Some(&d));
        assert!(w > 0.6 && w < 0.8, "expected ~0.707, got {w}");
        // Missing either embedding → fallback 1.0 (never drop/zero an edge).
        assert_eq!(supported_by_weight(None, Some(&v)), 1.0);
        assert_eq!(supported_by_weight(Some(&v), None), 1.0);
        assert_eq!(supported_by_weight(None, None), 1.0);
    }

    #[test]
    fn date_augment_disabled_omits_prefix() {
        let t = ts(2024, 1, 15);
        let s = compose_embed_text(t, "Caroline", "has_pet", Some("cockatiel"), None, false);
        assert_eq!(s, "Caroline has_pet cockatiel");
    }

    #[test]
    fn embed_text_falls_back_to_subject_predicate_when_nothing_else() {
        let t = ts(2024, 6, 1);
        // Canonical None, fallback_content None → "{subject} {predicate}".
        let s = compose_embed_text(t, "Melanie", "likes", None, None, true);
        assert_eq!(s, "[June 2024] Melanie likes");
    }

    #[test]
    fn cosine_sim_basic() {
        assert!((cosine_sim(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_sim(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine_sim(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }
}
