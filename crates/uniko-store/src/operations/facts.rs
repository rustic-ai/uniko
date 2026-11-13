//! Fact node CRUD for P4 Consolidation.
//!
//! Provides upsert-by-triple semantics for derived Facts plus the
//! supporting edges (`SUPPORTED_BY`, `CREATED`, `INVOLVED`) and the
//! `ConsolidationCycle` audit node.  Embeddings are computed by Layer 2
//! and supplied as an opaque vector — this module never calls Xervo.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uni_db::Value;
use uni_db::common::{TemporalValue, uni_btic::Btic};

use crate::error::Result;
use crate::id::new_id;
use crate::schema::btic::{CERTAINTY_THRESHOLD, btic_active, btic_upgrade_certainty};
use crate::schema::constants::{edges, labels};
use crate::storage::KnowledgeBase;
use crate::types::NodeId;

/// Outcome of a single Fact upsert call.
#[derive(Debug, Clone, Copy)]
pub struct FactUpsert {
    /// Internal node id of the (created or reinforced) Fact.
    pub node_id: NodeId,
    /// `true` when this call inserted a new Fact node.
    pub was_created: bool,
    /// Total `observation_count` after this upsert.
    pub observation_count: i64,
}

/// One Fact to upsert in [`KnowledgeBase::batch_upsert_facts`].
///
/// Mirrors the per-Fact inputs of [`KnowledgeBase::upsert_fact_by_triple`].
/// `embedding` may be `None` for individual entries when the Layer 2
/// batch embed succeeded only for a subset — those Facts will be stored
/// without an embedding (same semantic as the single-call helper).
#[derive(Debug, Clone)]
pub struct FactUpsertInput {
    /// Triple subject (lower-cased by `fact_id_for` for the id).
    pub subject: String,
    /// Triple predicate.
    pub predicate: String,
    /// Triple object; `None` collapses to empty string in `fact_id`.
    pub object: Option<String>,
    /// Number of observations contributing to this upsert.
    pub observation_count: i64,
    /// Earliest observed_at among the contributing observations.
    pub observed_at: DateTime<Utc>,
    /// Pre-computed embedding from Layer 2, or `None` to store without.
    pub embedding: Option<Vec<f32>>,
    /// The Fact's visibility (`private:{id}`, `team:{id}`, `org:{id}`), or
    /// `None` / `"public"` for a public Fact. A scoped Fact is a separate
    /// node from the public Fact of the same triple; see
    /// [`fact_id_for_scoped`].
    pub visibility: Option<String>,
}

/// Build the deterministic external `fact_id` for a triple.
///
/// Idempotency anchor: callers upserting the same triple on subsequent
/// cycles resolve to the same Fact node via the
/// `Fact.fact_id` hash index.  Lowercased to make grouping
/// case-insensitive (P3 already normalizes `predicate`; the other two
/// fields can vary by surface form).
pub fn fact_id_for(subject: &str, predicate: &str, object: Option<&str>) -> String {
    format!(
        "{}|{}|{}",
        subject.to_ascii_lowercase(),
        predicate.to_ascii_lowercase(),
        object.unwrap_or("").to_ascii_lowercase(),
    )
}

/// The external `fact_id` for a triple held under `visibility`.
///
/// A public triple (no visibility, `""` or `"public"`) keeps
/// [`fact_id_for`]'s id. A scoped one appends its visibility, so the same
/// triple asserted privately and publicly are two Facts: reinforcing or
/// contradicting one never touches the other, and the scoped one's
/// existence never surfaces through the public one.
pub fn fact_id_for_scoped(
    subject: &str,
    predicate: &str,
    object: Option<&str>,
    visibility: Option<&str>,
) -> String {
    let base = fact_id_for(subject, predicate, object);
    match scoped_visibility(visibility) {
        Some(v) => format!("{base}|{v}"),
        None => base,
    }
}

/// `visibility` when it scopes a Fact, `None` when it is public.
#[must_use]
pub fn scoped_visibility(visibility: Option<&str>) -> Option<&str> {
    match visibility.map(str::trim) {
        None | Some("" | "public") => None,
        Some(v) => Some(v),
    }
}

/// Laplace-smoothed confidence: `(n + 1) / (n + 2)`.
///
/// Maps `n=0 → 0.5`, `n=1 → 0.67`, `n=4 → 0.83`, `n=10 → 0.92`.
/// Asymptotes toward 1.0 without ever reaching it — the system never
/// claims absolute certainty from observation counts alone.
pub fn laplace_confidence(observation_count: i64) -> f64 {
    let n = observation_count.max(0) as f64;
    (n + 1.0) / (n + 2.0)
}

/// Build the per-Fact RMW lock key used by [`KnowledgeBase::upsert_fact_by_triple`]
/// and [`KnowledgeBase::batch_upsert_facts`].
///
/// Both call sites lock in the same namespace so a single-call upsert
/// and a batch upsert that touch the same Fact serialize against each
/// other rather than racing.
fn fact_lock_key(fact_id: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(5 + fact_id.len());
    k.extend_from_slice(b"fact:");
    k.extend_from_slice(fact_id.as_bytes());
    k
}

impl KnowledgeBase {
    /// Upsert a Fact for the given `(subject, predicate, object)` triple.
    ///
    /// First call creates the Fact with `valid_at = btic_active(observed_at)`,
    /// `observation_count = n`, and Laplace-smoothed `confidence`.
    /// Subsequent calls reinforce: `observation_count += n`, `confidence`
    /// is recomputed, and the BTIC certainty is upgraded to
    /// [`Certainty::Definite`](uni_db::common::uni_btic::Certainty::Definite)
    /// once the cumulative count crosses [`CERTAINTY_THRESHOLD`].
    ///
    /// `embedding` is supplied by the caller (computed in Layer 2) and
    /// is only written on the create path; reinforcement does not
    /// re-embed since the canonical content text is unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on any database failure.
    pub async fn upsert_fact_by_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: Option<&str>,
        observation_count: i64,
        observed_at: DateTime<Utc>,
        embedding: Option<Vec<f32>>,
    ) -> Result<FactUpsert> {
        self.upsert_scoped_fact_by_triple(
            subject,
            predicate,
            object,
            observation_count,
            observed_at,
            embedding,
            None,
        )
        .await
    }

    /// [`upsert_fact_by_triple`](Self::upsert_fact_by_triple) for a Fact
    /// held under `visibility`. A scoped Fact is keyed apart from the
    /// public Fact of the same triple ([`fact_id_for_scoped`]) and carries
    /// the visibility from creation.
    ///
    /// # Errors
    ///
    /// As for [`upsert_fact_by_triple`](Self::upsert_fact_by_triple).
    #[allow(clippy::too_many_arguments)]
    pub async fn upsert_scoped_fact_by_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: Option<&str>,
        observation_count: i64,
        observed_at: DateTime<Utc>,
        embedding: Option<Vec<f32>>,
        visibility: Option<&str>,
    ) -> Result<FactUpsert> {
        let fact_id = fact_id_for_scoped(subject, predicate, object, visibility);

        // Serialize concurrent upserts of the same triple.  Without
        // this, two callers can both observe the same prior count and
        // each apply `prior + n`, losing one batch of observations.
        // Lock key is the fact_id namespace so this nests cleanly with
        // [`KnowledgeBase::batch_upsert_facts`], which acquires the same
        // namespace.
        let lock_key = fact_lock_key(&fact_id);
        let _rmw_guard = self.rmw_locks.lock(&lock_key).await;

        // Idempotency lookup via the fact_id hash index.
        if let Some((nid, props)) = self
            .get_node_by_ext_id(labels::FACT, "fact_id", &fact_id)
            .await?
        {
            let prior_count = props
                .get("observation_count")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let new_count = prior_count.saturating_add(observation_count);
            let confidence = laplace_confidence(new_count);

            let mut updates: HashMap<String, Value> = HashMap::new();
            updates.insert("observation_count".into(), Value::Int(new_count));
            updates.insert("confidence".into(), Value::Float(confidence));

            // Upgrade certainty once the cumulative count crosses the
            // threshold.  Idempotent: a Fact already at Definite stays
            // there.  `props` comes from `get_node_by_ext_id` which uses
            // `RETURN n` — Node-wrapped projection stringifies Temporal
            // columns, so we cannot reuse `props.get("valid_at")`.  Read
            // the raw BTIC via a bare-column projection.
            if (prior_count as u64) < CERTAINTY_THRESHOLD
                && (new_count as u64) >= CERTAINTY_THRESHOLD
            {
                let session = self.db.session();
                let r = session
                    .query_with("MATCH (f:Fact) WHERE id(f) = $vid RETURN f.valid_at AS valid_at")
                    .param("vid", nid)
                    .fetch_all()
                    .await?;
                if let Some(row) = r.rows().first()
                    && let Some(current) = extract_btic(row.value("valid_at"))
                {
                    updates.insert(
                        "valid_at".into(),
                        btic_to_value(&btic_upgrade_certainty(&current)),
                    );
                }
            }

            self.update_node(nid, &updates).await?;
            return Ok(FactUpsert {
                node_id: nid,
                was_created: false,
                observation_count: new_count,
            });
        }

        // Create path.
        let valid_at = btic_active(observed_at);
        let confidence = laplace_confidence(observation_count);

        let mut props: HashMap<String, Value> = HashMap::new();
        props.insert("fact_id".into(), Value::String(fact_id));
        props.insert("subject".into(), Value::String(subject.to_string()));
        props.insert("predicate".into(), Value::String(predicate.to_string()));
        if let Some(obj) = object {
            props.insert("object".into(), Value::String(obj.to_string()));
        }
        props.insert("confidence".into(), Value::Float(confidence));
        props.insert("observation_count".into(), Value::Int(observation_count));
        props.insert("valid_at".into(), btic_to_value(&valid_at));
        props.insert(
            "source_rule".into(),
            Value::String("consolidation_v1".into()),
        );
        if let Some(vec) = embedding {
            props.insert("embedding".into(), Value::Vector(vec));
        }
        if let Some(v) = scoped_visibility(visibility) {
            props.insert("visibility".into(), Value::String(v.to_string()));
        }

        let nid = self.create_node(labels::FACT, &props).await?;
        Ok(FactUpsert {
            node_id: nid,
            was_created: true,
            observation_count,
        })
    }

    /// Wire `SUPPORTED_BY` edges from a Fact to many Observations.
    ///
    /// All edges share `weight = 1.0`.  Idempotency is the caller's
    /// responsibility — passing the same `(fact_id, observation)` pair
    /// twice creates duplicate edges.  In the consolidation cycle this
    /// is avoided by only processing observations that have no
    /// inbound `PROCESSED` edge from any prior cycle.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn attach_supported_by(
        &self,
        fact_node_id: NodeId,
        observation_node_ids: &[NodeId],
    ) -> Result<()> {
        if observation_node_ids.is_empty() {
            return Ok(());
        }
        let mut weight_props: HashMap<String, Value> = HashMap::new();
        weight_props.insert("weight".into(), Value::Float(1.0));
        let edge_specs: Vec<(NodeId, NodeId, HashMap<String, Value>)> = observation_node_ids
            .iter()
            .map(|&oid| (fact_node_id, oid, weight_props.clone()))
            .collect();
        self.batch_create_edges_fast(
            edges::SUPPORTED_BY,
            Some(labels::FACT),
            Some(labels::OBSERVATION),
            &edge_specs,
        )
        .await
    }

    /// Upsert many Facts via the same per-Fact semantics as
    /// [`KnowledgeBase::upsert_fact_by_triple`], in batched operations.
    ///
    /// Issues at most four storage ops per call regardless of input
    /// size, replacing the per-Fact transaction loop. The split mirrors
    /// the entity-dedup batched path:
    ///
    /// 1. Read-only `UNWIND … MATCH (f:Fact {fact_id: fid})` to find
    ///    which `fact_id`s already exist (one indexed lookup per row).
    /// 2. [`KnowledgeBase::batch_create_nodes`] for the not-found subset.
    /// 3. `UNWIND … MATCH WHERE id(n)=u.nid SET …` for the reinforce
    ///    subset, including the conditional BTIC certainty upgrade
    ///    pre-computed per row.
    ///
    /// Returns one [`FactUpsert`] per input row in input order, with
    /// the freshly resolved node id and the post-upsert observation
    /// count.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::UnikoError::Storage`] on any storage
    /// failure. Partial state is possible if the call aborts between
    /// phases — same property as the per-Fact loop it replaces.
    pub async fn batch_upsert_facts(
        &self,
        inputs: Vec<FactUpsertInput>,
    ) -> Result<Vec<FactUpsert>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Pre-compute fact_ids once; reused across phases.
        let fact_ids: Vec<String> = inputs
            .iter()
            .map(|i| {
                fact_id_for_scoped(
                    &i.subject,
                    &i.predicate,
                    i.object.as_deref(),
                    i.visibility.as_deref(),
                )
            })
            .collect();

        // Within-batch dedup by fact_id. Multiple input rows can map
        // to the same fact_id when the caller's grouping key
        // (subject, predicate) differs only in case from another
        // entry — `fact_id_for` lower-cases both sides. The original
        // per-row `upsert_fact_by_triple` loop folded these via its
        // MATCH-then-create-or-update; the batched path must do the
        // same in Rust because two UNWIND rows over the same MERGE
        // node would otherwise clobber each other's counts (for the
        // existing path) or create duplicate Fact nodes (for the
        // new path — the schema has no unique constraint on
        // `fact_id`).
        //
        // First occurrence wins for subject / predicate / object /
        // embedding; `observation_count` is summed across all rows
        // with the same fact_id, and `observed_at` is reduced to the
        // earliest. `first_idx` records the input index that
        // introduced each fact_id so the output `was_created` flag
        // can mirror the per-row helper (true only for the first
        // appearance of a freshly-created Fact).
        let mut canonical: HashMap<String, FactUpsertInput> =
            HashMap::with_capacity(fact_ids.len());
        let mut first_idx: HashMap<String, usize> = HashMap::with_capacity(fact_ids.len());
        let mut unique_fids: Vec<String> = Vec::with_capacity(fact_ids.len());
        for (i, (input, fid)) in inputs.iter().zip(fact_ids.iter()).enumerate() {
            match canonical.entry(fid.clone()) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(input.clone());
                    first_idx.insert(fid.clone(), i);
                    unique_fids.push(fid.clone());
                }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    let cur = e.get_mut();
                    cur.observation_count = cur
                        .observation_count
                        .saturating_add(input.observation_count);
                    if input.observed_at < cur.observed_at {
                        cur.observed_at = input.observed_at;
                    }
                }
            }
        }

        // Serialize against concurrent upserts of the same Facts.
        // We acquire one stripe per distinct `fact_id` and hold the
        // guards for the whole call so Phase 1 / 2 / 3 see a coherent
        // snapshot per Fact.  Sorting by lock key gives a deterministic
        // acquisition order so two overlapping batches cannot deadlock.
        // Aligns with the single-call path
        // [`KnowledgeBase::upsert_fact_by_triple`], which locks in the
        // same namespace.
        let lock_keys: Vec<Vec<u8>> = unique_fids.iter().map(|fid| fact_lock_key(fid)).collect();
        // Acquire via `lock_many`, which maps keys to stripe indices and
        // de-duplicates them before locking. The previous hand-rolled loop
        // locked one guard per distinct `fact_id` without de-duplicating
        // stripe indices, so as soon as two fact_ids hashed to the same
        // stripe (near-certain past ~50 facts with 256 stripes) it re-locked
        // a stripe it already held — a non-reentrant `tokio::sync::Mutex`
        // self-deadlock that parked the whole consolidation cycle.
        let _rmw_guards = self.rmw_locks.lock_many(&lock_keys).await;

        // Phase 1: batched MATCH to discover which (deduped) fact_ids
        // exist. Returns one row per match; absent fact_ids simply
        // don't appear.
        let fid_list: Vec<Value> = unique_fids.iter().cloned().map(Value::String).collect();
        let match_cypher = "\
            UNWIND $fids AS fid \
            MATCH (n:Fact {fact_id: fid}) \
            RETURN fid AS fact_id, id(n) AS nid, \
                   n.observation_count AS observation_count, \
                   n.valid_at AS valid_at";

        let session = self.db.session();
        let match_result = session
            .query_with(match_cypher)
            .param("fids", Value::List(fid_list))
            .fetch_all()
            .await?;

        // Build fact_id → (nid, prior_count, prior_btic) map.
        let mut existing: HashMap<String, (NodeId, i64, Option<Btic>)> =
            HashMap::with_capacity(match_result.rows().len());
        for row in match_result.rows() {
            let fid: String = row.get("fact_id")?;
            let nid: i64 = row.get("nid")?;
            let prior_count: i64 = row.get("observation_count")?;
            // valid_at comes back as the BTIC Temporal Value; extract
            // structurally rather than going through a string roundtrip.
            // `Row::value` returns the raw `&Value`; `Row::get<T>` would
            // require `T: FromValue` which `Value` itself does not
            // implement.
            let prior_btic = extract_btic(row.value("valid_at"));
            existing.insert(fid, (nid, prior_count, prior_btic));
        }

        // Partition deduped canonical entries into existing (UPDATE)
        // and new (CREATE). `resolved` maps each unique fact_id to
        // its post-upsert node_id and post-upsert observation_count;
        // it's keyed by fact_id so the final output loop can fan it
        // back out across every original input row (including the
        // collapsed duplicates).
        let mut updates_list: Vec<Value> = Vec::new();
        let mut new_props: Vec<HashMap<String, Value>> = Vec::new();
        let mut new_fids_in_order: Vec<String> = Vec::new();
        let mut resolved: HashMap<String, (NodeId, i64)> =
            HashMap::with_capacity(unique_fids.len());

        for fid in &unique_fids {
            let canon = canonical
                .get(fid)
                .expect("canonical map populated above for every unique_fids entry");

            if let Some((nid, prior_count, prior_btic)) = existing.get(fid) {
                let new_count = prior_count.saturating_add(canon.observation_count);
                let confidence = laplace_confidence(new_count);

                let mut update = HashMap::with_capacity(4);
                update.insert("nid".into(), Value::Int(*nid));
                update.insert("new_count".into(), Value::Int(new_count));
                update.insert("new_confidence".into(), Value::Float(confidence));

                // Conditional certainty upgrade — match the single-row
                // path exactly: only upgrade once we cross the
                // threshold from below, and only when we still have a
                // readable prior BTIC to upgrade.
                let upgrade_valid_at = if (*prior_count as u64) < CERTAINTY_THRESHOLD
                    && (new_count as u64) >= CERTAINTY_THRESHOLD
                {
                    prior_btic
                        .as_ref()
                        .map(|b| btic_to_value(&btic_upgrade_certainty(b)))
                } else {
                    None
                };
                // Always send a `valid_at` key so the SET clause is
                // the same shape across rows; rows that don't need an
                // upgrade get `Value::Null` and the COALESCE in the
                // UPDATE statement keeps the existing column value.
                update.insert("valid_at".into(), upgrade_valid_at.unwrap_or(Value::Null));
                updates_list.push(Value::Map(update));

                resolved.insert(fid.clone(), (*nid, new_count));
            } else {
                let valid_at = btic_active(canon.observed_at);
                let confidence = laplace_confidence(canon.observation_count);

                let mut props: HashMap<String, Value> = HashMap::with_capacity(9);
                props.insert("fact_id".into(), Value::String(fid.clone()));
                props.insert("subject".into(), Value::String(canon.subject.clone()));
                props.insert("predicate".into(), Value::String(canon.predicate.clone()));
                if let Some(obj) = &canon.object {
                    props.insert("object".into(), Value::String(obj.clone()));
                }
                props.insert("confidence".into(), Value::Float(confidence));
                props.insert(
                    "observation_count".into(),
                    Value::Int(canon.observation_count),
                );
                props.insert("valid_at".into(), btic_to_value(&valid_at));
                props.insert(
                    "source_rule".into(),
                    Value::String("consolidation_v1".into()),
                );
                if let Some(vec) = canon.embedding.clone() {
                    props.insert("embedding".into(), Value::Vector(vec));
                }
                if let Some(v) = scoped_visibility(canon.visibility.as_deref()) {
                    props.insert("visibility".into(), Value::String(v.to_string()));
                }
                new_props.push(props);
                new_fids_in_order.push(fid.clone());
            }
        }

        // Phase 2: batched CREATE for not-found Facts.
        if !new_props.is_empty() {
            let new_nids = self.batch_create_nodes(labels::FACT, &new_props).await?;
            if new_nids.len() != new_fids_in_order.len() {
                return Err(crate::error::UnikoError::Storage(format!(
                    "batch_upsert_facts: batch_create_nodes returned {} nids for {} new inputs",
                    new_nids.len(),
                    new_fids_in_order.len()
                )));
            }
            for (fid, nid) in new_fids_in_order.into_iter().zip(new_nids) {
                let canon_count = canonical
                    .get(&fid)
                    .map(|c| c.observation_count)
                    .unwrap_or(0);
                resolved.insert(fid, (nid, canon_count));
            }
        }

        // Phase 3: batched UPDATE for reinforce. The COALESCE keeps the
        // existing valid_at when u.valid_at is null (no certainty
        // upgrade this round); otherwise it adopts the upgraded BTIC.
        if !updates_list.is_empty() {
            let update_cypher = "\
                UNWIND $updates AS u \
                MATCH (n) WHERE id(n) = u.nid \
                SET n.observation_count = u.new_count, \
                    n.confidence = u.new_confidence, \
                    n.valid_at = COALESCE(u.valid_at, n.valid_at)";
            self.transact_with_retry(uni_db::RetryOptions::default(), move |tx| {
                let updates_list = updates_list.clone();
                async move {
                    let r = async {
                        tx.execute_with(update_cypher)
                            .param("updates", Value::List(updates_list))
                            .run()
                            .await?;
                        Ok(())
                    }
                    .await;
                    (tx, r)
                }
            })
            .await?;
        }

        // Fan resolved (fact_id → node_id) back out across every
        // original input row, preserving caller-visible order. The
        // `was_created` flag is true only for the input that
        // introduced its fact_id AND only when that fact_id wasn't
        // already in the DB before this call — mirroring the
        // per-row `upsert_fact_by_triple` exactly.
        let mut outputs: Vec<FactUpsert> = Vec::with_capacity(inputs.len());
        for (i, fid) in fact_ids.iter().enumerate() {
            let (nid, post_count) = *resolved
                .get(fid)
                .expect("resolved populated for every fact_id");
            let was_first_input = first_idx.get(fid).map(|&first| first == i).unwrap_or(false);
            let was_newly_created = !existing.contains_key(fid);
            outputs.push(FactUpsert {
                node_id: nid,
                was_created: was_first_input && was_newly_created,
                observation_count: post_count,
            });
        }

        Ok(outputs)
    }

    /// Create a `ConsolidationCycle` audit node and wire its edges.
    ///
    /// Wires:
    /// - `PROCESSED` to every observation visited in the cycle
    ///   (idempotency anchor for future cycles)
    /// - `CREATED` to every newly-created Fact
    /// - `INVOLVED` to every reinforced Fact (existing Fact whose
    ///   `observation_count` was incremented)
    /// - `APPLIED_RULE` to every Locy rule executed in the cycle
    ///
    /// `INVOLVED` is reused for "reinforced Fact" rather than "Episode"
    /// because the spec's audit edge for "Fact touched by this cycle
    /// but not created" maps to the same semantic; Episodes get their
    /// own edge type when the procedural-memory pipeline ships.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    #[expect(
        clippy::too_many_arguments,
        reason = "ConsolidationCycle is an audit record with several \
                  edge groupings (processed/created/reinforced/invalidated/ \
                  applied rules) plus drift alerts and two timestamps; \
                  grouping into a struct would obscure the call site \
                  without simplifying it."
    )]
    pub async fn write_consolidation_cycle(
        &self,
        agent_id: &str,
        started_at: DateTime<Utc>,
        completed_at: DateTime<Utc>,
        processed_observations: &[NodeId],
        created_facts: &[NodeId],
        reinforced_facts: &[NodeId],
        invalidated_facts: &[NodeId],
        drift_alerts: i64,
        applied_rules: &[NodeId],
    ) -> Result<NodeId> {
        let mut props: HashMap<String, Value> = HashMap::new();
        props.insert("cycle_id".into(), Value::String(new_id()));
        props.insert("agent_id".into(), Value::String(agent_id.to_string()));
        props.insert(
            "started_at".into(),
            crate::types::datetime_value(started_at),
        );
        props.insert(
            "completed_at".into(),
            crate::types::datetime_value(completed_at),
        );
        props.insert(
            "observations_processed".into(),
            Value::Int(processed_observations.len() as i64),
        );
        props.insert(
            "facts_created".into(),
            Value::Int(created_facts.len() as i64),
        );
        props.insert(
            "facts_reinforced".into(),
            Value::Int(reinforced_facts.len() as i64),
        );
        props.insert(
            "facts_invalidated".into(),
            Value::Int(invalidated_facts.len() as i64),
        );
        props.insert("drift_alerts".into(), Value::Int(drift_alerts));

        let cycle_nid = self
            .create_node(labels::CONSOLIDATION_CYCLE, &props)
            .await?;

        let empty: HashMap<String, Value> = HashMap::new();
        let mk_edges = |targets: &[NodeId]| -> Vec<(NodeId, NodeId, HashMap<String, Value>)> {
            targets
                .iter()
                .map(|&t| (cycle_nid, t, empty.clone()))
                .collect()
        };

        // Every outgoing edge from the cycle node shares the source label
        // (CONSOLIDATION_CYCLE) and `mk_edges` shape; the only per-batch
        // variation is the edge type, the target label, and the targets vec.
        let edge_batches: [(&str, &str, &[NodeId]); 5] = [
            (
                edges::PROCESSED,
                labels::OBSERVATION,
                processed_observations,
            ),
            (edges::CREATED, labels::FACT, created_facts),
            (edges::REINFORCED, labels::FACT, reinforced_facts),
            (edges::INVALIDATED, labels::FACT, invalidated_facts),
            (edges::APPLIED_RULE, labels::RULE, applied_rules),
        ];
        for (edge_type, target_label, targets) in edge_batches {
            if targets.is_empty() {
                continue;
            }
            self.batch_create_edges_fast(
                edge_type,
                Some(labels::CONSOLIDATION_CYCLE),
                Some(target_label),
                &mk_edges(targets),
            )
            .await?;
        }

        Ok(cycle_nid)
    }
}

/// Convert an in-memory [`Btic`] to the [`Value::Temporal`] wire form.
pub(crate) fn btic_to_value(b: &Btic) -> Value {
    Value::Temporal(TemporalValue::Btic {
        lo: b.lo(),
        hi: b.hi(),
        meta: b.meta(),
    })
}

/// Decode a stored `valid_at` value back into an in-memory [`Btic`].
///
/// Returns `None` when the property is absent, null, or not a BTIC
/// temporal value (defensive — uni-db should not produce other shapes
/// for a column declared as [`uni_db::DataType::Btic`]).
pub(crate) fn extract_btic(value: Option<&Value>) -> Option<Btic> {
    match value? {
        Value::Temporal(TemporalValue::Btic { lo, hi, meta }) => Btic::new(*lo, *hi, *meta).ok(),
        _ => None,
    }
}

impl KnowledgeBase {
    /// Load the `valid_at` BTIC of a single Fact by node id.
    ///
    /// Returns `None` when the node is absent or has no decodable
    /// `valid_at`.  Used by the `invalidate_fact` agent tool to obtain
    /// the interval it must close without a second-guessing read.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn fact_valid_at(&self, fact_node: NodeId) -> Result<Option<Btic>> {
        let session = self.db.session();
        // Bare projection returns the raw `Value::Temporal(Btic{..})`
        // — same readback pattern as `find_stale_open_facts`.
        let result = session
            .query_with("MATCH (f:Fact) WHERE id(f) = $vid RETURN f.valid_at AS valid_at")
            .param("vid", fact_node)
            .fetch_all()
            .await?;
        let Some(row) = result.rows().first() else {
            return Ok(None);
        };
        Ok(extract_btic(row.value("valid_at")))
    }

    /// Find every open-BTIC Fact for `(subject, predicate)` whose
    /// `object` differs from `current_object`.
    ///
    /// "Open" = `valid_at.hi == POS_INF`, i.e. not previously closed.
    /// Used by F38 contradiction detection: when a consolidation cycle
    /// promotes a new canonical object, prior open Facts with stale
    /// objects must be invalidated.
    ///
    /// Returns each match's `(node_id, valid_at)` pair so the caller
    /// can close the BTIC without a second read.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn find_stale_open_facts(
        &self,
        subject: &str,
        predicate: &str,
        current_object: Option<&str>,
    ) -> Result<Vec<(NodeId, Btic)>> {
        let session = self.db.session();
        // Bare property projection (`f.valid_at AS valid_at`) returns the
        // raw `Value::Temporal(Btic{..})` — see `batch_upsert_facts` for
        // the same readback pattern.  Avoids the lossy
        // `toString` → RFC 3339 → granularity/certainty-name roundtrip
        // that the previous implementation used.
        let cypher = "MATCH (f:Fact) \
                      WHERE f.subject = $s AND f.predicate = $p \
                        AND btic_is_unbounded(f.valid_at) \
                      RETURN id(f) AS nid, f.object AS obj, \
                             f.valid_at AS valid_at";
        let result = session
            .query_with(cypher)
            .param("s", subject)
            .param("p", predicate)
            .fetch_all()
            .await?;

        let current = current_object.unwrap_or("").to_ascii_lowercase();
        let mut out = Vec::new();
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            let obj: String = row.get::<String>("obj").unwrap_or_default();
            if !current.is_empty() && obj.to_ascii_lowercase() == current {
                continue;
            }
            if let Some(btic) = extract_btic(row.value("valid_at")) {
                out.push((nid, btic));
            }
        }
        Ok(out)
    }

    /// Batched variant of [`KnowledgeBase::find_stale_open_facts`].
    ///
    /// Replaces N per-key session queries with one `UNWIND $keys ...
    /// MATCH (f:Fact) ...` call. Also returns the prior `object`
    /// surface form per Fact (eliminating the follow-up
    /// `MATCH (f) WHERE id(f) = $vid RETURN f.object` that the
    /// per-group caller would otherwise need).
    ///
    /// Each key is `(subject, predicate, scope)`, where `scope` is a
    /// Fact visibility or `""` for public. Only Facts held under that
    /// scope match, so a scoped observation never reinforces or
    /// contradicts a Fact in another scope. Keys with no open Facts are
    /// absent from the map (caller should treat absence as empty).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn find_stale_open_facts_batched(
        &self,
        keys: &[(String, String, String)],
    ) -> Result<HashMap<(String, String, String), Vec<(NodeId, Btic, String)>>> {
        use std::collections::hash_map::Entry;

        if keys.is_empty() {
            return Ok(HashMap::new());
        }

        // Build the UNWIND list as `[{subject, predicate, scope}, …]`.
        let list: Vec<Value> = keys
            .iter()
            .map(|(s, p, scope)| {
                let mut m = HashMap::with_capacity(3);
                m.insert("subject".to_string(), Value::String(s.clone()));
                m.insert("predicate".to_string(), Value::String(p.clone()));
                m.insert("scope".to_string(), Value::String(scope.clone()));
                Value::Map(m)
            })
            .collect();

        // Bare `f.valid_at AS valid_at` projection returns the raw
        // `Value::Temporal(Btic{..})` — same readback shape as
        // `batch_upsert_facts`. The scope is matched in Rust, where
        // `""` and `"public"` normalize together.
        let cypher = "UNWIND $keys AS k \
                      MATCH (f:Fact) \
                      WHERE f.subject = k.subject \
                        AND f.predicate = k.predicate \
                        AND btic_is_unbounded(f.valid_at) \
                      RETURN k.subject AS subject, k.predicate AS predicate, \
                             k.scope AS scope, \
                             id(f) AS nid, f.object AS obj, \
                             f.valid_at AS valid_at, \
                             coalesce(f.visibility, '') AS vis";

        let session = self.db.session();
        let result = session
            .query_with(cypher)
            .param("keys", Value::List(list))
            .fetch_all()
            .await?;

        type FactRow = (NodeId, Btic, String);
        let mut out: HashMap<(String, String, String), Vec<FactRow>> = HashMap::new();
        for row in result.rows() {
            let Ok(subject) = row.get::<String>("subject") else {
                continue;
            };
            let Ok(predicate) = row.get::<String>("predicate") else {
                continue;
            };
            let scope: String = row.get::<String>("scope").unwrap_or_default();
            let vis: String = row.get::<String>("vis").unwrap_or_default();
            if scoped_visibility(Some(&vis)).unwrap_or("") != scope {
                continue;
            }
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            let obj: String = row.get::<String>("obj").unwrap_or_default();
            let Some(btic) = extract_btic(row.value("valid_at")) else {
                continue;
            };

            match out.entry((subject, predicate, scope)) {
                Entry::Vacant(e) => {
                    e.insert(vec![(nid, btic, obj)]);
                }
                Entry::Occupied(mut e) => {
                    e.get_mut().push((nid, btic, obj));
                }
            }
        }
        Ok(out)
    }

    /// Close a Fact's BTIC interval as of `now` and (optionally) wire an
    /// `INVALIDATES` edge from `replacement` → this fact.
    ///
    /// Used by F38 when a contradicting canonical object is promoted
    /// in a consolidation cycle.  The fact remains in the graph (for
    /// audit) but its BTIC `hi` is closed so `btic.contains(now)` is
    /// false going forward.
    ///
    /// `reason` populates the `INVALIDATES.reason` property; pass
    /// `None` to skip the edge entirely (useful when the caller wants
    /// to invalidate without recording a successor).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn invalidate_fact(
        &self,
        stale_fact: NodeId,
        old_interval: &Btic,
        now: DateTime<Utc>,
        replacement: Option<NodeId>,
        reason: Option<&str>,
    ) -> Result<()> {
        let closed = crate::schema::btic::btic_invalidate(old_interval, now);
        let mut updates: HashMap<String, Value> = HashMap::new();
        updates.insert("valid_at".into(), btic_to_value(&closed));
        // Stamp the node-level invalidation audit (recorded even when there
        // is no successor fact, i.e. no INVALIDATES edge below).
        updates.insert("invalidated_at".into(), crate::types::datetime_value(now));
        if let Some(r) = reason {
            updates.insert("invalidation_reason".into(), Value::String(r.to_string()));
        }
        self.update_node(stale_fact, &updates).await?;

        if let Some(new_fact) = replacement {
            let mut edge_props: HashMap<String, Value> = HashMap::new();
            if let Some(r) = reason {
                edge_props.insert("reason".into(), Value::String(r.to_string()));
            }
            // Stamp the edge so F39 can window invalidations by time.
            edge_props.insert("invalidated_at".into(), crate::types::datetime_value(now));
            self.create_edge(edges::INVALIDATES, new_fact, stale_fact, &edge_props)
                .await?;
        }
        Ok(())
    }

    /// Record that an Entity's subject was invalidated and flip the
    /// `unstable` flag once invalidations within the last 30 days
    /// exceed `drift_threshold`.
    ///
    /// Returns whether the Entity transitioned to unstable on this
    /// call (`false` when it was already unstable or remains stable).
    /// Creates an Entity node implicitly when `subject` has none — F38
    /// must not silently swallow drift when the upstream NER missed
    /// the subject.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn record_entity_invalidation(
        &self,
        subject: &str,
        now: DateTime<Utc>,
        drift_threshold: i64,
    ) -> Result<bool> {
        let key = subject.trim();
        if key.is_empty() {
            return Ok(false);
        }

        // Serialize concurrent invalidations for the same entity name.
        // Without this, two callers can both read `prior_count` and
        // each write `prior + 1`, losing one invalidation increment.
        let mut lock_key = Vec::with_capacity(7 + key.len());
        lock_key.extend_from_slice(b"entity:");
        lock_key.extend_from_slice(key.as_bytes());
        let _rmw_guard = self.rmw_locks.lock(&lock_key).await;

        // Fetch (or implicitly create) the Entity by name.
        let session = self.db.session();
        let lookup = session
            .query_with("MATCH (e:Entity {name: $n}) RETURN id(e) AS nid, e.invalidation_count AS c, e.unstable AS u")
            .param("n", key)
            .fetch_all()
            .await
            ?;

        let (nid, prior_count, prior_unstable) = match lookup.rows().first() {
            Some(row) => {
                let nid: i64 = row.get("nid").unwrap_or(0);
                let c: i64 = row.get::<i64>("c").unwrap_or(0);
                let u: bool = row.get::<bool>("u").unwrap_or(false);
                (nid, c, u)
            }
            None => {
                // Implicit create when NER missed the subject. Use the canonical
                // entity_id scheme (issue #1) — no type is available here, so
                // classify as `misc`. (Lookup above is by `name`, so an existing
                // typed entity is found and updated instead of duplicated.)
                let entity_id = crate::id::entity_id(key, "misc");
                let mut props: HashMap<String, Value> = HashMap::new();
                props.insert("name".into(), Value::String(key.to_string()));
                props.insert("entity_type".into(), Value::String("misc".into()));
                let nid = self
                    .merge_node(labels::ENTITY, "entity_id", &entity_id, &props)
                    .await?;
                (nid, 0, false)
            }
        };

        let new_count = prior_count.saturating_add(1);
        let mut updates: HashMap<String, Value> = HashMap::new();
        updates.insert("invalidation_count".into(), Value::Int(new_count));
        updates.insert(
            "last_invalidation_at".into(),
            crate::types::datetime_value(now),
        );

        // Drift gate (F39): count INVALIDATES edges on this entity's Facts
        // stamped within the last 30 days. The contradiction path creates
        // the current edge (with `invalidated_at`) before this runs, so it
        // is included. Edges predating the `invalidated_at` field (NULL)
        // fall outside the window and are not counted — acceptable
        // migration behavior.
        const DRIFT_WINDOW_DAYS: i64 = 30;
        let since = now - chrono::Duration::days(DRIFT_WINDOW_DAYS);
        let window_count = self.count_recent_invalidations(key, since).await?;
        let now_unstable = window_count > drift_threshold;
        updates.insert("unstable".into(), Value::Bool(now_unstable));
        self.update_node(nid, &updates).await?;

        Ok(now_unstable && !prior_unstable)
    }

    /// Count `INVALIDATES` edges on Facts with `subject = entity_name`
    /// stamped at or after `since`.
    ///
    /// Backs F39's rolling-window drift gate. Edges without an
    /// `invalidated_at` (created before that field existed) are excluded
    /// by the range predicate.
    async fn count_recent_invalidations(
        &self,
        entity_name: &str,
        since: DateTime<Utc>,
    ) -> Result<i64> {
        let session = self.db.session();
        let cypher = "MATCH (new:Fact)-[r:INVALIDATES]->(old:Fact) \
                      WHERE old.subject = $n AND r.invalidated_at >= $since \
                      RETURN count(r) AS k";
        let result = session
            .query_with(cypher)
            .param("n", entity_name)
            .param("since", crate::types::datetime_value(since))
            .fetch_all()
            .await?;
        match result.rows().first() {
            Some(row) => Ok(row.get::<i64>("k").unwrap_or(0)),
            None => Ok(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_id_is_lowercased_and_pipe_separated() {
        assert_eq!(
            fact_id_for("Caroline", "Researches", Some("adoption agencies")),
            "caroline|researches|adoption agencies",
        );
    }

    #[test]
    fn fact_id_treats_missing_object_as_empty() {
        assert_eq!(fact_id_for("Jon", "is_happy", None), "jon|is_happy|");
    }

    #[test]
    fn laplace_grows_monotonically_and_is_bounded() {
        let c0 = laplace_confidence(0);
        let c4 = laplace_confidence(4);
        let c100 = laplace_confidence(100);
        assert!((c0 - 0.5).abs() < 1e-9);
        assert!(c0 < c4 && c4 < c100);
        assert!(c100 < 1.0);
    }
}
