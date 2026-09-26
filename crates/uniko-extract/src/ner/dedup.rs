//! Entity deduplication and graph persistence.
//!
//! Three-tier dedup cascade: (1) exact `entity_id` match, (2) embedding
//! similarity above threshold, (3) create new.  Creates MENTIONS edges
//! from source nodes to Entity nodes.

use std::collections::HashMap;

use uniko_store::schema::constants::{edges, labels};
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId, Transaction, UnikoError, Value};

use super::types::{EntityMatch, EntityType, ExtractionSource, RawEntity};

/// Drop ONNX-NER entities overlapping a format-structured rule entity.
///
/// Email and URL entities are defined by their surface *format*, so the
/// regex extractor is authoritative for them. When the ONNX NER cascade
/// tags the same characters differently — an email address reads as
/// name-like and gets a `Person` tag, for instance — that guess is
/// spurious: keeping it would mint a second, bogus `:Entity` for one
/// span. This drops any [`ExtractionSource::OnnxModel`] entity whose byte
/// span overlaps a rule-based [`EntityType::Email`] or
/// [`EntityType::Url`], leaving the rule entity as the single canonical
/// one. Cross-source overlap resolution: the in-regex pass already
/// suppresses overlaps within `rules`, but ONNX entities are merged in
/// afterward and never cross that filter.
///
/// A no-op on builds without the `onnx` feature, which emit no
/// `OnnxModel` entities.
///
/// # Examples
///
/// ```ignore
/// let kept = suppress_onnx_over_structured(all_raw);
/// ```
pub fn suppress_onnx_over_structured(mut raw: Vec<RawEntity>) -> Vec<RawEntity> {
    let structured: Vec<(usize, usize)> = raw
        .iter()
        .filter(|e| {
            e.source == ExtractionSource::RuleBased
                && matches!(e.entity_type, EntityType::Email | EntityType::Url)
        })
        .map(|e| (e.start_byte, e.end_byte))
        .collect();
    if structured.is_empty() {
        return raw;
    }
    raw.retain(|e| {
        e.source != ExtractionSource::OnnxModel
            || !structured
                .iter()
                .any(|&(s, t)| spans_overlap(e.start_byte, e.end_byte, s, t))
    });
    raw
}

/// Whether two half-open byte ranges `[s1, e1)` and `[s2, e2)` overlap.
fn spans_overlap(s1: usize, e1: usize, s2: usize, e2: usize) -> bool {
    s1 < e2 && s2 < e1
}

/// Entity admission policy — keep only genuine *named* entities.
///
/// Diagnosed: ~40% of extracted entities were `Date` noise and ~58% were
/// non-entities overall (greeting fragments, punctuation variants). This
/// drops the NER outputs that belong elsewhere in the graph:
/// - `Date` → already captured by observation temporal anchors,
/// - `Measurement` / `Preference` / `QuotedString` → captured in
///   observation triples,
/// - `Other` (the ONNX Event/Product/WorkOfArt/Group/Misc catch-all) →
///   admitted only above `other_min_confidence`,
/// - `Person` greeting/discourse fragments ("Hey Melanie") → dropped via
///   the shared filler vocabulary.
///
/// Real named types (Person, Organization, Location, Url, Email,
/// CodeSymbol, CodeImport) are always kept. When `strict` is `false` the
/// input passes through unchanged (legacy admit-everything, for A/B).
#[must_use]
pub fn admit_entities(
    raw: Vec<RawEntity>,
    strict: bool,
    other_min_confidence: f64,
) -> Vec<RawEntity> {
    if !strict {
        return raw;
    }
    raw.into_iter()
        .filter(|e| match e.entity_type {
            EntityType::Date
            | EntityType::Measurement
            | EntityType::Preference
            | EntityType::QuotedString => false,
            EntityType::Other => e.confidence >= other_min_confidence,
            EntityType::Person => {
                !crate::observations::filter::starts_with_filler(&e.canonical_name)
            }
            EntityType::Organization
            | EntityType::Location
            | EntityType::Url
            | EntityType::Email
            | EntityType::CodeSymbol
            | EntityType::CodeImport => true,
        })
        .collect()
}

/// Merge raw entities by canonical name within a single extraction batch.
///
/// Entities with the same `canonical_name` are collapsed: the highest
/// confidence extraction is kept, and mention count reflects total
/// occurrences.
pub fn deduplicate_raw(raw: Vec<RawEntity>) -> Vec<(RawEntity, u32)> {
    let mut map: HashMap<String, (RawEntity, u32)> = HashMap::new();
    for entity in raw {
        let key = entity.canonical_name.clone();
        map.entry(key)
            .and_modify(|(best, count)| {
                *count += 1;
                if entity.confidence > best.confidence {
                    *best = entity.clone();
                }
            })
            .or_insert((entity, 1));
    }
    map.into_values().collect()
}

/// Pre-tx prep: compute the canonical `entity_id`s for the batch and
/// snapshot `now`. The caller uses the ids to acquire the per-entity RMW
/// locks ([`KnowledgeBase::lock_entity_ids`]) BEFORE opening the write
/// tx; the authoritative existence read then happens in
/// [`apply_entity_upsert_nodes`] *inside* that tx and *under* those locks.
///
/// (Previously this did the existence read here, outside any tx — but a
/// pre-tx, pre-lock read is non-authoritative: a concurrent ingest could
/// create the same entity in the gap before the writer locked and opened
/// its tx, so the writer would still CREATE a duplicate. The read now
/// lives in `apply` on a post-lock snapshot. See issue #1 / RC2.)
pub async fn prepare_entity_upsert(
    _kb: &KnowledgeBase,
    deduped: Vec<(RawEntity, u32)>,
) -> uniko_store::Result<EntityUpsertPrep> {
    // Canonical entity_id (issue #1): the single shared derivation, keyed on
    // (lower name, canonical type). `EntityType::as_str()` already yields the
    // shared lowercase vocabulary, so overlapping types unify with the
    // action/consolidation paths.
    let entity_ids: Vec<String> = deduped
        .iter()
        .map(|(entity, _)| {
            uniko_store::id::entity_id(&entity.canonical_name, entity.entity_type.as_str())
        })
        .collect();

    Ok(EntityUpsertPrep {
        deduped,
        entity_ids,
        now_value: datetime_value(chrono::Utc::now()),
    })
}

/// Output of [`prepare_entity_upsert`]. Holds the new-vs-existing split
/// inputs needed at apply time. The existence map is *not* precomputed
/// here — `apply_entity_upsert_nodes` reads it authoritatively under lock.
///
/// `Clone` so the atomic ingest retry loop can hand a fresh copy to
/// [`apply_entity_upsert_nodes`] (which consumes it by value) on each attempt.
#[derive(Debug, Clone)]
pub struct EntityUpsertPrep {
    /// Raw entities (with mention counts) to upsert, in input order.
    pub deduped: Vec<(RawEntity, u32)>,
    /// Pre-computed canonical entity ids, parallel to `deduped`.
    pub entity_ids: Vec<String>,
    /// `Utc::now()` snapped at prep time; used as `last_seen`/`first_seen`.
    pub now_value: Value,
}

impl EntityUpsertPrep {
    /// True when there are no entities to write — apply step short-circuits.
    pub fn is_empty(&self) -> bool {
        self.deduped.is_empty()
    }
}

/// Writes-only inside the caller's tx. Issues:
/// (1) the authoritative existence read, under the caller's RMW locks,
/// (2) batch CREATE for not-found entities,
/// (3) batched UPDATE on existing entities (freq + last_seen + confidence).
///
/// Deliberately does **not** write MENTIONS. A multi-turn unit runs this
/// ONCE over the whole unit's merged batch — so `new_freq = old_freq +
/// summed mention_count` is computed from one snapshot read, and an entity
/// named in two turns yields one `:Entity` row — then emits the per-message
/// MENTIONS separately via [`apply_entity_mentions_in_tx`]. Calling this
/// once per turn instead would either duplicate the row (the second read
/// cannot see the first turn's uncommitted CREATE) or compute the second
/// turn's frequency from a stale `frequency`.
///
/// Returns one [`EntityMatch`] per input row in original order. The
/// caller owns the commit.
pub async fn apply_entity_upsert_nodes(
    kb: &KnowledgeBase,
    tx: &Transaction,
    prep: EntityUpsertPrep,
) -> uniko_store::Result<Vec<EntityMatch>> {
    if prep.is_empty() {
        return Ok(Vec::new());
    }
    let EntityUpsertPrep {
        deduped,
        entity_ids,
        now_value,
    } = prep;

    // Authoritative existence read, INSIDE the caller's tx and UNDER the
    // per-entity RMW locks the caller holds (see `lock_entity_ids`). The
    // tx snapshot was taken after the locks were acquired, so this read
    // reflects every committed entity and no concurrent writer can create
    // one of `entity_ids` until we commit and release. This is what makes
    // the check-then-create below race-free (issue #1 / RC2).
    let existing: HashMap<String, (NodeId, i64, f64)> =
        kb.fetch_entities_for_upsert_in_tx(tx, &entity_ids).await?;

    // Partition into (existing → batched UPDATE) and (new → batched
    // CREATE). `matches` is pre-filled with the right length so we can
    // fill node_id by index for the new rows once batch_create_nodes
    // returns them.
    let mut matches: Vec<EntityMatch> = Vec::with_capacity(deduped.len());
    let mut updates: Vec<(NodeId, i64, f64)> = Vec::new();
    let mut new_props: Vec<HashMap<String, Value>> = Vec::new();
    let mut new_indices: Vec<usize> = Vec::new();

    for (i, ((entity, mention_count), entity_id)) in
        deduped.iter().zip(entity_ids.iter()).enumerate()
    {
        if let Some(&(nid, old_freq, old_conf)) = existing.get(entity_id) {
            let new_freq = old_freq + i64::from(*mention_count);
            let new_conf = if entity.confidence > old_conf {
                entity.confidence
            } else {
                old_conf
            };

            updates.push((nid, new_freq, new_conf));

            matches.push(EntityMatch {
                canonical_name: entity.canonical_name.clone(),
                node_id: nid,
                was_existing: true,
                mention_count: *mention_count,
            });
        } else {
            let mut props = HashMap::with_capacity(7);
            props.insert("entity_id".into(), Value::String(entity_id.clone()));
            props.insert("name".into(), Value::String(entity.canonical_name.clone()));
            props.insert(
                "entity_type".into(),
                Value::String(entity.entity_type.as_str().to_string()),
            );
            props.insert("first_seen".into(), now_value.clone());
            props.insert("last_seen".into(), now_value.clone());
            props.insert("frequency".into(), Value::Int(i64::from(*mention_count)));
            props.insert("confidence".into(), Value::Float(entity.confidence));
            new_props.push(props);
            new_indices.push(i);

            // Placeholder node_id; filled in after Phase 2.
            matches.push(EntityMatch {
                canonical_name: entity.canonical_name.clone(),
                node_id: 0,
                was_existing: false,
                mention_count: *mention_count,
            });
        }
    }

    // ── Phase 2: batched CREATE for not-found entities.
    let phase2_start = std::time::Instant::now();
    let new_count = new_props.len();
    if !new_props.is_empty() {
        let new_nids = kb
            .batch_create_nodes_in_tx(tx, labels::ENTITY, &new_props)
            .await?;
        if new_nids.len() != new_indices.len() {
            return Err(UnikoError::Storage(format!(
                "apply_entity_upsert_nodes: batch_create_nodes returned {} nids for {} new inputs",
                new_nids.len(),
                new_indices.len()
            )));
        }
        for (&i, &nid) in new_indices.iter().zip(new_nids.iter()) {
            matches[i].node_id = nid;
        }
    }
    let phase2_create_ms = phase2_start.elapsed().as_millis();

    // ── Phase 3: batched UPDATE for found entities.
    // uni-db 2.0 (#53/#54) rewrites `UNWIND … MATCH WHERE id(n) = col`
    // into a HashJoin, so the id()-equality match no longer needs an
    // :Entity label hint to avoid the per-row multi-label scan that
    // previously cost ~18 ms/row (investigated 2026-05-20, fixed upstream).
    let phase3_start = std::time::Instant::now();
    let update_count = updates.len();
    if !updates.is_empty() {
        kb.batch_update_entity_counters_in_tx(tx, &updates, now_value)
            .await?;
    }
    let phase3_update_ms = phase3_start.elapsed().as_millis();

    tracing::info!(
        target: "apply_entity_breakdown",
        phase2_create_ms = phase2_create_ms as u64,
        phase3_update_ms = phase3_update_ms as u64,
        new_count = new_count as u64,
        update_count = update_count as u64,
        "apply_entity_nodes breakdown",
    );

    Ok(matches)
}

/// Phase 4, split out of [`apply_entity_upsert_nodes`]: one batched
/// `MENTIONS` CREATE for every (message, entity) pair in the unit.
///
/// Each tuple is `(message_node_id, entity_node_id, mention_count)`; the
/// same entity may legitimately appear under several messages, which is
/// exactly why this is separate from the node upsert — the entity row is
/// deduped across the unit, its provenance edges are not.
///
/// Sources are always freshly-created Message nodes in the caller's
/// still-open tx, so no MENTIONS edge from them can exist yet and plain
/// CREATE is safe (it skips MERGE's per-row penalty). Because every source
/// carries the same label hint, one call covers the whole unit.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on database failure.
pub async fn apply_entity_mentions_in_tx(
    kb: &KnowledgeBase,
    tx: &Transaction,
    mentions: &[(NodeId, NodeId, u32)],
) -> uniko_store::Result<()> {
    if mentions.is_empty() {
        return Ok(());
    }
    let edges_vec: Vec<(NodeId, NodeId, HashMap<String, Value>)> = mentions
        .iter()
        .map(|&(src, dst, count)| {
            let mut props = HashMap::with_capacity(1);
            props.insert("count".into(), Value::Int(i64::from(count)));
            (src, dst, props)
        })
        .collect();
    let start = std::time::Instant::now();
    kb.batch_create_edges_fast_in_tx(
        tx,
        edges::MENTIONS,
        Some(labels::MESSAGE),
        Some(labels::ENTITY),
        &edges_vec,
    )
    .await?;
    tracing::info!(
        target: "apply_entity_breakdown",
        phase4_mentions_ms = start.elapsed().as_millis() as u64,
        mentions_count = edges_vec.len() as u64,
        "apply_entity mentions",
    );
    Ok(())
}

/// One turn unit's entity batch: a single merged [`EntityUpsertPrep`] plus
/// the per-turn slices needed to rebuild each turn's own view of it.
#[derive(Debug, Clone)]
pub struct UnitEntityPrep {
    /// One batch for the whole unit, mention counts summed per
    /// `entity_id`. Fed to [`apply_entity_upsert_nodes`] exactly once.
    pub merged: EntityUpsertPrep,
    /// `per_turn[i]` lists `(index into `merged`, that turn's own mention
    /// count)` for every entity turn `i` mentioned.
    pub per_turn: Vec<Vec<(usize, u32)>>,
}

impl UnitEntityPrep {
    /// The `(node_id, canonical_name)` pairs turn `turn` mentioned, sliced
    /// out of the unit-wide match list by index.
    ///
    /// This is what feeds that turn's `ObservationInputs::extracted_entities`
    /// — each turn must see only its own entities even though the upsert
    /// was done for the whole unit at once.
    #[must_use]
    pub fn entities_for_turn(&self, turn: usize, matches: &[EntityMatch]) -> Vec<(NodeId, String)> {
        self.per_turn
            .get(turn)
            .map(|slots| {
                slots
                    .iter()
                    .filter_map(|&(i, _)| matches.get(i))
                    .map(|m| (m.node_id, m.canonical_name.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The `(message_node_id, entity_node_id, mention_count)` tuples for the
    /// whole unit, given the per-turn message nids in unit order.
    ///
    /// Pass the result straight to [`apply_entity_mentions_in_tx`].
    #[must_use]
    pub fn mentions(
        &self,
        message_nids: &[NodeId],
        matches: &[EntityMatch],
    ) -> Vec<(NodeId, NodeId, u32)> {
        let mut out = Vec::new();
        for (turn, slots) in self.per_turn.iter().enumerate() {
            let Some(&source) = message_nids.get(turn) else {
                continue;
            };
            for &(i, count) in slots {
                if let Some(m) = matches.get(i) {
                    out.push((source, m.node_id, count));
                }
            }
        }
        out
    }
}

/// Fold per-turn [`EntityUpsertPrep`]s into one unit-wide batch.
///
/// Keyed on the canonical `entity_id` — not the canonical name — so the
/// merge agrees exactly with the existence read in
/// [`apply_entity_upsert_nodes`], which is keyed the same way. Mention
/// counts sum across turns and the highest-confidence `RawEntity` wins,
/// matching [`deduplicate_raw`]'s rule within a single message.
///
/// `now_value` is taken from the first prep, since
/// [`prepare_entity_upsert`] snaps `Utc::now()` per call and a unit's
/// `first_seen`/`last_seen` should be one timestamp rather than an
/// arbitrary one of N.
#[must_use]
pub fn merge_entity_preps(per_turn_preps: Vec<EntityUpsertPrep>) -> UnitEntityPrep {
    let mut index_of: HashMap<String, usize> = HashMap::new();
    let mut deduped: Vec<(RawEntity, u32)> = Vec::new();
    let mut entity_ids: Vec<String> = Vec::new();
    let mut per_turn: Vec<Vec<(usize, u32)>> = Vec::with_capacity(per_turn_preps.len());
    let mut now_value: Option<Value> = None;

    for prep in per_turn_preps {
        if now_value.is_none() {
            now_value = Some(prep.now_value.clone());
        }
        let mut slots = Vec::with_capacity(prep.deduped.len());
        for ((entity, count), entity_id) in prep.deduped.into_iter().zip(prep.entity_ids) {
            if let Some(&i) = index_of.get(&entity_id) {
                deduped[i].1 += count;
                if entity.confidence > deduped[i].0.confidence {
                    deduped[i].0 = entity;
                }
                slots.push((i, count));
            } else {
                let i = deduped.len();
                index_of.insert(entity_id.clone(), i);
                entity_ids.push(entity_id);
                deduped.push((entity, count));
                slots.push((i, count));
            }
        }
        per_turn.push(slots);
    }

    UnitEntityPrep {
        merged: EntityUpsertPrep {
            deduped,
            entity_ids,
            now_value: now_value.unwrap_or_else(|| datetime_value(chrono::Utc::now())),
        },
        per_turn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ner::types::{EntityType, ExtractionSource};

    fn make_raw(name: &str, confidence: f64) -> RawEntity {
        RawEntity {
            surface_form: name.to_string(),
            canonical_name: name.to_lowercase(),
            entity_type: EntityType::Person,
            confidence,
            source: ExtractionSource::RuleBased,
            start_byte: 0,
            end_byte: name.len(),
        }
    }

    /// Build a one-turn prep the way `prepare_entity_upsert` would.
    fn prep_of(entities: Vec<(RawEntity, u32)>) -> EntityUpsertPrep {
        let entity_ids = entities
            .iter()
            .map(|(e, _)| uniko_store::id::entity_id(&e.canonical_name, e.entity_type.as_str()))
            .collect();
        EntityUpsertPrep {
            deduped: entities,
            entity_ids,
            now_value: datetime_value(chrono::Utc::now()),
        }
    }

    #[test]
    fn test_merge_entity_preps_sums_mentions_and_keeps_best_confidence() {
        // The same person named in two turns of one unit must collapse to a
        // single row — otherwise the unit CREATEs a duplicate `:Entity`,
        // because turn 2's existence read cannot see turn 1's uncommitted
        // CREATE.
        let unit = merge_entity_preps(vec![
            prep_of(vec![(make_raw("Alice", 0.6), 2)]),
            prep_of(vec![(make_raw("Alice", 0.9), 3), (make_raw("Bob", 0.8), 1)]),
        ]);

        assert_eq!(unit.merged.deduped.len(), 2, "alice must not be duplicated");
        assert_eq!(unit.merged.entity_ids.len(), 2);

        let alice = unit
            .merged
            .deduped
            .iter()
            .find(|(e, _)| e.canonical_name == "alice")
            .expect("alice present");
        assert_eq!(alice.1, 5, "mention counts sum across turns");
        assert!(
            (alice.0.confidence - 0.9).abs() < f64::EPSILON,
            "highest-confidence RawEntity wins, as in deduplicate_raw"
        );
    }

    #[test]
    fn test_merge_entity_preps_records_per_turn_indices() {
        let unit = merge_entity_preps(vec![
            prep_of(vec![(make_raw("Alice", 0.6), 2)]),
            prep_of(vec![(make_raw("Alice", 0.9), 3), (make_raw("Bob", 0.8), 1)]),
        ]);

        assert_eq!(unit.per_turn.len(), 2);
        assert_eq!(unit.per_turn[0], vec![(0, 2)], "turn 0 saw alice twice");
        assert_eq!(
            unit.per_turn[1],
            vec![(0, 3), (1, 1)],
            "turn 1 points at the SAME alice slot, with its own count"
        );

        // A fabricated match list lets us check the slicing without a KB.
        let matches = vec![
            EntityMatch {
                canonical_name: "alice".into(),
                node_id: 10,
                was_existing: false,
                mention_count: 5,
            },
            EntityMatch {
                canonical_name: "bob".into(),
                node_id: 11,
                was_existing: false,
                mention_count: 1,
            },
        ];

        assert_eq!(
            unit.entities_for_turn(0, &matches),
            vec![(10, "alice".to_string())],
            "turn 0 must not see bob"
        );
        assert_eq!(
            unit.entities_for_turn(1, &matches),
            vec![(10, "alice".to_string()), (11, "bob".to_string())]
        );

        // One entity row, but one MENTIONS edge per message — dedup must not
        // collapse provenance.
        assert_eq!(
            unit.mentions(&[100, 101], &matches),
            vec![(100, 10, 2), (101, 10, 3), (101, 11, 1)]
        );
    }

    #[test]
    fn test_merge_entity_preps_empty_unit() {
        let unit = merge_entity_preps(Vec::new());
        assert!(unit.merged.is_empty());
        assert!(unit.per_turn.is_empty());
        assert!(unit.mentions(&[], &[]).is_empty());
        assert!(unit.entities_for_turn(0, &[]).is_empty());
    }

    #[test]
    fn test_merge_entity_preps_takes_one_now_value() {
        // `prepare_entity_upsert` snaps Utc::now() per call, so N turns carry
        // N timestamps; the unit must settle on one or first_seen/last_seen
        // become arbitrary.
        let first = prep_of(vec![(make_raw("Alice", 0.6), 1)]);
        let expected = first.now_value.clone();
        let unit = merge_entity_preps(vec![first, prep_of(vec![(make_raw("Bob", 0.6), 1)])]);
        assert_eq!(unit.merged.now_value, expected);
    }

    #[test]
    fn test_dedup_merges_same_name() {
        let raw = vec![
            make_raw("Alice", 0.7),
            make_raw("Alice", 0.9),
            make_raw("Bob", 0.8),
        ];
        let deduped = deduplicate_raw(raw);
        assert_eq!(deduped.len(), 2);

        let alice = deduped.iter().find(|(e, _)| e.canonical_name == "alice");
        assert!(alice.is_some());
        let (alice_ent, alice_count) = alice.unwrap();
        assert_eq!(*alice_count, 2);
        // Highest confidence kept.
        assert!((alice_ent.confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dedup_collapses_normalized_case_and_punct_variants() {
        // Cross-path scenario the shared normalizer fixes: rules emits
        // "Melanie." → "melanie"; the ONNX path emits "Melanie" (was
        // title_case, now normalized) → "melanie". Both canonical names now
        // match, so the rows collapse to one entity (previously they split
        // into "melanie" vs "Melanie").
        let mk = |surface: &str, source: ExtractionSource| RawEntity {
            surface_form: surface.to_string(),
            canonical_name: uniko_store::text::normalize_canonical(surface),
            entity_type: EntityType::Person,
            confidence: 0.8,
            source,
            start_byte: 0,
            end_byte: surface.len(),
        };
        let raw = vec![
            mk("Melanie.", ExtractionSource::RuleBased),
            mk("Melanie", ExtractionSource::OnnxModel),
            mk("melanie", ExtractionSource::RuleBased),
        ];
        let deduped = deduplicate_raw(raw);
        assert_eq!(deduped.len(), 1, "all three variants must collapse to one");
        assert_eq!(deduped[0].0.canonical_name, "melanie");
        assert_eq!(deduped[0].1, 3, "mention count sums across variants");
    }

    #[test]
    fn test_dedup_single_entity() {
        let raw = vec![make_raw("Carol", 0.8)];
        let deduped = deduplicate_raw(raw);
        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].1, 1);
    }

    #[test]
    fn test_dedup_empty() {
        let deduped = deduplicate_raw(Vec::new());
        assert!(deduped.is_empty());
    }

    fn raw_span(
        name: &str,
        entity_type: EntityType,
        source: ExtractionSource,
        start: usize,
        end: usize,
    ) -> RawEntity {
        RawEntity {
            surface_form: name.to_string(),
            canonical_name: name.to_lowercase(),
            entity_type,
            confidence: 0.9,
            source,
            start_byte: start,
            end_byte: end,
        }
    }

    #[test]
    fn test_suppress_onnx_person_overlapping_email() {
        // The NER cascade tags an email address as a PERSON over the same
        // span the email regex matched; the ONNX guess must be dropped.
        let raw = vec![
            raw_span(
                "dedup@example.com",
                EntityType::Email,
                ExtractionSource::RuleBased,
                12,
                29,
            ),
            raw_span(
                "dedup@example.com",
                EntityType::Person,
                ExtractionSource::OnnxModel,
                12,
                29,
            ),
        ];
        let kept = suppress_onnx_over_structured(raw);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].entity_type, EntityType::Email);
        assert_eq!(kept[0].source, ExtractionSource::RuleBased);
    }

    #[test]
    fn test_suppress_keeps_nonoverlapping_onnx() {
        // A PERSON elsewhere in the text is unrelated to the email span
        // and must survive.
        let raw = vec![
            raw_span(
                "dedup@example.com",
                EntityType::Email,
                ExtractionSource::RuleBased,
                12,
                29,
            ),
            raw_span(
                "Alice",
                EntityType::Person,
                ExtractionSource::OnnxModel,
                0,
                5,
            ),
        ];
        let kept = suppress_onnx_over_structured(raw);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_suppress_noop_without_structured() {
        // No Email/URL entity → nothing is suppressed even on overlap.
        let raw = vec![
            raw_span(
                "Alice",
                EntityType::Person,
                ExtractionSource::RuleBased,
                0,
                5,
            ),
            raw_span(
                "Alice",
                EntityType::Person,
                ExtractionSource::OnnxModel,
                0,
                5,
            ),
        ];
        let kept = suppress_onnx_over_structured(raw);
        assert_eq!(kept.len(), 2);
    }

    fn raw_typed(name: &str, ty: EntityType, conf: f64) -> RawEntity {
        RawEntity {
            surface_form: name.to_string(),
            canonical_name: uniko_store::text::normalize_canonical(name),
            entity_type: ty,
            confidence: conf,
            source: ExtractionSource::OnnxModel,
            start_byte: 0,
            end_byte: name.len(),
        }
    }

    #[test]
    fn test_admit_entities_drops_noise_types_keeps_named() {
        let raw = vec![
            raw_typed("Caroline Smith", EntityType::Person, 0.9),
            raw_typed("Google", EntityType::Organization, 0.9),
            raw_typed("Sweden", EntityType::Location, 0.9),
            raw_typed("https://x.com", EntityType::Url, 0.95),
            raw_typed("a@b.com", EntityType::Email, 0.95),
            raw_typed("yesterday", EntityType::Date, 0.9),
            raw_typed("5 gb", EntityType::Measurement, 0.85),
            raw_typed("vscode", EntityType::Preference, 0.7),
            raw_typed("adoption agencies", EntityType::QuotedString, 0.6),
        ];
        let kept = admit_entities(raw, true, 0.9);
        let kept_types: Vec<EntityType> = kept.iter().map(|e| e.entity_type).collect();
        assert_eq!(kept.len(), 5, "only the 5 named-entity types survive");
        for t in [
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Url,
            EntityType::Email,
        ] {
            assert!(kept_types.contains(&t), "{t:?} should be kept");
        }
        for t in [
            EntityType::Date,
            EntityType::Measurement,
            EntityType::Preference,
            EntityType::QuotedString,
        ] {
            assert!(!kept_types.contains(&t), "{t:?} should be dropped");
        }
    }

    #[test]
    fn test_admit_entities_gates_other_by_confidence() {
        let raw = vec![
            raw_typed("Some Event", EntityType::Other, 0.95),
            raw_typed("vague thing", EntityType::Other, 0.80),
        ];
        let kept = admit_entities(raw, true, 0.9);
        assert_eq!(kept.len(), 1, "only the high-confidence Other survives");
        assert!((kept[0].confidence - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_admit_entities_drops_greeting_person() {
        let raw = vec![
            raw_typed("hey melanie", EntityType::Person, 0.7),
            raw_typed("thanks mel", EntityType::Person, 0.7),
            raw_typed("congrats caroline", EntityType::Person, 0.7),
            raw_typed("melanie smith", EntityType::Person, 0.7),
        ];
        let kept = admit_entities(raw, true, 0.9);
        assert_eq!(kept.len(), 1, "only the real person name survives");
        assert_eq!(kept[0].canonical_name, "melanie smith");
    }

    #[test]
    fn test_admit_entities_passthrough_when_not_strict() {
        let raw = vec![
            raw_typed("yesterday", EntityType::Date, 0.9),
            raw_typed("hey melanie", EntityType::Person, 0.7),
        ];
        let n = raw.len();
        let kept = admit_entities(raw, false, 0.9);
        assert_eq!(kept.len(), n, "non-strict admits everything unchanged");
    }
}
