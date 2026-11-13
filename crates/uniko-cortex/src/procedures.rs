//! Procedure promotion (P5) — F41/F42/F43.
//!
//! Recurring successful action/episode sequences are detected by the
//! stdlib `sequence_detector` Locy rule.  Each surfaced pair becomes
//! (or reinforces) a Procedure node with a precondition rule that
//! recall and planning can match against.
//!
//! Lifecycle (F41):
//!
//! ```text
//! candidate ──(promote_threshold reached)──▶ active
//!     │                                       │
//!     └──(stale)──▶ deprecated ◀──(degrade)───┘
//! ```
//!
//! - `candidate` — first observation; not yet eligible for use.
//! - `active` — matched the promotion threshold; surfaced to recall
//!   and planning.
//! - `deprecated` — effectiveness fell below the demotion threshold;
//!   kept for provenance but excluded from active use.
//!
//! `record_procedure_use` updates counters and applies the lifecycle
//! state machine on every call; `promote_procedures_once` runs the
//! sequence detector and creates/refreshes Procedure nodes.

use std::collections::HashMap;

use serde::Serialize;
use uniko_store::schema::labels;
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, UnikoError, Value};

/// Status string for a Procedure that has been observed once but
/// hasn't yet crossed the promotion threshold.
pub const STATUS_CANDIDATE: &str = "candidate";
/// Status string for a Procedure that is eligible for matching and
/// execution.
pub const STATUS_ACTIVE: &str = "active";
/// Status string for a Procedure that was demoted because its
/// effectiveness fell below the demotion threshold.
pub const STATUS_DEPRECATED: &str = "deprecated";

/// The `sequence_detector` Locy rule, invoked by name via
/// [`KnowledgeBase::query_rule`].
///
/// This is the **canonical source** for the rule. uniko-memory's
/// `register_stdlib_rules` references this same constant (uniko-memory depends
/// on uniko-cortex), so the rule is defined in exactly one place and registered
/// both at startup and (idempotently) by [`promote_procedures_once`] for
/// standalone use.
///
/// Detects every recurring `(action_a → action_b)` pair where both
/// episodes succeeded; `success_count` is the occurrence count.
/// [`upsert_procedure`] classifies candidate-vs-active against
/// `LifecycleConfig::promote_threshold`, so the rule itself surfaces ALL
/// pairs (no HAVING filter).
///
/// Locy is not Cypher: the two relationships are one comma-joined
/// `MATCH` (a second `MATCH` clause is a parse error), aggregate columns
/// are `expr AS name` (there is no `VALUE` keyword), and a `$param` in a
/// post-`FOLD` HAVING clause does not resolve — all three were latent
/// bugs in the prior rule, which is why it never registered and a Cypher
/// fallback was load-bearing (RC12, now resolved).
pub const SEQUENCE_DETECTOR_RULE: &str = "CREATE RULE sequence_detector AS \
     MATCH (e1:Episode)-[:FOLLOWED_BY]->(e2:Episode), \
           (e1)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id}) \
     WHERE e1.outcome = 'success' AND e2.outcome = 'success' \
     FOLD n = COUNT(*) \
     YIELD KEY e1.action_type AS action_a, KEY e2.action_type AS action_b, \
           n AS success_count";

/// Promotion / demotion thresholds for the Procedure lifecycle.
///
/// Values come from the spec's P5 description (Part VI) and the
/// `sequence_detector` `$promotion_threshold` parameter.  Demotion is
/// asymmetric so a Procedure that briefly failed isn't immediately
/// dropped.
#[derive(Debug, Clone, Copy)]
pub struct LifecycleConfig {
    /// Sequence-count threshold for promoting a candidate to active.
    pub promote_threshold: i64,
    /// Effectiveness threshold (success / (success + failure)) below
    /// which active Procedures are demoted to deprecated.
    pub demote_effectiveness: f64,
    /// Effectiveness threshold above which deprecated Procedures
    /// return to active.  Higher than `demote_effectiveness` for
    /// hysteresis.
    pub repromote_effectiveness: f64,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            promote_threshold: 3,
            demote_effectiveness: 0.4,
            repromote_effectiveness: 0.6,
        }
    }
}

/// Outcome of one [`promote_procedures_once`] call.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PromotionReport {
    /// Number of new candidate Procedure nodes created.
    pub created: usize,
    /// Number of existing Procedures whose support_count was bumped.
    pub reinforced: usize,
    /// Number of Procedures whose status flipped to `active` this run.
    pub promoted: usize,
}

/// Detect repeating successful action/episode sequences for `agent_id`
/// and create or update Procedure nodes accordingly.
///
/// Runs the stdlib `sequence_detector` Locy rule with the configured
/// promotion threshold.  Each pair `(action_a → action_b)` becomes a
/// Procedure with `name = "{a} → {b}"`.  Idempotent: re-running merges
/// into the same Procedure by deterministic id.
///
/// # Errors
///
/// - [`UnikoError::Locy`] when the `sequence_detector` rule cannot be
///   executed.
/// - [`UnikoError::Storage`] when graph writes fail.
pub async fn promote_procedures_once(
    kb: &KnowledgeBase,
    agent_id: &str,
    cfg: LifecycleConfig,
) -> Result<PromotionReport, UnikoError> {
    // Register the rule (idempotent — registering an exact-duplicate
    // program is a no-op) and invoke it by name via the QUERY goal-query
    // form. `upsert_procedure` applies the promotion threshold, so the
    // rule surfaces every recurring pair.
    kb.create_rule(SEQUENCE_DETECTOR_RULE).await?;
    let mut params = HashMap::new();
    params.insert("agent_id".into(), Value::String(agent_id.to_string()));
    let records = kb
        .query_rule(
            "sequence_detector",
            &["action_a", "action_b", "success_count"],
            &params,
        )
        .await?;

    let mut report = PromotionReport::default();
    for rec in records {
        let (Some(Value::String(a)), Some(Value::String(b)), Some(count)) = (
            rec.get("action_a"),
            rec.get("action_b"),
            rec.get("success_count").and_then(Value::as_i64),
        ) else {
            continue;
        };

        let (created, promoted) = upsert_procedure(kb, agent_id, a, b, count, cfg).await?;
        if created {
            report.created += 1;
        } else {
            report.reinforced += 1;
            if promoted {
                report.promoted += 1;
            }
        }
    }
    Ok(report)
}

/// Bump a Procedure's use counters after a real attempt and apply the
/// lifecycle state machine (F42).
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] when the Procedure is missing or
/// the update fails.
pub async fn record_procedure_use(
    kb: &KnowledgeBase,
    procedure_id: &str,
    succeeded: bool,
    cfg: LifecycleConfig,
) -> Result<(), UnikoError> {
    let proc_node = kb
        .get_node_by_ext_id(labels::PROCEDURE, "procedure_id", procedure_id)
        .await?
        .ok_or_else(|| {
            UnikoError::Storage(format!(
                "record_procedure_use: Procedure '{procedure_id}' not found"
            ))
        })?;

    let snapshot = kb
        .read_procedure_snapshot(procedure_id)
        .await?
        .ok_or_else(|| {
            UnikoError::Storage(format!(
                "record_procedure_use: Procedure '{procedure_id}' not found"
            ))
        })?;
    let use_count = snapshot.use_count + 1;
    let success = snapshot.success_count + i64::from(succeeded);
    let failure = snapshot.failure_count + i64::from(!succeeded);
    let denom = (success + failure) as f64;
    let effectiveness = if denom > 0.0 {
        success as f64 / denom
    } else {
        0.0
    };

    let new_status = match snapshot.status.as_str() {
        STATUS_CANDIDATE => snapshot.status.clone(),
        STATUS_ACTIVE if effectiveness < cfg.demote_effectiveness => STATUS_DEPRECATED.to_string(),
        STATUS_DEPRECATED if effectiveness >= cfg.repromote_effectiveness => {
            STATUS_ACTIVE.to_string()
        }
        _ => snapshot.status.clone(),
    };

    let mut props = HashMap::new();
    props.insert("use_count".into(), Value::Int(use_count));
    props.insert("success_count".into(), Value::Int(success));
    props.insert("failure_count".into(), Value::Int(failure));
    props.insert("effectiveness".into(), Value::Float(effectiveness));
    props.insert("status".into(), Value::String(new_status));
    props.insert("last_used_at".into(), datetime_value(chrono::Utc::now()));
    kb.update_node(proc_node.0, &props).await?;
    Ok(())
}

/// Find active Procedures whose `precondition_rule` matches the
/// current state (F43).
///
/// Each active Procedure stores its precondition as a Locy WHERE
/// fragment in `precondition_rule`.  For the MVP we evaluate it as a
/// comma-separated list of `key=value` clauses that must all appear in
/// `state` (see [`precondition_matches`]).  The signature is shaped to
/// accept a richer evaluator without breaking callers.
///
/// Procedures with no precondition_rule match any state.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] if the candidate query fails.
pub async fn match_procedures(
    kb: &KnowledgeBase,
    state: &HashMap<String, String>,
) -> Result<Vec<MatchedProcedure>, UnikoError> {
    let candidates = kb.fetch_procedures_by_status(STATUS_ACTIVE).await?;

    let mut out = Vec::new();
    for c in candidates {
        if precondition_matches(&c.precondition_rule, state) {
            out.push(MatchedProcedure {
                procedure_id: c.procedure_id,
                name: c.name,
                effectiveness: c.effectiveness,
            });
        }
    }
    Ok(out)
}

/// One Procedure that matched the input state.
#[derive(Debug, Clone, Serialize)]
pub struct MatchedProcedure {
    /// External `procedure_id` for downstream tools.
    pub procedure_id: String,
    /// Human-readable name (e.g. `"investigate → implement"`).
    pub name: String,
    /// Current effectiveness score in `[0.0, 1.0]`.
    pub effectiveness: f64,
}

/// Returns `(created, promoted)`:
/// - `created` — a fresh Procedure node was inserted this call.
/// - `promoted` — an existing candidate crossed the threshold and
///   transitioned to active in this call.
async fn upsert_procedure(
    kb: &KnowledgeBase,
    agent_id: &str,
    action_a: &str,
    action_b: &str,
    support_count: i64,
    cfg: LifecycleConfig,
) -> Result<(bool, bool), UnikoError> {
    let name = format!("{action_a} → {action_b}");
    let procedure_id = stable_procedure_id(agent_id, action_a, action_b);

    let existing = kb.read_procedure_snapshot(&procedure_id).await?;
    let new = existing.is_none();
    let snapshot = existing.unwrap_or_default();

    let next_status = if new {
        if support_count >= cfg.promote_threshold {
            STATUS_ACTIVE
        } else {
            STATUS_CANDIDATE
        }
    } else if snapshot.status == STATUS_CANDIDATE && support_count >= cfg.promote_threshold {
        STATUS_ACTIVE
    } else if snapshot.status.is_empty() {
        STATUS_CANDIDATE
    } else {
        // Keep existing status; demotion/repromotion is owned by
        // `record_procedure_use`.
        snapshot.status.as_str()
    };
    let promoted_this_call =
        !new && snapshot.status == STATUS_CANDIDATE && next_status == STATUS_ACTIVE;

    let now = datetime_value(chrono::Utc::now());
    let success_count = snapshot.success_count.max(support_count);
    let denom = (success_count + snapshot.failure_count) as f64;
    let eff = if denom > 0.0 {
        success_count as f64 / denom
    } else {
        1.0
    };

    let mut props = HashMap::new();
    props.insert("name".into(), Value::String(name.clone()));
    props.insert(
        "description".into(),
        Value::String(format!(
            "Detected sequence: when {action_a} succeeds it is often followed by {action_b}."
        )),
    );
    props.insert(
        "precondition_rule".into(),
        Value::String(format!("last_action_type={action_a}")),
    );
    props.insert("status".into(), Value::String(next_status.to_string()));
    props.insert(
        "use_count".into(),
        Value::Int(snapshot.use_count.max(support_count)),
    );
    props.insert("success_count".into(), Value::Int(success_count));
    props.insert("failure_count".into(), Value::Int(snapshot.failure_count));
    props.insert("effectiveness".into(), Value::Float(eff));
    if new {
        props.insert("created_at".into(), now.clone());
        // Recall's phase-1 search finds procedures by their embedding, which
        // nothing set before, so no procedure could surface in recall.
        // Embed the description once, on creation; a failure only means
        // this procedure is reachable by `match_procedures` and not recall.
        let description =
            format!("{name}: when {action_a} succeeds it is often followed by {action_b}.");
        let text = match kb.config().embedding.document_prefix.as_deref() {
            Some(prefix) => format!("{prefix}{description}"),
            None => description,
        };
        match kb
            .embed(uniko_store::schema::EMBED_ALIAS, &[text.as_str()])
            .await
        {
            Ok(mut vectors) if !vectors.is_empty() => {
                props.insert("embedding".into(), Value::Vector(vectors.remove(0)));
            }
            Ok(_) => {}
            Err(e) => {
                tracing::debug!(error = %e, procedure_id, "procedure embedding failed; stored without one");
            }
        }
    }
    props.insert("last_used_at".into(), now);

    kb.merge_node(labels::PROCEDURE, "procedure_id", &procedure_id, &props)
        .await?;

    Ok((new, promoted_this_call))
}

/// Build a deterministic procedure_id for the (agent, action_a, action_b) triple.
///
/// Wraps [`uniko_store::id::stable_hex64`] with NUL separators between
/// the three components so distinct triples cannot alias.  IDs are
/// persisted as `Procedure.procedure_id`.
fn stable_procedure_id(agent_id: &str, a: &str, b: &str) -> String {
    uniko_store::id::stable_hex64("proc", |h| {
        h.update(agent_id.as_bytes());
        h.update(b"\x00");
        h.update(a.as_bytes());
        h.update(b"\x00");
        h.update(b.as_bytes());
    })
}

/// MVP precondition matcher: each clause is `key=value`, multiple
/// clauses joined by `,`.  Matches when every clause finds its key in
/// `state` with the exact value.  An empty precondition matches any
/// state.
fn precondition_matches(rule: &str, state: &HashMap<String, String>) -> bool {
    let rule = rule.trim();
    if rule.is_empty() {
        return true;
    }
    for clause in rule.split(',') {
        let clause = clause.trim();
        let Some((k, v)) = clause.split_once('=') else {
            return false;
        };
        let k = k.trim();
        let v = v.trim();
        match state.get(k) {
            Some(actual) if actual == v => continue,
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_defaults_are_spec_aligned() {
        let c = LifecycleConfig::default();
        assert_eq!(c.promote_threshold, 3);
        assert!((c.demote_effectiveness - 0.4).abs() < 1e-9);
        assert!((c.repromote_effectiveness - 0.6).abs() < 1e-9);
    }

    #[test]
    fn precondition_empty_matches() {
        let s = HashMap::new();
        assert!(precondition_matches("", &s));
        assert!(precondition_matches("   ", &s));
    }

    #[test]
    fn precondition_single_clause() {
        let mut s = HashMap::new();
        s.insert("last_action_type".to_string(), "build".to_string());
        assert!(precondition_matches("last_action_type=build", &s));
        assert!(!precondition_matches("last_action_type=test", &s));
    }

    #[test]
    fn precondition_multi_clause_all_must_match() {
        let mut s = HashMap::new();
        s.insert("a".into(), "1".into());
        s.insert("b".into(), "2".into());
        assert!(precondition_matches("a=1,b=2", &s));
        assert!(!precondition_matches("a=1,b=3", &s));
    }

    #[test]
    fn precondition_malformed_clause_fails_closed() {
        let s = HashMap::new();
        assert!(!precondition_matches("just_a_key", &s));
    }

    #[test]
    fn stable_procedure_id_is_deterministic() {
        let a = stable_procedure_id("agent-1", "build", "test");
        let b = stable_procedure_id("agent-1", "build", "test");
        let c = stable_procedure_id("agent-1", "test", "build");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
