//! Generic execution of active Locy rules during the cortex sweep.
//!
//! Every `status = "active"` `:Rule` is run once per sweep. Stdlib rules with
//! a dedicated consumer (`episode_pattern_detector`, `contradiction_detector`)
//! perform their side effects here; any other active rule (authored / induced,
//! user-defined) is run purely so a match feeds the confidence lifecycle via
//! [`record_rule_match`]. This is what makes the lifecycle bidirectional —
//! before this, rules could only ever decay.
//!
//! `sequence_detector` and `relevance_decay` are executed by their own
//! dedicated sweeps (`run_procedure_sweep` / `run_decay_sweep`) and are skipped
//! here to avoid double execution.

use std::collections::HashMap;

use uniko_store::id::stable_hex64;
use uniko_store::schema::{edges, labels};
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, UnikoError, Value};

use super::lifecycle::{RuleLifecycleConfig, STATUS_ACTIVE, STATUS_CANDIDATE, record_rule_match};
use super::relevance_decay_params;
use crate::fact::{InvalidateFactParams, invalidate_fact};

/// Stdlib rules run by their own dedicated cortex sweeps; skipped here to avoid
/// executing them twice in one pass.
const DEDICATED_SWEEP_RULES: &[&str] = &["sequence_detector", "relevance_decay"];

/// Minimum number of contradicting episodes before a Fact is invalidated.
///
/// The `contradiction_detector` rule surfaces one row per (stale Fact,
/// contradicting Episode) pair; the threshold is applied here (the
/// `sequence_detector` precedent — Locy can't resolve a `$param` in a post-FOLD
/// HAVING clause).
const CONTRADICTION_MIN_EPISODES: usize = 2;

/// Summary of one [`run_active_rules`] pass.
#[derive(Debug, Clone, Default)]
pub struct RuleExecutionReport {
    /// Active rules executed (excluding the dedicated-sweep rules).
    pub rules_run: usize,
    /// Rules that bound at least one match this pass.
    pub rules_matched: usize,
    /// Total side effects applied (patterns upserted, facts invalidated, or —
    /// for consumer-less rules — matched rows).
    pub effects: usize,
}

/// Run every active rule for `agent_id`, applying consumers and recording
/// matches for the confidence lifecycle.
///
/// Per-rule failures are logged and skipped — rule execution must never
/// destabilize consolidation.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] only when the initial `:Rule` scan fails.
pub async fn run_active_rules(
    kb: &KnowledgeBase,
    agent_id: &str,
    cfg: RuleLifecycleConfig,
) -> Result<RuleExecutionReport, UnikoError> {
    let rules = kb.fetch_lifecycle_rules().await?;
    let mut report = RuleExecutionReport::default();

    // uni-db evaluates ALL registered rules together on any QUERY, so every
    // call must supply the union of all registered rules' parameters —
    // `relevance_decay` references `$decay_rate` / `$decay_threshold`, so an
    // unresolved param in it would fail every other rule's query too.
    let cfg_store = kb.config();
    let params = stdlib_rule_params(agent_id, cfg_store.half_life_days, cfg_store.prune_below);

    for r in rules {
        // Run active rules AND candidates: a candidate authored rule must run
        // to bind a match, which is what promotes it to active (decay only ever
        // lowers confidence, so without this candidates would never promote).
        // Stdlib rules are registered as `active`.
        let runnable = matches!(r.status.as_str(), STATUS_ACTIVE | STATUS_CANDIDATE);
        if !runnable || DEDICATED_SWEEP_RULES.contains(&r.name.as_str()) {
            continue;
        }
        report.rules_run += 1;

        let effects = match run_one_rule(kb, &r.name, agent_id, &params).await {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!(rule = %r.name, error = %e, "rule execution failed — continuing");
                continue;
            }
        };

        if effects > 0 {
            report.rules_matched += 1;
            report.effects += effects;
            // No-op for stdlib rules (fixed confidence); meaningful for
            // authored / induced rules — resets missed_cycles and rewards.
            if let Err(e) = record_rule_match(kb, &r.name, cfg).await {
                tracing::warn!(rule = %r.name, error = %e, "record_rule_match failed — continuing");
            }
        }
    }
    Ok(report)
}

/// Dispatch a single rule to its consumer, returning the number of side
/// effects (or matched rows for consumer-less rules). `params` is the union of
/// all stdlib rules' parameters (see [`run_active_rules`]).
async fn run_one_rule(
    kb: &KnowledgeBase,
    name: &str,
    agent_id: &str,
    params: &HashMap<String, Value>,
) -> Result<usize, UnikoError> {
    match name {
        "episode_pattern_detector" => consume_episode_patterns(kb, agent_id, params).await,
        "contradiction_detector" => consume_contradictions(kb, params).await,
        // Consumer-less (user-defined) rule: run it for lifecycle only.
        _ => Ok(kb.query_rule(name, &[], params).await?.len()),
    }
}

/// The union of every registered stdlib rule's parameters: `agent_id` plus
/// `relevance_decay`'s `decay_rate` / `decay_threshold`. Because uni-db
/// evaluates all registered rules together, this union must be passed to every
/// `query_rule` call. When decay is disabled (`half_life_days <= 0`), the decay
/// params are set so `relevance_decay` yields nothing (it is consumed by its own
/// sweep anyway) while staying resolvable.
///
/// This is not specific to rule execution: uni-db resolves every *registered*
/// rule as a sub-plan of **any** Locy program, including one that references no
/// rules at all. So every Locy entry point uniko exposes has to supply this
/// union, not just this module — see [`Agent::assume`](crate::Agent::assume),
/// [`Agent::abduce`](crate::Agent::abduce) and
/// [`Agent::run_rule`](crate::Agent::run_rule), which bind it as defaults.
pub(crate) fn stdlib_rule_params(
    agent_id: &str,
    half_life_days: f64,
    prune_below: f64,
) -> HashMap<String, Value> {
    let (decay_rate, decay_threshold) = if half_life_days > 0.0 {
        (std::f64::consts::LN_2 / half_life_days, prune_below)
    } else {
        // decay_rate 0 ⇒ decayed == importance; threshold -1 ⇒ never yields.
        (0.0, -1.0)
    };
    let mut params = HashMap::new();
    params.insert("agent_id".into(), Value::String(agent_id.to_string()));
    params.insert("decay_rate".into(), Value::Float(decay_rate));
    params.insert("decay_threshold".into(), Value::Float(decay_threshold));
    params
}

/// Prune an agent's Episodes that the `relevance_decay` Locy rule yields (those
/// whose age-decayed importance has fallen to or below `prune_below`).
///
/// The Locy rule is the single source of truth for the decay formula; this fn
/// performs the delete the rule cannot (`DERIVE` is additive-only). Invoked by
/// the worker's decay sweep (not [`run_active_rules`]) because it needs the
/// half-life / threshold config. A non-positive `half_life_days` disables decay.
///
/// # Errors
///
/// Returns [`UnikoError::Locy`] if the rule cannot be evaluated, or
/// [`UnikoError::Storage`] if a delete fails.
pub async fn consume_relevance_decay(
    kb: &KnowledgeBase,
    agent_id: &str,
    half_life_days: f64,
    prune_below: f64,
) -> Result<usize, UnikoError> {
    if half_life_days <= 0.0 {
        return Ok(0);
    }
    // Builds { agent_id, decay_rate = ln(2)/half_life_days, decay_threshold }.
    let params = relevance_decay_params(agent_id, half_life_days, prune_below);
    let rows = kb
        .query_rule("relevance_decay", &["episode_id"], &params)
        .await?;

    let mut pruned = 0;
    for rec in &rows {
        let Some(episode_id) = rec.get("episode_id").and_then(Value::as_str) else {
            continue;
        };
        if let Some((nid, _)) = kb
            .get_node_by_ext_id(labels::EPISODE, "episode_id", episode_id)
            .await?
            && kb.delete_node(nid).await?
        {
            pruned += 1;
        }
    }
    Ok(pruned)
}

/// Mean-importance threshold a recurring pattern must clear to be recorded
/// (the `episode_pattern_detector` rule's original `avg_importance > 0.3`).
const PATTERN_MIN_MEAN_IMPORTANCE: f64 = 0.3;

/// Upsert a `:Pattern` node for each recurring (action_type, outcome) group
/// surfaced by `episode_pattern_detector`. Returns the number of patterns
/// upserted.
///
/// The rule only supplies `support` (`COUNT`); `mean_importance` is computed
/// here in Rust because a FOLD `AVG` comes back 0.0 when the rule's YIELD
/// renames the aggregate to a different alias (uni-db #145). Groups whose mean
/// importance is at or below [`PATTERN_MIN_MEAN_IMPORTANCE`] are skipped,
/// preserving the rule's original intent.
async fn consume_episode_patterns(
    kb: &KnowledgeBase,
    agent_id: &str,
    params: &HashMap<String, Value>,
) -> Result<usize, UnikoError> {
    let rows = kb
        .query_rule(
            "episode_pattern_detector",
            &["action_type", "outcome", "support"],
            params,
        )
        .await?;

    let now = datetime_value(chrono::Utc::now());
    let mut count = 0;
    for rec in &rows {
        let (Some(action_type), Some(outcome)) = (
            rec.get("action_type").and_then(Value::as_str),
            rec.get("outcome").and_then(Value::as_str),
        ) else {
            continue;
        };
        let support = rec.get("support").and_then(Value::as_i64).unwrap_or(0);
        let mean_importance = kb
            .mean_episode_importance(agent_id, action_type, outcome)
            .await?;
        if mean_importance <= PATTERN_MIN_MEAN_IMPORTANCE {
            continue;
        }

        let pattern_id = stable_hex64("pattern", |h| {
            h.update(agent_id.as_bytes());
            h.update(b"\x00");
            h.update(action_type.as_bytes());
            h.update(b"\x00");
            h.update(outcome.as_bytes());
        });

        let existing = kb
            .get_node_by_ext_id(labels::PATTERN, "pattern_id", &pattern_id)
            .await?;

        let mut props = HashMap::new();
        props.insert("action_type".into(), Value::String(action_type.to_string()));
        props.insert("outcome".into(), Value::String(outcome.to_string()));
        props.insert("support".into(), Value::Int(support));
        props.insert("mean_importance".into(), Value::Float(mean_importance));
        props.insert("status".into(), Value::String("active".into()));
        if existing.is_none() {
            props.insert("created_at".into(), now.clone());
        }
        props.insert("last_seen_at".into(), now.clone());

        kb.merge_node(labels::PATTERN, "pattern_id", &pattern_id, &props)
            .await?;
        count += 1;
    }
    Ok(count)
}

/// Invalidate Facts contradicted by enough episodes (per
/// `contradiction_detector`) and draw a `CONTRADICTED_BY` marker edge to each
/// contradicting episode. Coexists with the F38 embedding-vote path — this is
/// an additional, episode-outcome-driven signal. Returns the number of Facts
/// invalidated.
async fn consume_contradictions(
    kb: &KnowledgeBase,
    params: &HashMap<String, Value>,
) -> Result<usize, UnikoError> {
    let rows = kb
        .query_rule(
            "contradiction_detector",
            &["stale_fact", "episode_id"],
            params,
        )
        .await?;

    // Group contradicting episodes by stale fact so the threshold and the
    // single invalidation are applied per Fact.
    let mut by_fact: HashMap<String, Vec<String>> = HashMap::new();
    for rec in &rows {
        let (Some(fact_id), Some(episode_id)) = (
            rec.get("stale_fact").and_then(Value::as_str),
            rec.get("episode_id").and_then(Value::as_str),
        ) else {
            continue;
        };
        by_fact
            .entry(fact_id.to_string())
            .or_default()
            .push(episode_id.to_string());
    }

    let now = chrono::Utc::now();
    let mut invalidated = 0;
    for (fact_id, episodes) in by_fact {
        if episodes.len() < CONTRADICTION_MIN_EPISODES {
            continue;
        }

        // Resolve the fact node up-front for the marker edges; skip if it has
        // vanished since the rule ran.
        let Some((fact_nid, _)) = kb
            .get_node_by_ext_id(labels::FACT, "fact_id", &fact_id)
            .await?
        else {
            continue;
        };

        invalidate_fact(
            kb,
            InvalidateFactParams {
                fact_id: fact_id.clone(),
                replacement_fact_id: None,
                reason: Some(format!("contradicted by {} episodes", episodes.len())),
                now: Some(now),
            },
        )
        .await?;
        invalidated += 1;

        // Best-effort provenance edges — a duplicate or missing endpoint must
        // not abort the invalidation we already committed.
        for episode_id in &episodes {
            let Some((ep_nid, _)) = kb
                .get_node_by_ext_id(labels::EPISODE, "episode_id", episode_id)
                .await?
            else {
                continue;
            };
            let mut props = HashMap::new();
            props.insert("detected_at".into(), datetime_value(now));
            if let Err(e) = kb
                .create_edge(edges::CONTRADICTED_BY, fact_nid, ep_nid, &props)
                .await
            {
                tracing::debug!(
                    fact = %fact_id,
                    episode = %episode_id,
                    error = %e,
                    "CONTRADICTED_BY edge create failed — continuing",
                );
            }
        }
    }
    Ok(invalidated)
}
