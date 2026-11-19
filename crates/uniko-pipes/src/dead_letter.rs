//! Persistent dead-letter queue backed by the knowledge graph.
//!
//! A streaming ingest item whose step fails is stored as a `DeadLetter`
//! node together with its payload, so it can be inspected, retried up to a
//! bound and resolved. Retries replay the stored [`IngestTask`] unchanged,
//! so the agent and source it was submitted under carry over, and ingest's
//! own id and content dedup keep a replay from writing twice.
//!
//! Lifecycle (`status`):
//!
//! - `pending`: waiting for a retry (`next_retry_at` says when one is due).
//! - `retrying`: a retry has been claimed and resubmitted.
//! - `exhausted`: the retry limit is reached, or the error cannot succeed on
//!   a retry. It stays for inspection until resolved.
//! - `resolved`: a retry succeeded, or the host resolved it.
//!
//! Rows written before this lifecycle existed have no payload or status;
//! they list as `legacy` and can only be resolved.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uniko_store::schema::constants::labels::DEAD_LETTER;
use uniko_store::temporal::TemporalValue;
use uniko_store::{KnowledgeBase, NodeId, UnikoError, Value, datetime_value};

use crate::types::IngestTask;

/// Where a dead letter is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DeadLetterStatus {
    /// Waiting for a retry.
    Pending,
    /// A retry has been claimed and resubmitted.
    Retrying,
    /// Out of retries, or failed in a way a retry cannot fix.
    Exhausted,
    /// Retried successfully, or resolved by the host.
    Resolved,
    /// Written before payloads were kept: not replayable.
    Legacy,
}

impl DeadLetterStatus {
    /// The stable lowercase name stored in the graph.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Retrying => "retrying",
            Self::Exhausted => "exhausted",
            Self::Resolved => "resolved",
            Self::Legacy => "legacy",
        }
    }

    /// Decode a stored status: anything unknown or missing is `Legacy`.
    fn parse(s: Option<&str>) -> Self {
        match s {
            Some("pending") => Self::Pending,
            Some("retrying") => Self::Retrying,
            Some("exhausted") => Self::Exhausted,
            Some("resolved") => Self::Resolved,
            _ => Self::Legacy,
        }
    }
}

impl std::str::FromStr for DeadLetterStatus {
    type Err = UnikoError;

    /// Parse a status name: `pending`, `retrying`, `exhausted`, `resolved`
    /// or `legacy`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "retrying" => Ok(Self::Retrying),
            "exhausted" => Ok(Self::Exhausted),
            "resolved" => Ok(Self::Resolved),
            "legacy" => Ok(Self::Legacy),
            other => Err(UnikoError::Config(format!(
                "unknown dead-letter status '{other}'; expected pending, retrying, \
                 exhausted, resolved or legacy"
            ))),
        }
    }
}

impl std::fmt::Display for DeadLetterStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One dead-lettered item, as read back from the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DeadLetter {
    /// Stable id; `legacy:{node_id}` for a row written before ids existed.
    pub id: String,
    /// The step that failed.
    pub step: String,
    /// The latest failure's message.
    pub error: String,
    /// The latest failure's [`UnikoError`] variant, e.g. `embedding`.
    pub error_kind: Option<String>,
    /// Whether the latest failure can succeed on a retry.
    pub retriable: bool,
    /// Where the letter is in its lifecycle.
    pub status: DeadLetterStatus,
    /// Retries attempted so far.
    pub retry_count: u32,
    /// Retries allowed before the letter is exhausted.
    pub max_retries: u32,
    /// `message`, `artifact`, `pdf` or `source`.
    pub ingest_type: Option<String>,
    /// The agent the item was submitted under, when known.
    pub agent_id: Option<String>,
    /// The stored payload can be replayed.
    pub replayable: bool,
    /// When the item was first dead-lettered.
    pub created_at: Option<DateTime<Utc>>,
    /// When the letter last changed.
    pub updated_at: Option<DateTime<Utc>>,
    /// When the next retry is due, for a pending letter.
    pub next_retry_at: Option<DateTime<Utc>>,
    /// When the letter was resolved.
    pub resolved_at: Option<DateTime<Utc>>,
    /// Why it was resolved: `retried`, or the host's note.
    pub resolution: Option<String>,
}

/// Which dead letters [`DeadLetterQueue::list`] returns.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DeadLetterFilter {
    /// Only letters in this status.
    pub status: Option<DeadLetterStatus>,
    /// At most this many, oldest first.
    pub limit: Option<usize>,
}

impl DeadLetterFilter {
    /// Every letter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Only letters in `status`.
    #[must_use]
    pub fn status(mut self, status: DeadLetterStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// At most `limit` letters.
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// How many dead letters are in each status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DeadLetterCounts {
    /// Waiting for a retry.
    pub pending: u64,
    /// Being retried.
    pub retrying: u64,
    /// Out of retries or not retriable.
    pub exhausted: u64,
    /// Retried successfully or resolved by the host.
    pub resolved: u64,
    /// Written before payloads were kept.
    pub legacy: u64,
}

/// Retry limit and backoff for dead letters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RetryPolicy {
    /// Retries before a letter is exhausted.
    pub max_retries: u32,
    /// Wait before the first retry, in milliseconds.
    pub backoff_ms: u64,
    /// Longest wait between retries, in milliseconds.
    pub max_backoff_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff_ms: 30_000,
            max_backoff_ms: 3_600_000,
        }
    }
}

impl RetryPolicy {
    /// A policy allowing `max_retries` retries, waiting `backoff_ms` before
    /// the first and doubling up to `max_backoff_ms`.
    #[must_use]
    pub fn new(max_retries: u32, backoff_ms: u64, max_backoff_ms: u64) -> Self {
        Self {
            max_retries,
            backoff_ms,
            max_backoff_ms,
        }
    }

    /// The wait before retry number `attempt + 1`: the base backoff doubled
    /// `attempt` times, capped at the maximum.
    #[must_use]
    pub fn backoff_after(&self, attempt: u32) -> chrono::Duration {
        let factor = 1u64.checked_shl(attempt.min(62)).unwrap_or(u64::MAX);
        let ms = self
            .backoff_ms
            .saturating_mul(factor)
            .min(self.max_backoff_ms);
        chrono::Duration::milliseconds(i64::try_from(ms).unwrap_or(i64::MAX))
    }
}

/// Whether `err` can succeed on a retry of the same payload.
///
/// A conflicting id, an unsupported or invalid payload, and configuration or
/// schema errors fail the same way every time; model, timeout, storage and
/// internal failures may clear up.
#[must_use]
pub fn is_retriable(err: &UnikoError) -> bool {
    !matches!(
        err,
        UnikoError::IdConflict(_)
            | UnikoError::Unsupported(_)
            | UnikoError::Pipeline(_)
            | UnikoError::Config(_)
            | UnikoError::Schema(_)
            | UnikoError::Locy(_)
    )
}

/// The stable name of `err`'s variant, stored as `error_kind`.
#[must_use]
pub fn error_kind(err: &UnikoError) -> &'static str {
    match err {
        UnikoError::Storage(_) => "storage",
        UnikoError::Search(_) => "search",
        UnikoError::Schema(_) => "schema",
        UnikoError::Pipeline(_) => "pipeline",
        UnikoError::Locy(_) => "locy",
        UnikoError::Config(_) => "config",
        UnikoError::Embedding(_) => "embedding",
        UnikoError::Llm(_) => "llm",
        UnikoError::Timeout(_) => "timeout",
        UnikoError::Conflict(_) => "conflict",
        UnikoError::IdConflict(_) => "id_conflict",
        UnikoError::Internal(_) => "internal",
        UnikoError::Unsupported(_) => "unsupported",
        UnikoError::NotReady(_) => "not_ready",
        UnikoError::Degraded(_) => "degraded",
    }
}

fn ingest_type(task: &IngestTask) -> &'static str {
    match task {
        IngestTask::Message(_) => "message",
        IngestTask::Artifact(_) => "artifact",
        IngestTask::Pdf(_) => "pdf",
        IngestTask::Source(_) => "source",
    }
}

fn task_agent_id(task: &IngestTask) -> Option<String> {
    let metadata = match task {
        IngestTask::Message(m) => &m.metadata,
        IngestTask::Artifact(a) => &a.metadata,
        IngestTask::Pdf(p) => &p.metadata,
        IngestTask::Source(s) => &s.metadata,
    };
    metadata
        .get("agent_id")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

const LEGACY_PREFIX: &str = "legacy:";

/// The DeadLetter properties [`DeadLetterQueue::list`] reads: all but the
/// payload.
const LIST_FIELDS: &[&str] = &[
    "dead_letter_id",
    "step",
    "error",
    "error_kind",
    "retriable",
    "status",
    "retry_count",
    "max_retries",
    "ingest_type",
    "agent_id",
    "created_at",
    "updated_at",
    "next_retry_at",
    "resolved_at",
    "resolution",
];

/// Persistent dead-letter queue using `DeadLetter` graph nodes.
#[derive(Debug, Clone)]
pub struct DeadLetterQueue {
    kb: Arc<KnowledgeBase>,
    policy: RetryPolicy,
    /// Serializes read-check-write transitions within this process, so two
    /// callers can't both claim the same retry, and holds the ids of the
    /// retries this process has claimed and not yet finished. A letter
    /// marked `retrying` that is not in this set was left by a process that
    /// stopped mid-retry.
    transitions: Arc<Mutex<HashSet<String>>>,
}

impl DeadLetterQueue {
    /// Wrap a knowledge base for DLQ operations, with the default policy.
    pub fn new(kb: Arc<KnowledgeBase>) -> Self {
        Self::with_policy(kb, RetryPolicy::default())
    }

    /// Wrap a knowledge base with a retry policy.
    pub fn with_policy(kb: Arc<KnowledgeBase>, policy: RetryPolicy) -> Self {
        Self {
            kb,
            policy,
            transitions: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// The retry policy in force.
    #[must_use]
    pub fn policy(&self) -> RetryPolicy {
        self.policy
    }

    /// Store a failed item, keeping its payload for retries. Returns the
    /// new letter's id.
    ///
    /// A retriable failure is stored `pending`, with its first retry due
    /// after the base backoff. It is stored `exhausted` instead when the
    /// failure cannot succeed on a retry, when the policy allows no retries,
    /// or when there is no payload to replay: no `task` (a failure outside
    /// an ingest item), or one that could not be serialized (for example a
    /// path that isn't valid UTF-8), in which case the letter is still kept
    /// for inspection.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn store(
        &self,
        step: &str,
        error: &UnikoError,
        task: Option<&IngestTask>,
        node_ref: NodeId,
    ) -> Result<String, UnikoError> {
        let id = uniko_store::id::new_id();
        let now = Utc::now();
        let retriable = is_retriable(error);
        let payload = task.and_then(|t| match serde_json::to_string(t) {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!(error = %e, "dead letter payload not serializable; storing it unreplayable");
                None
            }
        });
        let replayable = payload.is_some();
        let status = if retriable && replayable && self.policy.max_retries > 0 {
            DeadLetterStatus::Pending
        } else {
            DeadLetterStatus::Exhausted
        };

        let mut props = HashMap::new();
        props.insert("dead_letter_id".into(), Value::String(id.clone()));
        props.insert("step".into(), Value::String(step.to_string()));
        props.insert("error".into(), Value::String(error.to_string()));
        props.insert("error_kind".into(), Value::String(error_kind(error).into()));
        props.insert("retriable".into(), Value::Bool(retriable));
        props.insert("node_ref".into(), Value::Int(node_ref));
        props.insert("retry_count".into(), Value::Int(0));
        props.insert(
            "max_retries".into(),
            Value::Int(i64::from(self.policy.max_retries)),
        );
        props.insert("status".into(), Value::String(status.as_str().into()));
        props.insert("created_at".into(), datetime_value(now));
        props.insert("updated_at".into(), datetime_value(now));
        if status == DeadLetterStatus::Pending {
            props.insert(
                "next_retry_at".into(),
                datetime_value(now + self.policy.backoff_after(0)),
            );
        }
        if let Some(t) = task {
            props.insert("ingest_type".into(), Value::String(ingest_type(t).into()));
            if let Some(agent) = task_agent_id(t) {
                props.insert("agent_id".into(), Value::String(agent));
            }
        }
        if let Some(p) = payload {
            props.insert("payload".into(), Value::String(p));
        }
        self.kb.create_node(DEAD_LETTER, &props).await?;
        Ok(id)
    }

    /// Dead letters matching `filter`, oldest first.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn list(&self, filter: &DeadLetterFilter) -> Result<Vec<DeadLetter>, UnikoError> {
        // Every field but the payload, which can be large: listing and the
        // scheduler's `due` never need it, and `begin_retry` reads it back.
        let (filter_clause, params) = match filter.status {
            None => (String::new(), HashMap::new()),
            Some(DeadLetterStatus::Legacy) => {
                ("WHERE d.status IS NULL".to_string(), HashMap::new())
            }
            Some(status) => (
                "WHERE d.status = $status".to_string(),
                HashMap::from([(
                    "status".to_string(),
                    Value::String(status.as_str().to_string()),
                )]),
            ),
        };
        let fields = LIST_FIELDS
            .iter()
            .map(|f| format!("d.{f} AS {f}"))
            .collect::<Vec<_>>()
            .join(", ");
        let cypher = format!(
            "MATCH (d:DeadLetter) {filter_clause} \
             RETURN id(d) AS nid, {fields}, d.payload IS NOT NULL AS has_payload"
        );
        let mut letters: Vec<DeadLetter> = self
            .kb
            .query_cypher(&cypher, &params)
            .await?
            .into_iter()
            .filter_map(|row| match row.get("nid") {
                Some(Value::Int(nid)) => Some(decode(*nid, &row)),
                _ => None,
            })
            .collect();
        letters.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
        if let Some(limit) = filter.limit {
            letters.truncate(limit);
        }
        Ok(letters)
    }

    /// One dead letter by id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn get(&self, id: &str) -> Result<Option<DeadLetter>, UnikoError> {
        Ok(self.find(id).await?.map(|(nid, props)| decode(nid, &props)))
    }

    /// How many letters are in each status.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn counts(&self) -> Result<DeadLetterCounts, UnikoError> {
        let rows = self
            .kb
            .query_cypher(
                "MATCH (d:DeadLetter) RETURN d.status AS status, count(d) AS c",
                &HashMap::new(),
            )
            .await?;
        let mut counts = DeadLetterCounts::default();
        for row in rows {
            let status = match row.get("status") {
                Some(Value::String(s)) => Some(s.as_str()),
                _ => None,
            };
            let n = match row.get("c") {
                Some(Value::Int(n)) => u64::try_from(*n).unwrap_or(0),
                _ => 0,
            };
            match DeadLetterStatus::parse(status) {
                DeadLetterStatus::Pending => counts.pending += n,
                DeadLetterStatus::Retrying => counts.retrying += n,
                DeadLetterStatus::Exhausted => counts.exhausted += n,
                DeadLetterStatus::Resolved => counts.resolved += n,
                DeadLetterStatus::Legacy => counts.legacy += n,
            }
        }
        Ok(counts)
    }

    /// Claim a retry of `id`: count the attempt, mark the letter `retrying`
    /// and return its payload to resubmit.
    ///
    /// Refused with [`UnikoError::Config`] when the letter is unknown,
    /// resolved, already being retried, has no payload, or is out of
    /// retries; and when its failure cannot succeed on a retry, unless
    /// `force` is set.
    ///
    /// # Errors
    ///
    /// As above, or [`UnikoError::Storage`] on graph failure.
    pub async fn begin_retry(&self, id: &str, force: bool) -> Result<IngestTask, UnikoError> {
        let mut in_flight = self.transitions.lock().await;
        let (nid, props) = self
            .find(id)
            .await?
            .ok_or_else(|| UnikoError::Config(format!("no dead letter '{id}'")))?;
        let letter = decode(nid, &props);
        let refuse = |why: &str| Err(UnikoError::Config(format!("dead letter '{id}' {why}")));
        match letter.status {
            DeadLetterStatus::Resolved => return refuse("is already resolved"),
            DeadLetterStatus::Retrying => return refuse("is already being retried"),
            DeadLetterStatus::Legacy => {
                return refuse("predates stored payloads and cannot be retried");
            }
            DeadLetterStatus::Pending | DeadLetterStatus::Exhausted => {}
        }
        if letter.retry_count >= letter.max_retries && !force {
            return refuse("has used all its retries; pass force to retry anyway");
        }
        if !letter.retriable && !force {
            return refuse(&format!(
                "failed with a {} error a retry cannot fix; pass force to retry anyway",
                letter.error_kind.as_deref().unwrap_or("permanent")
            ));
        }
        let payload = match props.get("payload") {
            Some(Value::String(p)) => p,
            _ => return refuse("has no stored payload"),
        };
        let task: IngestTask = serde_json::from_str(payload).map_err(|e| {
            UnikoError::Internal(format!(
                "dead letter '{id}': stored payload is invalid: {e}"
            ))
        })?;
        let mut update = HashMap::new();
        update.insert(
            "retry_count".into(),
            Value::Int(i64::from(letter.retry_count) + 1),
        );
        update.insert(
            "status".into(),
            Value::String(DeadLetterStatus::Retrying.as_str().into()),
        );
        update.insert("updated_at".into(), datetime_value(Utc::now()));
        update.insert("next_retry_at".into(), Value::Null);
        self.kb.update_node(nid, &update).await?;
        in_flight.insert(id.to_string());
        Ok(task)
    }

    /// Undo a claim from [`begin_retry`](Self::begin_retry) whose payload
    /// could not be resubmitted (the ingest queue was full or shutting
    /// down): the attempt is not counted and the letter is pending again.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn release_claim(&self, id: &str) -> Result<(), UnikoError> {
        let mut in_flight = self.transitions.lock().await;
        in_flight.remove(id);
        let Some((nid, props)) = self.find(id).await? else {
            return Ok(());
        };
        self.uncount_claim(nid, &decode(nid, &props)).await
    }

    /// Put a `retrying` letter back to `pending`, due now, without counting
    /// the attempt it was claimed for.
    async fn uncount_claim(&self, nid: NodeId, letter: &DeadLetter) -> Result<(), UnikoError> {
        if letter.status != DeadLetterStatus::Retrying {
            return Ok(());
        }
        let now = Utc::now();
        let mut update = HashMap::new();
        update.insert(
            "retry_count".into(),
            Value::Int(i64::from(letter.retry_count.saturating_sub(1))),
        );
        update.insert(
            "status".into(),
            Value::String(DeadLetterStatus::Pending.as_str().into()),
        );
        update.insert("next_retry_at".into(), datetime_value(now));
        update.insert("updated_at".into(), datetime_value(now));
        self.kb.update_node(nid, &update).await
    }

    /// Record that a claimed retry of `id` failed again with `error`.
    ///
    /// The letter goes back to `pending` with its next retry due after the
    /// doubled backoff, or to `exhausted` once its retries are used up or
    /// the new failure cannot succeed on a retry.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn retry_failed(
        &self,
        id: &str,
        step: &str,
        error: &UnikoError,
    ) -> Result<(), UnikoError> {
        let mut in_flight = self.transitions.lock().await;
        in_flight.remove(id);
        let Some((nid, props)) = self.find(id).await? else {
            return Ok(());
        };
        let letter = decode(nid, &props);
        let retriable = is_retriable(error);
        let now = Utc::now();
        let mut update = HashMap::new();
        update.insert("step".into(), Value::String(step.to_string()));
        update.insert("error".into(), Value::String(error.to_string()));
        update.insert("error_kind".into(), Value::String(error_kind(error).into()));
        update.insert("retriable".into(), Value::Bool(retriable));
        update.insert("updated_at".into(), datetime_value(now));
        if retriable && letter.retry_count < letter.max_retries {
            update.insert(
                "status".into(),
                Value::String(DeadLetterStatus::Pending.as_str().into()),
            );
            update.insert(
                "next_retry_at".into(),
                datetime_value(now + self.policy.backoff_after(letter.retry_count)),
            );
        } else {
            update.insert(
                "status".into(),
                Value::String(DeadLetterStatus::Exhausted.as_str().into()),
            );
            update.insert("next_retry_at".into(), Value::Null);
        }
        self.kb.update_node(nid, &update).await?;
        Ok(())
    }

    /// Record that a claimed retry of `id` succeeded.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn retry_succeeded(&self, id: &str) -> Result<(), UnikoError> {
        let mut in_flight = self.transitions.lock().await;
        in_flight.remove(id);
        if let Some((nid, _)) = self.find(id).await? {
            self.mark_resolved(nid, "retried").await?;
        }
        Ok(())
    }

    /// Resolve `id` without retrying it, recording `note` as the reason.
    /// Nothing outside the letter is changed, and its stored payload is
    /// dropped. Returns `false` when there is no such letter or it was
    /// already resolved.
    ///
    /// A letter `retrying` because a process stopped mid-retry can be
    /// resolved; only a retry in flight in this process is refused.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] while a retry of it is in flight here,
    /// or [`UnikoError::Storage`] on graph failure.
    pub async fn resolve(&self, id: &str, note: Option<&str>) -> Result<bool, UnikoError> {
        let in_flight = self.transitions.lock().await;
        let Some((nid, props)) = self.find(id).await? else {
            return Ok(false);
        };
        let letter = decode(nid, &props);
        match letter.status {
            DeadLetterStatus::Resolved => Ok(false),
            DeadLetterStatus::Retrying if in_flight.contains(&letter.id) => {
                Err(UnikoError::Config(format!(
                    "dead letter '{id}' is being retried; resolve it once the retry finishes"
                )))
            }
            _ => {
                self.mark_resolved(nid, note.unwrap_or("resolved")).await?;
                Ok(true)
            }
        }
    }

    /// Pending, retriable letters whose next retry is due by `now`, oldest
    /// due first, at most `limit`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn due(&self, now: DateTime<Utc>, limit: usize) -> Result<Vec<String>, UnikoError> {
        let mut due: Vec<(Option<DateTime<Utc>>, String)> = self
            .list(&DeadLetterFilter::new().status(DeadLetterStatus::Pending))
            .await?
            .into_iter()
            .filter(|l| {
                l.retriable
                    && l.replayable
                    && l.retry_count < l.max_retries
                    && l.next_retry_at.is_none_or(|at| at <= now)
            })
            .map(|l| (l.next_retry_at, l.id))
            .collect();
        due.sort();
        Ok(due.into_iter().take(limit).map(|(_, id)| id).collect())
    }

    /// Return letters left `retrying` by a process that stopped mid-retry
    /// to `pending`, due now, without counting the attempt that never
    /// finished. Retries claimed by this process are left alone. A pipeline
    /// runs this when it starts. Returns how many were reset.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn reset_interrupted(&self) -> Result<usize, UnikoError> {
        let in_flight = self.transitions.lock().await;
        let mut reset = 0;
        for letter in self
            .list(&DeadLetterFilter::new().status(DeadLetterStatus::Retrying))
            .await?
        {
            if in_flight.contains(&letter.id) {
                continue;
            }
            let Some((nid, props)) = self.find(&letter.id).await? else {
                continue;
            };
            self.uncount_claim(nid, &decode(nid, &props)).await?;
            reset += 1;
        }
        Ok(reset)
    }

    async fn mark_resolved(&self, nid: NodeId, resolution: &str) -> Result<(), UnikoError> {
        let now = Utc::now();
        let mut update = HashMap::new();
        update.insert(
            "status".into(),
            Value::String(DeadLetterStatus::Resolved.as_str().into()),
        );
        update.insert("resolution".into(), Value::String(resolution.to_string()));
        update.insert("resolved_at".into(), datetime_value(now));
        update.insert("updated_at".into(), datetime_value(now));
        update.insert("next_retry_at".into(), Value::Null);
        // A resolved letter is never replayed, so its payload, which can be
        // large, is dropped.
        update.insert("payload".into(), Value::Null);
        self.kb.update_node(nid, &update).await
    }

    async fn find(&self, id: &str) -> Result<Option<(NodeId, HashMap<String, Value>)>, UnikoError> {
        if let Some(raw) = id.strip_prefix(LEGACY_PREFIX) {
            let Ok(nid) = raw.parse::<NodeId>() else {
                return Ok(None);
            };
            return Ok(self
                .kb
                .get_node(nid)
                .await?
                .filter(|(label, _)| label == DEAD_LETTER)
                .map(|(_, props)| (nid, props)));
        }
        self.kb
            .get_node_by_ext_id(DEAD_LETTER, "dead_letter_id", id)
            .await
    }
}

fn string(props: &HashMap<String, Value>, key: &str) -> Option<String> {
    match props.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn count(props: &HashMap<String, Value>, key: &str) -> u32 {
    match props.get(key) {
        Some(Value::Int(n)) => u32::try_from(*n).unwrap_or(0),
        _ => 0,
    }
}

fn time(props: &HashMap<String, Value>, key: &str) -> Option<DateTime<Utc>> {
    match props.get(key) {
        Some(Value::Temporal(t @ TemporalValue::DateTime { .. })) => {
            uniko_store::temporal_epoch_millis(t).and_then(DateTime::from_timestamp_millis)
        }
        _ => None,
    }
}

fn decode(nid: NodeId, props: &HashMap<String, Value>) -> DeadLetter {
    let status = DeadLetterStatus::parse(string(props, "status").as_deref());
    let id = string(props, "dead_letter_id").unwrap_or_else(|| format!("{LEGACY_PREFIX}{nid}"));
    DeadLetter {
        id,
        step: string(props, "step").unwrap_or_default(),
        error: string(props, "error").unwrap_or_default(),
        error_kind: string(props, "error_kind"),
        retriable: matches!(props.get("retriable"), Some(Value::Bool(true))),
        status,
        retry_count: count(props, "retry_count"),
        max_retries: count(props, "max_retries"),
        ingest_type: string(props, "ingest_type"),
        agent_id: string(props, "agent_id"),
        replayable: status != DeadLetterStatus::Legacy
            && (matches!(props.get("payload"), Some(Value::String(_)))
                || matches!(props.get("has_payload"), Some(Value::Bool(true)))),
        created_at: time(props, "created_at"),
        updated_at: time(props, "updated_at"),
        next_retry_at: time(props, "next_retry_at"),
        resolved_at: time(props, "resolved_at"),
        resolution: string(props, "resolution"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_and_caps() {
        let p = RetryPolicy::new(5, 1_000, 5_000);
        let ms = |n| p.backoff_after(n).num_milliseconds();
        assert_eq!(ms(0), 1_000);
        assert_eq!(ms(1), 2_000);
        assert_eq!(ms(2), 4_000);
        assert_eq!(ms(3), 5_000);
        assert_eq!(ms(200), 5_000);
    }

    #[test]
    fn status_names_round_trip() {
        for s in [
            DeadLetterStatus::Pending,
            DeadLetterStatus::Retrying,
            DeadLetterStatus::Exhausted,
            DeadLetterStatus::Resolved,
            DeadLetterStatus::Legacy,
        ] {
            assert_eq!(s.as_str().parse::<DeadLetterStatus>().unwrap(), s);
            assert_eq!(s.to_string(), s.as_str());
        }
        assert!("stuck".parse::<DeadLetterStatus>().is_err());
    }

    #[test]
    fn permanent_errors_are_not_retriable() {
        assert!(!is_retriable(&UnikoError::IdConflict("x".into())));
        assert!(!is_retriable(&UnikoError::Unsupported("x".into())));
        assert!(!is_retriable(&UnikoError::Pipeline("bad payload".into())));
        assert!(is_retriable(&UnikoError::Embedding("offline".into())));
        assert!(is_retriable(&UnikoError::Internal("captioner".into())));
        assert!(is_retriable(&UnikoError::Timeout(5)));
    }
}
