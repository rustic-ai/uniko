//! Ingest pipeline worker with biased select, semaphore concurrency,
//! and per-item step chain execution.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{Semaphore, mpsc};
use tokio_util::sync::CancellationToken;

use uniko_pipes::circuit_breaker::CircuitBreaker;
use uniko_pipes::dead_letter::DeadLetterQueue;
use uniko_pipes::health::HealthTracker;
use uniko_pipes::metrics;
use uniko_pipes::step::{PipelineContext, Step};
use uniko_pipes::types::{
    ConsolidationTask, IngestTask, ItemResult, ObservationsReady, StepErrorPolicy, StepOutcome,
};
use uniko_store::{KnowledgeBase, UnikoError};

/// One item on the ingest channel: a task, and the dead letter it retries
/// when it is a retry.
#[derive(Debug)]
pub(crate) struct IngestJob {
    pub(crate) task: IngestTask,
    pub(crate) retry_of: Option<String>,
}

/// Long-running ingest worker receiving tasks via a bounded channel.
pub(crate) struct IngestWorker {
    rx: mpsc::Receiver<IngestJob>,
    steps: Arc<Vec<Box<dyn Step>>>,
    semaphore: Arc<Semaphore>,
    cancel: CancellationToken,
    kb: Arc<KnowledgeBase>,
    llm_breaker: Arc<CircuitBreaker>,
    dlq: Arc<DeadLetterQueue>,
    health: Arc<Mutex<HealthTracker>>,
    /// Sender for notifying the consolidation worker that a batch of
    /// Observations landed. Without this the per-agent counter never
    /// advances and neither the threshold nor the periodic timer fires.
    consolidation_tx: mpsc::Sender<ConsolidationTask>,
    /// Count of submitted-but-not-yet-finished ingest items.
    ///
    /// Incremented at submit time and decremented when a spawned item
    /// finishes; [`PipelineSystem::quiesce`](super::PipelineSystem::quiesce)
    /// awaits this reaching zero.
    inflight: Arc<AtomicUsize>,
}

impl IngestWorker {
    #[expect(
        clippy::too_many_arguments,
        reason = "constructor assembles all worker dependencies"
    )]
    pub(crate) fn new(
        rx: mpsc::Receiver<IngestJob>,
        steps: Vec<Box<dyn Step>>,
        concurrency: usize,
        cancel: CancellationToken,
        kb: Arc<KnowledgeBase>,
        llm_breaker: Arc<CircuitBreaker>,
        dlq: Arc<DeadLetterQueue>,
        health: Arc<Mutex<HealthTracker>>,
        consolidation_tx: mpsc::Sender<ConsolidationTask>,
        inflight: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            rx,
            steps: Arc::new(steps),
            semaphore: Arc::new(Semaphore::new(concurrency)),
            cancel,
            kb,
            llm_breaker,
            dlq,
            health,
            consolidation_tx,
            inflight,
        }
    }

    /// Run the worker loop until cancelled or the channel closes.
    pub(crate) async fn run(mut self) {
        tracing::info!("ingest worker started");
        loop {
            tokio::select! {
                biased;

                () = self.cancel.cancelled() => {
                    tracing::info!("ingest worker shutting down");
                    break;
                }

                task = self.rx.recv() => {
                    match task {
                        Some(job) => self.process(job),
                        None => {
                            tracing::info!("ingest channel closed");
                            break;
                        }
                    }
                }
            }
        }
        tracing::info!("ingest worker stopped");
    }

    /// Spawn a concurrent task for one ingest item.
    fn process(&self, job: IngestJob) {
        let IngestJob { task, retry_of } = job;
        let permit = self.semaphore.clone();
        let steps = self.steps.clone();
        let item_cancel = self.cancel.child_token();
        let kb = self.kb.clone();
        let breaker = self.llm_breaker.clone();
        let dlq = self.dlq.clone();
        let health = self.health.clone();
        let consolidation_tx = self.consolidation_tx.clone();
        let inflight = self.inflight.clone();

        tokio::spawn(async move {
            // Decrement the in-flight counter on every exit path (including
            // the early shutdown return below), so `quiesce` is exact.
            let _inflight_guard = InflightGuard(inflight);

            // Acquire semaphore permit (bounds concurrency).  An `Err`
            // here means the semaphore was closed during shutdown —
            // nothing to do, drop the task.
            let Ok(_permit) = permit.acquire().await else {
                return;
            };

            let start = std::time::Instant::now();

            // Extract content from the task.
            let (content, content_type) = match &task {
                IngestTask::Message(m) => (m.content.clone(), m.content_type.clone()),
                IngestTask::Artifact(a) => (a.content.clone(), a.kind.clone()),
                // PDF ingest is routed through `uniko_extract::ingest_pdf`
                // directly; it never lands on the generic worker step
                // chain because `content` is binary (not UTF-8 String)
                // and chunker selection is handled internally.
                IngestTask::Pdf(p) => (String::new(), format!("pdf:{}", p.artifact_id)),
                // Source ingest is MIME-routed inside the step from the
                // payload; the worker only needs a label here.
                IngestTask::Source(_) => (String::new(), "source".to_string()),
            };

            let mut ctx = PipelineContext::new(
                0, // Node ID set by the first step (P1 ingest).
                content,
                content_type,
                item_cancel,
                kb,
                breaker,
            );

            // Hand `IngestStep` the typed payload it deserializes. Without
            // these keys the step has nothing to ingest. PDFs are routed
            // through `ingest_pdf` directly, so they carry no payload.
            populate_ingest_metadata(&task, &mut ctx);

            metrics::emit_ingest_item();
            let (chain, diagnostics) =
                uniko_store::diagnostics::collect(run_step_chain(&steps, &mut ctx)).await;
            let elapsed_ms = start.elapsed().as_millis() as f64;
            let result = record_dead_letter(
                &dlq,
                &health,
                &task,
                retry_of.as_deref(),
                chain,
                ctx.node_id,
            )
            .await;

            // Tell the consolidation worker that new Observations landed, so
            // its per-agent counter advances and the threshold / periodic
            // timer can fire. Best-effort: a full or closed channel must not
            // fail an ingest that already committed.
            if !ctx.extracted_observations.is_empty()
                && let Some(agent_id) = agent_id_for(&task)
            {
                let notice = ConsolidationTask::ObservationsReady(ObservationsReady {
                    agent_id,
                    observation_count: ctx.extracted_observations.len() as u32,
                    source_node_ids: ctx.extracted_observations.clone(),
                });
                if let Err(e) = consolidation_tx.try_send(notice) {
                    tracing::debug!(error = %e, "consolidation notify skipped");
                }
            }

            // Update health tracker.
            let mut tracker = health.lock().unwrap();
            if result.steps_failed.is_empty() {
                tracker.record_degradations(&diagnostics);
                tracker.record_success(elapsed_ms);
            } else {
                tracker.record_failure();
                for (step_name, _) in &result.steps_failed {
                    metrics::emit_ingest_failure(step_name);
                }
            }
            metrics::emit_ingest_duration(elapsed_ms);
        });
    }
}

/// The agent an ingest task should be consolidated under.
///
/// Carried as the reserved `agent_id` metadata key, set by
/// [`Session::submit`](crate::Session::submit). Tasks submitted directly to
/// [`PipelineSystem`](super::PipelineSystem) without it simply do not notify
/// the consolidation worker.
fn agent_id_for(task: &IngestTask) -> Option<String> {
    let metadata = match task {
        IngestTask::Message(m) => &m.metadata,
        _ => return None,
    };
    metadata
        .get("agent_id")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// Decrements the in-flight counter when a spawned ingest item finishes.
///
/// Held as the first binding in the spawned task so every exit path —
/// including the early return when the semaphore is closed during
/// shutdown — releases the count, keeping `quiesce` exact.
struct InflightGuard(Arc<AtomicUsize>);

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Serialize the task payload into the metadata keys [`IngestStep`] reads.
///
/// [`IngestStep`](uniko_extract::ingest::IngestStep) dispatches on
/// `ingest_type` and deserializes `ingest_payload`; without them the step
/// has nothing to ingest. Every variant (message / artifact / pdf / source)
/// serializes its payload here for the step to route on.
fn populate_ingest_metadata(task: &IngestTask, ctx: &mut PipelineContext) {
    let (ingest_type, payload) = match task {
        IngestTask::Message(m) => ("message", serde_json::to_value(m)),
        IngestTask::Artifact(a) => ("artifact", serde_json::to_value(a)),
        IngestTask::Pdf(p) => ("pdf", serde_json::to_value(p)),
        IngestTask::Source(s) => ("source", serde_json::to_value(s)),
    };
    ctx.metadata.insert(
        "ingest_type".to_string(),
        serde_json::Value::String(ingest_type.to_string()),
    );
    if let Ok(payload) = payload
        && !payload.is_null()
    {
        ctx.metadata.insert("ingest_payload".to_string(), payload);
    }
}

/// How a step chain ended.
#[derive(Debug)]
pub(crate) enum ChainEnd {
    /// Every step ran (some may have failed with `Skip` or `DeadLetter`).
    Completed,
    /// The cancellation token fired before every step ran.
    Cancelled,
    /// A step failed with `Abort`; its error.
    Aborted(String, UnikoError),
}

/// What [`run_step_chain`] produced for one item.
pub(crate) struct ChainOutcome {
    pub(crate) result: ItemResult,
    /// Every failure whose policy is `DeadLetter`, with its error intact so
    /// the dead letter can record whether a retry may succeed.
    pub(crate) dead: Vec<(String, UnikoError)>,
    pub(crate) end: ChainEnd,
}

/// Execute a chain of steps with per-item error isolation.
///
/// Each step's failure is handled according to its
/// [`StepErrorPolicy`].  The chain continues unless `Abort` is
/// declared or the cancellation token fires, and reports which of the
/// three ended it.
pub(crate) async fn run_step_chain(
    steps: &[Box<dyn Step>],
    ctx: &mut PipelineContext,
) -> ChainOutcome {
    let mut result = ItemResult::new(ctx.node_id);
    let mut dead = Vec::new();

    for step in steps {
        if ctx.cancel.is_cancelled() {
            return ChainOutcome {
                result,
                dead,
                end: ChainEnd::Cancelled,
            };
        }
        if !step.should_run(ctx) {
            result.steps_skipped.push(step.name().to_owned());
            continue;
        }

        let step_name = step.name().to_owned();
        tracing::debug!(step = %step_name, "executing step");
        let (error, policy) = match step.execute(ctx).await {
            Ok(StepOutcome::Completed) => {
                tracing::debug!("step completed");
                result.steps_succeeded.push(step_name);
                continue;
            }
            Ok(StepOutcome::Skipped { reason }) => {
                tracing::debug!(reason = %reason, "step skipped");
                result.steps_skipped.push(step_name);
                continue;
            }
            // A step's own failure report carries only text, so it is
            // treated as one a retry may fix.
            Ok(StepOutcome::Failed { error, policy }) => (UnikoError::Internal(error), policy),
            // Transport-level failure (Step::execute returned Err). No
            // StepOutcome to consult — default to DeadLetter so the item
            // is preserved for inspection / retry.
            Err(e) => (e, StepErrorPolicy::DeadLetter),
        };
        tracing::warn!(error = %error, policy = ?policy, "step failed");
        result
            .steps_failed
            .push((step_name.clone(), error.to_string()));
        match policy {
            StepErrorPolicy::Skip => {}
            StepErrorPolicy::DeadLetter => dead.push((step_name, error)),
            StepErrorPolicy::Abort => {
                return ChainOutcome {
                    result,
                    dead,
                    end: ChainEnd::Aborted(step_name, error),
                };
            }
        }
    }
    ChainOutcome {
        result,
        dead,
        end: ChainEnd::Completed,
    }
}

/// Record the outcome of an item against the dead-letter queue, and return
/// the item's result.
///
/// - A new item with a dead-letter failure becomes one letter (the first
///   such failure; later ones in the same chain are only logged).
/// - A retry updates the letter it retries:
///   - resolved only when its chain completed with no dead-letter failure;
///   - pending again or exhausted when it failed or aborted;
///   - released, uncounted, when shutdown cancelled it before it finished,
///     since the payload never fully ran.
///
/// Only a failure under the `DeadLetter` error policy counts. A step whose
/// policy is `Skip` is best effort: its failure creates no letter on a
/// first run, so a retry whose only failure is such a step is resolved.
///
/// A queue write failure is logged, never raised: the item itself has
/// already been processed.
pub(crate) async fn record_dead_letter(
    dlq: &DeadLetterQueue,
    health: &Mutex<HealthTracker>,
    task: &IngestTask,
    retry_of: Option<&str>,
    chain: ChainOutcome,
    node_ref: uniko_store::NodeId,
) -> ItemResult {
    let ChainOutcome { result, dead, end } = chain;
    let first_failure = dead.into_iter().next();
    match (retry_of, end, first_failure) {
        (Some(id), ChainEnd::Cancelled, _) => {
            if let Err(e) = dlq.release_claim(id).await {
                tracing::error!(error = %e, dead_letter = id, "failed to release retry claim");
            }
        }
        (Some(id), ChainEnd::Completed, None) => {
            if let Err(e) = dlq.retry_succeeded(id).await {
                tracing::error!(error = %e, dead_letter = id, "failed to resolve dead letter");
            }
            health.lock().unwrap().record_retry_success();
        }
        (Some(id), ChainEnd::Aborted(step, err), None) | (Some(id), _, Some((step, err))) => {
            if let Err(e) = dlq.retry_failed(id, &step, &err).await {
                tracing::error!(error = %e, dead_letter = id, "failed to update dead letter");
            }
        }
        (None, _, Some((step, err))) => match dlq.store(&step, &err, Some(task), node_ref).await {
            Ok(id) => {
                tracing::info!(dead_letter = %id, step = %step, "item dead-lettered");
                health.lock().unwrap().record_dead_letter();
            }
            Err(e) => tracing::error!(error = %e, "failed to store dead letter"),
        },
        (None, _, None) => {}
    }
    result
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use uniko_pipes::dead_letter::{DeadLetterStatus, RetryPolicy};
    use uniko_pipes::types::IngestMessage;
    use uniko_store::config::UnikoConfig;

    use super::*;

    fn task() -> IngestTask {
        IngestTask::Message(IngestMessage {
            message_id: "m".into(),
            content: "hello".into(),
            content_type: "text".into(),
            sender_id: "user".into(),
            session_id: "s".into(),
            addressed_to: None,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
            category: None,
            source_id: None,
            revision_id: None,
            visibility: None,
        })
    }

    fn outcome(end: ChainEnd) -> ChainOutcome {
        ChainOutcome {
            result: ItemResult::new(0),
            dead: Vec::new(),
            end,
        }
    }

    /// A claimed retry and the queue it lives in.
    async fn claimed() -> Option<(DeadLetterQueue, String)> {
        let kb = match KnowledgeBase::in_memory(UnikoConfig::default()).await {
            Ok(kb) => Arc::new(kb),
            Err(e) => {
                eprintln!("skipping: in-memory store unavailable: {e}");
                return None;
            }
        };
        let dlq = DeadLetterQueue::with_policy(kb, RetryPolicy::new(3, 10, 40));
        let id = dlq
            .store("ingest", &UnikoError::Timeout(1), Some(&task()), 0)
            .await
            .unwrap();
        dlq.begin_retry(&id, false).await.unwrap();
        Some((dlq, id))
    }

    /// Shutdown cutting a retry short must not mark it resolved: the
    /// payload never ran, so the claim is released and not counted.
    #[tokio::test]
    async fn a_cancelled_retry_is_released_not_resolved() {
        let Some((dlq, id)) = claimed().await else {
            return;
        };
        let health = Mutex::new(HealthTracker::new());
        record_dead_letter(
            &dlq,
            &health,
            &task(),
            Some(&id),
            outcome(ChainEnd::Cancelled),
            0,
        )
        .await;
        let letter = dlq.get(&id).await.unwrap().unwrap();
        assert_eq!(
            (letter.status, letter.retry_count),
            (DeadLetterStatus::Pending, 0)
        );
        assert_eq!(
            health.lock().unwrap().health(0, 1, false).retries_succeeded,
            0
        );
    }

    /// A retry that aborts is a failure, not a success.
    #[tokio::test]
    async fn an_aborted_retry_counts_as_failed() {
        let Some((dlq, id)) = claimed().await else {
            return;
        };
        let health = Mutex::new(HealthTracker::new());
        let end = ChainEnd::Aborted("ingest".into(), UnikoError::Timeout(2));
        record_dead_letter(&dlq, &health, &task(), Some(&id), outcome(end), 0).await;
        let letter = dlq.get(&id).await.unwrap().unwrap();
        assert_eq!(
            (letter.status, letter.retry_count),
            (DeadLetterStatus::Pending, 1)
        );
        assert_eq!(letter.error_kind.as_deref(), Some("timeout"));
    }

    /// Only a completed chain with no dead-letter failure resolves a retry.
    #[tokio::test]
    async fn a_completed_retry_is_resolved() {
        let Some((dlq, id)) = claimed().await else {
            return;
        };
        let health = Mutex::new(HealthTracker::new());
        record_dead_letter(
            &dlq,
            &health,
            &task(),
            Some(&id),
            outcome(ChainEnd::Completed),
            0,
        )
        .await;
        assert_eq!(
            dlq.get(&id).await.unwrap().unwrap().status,
            DeadLetterStatus::Resolved
        );
        assert_eq!(
            health.lock().unwrap().health(0, 1, false).retries_succeeded,
            1
        );
    }
}
