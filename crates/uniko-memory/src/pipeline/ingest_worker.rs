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
use uniko_store::KnowledgeBase;

/// Long-running ingest worker receiving tasks via a bounded channel.
pub(crate) struct IngestWorker {
    rx: mpsc::Receiver<IngestTask>,
    steps: Arc<Vec<Box<dyn Step>>>,
    semaphore: Arc<Semaphore>,
    cancel: CancellationToken,
    kb: Arc<KnowledgeBase>,
    llm_breaker: Arc<CircuitBreaker>,
    dlq: Arc<DeadLetterQueue>,
    health: Arc<Mutex<HealthTracker>>,
    dlq_max_retries: u32,
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
        rx: mpsc::Receiver<IngestTask>,
        steps: Vec<Box<dyn Step>>,
        concurrency: usize,
        cancel: CancellationToken,
        kb: Arc<KnowledgeBase>,
        llm_breaker: Arc<CircuitBreaker>,
        dlq: Arc<DeadLetterQueue>,
        health: Arc<Mutex<HealthTracker>>,
        dlq_max_retries: u32,
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
            dlq_max_retries,
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
                        Some(task) => self.process(task),
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
    fn process(&self, task: IngestTask) {
        let permit = self.semaphore.clone();
        let steps = self.steps.clone();
        let item_cancel = self.cancel.child_token();
        let kb = self.kb.clone();
        let breaker = self.llm_breaker.clone();
        let dlq = self.dlq.clone();
        let health = self.health.clone();
        let dlq_max = self.dlq_max_retries;
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
            let result = run_step_chain(&steps, &mut ctx, &dlq, dlq_max).await;
            let elapsed_ms = start.elapsed().as_millis() as f64;

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

/// Execute a chain of steps with per-item error isolation.
///
/// Each step's failure is handled according to its
/// [`StepErrorPolicy`].  The chain continues unless `Abort` is
/// declared or the cancellation token fires.
pub(crate) async fn run_step_chain(
    steps: &[Box<dyn Step>],
    ctx: &mut PipelineContext,
    dlq: &DeadLetterQueue,
    dlq_max_retries: u32,
) -> ItemResult {
    let mut result = ItemResult::new(ctx.node_id);

    for step in steps {
        if ctx.cancel.is_cancelled() {
            break;
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
            Ok(StepOutcome::Failed { error, policy }) => (error, policy),
            // Transport-level failure (Step::execute returned Err). No
            // StepOutcome to consult — default to DeadLetter so the item
            // is preserved for inspection / retry.
            Err(e) => (e.to_string(), StepErrorPolicy::DeadLetter),
        };
        if handle_failure(
            &step_name,
            &error,
            policy,
            ctx,
            dlq,
            dlq_max_retries,
            &mut result,
        )
        .await
        {
            break;
        }
    }
    result
}

/// Handle a step failure. Returns `true` if the chain should abort.
async fn handle_failure(
    step_name: &str,
    error: &str,
    policy: StepErrorPolicy,
    ctx: &PipelineContext,
    dlq: &DeadLetterQueue,
    dlq_max_retries: u32,
    result: &mut ItemResult,
) -> bool {
    tracing::warn!(error = %error, policy = ?policy, "step failed");
    result
        .steps_failed
        .push((step_name.to_owned(), error.to_owned()));

    match policy {
        StepErrorPolicy::Skip => false,
        StepErrorPolicy::DeadLetter => {
            if let Err(e) = dlq
                .store(step_name, error, ctx.node_id, dlq_max_retries)
                .await
            {
                tracing::error!(error = %e, "failed to store dead letter");
            }
            false
        }
        StepErrorPolicy::Abort => true,
    }
}
