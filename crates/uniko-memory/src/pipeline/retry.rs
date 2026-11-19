//! Dead-letter retries: explicit ones and the opt-in background scheduler.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::sync::{OnceCell, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uniko_pipes::dead_letter::DeadLetterQueue;
use uniko_pipes::health::HealthTracker;
use uniko_store::UnikoError;

use super::RetryReport;
use super::ingest_worker::IngestJob;

/// Letters the scheduler resubmits per pass, so a large backlog drains over
/// several passes instead of flooding the ingest queue at once.
const SCHEDULER_BATCH: usize = 32;

/// Longest gap between scheduler passes.
const MAX_POLL: Duration = Duration::from_secs(60);

/// Everything a retry needs, shared by [`PipelineSystem`](super::PipelineSystem)
/// and the scheduler task.
pub(crate) struct Retrier {
    pub(crate) dlq: Arc<DeadLetterQueue>,
    ingest_tx: mpsc::Sender<IngestJob>,
    ingest_inflight: Arc<AtomicUsize>,
    ingest_health: Arc<Mutex<HealthTracker>>,
    /// Letters left `retrying` by an earlier process go back to `pending`
    /// once, before this pipeline claims any retry.
    interrupted_reset: OnceCell<()>,
}

impl Retrier {
    pub(crate) fn new(
        dlq: Arc<DeadLetterQueue>,
        ingest_tx: mpsc::Sender<IngestJob>,
        ingest_inflight: Arc<AtomicUsize>,
        ingest_health: Arc<Mutex<HealthTracker>>,
    ) -> Self {
        Self {
            dlq,
            ingest_tx,
            ingest_inflight,
            ingest_health,
            interrupted_reset: OnceCell::new(),
        }
    }

    /// Queue `job` and count it in flight, so `quiesce` waits for it.
    pub(crate) fn submit_job(&self, job: IngestJob) -> Result<(), UnikoError> {
        self.ingest_tx
            .try_send(job)
            .map_err(|e| UnikoError::Pipeline(format!("ingest channel: {e}")))?;
        // Count the item only once accepted; the worker decrements when it
        // finishes (see `IngestWorker`'s drop guard).
        self.ingest_inflight.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    pub(crate) async fn retry(&self, id: &str, force: bool) -> Result<(), UnikoError> {
        self.reset_interrupted_once().await;
        let task = self.dlq.begin_retry(id, force).await?;
        if let Err(e) = self.submit_job(IngestJob {
            task,
            retry_of: Some(id.to_string()),
        }) {
            if let Err(release) = self.dlq.release_claim(id).await {
                tracing::error!(error = %release, dead_letter = id, "failed to release retry claim");
            }
            return Err(e);
        }
        self.ingest_health.lock().unwrap().record_retry();
        Ok(())
    }

    /// Retry up to `limit` pending, retriable letters due by `now`. Stops
    /// early, leaving the rest pending, when the ingest queue is full.
    pub(crate) async fn retry_due(
        &self,
        now: DateTime<Utc>,
        limit: usize,
    ) -> Result<RetryReport, UnikoError> {
        self.reset_interrupted_once().await;
        let mut report = RetryReport::default();
        for id in self.dlq.due(now, limit).await? {
            match self.retry(&id, false).await {
                Ok(()) => report.retried.push(id),
                Err(e @ UnikoError::Pipeline(_)) => {
                    report.refused.push((id, e.to_string()));
                    break;
                }
                Err(e) => report.refused.push((id, e.to_string())),
            }
        }
        Ok(report)
    }

    async fn reset_interrupted_once(&self) {
        self.interrupted_reset
            .get_or_init(|| async {
                match self.dlq.reset_interrupted().await {
                    Ok(0) => {}
                    Ok(n) => tracing::info!(count = n, "dead letters interrupted mid-retry reset"),
                    Err(e) => {
                        tracing::warn!(error = %e, "could not reset interrupted dead letters");
                    }
                }
            })
            .await;
    }

    /// Spawn the background scheduler: every pass retries the letters whose
    /// backoff has elapsed, until `cancel` fires.
    ///
    /// Passes run at the base backoff, at most a minute apart. Only
    /// retriable letters under their retry limit are picked, so a letter
    /// stops at `max_retries` exactly as with explicit retries. A retry it
    /// starts counts as in flight; one not yet due does not, so `quiesce`
    /// never waits for a future retry.
    pub(crate) fn spawn_scheduler(
        self: Arc<Self>,
        cancel: CancellationToken,
        backoff_ms: u64,
    ) -> JoinHandle<()> {
        let poll = Duration::from_millis(backoff_ms.max(1)).min(MAX_POLL);
        tokio::spawn(async move {
            tracing::info!(
                poll_ms = poll.as_millis() as u64,
                "dead-letter retry scheduler started"
            );
            loop {
                tokio::select! {
                    biased;
                    () = cancel.cancelled() => break,
                    () = tokio::time::sleep(poll) => {}
                }
                match self.retry_due(Utc::now(), SCHEDULER_BATCH).await {
                    Ok(report) if !report.retried.is_empty() => tracing::info!(
                        retried = report.retried.len(),
                        refused = report.refused.len(),
                        "dead-letter retries resubmitted"
                    ),
                    Ok(_) => {}
                    Err(e) => tracing::warn!(error = %e, "dead-letter retry pass failed"),
                }
            }
            tracing::info!("dead-letter retry scheduler stopped");
        })
    }
}
