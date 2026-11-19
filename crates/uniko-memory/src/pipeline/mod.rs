//! Pipeline orchestration: system lifecycle, task routing, health.
//!
//! [`PipelineSystem`] owns all workers, channels, and the circuit breaker.
//! It is the single entry point for submitting ingest and consolidation
//! tasks.

pub(crate) mod consolidation_worker;
pub(crate) mod ingest_worker;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use uniko_pipes::cancel::ShutdownCoordinator;
use uniko_pipes::circuit_breaker::{CircuitBreaker, CircuitState};
use uniko_pipes::config::PipelineConfig;
use uniko_pipes::dead_letter::{
    DeadLetter, DeadLetterCounts, DeadLetterFilter, DeadLetterQueue, RetryPolicy,
};
use uniko_pipes::health::{HealthTracker, PipelineHealth};
use uniko_pipes::metrics::register_pipeline_metrics;
use uniko_pipes::step::Step;
use uniko_pipes::types::{ConsolidationTask, IngestTask};
use uniko_store::KnowledgeBase;

use self::consolidation_worker::ConsolidationWorker;
use self::ingest_worker::{IngestJob, IngestWorker};

mod retry;

/// Root orchestrator for all pipeline workers.
///
/// Owns bounded channels, the LLM circuit breaker, and worker handles.
/// Submitting tasks is non-blocking (bounded channel with backpressure).
///
/// # Examples
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use std::sync::Arc;
/// use uniko_pipes::PipelineConfig;
/// use uniko_store::{KnowledgeBase, config::UnikoConfig};
/// use uniko_memory::pipeline::PipelineSystem;
///
/// let kb = Arc::new(KnowledgeBase::in_memory(UnikoConfig::default()).await?);
/// let ps = PipelineSystem::new(PipelineConfig::default(), kb, vec![]);
/// // ... submit tasks ...
/// ps.shutdown().await?;
/// # Ok(())
/// # }
/// ```
pub struct PipelineSystem {
    shutdown: ShutdownCoordinator,
    ingest_tx: mpsc::Sender<IngestJob>,
    consolidation_tx: mpsc::Sender<ConsolidationTask>,
    llm_breaker: Arc<CircuitBreaker>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    ingest_health: Arc<Mutex<HealthTracker>>,
    consolidation_health: Arc<Mutex<HealthTracker>>,
    /// Submitted-but-unfinished ingest items, awaited by [`Self::quiesce`].
    ingest_inflight: Arc<AtomicUsize>,
    started_at: Instant,
    config: PipelineConfig,
    retrier: Arc<retry::Retrier>,
    /// Cancels every worker when the system is dropped, even without
    /// [`shutdown`](Self::shutdown): otherwise the retry scheduler, which
    /// holds its own sender and store handle, would keep running.
    _cancel_on_drop: tokio_util::sync::DropGuard,
}

/// What [`PipelineSystem::retry_pending`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RetryReport {
    /// Letters resubmitted for a retry.
    pub retried: Vec<String>,
    /// Letters that could not be resubmitted, with the reason.
    pub refused: Vec<RetryRefusal>,
}

/// A dead letter [`PipelineSystem::retry_pending`] could not resubmit.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RetryRefusal {
    /// The letter's id.
    pub id: String,
    /// Why it was not resubmitted.
    pub reason: String,
}

impl PipelineSystem {
    /// Create the system, spawn workers, and begin processing.
    ///
    /// Workers start their `select!` loops immediately.
    pub fn new(
        config: PipelineConfig,
        kb: Arc<KnowledgeBase>,
        ingest_steps: Vec<Box<dyn Step>>,
    ) -> Self {
        register_pipeline_metrics();

        let shutdown = ShutdownCoordinator::new();
        let llm_breaker = Arc::new(CircuitBreaker::new(
            config.circuit_failure_threshold,
            config.circuit_recovery_ms,
        ));
        let dlq = Arc::new(DeadLetterQueue::with_policy(
            kb.clone(),
            RetryPolicy::new(
                config.dead_letter_max_retries,
                config.dead_letter_retry_backoff_ms,
                config.dead_letter_retry_max_backoff_ms,
            ),
        ));

        // Bounded channels with backpressure.
        let (ingest_tx, ingest_rx) = mpsc::channel(config.ingest_queue_capacity);
        let (consolidation_tx, consolidation_rx) =
            mpsc::channel(config.consolidation_queue_capacity);

        let ingest_health = Arc::new(Mutex::new(HealthTracker::new()));
        let consolidation_health = Arc::new(Mutex::new(HealthTracker::new()));
        let ingest_inflight = Arc::new(AtomicUsize::new(0));

        // Spawn ingest worker.
        let ingest_worker = IngestWorker::new(
            ingest_rx,
            ingest_steps,
            config.ingest_concurrency,
            shutdown.ingest_token(),
            kb.clone(),
            llm_breaker.clone(),
            dlq.clone(),
            ingest_health.clone(),
            consolidation_tx.clone(),
            ingest_inflight.clone(),
        );
        let ingest_handle = tokio::spawn(ingest_worker.run());

        // Spawn consolidation worker.
        let consolidation_worker = ConsolidationWorker::new(
            consolidation_rx,
            shutdown.consolidation_token(),
            kb,
            consolidation_health.clone(),
            &config,
        );
        let consolidation_handle = tokio::spawn(consolidation_worker.run());

        let retrier = Arc::new(retry::Retrier::new(
            dlq,
            ingest_tx.clone(),
            ingest_inflight.clone(),
            ingest_health.clone(),
        ));
        // Return retries a stopped process left mid-flight to `pending`
        // now, so they can be resolved or retried without waiting for this
        // pipeline's first retry.
        let reset = retrier.clone();
        tokio::spawn(async move { reset.reset_interrupted_once().await });
        let mut workers = vec![ingest_handle, consolidation_handle];
        if config.dead_letter_auto_retry {
            workers.push(
                retrier
                    .clone()
                    .spawn_scheduler(shutdown.ingest_token(), config.dead_letter_retry_backoff_ms),
            );
        }

        let cancel_on_drop = shutdown.cancel_on_drop();
        Self {
            shutdown,
            ingest_tx,
            consolidation_tx,
            llm_breaker,
            workers: Mutex::new(workers),
            ingest_health,
            consolidation_health,
            ingest_inflight,
            started_at: Instant::now(),
            config,
            retrier,
            _cancel_on_drop: cancel_on_drop,
        }
    }

    /// Submit an ingest task.
    ///
    /// Returns `Err` if the channel is full (backpressure) or the system
    /// is shutting down.
    ///
    /// # Errors
    ///
    /// Returns [`uniko_store::UnikoError::Pipeline`] on channel failure.
    pub fn submit_ingest(&self, task: IngestTask) -> Result<(), uniko_store::UnikoError> {
        self.submit_job(IngestJob {
            task,
            retry_of: None,
        })
    }

    fn submit_job(&self, job: IngestJob) -> Result<(), uniko_store::UnikoError> {
        self.retrier.submit_job(job)
    }

    /// Await until every submitted ingest item has finished processing.
    ///
    /// A true barrier: returns only when no submitted-but-unfinished ingest
    /// item remains, so a caller can `submit_ingest(...)` then `quiesce()`
    /// before a recall that must observe those writes. Polls on a short
    /// interval; intended for steady-state draining, not shutdown.
    pub async fn quiesce(&self) {
        while self.ingest_inflight.load(Ordering::SeqCst) != 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Submit a consolidation task.
    ///
    /// # Errors
    ///
    /// Returns [`uniko_store::UnikoError::Pipeline`] on channel failure.
    pub fn submit_consolidation(
        &self,
        task: ConsolidationTask,
    ) -> Result<(), uniko_store::UnikoError> {
        self.consolidation_tx
            .try_send(task)
            .map_err(|e| uniko_store::UnikoError::Pipeline(format!("consolidation channel: {e}")))
    }

    /// The pipeline's dead-letter queue, for listing, counting and
    /// resolving letters.
    #[must_use]
    pub fn dead_letter_queue(&self) -> &DeadLetterQueue {
        &self.retrier.dlq
    }

    /// Dead letters matching `filter`, oldest first.
    ///
    /// # Errors
    ///
    /// Returns [`uniko_store::UnikoError::Storage`] on graph failure.
    pub async fn dead_letters(
        &self,
        filter: &DeadLetterFilter,
    ) -> Result<Vec<DeadLetter>, uniko_store::UnikoError> {
        self.retrier.dlq.list(filter).await
    }

    /// How many dead letters are in each status.
    ///
    /// # Errors
    ///
    /// Returns [`uniko_store::UnikoError::Storage`] on graph failure.
    pub async fn dead_letter_counts(&self) -> Result<DeadLetterCounts, uniko_store::UnikoError> {
        self.retrier.dlq.counts().await
    }

    /// Retry dead letter `id`: count the attempt and resubmit its stored
    /// payload unchanged. The outcome lands on the letter once the item is
    /// processed (resolved, pending again, or exhausted); a following
    /// [`quiesce`](Self::quiesce) waits for it.
    ///
    /// `force` retries a letter whose failure a retry can't normally fix,
    /// or one that has used all its retries.
    ///
    /// # Errors
    ///
    /// Returns [`uniko_store::UnikoError::Config`] when the letter can't be
    /// retried (see [`DeadLetterQueue::begin_retry`]), or
    /// [`uniko_store::UnikoError::Pipeline`] when the ingest queue is full;
    /// the attempt is then not counted.
    pub async fn retry_dead_letter(
        &self,
        id: &str,
        force: bool,
    ) -> Result<(), uniko_store::UnikoError> {
        self.retrier.retry(id, force).await
    }

    /// Retry up to `limit` pending dead letters whose failure a retry can
    /// fix, oldest first, regardless of their backoff.
    ///
    /// Stops early, leaving the rest pending, if the ingest queue fills.
    ///
    /// # Errors
    ///
    /// Returns [`uniko_store::UnikoError::Storage`] when the letters can't
    /// be read.
    pub async fn retry_pending(
        &self,
        limit: usize,
    ) -> Result<RetryReport, uniko_store::UnikoError> {
        self.retrier
            .retry_due(chrono::DateTime::<chrono::Utc>::MAX_UTC, limit)
            .await
    }

    /// Query aggregate pipeline health.
    pub fn health(&self) -> PipelineHealth {
        let circuit_open = self.llm_breaker.state() == CircuitState::Open;

        let ingest_depth = self.ingest_tx.max_capacity() - self.ingest_tx.capacity();
        let ingest = self.ingest_health.lock().unwrap().health(
            ingest_depth,
            self.ingest_tx.max_capacity(),
            circuit_open,
        );

        let consol_depth = self.consolidation_tx.max_capacity() - self.consolidation_tx.capacity();
        let consolidation = self.consolidation_health.lock().unwrap().health(
            consol_depth,
            self.consolidation_tx.max_capacity(),
            circuit_open,
        );

        PipelineHealth {
            ingest,
            consolidation,
            llm_circuit: self.llm_breaker.state(),
            uptime_secs: self.started_at.elapsed().as_secs(),
        }
    }

    /// Graceful shutdown: drain workers within the configured timeout.
    ///
    /// # Errors
    ///
    /// Returns an error string if shutdown times out.
    pub async fn shutdown(self) -> Result<(), String> {
        let timeout = Duration::from_secs(self.config.shutdown_timeout_secs);
        let mut workers = self.workers.into_inner().unwrap();
        self.shutdown
            .shutdown(&mut workers, timeout)
            .await
            .map_err(|e| e.to_string())
    }
}

impl std::fmt::Debug for PipelineSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineSystem")
            .field("uptime_secs", &self.started_at.elapsed().as_secs())
            .finish_non_exhaustive()
    }
}
