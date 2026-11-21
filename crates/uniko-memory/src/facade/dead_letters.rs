//! [`DeadLetters`]: inspect, retry and resolve streaming-ingest failures.

use std::sync::Arc;

use uniko_pipes::dead_letter::{DeadLetter, DeadLetterCounts, DeadLetterFilter, DeadLetterQueue};
use uniko_store::{KnowledgeBase, UnikoError};

use crate::pipeline::{PipelineSystem, RetryReport};

/// The dead-letter queue of an instance's streaming pipeline.
///
/// A streaming item ([`Session::submit`](crate::Session::submit),
/// [`Session::submit_source`](crate::Session::submit_source)) whose ingest
/// fails is kept as a dead letter with its payload. Obtained from
/// [`Uniko::dead_letters`](crate::Uniko::dead_letters).
///
/// Listing, counting and resolving work on any instance, so letters left by
/// an earlier streaming run can be inspected. Retrying needs the instance to
/// be built with [`streaming`](crate::UnikoBuilder::streaming).
#[derive(Clone)]
pub struct DeadLetters {
    queue: DeadLetterQueue,
    pipeline: Option<Arc<PipelineSystem>>,
}

impl DeadLetters {
    pub(super) fn new(kb: &KnowledgeBase, pipeline: Option<Arc<PipelineSystem>>) -> Self {
        let queue = match &pipeline {
            Some(p) => p.dead_letter_queue().clone(),
            None => DeadLetterQueue::new(Arc::new(kb.clone())),
        };
        Self { queue, pipeline }
    }

    /// Dead letters matching `filter`, oldest first.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn list(&self, filter: &DeadLetterFilter) -> Result<Vec<DeadLetter>, UnikoError> {
        self.queue.list(filter).await
    }

    /// One dead letter by id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn get(&self, id: &str) -> Result<Option<DeadLetter>, UnikoError> {
        self.queue.get(id).await
    }

    /// How many dead letters are in each status.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn counts(&self) -> Result<DeadLetterCounts, UnikoError> {
        self.queue.counts().await
    }

    /// Retry dead letter `id`: count the attempt and resubmit its stored
    /// payload unchanged, so it keeps its agent and source attribution.
    ///
    /// Returns once the item is queued; its outcome lands on the letter when
    /// it is processed, and [`Session::flush`](crate::Session::flush) waits
    /// for it. Ingest dedups by id and content, so a retry of an item that
    /// partly landed writes nothing twice.
    ///
    /// `force` also retries a letter whose failure a retry can't normally
    /// fix (an id conflict, an unsupported payload) or one that has used all
    /// its retries.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] without streaming, or when the letter
    /// is unknown, resolved, being retried, not replayable, out of retries or
    /// permanently failed (without `force`); [`UnikoError::Pipeline`] when
    /// the ingest queue is full, in which case the attempt isn't counted.
    pub async fn retry(&self, id: &str, force: bool) -> Result<(), UnikoError> {
        self.require_pipeline("retry")?
            .retry_dead_letter(id, force)
            .await
    }

    /// Retry up to `limit` pending dead letters whose failure a retry can
    /// fix, without waiting for their backoff. Letters go in the order their
    /// next retry falls due, soonest first.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] without streaming, or
    /// [`UnikoError::Storage`] when the letters can't be read.
    pub async fn retry_pending(&self, limit: usize) -> Result<RetryReport, UnikoError> {
        self.require_pipeline("retry_pending")?
            .retry_pending(limit)
            .await
    }

    /// Resolve `id` without retrying it, recording `note` as the reason.
    /// Only the letter changes; memory it may have partly written is left
    /// alone. Returns `false` when there is no such letter or it was already
    /// resolved.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] while a retry of it is in flight, or
    /// [`UnikoError::Storage`] on graph failure.
    pub async fn resolve(&self, id: &str, note: Option<&str>) -> Result<bool, UnikoError> {
        self.queue.resolve(id, note).await
    }

    fn require_pipeline(&self, op: &str) -> Result<&PipelineSystem, UnikoError> {
        self.pipeline.as_deref().ok_or_else(|| {
            UnikoError::Config(format!(
                "dead_letters().{op} needs an instance built with .streaming(true)"
            ))
        })
    }
}
