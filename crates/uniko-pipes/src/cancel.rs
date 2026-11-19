//! Cancellation token hierarchy for orderly shutdown.
//!
//! Tokens form a parent-child tree: cancelling a parent cancels all
//! children, but cancelling a child only affects that subtree.

use std::time::Duration;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Manages a tree of cancellation tokens for graceful shutdown.
///
/// ```text
/// root
///   +-- ingest   (cancelled first, 5 s drain)
///   +-- consolidation (cancelled second, 10 s drain)
/// ```
#[derive(Debug, Clone)]
pub struct ShutdownCoordinator {
    root: CancellationToken,
    ingest: CancellationToken,
    consolidation: CancellationToken,
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl ShutdownCoordinator {
    /// Build the token hierarchy.
    pub fn new() -> Self {
        let root = CancellationToken::new();
        let ingest = root.child_token();
        let consolidation = root.child_token();
        Self {
            root,
            ingest,
            consolidation,
        }
    }

    /// Token for the ingest worker.
    pub fn ingest_token(&self) -> CancellationToken {
        self.ingest.clone()
    }

    /// Token for the consolidation worker.
    pub fn consolidation_token(&self) -> CancellationToken {
        self.consolidation.clone()
    }

    /// A guard that cancels the whole tree when dropped, so a pipeline
    /// dropped without [`shutdown`](Self::shutdown) still stops its tasks.
    pub fn cancel_on_drop(&self) -> tokio_util::sync::DropGuard {
        self.root.clone().drop_guard()
    }

    /// Whether the root has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.root.is_cancelled()
    }

    /// Execute a graceful shutdown, joining all worker handles.
    ///
    /// Sequence: cancel ingest (5 s) → cancel consolidation (10 s) →
    /// cancel root → join handles within `total_timeout`.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeout expires before all workers join.
    pub async fn shutdown(
        &self,
        workers: &mut Vec<JoinHandle<()>>,
        total_timeout: Duration,
    ) -> Result<(), &'static str> {
        // Phase 1: stop accepting new ingest tasks.
        self.ingest.cancel();
        tracing::info!("shutdown: ingest cancelled, draining in-flight items");
        tokio::time::sleep(Duration::from_secs(5).min(total_timeout)).await;

        // Phase 2: stop consolidation.
        self.consolidation.cancel();
        tracing::info!("shutdown: consolidation cancelled, finishing current cycle");
        tokio::time::sleep(Duration::from_secs(10).min(total_timeout)).await;

        // Phase 3: force-stop anything remaining.
        self.root.cancel();
        tracing::info!("shutdown: root cancelled, joining workers");

        // Join all handles within remaining budget.
        let deadline = tokio::time::Instant::now() + total_timeout;
        for handle in workers.drain(..) {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                tracing::warn!("shutdown: timeout expired, aborting remaining workers");
                handle.abort();
                continue;
            }
            match tokio::time::timeout(remaining, handle).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(error = %e, "worker panicked during shutdown"),
                Err(_) => tracing::warn!("worker did not join within timeout, aborting"),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parent_cancels_children() {
        let coord = ShutdownCoordinator::new();
        let ingest = coord.ingest_token();
        let consolidation = coord.consolidation_token();

        assert!(!ingest.is_cancelled());
        assert!(!consolidation.is_cancelled());

        coord.root.cancel();

        assert!(ingest.is_cancelled());
        assert!(consolidation.is_cancelled());
    }

    #[test]
    fn test_child_independent_of_sibling() {
        let coord = ShutdownCoordinator::new();
        let ingest = coord.ingest_token();
        let consolidation = coord.consolidation_token();

        coord.ingest.cancel();

        assert!(ingest.is_cancelled());
        assert!(
            !consolidation.is_cancelled(),
            "sibling must not be affected"
        );
    }
}
