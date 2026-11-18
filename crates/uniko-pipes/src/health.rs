//! Worker health tracking and status classification.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::circuit_breaker::CircuitState;

/// Per-worker health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    /// Queue depth < 60 % capacity.
    Healthy,
    /// LLM circuit open, using fallbacks.
    Degraded,
    /// Queue depth > 80 % capacity.
    Backpressured,
    /// No items processed in > 5 minutes with pending work.
    Stalled,
}

/// Snapshot of a single worker's health.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerHealth {
    /// Current status classification.
    pub status: WorkerStatus,
    /// Current queue occupancy.
    pub queue_depth: usize,
    /// Maximum channel capacity.
    pub queue_capacity: usize,
    /// Lifetime items processed successfully.
    pub items_processed: u64,
    /// Lifetime items that failed.
    pub items_failed: u64,
    /// Exponential moving average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Lifetime count of degraded stages (fell back or failed) in items
    /// that still succeeded, keyed `stage:code` (e.g. `nlp:nlp_unavailable`).
    /// Streaming ingest returns no per-call result, so this is where its
    /// stage diagnostics surface.
    pub degradations: BTreeMap<String, u64>,
}

/// Aggregated health of the entire pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct PipelineHealth {
    /// Ingest worker health.
    pub ingest: WorkerHealth,
    /// Consolidation worker health.
    pub consolidation: WorkerHealth,
    /// LLM circuit breaker state.
    pub llm_circuit: CircuitState,
    /// Seconds since pipeline start.
    pub uptime_secs: u64,
}

/// Tracks per-worker counters and EMA latency.
///
/// Shared between the worker task and the health query via
/// `Arc<std::sync::Mutex<HealthTracker>>`.  Critical sections are
/// trivially short (primitive reads/writes only).
#[derive(Debug)]
pub struct HealthTracker {
    items_processed: u64,
    items_failed: u64,
    last_processed_at: Option<Instant>,
    avg_latency_ms: f64,
    degradations: BTreeMap<String, u64>,
}

/// EMA smoothing factor — higher values weight recent samples more.
const EMA_ALPHA: f64 = 0.1;

/// Seconds with no processing before a worker is considered stalled.
const STALL_THRESHOLD_SECS: u64 = 300;

impl HealthTracker {
    /// Create a fresh tracker with no history.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            items_processed: 0,
            items_failed: 0,
            last_processed_at: None,
            avg_latency_ms: 0.0,
            degradations: BTreeMap::new(),
        }
    }

    /// Record a successful item processing.
    pub fn record_success(&mut self, latency_ms: f64) {
        self.items_processed += 1;
        self.last_processed_at = Some(Instant::now());
        self.avg_latency_ms = EMA_ALPHA * latency_ms + (1.0 - EMA_ALPHA) * self.avg_latency_ms;
    }

    /// Count the degraded (fallback or failed) stages among `diagnostics`.
    pub fn record_degradations(&mut self, diagnostics: &[uniko_store::Diagnostic]) {
        for d in diagnostics.iter().filter(|d| d.status.is_degraded()) {
            let stage = serde_json::to_value(d.stage)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            let code = d.code.map_or("unknown", |c| c.as_str());
            *self
                .degradations
                .entry(format!("{stage}:{code}"))
                .or_insert(0) += 1;
        }
    }

    /// Record a failed item.
    pub fn record_failure(&mut self) {
        self.items_failed += 1;
        self.last_processed_at = Some(Instant::now());
    }

    /// Compute the health snapshot for this worker.
    pub fn health(
        &self,
        queue_depth: usize,
        queue_capacity: usize,
        circuit_open: bool,
    ) -> WorkerHealth {
        let status = classify(
            queue_depth,
            queue_capacity,
            circuit_open,
            self.last_processed_at,
        );
        WorkerHealth {
            status,
            queue_depth,
            queue_capacity,
            items_processed: self.items_processed,
            items_failed: self.items_failed,
            avg_latency_ms: self.avg_latency_ms,
            degradations: self.degradations.clone(),
        }
    }
}

/// Classify worker status from queue depth and external signals.
fn classify(
    queue_depth: usize,
    queue_capacity: usize,
    circuit_open: bool,
    last_processed_at: Option<Instant>,
) -> WorkerStatus {
    if circuit_open {
        return WorkerStatus::Degraded;
    }
    if let Some(last) = last_processed_at
        && last.elapsed().as_secs() > STALL_THRESHOLD_SECS
        && queue_depth > 0
    {
        return WorkerStatus::Stalled;
    }
    let ratio = if queue_capacity > 0 {
        queue_depth as f64 / queue_capacity as f64
    } else {
        0.0
    };
    if ratio > 0.8 {
        WorkerStatus::Backpressured
    } else {
        WorkerStatus::Healthy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healthy_low_depth() {
        assert_eq!(classify(10, 200, false, None), WorkerStatus::Healthy);
    }

    #[test]
    fn test_backpressured() {
        assert_eq!(classify(170, 200, false, None), WorkerStatus::Backpressured);
    }

    #[test]
    fn test_degraded_circuit_open() {
        assert_eq!(classify(10, 200, true, None), WorkerStatus::Degraded);
    }

    #[test]
    fn test_ema_latency() {
        let mut t = HealthTracker::new();
        t.record_success(100.0);
        assert!((t.avg_latency_ms - 10.0).abs() < 0.01); // 0.1 * 100
        t.record_success(100.0);
        // 0.1 * 100 + 0.9 * 10 = 19.0
        assert!((t.avg_latency_ms - 19.0).abs() < 0.01);
    }
}
