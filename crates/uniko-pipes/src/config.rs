//! Pipeline configuration with spec-mandated defaults.

use serde::{Deserialize, Serialize};

/// Configuration for the pipeline execution engine.
///
/// All fields have defaults matching the uniko specification v6.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Bounded channel capacity for ingest tasks.
    pub ingest_queue_capacity: usize,
    /// Maximum concurrent ingest item processing.
    pub ingest_concurrency: usize,
    /// Bounded channel capacity for consolidation tasks.
    pub consolidation_queue_capacity: usize,
    /// Observation count that triggers consolidation for an agent.
    pub consolidation_threshold: u32,
    /// Seconds between periodic consolidation sweeps.
    pub consolidation_interval_secs: u64,
    /// Maximum unprocessed Observations to derive Facts from per cycle.
    ///
    /// Caps work per cycle so a long-running ingest doesn't starve
    /// other agents.  Spillover is picked up on the next sweep.
    pub consolidation_batch_size: u32,
    /// Run P5 (procedure promotion) + P6 (topic detection) every N
    /// successful consolidation cycles for an agent.
    ///
    /// `0` disables the cortex sweep entirely.
    pub cortex_cycle_every_n_consolidations: u32,
    /// Minimum wall-clock gap (seconds) between cortex sweeps.
    ///
    /// Acts as an upper bound on cortex frequency even when many
    /// consolidation cycles fire rapidly.  Applied independently per
    /// agent for procedures and globally for topics.
    pub cortex_min_interval_secs: u64,
    /// Execute active Locy rules during the cortex sweep
    /// (`episode_pattern_detector`, `contradiction_detector`, and any
    /// user-defined rules), recording matches into the confidence lifecycle.
    ///
    /// Enabled by default; set `false` to skip rule execution entirely.
    pub run_rules_in_sweep: bool,
    /// Consecutive LLM failures before the circuit breaker opens.
    pub circuit_failure_threshold: u32,
    /// Milliseconds the circuit breaker stays open before probing.
    pub circuit_recovery_ms: u64,
    /// Maximum automatic retries for dead-letter items.
    pub dead_letter_max_retries: u32,
    /// Total seconds allowed for graceful shutdown.
    pub shutdown_timeout_secs: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            ingest_queue_capacity: 200,
            ingest_concurrency: 8,
            consolidation_queue_capacity: 32,
            consolidation_threshold: 20,
            consolidation_interval_secs: 900,
            consolidation_batch_size: 500,
            cortex_cycle_every_n_consolidations: 4,
            cortex_min_interval_secs: 600,
            run_rules_in_sweep: true,
            circuit_failure_threshold: 5,
            circuit_recovery_ms: 60_000,
            dead_letter_max_retries: 3,
            shutdown_timeout_secs: 30,
        }
    }
}

impl PipelineConfig {
    /// The pipeline settings a [`UnikoConfig`](uniko_store::config::UnikoConfig)
    /// carries, with the rest at their defaults.
    ///
    /// `UnikoConfig` holds the queue capacities, consolidation schedule and
    /// circuit-breaker settings a host configures; the remaining knobs here
    /// (concurrency, batch size, cortex cadence, shutdown) have no
    /// `UnikoConfig` counterpart and keep their defaults.
    pub fn from_uniko_config(config: &uniko_store::config::UnikoConfig) -> Self {
        Self {
            ingest_queue_capacity: config.ingest_queue_capacity,
            consolidation_queue_capacity: config.consolidation_queue_capacity,
            consolidation_threshold: config.consolidation_threshold,
            consolidation_interval_secs: config.consolidation_interval_secs,
            circuit_failure_threshold: config.circuit_failure_threshold,
            circuit_recovery_ms: config.circuit_recovery_ms,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniko_store::config::UnikoConfig;

    #[test]
    fn default_uniko_config_maps_to_default_pipeline_config() {
        let mapped = PipelineConfig::from_uniko_config(&UnikoConfig::default());
        let default = PipelineConfig::default();
        assert_eq!(mapped.ingest_queue_capacity, default.ingest_queue_capacity);
        assert_eq!(
            mapped.consolidation_queue_capacity,
            default.consolidation_queue_capacity
        );
        assert_eq!(
            mapped.consolidation_threshold,
            default.consolidation_threshold
        );
        assert_eq!(
            mapped.consolidation_interval_secs,
            default.consolidation_interval_secs
        );
        assert_eq!(
            mapped.circuit_failure_threshold,
            default.circuit_failure_threshold
        );
        assert_eq!(mapped.circuit_recovery_ms, default.circuit_recovery_ms);
    }

    #[test]
    fn uniko_config_pipeline_fields_carry_over() {
        let config = UnikoConfig {
            ingest_queue_capacity: 7,
            consolidation_queue_capacity: 3,
            consolidation_threshold: 11,
            consolidation_interval_secs: 42,
            circuit_failure_threshold: 2,
            circuit_recovery_ms: 1_234,
            ..UnikoConfig::default()
        };
        let mapped = PipelineConfig::from_uniko_config(&config);
        assert_eq!(mapped.ingest_queue_capacity, 7);
        assert_eq!(mapped.consolidation_queue_capacity, 3);
        assert_eq!(mapped.consolidation_threshold, 11);
        assert_eq!(mapped.consolidation_interval_secs, 42);
        assert_eq!(mapped.circuit_failure_threshold, 2);
        assert_eq!(mapped.circuit_recovery_ms, 1_234);
        assert_eq!(
            mapped.ingest_concurrency,
            PipelineConfig::default().ingest_concurrency
        );
    }
}
