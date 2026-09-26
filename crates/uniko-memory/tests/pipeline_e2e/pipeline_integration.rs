//! Integration tests for the PipelineSystem lifecycle.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;

use uniko_pipes::config::PipelineConfig;
use uniko_pipes::step::{PipelineContext, Step};
use uniko_pipes::types::{ConsolidationTask, IngestMessage, IngestTask, StepOutcome};
use uniko_store::config::UnikoConfig;
use uniko_store::{KnowledgeBase, UnikoError};

use uniko_memory::PipelineSystem;

// ── Mock step for testing ───────────────────────────────────────────

struct MockStep {
    name: String,
    outcome: StepOutcome,
}

impl MockStep {
    fn succeeding(name: &str) -> Box<dyn Step> {
        Box::new(Self {
            name: name.to_string(),
            outcome: StepOutcome::Completed,
        })
    }
}

#[async_trait]
impl Step for MockStep {
    fn name(&self) -> &str {
        &self.name
    }

    fn should_run(&self, _ctx: &PipelineContext) -> bool {
        true
    }

    async fn execute(&self, _ctx: &mut PipelineContext) -> Result<StepOutcome, UnikoError> {
        Ok(self.outcome.clone())
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

async fn test_kb() -> Arc<KnowledgeBase> {
    Arc::new(
        KnowledgeBase::in_memory(UnikoConfig::default())
            .await
            .expect("in-memory KB"),
    )
}

fn test_message(id: &str) -> IngestTask {
    IngestTask::Message(IngestMessage {
        message_id: id.to_string(),
        content: format!("test message {id}"),
        content_type: "text".to_string(),
        sender_id: "test-sender".to_string(),
        session_id: "test-session".to_string(),
        addressed_to: None,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
        category: None,
        source_id: None,
        revision_id: None,
    })
}

// ── Tests ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_pipeline_system_lifecycle() {
    let kb = test_kb().await;
    let steps: Vec<Box<dyn Step>> = vec![MockStep::succeeding("mock_step")];
    let ps = PipelineSystem::new(PipelineConfig::default(), kb, steps);

    // Submit a task.
    ps.submit_ingest(test_message("m-1")).unwrap();

    // Brief pause for processing.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Health should show the system is alive.
    let health = ps.health();
    assert!(health.uptime_secs < 5);

    // Shutdown should succeed quickly.
    ps.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_pipeline_backpressure() {
    let kb = test_kb().await;
    let config = PipelineConfig {
        ingest_queue_capacity: 5,
        ..PipelineConfig::default()
    };
    let ps = PipelineSystem::new(config, kb, vec![]);

    // Fill the channel to capacity.
    for i in 0..5 {
        ps.submit_ingest(test_message(&format!("bp-{i}"))).unwrap();
    }

    // Next submission should fail (channel full).
    let result = ps.submit_ingest(test_message("bp-overflow"));
    assert!(result.is_err(), "should get backpressure error");

    ps.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_pipeline_consolidation_submit() {
    let kb = test_kb().await;
    let ps = PipelineSystem::new(PipelineConfig::default(), kb, vec![]);

    // Submit a force-consolidate task.
    ps.submit_consolidation(ConsolidationTask::ForceConsolidate {
        agent_id: "agent-1".to_string(),
    })
    .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    ps.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_pipeline_concurrent_ingest() {
    let kb = test_kb().await;
    let steps: Vec<Box<dyn Step>> = vec![MockStep::succeeding("noop")];
    let ps = Arc::new(PipelineSystem::new(PipelineConfig::default(), kb, steps));

    // Submit 50 tasks concurrently.
    let mut handles = Vec::new();
    for i in 0..50 {
        let ps_clone = ps.clone();
        handles.push(tokio::spawn(async move {
            let _ = ps_clone.submit_ingest(test_message(&format!("conc-{i}")));
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Should not panic.  Check health is accessible.
    let health = ps.health();
    assert!(health.uptime_secs < 10);

    // Shutdown (need to unwrap Arc first).
    let ps_owned = Arc::try_unwrap(ps).expect("should be sole owner");
    ps_owned.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_pipeline_health_reports() {
    let kb = test_kb().await;
    let ps = PipelineSystem::new(PipelineConfig::default(), kb, vec![]);

    let health = ps.health();
    assert_eq!(health.llm_circuit, uniko_pipes::CircuitState::Closed);
    assert_eq!(health.ingest.queue_depth, 0);
    assert_eq!(health.consolidation.queue_depth, 0);

    ps.shutdown().await.unwrap();
}
