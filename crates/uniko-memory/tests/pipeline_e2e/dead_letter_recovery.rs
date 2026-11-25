//! Dead-letter recovery (#65): old stores, queue transitions, and a message
//! retried through a real pipeline.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use chrono::Utc;

use uniko_extract::ingest::{IngestStep, ModalityRegistry};
use uniko_pipes::config::PipelineConfig;
use uniko_pipes::dead_letter::{DeadLetterFilter, DeadLetterQueue, DeadLetterStatus, RetryPolicy};
use uniko_pipes::step::{PipelineContext, Step};
use uniko_pipes::types::{IngestMessage, IngestTask, StepOutcome};
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::labels;
use uniko_store::{KnowledgeBase, UnikoError, Value};

use uniko_memory::PipelineSystem;

/// The DeadLetter properties and indexes #65 added.
const NEW_PROPERTIES: &[&str] = &[
    "dead_letter_id",
    "ingest_type",
    "payload",
    "status",
    "error_kind",
    "retriable",
    "agent_id",
    "updated_at",
    "resolved_at",
    "resolution",
];

/// The schema as it was before #65: the current snapshot with the new
/// DeadLetter properties and indexes taken out.
fn schema_before_retry_columns(dir: &std::path::Path) -> std::path::PathBuf {
    let current = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/schema.json");
    let mut schema: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(current).expect("schema.json")).unwrap();
    let props = schema["properties"]["DeadLetter"]
        .as_object_mut()
        .expect("DeadLetter properties");
    for p in NEW_PROPERTIES {
        assert!(props.remove(*p).is_some(), "snapshot lacks DeadLetter.{p}");
    }
    let indexes = schema["indexes"].as_array_mut().expect("indexes");
    let before = indexes.len();
    indexes.retain(|ix| {
        !(ix["label"] == "DeadLetter"
            && ix["properties"]
                .as_array()
                .is_some_and(|p| p.iter().any(|f| f == "dead_letter_id" || f == "status")))
    });
    assert_eq!(before - indexes.len(), 2, "two DeadLetter indexes removed");
    let path = dir.join("schema_before.json");
    std::fs::write(&path, serde_json::to_string(&schema).unwrap()).unwrap();
    path
}

/// A store created before the retry columns existed, holding a dead letter
/// written the old way, reopens under the current schema: the row lists as
/// legacy, and new letters get the full lifecycle.
#[tokio::test(flavor = "multi_thread")]
async fn an_old_store_reopens_with_the_retry_schema() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("kb");
    let old = UnikoConfig {
        schema_path: Some(schema_before_retry_columns(dir.path())),
        ..UnikoConfig::default()
    };
    let kb = KnowledgeBase::open(&store, old).await.expect("open old");
    let props = HashMap::from([
        ("step".to_string(), Value::String("ingest".into())),
        ("error".to_string(), Value::String("old failure".into())),
        ("node_ref".to_string(), Value::Int(0)),
        ("retry_count".to_string(), Value::Int(0)),
        ("max_retries".to_string(), Value::Int(3)),
    ]);
    kb.create_node(labels::DEAD_LETTER, &props).await.unwrap();
    kb.shutdown().await.expect("close old");

    let kb = Arc::new(
        KnowledgeBase::open(&store, UnikoConfig::default())
            .await
            .expect("reopen under the current schema"),
    );
    let dlq = DeadLetterQueue::new(kb.clone());
    let letters = dlq.list(&DeadLetterFilter::new()).await.unwrap();
    let [old_letter] = letters.as_slice() else {
        panic!("{letters:?}");
    };
    assert_eq!(old_letter.status, DeadLetterStatus::Legacy);
    assert_eq!(old_letter.error, "old failure");
    assert!(!old_letter.replayable);

    let id = dlq
        .store(
            "ingest",
            &UnikoError::Embedding("offline".into()),
            Some(&message("m-new", "user", None)),
            0,
        )
        .await
        .unwrap();
    let new_letter = dlq.get(&id).await.unwrap().unwrap();
    assert_eq!(new_letter.status, DeadLetterStatus::Pending);
    assert!(new_letter.replayable);
    let counts = dlq.counts().await.unwrap();
    assert_eq!((counts.legacy, counts.pending), (1, 1));
    drop(dlq);
    Arc::try_unwrap(kb)
        .ok()
        .expect("sole kb")
        .shutdown()
        .await
        .unwrap();
}

fn message(id: &str, sender: &str, agent: Option<&str>) -> IngestTask {
    let mut metadata = HashMap::new();
    if let Some(a) = agent {
        metadata.insert("agent_id".to_string(), serde_json::json!(a));
    }
    IngestTask::Message(IngestMessage {
        message_id: id.to_string(),
        content: format!("the quarterly report is due on friday ({id})"),
        content_type: "text".to_string(),
        sender_id: sender.to_string(),
        session_id: "research".to_string(),
        addressed_to: None,
        timestamp: Utc::now(),
        metadata,
        category: None,
        source_id: Some("crm".to_string()),
        revision_id: None,
        visibility: None,
    })
}

async fn kb() -> Arc<KnowledgeBase> {
    Arc::new(
        KnowledgeBase::in_memory(UnikoConfig::default())
            .await
            .expect("in-memory KB"),
    )
}

/// Claims, releases and resolution at the queue level.
#[tokio::test]
async fn queue_transitions() {
    let kb = kb().await;
    let dlq = DeadLetterQueue::with_policy(kb.clone(), RetryPolicy::new(2, 10, 40));
    let task = message("m-1", "user", Some("reporter"));
    let id = dlq
        .store("ingest", &UnikoError::Timeout(5), Some(&task), 0)
        .await
        .unwrap();
    let letter = dlq.get(&id).await.unwrap().unwrap();
    assert_eq!(letter.agent_id.as_deref(), Some("reporter"));
    assert_eq!(letter.ingest_type.as_deref(), Some("message"));

    // A claim in flight here can't be resolved; releasing it doesn't count.
    let replay = dlq.begin_retry(&id, false).await.unwrap();
    assert!(
        matches!(&replay, IngestTask::Message(m) if m.sender_id == "user"
        && m.source_id.as_deref() == Some("crm"))
    );
    let err = dlq.resolve(&id, None).await.unwrap_err();
    assert!(matches!(err, UnikoError::Config(_)), "{err}");
    dlq.release_claim(&id).await.unwrap();
    let letter = dlq.get(&id).await.unwrap().unwrap();
    assert_eq!(
        (letter.status, letter.retry_count),
        (DeadLetterStatus::Pending, 0)
    );

    // A claim left by another process (here: another queue value) is stale:
    // it can be resolved, and a reset puts it back pending, due now, with
    // the attempt still counted, since the payload may be what stopped it.
    DeadLetterQueue::new(kb.clone())
        .begin_retry(&id, false)
        .await
        .unwrap();
    assert_eq!(dlq.reset_interrupted().await.unwrap(), 1);
    let letter = dlq.get(&id).await.unwrap().unwrap();
    assert_eq!(
        (letter.status, letter.retry_count),
        (DeadLetterStatus::Pending, 1)
    );
    assert_eq!(
        dlq.due(chrono::Utc::now(), 10).await.unwrap(),
        vec![id.clone()]
    );
    // Interrupted again: that was its last retry, so it is exhausted rather
    // than crashing every restart.
    DeadLetterQueue::new(kb.clone())
        .begin_retry(&id, false)
        .await
        .unwrap();
    assert_eq!(dlq.reset_interrupted().await.unwrap(), 1);
    let letter = dlq.get(&id).await.unwrap().unwrap();
    assert_eq!(
        (letter.status, letter.retry_count),
        (DeadLetterStatus::Exhausted, 2)
    );
    assert!(dlq.due(chrono::Utc::now(), 10).await.unwrap().is_empty());
    DeadLetterQueue::new(kb.clone())
        .begin_retry(&id, true)
        .await
        .unwrap();
    assert!(dlq.resolve(&id, Some("stale")).await.unwrap());

    // A retry that fails permanently is exhausted at once; resolving it
    // drops the payload.
    let id = dlq
        .store("ingest", &UnikoError::Timeout(5), Some(&task), 0)
        .await
        .unwrap();
    dlq.begin_retry(&id, false).await.unwrap();
    dlq.retry_failed(&id, "ingest", &UnikoError::IdConflict("m-1".into()))
        .await
        .unwrap();
    let letter = dlq.get(&id).await.unwrap().unwrap();
    assert_eq!(letter.status, DeadLetterStatus::Exhausted);
    assert_eq!(letter.error_kind.as_deref(), Some("id_conflict"));
    assert!(!letter.retriable && letter.replayable);
    assert!(dlq.resolve(&id, None).await.unwrap());
    let letter = dlq.get(&id).await.unwrap().unwrap();
    assert_eq!(letter.status, DeadLetterStatus::Resolved);
    assert!(!letter.replayable, "a resolved letter keeps no payload");

    // No retries allowed: a retriable failure is exhausted, not stuck pending.
    let none = DeadLetterQueue::with_policy(kb.clone(), RetryPolicy::new(0, 10, 40));
    let id = none
        .store("ingest", &UnikoError::Timeout(5), Some(&task), 0)
        .await
        .unwrap();
    assert_eq!(
        none.get(&id).await.unwrap().unwrap().status,
        DeadLetterStatus::Exhausted
    );

    // Listing by status uses the stored status.
    let exhausted = dlq
        .list(&DeadLetterFilter::new().status(DeadLetterStatus::Exhausted))
        .await
        .unwrap();
    assert_eq!(exhausted.len(), 1);
}

/// Runs the real ingest step, then fails the first `fail_times` items after
/// it ran, so a retry replays an item that already partly landed.
struct FailsAfterIngest {
    inner: IngestStep,
    failing: Arc<AtomicBool>,
}

#[async_trait]
impl Step for FailsAfterIngest {
    fn name(&self) -> &str {
        "ingest"
    }

    fn should_run(&self, ctx: &PipelineContext) -> bool {
        self.inner.should_run(ctx)
    }

    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepOutcome, UnikoError> {
        let outcome = self.inner.execute(ctx).await?;
        if self.failing.load(Ordering::SeqCst) {
            return Err(UnikoError::Timeout(1));
        }
        Ok(outcome)
    }
}

async fn count(kb: &KnowledgeBase, cypher: &str) -> i64 {
    match kb
        .query_cypher(cypher, &HashMap::new())
        .await
        .unwrap()
        .first()
        .and_then(|r| r.get("c"))
    {
        Some(Value::Int(n)) => *n,
        _ => 0,
    }
}

/// A message retried after it partly landed keeps its sender, agent and
/// source, and is not written twice.
#[tokio::test(flavor = "multi_thread")]
async fn a_message_retry_keeps_its_attribution_and_writes_once() {
    let kb = kb().await;
    let failing = Arc::new(AtomicBool::new(true));
    let step = FailsAfterIngest {
        inner: IngestStep::new(Arc::new(ModalityRegistry::new())),
        failing: failing.clone(),
    };
    let pipeline = PipelineSystem::new(PipelineConfig::default(), kb.clone(), vec![Box::new(step)]);
    pipeline
        .submit_ingest(message("m-attr", "user", Some("reporter")))
        .unwrap();
    pipeline.quiesce().await;

    let letters = pipeline
        .dead_letters(&DeadLetterFilter::new())
        .await
        .unwrap();
    let [letter] = letters.as_slice() else {
        if count(&kb, "MATCH (m:Message) RETURN count(m) AS c").await == 0 {
            eprintln!("skipping: the ingest step could not run (no embedding model?)");
            return;
        }
        panic!("{letters:?}");
    };
    assert_eq!(letter.agent_id.as_deref(), Some("reporter"));
    assert_eq!(letter.error_kind.as_deref(), Some("timeout"));
    let message_count = "MATCH (m:Message {message_id: 'm-attr'}) RETURN count(m) AS c";
    assert_eq!(
        count(&kb, message_count).await,
        1,
        "the first attempt landed"
    );

    failing.store(false, Ordering::SeqCst);
    pipeline.retry_dead_letter(&letter.id, false).await.unwrap();
    pipeline.quiesce().await;

    let after = pipeline
        .dead_letter_queue()
        .get(&letter.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(after.status, DeadLetterStatus::Resolved);
    assert_eq!(
        count(&kb, message_count).await,
        1,
        "the retry wrote nothing twice"
    );
    assert_eq!(
        count(
            &kb,
            "MATCH (:Message {message_id: 'm-attr'})-[:SENT_BY]->(p:Participant) \
             WHERE p.participant_id = 'user' RETURN count(p) AS c"
        )
        .await,
        1,
        "the retried message keeps its sender"
    );
    assert_eq!(
        count(
            &kb,
            "MATCH (m:Message {message_id: 'm-attr'}) WHERE m.source_id = 'crm' \
             RETURN count(m) AS c"
        )
        .await,
        1,
        "and its source"
    );
    let health = pipeline.health().ingest;
    assert_eq!((health.retries, health.retries_succeeded), (1, 1));
    pipeline.shutdown().await.unwrap();
}
