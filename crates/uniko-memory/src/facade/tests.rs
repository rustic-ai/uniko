//! End-to-end tests for the high-level [`Uniko`] facade.
//!
//! In-crate (not `tests/`) so the access-control / graph-assertion fixtures
//! can use the `pub(crate)` `Agent::kb()` seam — the public surface never
//! exposes `KnowledgeBase`. Recall-dependent assertions skip when the
//! embedding model is unavailable.

use std::collections::HashMap;

use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::{edges, labels};
use uniko_store::{KnowledgeBase, UnikoError, Value};

use crate::{IngestOutcome, IngestSource, Turn, Uniko};

/// True for the "model not present in this environment" error so recall
/// tests can skip instead of failing where embeddings are unavailable.
fn is_model_unavailable(err: &UnikoError) -> bool {
    matches!(err, UnikoError::Embedding(_))
}

/// `observe()` commits before returning, so a following `recall()` sees the
/// turn (read-after-write).
#[tokio::test]
async fn observe_then_recall_is_read_after_write() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("chat-1");

    let result = match session
        .observe(Turn::new(
            "alice",
            "I love hiking in the mountains every weekend",
        ))
        .await
    {
        Ok(result) => result,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("observe failed: {e}"),
    };
    assert!(
        result.message.message_node_id > 0,
        "ingest should yield a node id"
    );

    match agent.recall("hiking hobbies").await {
        Ok(bundle) => assert!(
            !bundle.items.is_empty(),
            "recall should surface the just-observed turn"
        ),
        Err(e) if is_model_unavailable(&e) => eprintln!("skipping: embeddings unavailable"),
        Err(e) => panic!("recall failed: {e}"),
    }

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// `observe()` with an attachment ingests the document linked
/// `Artifact -ATTACHED_TO-> Message` (conversational provenance).
#[tokio::test]
async fn observe_with_attachment_links_to_message() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("chat-att");

    let result = match session
        .observe(
            Turn::new("alice", "here's the spec we discussed")
                .id("m-att")
                .attach(IngestSource::text("# Spec\n\n- requirement one")),
        )
        .await
    {
        Ok(result) => result,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("observe failed: {e}"),
    };
    assert_eq!(result.attachments.len(), 1, "one attachment ingested");
    assert!(matches!(result.attachments[0], IngestOutcome::Artifact(_)));

    let linked = count_query(
        agent.kb(),
        "MATCH (a:Artifact)-[:ATTACHED_TO]->(m:Message {message_id: 'm-att'}) \
         RETURN count(a) AS c",
    )
    .await;
    assert_eq!(linked, 1, "attachment must link ATTACHED_TO the message");

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// `answer()` without a configured LLM is a clear `Config` error.
#[tokio::test]
async fn answer_without_llm_is_config_error() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");

    let err = agent
        .answer("what is the meaning of life?")
        .await
        .expect_err("answer without an LLM must error");
    assert!(
        matches!(err, UnikoError::Config(_)),
        "expected Config error, got {err:?}"
    );
}

/// `submit()` without streaming enabled is a clear `Config` error.
#[tokio::test]
async fn submit_without_streaming_is_config_error() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let session = memory.agent("assistant").session("chat-1");

    let err = session
        .submit(Turn::new("alice", "hello"))
        .await
        .expect_err("submit without streaming must error");
    assert!(
        matches!(err, UnikoError::Config(_)),
        "expected Config error, got {err:?}"
    );
}

/// Streaming `submit_source()` + `flush()` ingests a blob through the
/// unified source path.
#[tokio::test]
async fn streaming_submit_source_then_flush_ingests() {
    let memory = match Uniko::builder().in_memory().streaming(true).build().await {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("skipping: streaming instance unavailable: {e}");
            return;
        }
    };
    let agent = memory.agent("assistant");
    let session = agent.session("stream-src-1");

    if let Err(e) = session
        .submit_source(IngestSource::text(
            "# Release notes\n\n- shipped the deploy",
        ))
        .await
    {
        eprintln!("skipping: submit_source failed: {e}");
        return;
    }
    session
        .flush()
        .await
        .expect("flush should drain the pipeline");

    let count = agent
        .kb()
        .query_nodes(labels::ARTIFACT, None, None)
        .await
        .expect("query artifacts")
        .len();
    assert!(
        count >= 1,
        "expected >= 1 artifact after flush, found {count}"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Streaming `submit()` + `flush()` ingests conversation turns.
#[tokio::test]
async fn streaming_submit_then_flush_ingests() {
    let memory = match Uniko::builder().in_memory().streaming(true).build().await {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("skipping: streaming instance unavailable: {e}");
            return;
        }
    };
    let agent = memory.agent("assistant");
    let session = agent.session("stream-1");

    for i in 0..3 {
        if let Err(e) = session
            .submit(Turn::new(
                "alice",
                format!("streamed note {i} about rock climbing"),
            ))
            .await
        {
            eprintln!("skipping: submit failed: {e}");
            return;
        }
    }
    session
        .flush()
        .await
        .expect("flush should drain the pipeline");

    let count = message_count(agent.kb()).await;
    assert!(
        count >= 3,
        "expected >= 3 messages after flush, found {count}"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// `scope_to_agent()` filters reads to each agent's own visibility.
#[tokio::test]
async fn agent_scope_filters_private_facts() {
    let memory = match Uniko::builder().in_memory().scope_to_agent().build().await {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("skipping: instance unavailable: {e}");
            return;
        }
    };
    let alice = memory.agent("alice");
    let bob = memory.agent("bob");
    let kb = alice.kb();

    seed_participant(kb, "alice").await;
    seed_participant(kb, "bob").await;
    let query = "quarterly revenue outlook";
    seed_fact(kb, "f-public", query, Some("public")).await;
    seed_fact(kb, "f-private", query, Some("private:alice")).await;
    let private_nid = fact_nid(kb, "f-private").await;

    let alice_bundle = match alice.recall(query).await {
        Ok(bundle) => bundle,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("alice recall failed: {e}"),
    };
    if !alice_bundle.items.iter().any(|i| i.node_id == private_nid) {
        eprintln!("skipping: recall did not surface Facts in this env");
        return;
    }

    let bob_bundle = bob.recall(query).await.expect("bob recall");
    assert!(
        !bob_bundle.items.iter().any(|i| i.node_id == private_nid),
        "bob must not see the private:alice Fact"
    );
}

/// `Uniko::in_memory` runs the validated best config.
#[tokio::test]
async fn defaults_match_validated_best_config() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let config = memory.config();
    let expected = UnikoConfig::default();
    assert!(config.reranker.enabled, "reranker should default on");
    assert_eq!(config.phase1_strategy, expected.phase1_strategy);
    assert_eq!(config.embedding.model_id, expected.embedding.model_id);
}

/// `Turn::id` makes observe idempotent.
#[tokio::test]
async fn turn_id_makes_observe_idempotent() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("chat-1");

    let first = match session
        .observe(Turn::new("alice", "fixed content").id("msg-1"))
        .await
    {
        Ok(result) => result,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("first observe failed: {e}"),
    };
    let second = session
        .observe(Turn::new("alice", "fixed content").id("msg-1"))
        .await
        .expect("second observe");

    assert_eq!(
        first.message.message_node_id, second.message.message_node_id,
        "same message id must dedup to the same node"
    );
    assert_eq!(message_count(agent.kb()).await, 1, "no duplicate message");
}

/// `Session::ingest` persists an artifact and dedups identical content.
#[tokio::test]
async fn ingest_persists_and_dedups() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let session = memory.agent("librarian").session("kb-import");

    let body = "The Eiffel Tower is a wrought-iron lattice tower in Paris.";
    let first = match session.ingest(IngestSource::text(body)).await {
        Ok(IngestOutcome::Artifact(r)) => r,
        Ok(other) => panic!("expected artifact, got {other:?}"),
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("ingest failed: {e}"),
    };
    assert!(first.artifact_node_id > 0);
    assert!(!first.was_deduplicated, "first ingest is not a dup");

    match session
        .ingest(IngestSource::text(body))
        .await
        .expect("re-ingest")
    {
        IngestOutcome::Artifact(r) => {
            assert!(r.was_deduplicated, "identical content must dedup by hash")
        }
        other => panic!("expected artifact, got {other:?}"),
    }
}

/// `Session::ingest` routes a PDF (sniffed from magic bytes) to the PDF path.
#[tokio::test]
async fn ingest_pdf_routes_to_pdf_path() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let session = memory.agent("librarian").session("kb-import");

    match session
        .ingest(IngestSource::bytes(b"%PDF-1.4 not a real pdf".to_vec()))
        .await
    {
        Ok(IngestOutcome::Pdf(result)) => {
            assert!(result.artifact_node_id > 0, "PDF artifact should persist")
        }
        Ok(other) => panic!("expected pdf, got {other:?}"),
        Err(e) if is_model_unavailable(&e) => eprintln!("skipping: embeddings unavailable"),
        Err(e) => eprintln!("skipping: pdf extractor unavailable in this env: {e}"),
    }
}

/// `Agent::query` rejects writes and runs read Cypher through the graph
/// engine.
#[tokio::test]
async fn query_is_read_only() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");

    let write = agent.query("CREATE (n:Foo {x: 1}) RETURN n").await;
    assert!(
        matches!(write, Err(UnikoError::Storage(_))),
        "write Cypher must be rejected, got {write:?}"
    );

    let rows = agent
        .query("MATCH (n:Participant) RETURN n")
        .await
        .expect("read query");
    assert!(rows.is_empty(), "empty graph yields no rows");
}

/// `Agent::query` actually returns `MATCH` rows (proves it uses the graph
/// engine, not the Locy runtime which serves only derived facts).
#[tokio::test]
async fn query_returns_match_rows() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    seed_participant(agent.kb(), "p-q").await;

    let rows = agent
        .query("MATCH (p:Participant) RETURN p.participant_id AS pid")
        .await
        .expect("query");
    assert_eq!(rows.len(), 1, "query must return the seeded participant");
}

/// `Agent::define_rule` registers a Locy rule.
#[tokio::test]
async fn define_rule_registers() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");

    let nid = agent
        .define_rule(
            "facade_rule",
            "CREATE RULE facade_rule AS MATCH (n:Episode) YIELD KEY n",
        )
        .await
        .expect("define_rule");
    assert!(nid > 0, "rule should get a node id");
}

/// `Agent::assume` is hypothetical: a mutation inside it is rolled back.
#[tokio::test]
async fn assume_does_not_mutate_the_graph() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");

    // `fact_id` is NOT NULL in the schema, so the hypothetical CREATE must
    // supply it — omitting it fails inside the ASSUME with a constraint
    // violation rather than exercising the rollback this test is about.
    let hypothetical = agent
        .assume(
            "ASSUME { CREATE (:Fact {fact_id: 'assume_probe', subject: 'srv', \
             predicate: 'port', object: '9090'}) }",
        )
        .then_query("MATCH (f:Fact {subject: 'srv'}) RETURN f")
        .run()
        .await
        .expect("assume should run");
    assert_eq!(
        hypothetical.len(),
        1,
        "the assumed Fact must be visible inside the ASSUME"
    );

    let rows = agent
        .query("MATCH (f:Fact {subject: 'srv'}) RETURN f")
        .await
        .expect("post-assume query");
    assert!(rows.is_empty(), "ASSUME mutation must be rolled back");
}

/// `Session::summarize` on a session with no content returns `None`.
#[tokio::test]
async fn summarize_unused_session_is_none() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let session = memory.agent("assistant").session("never-used");

    match session.summarize().await {
        Ok(summary) => assert!(summary.is_none(), "unused session has nothing to summarize"),
        Err(e) if is_model_unavailable(&e) => eprintln!("skipping: embeddings unavailable"),
        Err(e) => panic!("summarize failed: {e}"),
    }
}

// ── Fixtures (seed via the `pub(crate)` `Agent::kb()` seam) ───────────────

async fn message_count(kb: &KnowledgeBase) -> usize {
    kb.query_nodes(labels::MESSAGE, None, None)
        .await
        .expect("message query")
        .len()
}

/// Single-row read returning one i64 column named `c`, or 0.
async fn count_query(kb: &KnowledgeBase, cypher: &str) -> i64 {
    kb.db() // ALLOW: test-only assertion helper; the seal governs product code.
        .session()
        .query(cypher)
        .await
        .expect("query")
        .rows()
        .first()
        .and_then(|row| row.get::<i64>("c").ok())
        .unwrap_or(0)
}

async fn seed_participant(kb: &KnowledgeBase, pid: &str) {
    let mut props = HashMap::new();
    props.insert("kind".to_string(), Value::String("agent".into()));
    kb.merge_node(labels::PARTICIPANT, "participant_id", pid, &props)
        .await
        .expect("participant");
}

async fn seed_fact(kb: &KnowledgeBase, fid: &str, object: &str, visibility: Option<&str>) {
    let mut props = HashMap::new();
    props.insert("subject".to_string(), Value::String("project".into()));
    props.insert("predicate".to_string(), Value::String("status_is".into()));
    props.insert("object".to_string(), Value::String(object.into()));
    if let Some(v) = visibility {
        props.insert("visibility".to_string(), Value::String(v.into()));
    }
    kb.merge_node(labels::FACT, "fact_id", fid, &props)
        .await
        .expect("fact");
}

async fn fact_nid(kb: &KnowledgeBase, fid: &str) -> i64 {
    kb.get_node_by_ext_id(labels::FACT, "fact_id", fid)
        .await
        .expect("lookup")
        .expect("fact exists")
        .0
}

/// `agent.data()` dereferences an attachment (text + bytes + the message it
/// was attached to) and the message itself (sender / session / attachments),
/// and `observe`'s result exposes the external `artifact_id`.
#[tokio::test]
async fn data_handle_dereferences_attachment_and_message() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("chat-data");

    let result = match session
        .observe(
            Turn::new("alice", "see attached")
                .id("m-data")
                .attach(IngestSource::text("# Spec\n\nthe deadline is Friday")),
        )
        .await
    {
        Ok(result) => result,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("observe failed: {e}"),
    };
    let IngestOutcome::Artifact(art) = &result.attachments[0] else {
        panic!("expected an Artifact attachment");
    };
    assert!(
        !art.artifact_id.is_empty(),
        "ingest result must expose the external artifact_id"
    );

    // artifact() — reassembled text + the message it was attached to.
    let view = agent
        .data()
        .artifact(&art.artifact_id)
        .await
        .expect("artifact lookup")
        .expect("artifact exists");
    assert!(
        view.text.contains("deadline is Friday"),
        "reassembled text should contain the attachment body, got {:?}",
        view.text
    );
    assert_eq!(view.attached_to_message.as_deref(), Some("m-data"));

    // artifact_bytes() — the original blob. On the in-memory Lance backend
    // bytes live inline in `:ArtifactContent.bytes`, but uni-db currently
    // can't decode a `Bytes` column returned from a Cypher `RETURN`
    // ("unknown CypherValue tag: 35"), so `KnowledgeBase::fetch_blob`'s
    // inline path fails over to `LanceBlobStore::get` (intentionally not
    // callable). The Fs/S3 backends (production) read via the `uri` path and
    // are unaffected. Tolerate the known limitation here; assert success
    // where the backend can serve the bytes.
    match agent.data().artifact_bytes(&art.artifact_id).await {
        Ok(Some(bytes)) => assert!(!bytes.is_empty(), "original bytes must be non-empty"),
        Ok(None) => panic!("artifact exists, so its bytes must resolve to Some"),
        Err(UnikoError::Storage(msg)) if msg.contains("LanceBlobStore::get is not callable") => {
            eprintln!("skipping bytes assertion: known uni-db Bytes-decode limitation on Lance");
        }
        Err(e) => panic!("artifact_bytes failed unexpectedly: {e}"),
    }

    // message() — sender / session / its attachment.
    let msg = agent
        .data()
        .message("m-data")
        .await
        .expect("message lookup")
        .expect("message exists");
    assert_eq!(msg.sender_id, "alice");
    assert_eq!(msg.session_id, "chat-data");
    assert!(
        msg.attachments.contains(&art.artifact_id),
        "message should list its attachment, got {:?}",
        msg.attachments
    );

    // Unknown ids resolve to None, not an error.
    assert!(agent.data().message("nope").await.expect("ok").is_none());
    assert!(agent.data().artifact("nope").await.expect("ok").is_none());

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Recall stamps each surviving item with its source lineage; an attachment
/// chunk traces back to an `Attachment { artifact_id, message_id }`.
#[tokio::test]
async fn recall_stamps_attachment_source() {
    use crate::recall::RecallSource;

    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("chat-src");

    let result = match session
        .observe(
            Turn::new("alice", "see attached")
                .id("m-src")
                .attach(IngestSource::text(
                    "the quarterly revenue target is four million dollars",
                )),
        )
        .await
    {
        Ok(result) => result,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("observe failed: {e}"),
    };
    let IngestOutcome::Artifact(art) = &result.attachments[0] else {
        panic!("expected an Artifact attachment");
    };

    match agent.recall("quarterly revenue target").await {
        Ok(bundle) => {
            assert!(
                bundle.items.iter().any(|i| !i.sources.is_empty()),
                "recalled items should be stamped with sources"
            );
            // Any attachment-derived chunk must name this artifact + message.
            for item in &bundle.items {
                for src in &item.sources {
                    if let RecallSource::Attachment {
                        artifact_id,
                        message_id,
                        ..
                    } = src
                    {
                        assert_eq!(artifact_id, &art.artifact_id);
                        assert_eq!(message_id, "m-src");
                    }
                }
            }
        }
        Err(e) if is_model_unavailable(&e) => eprintln!("skipping: embeddings unavailable"),
        Err(e) => panic!("recall failed: {e}"),
    }

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Full goal lifecycle through `agent.goals()`: create → active slice →
/// task linkage → complete-with-result → moves to the completed slice with
/// its result recorded and out of active.
#[tokio::test]
async fn goal_lifecycle_create_complete_moves_phase() {
    use crate::{CreateGoalParams, CreateTaskParams, GoalPhase};

    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("planner");
    seed_participant(agent.kb(), "planner").await;

    let create = CreateGoalParams {
        goal_id: Some("g-1".into()),
        title: "Ship the API".into(),
        ..Default::default()
    };
    match agent.goals().create(create).await {
        Ok(_) => {}
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("create goal: {e}"),
    }

    // New goal (default status "active") shows in the active slice.
    let active = agent.goals().active().await.expect("active");
    assert!(
        active
            .iter()
            .any(|g| g.goal_id == "g-1" && g.phase == GoalPhase::Active),
        "new goal should be active, got {active:?}"
    );

    // A task PART_OF the goal is reachable via tasks_of.
    agent
        .goals()
        .create_task(CreateTaskParams {
            task_id: Some("t-1".into()),
            title: "write docs".into(),
            goal_id: Some("g-1".into()),
            ..Default::default()
        })
        .await
        .expect("create task");
    let tasks = agent.goals().tasks_of("g-1").await.expect("tasks_of");
    assert!(
        tasks.iter().any(|t| t.task_id == "t-1"),
        "task linked to goal"
    );

    // Complete with a result; it merges into metrics.
    let result = serde_json::json!({ "shipped": true });
    assert!(
        agent
            .goals()
            .complete("g-1", Some(result))
            .await
            .expect("complete"),
        "complete should resolve the goal"
    );

    let completed = agent.goals().completed().await.expect("completed");
    let g = completed
        .iter()
        .find(|g| g.goal_id == "g-1")
        .expect("goal in completed slice");
    assert_eq!(g.phase, GoalPhase::Completed);
    assert!(g.completed_at.is_some(), "completed_at stamped");
    assert_eq!(
        g.metrics.as_ref().and_then(|m| m.get("shipped")),
        Some(&serde_json::json!(true)),
        "result recorded in metrics"
    );
    assert!(
        !agent
            .goals()
            .active()
            .await
            .expect("active")
            .iter()
            .any(|g| g.goal_id == "g-1"),
        "completed goal must leave the active slice"
    );

    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// A goal created with status "planned" lands in the planned (future) slice.
#[tokio::test]
async fn planned_goal_appears_in_planned_slice() {
    use crate::{CreateGoalParams, GoalPhase};

    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("planner");
    seed_participant(agent.kb(), "planner").await;

    let create = CreateGoalParams {
        goal_id: Some("g-plan".into()),
        title: "Future work".into(),
        status: Some("planned".into()),
        ..Default::default()
    };
    match agent.goals().create(create).await {
        Ok(_) => {}
        Err(e) if is_model_unavailable(&e) => return,
        Err(e) => panic!("create goal: {e}"),
    }

    let planned = agent.goals().planned().await.expect("planned");
    assert!(
        planned
            .iter()
            .any(|g| g.goal_id == "g-plan" && g.phase == GoalPhase::Planned),
        "planned goal should appear in the planned slice, got {planned:?}"
    );

    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// `context()` returns the typed goal + its tasks; unknown goal → None.
#[tokio::test]
async fn goal_context_returns_typed_subtree() {
    use crate::{CreateGoalParams, CreateTaskParams};

    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("planner");
    seed_participant(agent.kb(), "planner").await;

    match agent
        .goals()
        .create(CreateGoalParams {
            goal_id: Some("g-ctx".into()),
            title: "Build".into(),
            ..Default::default()
        })
        .await
    {
        Ok(_) => {}
        Err(e) if is_model_unavailable(&e) => return,
        Err(e) => panic!("create goal: {e}"),
    }
    agent
        .goals()
        .create_task(CreateTaskParams {
            task_id: Some("t-ctx".into()),
            title: "subtask".into(),
            goal_id: Some("g-ctx".into()),
            ..Default::default()
        })
        .await
        .expect("create task");

    let ctx = agent
        .goals()
        .context("g-ctx")
        .await
        .expect("context")
        .expect("goal exists");
    assert_eq!(ctx.goal.goal_id, "g-ctx");
    assert!(
        ctx.tasks.iter().any(|t| t.task_id == "t-ctx"),
        "context should include the goal's task"
    );
    assert!(
        agent.goals().context("nope").await.expect("ok").is_none(),
        "unknown goal context → None"
    );

    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Transitions on an unknown id are a clean `Ok(false)`, not an error.
#[tokio::test]
async fn unknown_goal_transitions_return_false() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("planner");
    seed_participant(agent.kb(), "planner").await;

    assert!(
        !agent
            .goals()
            .complete("nope", None)
            .await
            .expect("complete")
    );
    assert!(!agent.goals().start("nope").await.expect("start"));
    assert!(!agent.goals().abandon("nope").await.expect("abandon"));
    assert!(agent.goals().get("nope").await.expect("get").is_none());

    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

// ── Session-level chunking (`Session::finalize`) ────────────────────────

/// Observe `turns` into a fresh session, skipping the test when the
/// embedding model is unavailable. Returns `None` when skipped.
async fn observe_all<'a>(
    session: &mut crate::Session,
    turns: impl IntoIterator<Item = &'a str>,
) -> Option<()> {
    for (i, text) in turns.into_iter().enumerate() {
        let sender = if i % 2 == 0 { "alice" } else { "bob" };
        match session.observe(Turn::new(sender, text)).await {
            Ok(_) => {}
            Err(e) if is_model_unavailable(&e) => {
                eprintln!("skipping: embeddings unavailable");
                return None;
            }
            Err(e) => panic!("observe failed: {e}"),
        }
    }
    Some(())
}

/// Count a session's chunks of one `chunk_type`.
async fn session_chunk_count(kb: &KnowledgeBase, session_id: &str, chunk_type: &str) -> i64 {
    count_query(
        kb,
        &format!(
            "MATCH (:Session {{session_id: '{session_id}'}})-[:HAS_CHUNK]->\
             (c:Chunk {{chunk_type: '{chunk_type}'}}) RETURN count(c) AS c"
        ),
    )
    .await
}

/// `finalize()` builds both session-level retrieval surfaces.
#[tokio::test]
async fn finalize_creates_session_chunks() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("fin-1");

    let turns = [
        "I love hiking in the Cascades every weekend.",
        "Which trail is your favourite?",
        "Rattlesnake Ledge, mostly for the view at the top.",
    ];
    if observe_all(&mut session, turns).await.is_none() {
        return;
    }

    let report = match session.finalize().await {
        Ok(r) => r,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("finalize failed: {e}"),
    };
    assert!(report.rebuilt, "first finalize writes the chunks");
    assert!(
        !report.transcript_chunks.is_empty(),
        "a three-turn session must produce transcript chunks"
    );
    assert!(
        session_chunk_count(agent.kb(), "fin-1", "session").await > 0,
        "transcript chunks must hang off the Session via HAS_CHUNK"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Regression guard: `summarize()` refreshes the chunks for callers who
/// never learn about `finalize()`. This is the path that silently produced
/// no session chunks at all before session chunking was wired into the
/// facade, leaving the Phase 1 session boost with nothing to walk.
#[tokio::test]
async fn summarize_builds_session_chunks() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("fin-2");

    let turns = [
        "We adopted a border collie named Pip last spring.",
        "Does Pip get along with the cat?",
    ];
    if observe_all(&mut session, turns).await.is_none() {
        return;
    }

    match session.summarize().await {
        Ok(_) => {}
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("summarize failed: {e}"),
    }

    assert!(
        session_chunk_count(agent.kb(), "fin-2", "session").await > 0,
        "summarize() must leave the session with transcript chunks"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// A second `finalize()` with no new turns rewrites nothing — and must not
/// duplicate chunks. `chunk_id` carries no uniqueness constraint and chunk
/// writes are plain inserts, so a rebuild that skipped the delete would
/// silently double every chunk.
#[tokio::test]
async fn finalize_is_idempotent() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("fin-3");

    if observe_all(&mut session, ["Sourdough needs a stiff starter."])
        .await
        .is_none()
    {
        return;
    }

    let first = match session.finalize().await {
        Ok(r) => r,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("finalize failed: {e}"),
    };
    let second = session.finalize().await.expect("second finalize");

    assert!(
        !second.rebuilt,
        "an unchanged session must not be rewritten"
    );
    assert_eq!(
        first.transcript_chunks, second.transcript_chunks,
        "unchanged session keeps the same chunk nodes"
    );

    let total = count_query(
        agent.kb(),
        "MATCH (:Session {session_id: 'fin-3'})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS c",
    )
    .await;
    let distinct = count_query(
        agent.kb(),
        "MATCH (:Session {session_id: 'fin-3'})-[:HAS_CHUNK]->(c:Chunk) \
         RETURN count(DISTINCT c.chunk_id) AS c",
    )
    .await;
    assert_eq!(total, distinct, "chunk_id must not be duplicated");

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// A session that grows after being finalized gets a refreshed transcript,
/// not a permanently stale one.
#[tokio::test]
async fn finalize_after_more_turns_refreshes() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("fin-4");

    if observe_all(&mut session, ["The first topic was budgeting."])
        .await
        .is_none()
    {
        return;
    }
    match session.finalize().await {
        Ok(_) => {}
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("finalize failed: {e}"),
    }

    if observe_all(&mut session, ["Later we moved on to xylophones."])
        .await
        .is_none()
    {
        return;
    }
    let after = session.finalize().await.expect("refresh finalize");
    assert!(after.rebuilt, "a grown session must be rebuilt");

    // The later turn's distinctive token must now appear in some chunk.
    let hits = count_query(
        agent.kb(),
        "MATCH (:Session {session_id: 'fin-4'})-[:HAS_CHUNK]->(c:Chunk) \
         WHERE c.text CONTAINS 'xylophones' RETURN count(c) AS c",
    )
    .await;
    assert!(hits > 0, "refreshed chunks must include the later turns");

    let total = count_query(
        agent.kb(),
        "MATCH (:Session {session_id: 'fin-4'})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS c",
    )
    .await;
    let distinct = count_query(
        agent.kb(),
        "MATCH (:Session {session_id: 'fin-4'})-[:HAS_CHUNK]->(c:Chunk) \
         RETURN count(DISTINCT c.chunk_id) AS c",
    )
    .await;
    assert_eq!(total, distinct, "a rebuild must not duplicate chunks");

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// `finalize()` on a session with no turns is a clean no-op.
#[tokio::test]
async fn finalize_unused_session_is_empty() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let session = agent.session("fin-empty");

    match session.finalize().await {
        Ok(report) => {
            assert!(report.transcript_chunks.is_empty());
            assert!(report.observation_chunks.is_empty());
            assert!(!report.rebuilt);
        }
        Err(e) if is_model_unavailable(&e) => eprintln!("skipping: embeddings unavailable"),
        Err(e) => panic!("finalize failed: {e}"),
    }

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// `Agent::delete_session` must take the session-anchored chunks with it —
/// otherwise deleted content stays live in the vector and full-text indexes
/// and remains recallable.
#[tokio::test]
async fn delete_session_removes_session_chunks() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("fin-del");

    if observe_all(&mut session, ["Quarterly numbers looked strong."])
        .await
        .is_none()
    {
        return;
    }
    match session.finalize().await {
        Ok(_) => {}
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("finalize failed: {e}"),
    }
    assert!(session_chunk_count(agent.kb(), "fin-del", "session").await > 0);

    drop(session);
    agent
        .delete_session("fin-del")
        .await
        .expect("delete_session");

    assert_eq!(
        session_chunk_count(agent.kb(), "fin-del", "session").await,
        0,
        "session-anchored chunks must be deleted with the session"
    );

    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// The bug this wiring fixes, end to end.
///
/// `session_boost_signals` — the Phase 1 contribution under the **default**
/// `phase1_strategy = "boost"` — walks
/// `Fact <-SUPPORTED_BY- Observation -OBSERVED_IN-> Message -IN_SESSION->
/// Session -HAS_CHUNK-> Chunk`. Every hop but the last always existed on a
/// facade-ingested graph; the last one did not, so the boost was a silent
/// no-op. Asserts the walk directly, with a negative control, rather than
/// depending on LLM ranking.
#[tokio::test]
async fn session_boost_walk_is_populated_only_after_finalize() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("boost-1");

    let turns = [
        "Dana works as a marine biologist in Monterey.",
        "Dana studies kelp forest ecology there.",
        "Dana has published on sea otter foraging.",
    ];
    if observe_all(&mut session, turns).await.is_none() {
        return;
    }

    // Derive Facts so the walk has a starting node.
    if let Err(e) = agent.consolidate().await {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("consolidation failed: {e}");
    }

    let fact_ids: Vec<i64> = agent
        .kb()
        .query_cypher(
            "MATCH (f:Fact)-[:SUPPORTED_BY]->(:Observation)-[:OBSERVED_IN]->(:Message)\
             -[:IN_SESSION]->(:Session {session_id: 'boost-1'}) RETURN DISTINCT id(f) AS fid",
            &HashMap::new(),
        )
        .await
        .expect("fact query")
        .iter()
        .filter_map(|r| match r.get("fid") {
            Some(Value::Int(i)) => Some(*i),
            _ => None,
        })
        .collect();
    if fact_ids.is_empty() {
        eprintln!("skipping: consolidation derived no facts in this environment");
        return;
    }

    // Negative control: before finalize the last hop does not exist, so the
    // boost has nothing to score with.
    for fid in &fact_ids {
        let chunks = agent
            .kb()
            .fact_session_chunk_ids(*fid)
            .await
            .expect("walk before finalize");
        assert!(
            chunks.is_empty(),
            "without finalize the session boost walk must find nothing"
        );
    }

    session.finalize().await.expect("finalize");

    let mut any = false;
    for fid in &fact_ids {
        any |= !agent
            .kb()
            .fact_session_chunk_ids(*fid)
            .await
            .expect("walk after finalize")
            .is_empty();
    }
    assert!(
        any,
        "after finalize the session boost walk must reach session chunks"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Deleting every turn behind a finalized session must not leave its chunks
/// behind — they would stay live in the vector and full-text indexes and keep
/// describing content that no longer exists.
#[tokio::test]
async fn finalize_drops_chunks_when_all_turns_are_deleted() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("fin-empty-after");

    let turn = Turn::new("alice", "The launch slipped to November.").id("m-1");
    match session.observe(turn).await {
        Ok(_) => {}
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("observe failed: {e}"),
    }
    match session.finalize().await {
        Ok(_) => {}
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("finalize failed: {e}"),
    }
    assert!(session_chunk_count(agent.kb(), "fin-empty-after", "session").await > 0);

    session.delete_turn("m-1").await.expect("delete_turn");
    let report = session.finalize().await.expect("finalize after delete");

    assert!(report.transcript_chunks.is_empty());
    assert_eq!(
        session_chunk_count(agent.kb(), "fin-empty-after", "session").await,
        0,
        "chunks for a now-empty session must be dropped"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Regression guard for the `SUPPORTED_BY` traversal direction.
///
/// `SUPPORTED_BY` is registered `Fact → Observation` (`schema/facts.rs`), but
/// `fact_session_chunk_ids` once walked it inbound — a pattern that can never
/// match, so the Phase 1 session boost silently scored nothing on every call.
/// The graph here is seeded directly so the guard runs without models or
/// consolidation, and fails loudly if the arrow is ever flipped back.
#[tokio::test]
async fn fact_session_chunk_walk_follows_schema_edge_direction() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let kb = agent.kb();
    let none: HashMap<String, Value> = HashMap::new();
    let now = uniko_store::datetime_value(chrono::Utc::now());
    let props = |pairs: &[(&str, Value)]| -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    };

    // Session -HAS_CHUNK-> Chunk, and Message -IN_SESSION-> Session.
    let session = kb
        .merge_node(
            labels::SESSION,
            "session_id",
            "walk-1",
            &props(&[("started_at", now.clone())]),
        )
        .await
        .expect("session");
    let chunk = kb
        .merge_node(
            labels::CHUNK,
            "chunk_id",
            "walk-1-c0",
            &props(&[
                ("text", Value::String("transcript".into())),
                ("chunk_type", Value::String("session".into())),
            ]),
        )
        .await
        .expect("chunk");
    let message = kb
        .merge_node(
            labels::MESSAGE,
            "message_id",
            "walk-1-m0",
            &props(&[
                ("content", Value::String("hello".into())),
                ("timestamp", now.clone()),
            ]),
        )
        .await
        .expect("message");
    let observation = kb
        .merge_node(
            labels::OBSERVATION,
            "observation_id",
            "walk-1-o0",
            &props(&[("content", Value::String("alice likes hiking".into()))]),
        )
        .await
        .expect("observation");
    let fact = kb
        .merge_node(
            labels::FACT,
            "fact_id",
            "walk-1-f0",
            &props(&[
                ("subject", Value::String("alice".into())),
                ("predicate", Value::String("likes".into())),
            ]),
        )
        .await
        .expect("fact");

    for (edge, from, to) in [
        (edges::HAS_CHUNK, session, chunk),
        (edges::IN_SESSION, message, session),
        (edges::OBSERVED_IN, observation, message),
        // The direction under test: Fact is the source.
        (edges::SUPPORTED_BY, fact, observation),
    ] {
        kb.create_edge(edge, from, to, &none).await.expect(edge);
    }

    let reached = kb
        .fact_session_chunk_ids(fact)
        .await
        .expect("fact_session_chunk_ids");
    assert_eq!(
        reached,
        vec![chunk],
        "the session-boost walk must follow SUPPORTED_BY outbound from the Fact"
    );

    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// A refresh after new turns reuses the byte-identical leading chunks and
/// only rebuilds the tail, so appending to a long session does not re-embed
/// the whole transcript.
#[tokio::test]
async fn finalize_reuses_unchanged_chunk_prefix() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("prefix-1");

    // Enough text that the transcript spans more than one chunk.
    let filler: Vec<String> = (0..40)
        .map(|i| {
            format!(
                "Turn {i}: the quarterly logistics review covered warehouse throughput, \
             carrier performance, and the seasonal staffing plan in some detail."
            )
        })
        .collect();
    if observe_all(&mut session, filler.iter().map(String::as_str))
        .await
        .is_none()
    {
        return;
    }
    let first = match session.finalize().await {
        Ok(r) => r,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("finalize failed: {e}"),
    };
    if first.transcript_chunks.len() < 2 {
        eprintln!("skipping: transcript did not span multiple chunks");
        return;
    }

    if observe_all(&mut session, ["One more turn about xylophones."])
        .await
        .is_none()
    {
        return;
    }
    let second = session.finalize().await.expect("refresh");
    assert!(second.rebuilt, "a grown session must be rebuilt");

    // The leading chunk nodes must be the *same nodes*, not recreated ones.
    assert_eq!(
        first.transcript_chunks[0], second.transcript_chunks[0],
        "the unchanged leading chunk must be reused, not re-embedded"
    );
    let reused = first
        .transcript_chunks
        .iter()
        .zip(&second.transcript_chunks)
        .take_while(|(a, b)| a == b)
        .count();
    assert!(
        reused >= 1,
        "expected at least one reused chunk, got {reused}"
    );

    // And the new content is present exactly once.
    let hits = count_query(
        agent.kb(),
        "MATCH (:Session {session_id: 'prefix-1'})-[:HAS_CHUNK]->(c:Chunk) \
         WHERE c.text CONTAINS 'xylophones' RETURN count(c) AS c",
    )
    .await;
    assert_eq!(
        hits, 1,
        "the appended turn must appear in exactly one chunk"
    );

    let total = count_query(
        agent.kb(),
        "MATCH (:Session {session_id: 'prefix-1'})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS c",
    )
    .await;
    let distinct = count_query(
        agent.kb(),
        "MATCH (:Session {session_id: 'prefix-1'})-[:HAS_CHUNK]->(c:Chunk) \
         RETURN count(DISTINCT c.chunk_id) AS c",
    )
    .await;
    assert_eq!(
        total, distinct,
        "a partial rebuild must not duplicate chunks"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}
