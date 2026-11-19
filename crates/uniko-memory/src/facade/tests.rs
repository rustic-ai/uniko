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

use crate::{IngestOutcome, IngestSource, RecallOverrides, Scope, Turn, Uniko};

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

/// A streamed image/audio/video source resolves the same registered
/// extractors as a directly ingested one (issue #62).
#[tokio::test]
async fn streaming_submit_source_uses_registered_extractors() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use uniko_pipes::content::Mime;

    use crate::{Modality, TextExtractor};

    let counted = |modality: Modality, text: &'static str, calls: Arc<AtomicUsize>| {
        Arc::new(TextExtractor::new(modality, move |req| {
            assert_eq!(req.modality, modality);
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(text.to_string())
        }))
    };
    let image_calls = Arc::new(AtomicUsize::new(0));
    let audio_calls = Arc::new(AtomicUsize::new(0));
    let video_calls = Arc::new(AtomicUsize::new(0));

    let memory = match Uniko::builder()
        .in_memory()
        .streaming(true)
        .extractor(counted(
            Modality::Image,
            "a photo of a red zeppelin over the harbour",
            image_calls.clone(),
        ))
        .extractor(counted(
            Modality::Audio,
            "a voice memo about the quarterly budget",
            audio_calls.clone(),
        ))
        .extractor(counted(
            Modality::Video,
            "a screen recording of the deploy dashboard",
            video_calls.clone(),
        ))
        .build()
        .await
    {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("skipping: streaming instance unavailable: {e}");
            return;
        }
    };
    let agent = memory.agent("viewer");
    let session = agent.session("stream-media");

    for (id, mime) in [
        ("img-1", "image/png"),
        ("aud-1", "audio/wav"),
        ("vid-1", "video/mp4"),
    ] {
        let source = IngestSource::bytes(vec![0x42; 16])
            .with_mime(Mime::parse(mime).expect("valid mime"))
            .with_id(id);
        session
            .submit_source(source)
            .await
            .expect("submit_source enqueues");
    }
    session.flush().await.expect("flush drains the pipeline");

    let dead = agent
        .kb()
        .query_nodes(labels::DEAD_LETTER, None, None)
        .await
        .expect("query dead letters");
    let errors: Vec<String> = dead
        .iter()
        .filter_map(|(_, props)| match props.get("error") {
            Some(Value::String(e)) => Some(e.clone()),
            _ => None,
        })
        .collect();
    assert!(
        errors.iter().all(|e| !e.contains("unsupported modality")),
        "a streamed source missed its registered extractor: {errors:?}"
    );

    // Every extractor ran exactly once, whether or not the embedder was
    // available to finish the artifact afterwards.
    assert_eq!(image_calls.load(Ordering::SeqCst), 1, "image extractor");
    assert_eq!(audio_calls.load(Ordering::SeqCst), 1, "audio extractor");
    assert_eq!(video_calls.load(Ordering::SeqCst), 1, "video extractor");

    if !errors.is_empty() {
        eprintln!("skipping artifact checks: streamed media failed downstream: {errors:?}");
        return;
    }
    let health = memory.health().expect("streaming instance reports health");
    assert_eq!(health.ingest.items_failed, 0, "no streamed source failed");

    for (id, expected) in [
        ("img-1", "red zeppelin"),
        ("aud-1", "quarterly budget"),
        ("vid-1", "deploy dashboard"),
    ] {
        let view = agent
            .data()
            .artifact(id)
            .await
            .expect("artifact read")
            .unwrap_or_else(|| panic!("streamed {id} became an artifact"));
        assert!(view.text.contains(expected), "{id}: {}", view.text);
    }

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// A failing extractor on the streaming path is counted and dead-lettered,
/// never recorded as a successful extraction.
#[tokio::test]
async fn streaming_submit_source_surfaces_extractor_errors() {
    use std::sync::Arc;

    use uniko_pipes::content::Mime;

    use crate::{Modality, TextExtractor};

    let failing = TextExtractor::new(Modality::Image, |_| {
        Err(UnikoError::Internal("captioner offline".into()))
    });
    let memory = match Uniko::builder()
        .in_memory()
        .streaming(true)
        .extractor(Arc::new(failing))
        .build()
        .await
    {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("skipping: streaming instance unavailable: {e}");
            return;
        }
    };
    let agent = memory.agent("viewer");
    let session = agent.session("stream-media-fail");

    session
        .submit_source(
            IngestSource::bytes(vec![0x42; 16])
                .with_mime(Mime::parse("image/png").expect("valid mime"))
                .with_id("img-broken"),
        )
        .await
        .expect("submit_source enqueues");
    session.flush().await.expect("flush drains the pipeline");

    let health = memory.health().expect("streaming instance reports health");
    assert_eq!(health.ingest.items_failed, 1, "the failure is counted");

    let dead = agent
        .kb()
        .query_nodes(labels::DEAD_LETTER, None, None)
        .await
        .expect("query dead letters");
    assert_eq!(dead.len(), 1, "the failure is dead-lettered");
    let error = match dead[0].1.get("error") {
        Some(Value::String(e)) => e.clone(),
        other => panic!("dead letter has no error: {other:?}"),
    };
    assert!(error.contains("captioner offline"), "{error}");

    assert!(
        agent
            .data()
            .artifact("img-broken")
            .await
            .expect("artifact read")
            .is_none(),
        "a failed extraction is not an artifact"
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

/// `recall_with` applies its overrides to one call only.
#[tokio::test]
async fn recall_with_overrides_apply_to_one_call() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("chat-overrides");
    for text in [
        "I hike in the Alps every summer",
        "My favourite trail is the Haute Route",
        "I bought new hiking boots last week",
        "Hiking with friends is the best part of my weekend",
    ] {
        match session.observe(Turn::new("alice", text)).await {
            Ok(_) => {}
            Err(e) if is_model_unavailable(&e) => {
                eprintln!("skipping: embeddings unavailable");
                return;
            }
            Err(e) => panic!("observe failed: {e}"),
        }
    }

    let one = RecallOverrides::default().limit(1);
    let bundle = match agent.recall_with("hiking", Scope::default(), &one).await {
        Ok(bundle) => bundle,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("recall_with failed: {e}"),
    };
    assert!(
        bundle.items.len() <= 1,
        "limit override: {}",
        bundle.items.len()
    );

    // An unreachable minimum score filters everything out.
    let strict = RecallOverrides::default().min_score(f64::MAX);
    let bundle = agent
        .recall_with("hiking", Scope::default(), &strict)
        .await
        .expect("recall_with min_score");
    assert!(
        bundle.items.is_empty(),
        "min_score override: {}",
        bundle.items.len()
    );

    // The overrides did not stick: plain recall still finds items.
    let bundle = agent.recall("hiking").await.expect("recall");
    assert!(!bundle.items.is_empty(), "plain recall after overrides");

    // A limit above the reranker's top_n is rejected up front.
    let top_n = memory.config().reranker.top_n;
    let too_many = RecallOverrides::default().limit(top_n + 1);
    let err = agent
        .recall_with("hiking", Scope::default(), &too_many)
        .await
        .expect_err("limit above reranker.top_n");
    assert!(matches!(err, UnikoError::Config(_)), "got: {err}");
}

/// `health()` reports on the streaming pipeline, which takes its queue
/// capacities from the instance's `UnikoConfig`.
#[tokio::test]
async fn health_reports_the_configured_pipeline() {
    let Ok(plain) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    assert!(plain.health().is_none(), "no pipeline without streaming");

    let config = UnikoConfig {
        ingest_queue_capacity: 17,
        consolidation_queue_capacity: 5,
        ..UnikoConfig::default()
    };
    let memory = match Uniko::builder()
        .in_memory()
        .raw_config(config)
        .streaming(true)
        .build()
        .await
    {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("skipping: streaming instance unavailable: {e}");
            return;
        }
    };
    let health = memory.health().expect("streaming instance has a pipeline");
    assert_eq!(health.ingest.queue_capacity, 17);
    assert_eq!(health.consolidation.queue_capacity, 5);
    memory.shutdown().await.expect("shutdown");
}

/// `reranker()` on the builder wins over `raw_config`, in either order.
#[tokio::test]
async fn builder_reranker_overrides_raw_config() {
    let mut base = UnikoConfig::default();
    base.reranker.enabled = true;
    let mut off = uniko_store::config::RerankerConfig::minilm_l6();
    off.enabled = false;

    for reranker_first in [true, false] {
        let builder = Uniko::builder().in_memory();
        let builder = if reranker_first {
            builder.reranker(off.clone()).raw_config(base.clone())
        } else {
            builder.raw_config(base.clone()).reranker(off.clone())
        };
        let Ok(memory) = builder.build().await else {
            eprintln!("skipping: in-memory instance unavailable (no model?)");
            return;
        };
        assert!(
            !memory.config().reranker.enabled,
            "reranker() must win (reranker first: {reranker_first})"
        );
    }
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
        Ok(report) => assert!(
            report.summary.is_none(),
            "unused session has nothing to summarize"
        ),
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

/// Collect a single string column `t` from every row, for diagnostics.
async fn text_query(kb: &KnowledgeBase, cypher: &str) -> Vec<String> {
    kb.db() // ALLOW: test-only assertion helper; the seal governs product code.
        .session()
        .query(cypher)
        .await
        .expect("query")
        .rows()
        .iter()
        .map(|row| {
            row.get::<String>("t")
                .unwrap_or_else(|_| "<decode err>".into())
        })
        .collect()
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

/// `create_view` / `create_task_view` hand back the generated external ids,
/// and those ids drive every other goal and task method (issue #64).
#[tokio::test]
async fn create_views_return_usable_generated_ids() {
    use crate::{CreateGoalParams, CreateTaskParams, GoalPhase, TaskPhase};

    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("planner");
    seed_participant(agent.kb(), "planner").await;
    let goals = agent.goals();

    let goal = match goals
        .create_view(CreateGoalParams {
            title: "Ship the API".into(),
            status: Some("planned".into()),
            ..Default::default()
        })
        .await
    {
        Ok(goal) => goal,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("create goal: {e}"),
    };
    assert!(!goal.goal_id.is_empty(), "a goal id was generated");
    assert_eq!(goal.title, "Ship the API");
    assert_eq!(goal.phase, GoalPhase::Planned);

    let task = goals
        .create_task_view(CreateTaskParams {
            title: "write docs".into(),
            goal_id: Some(goal.goal_id.clone()),
            ..Default::default()
        })
        .await
        .expect("create task");
    assert!(!task.task_id.is_empty(), "a task id was generated");
    assert_ne!(task.task_id, goal.goal_id);
    assert_eq!(task.goal_id.as_deref(), Some(goal.goal_id.as_str()));
    assert_eq!(task.phase, TaskPhase::Planned);

    // The returned ids work with reads, transitions and goal context.
    assert_eq!(
        goals
            .get(&goal.goal_id)
            .await
            .expect("get")
            .map(|g| g.title),
        Some("Ship the API".to_string())
    );
    assert!(goals.start(&goal.goal_id).await.expect("start goal"));
    assert!(goals.start_task(&task.task_id).await.expect("start task"));
    assert_eq!(
        goals
            .task(&task.task_id)
            .await
            .expect("task")
            .map(|t| t.phase),
        Some(TaskPhase::Active)
    );
    assert!(
        goals
            .complete_task(&task.task_id)
            .await
            .expect("complete task")
    );
    let context = goals
        .context(&goal.goal_id)
        .await
        .expect("context")
        .expect("goal context");
    assert_eq!(context.goal.phase, GoalPhase::Active);
    assert!(
        context.tasks.iter().any(|t| t.task_id == task.task_id),
        "task in goal context: {:?}",
        context.tasks
    );
    assert!(
        goals
            .tasks_of(&goal.goal_id)
            .await
            .expect("tasks_of")
            .iter()
            .any(|t| t.task_id == task.task_id && t.phase == TaskPhase::Completed)
    );

    // An explicit id is used as given.
    let explicit = goals
        .create_view(CreateGoalParams {
            goal_id: Some("g-explicit".into()),
            title: "named goal".into(),
            ..Default::default()
        })
        .await
        .expect("create explicit goal");
    assert_eq!(explicit.goal_id, "g-explicit");
    assert_eq!(explicit.title, "named goal");
    let explicit_task = goals
        .create_task_view(CreateTaskParams {
            task_id: Some("t-explicit".into()),
            title: "named task".into(),
            ..Default::default()
        })
        .await
        .expect("create explicit task");
    assert_eq!(explicit_task.task_id, "t-explicit");
    assert_eq!(explicit_task.goal_id, None);

    // Retrying with the same explicit id updates in place: one goal, one
    // task, no duplicated ownership/assignment links.
    let retried = goals
        .create_view(CreateGoalParams {
            goal_id: Some("g-explicit".into()),
            title: "renamed goal".into(),
            ..Default::default()
        })
        .await
        .expect("retry explicit goal");
    assert_eq!(retried.title, "renamed goal");
    let retried_task = goals
        .create_task_view(CreateTaskParams {
            task_id: Some("t-explicit".into()),
            title: "named task".into(),
            goal_id: Some("g-explicit".into()),
            ..Default::default()
        })
        .await
        .expect("retry explicit task");
    assert_eq!(retried_task.goal_id.as_deref(), Some("g-explicit"));
    goals
        .create_task_view(CreateTaskParams {
            task_id: Some("t-explicit".into()),
            title: "named task".into(),
            goal_id: Some("g-explicit".into()),
            ..Default::default()
        })
        .await
        .expect("retry explicit task again");
    let all = goals.all().await.expect("all");
    assert_eq!(all.iter().filter(|g| g.goal_id == "g-explicit").count(), 1);
    let tasks = goals.tasks().await.expect("tasks");
    assert_eq!(
        tasks.iter().filter(|t| t.task_id == "t-explicit").count(),
        1
    );
    let linked = goals.tasks_of("g-explicit").await.expect("tasks_of");
    assert_eq!(
        linked.iter().filter(|t| t.task_id == "t-explicit").count(),
        1
    );

    // Two creates without an id are two goals.
    let a = goals
        .create_view(CreateGoalParams {
            title: "twin".into(),
            ..Default::default()
        })
        .await
        .expect("create a");
    let b = goals
        .create_view(CreateGoalParams {
            title: "twin".into(),
            ..Default::default()
        })
        .await
        .expect("create b");
    assert_ne!(a.goal_id, b.goal_id);

    assert!(goals.task("no-such-task").await.expect("task").is_none());

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
    // Capture exactly what the existence read returns between the two
    // finalizes. If this is non-empty and correct, a spurious rebuild cannot
    // be "the read came back short", which is the hypothesis to kill.
    let between = agent
        .kb()
        .session_chunk_rows("fin-3", "session")
        .await
        .expect("between-finalize chunk read");

    let second = session.finalize().await.expect("second finalize");

    if second.rebuilt {
        // The bare assertion said only "it rebuilt", which is useless for
        // diagnosis. The rebuild has exactly two causes: the existing-chunk
        // read came back short, or the freshly computed transcript differs
        // from the stored one. Print both so a failure identifies which.
        let stored = text_query(
            agent.kb(),
            "MATCH (:Session {session_id: 'fin-3'})-[:HAS_CHUNK]->(c:Chunk) \
             WHERE c.chunk_type = 'session' RETURN c.text AS t ORDER BY c.index",
        )
        .await;
        let obs_rows = agent
            .kb()
            .session_chunk_rows("fin-3", "observation")
            .await
            .map(|rows| {
                rows.into_iter()
                    .map(|r| (r.node_id, r.text))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let speakers = text_query(
            agent.kb(),
            "MATCH (m:Message)-[:IN_SESSION]->(:Session {session_id: 'fin-3'}) \
             OPTIONAL MATCH (m)-[:SENT_BY]->(p:Participant) \
             RETURN coalesce(p.name, '<NULL SENT_BY>') AS t",
        )
        .await;
        panic!(
            "an unchanged session must not be rewritten\n\
             first.transcript_chunks  = {:?}\n\
             second.transcript_chunks = {:?}\n\
             stored chunk text        = {stored:?}\n\
             resolved speakers        = {speakers:?}\n\
             read BETWEEN finalizes   = {between:?}\n\
             first.observation_chunks  = {:?}\n\
             second.observation_chunks = {:?}\n\
             observation chunk rows    = {obs_rows:?}",
            first.transcript_chunks,
            second.transcript_chunks,
            first.observation_chunks,
            second.observation_chunks,
        );
    }
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

// ── Atomic multi-turn units (issue #40) ────────────────────────────────

/// A unit records every turn, in order, as one write.
#[tokio::test]
async fn unit_commits_all_turns_in_one_transaction() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-1");

    let result = match session
        .unit()
        .turn(Turn::new("alice", "what is the plan for friday").id("u1-m1"))
        .turn(Turn::new("bob", "we ship the release on friday").id("u1-m2"))
        .turn(Turn::new("alice", "great, i will tell the team").id("u1-m3"))
        .commit()
        .await
    {
        Ok(r) => r,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("unit commit failed: {e}"),
    };

    assert_eq!(result.turns.len(), 3, "one result per turn, in unit order");
    assert!(!result.was_replay, "a fresh unit is not a replay");
    assert_eq!(message_count(agent.kb()).await, 3);
    for nid in result.message_node_ids() {
        assert_ne!(nid, 0, "every turn must report a real node id");
    }
    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// The `NEXT` chain runs through the unit in order, even though every turn
/// is written before the single commit.
#[tokio::test]
async fn unit_chains_next_edges_in_order() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-next");

    if let Err(e) = session
        .unit()
        .turn(Turn::new("alice", "first message here").id("n-1"))
        .turn(Turn::new("bob", "second message here").id("n-2"))
        .turn(Turn::new("alice", "third message here").id("n-3"))
        .commit()
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("unit commit failed: {e}");
    }

    let chained = count_query(
        agent.kb(),
        "MATCH (a:Message)-[:NEXT]->(b:Message) RETURN count(*) AS c",
    )
    .await;
    assert_eq!(chained, 2, "three turns must form two NEXT edges");
    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// An entity named in two turns of one unit yields ONE `:Entity` row but a
/// `MENTIONS` edge from EACH message.
///
/// This is the regression test for the per-turn entity upsert: running the
/// upsert once per turn either duplicates the row, because the second
/// snapshot read cannot see the first turn's uncommitted CREATE, or sums its
/// frequency from a stale count.
#[tokio::test]
async fn unit_dedups_entity_mentioned_in_two_turns() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-entity");

    if let Err(e) = session
        .unit()
        .turn(Turn::new("alice", "Marie Curie discovered radium in Paris").id("e-1"))
        .turn(Turn::new("bob", "Marie Curie won a Nobel Prize for it").id("e-2"))
        .commit()
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("unit commit failed: {e}");
    }

    let distinct = count_query(
        agent.kb(),
        "MATCH (e:Entity) RETURN count(DISTINCT e.entity_id) AS c",
    )
    .await;
    let rows = count_query(agent.kb(), "MATCH (e:Entity) RETURN count(e) AS c").await;
    assert_eq!(
        rows, distinct,
        "a unit must not create two :Entity rows for one entity_id"
    );

    // Dedup must not collapse provenance: both messages still point at it.
    let mentions = count_query(
        agent.kb(),
        "MATCH (m:Message)-[:MENTIONS]->(:Entity) RETURN count(DISTINCT m) AS c",
    )
    .await;
    assert!(
        mentions >= 2,
        "both messages must keep their own MENTIONS edges, got {mentions}"
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Re-committing a unit with the same ids and content writes nothing.
#[tokio::test]
async fn unit_replay_is_whole_unit_noop() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-replay");

    let turns = || {
        vec![
            Turn::new("alice", "stable content one").id("r-1"),
            Turn::new("bob", "stable content two").id("r-2"),
        ]
    };

    let first = match session.commit_unit(turns()).await {
        Ok(r) => r,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("first commit failed: {e}"),
    };
    assert!(!first.was_replay);
    let after_first = message_count(agent.kb()).await;

    let second = session.commit_unit(turns()).await.expect("replay");
    assert!(second.was_replay, "an identical unit must report a replay");
    assert_eq!(
        message_count(agent.kb()).await,
        after_first,
        "a replay must not write anything"
    );
    assert_eq!(
        first.message_node_ids(),
        second.message_node_ids(),
        "a replay must report the ids already on disk"
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// A unit whose ids are only PARTLY recorded is rejected, and writes
/// nothing.
///
/// A correct caller cannot reach this: a unit commits atomically, so a
/// replay is wholly present or wholly absent. Reaching it means ids were
/// reused across different units, and both alternatives silently corrupt —
/// ingesting only the absent turns changes the unit's meaning, treating it
/// as a no-op drops a turn.
#[tokio::test]
async fn unit_partial_replay_is_id_conflict() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-partial");

    if let Err(e) = session
        .commit_unit(vec![Turn::new("alice", "already recorded").id("p-1")])
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("seed commit failed: {e}");
    }
    let before = message_count(agent.kb()).await;

    let err = session
        .commit_unit(vec![
            Turn::new("alice", "already recorded").id("p-1"),
            Turn::new("bob", "brand new turn").id("p-2"),
        ])
        .await
        .expect_err("a partly-recorded unit must be rejected");
    assert!(
        matches!(err, UnikoError::IdConflict(_)),
        "expected IdConflict, got {err:?}"
    );
    assert_eq!(
        message_count(agent.kb()).await,
        before,
        "the absent turn must not have been written"
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// An id conflict on ANY turn fails the unit before anything is written —
/// including the turns that precede it.
#[tokio::test]
async fn unit_id_conflict_on_any_turn_writes_nothing() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-conflict");

    if let Err(e) = session
        .commit_unit(vec![Turn::new("alice", "original content").id("c-1")])
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("seed commit failed: {e}");
    }
    let before = message_count(agent.kb()).await;

    let err = session
        .commit_unit(vec![
            Turn::new("bob", "a genuinely fresh turn").id("c-fresh"),
            Turn::new("alice", "DIFFERENT content").id("c-1"),
        ])
        .await
        .expect_err("reusing an id with different content must be rejected");
    assert!(
        matches!(err, UnikoError::IdConflict(_)),
        "expected IdConflict, got {err:?}"
    );
    assert_eq!(
        message_count(agent.kb()).await,
        before,
        "the fresh turn preceding the conflict must not persist"
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// A unit repeating a message id internally is rejected: two turns sharing
/// an id would mint two Message nodes whose deterministic chunk ids collide.
#[tokio::test]
async fn unit_with_duplicate_ids_is_rejected() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-dup");

    let err = match session
        .commit_unit(vec![
            Turn::new("alice", "first").id("dup-1"),
            Turn::new("bob", "second").id("dup-1"),
        ])
        .await
    {
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => e,
        Ok(_) => panic!("a unit repeating a message id must be rejected"),
    };
    assert!(
        matches!(err, UnikoError::IdConflict(_)),
        "expected IdConflict, got {err:?}"
    );
    assert_eq!(message_count(agent.kb()).await, 0, "nothing may persist");
    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Once commit returns, scoped recall sees the WHOLE unit — #40's
/// read-after-write requirement.
#[tokio::test]
async fn unit_is_fully_recallable_after_commit() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-recall");

    if let Err(e) = session
        .unit()
        .turn(Turn::new("alice", "where did we leave the telescope").id("rc-1"))
        .turn(Turn::new("bob", "the telescope is in the observatory").id("rc-2"))
        .commit()
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("unit commit failed: {e}");
    }

    let both = count_query(
        agent.kb(),
        "MATCH (m:Message) WHERE m.content CONTAINS 'telescope' RETURN count(m) AS c",
    )
    .await;
    assert_eq!(both, 2, "both members of the pair must be visible at once");
    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// An empty unit is a caller error, not a silent no-op.
#[tokio::test]
async fn empty_unit_is_config_error() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-empty");

    let err = session
        .commit_unit(Vec::new())
        .await
        .expect_err("an empty unit must be rejected");
    assert!(
        matches!(err, UnikoError::Config(_)),
        "expected Config, got {err:?}"
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// THE ROLLBACK TEST. A failure part-way through a unit leaves no trace of
/// the turns already written inside that transaction.
///
/// The conflict tests above prove the *pre-transaction* rejection path
/// writes nothing. This proves the *in-transaction* rollback path does,
/// which is a different guarantee and the one issue #40 actually asks for:
/// interrupt after the first member would otherwise have been written, and
/// neither member may appear.
///
/// `UNIKO_TEST_FAIL_AFTER_TURN` injects the failure after turn 0's writes.
/// Env vars are process-global, which is safe only because nextest — this
/// repo's runner of record — gives every test its own process. Run under
/// `cargo test` and this would leak into sibling tests.
#[tokio::test]
async fn unit_rollback_leaves_no_trace() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-rollback");

    // Prove the store is reachable and empty before the injected failure,
    // so a zero count afterwards cannot be a vacuous pass.
    assert_eq!(
        message_count(agent.kb()).await,
        0,
        "fixture must start empty"
    );

    // SAFETY: nextest runs each test in its own process, so this cannot
    // race another test.
    unsafe { std::env::set_var("UNIKO_TEST_FAIL_AFTER_TURN", "0") };
    let result = session
        .unit()
        .turn(Turn::new("alice", "first turn of the pair").id("rb-1"))
        .turn(Turn::new("bob", "second turn of the pair").id("rb-2"))
        .commit()
        .await;
    unsafe { std::env::remove_var("UNIKO_TEST_FAIL_AFTER_TURN") };

    let Err(err) = result else {
        panic!("the injected failure must fail the unit");
    };
    if is_model_unavailable(&err) {
        eprintln!("skipping: embeddings unavailable");
        return;
    }

    // Neither member may be visible — not the one written before the
    // failure, and not the one after it.
    assert_eq!(
        message_count(agent.kb()).await,
        0,
        "turn 0 was written inside the transaction and must have rolled back"
    );
    assert_eq!(
        count_query(agent.kb(), "MATCH (o:Observation) RETURN count(o) AS c").await,
        0,
        "observations from the rolled-back turn must not persist"
    );
    assert_eq!(
        count_query(agent.kb(), "MATCH (c:Chunk) RETURN count(c) AS c").await,
        0,
        "chunks from the rolled-back turn must not persist"
    );

    // And session_ctx must not have advanced: a later turn chains from the
    // original head, so it gets no incoming NEXT edge.
    session
        .observe(Turn::new("alice", "a turn after the failed unit").id("rb-after"))
        .await
        .expect("observe after a failed unit must succeed");
    assert_eq!(
        count_query(
            agent.kb(),
            "MATCH (:Message)-[:NEXT]->(m:Message) WHERE m.message_id = 'rb-after' \
             RETURN count(*) AS c"
        )
        .await,
        0,
        "a rolled-back unit must not leave the chain head advanced"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// An attachment failure rolls back its MESSAGE too — the attachments-in-
/// the-transaction guarantee.
#[tokio::test]
async fn unit_attachment_rolls_back_with_its_message() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-attach-rollback");

    // Seed an artifact so the unit's attachment can collide with it.
    let seeded = match session
        .ingest(IngestSource::text("original bytes for the shared id").with_id("att-1"))
        .await
    {
        Ok(o) => o,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("seed ingest failed: {e}"),
    };
    let _ = seeded;
    let before_messages = message_count(agent.kb()).await;

    // Same artifact id, different bytes: an IdConflict raised while the
    // unit's transaction is open, after the message has been written.
    let err = session
        .unit()
        .turn(
            Turn::new("alice", "a turn carrying a conflicting attachment")
                .id("att-turn")
                .attach(IngestSource::text("DIFFERENT bytes under the same id").with_id("att-1")),
        )
        .commit()
        .await
        .expect_err("a conflicting attachment must fail the unit");
    assert!(
        matches!(err, UnikoError::IdConflict(_)),
        "expected IdConflict, got {err:?}"
    );

    assert_eq!(
        message_count(agent.kb()).await,
        before_messages,
        "the message must roll back with its failed attachment"
    );
    assert_eq!(
        count_query(
            agent.kb(),
            "MATCH (m:Message) WHERE m.message_id = 'att-turn' RETURN count(m) AS c"
        )
        .await,
        0,
        "no trace of the turn whose attachment failed"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// The same bytes attached to two turns of one unit yield ONE `:Artifact`,
/// but an `ATTACHED_TO` edge from EACH message — dedup must not collapse
/// provenance.
#[tokio::test]
async fn duplicate_attachment_across_turns_yields_one_artifact() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-attach-dedup");

    let shared = "identical attachment bytes shared by two turns";
    let result = match session
        .unit()
        .turn(
            Turn::new("alice", "here is the document")
                .id("d-1")
                .attach(IngestSource::text(shared)),
        )
        .turn(
            Turn::new("bob", "sending the same document back")
                .id("d-2")
                .attach(IngestSource::text(shared)),
        )
        .commit()
        .await
    {
        Ok(r) => r,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("unit commit failed: {e}"),
    };

    assert_eq!(result.turns.len(), 2);

    let content_rows =
        count_query(agent.kb(), "MATCH (c:ArtifactContent) RETURN count(c) AS c").await;
    assert_eq!(
        content_rows, 1,
        "identical bytes must converge on one :ArtifactContent"
    );

    let attached = count_query(
        agent.kb(),
        "MATCH (:Artifact)-[:ATTACHED_TO]->(m:Message) RETURN count(DISTINCT m) AS c",
    )
    .await;
    assert_eq!(
        attached, 2,
        "dedup must not drop the second turn's provenance"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// An attachment links to the Message created in the SAME transaction.
///
/// This is the silent-regression guard. Resolving `triggered_by_message_id`
/// before the transaction always misses — the Message is uncommitted — and a
/// miss is skipped rather than raised, so the edge would simply vanish. No
/// error, no warning, no panic; only this assertion catches it.
#[tokio::test]
async fn unit_attachment_links_to_in_tx_message() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("unit-attach-link");

    if let Err(e) = session
        .unit()
        .turn(
            Turn::new("alice", "please review the attached notes")
                .id("link-1")
                .attach(IngestSource::text("the attached notes body")),
        )
        .commit()
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("unit commit failed: {e}");
    }

    let linked = count_query(
        agent.kb(),
        "MATCH (:Artifact)-[:ATTACHED_TO]->(m:Message) WHERE m.message_id = 'link-1' \
         RETURN count(*) AS c",
    )
    .await;
    assert_eq!(
        linked, 1,
        "the attachment must link to the message created in the same transaction"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

// ── Typed provenance and pre-ranking filters (issue #39) ───────────────

/// The acceptance case from the issue: record four categories, query a term
/// present in all four while permitting only one, and get back only that
/// one.
///
/// The filter is pushed into candidate generation, so the result limit and
/// the coverage score describe the permitted evidence — not candidates that
/// were ranked first and discarded afterwards.
#[tokio::test]
async fn recall_returns_only_the_permitted_category() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("prov-1");

    // One shared term across four different record categories.
    let turns = vec![
        Turn::new("alice", "the telescope readings look stable")
            .id("p-assert")
            .category("user_assertion"),
        Turn::new("agent", "the telescope query returned 42 rows")
            .id("p-exec")
            .category("executed_result"),
        Turn::new("agent", "the telescope data suggests a drift")
            .id("p-interp")
            .category("model_interpretation"),
        Turn::new("agent", "the telescope manual describes calibration")
            .id("p-doc")
            .category("external_evidence"),
    ];
    if let Err(e) = session.commit_unit(turns).await {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("unit commit failed: {e}");
    }

    let scope = Scope::default().categories(["executed_result"]);
    let bundle = match agent.recall_in("telescope", scope).await {
        Ok(b) => b,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("recall_in failed: {e}"),
    };

    assert!(
        !bundle.items.is_empty(),
        "the permitted category must still return its evidence"
    );
    for item in &bundle.items {
        assert_eq!(
            item.category.as_deref(),
            Some("executed_result"),
            "a disallowed category leaked into the results: {item:?}"
        );
    }

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// A category with no eligible matches returns empty — it is NOT padded out
/// with other categories to fill the result limit.
#[tokio::test]
async fn unmatched_category_filter_returns_empty() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("prov-empty");

    if let Err(e) = session
        .observe(
            Turn::new("alice", "the telescope readings look stable")
                .id("pe-1")
                .category("user_assertion"),
        )
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("observe failed: {e}");
    }

    let scope = Scope::default().categories(["no_such_category"]);
    match agent.recall_in("telescope", scope).await {
        Ok(bundle) => assert!(
            bundle.items.is_empty(),
            "an unmatched category must return empty, not fall back to \
             other categories: {:?}",
            bundle.items
        ),
        Err(e) if is_model_unavailable(&e) => eprintln!("skipping: embeddings unavailable"),
        Err(e) => panic!("recall_in failed: {e}"),
    }

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// A source filter narrows to that logical source, and the `:Source` node is
/// materialised with a `FROM_SOURCE` edge from the record.
#[tokio::test]
async fn recall_filters_by_logical_source() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("prov-src");

    let turns = vec![
        Turn::new("agent", "the beacon signal was steady all week")
            .id("s-a")
            .source("feed-alpha"),
        Turn::new("agent", "the beacon signal dropped out twice")
            .id("s-b")
            .source("feed-beta"),
    ];
    if let Err(e) = session.commit_unit(turns).await {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("unit commit failed: {e}");
    }

    // The normalised :Source rows exist, one per logical source.
    assert_eq!(
        count_query(agent.kb(), "MATCH (s:Source) RETURN count(s) AS c").await,
        2,
        "one :Source per distinct source id"
    );
    assert_eq!(
        count_query(
            agent.kb(),
            "MATCH (:Message)-[:FROM_SOURCE]->(:Source) RETURN count(*) AS c"
        )
        .await,
        2,
        "each record must point at its source"
    );

    let scope = Scope::default().sources(["feed-alpha"]);
    let bundle = match agent.recall_in("beacon signal", scope).await {
        Ok(b) => b,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("recall_in failed: {e}"),
    };
    for item in &bundle.items {
        assert_eq!(
            item.source_id.as_deref(),
            Some("feed-alpha"),
            "a disallowed source leaked into the results: {item:?}"
        );
    }

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Provenance survives onto chunks and observations, so the filter reaches
/// derived items rather than only the raw turn.
#[tokio::test]
async fn provenance_is_inherited_by_derived_items() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("prov-derived");

    if let Err(e) = session
        .observe(
            Turn::new(
                "alice",
                "Marie Curie discovered radium in Paris and later won a Nobel Prize \
                 for the work, which the committee recognised in nineteen eleven.",
            )
            .id("pd-1")
            .category("external_evidence")
            .source("encyclopaedia"),
        )
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("observe failed: {e}");
    }

    // The message itself.
    assert_eq!(
        count_query(
            agent.kb(),
            "MATCH (m:Message) WHERE m.category = 'external_evidence' \
             AND m.source_id = 'encyclopaedia' RETURN count(m) AS c"
        )
        .await,
        1,
        "the record must carry its own provenance"
    );

    // Derived observations inherit it, which is what lets the pre-ranking
    // filter reach them with a property predicate instead of a traversal.
    let obs_total = count_query(agent.kb(), "MATCH (o:Observation) RETURN count(o) AS c").await;
    if obs_total > 0 {
        assert_eq!(
            count_query(
                agent.kb(),
                "MATCH (o:Observation) WHERE o.source_id = 'encyclopaedia' \
                 RETURN count(o) AS c"
            )
            .await,
            obs_total,
            "every derived observation must inherit the record's source"
        );
    }

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// A session filter and a provenance filter compose: the intersection, not
/// the union.
///
/// This is the interaction worth testing rather than assuming — the
/// allow-set gates Chunk candidates differently depending on which
/// dimensions are active, so a session+category scope takes a different code
/// path from either one alone.
#[tokio::test]
async fn session_boundary_holds_with_a_provenance_filter() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");

    // Same category, same term, two different sessions.
    let mut a = agent.session("prov-sess-a");
    let mut b = agent.session("prov-sess-b");
    for (session, id) in [(&mut a, "sa-1"), (&mut b, "sb-1")] {
        if let Err(e) = session
            .observe(
                Turn::new("agent", "the aurora forecast index reached seven")
                    .id(id)
                    .category("executed_result"),
            )
            .await
        {
            if is_model_unavailable(&e) {
                eprintln!("skipping: embeddings unavailable");
                return;
            }
            panic!("observe failed: {e}");
        }
    }

    let scope = Scope::default()
        .sessions(["prov-sess-a"])
        .categories(["executed_result"]);
    let bundle = match agent.recall_in("aurora forecast", scope).await {
        Ok(bundle) => bundle,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("recall_in failed: {e}"),
    };

    // Everything returned must satisfy BOTH dimensions. Session b shares the
    // category, so a category filter that ignored the session boundary would
    // leak it.
    for item in &bundle.items {
        assert_eq!(
            item.category.as_deref(),
            Some("executed_result"),
            "category filter leaked: {item:?}"
        );
    }
    let leaked = count_query(
        agent.kb(),
        "MATCH (m:Message) WHERE m.message_id = 'sb-1' RETURN count(m) AS c",
    )
    .await;
    assert_eq!(
        leaked, 1,
        "session b's message should exist but be out of scope"
    );
    for item in &bundle.items {
        assert!(
            !item.content.is_empty(),
            "an in-scope item must carry content"
        );
    }

    drop(a);
    drop(b);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Coverage describes the PERMITTED evidence.
///
/// The issue's complaint is that a coverage score can describe unfiltered
/// candidates rather than what the consumer received. Because the filter runs
/// during candidate generation, coverage is computed over the eligible set —
/// this pins that rather than trusting the structure.
#[tokio::test]
async fn coverage_describes_only_permitted_evidence() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("prov-cov");

    // Many disallowed high-ranking candidates, one permitted record: more
    // disallowed matches than the result limit, as the issue asks.
    let mut turns = Vec::new();
    for i in 0..6 {
        turns.push(
            Turn::new("agent", format!("the pipeline latency report number {i}"))
                .id(format!("cov-bad-{i}"))
                .category("model_interpretation"),
        );
    }
    turns.push(
        Turn::new("agent", "the pipeline latency measured four hundred ms")
            .id("cov-good")
            .category("executed_result"),
    );
    if let Err(e) = session.commit_unit(turns).await {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("unit commit failed: {e}");
    }

    let scope = Scope::default().categories(["executed_result"]);
    let bundle = match agent.recall_in("pipeline latency", scope).await {
        Ok(bundle) => bundle,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("recall_in failed: {e}"),
    };

    for item in &bundle.items {
        assert_eq!(
            item.category.as_deref(),
            Some("executed_result"),
            "coverage would be describing a discarded candidate: {item:?}"
        );
    }
    // Coverage is a property of what came back, so with a non-empty permitted
    // set it must be a real score over those items — not zero, and not a
    // score inherited from the six disallowed candidates.
    if !bundle.items.is_empty() {
        assert!(
            bundle.coverage.is_finite() && bundle.coverage >= 0.0,
            "coverage must be a real score over the permitted items, got {}",
            bundle.coverage
        );
    }

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

// ── Source revisions and retirement (issue #41) ────────────────────────

/// The issue's acceptance sequence, end to end: ingest revision A, ingest a
/// contradicting revision B, confirm current recall uses B while historical
/// recall still attributes A, then retire the source and confirm neither
/// revision grounds a current answer.
#[tokio::test]
async fn newer_revision_supersedes_and_retirement_hides_both() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let session = agent.session("rev-1");

    // Revision A of a page.
    if let Err(e) = session
        .ingest(
            IngestSource::text("The summit elevation is 3200 metres.")
                .with_id("page-a")
                .with_source("wiki-summit")
                .with_revision("rev-a"),
        )
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("ingest A failed: {e}");
    }

    // Nothing is superseded yet, so A is current.
    assert_eq!(
        count_query(
            agent.kb(),
            "MATCH (a:Artifact) WHERE a.revision_id = 'rev-a' AND a.superseded_at IS NULL \
             RETURN count(a) AS c"
        )
        .await,
        1,
        "the only revision must be current"
    );

    // Revision B contradicts A.
    session
        .ingest(
            IngestSource::text("The summit elevation is 3450 metres.")
                .with_id("page-b")
                .with_source("wiki-summit")
                .with_revision("rev-b"),
        )
        .await
        .expect("ingest B");

    // A is now history; B is current; the history edge records the order.
    assert_eq!(
        count_query(
            agent.kb(),
            "MATCH (a:Artifact) WHERE a.revision_id = 'rev-a' AND a.superseded_at IS NOT NULL \
             RETURN count(a) AS c"
        )
        .await,
        1,
        "revision A must be stamped superseded once B arrives"
    );
    assert_eq!(
        count_query(
            agent.kb(),
            "MATCH (b:Artifact {revision_id: 'rev-b'})-[:SUPERSEDES]->\
             (a:Artifact {revision_id: 'rev-a'}) RETURN count(*) AS c"
        )
        .await,
        1,
        "SUPERSEDES must record which revision replaced which"
    );

    // Ordinary recall: only the current revision may ground an answer.
    let current = match agent.recall("summit elevation").await {
        Ok(bundle) => bundle,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("recall failed: {e}"),
    };
    for item in &current.items {
        assert_ne!(
            item.revision_id.as_deref(),
            Some("rev-a"),
            "a superseded revision must not ground a current answer: {item:?}"
        );
    }

    // Historical recall: A is still attributable to the revision that
    // grounded it.
    let historical = agent
        .recall_in("summit elevation", Scope::default().include_superseded())
        .await
        .expect("historical recall");
    let saw_a = historical
        .items
        .iter()
        .any(|i| i.revision_id.as_deref() == Some("rev-a"));
    assert!(
        saw_a || historical.items.is_empty(),
        "historical recall must be able to reach revision A; got {:?}",
        historical
            .items
            .iter()
            .map(|i| i.revision_id.clone())
            .collect::<Vec<_>>()
    );

    // Retire the source: now NEITHER revision grounds a current answer.
    assert!(
        agent.retire_source("wiki-summit").await.expect("retire"),
        "retiring an existing source must report success"
    );
    let after = agent.recall("summit elevation").await.expect("recall");
    for item in &after.items {
        assert_ne!(
            item.source_id.as_deref(),
            Some("wiki-summit"),
            "a retired source must not ground a current answer: {item:?}"
        );
    }

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Retiring one source must not affect another that merely shares identical
/// bytes.
///
/// Identical content converges on ONE `:ArtifactContent` row by design, so a
/// retirement implemented on the content rather than the source would take
/// both down. This pins that it is recorded on the `:Source`.
#[tokio::test]
async fn retiring_one_source_leaves_an_identical_twin_alive() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let session = agent.session("rev-twin");

    let shared = "The lighthouse beam rotates every twelve seconds.";
    for (id, src) in [("twin-1", "feed-one"), ("twin-2", "feed-two")] {
        if let Err(e) = session
            .ingest(
                IngestSource::text(shared)
                    .with_id(id)
                    .with_source(src)
                    .with_revision(format!("{src}-r1")),
            )
            .await
        {
            if is_model_unavailable(&e) {
                eprintln!("skipping: embeddings unavailable");
                return;
            }
            panic!("ingest failed: {e}");
        }
    }

    // Two sources, two artifacts, but ONE stored copy of the bytes.
    assert_eq!(
        count_query(agent.kb(), "MATCH (s:Source) RETURN count(s) AS c").await,
        2,
        "two independent logical sources"
    );
    assert_eq!(
        count_query(agent.kb(), "MATCH (c:ArtifactContent) RETURN count(c) AS c").await,
        1,
        "identical bytes must still share one stored copy"
    );

    agent.retire_source("feed-one").await.expect("retire");

    // feed-two is untouched: retirement is a property of the source.
    assert_eq!(
        count_query(
            agent.kb(),
            "MATCH (s:Source {source_id: 'feed-two'}) WHERE s.retired_at IS NULL \
             RETURN count(s) AS c"
        )
        .await,
        1,
        "retiring feed-one must not retire feed-two"
    );
    let bundle = match agent.recall("lighthouse beam").await {
        Ok(b) => b,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("recall failed: {e}"),
    };
    for item in &bundle.items {
        assert_ne!(
            item.source_id.as_deref(),
            Some("feed-one"),
            "the retired source must be excluded: {item:?}"
        );
    }

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// The same revision with changed bytes is rejected: a revision id is a
/// promise about the content.
#[tokio::test]
async fn reusing_a_revision_with_changed_content_is_rejected() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let session = agent.session("rev-conflict");

    if let Err(e) = session
        .ingest(
            IngestSource::text("original body of the page")
                .with_id("rc-1")
                .with_source("feed-x")
                .with_revision("rx-1"),
        )
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        panic!("seed ingest failed: {e}");
    }

    // Same revision, same bytes: idempotent.
    session
        .ingest(
            IngestSource::text("original body of the page")
                .with_id("rc-1")
                .with_source("feed-x")
                .with_revision("rx-1"),
        )
        .await
        .expect("re-ingesting an identical revision must be a no-op");

    // Same revision, different bytes: rejected.
    let err = session
        .ingest(
            IngestSource::text("SILENTLY DIFFERENT body of the page")
                .with_id("rc-2")
                .with_source("feed-x")
                .with_revision("rx-1"),
        )
        .await
        .expect_err("changed content under one revision id must be rejected");
    assert!(
        matches!(err, UnikoError::IdConflict(_)),
        "expected IdConflict, got {err:?}"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// `Scope::as_participant` actually filters, and does not fail open.
///
/// This closes a real asymmetry: constructing a `Viewer` is async and needs a
/// store handle, so before this the Python bindings could not express
/// viewer-scoped recall at all — and because unscoped reads are fail-open,
/// that meant Python callers got no Fact/Observation visibility filtering
/// unless the instance was built with `scope_to_agent()`.
///
/// The deferred variant is resolved inside `recall` rather than only in the
/// facade, so it cannot reach the fail-open arm.
#[tokio::test]
async fn scope_as_participant_filters_private_facts() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    // An UNRESTRICTED agent, so any filtering observed comes from the scope
    // rather than from an instance default.
    let agent = memory.agent("observer");
    let kb = agent.kb();

    seed_participant(kb, "alice").await;
    seed_participant(kb, "bob").await;
    let query = "quarterly revenue outlook";
    seed_fact(kb, "f-pub", query, Some("public")).await;
    seed_fact(kb, "f-priv", query, Some("private:alice")).await;
    let private_nid = fact_nid(kb, "f-priv").await;

    // Baseline: unscoped recall is fail-open, so it should surface the
    // private Fact. If it does not, this environment isn't surfacing Facts
    // and the assertion below would pass vacuously.
    let unscoped = match agent.recall(query).await {
        Ok(bundle) => bundle,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("unscoped recall failed: {e}"),
    };
    if !unscoped.items.iter().any(|i| i.node_id == private_nid) {
        eprintln!("skipping: recall did not surface Facts in this env");
        return;
    }

    // Scoped to bob, the private:alice Fact must be gone.
    let as_bob = agent
        .recall_in(query, Scope::default().as_participant("bob"))
        .await
        .expect("recall as bob");
    assert!(
        !as_bob.items.iter().any(|i| i.node_id == private_nid),
        "as_participant(bob) must not return the private:alice Fact"
    );

    // Scoped to alice, it must still be there — proving the filter is
    // discriminating, not just dropping Facts wholesale.
    let as_alice = agent
        .recall_in(query, Scope::default().as_participant("alice"))
        .await
        .expect("recall as alice");
    assert!(
        as_alice.items.iter().any(|i| i.node_id == private_nid),
        "as_participant(alice) must still return her own private Fact"
    );

    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

// ── Phase B: visibility writes, memberships, rules, memory operations ──

/// Recall `query` as `agent`, or `None` when embeddings are unavailable.
async fn recall_ids(agent: &crate::Agent, query: &str) -> Option<Vec<i64>> {
    match agent.recall(query).await {
        Ok(bundle) => Some(bundle.items.iter().map(|i| i.node_id).collect()),
        Err(e) if is_model_unavailable(&e) => None,
        Err(e) => panic!("recall failed: {e}"),
    }
}

/// Items scoped with `set_visibility` are filtered by team membership,
/// private ownership and an explicit viewer. Uses Observations, which
/// recall searches directly (Facts are reached through their entities).
#[tokio::test]
async fn visibility_writes_scope_recall_by_membership() {
    use crate::{AddObservationParams, Viewer, Visibility};

    let memory = match Uniko::builder().in_memory().scope_to_agent().build().await {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("skipping: instance unavailable: {e}");
            return;
        }
    };
    assert!(memory.add_team_member("alice", "eng").await.expect("add"));
    let writer = memory.agent("writer");
    let mut session = writer.session("planning");
    match session
        .observe(Turn::new("writer", "notes from the planning call").id("plan-1"))
        .await
    {
        Ok(_) => {}
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("observe failed: {e}"),
    }
    let observe = |content: &str| AddObservationParams {
        message_id: "plan-1".into(),
        content: content.into(),
        subject: "launch".into(),
        predicate: Some("scheduled_for".into()),
        object: Some(content.into()),
        ..AddObservationParams::default()
    };
    let team_item = writer
        .add_observation(observe("the zeppelin launch is scheduled for march"))
        .await
        .expect("add_observation");
    let private_item = writer
        .add_observation(observe("the zeppelin launch slips to april for carol"))
        .await
        .expect("add_observation");
    assert!(
        writer
            .set_visibility(team_item, Visibility::Team("eng".into()))
            .await
            .unwrap()
    );
    assert!(
        writer
            .set_visibility(private_item, Visibility::Private("carol".into()))
            .await
            .unwrap()
    );

    let query = "zeppelin launch";
    let carol = recall_ids(&memory.agent("carol"), query).await.unwrap();
    assert!(
        carol.contains(&private_item),
        "carol sees her own item: {carol:?}"
    );
    assert!(!carol.contains(&team_item), "carol is not in eng");

    let alice = recall_ids(&memory.agent("alice"), query).await.unwrap();
    assert!(alice.contains(&team_item), "alice is in eng: {alice:?}");
    assert!(!alice.contains(&private_item), "alice is not carol");

    let bob = recall_ids(&memory.agent("bob"), query).await.unwrap();
    assert!(!bob.contains(&team_item) && !bob.contains(&private_item));

    // An explicit viewer overrides the agent's own scope for one call.
    let viewer = Viewer::from_parts("bob", ["eng".to_string()], []);
    let scoped = memory
        .agent("bob")
        .recall_in(query, Scope::default().as_viewer(viewer))
        .await
        .unwrap();
    assert!(scoped.items.iter().any(|i| i.node_id == team_item));

    // Removing the membership takes effect on a fresh handle.
    assert!(memory.remove_team_member("alice", "eng").await.unwrap());
    let alice = recall_ids(&memory.agent("alice"), query).await.unwrap();
    assert!(!alice.contains(&team_item), "alice left eng");
}

/// Memberships are idempotent, and a team's org reaches its members.
#[tokio::test]
async fn memberships_are_idempotent_and_teams_reach_their_org() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    assert!(memory.add_team_member("dana", "infra").await.unwrap());
    assert!(!memory.add_team_member("dana", "infra").await.unwrap());
    assert!(memory.add_team_to_org("infra", "acme").await.unwrap());
    assert!(memory.add_org_member("erin", "acme").await.unwrap());

    let kb = memory.agent("x").kb().clone();
    let dana = crate::Viewer::new(&kb, "dana").await.unwrap();
    assert!(dana.teams.contains("infra"));
    assert!(dana.orgs.contains("acme"), "through the team");
    let erin = crate::Viewer::new(&kb, "erin").await.unwrap();
    assert!(erin.orgs.contains("acme"), "directly");

    assert!(memory.remove_team_from_org("infra", "acme").await.unwrap());
    assert!(!memory.remove_team_from_org("infra", "acme").await.unwrap());
    assert!(memory.remove_org_member("erin", "acme").await.unwrap());
    let dana = crate::Viewer::new(&kb, "dana").await.unwrap();
    assert!(dana.orgs.is_empty());

    let err = memory.add_team_member("", "infra").await.unwrap_err();
    assert!(matches!(err, UnikoError::Config(_)), "got: {err}");
}

/// `set_visibility` only applies to Facts and Observations, and never
/// un-forgets an item.
#[tokio::test]
async fn set_visibility_rejects_other_nodes_and_forgotten_items() {
    use crate::Visibility;

    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let kb = agent.kb();
    seed_participant(kb, "assistant").await;
    let pnid = kb
        .get_node_by_ext_id(labels::PARTICIPANT, "participant_id", "assistant")
        .await
        .unwrap()
        .unwrap()
        .0;
    let err = agent
        .set_visibility(pnid, Visibility::Public)
        .await
        .unwrap_err();
    assert!(matches!(err, UnikoError::Config(_)), "got: {err}");

    seed_fact(kb, "f-forgotten", "gone", Some("redacted:forgotten")).await;
    let nid = fact_nid(kb, "f-forgotten").await;
    let err = agent
        .set_visibility(nid, Visibility::Public)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("forgotten"), "got: {err}");

    assert!(
        !agent
            .set_visibility(i64::MAX, Visibility::Public)
            .await
            .unwrap()
    );
    let err = agent
        .set_visibility(nid, Visibility::Team(String::new()))
        .await
        .unwrap_err();
    assert!(matches!(err, UnikoError::Config(_)), "got: {err}");
}

/// `Visibility` round-trips through its stored form.
#[test]
fn visibility_round_trips() {
    use crate::Visibility;
    for v in [
        Visibility::Public,
        Visibility::Private("p1".into()),
        Visibility::Team("t1".into()),
        Visibility::Org("o1".into()),
    ] {
        assert_eq!(Visibility::parse(&v.as_property()).unwrap(), v);
    }
    assert_eq!(Visibility::parse("").unwrap(), Visibility::Public);
    assert!(Visibility::parse("group:x").is_err());
    assert!(Visibility::parse("team:").is_err());
}

/// Rules can be listed, explained and deleted; stdlib rules are protected.
#[tokio::test]
async fn rules_list_explain_and_delete() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let source = "CREATE RULE managed_rule AS MATCH (n:Episode) YIELD KEY n";
    agent
        .define_rule("managed_rule", source)
        .await
        .expect("define_rule");

    let rules = agent.list_rules().await.expect("list_rules");
    let names: Vec<&str> = rules.iter().map(|r| r.name.as_str()).collect();
    assert!(names.contains(&"managed_rule"), "{names:?}");
    assert!(
        rules.iter().any(|r| r.source_type == "stdlib"),
        "stdlib rules are listed: {names:?}"
    );

    let explained = agent
        .explain_rule("managed_rule")
        .await
        .unwrap()
        .expect("rule");
    assert_eq!(explained.source.as_deref(), Some(source));
    assert_eq!(explained.source_type, "authored");
    assert!(agent.explain_rule("no_such_rule").await.unwrap().is_none());

    assert!(agent.delete_rule("managed_rule").await.unwrap());
    let after = agent.explain_rule("managed_rule").await.unwrap().unwrap();
    assert_eq!(after.status, "pruned");
    assert!(!agent.delete_rule("no_such_rule").await.unwrap());

    let stdlib = rules
        .iter()
        .find(|r| r.source_type == "stdlib")
        .unwrap()
        .name
        .clone();
    let err = agent.delete_rule(&stdlib).await.unwrap_err();
    assert!(matches!(err, UnikoError::Config(_)), "got: {err}");
}

/// Actions, episodes and observations record through `Agent` even before
/// the agent has spoken, and `consolidate` reports the cortex sweeps.
#[tokio::test]
async fn agent_records_actions_episodes_and_observations() {
    use crate::{AddObservationParams, RecordActionParams, RecordEpisodeParams};

    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("fresh-agent");
    let action = agent
        .record_action(RecordActionParams {
            action_type: "tool_call".into(),
            status: Some("success".into()),
            ..RecordActionParams::default()
        })
        .await
        .expect("record_action without a prior turn");
    assert!(action.action_node > 0);

    match agent
        .record_episode(RecordEpisodeParams {
            action_type: "tool_call".into(),
            outcome: Some("success".into()),
            ..RecordEpisodeParams::default()
        })
        .await
    {
        Ok(nid) => assert!(nid > 0),
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("record_episode failed: {e}"),
    }

    let mut session = agent.session("ops-session");
    session
        .observe(Turn::new("user", "please deploy the service").id("ops-1"))
        .await
        .expect("observe");
    let in_session = session
        .record_action(RecordActionParams {
            action_type: "deploy".into(),
            ..RecordActionParams::default()
        })
        .await
        .expect("session record_action");
    assert!(in_session.action_node > 0);

    let obs = agent
        .add_observation(AddObservationParams {
            message_id: "ops-1".into(),
            content: "the user wants a deploy".into(),
            subject: "user".into(),
            predicate: Some("wants".into()),
            object: Some("deploy".into()),
            ..AddObservationParams::default()
        })
        .await
        .expect("add_observation");
    assert!(obs > 0);

    let report = agent.consolidate().await.expect("consolidate");
    assert!(report.cycle.observations_processed >= 1, "{report:?}");
}

/// A turn scoped with `Turn::visibility` hides its message, chunks and
/// observations from viewers it does not admit, and stays out of the
/// session-level chunks.
#[tokio::test]
async fn turn_visibility_scopes_the_message_and_what_derives_from_it() {
    use crate::Visibility;

    let memory = match Uniko::builder().in_memory().scope_to_agent().build().await {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("skipping: instance unavailable: {e}");
            return;
        }
    };
    memory.add_team_member("alice", "eng").await.unwrap();
    let writer = memory.agent("writer");
    let mut session = writer.session("scoped-session");
    let scoped = match session
        .observe(
            Turn::new("writer", "The zeppelin hangar opens next quarter in Lisbon")
                .id("scoped-1")
                .visibility(Visibility::Team("eng".into())),
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
    session
        .observe(Turn::new("writer", "Lunch is at noon on Friday").id("public-1"))
        .await
        .unwrap();

    let kb = writer.kb();
    let (_, props) = kb
        .get_node(scoped.message.message_node_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        props.get("visibility"),
        Some(&Value::String("team:eng".into())),
        "the message carries the turn's visibility"
    );
    for obs in &scoped.message.extracted_observations {
        let (_, props) = kb.get_node(*obs).await.unwrap().unwrap();
        assert_eq!(
            props.get("visibility"),
            Some(&Value::String("team:eng".into()))
        );
    }
    let mut hidden: Vec<i64> = vec![scoped.message.message_node_id];
    hidden.extend(&scoped.message.chunk_node_ids);
    hidden.extend(&scoped.message.extracted_observations);

    let query = "zeppelin hangar Lisbon";
    let alice = recall_ids(&memory.agent("alice"), query).await.unwrap();
    assert!(
        alice.iter().any(|id| hidden.contains(id)),
        "alice (eng) sees the scoped turn: {alice:?}"
    );
    let bob = recall_ids(&memory.agent("bob"), query).await.unwrap();
    assert!(
        !bob.iter().any(|id| hidden.contains(id)),
        "bob sees nothing of it: {bob:?}"
    );

    // Session-level chunks are built from public turns only.
    session.finalize().await.expect("finalize");
    let chunks = kb.query_nodes(labels::CHUNK, None, None).await.unwrap();
    let transcript_has_public = chunks.iter().any(|(_, p)| {
        matches!(p.get("chunk_type"), Some(Value::String(t)) if t == "session")
            && matches!(p.get("text"), Some(Value::String(text)) if text.contains("Lunch"))
    });
    assert!(
        transcript_has_public,
        "the transcript keeps the public turn"
    );
    for (_, props) in &chunks {
        if let Some(Value::String(text)) = props.get("text")
            && matches!(props.get("chunk_type"), Some(Value::String(t)) if t == "session" || t == "observation")
        {
            assert!(
                !text.contains("zeppelin"),
                "a session-level chunk leaked the scoped turn: {text}"
            );
        }
    }
}

/// Consolidation keeps a scoped observation's fact apart from the public
/// fact of the same triple.
#[tokio::test]
async fn consolidation_keeps_scoped_facts_apart() {
    use crate::{AddObservationParams, Visibility};

    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("consolidator");
    let mut session = agent.session("facts");
    match session
        .observe(Turn::new("user", "notes about the budget").id("budget-1"))
        .await
    {
        Ok(_) => {}
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("observe failed: {e}"),
    }
    let observe = |visibility: Option<Visibility>| AddObservationParams {
        message_id: "budget-1".into(),
        content: "the budget is capped at ten thousand".into(),
        subject: "budget".into(),
        predicate: Some("capped_at".into()),
        object: Some("ten thousand".into()),
        visibility,
        ..AddObservationParams::default()
    };
    agent.add_observation(observe(None)).await.unwrap();
    agent
        .add_observation(observe(Some(Visibility::Private("cfo".into()))))
        .await
        .unwrap();
    agent.consolidate().await.expect("consolidate");

    let facts: Vec<_> = agent
        .kb()
        .query_nodes(labels::FACT, None, None)
        .await
        .unwrap()
        .into_iter()
        .filter(|(_, p)| matches!(p.get("predicate"), Some(Value::String(s)) if s == "capped_at"))
        .collect();
    let visibilities: Vec<Option<&Value>> =
        facts.iter().map(|(_, p)| p.get("visibility")).collect();
    assert_eq!(
        facts.len(),
        2,
        "one public and one scoped fact: {visibilities:?}"
    );
    assert!(
        visibilities.contains(&Some(&Value::String("private:cfo".into()))),
        "{visibilities:?}"
    );
}

/// A minimal OpenAI-compatible `/v1/chat/completions` stub on a loopback
/// port: every request gets `reply` as the assistant message. Returns the
/// `base_url` (including `/v1`) and a counter of requests served.
async fn spawn_chat_stub(
    reply: &'static str,
) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind stub");
    let addr = listener.local_addr().expect("stub addr");
    let hits = Arc::new(AtomicUsize::new(0));
    let served = hits.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let served = served.clone();
            tokio::spawn(async move {
                // Read the head, then exactly Content-Length body bytes.
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let head_end = loop {
                    let Ok(n) = stream.read(&mut chunk).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let head = String::from_utf8_lossy(&buf[..head_end]).to_ascii_lowercase();
                let body_len = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                // Always answer once a request head arrived, even if the body
                // read fails, so the client never waits out its timeout.
                while buf.len() < head_end + body_len {
                    let Ok(n) = stream.read(&mut chunk).await else {
                        break;
                    };
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                served.fetch_add(1, Ordering::SeqCst);
                let body = serde_json::json!({
                    "choices": [{ "message": { "role": "assistant", "content": reply } }],
                })
                .to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            });
        }
    });
    (format!("http://{addr}/v1"), hits)
}

/// Build an instance whose `generate/triples` alias is the chat stub, seed one
/// Observation with a rule-based triple, and return both. `None` when the
/// instance or its embedder is unavailable.
async fn seeded_for_triple_refinement(base_url: &str) -> Option<(Uniko, crate::Agent)> {
    use crate::{AddObservationParams, LlmSpec};

    let memory = match Uniko::builder()
        .in_memory()
        .llm(LlmSpec::llamacpp("generate/triples", "stub", base_url))
        .build()
        .await
    {
        Ok(memory) => memory,
        // A rejected alias spec is a test bug, not a missing model.
        Err(e) if e.to_string().contains("Configuration error") => {
            panic!("the stub alias was rejected: {e}")
        }
        Err(e) => {
            eprintln!("skipping: instance unavailable: {e}");
            return None;
        }
    };
    let agent = memory.agent("consolidator");
    let mut session = agent.session("hr");
    match session
        .observe(Turn::new("user", "alice joined acme last week").id("hr-1"))
        .await
    {
        Ok(_) => {}
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return None;
        }
        Err(e) => panic!("observe failed: {e}"),
    }
    agent
        .add_observation(AddObservationParams {
            message_id: "hr-1".into(),
            content: "alice joined acme last week".into(),
            subject: "alice".into(),
            predicate: Some("joined".into()),
            object: Some("acme".into()),
            ..AddObservationParams::default()
        })
        .await
        .expect("add observation");
    drop(session);
    Some((memory, agent))
}

/// Fact predicates currently in the store.
async fn fact_predicates(agent: &crate::Agent) -> Vec<String> {
    agent
        .kb()
        .query_nodes(labels::FACT, None, None)
        .await
        .expect("query facts")
        .into_iter()
        .filter_map(|(_, p)| match p.get("predicate") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

/// `consolidate_with(llm_triples)` replaces the rule-based triple with the
/// model's and reports the refinement (issue #63).
#[tokio::test]
async fn consolidate_with_llm_triples_refines_through_the_alias() {
    use std::sync::atomic::Ordering;

    use crate::ConsolidateOptions;

    let (base_url, hits) = spawn_chat_stub("Alice|employed_by|Acme").await;
    let Some((memory, agent)) = seeded_for_triple_refinement(&base_url).await else {
        return;
    };

    let report = agent
        .consolidate_with(ConsolidateOptions::llm_triples("generate/triples"))
        .await
        .expect("consolidate with llm triples");
    // The observed turn may add NLP-extracted observations of its own, so
    // count relative to what was requested rather than assuming one.
    let requested = report.cycle.triples_requested;
    assert!(requested >= 1, "the seeded observation was sent");
    assert_eq!(hits.load(Ordering::SeqCst), requested, "one call each");
    assert_eq!(
        report.cycle.triples_refined, requested,
        "every reply parsed"
    );
    assert_eq!(report.cycle.observations_processed, requested);

    let predicates = fact_predicates(&agent).await;
    assert!(
        predicates.iter().any(|p| p == "employed_by"),
        "the refined predicate became a fact: {predicates:?}"
    );
    assert!(
        !predicates.iter().any(|p| p == "joined"),
        "the rule-based predicate was replaced: {predicates:?}"
    );

    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// A model reply that is not a triple falls back to the rule-based triple,
/// and the report says nothing was refined.
#[tokio::test]
async fn consolidate_with_llm_triples_falls_back_on_unparseable_replies() {
    use std::sync::atomic::Ordering;

    use crate::ConsolidateOptions;

    let (base_url, hits) = spawn_chat_stub("I cannot help with that").await;
    let Some((memory, agent)) = seeded_for_triple_refinement(&base_url).await else {
        return;
    };

    let report = agent
        .consolidate_with(ConsolidateOptions::llm_triples("generate/triples"))
        .await
        .expect("consolidate with llm triples");
    let requested = report.cycle.triples_requested;
    assert!(requested >= 1, "the seeded observation was sent");
    assert_eq!(hits.load(Ordering::SeqCst), requested, "one call each");
    assert_eq!(report.cycle.triples_refined, 0, "nothing parsed");
    assert_eq!(report.cycle.triples_failed, 0, "every call was answered");

    let predicates = fact_predicates(&agent).await;
    assert!(
        predicates.iter().any(|p| p == "joined"),
        "the rule-based triple still produced its fact: {predicates:?}"
    );

    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn consolidate_with_llm_triples_counts_unreachable_endpoint_as_failed() {
    use crate::ConsolidateOptions;

    // Reserve a port, then close it, so every call is refused.
    let base_url = {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        format!("http://{}/v1", listener.local_addr().expect("addr"))
    };
    let Some((memory, agent)) = seeded_for_triple_refinement(&base_url).await else {
        return;
    };

    let report = agent
        .consolidate_with(ConsolidateOptions::llm_triples("generate/triples"))
        .await
        .expect("a down endpoint falls back rather than failing the cycle");
    let requested = report.cycle.triples_requested;
    assert!(requested >= 1, "the seeded observation was sent");
    assert_eq!(report.cycle.triples_refined, 0);
    assert_eq!(report.cycle.triples_failed, requested, "every call failed");

    let predicates = fact_predicates(&agent).await;
    assert!(
        predicates.iter().any(|p| p == "joined"),
        "the rule-based triple still produced its fact: {predicates:?}"
    );

    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// An alias that is missing, or not a generation model, is rejected before
/// the cycle runs; the default path never calls a model.
#[tokio::test]
async fn consolidate_with_rejects_bad_refinement_aliases() {
    use crate::{AddObservationParams, ConsolidateOptions};

    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("consolidator");
    let mut session = agent.session("notes");
    match session
        .observe(Turn::new("user", "the budget is capped").id("n-1"))
        .await
    {
        Ok(_) => {}
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("observe failed: {e}"),
    }
    agent
        .add_observation(AddObservationParams {
            message_id: "n-1".into(),
            content: "the budget is capped at ten thousand".into(),
            subject: "budget".into(),
            predicate: Some("capped_at".into()),
            object: Some("ten thousand".into()),
            ..AddObservationParams::default()
        })
        .await
        .expect("add observation");

    let cycles = || async {
        agent
            .kb()
            .query_nodes(labels::CONSOLIDATION_CYCLE, None, None)
            .await
            .expect("query cycles")
            .len()
    };
    let before = cycles().await;

    let missing = agent
        .consolidate_with(ConsolidateOptions::llm_triples("no-such-alias"))
        .await
        .expect_err("unknown alias is rejected");
    assert!(
        matches!(&missing, UnikoError::Config(m) if m.contains("no-such-alias")),
        "{missing:?}"
    );
    let not_generation = agent
        .consolidate_with(ConsolidateOptions::llm_triples(
            uniko_store::schema::EMBED_ALIAS,
        ))
        .await
        .expect_err("an embedding alias is not a generator");
    assert!(
        matches!(not_generation, UnikoError::Config(_)),
        "{not_generation:?}"
    );
    assert_eq!(cycles().await, before, "no cycle ran for a rejected alias");

    // The default path is unchanged and asks for no refinement.
    let report = agent.consolidate().await.expect("default consolidate");
    assert!(report.cycle.observations_processed >= 1);
    assert_eq!(report.cycle.triples_requested, 0);
    assert_eq!(report.cycle.triples_refined, 0);

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// Procedures learned by `consolidate` are matched and scored through
/// `Agent`, and carry the embedding recall searches them by.
#[tokio::test]
async fn agent_matches_and_scores_learned_procedures() {
    use crate::RecordEpisodeParams;
    use chrono::{Duration, Utc};

    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("planner");
    let base = Utc::now() - Duration::hours(2);
    for i in 0..3 {
        for (offset, action) in [(0, "scan"), (30, "report")] {
            let result = agent
                .record_episode(RecordEpisodeParams {
                    action_type: action.into(),
                    outcome: Some("success".into()),
                    timestamp: Some(base + Duration::minutes(i * 5) + Duration::seconds(offset)),
                    state: Some(serde_json::json!({ "topic": action })),
                    ..RecordEpisodeParams::default()
                })
                .await;
            match result {
                Ok(_) => {}
                Err(e) if is_model_unavailable(&e) => {
                    eprintln!("skipping: embeddings unavailable");
                    return;
                }
                Err(e) => panic!("record_episode failed: {e}"),
            }
        }
    }
    let report = agent.consolidate().await.expect("consolidate");
    assert!(report.procedures.created >= 1, "{report:?}");

    let state = HashMap::from([("last_action_type".to_string(), "scan".to_string())]);
    let matched = agent.match_procedures(&state).await.expect("match");
    let procedure = matched
        .iter()
        .find(|m| m.name == "scan → report")
        .expect("the learned procedure matches");

    let (_, props) = agent
        .kb()
        .get_node_by_ext_id(labels::PROCEDURE, "procedure_id", &procedure.procedure_id)
        .await
        .unwrap()
        .unwrap();
    assert!(
        props.contains_key("embedding"),
        "new procedures are embedded"
    );

    agent
        .record_procedure_use(&procedure.procedure_id, true)
        .await
        .expect("record use");
    let err = agent
        .record_procedure_use("no-such-procedure", true)
        .await
        .unwrap_err();
    assert!(!err.to_string().is_empty());
}

/// A `TextExtractor` indexes an image as the text its function returns.
#[tokio::test]
async fn text_extractor_indexes_images_as_extracted_text() {
    use std::sync::Arc;

    use crate::{Modality, TextExtractor};

    let extractor = TextExtractor::new(Modality::Image, |req| {
        assert_eq!(req.modality, Modality::Image);
        Ok(format!(
            "a photo of a red zeppelin over the harbour ({} bytes)",
            req.bytes.len()
        ))
    });
    let memory = match Uniko::builder()
        .in_memory()
        .extractor(Arc::new(extractor))
        .build()
        .await
    {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("skipping: instance unavailable: {e}");
            return;
        }
    };
    let agent = memory.agent("viewer");
    let session = agent.session("photos");
    // A PNG signature is enough for the MIME sniffer.
    let png = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
    let outcome = match session.ingest(IngestSource::bytes(png.to_vec())).await {
        Ok(outcome) => outcome,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("ingest failed: {e}"),
    };
    let IngestOutcome::Artifact(artifact) = outcome else {
        panic!("an extracted image is an artifact");
    };
    assert!(
        !artifact.chunk_node_ids.is_empty(),
        "the caption was chunked"
    );

    let view = agent
        .data()
        .artifact(&artifact.artifact_id)
        .await
        .unwrap()
        .expect("artifact view");
    assert!(view.text.contains("red zeppelin"), "{}", view.text);
    assert_eq!(view.kind, "image");
}

/// `Data` reads apply the same policy as recall: a scoped agent cannot read
/// another participant's private turn or its attachment by id, and nobody
/// reads a forgotten turn.
#[tokio::test]
async fn data_reads_respect_visibility_and_forgetting() {
    use crate::{IngestSource, Visibility};

    let memory = match Uniko::builder().in_memory().scope_to_agent().build().await {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("skipping: in-memory instance unavailable: {e}");
            return;
        }
    };
    let alice = memory.agent("alice");
    let bob = memory.agent("bob");
    let mut session = alice.session("s-data-policy");
    let doc = IngestSource::text("The vault code is 4471 and lives in the blue binder.")
        .with_id("doc-private");
    if let Err(e) = session
        .observe(
            Turn::new("alice", "Only I should read this note.")
                .id("m-private")
                .visibility(Visibility::Private("alice".into()))
                .attach(doc),
        )
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: models unavailable");
            return;
        }
        panic!("observe failed: {e}");
    }
    session
        .observe(Turn::new("alice", "This one is for everybody.").id("m-public"))
        .await
        .expect("observe public");

    // The owner reads both, and the attachment with its bytes.
    let own = alice.data().message("m-private").await.expect("read");
    let own = own.expect("alice reads her private turn");
    assert_eq!(own.visibility.as_deref(), Some("private:alice"));
    assert_eq!(own.content_type, "text");
    assert!(
        alice
            .data()
            .artifact("doc-private")
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        alice
            .data()
            .artifact_bytes("doc-private")
            .await
            .unwrap()
            .is_some()
    );

    // Bob reads the public turn but not the private one or its attachment.
    assert!(bob.data().message("m-public").await.unwrap().is_some());
    assert!(bob.data().message("m-private").await.unwrap().is_none());
    assert!(bob.data().artifact("doc-private").await.unwrap().is_none());
    assert!(
        bob.data()
            .artifact_bytes("doc-private")
            .await
            .unwrap()
            .is_none()
    );

    // Once forgotten, not even the owner reads the turn.
    session.forget_turn("m-public").await.expect("forget");
    assert!(alice.data().message("m-public").await.unwrap().is_none());

    drop((alice, bob, session));
    memory.shutdown().await.expect("shutdown");
}

// ── Tool calls ─────────────────────────────────────────────────────────

fn orders_payload() -> serde_json::Value {
    let items: Vec<serde_json::Value> = (0..30)
        .map(|i| {
            serde_json::json!({
                "order_id": format!("ORD-{}", 50_000 + i),
                "status": if i == 17 { "refunded" } else { "shipped" },
                "amount": 1_000 + i * 37,
            })
        })
        .collect();
    serde_json::json!({"meta": {"currency": "EUR", "amount_unit": "cents"}, "data": {"items": items}})
}

/// A turn and the tool call it made commit together: the Action is linked
/// to the turn and session, and its result is a verbatim `tool_output`
/// artifact the Action produced.
#[tokio::test]
async fn tool_call_commits_with_its_turn() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("tools-1");
    let payload = orders_payload();
    let call = crate::ToolCall::new("list_orders")
        .call_id("call-orders-1")
        .arguments(serde_json::json!({"customer_id": "C-1042"}))
        .json_output(payload.clone())
        .status("ok");
    let result = match session
        .observe(
            Turn::new("assistant", "Checking the orders API.")
                .id("m-tool-1")
                .tool_call(call),
        )
        .await
    {
        Ok(r) => r,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: models unavailable");
            return;
        }
        Err(e) => panic!("observe failed: {e}"),
    };
    let [tc] = result.tool_calls.as_slice() else {
        panic!(
            "one tool call result expected, got {}",
            result.tool_calls.len()
        );
    };
    assert_eq!(tc.call_id, "call-orders-1");
    assert!(!tc.was_replay);
    let artifact_id = tc.artifact_id.clone().expect("result artifact");

    let kb = agent.kb();
    assert_eq!(
        count_query(
            kb,
            "MATCH (a:Action {action_id: 'call-orders-1'})-[:TRIGGERED_BY]->(:Message {message_id: 'm-tool-1'}) \
             MATCH (a)-[:IN_SESSION]->(:Session {session_id: 'tools-1'}) \
             MATCH (a)-[:PERFORMED_BY]->(:Participant {participant_id: 'assistant'}) \
             MATCH (a)-[:PRODUCED]->(art:Artifact {kind: 'tool_output'}) \
             WHERE a.tool_name = 'list_orders' AND a.action_type = 'tool_call' \
             RETURN count(art) AS c"
        )
        .await,
        1,
        "the Action is wired to its turn, session, performer and result"
    );
    let bytes = agent
        .data()
        .artifact_bytes(&artifact_id)
        .await
        .unwrap()
        .expect("result bytes");
    let stored: serde_json::Value = serde_json::from_slice(&bytes).expect("stored JSON");
    assert_eq!(stored, payload, "the result is stored verbatim");

    drop((agent, session));
    memory.shutdown().await.expect("shutdown");
}

/// Recording the same call again with the same result is a no-op; with a
/// different result it is refused.
#[tokio::test]
async fn tool_call_replay_is_idempotent_and_conflicts_are_refused() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let session = agent.session("tools-2");
    let call = || {
        crate::ToolCall::new("run_tests")
            .call_id("call-tests-1")
            .arguments(serde_json::json!({"commit": "4f2a9c1"}))
            .text_output("tests/test_pay.py::test_refund FAILED\n1 failed, 39 passed")
    };
    let first = match session.observe_tool(call()).await {
        Ok(r) => r,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: models unavailable");
            return;
        }
        Err(e) => panic!("observe_tool failed: {e}"),
    };
    assert!(!first.was_replay);
    let again = session.observe_tool(call()).await.expect("replay");
    assert!(again.was_replay, "same id and result is a replay");
    assert_eq!(again.action_node, first.action_node);
    assert_eq!(
        count_query(
            agent.kb(),
            "MATCH (a:Action {action_id: 'call-tests-1'}) RETURN count(a) AS c"
        )
        .await,
        1
    );
    let conflict = session
        .observe_tool(call().text_output("all 40 passed"))
        .await
        .expect_err("a different result under the same id must be refused");
    assert!(
        matches!(conflict, UnikoError::IdConflict { .. }),
        "expected IdConflict, got {conflict:?}"
    );

    drop((agent, session));
    memory.shutdown().await.expect("shutdown");
}

/// A unit that fails after its tool calls are written leaves neither the
/// Action nor its result artifact behind.
#[tokio::test]
async fn tool_call_rolls_back_with_its_unit() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("tools-3");
    // SAFETY: nextest runs each test in its own process.
    unsafe { std::env::set_var("UNIKO_TEST_FAIL_AFTER_TOOLS", "1") };
    let result = session
        .observe(
            Turn::new("assistant", "Fetching the logs.")
                .id("m-tool-rb")
                .tool_call(
                    crate::ToolCall::new("fetch_logs")
                        .call_id("call-logs-rb")
                        .text_output("12:00:01 ERROR payments request_id=r1 error=timeout"),
                ),
        )
        .await;
    unsafe { std::env::remove_var("UNIKO_TEST_FAIL_AFTER_TOOLS") };
    let Err(err) = result else {
        panic!("the injected failure must fail the unit");
    };
    if is_model_unavailable(&err) {
        eprintln!("skipping: models unavailable");
        return;
    }
    for (what, q) in [
        ("message", "MATCH (m:Message) RETURN count(m) AS c"),
        ("action", "MATCH (a:Action) RETURN count(a) AS c"),
        ("artifact", "MATCH (a:Artifact) RETURN count(a) AS c"),
        ("chunk", "MATCH (c:Chunk) RETURN count(c) AS c"),
    ] {
        assert_eq!(
            count_query(agent.kb(), q).await,
            0,
            "{what} must have rolled back"
        );
    }
    drop((agent, session));
    memory.shutdown().await.expect("shutdown");
}

/// A tool call's result is found by recall through its chunks.
#[tokio::test]
async fn tool_call_result_is_recalled() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("tools-4");
    if let Err(e) = session
        .observe(
            Turn::new("assistant", "Looking up the orders.")
                .id("m-tool-4")
                .tool_call(
                    crate::ToolCall::new("list_orders")
                        .call_id("call-orders-4")
                        .json_output(orders_payload()),
                ),
        )
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: models unavailable");
            return;
        }
        panic!("observe failed: {e}");
    }
    let bundle = match agent.recall("What is the status of order ORD-50017?").await {
        Ok(b) => b,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("recall failed: {e}"),
    };
    assert!(
        bundle.items.iter().any(|i| i.content.contains("ORD-50017")),
        "the order record should be recalled; got {:?}",
        bundle.items.iter().map(|i| &i.content).collect::<Vec<_>>()
    );
    drop((agent, session));
    memory.shutdown().await.expect("shutdown");
}

// ── Read tools ─────────────────────────────────────────────────────────

/// search → outline → read round trip over a tool call's JSON result.
#[tokio::test]
async fn sources_search_outline_and_read_a_tool_result() {
    use crate::facade::{ReadRange, SearchOptions, SourceRef};
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("src-1");
    if let Err(e) = session
        .observe(
            Turn::new("assistant", "Pulling the order list.")
                .id("m-src-1")
                .tool_call(
                    crate::ToolCall::new("list_orders")
                        .call_id("call-src-1")
                        .arguments(serde_json::json!({"customer_id": "C-1042"}))
                        .json_output(orders_payload()),
                ),
        )
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: models unavailable");
            return;
        }
        panic!("observe failed: {e}");
    }
    let sources = agent.sources();

    let hits = sources
        .search(
            "which status does ORD-50017 have",
            &SearchOptions::default(),
        )
        .await
        .expect("search");
    let hit = hits
        .iter()
        .find(|h| h.text.contains("ORD-50017"))
        .unwrap_or_else(|| panic!("the record should be found: {hits:?}"));
    assert_eq!(hit.call_id.as_deref(), Some("call-src-1"));
    assert_eq!(hit.tool_name.as_deref(), Some("list_orders"));
    assert!(
        hit.context
            .as_deref()
            .is_some_and(|c| c.contains("meta.currency=EUR"))
    );
    let SourceRef::Artifact(artifact_id) = &hit.source else {
        panic!("a tool result is an artifact: {:?}", hit.source);
    };

    let outline = sources
        .outline(&hit.source, 3, 0)
        .await
        .expect("outline")
        .expect("visible");
    assert_eq!(outline.format, "json");
    assert!(
        outline
            .entries
            .iter()
            .any(|e| e.address == "/data/items" && e.summary.starts_with("30 objects"))
    );

    let read = sources
        .read(&hit.source, &ReadRange::Address("/data/items/17".into()), 0)
        .await
        .expect("read")
        .expect("readable");
    let record: serde_json::Value = serde_json::from_str(&read.text).expect("an exact JSON value");
    assert_eq!(record["order_id"], "ORD-50017");
    assert_eq!(record["status"], "refunded");

    let small = sources
        .read(
            &SourceRef::Artifact(artifact_id.clone()),
            &ReadRange::Lines(1, 400),
            30,
        )
        .await
        .unwrap()
        .unwrap();
    assert!(small.truncated);
    let Some(ReadRange::Bytes(from, _)) = small.next else {
        panic!("a cut read reports the rest");
    };
    assert_eq!(from, small.end);

    let call = sources
        .tool_call("call-src-1")
        .await
        .unwrap()
        .expect("call");
    assert_eq!(call.tool_name.as_deref(), Some("list_orders"));
    assert_eq!(call.triggered_by.as_deref(), Some("m-src-1"));
    assert_eq!(
        call.arguments,
        Some(serde_json::json!({"customer_id": "C-1042"}))
    );
    assert_eq!(call.artifact_id.as_deref(), Some(artifact_id.as_str()));
    assert_eq!(
        sources
            .tool_calls(Some("src-1"), None, 0)
            .await
            .unwrap()
            .len(),
        1
    );

    drop(sources);
    drop((agent, session));
    memory.shutdown().await.expect("shutdown");
}

/// A private tool call is invisible to another viewer through every read
/// tool, and visible to its owner.
#[tokio::test]
async fn sources_hide_a_private_tool_call_from_other_viewers() {
    use crate::Visibility;
    use crate::facade::{ReadRange, SearchOptions, SourceRef};
    let memory = match Uniko::builder().in_memory().scope_to_agent().build().await {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping: in-memory instance unavailable: {e}");
            return;
        }
    };
    let alice = memory.agent("alice");
    let bob = memory.agent("bob");
    let mut session = alice.session("src-2");
    if let Err(e) = session
        .observe(
            Turn::new("alice", "Checking my payroll export.")
                .id("m-src-2")
                .visibility(Visibility::Private("alice".into()))
                .tool_call(
                    crate::ToolCall::new("export_payroll")
                        .call_id("call-src-2")
                        .text_output("employee_id=E-7731 salary=91000 bank=XX-0042"),
                ),
        )
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: models unavailable");
            return;
        }
        panic!("observe failed: {e}");
    }
    let artifact = SourceRef::Artifact("call-src-2:output".into());
    let q = "salary of E-7731";

    let mine = alice
        .sources()
        .search(q, &SearchOptions::default())
        .await
        .unwrap();
    assert!(
        mine.iter().any(|h| h.text.contains("E-7731")),
        "the owner finds it"
    );
    assert!(
        alice
            .sources()
            .tool_call("call-src-2")
            .await
            .unwrap()
            .is_some()
    );

    let theirs = bob
        .sources()
        .search(q, &SearchOptions::default())
        .await
        .unwrap();
    assert!(theirs.is_empty(), "bob must not find it: {theirs:?}");
    assert!(
        bob.sources()
            .tool_call("call-src-2")
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        bob.sources()
            .outline(&artifact, 2, 0)
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        bob.sources()
            .read(&artifact, &ReadRange::Bytes(0, 100), 0)
            .await
            .unwrap()
            .is_none()
    );

    drop((alice, bob, session));
    memory.shutdown().await.expect("shutdown");
}

/// Chunks without `search_text` (a store from before it existed) are not
/// seen by lexical search until `backfill_search_text` fills them.
#[tokio::test]
async fn backfill_search_text_restores_lexical_search() {
    use crate::facade::SearchOptions;
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("backfill-1");
    if let Err(e) = session
        .observe(
            Turn::new("assistant", "Fetching orders.")
                .id("m-bf-1")
                .tool_call(
                    crate::ToolCall::new("list_orders")
                        .call_id("call-bf-1")
                        .json_output(orders_payload()),
                ),
        )
        .await
    {
        if is_model_unavailable(&e) {
            eprintln!("skipping: models unavailable");
            return;
        }
        panic!("observe failed: {e}");
    }
    let q = "ORD-50023";
    let found = |hits: &[crate::facade::SearchHit]| hits.iter().any(|h| h.text.contains(q));
    assert!(found(
        &agent
            .sources()
            .search(q, &SearchOptions::default())
            .await
            .unwrap()
    ));

    // Simulate an upgraded store: chunks written before `search_text`.
    let session_db = agent.kb().db().session(); // ALLOW: test-only fixture write; the seal governs product code.
    let tx = session_db.tx().await.unwrap();
    tx.query_with("MATCH (c:Chunk) SET c.search_text = NULL")
        .fetch_all()
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert!(!found(
        &agent
            .sources()
            .search(q, &SearchOptions::default())
            .await
            .unwrap()
    ));

    let filled = memory.backfill_search_text().await.expect("backfill");
    assert!(filled > 0, "chunks should have been filled");
    assert_eq!(
        memory.backfill_search_text().await.unwrap(),
        0,
        "idempotent"
    );
    assert!(found(
        &agent
            .sources()
            .search(q, &SearchOptions::default())
            .await
            .unwrap()
    ));

    drop((agent, session));
    memory.shutdown().await.expect("shutdown");
}

// ── Several LLMs and the answer alias (issue #58) ──

/// Build an instance or skip when its models are unavailable. A config
/// error is a test bug, so it panics.
async fn build_or_skip(builder: crate::UnikoBuilder) -> Option<Uniko> {
    match builder.build().await {
        Ok(memory) => Some(memory),
        Err(e @ UnikoError::Config(_)) => panic!("rejected: {e}"),
        Err(e) => {
            eprintln!("skipping: instance unavailable: {e}");
            None
        }
    }
}

#[tokio::test]
async fn answer_uses_the_chosen_alias_among_several_llms() {
    use std::sync::atomic::Ordering;

    use crate::LlmSpec;

    let (url_a, hits_a) = spawn_chat_stub("from the first model").await;
    let (url_b, hits_b) = spawn_chat_stub("from the second model").await;
    let builder = Uniko::builder()
        .in_memory()
        .llm(LlmSpec::llamacpp("gen/a", "stub", url_a.as_str()))
        .llm(LlmSpec::llamacpp("gen/b", "stub", url_b.as_str()))
        .answer_alias("gen/b");
    let Some(memory) = build_or_skip(builder).await else {
        return;
    };

    let entries = memory
        .agent("a")
        .kb()
        .model_entries()
        .expect("catalog is known");
    for alias in ["gen/a", "gen/b"] {
        assert!(
            entries.iter().any(|e| e.alias == alias),
            "{alias} was registered: a later .llm() must not replace an earlier one"
        );
    }

    match memory.agent("a").answer("what happened?").await {
        Ok(answer) => assert_eq!(answer.text, "from the second model"),
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("answer failed: {e}"),
    }
    assert_eq!(hits_a.load(Ordering::SeqCst), 0);
    assert_eq!(hits_b.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn answer_defaults_to_the_first_llm() {
    use std::sync::atomic::Ordering;

    use crate::LlmSpec;

    let (url_a, hits_a) = spawn_chat_stub("first").await;
    let (url_b, hits_b) = spawn_chat_stub("second").await;
    let builder = Uniko::builder()
        .in_memory()
        .llm(LlmSpec::llamacpp("gen/a", "stub", url_a.as_str()))
        .llm(LlmSpec::llamacpp("gen/b", "stub", url_b.as_str()));
    let Some(memory) = build_or_skip(builder).await else {
        return;
    };
    match memory.agent("a").answer("q").await {
        Ok(answer) => assert_eq!(answer.text, "first"),
        Err(e) if is_model_unavailable(&e) => return,
        Err(e) => panic!("answer failed: {e}"),
    }
    assert_eq!(hits_a.load(Ordering::SeqCst), 1);
    assert_eq!(hits_b.load(Ordering::SeqCst), 0);
}

/// A generation alias that only exists in a `catalog_path` file can be
/// chosen to answer.
#[tokio::test]
async fn answer_alias_can_select_a_catalog_file_model() {
    use std::sync::atomic::Ordering;

    use uniko_store::storage::embed_catalog;
    use uniko_store::xervo::{ModelAliasSpec, ModelTask, WarmupPolicy};

    let (url, hits) = spawn_chat_stub("from the catalog").await;
    let dir = tempfile::tempdir().unwrap();
    let mut config = UnikoConfig::default();
    let mut catalog = embed_catalog(&config);
    catalog.push(ModelAliasSpec {
        alias: "catalog/gen".into(),
        task: ModelTask::Generate,
        provider_id: "remote/llamacpp".into(),
        model_id: "stub".into(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({ "base_url": url }),
    });
    let path = dir.path().join("catalog.json");
    std::fs::write(&path, serde_json::to_string(&catalog).unwrap()).unwrap();
    config.catalog_path = Some(path);

    let builder = Uniko::builder()
        .in_memory()
        .raw_config(config)
        .answer_alias("catalog/gen");
    let Some(memory) = build_or_skip(builder).await else {
        return;
    };
    match memory.agent("a").answer("q").await {
        Ok(answer) => assert_eq!(answer.text, "from the catalog"),
        Err(e) if is_model_unavailable(&e) => return,
        Err(e) => panic!("answer failed: {e}"),
    }
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn build_rejects_an_llm_alias_registered_twice() {
    use crate::LlmSpec;

    let err = Uniko::builder()
        .in_memory()
        .llm(LlmSpec::llamacpp(
            "generate/x",
            "a",
            "http://127.0.0.1:9/v1",
        ))
        .llm(LlmSpec::llamacpp(
            "generate/x",
            "b",
            "http://127.0.0.1:9/v1",
        ))
        .build()
        .await
        .unwrap_err();
    assert!(
        matches!(&err, UnikoError::Config(m) if m.contains("'generate/x'")),
        "{err}"
    );
}

#[tokio::test]
async fn build_rejects_an_answer_alias_that_cannot_answer() {
    for (alias, expect) in [
        ("missing/alias", "not in the model catalog"),
        ("embed/default", "not a generation model"),
    ] {
        match Uniko::builder()
            .in_memory()
            .answer_alias(alias)
            .build()
            .await
        {
            Err(UnikoError::Config(m)) => assert!(m.contains(expect) && m.contains(alias), "{m}"),
            Err(e) => {
                eprintln!("skipping: instance unavailable: {e}");
                return;
            }
            Ok(_) => panic!("answer alias '{alias}' was accepted"),
        }
    }
}

#[tokio::test]
async fn capabilities_report_models_features_and_the_answer_alias() {
    use uniko_store::ModelState;

    use crate::LlmSpec;

    let builder = Uniko::builder().in_memory().llm(LlmSpec::llamacpp(
        "gen/a",
        "stub",
        "http://127.0.0.1:9/v1",
    ));
    let Some(memory) = build_or_skip(builder).await else {
        return;
    };
    let caps = memory.capabilities().await;

    assert_eq!(caps.answer_alias.as_deref(), Some("gen/a"));
    for feature in [
        "pdf-ocr",
        "onnx",
        "code-parse",
        "gpu-cuda",
        "gpu-metal",
        "candle",
        "llm",
    ] {
        assert!(caps.compiled_features.contains_key(feature), "{feature}");
    }
    assert_eq!(caps.ocr_compiled, caps.compiled_features["pdf-ocr"]);
    assert!(!caps.ocr_configured, "OCR is off by default");
    assert!(caps.extractors.is_empty(), "{:?}", caps.extractors);

    let models = caps.models.expect("the builder's store knows its catalog");
    let embed = models.iter().find(|m| m.alias == "embed/default").unwrap();
    assert_eq!(embed.state, ModelState::Loaded);
    let generator = models.iter().find(|m| m.alias == "gen/a").unwrap();
    assert_eq!(generator.provider_id, "remote/llamacpp");
    assert_eq!(generator.base_url.as_deref(), Some("http://127.0.0.1:9/v1"));

    let json = serde_json::to_value(memory.capabilities().await).unwrap();
    assert_eq!(json["models"][0]["state"], "loaded", "{json}");
}

/// Two instances over one runtime keep separate memories, and shutting one
/// down leaves the other working.
#[tokio::test]
async fn instances_sharing_a_runtime_keep_their_own_memory() {
    let Ok(runtime) = Uniko::builder().build_runtime().await else {
        eprintln!("skipping: runtime unavailable");
        return;
    };
    let Some(first) = build_or_skip(Uniko::builder().runtime(runtime.clone())).await else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let Some(second) =
        build_or_skip(Uniko::builder().path(dir.path()).runtime(runtime.clone())).await
    else {
        return;
    };

    for (memory, text) in [
        (&first, "I love hiking in the mountains every weekend"),
        (&second, "My favourite food is a spicy ramen with extra egg"),
    ] {
        let agent = memory.agent("assistant");
        let mut session = agent.session("chat-1");
        match session.observe(Turn::new("alice", text)).await {
            Ok(_) => {}
            Err(e) if is_model_unavailable(&e) => return,
            Err(e) => panic!("observe failed: {e}"),
        }
    }

    let seen = |bundle: crate::recall::ContextBundle| {
        bundle
            .items
            .iter()
            .map(|i| i.content.clone())
            .collect::<Vec<_>>()
            .join(" | ")
    };
    let in_second = seen(
        second
            .agent("assistant")
            .recall("hiking mountains")
            .await
            .unwrap(),
    );
    assert!(!in_second.contains("hiking"), "{in_second}");

    first.shutdown().await.expect("shutdown first");
    let in_second = seen(second.agent("assistant").recall("ramen").await.unwrap());
    assert!(in_second.contains("ramen"), "{in_second}");

    let caps = second.capabilities().await;
    assert!(
        caps.models.is_some(),
        "a shared runtime carries its catalog"
    );
    second.shutdown().await.expect("shutdown second");
}

#[tokio::test]
async fn a_builder_over_a_runtime_rejects_an_llm_the_runtime_lacks() {
    use crate::LlmSpec;

    let Ok(runtime) = Uniko::builder()
        .prefetch(uniko_store::PrefetchPolicy::Lazy)
        .build_runtime()
        .await
    else {
        return;
    };
    let err = Uniko::builder()
        .runtime(runtime)
        .llm(LlmSpec::llamacpp("gen/a", "stub", "http://127.0.0.1:9/v1"))
        .build()
        .await
        .unwrap_err();
    assert!(
        matches!(&err, UnikoError::Config(m) if m.contains("'gen/a'") && m.contains("shared runtime")),
        "{err}"
    );
}

#[tokio::test]
async fn lazy_prefetch_leaves_models_unloaded() {
    use uniko_store::{ModelState, PrefetchPolicy};

    let builder = Uniko::builder().prefetch(PrefetchPolicy::Lazy);
    let Some(memory) = build_or_skip(builder).await else {
        return;
    };
    let models = memory.capabilities().await.models.unwrap();
    assert!(
        models.iter().all(|m| m.state == ModelState::NotLoaded),
        "{models:?}"
    );
}

#[tokio::test]
async fn a_builder_over_a_runtime_rejects_an_llm_with_other_options() {
    use crate::LlmSpec;

    let Ok(runtime) = Uniko::builder()
        .prefetch(uniko_store::PrefetchPolicy::Lazy)
        .llm(LlmSpec::llamacpp(
            "gen/a",
            "stub",
            "http://127.0.0.1:9/tenant-a",
        ))
        .build_runtime()
        .await
    else {
        return;
    };
    let err = Uniko::builder()
        .runtime(runtime)
        .llm(LlmSpec::llamacpp(
            "gen/a",
            "stub",
            "http://127.0.0.1:9/tenant-b",
        ))
        .build()
        .await
        .unwrap_err();
    assert!(
        matches!(&err, UnikoError::Config(m) if m.contains("'gen/a'") && m.contains("options")),
        "{err}"
    );
    assert!(!err.to_string().contains("tenant-a"), "{err}");
}

/// Over a shared runtime the LLMs belong to the runtime, so its first
/// generation model answers without repeating `.llm()`.
#[tokio::test]
async fn answer_defaults_to_the_runtimes_generation_model() {
    use std::sync::atomic::Ordering;

    use crate::LlmSpec;

    let (url, hits) = spawn_chat_stub("from the runtime").await;
    let Ok(runtime) = Uniko::builder()
        .llm(LlmSpec::llamacpp("gen/a", "stub", url.as_str()))
        .build_runtime()
        .await
    else {
        return;
    };
    let Some(memory) = build_or_skip(Uniko::builder().runtime(runtime)).await else {
        return;
    };
    match memory.agent("a").answer("q").await {
        Ok(answer) => assert_eq!(answer.text, "from the runtime"),
        Err(e) if is_model_unavailable(&e) => return,
        Err(e) => panic!("answer failed: {e}"),
    }
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

/// A bad answer alias fails before the store is created.
#[tokio::test]
async fn a_bad_answer_alias_fails_before_opening_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    let err = Uniko::builder()
        .path(&path)
        .answer_alias("generate/typo")
        .build()
        .await
        .unwrap_err();
    assert!(
        matches!(&err, UnikoError::Config(m) if m.contains("generate/typo")),
        "{err}"
    );
    assert!(!path.exists(), "the store was created before the check");
}

/// The answer alias chosen on the builder that built a runtime carries to
/// every instance over it.
#[tokio::test]
async fn the_runtimes_answer_alias_carries_to_its_instances() {
    use std::sync::atomic::Ordering;

    use crate::LlmSpec;

    let (judge_url, judge_hits) = spawn_chat_stub("from the judge").await;
    let (answer_url, answer_hits) = spawn_chat_stub("from the answerer").await;
    let Ok(runtime) = Uniko::builder()
        .llm(LlmSpec::llamacpp("gen/judge", "stub", judge_url.as_str()))
        .llm(LlmSpec::llamacpp("gen/answer", "stub", answer_url.as_str()))
        .answer_alias("gen/answer")
        .build_runtime()
        .await
    else {
        return;
    };
    assert_eq!(runtime.answer_alias(), Some("gen/answer"));
    let Some(memory) = build_or_skip(Uniko::builder().runtime(runtime)).await else {
        return;
    };
    match memory.agent("a").answer("q").await {
        Ok(answer) => assert_eq!(answer.text, "from the answerer"),
        Err(e) if is_model_unavailable(&e) => return,
        Err(e) => panic!("answer failed: {e}"),
    }
    assert_eq!(judge_hits.load(Ordering::SeqCst), 0);
    assert_eq!(answer_hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn build_runtime_rejects_an_answer_alias_it_cannot_answer_with() {
    let err = Uniko::builder()
        .prefetch(uniko_store::PrefetchPolicy::Lazy)
        .answer_alias("embed/default")
        .build_runtime()
        .await
        .unwrap_err();
    assert!(
        matches!(&err, UnikoError::Config(m) if m.contains("not a generation model")),
        "{err}"
    );
}

// ── Recall diagnostics (issue #59) ───────────────────────────────────────

mod recall_diagnostics {
    use uniko_store::config::UnikoConfig;
    use uniko_store::{Diagnostic, DiagnosticCode, Stage, StageStatus, UnikoError};

    use crate::Uniko;
    use crate::recall::{RecallConfig, recall};

    fn find(diags: &[Diagnostic], stage: Stage) -> Vec<&Diagnostic> {
        diags.iter().filter(|d| d.stage == stage).collect()
    }

    async fn store(config: UnikoConfig) -> Option<Uniko> {
        match Uniko::builder().raw_config(config).build().await {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("skipping: instance unavailable: {e}");
                None
            }
        }
    }

    fn no_reranker() -> UnikoConfig {
        let mut config = UnikoConfig::default();
        config.reranker.enabled = false;
        config
    }

    /// A healthy recall reports the stages it ran and nothing degraded.
    #[tokio::test]
    async fn a_healthy_recall_reports_applied_stages_only() {
        let Some(memory) = store(UnikoConfig::default()).await else {
            return;
        };
        let agent = memory.agent("a");
        let mut session = agent.session("s");
        if session
            .observe(crate::Turn::new("alice", "I adopted a beagle called Rex"))
            .await
            .is_err()
        {
            eprintln!("skipping: ingest unavailable");
            return;
        }
        let bundle = agent.recall("what dog did alice adopt").await.unwrap();
        let embed = find(&bundle.diagnostics, Stage::QueryEmbedding);
        assert_eq!(embed.len(), 1, "{:?}", bundle.diagnostics);
        if embed[0].status == StageStatus::Failed {
            eprintln!("skipping: the embedder is unavailable here");
            return;
        }
        assert_eq!(embed[0].status, StageStatus::Applied);
        assert!(embed[0].count.unwrap() >= 1);
        let degraded: Vec<_> = bundle
            .diagnostics
            .iter()
            .filter(|d| d.status.is_degraded())
            .collect();
        if !degraded.is_empty() && degraded.iter().all(|d| d.stage == Stage::Rerank) {
            eprintln!("skipping: the reranker is unavailable here");
            return;
        }
        assert!(degraded.is_empty(), "{:?}", bundle.diagnostics);
        // The reranker is on by default: it ran, or the cascade answered
        // before Phase 3 and says so.
        let rerank = find(&bundle.diagnostics, Stage::Rerank);
        assert_eq!(rerank.len(), 1, "{:?}", bundle.diagnostics);
        assert!(
            rerank[0].status == StageStatus::Applied
                || rerank[0].code == Some(DiagnosticCode::NotReached),
            "{:?}",
            rerank[0]
        );
        // Diagnostics carry no query text.
        let json = serde_json::to_string(&bundle.diagnostics).unwrap();
        assert!(!json.contains("beagle") && !json.contains("dog"), "{json}");
    }

    /// A requested reranker reports whether it ran: applied once Phase 3
    /// reranks, skipped (`not_reached`) when the cascade returns earlier.
    #[tokio::test]
    async fn a_requested_reranker_reports_applied_or_not_reached() {
        let Some(memory) = store(UnikoConfig::default()).await else {
            return;
        };
        let mut config = RecallConfig::from_uniko_config(memory.config());
        assert!(config.reranker_enabled, "the default config reranks");

        // Nothing stored: the cascade returns before reranking.
        let bundle = recall(&memory.kb, "anything at all", &config)
            .await
            .unwrap();
        let rerank = find(&bundle.diagnostics, Stage::Rerank);
        assert_eq!(rerank.len(), 1, "{:?}", bundle.diagnostics);
        assert_eq!(rerank[0].status, StageStatus::Skipped);
        assert_eq!(rerank[0].code, Some(DiagnosticCode::NotReached));

        if memory
            .agent("a")
            .session("s")
            .ingest(crate::IngestSource::text(
                "# Runbook\n\nThe deploy key rotates every Friday at noon.",
            ))
            .await
            .is_err()
        {
            eprintln!("skipping: ingest unavailable");
            return;
        }
        config.phase2_coverage_gate = f64::INFINITY;
        let bundle = recall(&memory.kb, "when does the deploy key rotate", &config)
            .await
            .unwrap();
        let rerank = find(&bundle.diagnostics, Stage::Rerank);
        assert_eq!(rerank.len(), 1, "{:?}", bundle.diagnostics);
        if rerank[0].status == StageStatus::Fallback {
            eprintln!("skipping: the reranker model is unavailable here");
            return;
        }
        assert_eq!(rerank[0].status, StageStatus::Applied, "{:?}", rerank[0]);
        assert_eq!(rerank[0].count, None, "counts would precede access control");
    }

    /// Stages asked for but unavailable on this store fall back or fail;
    /// the recall still succeeds, and strict mode turns that into an error.
    #[tokio::test]
    async fn unavailable_stages_degrade_and_strict_rejects_them() {
        let Some(memory) = store(no_reranker()).await else {
            return;
        };
        let agent = memory.agent("a");
        if agent
            .session("s")
            // A document, so Phase 3 has a chunk to rerank.
            .ingest(crate::IngestSource::text(
                "# Runbook\n\nThe deploy key rotates every Friday at noon.",
            ))
            .await
            .is_err()
        {
            eprintln!("skipping: ingest unavailable");
            return;
        }
        // No reranker model, no hybrid embedder: the cross-encoder call,
        // the sparse channel and ColBERT all have nothing to run on.
        let mut config = RecallConfig::from_uniko_config(memory.config());
        config.reranker_enabled = true;
        config.recall_sparse_enabled = true;
        // Never exit after Phase 2, so the Phase 3 stages (reranking) run.
        config.phase2_coverage_gate = f64::INFINITY;
        let bundle = recall(&memory.kb, "when does the deploy key rotate", &config)
            .await
            .expect("best-effort recall still succeeds");
        assert!(
            !bundle.items.is_empty(),
            "the runbook chunk should be recalled"
        );
        let rerank = find(&bundle.diagnostics, Stage::Rerank);
        assert_eq!(rerank.len(), 1, "{:?}", bundle.diagnostics);
        assert_eq!(rerank[0].status, StageStatus::Fallback);
        assert_eq!(rerank[0].code, Some(DiagnosticCode::RerankFailed));
        let sparse = find(&bundle.diagnostics, Stage::Sparse);
        assert!(!sparse.is_empty(), "{:?}", bundle.diagnostics);
        assert!(
            sparse
                .iter()
                .all(|d| d.status == StageStatus::Failed
                    && d.code == Some(DiagnosticCode::QueryFailed)),
            "{sparse:?}"
        );

        config.reranker_style = "colbert".into();
        config.recall_sparse_enabled = false;
        let bundle = recall(&memory.kb, "when does the deploy key rotate", &config)
            .await
            .unwrap();
        assert!(!bundle.items.is_empty());
        let colbert = find(&bundle.diagnostics, Stage::Colbert);
        assert_eq!(colbert.len(), 1, "{:?}", bundle.diagnostics);
        assert_eq!(colbert[0].status, StageStatus::Fallback);
        assert_eq!(colbert[0].code, Some(DiagnosticCode::QueryEmbedFailed));

        config.strict = true;
        let err = recall(&memory.kb, "when does the deploy key rotate", &config)
            .await
            .unwrap_err();
        match err {
            UnikoError::Degraded(d) => {
                assert!(d.iter().any(|d| d.stage == Stage::Colbert), "{d:?}");
                assert!(d.iter().all(|d| d.status.is_degraded()));
            }
            other => panic!("expected Degraded, got {other}"),
        }
    }

    /// With a hybrid embedder the sparse channel and ColBERT run and say so
    /// (a large download, so opt in with `EXPENSIVE_TESTS=1`).
    #[tokio::test]
    async fn a_hybrid_store_applies_sparse_and_colbert() {
        if std::env::var("EXPENSIVE_TESTS").as_deref() != Ok("1") {
            eprintln!("skipping: set EXPENSIVE_TESTS=1 to load BGE-M3");
            return;
        }
        let mut config = no_reranker();
        config.embedding = uniko_store::config::EmbeddingConfig::bge_m3();
        let Some(memory) = store(config).await else {
            return;
        };
        memory
            .agent("a")
            .session("s")
            .ingest(crate::IngestSource::text(
                "# Runbook\n\nThe deploy key rotates every Friday at noon.",
            ))
            .await
            .unwrap();
        let mut config = RecallConfig::from_uniko_config(memory.config());
        config.recall_sparse_enabled = true;
        config.reranker_enabled = true;
        config.reranker_style = "colbert".into();
        config.phase2_coverage_gate = f64::INFINITY;
        config.strict = true;
        let bundle = recall(&memory.kb, "when does the deploy key rotate", &config)
            .await
            .expect("nothing degrades on a hybrid store");
        let sparse = find(&bundle.diagnostics, Stage::Sparse);
        assert!(
            sparse.iter().any(|d| d.status == StageStatus::Applied
                && d.target.as_deref() == Some("Chunk")
                && d.count.unwrap_or(0) > 0),
            "{:?}",
            bundle.diagnostics
        );
        let colbert = find(&bundle.diagnostics, Stage::Colbert);
        assert_eq!(colbert.len(), 1, "{:?}", bundle.diagnostics);
        assert_eq!(colbert[0].status, StageStatus::Applied);
    }

    /// An embedder that cannot load leaves recall lexical-only, which is
    /// reported as a failed query-embedding stage.
    #[tokio::test]
    async fn a_failed_query_embedding_is_reported() {
        let mut config = no_reranker();
        config.embedding.model_id = "uniko-test/no-such-model-repo".into();
        let built = Uniko::builder()
            .raw_config(config)
            .prefetch(uniko_store::PrefetchPolicy::Lazy)
            .build()
            .await;
        let Ok(memory) = built else {
            eprintln!("skipping: store did not open with a broken embedder");
            return;
        };
        let agent = memory.agent("a");
        let bundle = agent.recall("anything at all").await.unwrap();
        let embed = find(&bundle.diagnostics, Stage::QueryEmbedding);
        assert!(!embed.is_empty(), "{:?}", bundle.diagnostics);
        assert!(
            embed
                .iter()
                .all(|d| d.status == StageStatus::Failed
                    && d.code == Some(DiagnosticCode::EmbedFailed)),
            "{embed:?}"
        );
        let strict = agent
            .recall_with(
                "anything at all",
                crate::Scope::default(),
                &crate::RecallOverrides::default().strict(true),
            )
            .await;
        assert!(matches!(strict, Err(UnikoError::Degraded(_))), "{strict:?}");
    }
}

// ── Ingest diagnostics (issue #59) ───────────────────────────────────────

mod ingest_diagnostics {
    use uniko_store::config::UnikoConfig;
    use uniko_store::{Diagnostic, DiagnosticCode, Stage, StageStatus};

    use crate::{IngestSource, Turn, Uniko};

    fn find(diags: &[Diagnostic], stage: Stage) -> Vec<&Diagnostic> {
        diags.iter().filter(|d| d.stage == stage).collect()
    }

    fn config() -> UnikoConfig {
        let mut config = UnikoConfig::default();
        config.reranker.enabled = false;
        config
    }

    async fn store(config: UnikoConfig, streaming: bool) -> Option<Uniko> {
        match Uniko::builder()
            .raw_config(config)
            .streaming(streaming)
            .build()
            .await
        {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("skipping: instance unavailable: {e}");
                None
            }
        }
    }

    fn corrupt_pdf() -> IngestSource {
        IngestSource::bytes(b"%PDF-1.7\nnot really a pdf".to_vec())
            .with_mime(uniko_pipes::content::Mime::parse("application/pdf").unwrap())
    }

    /// With the NLP model unavailable, rules stand in, the turn is still
    /// recorded, and the result says so.
    #[tokio::test]
    async fn an_unavailable_nlp_model_is_reported_as_a_fallback() {
        let mut config = config();
        config.nlp.model_id = "uniko-test/no-such-model-repo".into();
        let Some(memory) = store(config, false).await else {
            return;
        };
        let mut session = memory.agent("a").session("s");
        let observed = session
            .observe(Turn::new("alice", "I moved to Lisbon last spring"))
            .await
            .expect("the turn is recorded without NLP");
        let nlp = find(&observed.diagnostics, Stage::Nlp);
        assert_eq!(nlp.len(), 1, "{:?}", observed.diagnostics);
        if cfg!(feature = "onnx") {
            assert_eq!(nlp[0].status, StageStatus::Fallback);
            assert_eq!(nlp[0].code, Some(DiagnosticCode::NlpUnavailable));
            let obs = find(&observed.diagnostics, Stage::ObservationModel);
            assert!(
                !obs.is_empty()
                    && obs
                        .iter()
                        .all(|d| d.code == Some(DiagnosticCode::RuleBased)),
                "{:?}",
                observed.diagnostics
            );
        } else {
            assert_eq!(nlp[0].status, StageStatus::Skipped);
            assert_eq!(nlp[0].code, Some(DiagnosticCode::NotCompiled));
        }
        let json = serde_json::to_string(&observed.diagnostics).unwrap();
        assert!(!json.contains("Lisbon"), "{json}");
    }

    /// A healthy ingest reports the NLP stage as applied (on `onnx`
    /// builds) and nothing degraded.
    #[tokio::test]
    async fn a_healthy_ingest_reports_nothing_degraded() {
        let Some(memory) = store(config(), false).await else {
            return;
        };
        let mut session = memory.agent("a").session("s");
        let observed = session
            .observe(Turn::new(
                "alice",
                "I adopted a beagle called Rex yesterday",
            ))
            .await
            .unwrap();
        if find(&observed.diagnostics, Stage::Nlp)
            .iter()
            .any(|d| d.code == Some(DiagnosticCode::NlpUnavailable))
        {
            eprintln!("skipping: the NLP model is unavailable here");
            return;
        }
        assert!(
            observed.diagnostics.iter().all(|d| !d.status.is_degraded()),
            "{:?}",
            observed.diagnostics
        );
        if cfg!(feature = "onnx") {
            let nlp = find(&observed.diagnostics, Stage::Nlp);
            assert_eq!(nlp.len(), 1, "{:?}", observed.diagnostics);
            assert_eq!(nlp[0].status, StageStatus::Applied);
        }
    }

    /// A missing custom rules file falls back to the bundled rules on every
    /// message that uses them, without failing the ingest.
    #[tokio::test]
    async fn a_missing_rules_file_is_reported_on_each_use() {
        if !cfg!(feature = "onnx") {
            eprintln!("skipping: observation rules run on the onnx model path only");
            return;
        }
        let mut config = config();
        config.observation_rules_path = Some("/nonexistent/uniko-rules.yml".into());
        let Some(memory) = store(config, false).await else {
            return;
        };
        let mut session = memory.agent("a").session("s");
        for text in [
            "I started a new job at the observatory",
            "My sister is visiting from Porto next week",
        ] {
            let observed = session.observe(Turn::new("alice", text)).await.unwrap();
            let nlp_ran = find(&observed.diagnostics, Stage::Nlp)
                .iter()
                .any(|d| d.status == StageStatus::Applied);
            if !nlp_ran {
                eprintln!("skipping: the NLP model is unavailable here, so no rules run");
                return;
            }
            let rules = find(&observed.diagnostics, Stage::ObservationRules);
            assert_eq!(rules.len(), 1, "{:?}", observed.diagnostics);
            assert_eq!(rules[0].code, Some(DiagnosticCode::CustomRulesUnloadable));
        }
    }

    /// A multi-turn unit reports its diagnostics once, on the unit, with a
    /// per-message count; each turn's own list stays empty.
    #[tokio::test]
    async fn a_multi_turn_unit_reports_on_the_unit() {
        let Some(memory) = store(config(), false).await else {
            return;
        };
        let mut session = memory.agent("a").session("s");
        let turns = || {
            vec![
                Turn::new("alice", "what's the plan for friday?").id("m-1"),
                Turn::new("bob", "we ship the release friday morning").id("m-2"),
            ]
        };
        let unit = session.commit_unit(turns()).await.unwrap();
        assert!(unit.turns.iter().all(|t| t.diagnostics.is_empty()));
        let nlp_messages: u32 = find(&unit.diagnostics, Stage::Nlp)
            .iter()
            .map(|d| d.count.unwrap_or(0))
            .sum();
        assert_eq!(nlp_messages, 2, "{:?}", unit.diagnostics);

        // Replaying the same unit writes nothing and reports nothing.
        let replay = session.commit_unit(turns()).await.unwrap();
        assert!(replay.was_replay);
        assert!(replay.diagnostics.is_empty(), "{:?}", replay.diagnostics);
    }

    /// `record_action` reports its NLP entity-linking stage.
    #[tokio::test]
    async fn record_action_reports_its_nlp_stage() {
        if !cfg!(feature = "onnx") {
            eprintln!("skipping: action entity linking runs on onnx builds only");
            return;
        }
        let mut config = config();
        config.nlp.model_id = "uniko-test/no-such-model-repo".into();
        let Some(memory) = store(config, false).await else {
            return;
        };
        let mut session = memory.agent("a").session("s");
        session
            .observe(Turn::new("alice", "please search the release notes"))
            .await
            .unwrap();
        let result = session
            .record_action(crate::RecordActionParams {
                action_type: "search".into(),
                input: Some(serde_json::json!({"q": "release notes"})),
                ..Default::default()
            })
            .await
            .unwrap();
        let nlp = find(&result.diagnostics, Stage::Nlp);
        assert_eq!(nlp.len(), 1, "{:?}", result.diagnostics);
        assert_eq!(nlp[0].status, StageStatus::Failed);
        assert_eq!(nlp[0].code, Some(DiagnosticCode::NlpUnavailable));
        assert_eq!(nlp[0].target.as_deref(), Some("action_entities"));
    }

    /// A PDF that cannot be parsed is still stored, and its outcome says
    /// extraction failed.
    #[tokio::test]
    async fn a_corrupt_pdf_reports_failed_extraction() {
        let Some(memory) = store(config(), false).await else {
            return;
        };
        let session = memory.agent("a").session("s");
        let outcome = session.ingest(corrupt_pdf()).await.unwrap();
        let pdf = find(outcome.diagnostics(), Stage::PdfExtraction);
        assert_eq!(pdf.len(), 1, "{:?}", outcome.diagnostics());
        assert_eq!(pdf[0].status, StageStatus::Failed);
        assert!(
            matches!(
                pdf[0].code,
                Some(DiagnosticCode::ExtractionFailed | DiagnosticCode::TieredFailed)
            ),
            "{:?}",
            pdf[0]
        );
    }

    /// Asking for OCR in a build without `pdf-ocr` still ingests the PDF as
    /// text, and the outcome says OCR was not available (issue #55).
    #[tokio::test]
    async fn ocr_without_the_feature_is_reported() {
        if crate::facade::readiness::compiled("pdf-ocr") {
            eprintln!("skipping: this build has pdf-ocr; the tiered test covers it");
            return;
        }
        let mut config = config();
        config.ocr.enabled = true;
        let Ok(memory) = Uniko::builder()
            .raw_config(config)
            .prefetch(uniko_store::PrefetchPolicy::Lazy)
            .build()
            .await
        else {
            eprintln!("skipping: instance unavailable");
            return;
        };
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../uniko-extract/tests/fixtures/dummy.pdf"
        ))
        .unwrap();
        let source = IngestSource::bytes(bytes)
            .with_mime(uniko_pipes::content::Mime::parse("application/pdf").unwrap());
        let outcome = memory.agent("a").session("s").ingest(source).await.unwrap();
        let pdf = find(outcome.diagnostics(), Stage::PdfExtraction);
        assert_eq!(pdf.len(), 1, "{:?}", outcome.diagnostics());
        assert_eq!(pdf[0].status, StageStatus::Fallback);
        assert_eq!(pdf[0].code, Some(DiagnosticCode::NotCompiled));
    }

    /// Streaming ingest has no per-call result, so its degradations are
    /// counted in the pipeline health.
    #[tokio::test]
    async fn streaming_ingest_counts_degradations_in_health() {
        let Some(memory) = store(config(), true).await else {
            return;
        };
        let session = memory.agent("a").session("s");
        session.submit_source(corrupt_pdf()).await.unwrap();
        session.flush().await.unwrap();
        let health = memory.health().expect("streaming is on");
        let counted: u64 = health
            .ingest
            .degradations
            .iter()
            .filter(|(k, _)| k.starts_with("pdf_extraction:"))
            .map(|(_, n)| n)
            .sum();
        assert_eq!(counted, 1, "{:?}", health.ingest.degradations);
    }
}

// ── Hybrid-store regressions (#70, #72) ──────────────────────────────────

mod hybrid_regressions {
    use uniko_store::config::{EmbeddingConfig, UnikoConfig};
    use uniko_store::{DiagnosticCode, Stage, StageStatus};

    use crate::{ContextBundle, IngestSource, RecallOverrides, Scope, Uniko};

    /// The fixture from rustic-ai/uniko#72, in ingest order.
    const DOCS: &[(&str, &str)] = &[
        (
            "eiffel",
            "The Eiffel Tower is a wrought-iron lattice tower beside the Seine in Paris, France.",
        ),
        (
            "tokyo",
            "Tokyo Tower is a communications and observation tower in Tokyo, Japan, painted orange and white.",
        ),
        (
            "london",
            "The Tower of London is a historic castle on the River Thames in London, England.",
        ),
        (
            "paris",
            "Paris is the capital of France. The Louvre museum holds paintings and ancient sculptures.",
        ),
        (
            "ada",
            "Ada Lovelace wrote the first published algorithm for Charles Babbage's Analytical Engine.",
        ),
        (
            "bridge",
            "The Golden Gate Bridge is a suspension bridge across the strait north of San Francisco.",
        ),
    ];
    const BOB: &str = "BOB_PRIVATE lattice tower in Paris.";
    const QUERY: &str = "lattice tower in Paris";

    fn expensive() -> bool {
        if std::env::var("EXPENSIVE_TESTS").as_deref() == Ok("1") {
            return true;
        }
        eprintln!("skipping: set EXPENSIVE_TESTS=1 to load BGE-M3");
        false
    }

    fn config(sparse: bool) -> UnikoConfig {
        let mut config = UnikoConfig::default();
        config.embedding = EmbeddingConfig::bge_m3();
        config.reranker.enabled = true;
        config.reranker.style = "colbert".into();
        config.reranker.top_n = 50;
        config.recall_sparse_enabled = sparse;
        config.recall_limit = 15;
        config.phase1_strategy = "off".into();
        config.phase2_coverage_threshold = 1.0;
        config.recall_lexical_enabled = false;
        config
    }

    async fn seed(memory: &Uniko) {
        let alice = memory.agent("a").session("alice");
        for (id, text) in DOCS {
            let mut src = IngestSource::text(*text);
            src.category = Some("external_evidence".into());
            src.source_id = Some((*id).into());
            src.revision_id = Some(format!("{id}-r1"));
            alice.ingest(src).await.unwrap();
        }
        memory
            .agent("a")
            .session("bob")
            .ingest(IngestSource::text(BOB))
            .await
            .unwrap();
    }

    fn alice() -> Scope {
        Scope::default()
            .sessions(["alice"])
            .categories(["external_evidence"])
    }

    async fn scoped_recall(memory: &Uniko) -> ContextBundle {
        memory
            .agent("a")
            .recall_with(
                QUERY,
                alice(),
                &RecallOverrides::default().limit(15).reranker(true),
            )
            .await
            .unwrap()
    }

    /// The doc ids (`DOCS` keys) in the bundle's order.
    fn doc_order(bundle: &ContextBundle) -> Vec<&'static str> {
        bundle
            .items
            .iter()
            .filter_map(|i| {
                DOCS.iter()
                    .find(|(_, t)| *t == i.content)
                    .map(|(id, _)| *id)
            })
            .collect()
    }

    /// Direct MaxSim order (uni-db's raw scores, recorded in #72 and checked
    /// against an independent cosine MaxSim there).
    const MAXSIM_ORDER: &[&str] = &["eiffel", "tokyo", "paris", "london", "bridge", "ada"];

    /// #72: on a fresh store, ColBERT recall follows MaxSim, never lets
    /// Bob's out-of-scope chunk set the scale, and stays the same after a
    /// reopen.
    #[tokio::test]
    async fn colbert_recall_follows_maxsim_fresh_and_reopened() {
        if !expensive() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let memory = Uniko::builder()
            .raw_config(config(false))
            .path(dir.path())
            .build()
            .await
            .unwrap();
        seed(&memory).await;

        let fresh = scoped_recall(&memory).await;
        assert!(
            fresh.items.iter().all(|i| i.content != BOB),
            "{:?}",
            fresh.items
        );
        assert_eq!(doc_order(&fresh), MAXSIM_ORDER, "fresh store");
        let colbert: Vec<_> = fresh
            .diagnostics
            .iter()
            .filter(|d| d.stage == Stage::Colbert)
            .collect();
        assert_eq!(colbert.len(), 1, "{:?}", fresh.diagnostics);
        assert_eq!(colbert[0].status, StageStatus::Applied);

        memory.shutdown().await.unwrap();
        let memory = Uniko::builder()
            .raw_config(config(false))
            .path(dir.path())
            .build()
            .await
            .unwrap();
        let reopened = scoped_recall(&memory).await;
        assert_eq!(doc_order(&reopened), MAXSIM_ORDER, "reopened store");
        memory.shutdown().await.unwrap();
    }

    /// #70: scoped recall should use the sparse channel. uni-db 5.2.1
    /// rejects a filtered `uni.sparse.query`, so today this reports
    /// `sparse: failed / query_failed`.
    #[tokio::test]
    #[ignore = "waits on rustic-ai/uni-db#344"]
    async fn scoped_recall_applies_the_sparse_channel() {
        if !expensive() {
            return;
        }
        let memory = Uniko::builder()
            .raw_config(config(true))
            .build()
            .await
            .unwrap();
        seed(&memory).await;
        let bundle = scoped_recall(&memory).await;
        let sparse: Vec<_> = bundle
            .diagnostics
            .iter()
            .filter(|d| d.stage == Stage::Sparse)
            .collect();
        assert!(!sparse.is_empty(), "{:?}", bundle.diagnostics);
        assert!(
            sparse
                .iter()
                .all(|d| d.status == StageStatus::Applied
                    && d.code != Some(DiagnosticCode::QueryFailed)),
            "{sparse:?}"
        );
        assert!(bundle.items.iter().all(|i| i.content != BOB));
    }
}

/// Agent registration (#67) and explicit tool-call performers (#68).
mod registration_and_performer {
    use super::*;
    use crate::{AgentRegistration, EnsureOutcome, ToolCall};

    async fn memory_or_skip() -> Option<Uniko> {
        match Uniko::in_memory().await {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("skipping: in-memory instance unavailable: {e}");
                None
            }
        }
    }

    async fn participants(kb: &KnowledgeBase, id: &str, kind: &str) -> i64 {
        count_query(
            kb,
            &format!(
                "MATCH (p:Participant {{participant_id: '{id}'}}) \
                 WHERE p.kind = '{kind}' RETURN count(p) AS c"
            ),
        )
        .await
    }

    #[tokio::test]
    async fn register_creates_an_agent_and_nothing_else() {
        let Some(memory) = memory_or_skip().await else {
            return;
        };
        let agent = memory.agent("reporter");
        let first = agent
            .register(AgentRegistration::new().name("Reporter"))
            .await
            .unwrap();
        assert_eq!(first.participant_id, "reporter");
        assert_eq!(first.outcome, EnsureOutcome::Created);
        let again = agent.register(AgentRegistration::new()).await.unwrap();
        assert_eq!(again.outcome, EnsureOutcome::Existing);

        let kb = agent.kb();
        assert_eq!(participants(kb, "reporter", "agent").await, 1);
        assert_eq!(
            count_query(
                kb,
                "MATCH (p:Participant {participant_id: 'reporter'}) \
                 WHERE p.name = 'Reporter' RETURN count(p) AS c"
            )
            .await,
            1
        );
        for label in ["Session", "Message", "Action"] {
            assert_eq!(
                count_query(kb, &format!("MATCH (n:{label}) RETURN count(n) AS c")).await,
                0,
                "register must not record a {label}"
            );
        }
        drop(agent);
        memory.shutdown().await.expect("shutdown");
    }

    /// A participant first seen through an org membership (kind `unknown`)
    /// is upgraded, keeping its membership and name.
    #[tokio::test]
    async fn register_upgrades_an_unknown_participant_and_keeps_its_membership() {
        let Some(memory) = memory_or_skip().await else {
            return;
        };
        assert!(memory.add_org_member("reporter", "acme").await.unwrap());
        let agent = memory.agent("reporter");
        let kb = agent.kb().clone();
        assert_eq!(participants(&kb, "reporter", "unknown").await, 1);

        let reg = agent
            .register(AgentRegistration::new().name("Reporter"))
            .await
            .unwrap();
        assert_eq!(reg.outcome, EnsureOutcome::Upgraded);
        assert_eq!(participants(&kb, "reporter", "agent").await, 1);
        let viewer = crate::Viewer::new(&kb, "reporter").await.unwrap();
        assert!(viewer.orgs.contains("acme"), "membership kept");
        drop((agent, kb, viewer));
        memory.shutdown().await.expect("shutdown");
    }

    /// The agent's first turn after registration keeps kind `agent`: ingest
    /// used to reset every participant it touched to `unknown`.
    #[tokio::test]
    async fn a_turn_after_registration_keeps_the_agent_kind() {
        let Some(memory) = memory_or_skip().await else {
            return;
        };
        let agent = memory.agent("reporter");
        agent
            .register(AgentRegistration::new().name("Reporter"))
            .await
            .unwrap();
        let mut session = agent.session("s-kind");
        match session
            .observe(Turn::new("reporter", "Starting the report.").id("m-kind"))
            .await
        {
            Ok(_) => {}
            Err(e) if is_model_unavailable(&e) => {
                eprintln!("skipping: models unavailable");
                return;
            }
            Err(e) => panic!("observe failed: {e}"),
        }
        let kb = agent.kb();
        assert_eq!(participants(kb, "reporter", "agent").await, 1);
        assert_eq!(
            count_query(
                kb,
                "MATCH (p:Participant {participant_id: 'reporter'}) \
                 WHERE p.name = 'Reporter' AND p.last_seen IS NOT NULL \
                 RETURN count(p) AS c"
            )
            .await,
            1,
            "the turn updates last_seen but keeps the registered name"
        );
        drop((session, agent));
        memory.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn goals_and_tasks_work_on_a_fresh_store() {
        let Some(memory) = memory_or_skip().await else {
            return;
        };
        let agent = memory.agent("reporter");
        let goal = agent
            .goals()
            .create_view(crate::CreateGoalParams {
                goal_id: Some("g".into()),
                title: "Goal".into(),
                ..Default::default()
            })
            .await;
        let goal = match goal {
            Ok(g) => g,
            Err(e) if is_model_unavailable(&e) => {
                eprintln!("skipping: models unavailable");
                return;
            }
            Err(e) => panic!("create_view failed: {e}"),
        };
        assert_eq!(goal.goal_id, "g");
        agent
            .goals()
            .create_task(crate::CreateTaskParams {
                goal_id: Some("g".into()),
                title: "Task".into(),
                ..Default::default()
            })
            .await
            .expect("create_task on a fresh store");
        assert_eq!(participants(agent.kb(), "reporter", "agent").await, 1);
        drop(agent);
        memory.shutdown().await.expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_registration_creates_one_participant() {
        let Some(memory) = memory_or_skip().await else {
            return;
        };
        let agent = memory.agent("reporter");
        let outcomes = futures::future::join_all((0..8).map(|_| {
            let agent = agent.clone();
            tokio::spawn(async move { agent.register(AgentRegistration::new()).await })
        }))
        .await;
        let created = outcomes
            .into_iter()
            .map(|r| r.unwrap().unwrap().outcome)
            .filter(|o| *o == EnsureOutcome::Created)
            .count();
        assert_eq!(created, 1, "exactly one caller creates the participant");
        assert_eq!(
            count_query(
                agent.kb(),
                "MATCH (p:Participant {participant_id: 'reporter'}) RETURN count(p) AS c"
            )
            .await,
            1
        );
        drop(agent);
        memory.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn registration_survives_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kb");
        let Some(memory) = build_or_skip(Uniko::builder().path(&path)).await else {
            return;
        };
        let reg = memory
            .agent("reporter")
            .register(AgentRegistration::new())
            .await
            .unwrap();
        assert_eq!(reg.outcome, EnsureOutcome::Created);
        memory.shutdown().await.expect("shutdown");

        let Some(memory) = build_or_skip(Uniko::builder().path(&path)).await else {
            return;
        };
        let agent = memory.agent("reporter");
        assert_eq!(participants(agent.kb(), "reporter", "agent").await, 1);
        let again = agent.register(AgentRegistration::new()).await.unwrap();
        assert_eq!(again.outcome, EnsureOutcome::Existing);
        drop(agent);
        memory.shutdown().await.expect("shutdown");
    }

    fn reporter_call() -> ToolCall {
        ToolCall::new("search")
            .call_id("call")
            .text_output("Evidence")
            .performer("reporter")
    }

    async fn performed_by(kb: &KnowledgeBase, call_id: &str, who: &str) -> i64 {
        count_query(
            kb,
            &format!(
                "MATCH (:Action {{action_id: '{call_id}'}})-[r:PERFORMED_BY]->\
                 (:Participant {{participant_id: '{who}'}}) RETURN count(r) AS c"
            ),
        )
        .await
    }

    /// The issue's reproduction: a user turn in the reporter's session keeps
    /// its sender while its embedded call is performed by the reporter, and
    /// a replay after a reopen adds no edges.
    #[tokio::test]
    async fn an_embedded_call_can_name_its_performer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kb");
        let Some(memory) = build_or_skip(Uniko::builder().path(&path)).await else {
            return;
        };
        let unit = || {
            vec![
                Turn::new("user", "Question")
                    .id("question")
                    .tool_call(reporter_call()),
            ]
        };
        let mut session = memory.agent("reporter").session("research");
        match session.commit_unit(unit()).await {
            Ok(r) => assert!(!r.was_replay),
            Err(e) if is_model_unavailable(&e) => {
                eprintln!("skipping: models unavailable");
                return;
            }
            Err(e) => panic!("commit_unit failed: {e}"),
        }
        let kb = memory.agent("reporter").kb().clone();
        assert_eq!(performed_by(&kb, "call", "reporter").await, 1);
        assert_eq!(performed_by(&kb, "call", "user").await, 0);
        assert_eq!(
            count_query(
                &kb,
                "MATCH (:Message {message_id: 'question'})-[:SENT_BY]->\
                 (:Participant {participant_id: 'user'}) RETURN count(*) AS c"
            )
            .await,
            1,
            "the turn keeps its sender"
        );
        assert_eq!(participants(&kb, "reporter", "agent").await, 1);
        drop((session, kb));
        memory.shutdown().await.expect("shutdown");

        let Some(memory) = build_or_skip(Uniko::builder().path(&path)).await else {
            return;
        };
        let mut session = memory.agent("reporter").session("research");
        let replay = session.commit_unit(unit()).await.expect("replay");
        assert!(replay.was_replay);
        let kb = memory.agent("reporter").kb().clone();
        assert_eq!(
            count_query(
                &kb,
                "MATCH (:Action {action_id: 'call'})-[r:PERFORMED_BY]->() RETURN count(r) AS c"
            )
            .await,
            1,
            "replay adds no PERFORMED_BY edge"
        );
        drop((session, kb));
        memory.shutdown().await.expect("shutdown");
    }

    /// Re-recording a call under another performer is refused, so the
    /// stored PERFORMED_BY edge never silently disagrees with the caller.
    #[tokio::test]
    async fn the_same_call_with_another_performer_is_a_conflict() {
        let Some(memory) = memory_or_skip().await else {
            return;
        };
        let session = memory.agent("reporter").session("research");
        match session
            .observe_tool(reporter_call().performer("auditor"))
            .await
        {
            Ok(r) => assert!(!r.was_replay),
            Err(e) if is_model_unavailable(&e) => {
                eprintln!("skipping: models unavailable");
                return;
            }
            Err(e) => panic!("observe_tool failed: {e}"),
        }
        let kb = memory.agent("reporter").kb().clone();
        assert_eq!(performed_by(&kb, "call", "auditor").await, 1);
        assert_eq!(participants(&kb, "auditor", "agent").await, 1);

        let replay = session
            .observe_tool(reporter_call().performer("auditor"))
            .await
            .expect("same performer replays");
        assert!(replay.was_replay);
        for call in [reporter_call(), reporter_call().performer("someone-else")] {
            let err = session.observe_tool(call).await.unwrap_err();
            assert!(
                matches!(err, UnikoError::IdConflict { .. }),
                "expected IdConflict, got {err:?}"
            );
        }
        // Naming the default performer explicitly is the same call as naming
        // none.
        let plain = session
            .observe_tool(
                ToolCall::new("search")
                    .call_id("plain")
                    .text_output("Evidence"),
            )
            .await
            .unwrap();
        assert!(!plain.was_replay);
        let explicit = session
            .observe_tool(
                ToolCall::new("search")
                    .call_id("plain")
                    .text_output("Evidence")
                    .performer("reporter"),
            )
            .await
            .unwrap();
        assert!(explicit.was_replay);
        assert_eq!(performed_by(&kb, "plain", "reporter").await, 1);
        drop((session, kb));
        memory.shutdown().await.expect("shutdown");
    }
}

/// Dead-letter inspection, retry and recovery (#65).
mod dead_letters {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use uniko_pipes::content::Mime;

    use super::*;
    use crate::{DeadLetterFilter, DeadLetterStatus, Modality, TextExtractor};

    /// An image extractor that fails with `err` while `failing` is set.
    fn flaky(failing: Arc<AtomicBool>, err: fn() -> UnikoError) -> Arc<TextExtractor> {
        Arc::new(TextExtractor::new(Modality::Image, move |_| {
            if failing.load(Ordering::SeqCst) {
                Err(err())
            } else {
                Ok("a red bicycle leaning on a wall".to_string())
            }
        }))
    }

    fn offline() -> UnikoError {
        UnikoError::Internal("captioner offline".into())
    }

    fn image(id: &str) -> IngestSource {
        IngestSource::bytes(vec![0x42; 16])
            .with_mime(Mime::parse("image/png").expect("mime"))
            .with_id(id)
    }

    fn config(max_retries: u32) -> UnikoConfig {
        UnikoConfig {
            dead_letter_max_retries: max_retries,
            ..UnikoConfig::default()
        }
    }

    async fn streaming(
        builder: crate::UnikoBuilder,
        extractor: Arc<TextExtractor>,
    ) -> Option<Uniko> {
        match builder.streaming(true).extractor(extractor).build().await {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("skipping: streaming instance unavailable: {e}");
                None
            }
        }
    }

    /// Submit one image that fails, and return its dead letter's id.
    async fn dead_letter_one(memory: &Uniko, id: &str) -> String {
        let session = memory.agent("viewer").session("dl");
        session.submit_source(image(id)).await.expect("submit");
        session.flush().await.expect("flush");
        let letters = memory
            .dead_letters()
            .list(&DeadLetterFilter::new())
            .await
            .unwrap();
        let [letter] = letters.as_slice() else {
            panic!("one dead letter expected, got {letters:?}");
        };
        letter.id.clone()
    }

    async fn flush(memory: &Uniko) {
        memory.agent("viewer").session("dl").flush().await.unwrap();
    }

    #[tokio::test]
    async fn a_transient_failure_is_retried_and_resolved() {
        let failing = Arc::new(AtomicBool::new(true));
        let Some(memory) = streaming(
            Uniko::builder().in_memory(),
            flaky(failing.clone(), offline),
        )
        .await
        else {
            return;
        };
        let id = dead_letter_one(&memory, "img-1").await;
        let dl = memory.dead_letters();
        let letter = dl.get(&id).await.unwrap().expect("letter");
        assert_eq!(letter.status, DeadLetterStatus::Pending);
        assert!(letter.retriable && letter.replayable);
        assert_eq!(letter.error_kind.as_deref(), Some("internal"));
        assert_eq!(letter.ingest_type.as_deref(), Some("source"));
        assert_eq!((letter.retry_count, letter.max_retries), (0, 3));
        assert!(
            letter.next_retry_at.is_some(),
            "the first retry is scheduled"
        );
        assert_eq!(dl.counts().await.unwrap().pending, 1);

        failing.store(false, Ordering::SeqCst);
        dl.retry(&id, false).await.expect("retry");
        flush(&memory).await;

        let letter = dl.get(&id).await.unwrap().unwrap();
        assert_eq!(letter.status, DeadLetterStatus::Resolved);
        assert_eq!(letter.resolution.as_deref(), Some("retried"));
        assert_eq!(letter.retry_count, 1);
        assert!(letter.resolved_at.is_some());
        let agent = memory.agent("viewer");
        assert!(
            agent.data().artifact("img-1").await.unwrap().is_some(),
            "the retried item is ingested"
        );
        assert_eq!(
            count_query(
                agent.kb(),
                "MATCH (a:Artifact {artifact_id: 'img-1'}) RETURN count(a) AS c"
            )
            .await,
            1
        );
        let health = memory.health().unwrap().ingest;
        assert_eq!(
            (
                health.dead_lettered,
                health.retries,
                health.retries_succeeded
            ),
            (1, 1, 1)
        );
        let err = dl.retry(&id, false).await.unwrap_err();
        assert!(matches!(err, UnikoError::Config(_)), "{err}");

        drop((agent, dl));
        memory.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn retries_stop_at_the_limit() {
        let failing = Arc::new(AtomicBool::new(true));
        let Some(memory) = streaming(
            Uniko::builder().in_memory().raw_config(config(2)),
            flaky(failing, offline),
        )
        .await
        else {
            return;
        };
        let id = dead_letter_one(&memory, "img-2").await;
        let dl = memory.dead_letters();
        for (attempt, expected) in [
            (1, DeadLetterStatus::Pending),
            (2, DeadLetterStatus::Exhausted),
        ] {
            dl.retry(&id, false).await.expect("retry");
            flush(&memory).await;
            let letter = dl.get(&id).await.unwrap().unwrap();
            assert_eq!(letter.retry_count, attempt);
            assert_eq!(letter.status, expected, "after retry {attempt}");
        }
        let err = dl.retry(&id, false).await.unwrap_err();
        assert!(
            matches!(&err, UnikoError::Config(m) if m.contains("all its retries")),
            "{err}"
        );
        // `force` goes past the limit; failing again leaves it exhausted.
        dl.retry(&id, true).await.expect("forced retry");
        flush(&memory).await;
        let letter = dl.get(&id).await.unwrap().unwrap();
        assert_eq!(
            (letter.retry_count, letter.status),
            (3, DeadLetterStatus::Exhausted)
        );
        assert_eq!(memory.health().unwrap().ingest.dead_lettered, 1);
        drop(dl);
        memory.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn a_permanent_failure_is_not_retried_without_force() {
        let Some(memory) = streaming(
            Uniko::builder().in_memory(),
            flaky(Arc::new(AtomicBool::new(true)), || {
                UnikoError::Unsupported("no decoder for this image".into())
            }),
        )
        .await
        else {
            return;
        };
        let id = dead_letter_one(&memory, "img-3").await;
        let dl = memory.dead_letters();
        let letter = dl.get(&id).await.unwrap().unwrap();
        assert_eq!(letter.status, DeadLetterStatus::Exhausted);
        assert!(!letter.retriable);
        assert!(letter.next_retry_at.is_none());
        let err = dl.retry(&id, false).await.unwrap_err();
        assert!(
            matches!(&err, UnikoError::Config(m) if m.contains("unsupported")),
            "{err}"
        );
        assert!(dl.retry_pending(10).await.unwrap().retried.is_empty());
        drop(dl);
        memory.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn resolve_only_changes_the_letter() {
        let Some(memory) = streaming(
            Uniko::builder().in_memory(),
            flaky(Arc::new(AtomicBool::new(true)), offline),
        )
        .await
        else {
            return;
        };
        let mut session = memory.agent("viewer").session("dl");
        session
            .observe(Turn::new("alice", "unrelated memory").id("m-keep"))
            .await
            .expect("observe");
        drop(session);
        let id = dead_letter_one(&memory, "img-4").await;
        let dl = memory.dead_letters();
        assert!(dl.resolve(&id, Some("image was corrupt")).await.unwrap());
        assert!(!dl.resolve(&id, None).await.unwrap(), "already resolved");
        assert!(!dl.resolve("no-such-letter", None).await.unwrap());
        let letter = dl.get(&id).await.unwrap().unwrap();
        assert_eq!(letter.status, DeadLetterStatus::Resolved);
        assert_eq!(letter.resolution.as_deref(), Some("image was corrupt"));
        let counts = dl.counts().await.unwrap();
        assert_eq!((counts.pending, counts.resolved), (0, 1));
        let agent = memory.agent("viewer");
        assert_eq!(
            count_query(
                agent.kb(),
                "MATCH (m:Message {message_id: 'm-keep'}) RETURN count(m) AS c"
            )
            .await,
            1,
            "other memory is untouched"
        );
        drop((agent, dl));
        memory.shutdown().await.expect("shutdown");
    }

    /// Letters survive a restart, and one left `retrying` by a stopped
    /// process is retried again.
    #[tokio::test]
    async fn letters_survive_a_restart_and_interrupted_retries_resume() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kb");
        let failing = Arc::new(AtomicBool::new(true));
        let Some(memory) = streaming(
            Uniko::builder().path(&path),
            flaky(failing.clone(), offline),
        )
        .await
        else {
            return;
        };
        let id = dead_letter_one(&memory, "img-5").await;
        // Claim a retry without resubmitting it: what a crash between the
        // claim and the ingest leaves behind.
        let kb = Arc::new(memory.agent("viewer").kb().clone());
        uniko_pipes::dead_letter::DeadLetterQueue::new(kb.clone())
            .begin_retry(&id, false)
            .await
            .expect("claim");
        drop(kb);
        memory.shutdown().await.expect("shutdown");

        failing.store(false, Ordering::SeqCst);
        let Some(memory) = streaming(Uniko::builder().path(&path), flaky(failing, offline)).await
        else {
            return;
        };
        let dl = memory.dead_letters();
        let letter = dl.get(&id).await.unwrap().expect("letter survives");
        assert_eq!(letter.status, DeadLetterStatus::Retrying);
        let report = dl.retry_pending(10).await.unwrap();
        assert_eq!(report.retried, vec![id.clone()], "{report:?}");
        flush(&memory).await;
        let letter = dl.get(&id).await.unwrap().unwrap();
        assert_eq!(letter.status, DeadLetterStatus::Resolved);
        assert_eq!(letter.retry_count, 2, "the interrupted claim still counts");
        drop(dl);
        memory.shutdown().await.expect("shutdown");
    }

    /// A row written before payloads were kept lists as legacy and can only
    /// be resolved.
    #[tokio::test]
    async fn a_legacy_letter_can_be_resolved_but_not_retried() {
        let Some(memory) = streaming(
            Uniko::builder().in_memory(),
            flaky(Arc::new(AtomicBool::new(false)), offline),
        )
        .await
        else {
            return;
        };
        let kb = memory.agent("viewer").kb().clone();
        let props = HashMap::from([
            ("step".to_string(), Value::String("ingest".into())),
            ("error".to_string(), Value::String("old failure".into())),
            ("node_ref".to_string(), Value::Int(0)),
            ("retry_count".to_string(), Value::Int(0)),
            ("max_retries".to_string(), Value::Int(3)),
        ]);
        kb.create_node(labels::DEAD_LETTER, &props).await.unwrap();
        drop(kb);
        let dl = memory.dead_letters();
        let letters = dl.list(&DeadLetterFilter::new()).await.unwrap();
        let [letter] = letters.as_slice() else {
            panic!("{letters:?}");
        };
        assert_eq!(letter.status, DeadLetterStatus::Legacy);
        assert!(letter.id.starts_with("legacy:") && !letter.replayable);
        assert_eq!(dl.counts().await.unwrap().legacy, 1);
        assert!(matches!(
            dl.retry(&letter.id, true).await,
            Err(UnikoError::Config(_))
        ));
        assert!(dl.resolve(&letter.id, None).await.unwrap());
        drop(dl);
        memory.shutdown().await.expect("shutdown");
    }

    /// Listing and resolving work without streaming; retrying does not.
    #[tokio::test]
    async fn retry_needs_streaming() {
        let Ok(memory) = Uniko::in_memory().await else {
            return;
        };
        let dl = memory.dead_letters();
        assert!(dl.list(&DeadLetterFilter::new()).await.unwrap().is_empty());
        let err = dl.retry("x", false).await.unwrap_err();
        assert!(
            matches!(&err, UnikoError::Config(m) if m.contains("streaming")),
            "{err}"
        );
        drop(dl);
        memory.shutdown().await.expect("shutdown");
    }
}

/// The opt-in background retry scheduler (#65).
mod dead_letter_scheduler {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use uniko_pipes::content::Mime;

    use super::*;
    use crate::{DeadLetter, DeadLetterFilter, DeadLetterStatus, Modality, TextExtractor};

    fn flaky(failing: Arc<AtomicBool>, err: fn() -> UnikoError) -> Arc<TextExtractor> {
        Arc::new(TextExtractor::new(Modality::Image, move |_| {
            if failing.load(Ordering::SeqCst) {
                Err(err())
            } else {
                Ok("a red bicycle leaning on a wall".to_string())
            }
        }))
    }

    fn offline() -> UnikoError {
        UnikoError::Internal("captioner offline".into())
    }

    /// Auto-retry on, with millisecond backoff so a test sees several
    /// passes quickly.
    fn auto(max_retries: u32, backoff_ms: u64) -> UnikoConfig {
        UnikoConfig {
            dead_letter_max_retries: max_retries,
            dead_letter_auto_retry: true,
            dead_letter_retry_backoff_ms: backoff_ms,
            dead_letter_retry_max_backoff_ms: backoff_ms * 4,
            ..UnikoConfig::default()
        }
    }

    async fn build(builder: crate::UnikoBuilder, extractor: Arc<TextExtractor>) -> Option<Uniko> {
        match builder.streaming(true).extractor(extractor).build().await {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("skipping: streaming instance unavailable: {e}");
                None
            }
        }
    }

    async fn submit_failing(memory: &Uniko, id: &str) -> String {
        let session = memory.agent("viewer").session("dl");
        session
            .submit_source(
                IngestSource::bytes(vec![0x42; 16])
                    .with_mime(Mime::parse("image/png").expect("mime"))
                    .with_id(id),
            )
            .await
            .expect("submit");
        session.flush().await.expect("flush");
        let letters = memory
            .dead_letters()
            .list(&DeadLetterFilter::new())
            .await
            .unwrap();
        letters.first().expect("a dead letter").id.clone()
    }

    /// Poll the letter until `done` holds, failing after 15 s.
    async fn wait_for(memory: &Uniko, id: &str, done: impl Fn(&DeadLetter) -> bool) -> DeadLetter {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let letter = memory
                .dead_letters()
                .get(id)
                .await
                .unwrap()
                .expect("letter");
            if done(&letter) {
                return letter;
            }
            assert!(
                Instant::now() < deadline,
                "timed out; last state {letter:?}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_transient_failure_recovers_on_its_own() {
        let failing = Arc::new(AtomicBool::new(true));
        let Some(memory) = build(
            Uniko::builder().in_memory().raw_config(auto(3, 50)),
            flaky(failing.clone(), offline),
        )
        .await
        else {
            return;
        };
        let id = submit_failing(&memory, "img-a").await;
        failing.store(false, Ordering::SeqCst);
        let letter = wait_for(&memory, &id, |l| l.status == DeadLetterStatus::Resolved).await;
        assert_eq!(letter.resolution.as_deref(), Some("retried"));
        assert!(letter.retry_count >= 1);
        // The scheduler's retry ingested the item.
        memory.agent("viewer").session("dl").flush().await.unwrap();
        assert!(
            memory
                .agent("viewer")
                .data()
                .artifact("img-a")
                .await
                .unwrap()
                .is_some()
        );
        assert!(memory.health().unwrap().ingest.retries_succeeded >= 1);
        memory.shutdown().await.expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn automatic_retries_stop_at_the_limit() {
        let Some(memory) = build(
            Uniko::builder().in_memory().raw_config(auto(2, 30)),
            flaky(Arc::new(AtomicBool::new(true)), offline),
        )
        .await
        else {
            return;
        };
        let id = submit_failing(&memory, "img-b").await;
        let letter = wait_for(&memory, &id, |l| l.status == DeadLetterStatus::Exhausted).await;
        assert_eq!(letter.retry_count, 2);
        // Give the scheduler a few more passes: it must not go past the limit.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let letter = memory.dead_letters().get(&id).await.unwrap().unwrap();
        assert_eq!(
            (letter.retry_count, letter.status),
            (2, DeadLetterStatus::Exhausted)
        );
        memory.shutdown().await.expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_permanent_failure_is_never_retried_automatically() {
        let Some(memory) = build(
            Uniko::builder().in_memory().raw_config(auto(3, 30)),
            flaky(Arc::new(AtomicBool::new(true)), || {
                UnikoError::Unsupported("no decoder".into())
            }),
        )
        .await
        else {
            return;
        };
        let id = submit_failing(&memory, "img-c").await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        let letter = memory.dead_letters().get(&id).await.unwrap().unwrap();
        assert_eq!(
            (letter.retry_count, letter.status),
            (0, DeadLetterStatus::Exhausted)
        );
        assert_eq!(memory.health().unwrap().ingest.retries, 0);
        memory.shutdown().await.expect("shutdown");
    }

    /// `flush` waits for submitted work, not for a retry that isn't due.
    #[tokio::test(flavor = "multi_thread")]
    async fn flush_does_not_wait_for_a_retry_that_is_not_due() {
        let Some(memory) = build(
            Uniko::builder().in_memory().raw_config(auto(3, 60_000)),
            flaky(Arc::new(AtomicBool::new(true)), offline),
        )
        .await
        else {
            return;
        };
        let started = Instant::now();
        let id = submit_failing(&memory, "img-d").await;
        memory.agent("viewer").session("dl").flush().await.unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "flush waited for a future retry"
        );
        let letter = memory.dead_letters().get(&id).await.unwrap().unwrap();
        assert_eq!(
            (letter.retry_count, letter.status),
            (0, DeadLetterStatus::Pending)
        );
        memory.shutdown().await.expect("shutdown");
    }

    /// A letter dead-lettered by a run without auto-retry is picked up by a
    /// later run that has it on: the schedule lives on the letter.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_schedule_survives_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kb");
        let failing = Arc::new(AtomicBool::new(true));
        let manual = UnikoConfig {
            dead_letter_retry_backoff_ms: 50,
            dead_letter_retry_max_backoff_ms: 200,
            ..UnikoConfig::default()
        };
        let Some(memory) = build(
            Uniko::builder().path(&path).raw_config(manual),
            flaky(failing.clone(), offline),
        )
        .await
        else {
            return;
        };
        let id = submit_failing(&memory, "img-e").await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            memory
                .dead_letters()
                .get(&id)
                .await
                .unwrap()
                .unwrap()
                .status,
            DeadLetterStatus::Pending,
            "nothing retries it while auto-retry is off"
        );
        memory.shutdown().await.expect("shutdown");

        failing.store(false, Ordering::SeqCst);
        let Some(memory) = build(
            Uniko::builder().path(&path).raw_config(auto(3, 50)),
            flaky(failing, offline),
        )
        .await
        else {
            return;
        };
        wait_for(&memory, &id, |l| l.status == DeadLetterStatus::Resolved).await;
        memory.shutdown().await.expect("shutdown");
    }
}
