//! Smoke test: ingest 5 messages, run recall, verify non-empty results.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;

use uni_db::ModelAliasSpec;
use uniko_extract::ingest::atomic::ingest_message_atomic;
use uniko_extract::ingest::context::SessionContext;
use uniko_memory::recall::{RecallConfig, recall};
use uniko_pipes::types::IngestMessage;
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

async fn make_kb() -> Arc<KnowledgeBase> {
    let config = UnikoConfig::default();
    let kb = KnowledgeBase::in_memory_with_xervo(config, Vec::<ModelAliasSpec>::new())
        .await
        .unwrap();
    Arc::new(kb)
}

async fn ingest_and_extract(
    kb: &Arc<KnowledgeBase>,
    msg: IngestMessage,
    session_ctx: &mut SessionContext,
) {
    let result = ingest_message_atomic(kb, &msg, session_ctx).await.unwrap();
    eprintln!(
        "  entities={}, observations={}",
        result.extracted_entities.len(),
        result.extracted_observations.len()
    );
}

fn msg(id: &str, content: &str, sender: &str) -> IngestMessage {
    IngestMessage {
        message_id: id.into(),
        content: content.into(),
        content_type: "text".into(),
        sender_id: sender.into(),
        session_id: "s-1".into(),
        addressed_to: Some(vec![if sender == "Jon" {
            "Gina".into()
        } else {
            "Jon".into()
        }]),
        timestamp: Utc::now(),
        metadata: HashMap::new(),
        category: None,
        source_id: None,
        revision_id: None,
    }
}

/// Stack for the test thread.
///
/// The default is 2 MiB — libtest runs each test on a spawned thread, not the
/// 8 MiB main thread — and an ingest-plus-recall pass sits right at that
/// limit: it passes at 2 MiB and overflows deterministically at 1 MiB. Under
/// full-suite load a slightly deeper call chain tipped it over in roughly
/// half of runs, aborting the process with SIGABRT rather than failing an
/// assertion, which silently invalidated whatever else that run reported.
///
/// The depth is inside the store's query execution (recursive query planning),
/// not in uniko's own frames: boxing the large ingest/recall futures moved the
/// 1 MiB threshold not at all. So this asks for headroom rather than
/// pretending the requirement is smaller than it is.
const TEST_STACK_BYTES: usize = 16 * 1024 * 1024;

#[test]
fn test_ingest_and_recall() {
    std::thread::Builder::new()
        .stack_size(TEST_STACK_BYTES)
        .name("test_ingest_and_recall".into())
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime")
                .block_on(ingest_and_recall_body());
        })
        .expect("spawn test thread")
        .join()
        .expect("test thread panicked");
}

async fn ingest_and_recall_body() {
    let kb = make_kb().await;

    eprintln!("--- Ingesting messages ---");
    let mut session_ctx = SessionContext::new("s-1".into(), 0);
    ingest_and_extract(
        &kb,
        msg("m1", "Hey Gina! Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business.", "Jon"),
        &mut session_ctx,
    ).await;
    ingest_and_extract(
        &kb,
        msg("m2", "Sorry about your job Jon, but starting your own business sounds awesome! I also lost my job at Door Dash last week.", "Gina"),
        &mut session_ctx,
    ).await;
    ingest_and_extract(
        &kb,
        msg(
            "m3",
            "We both like to destress by dancing, so let's go dancing this weekend!",
            "Jon",
        ),
        &mut session_ctx,
    )
    .await;

    eprintln!("\n--- Running recall ---");
    let config = RecallConfig {
        limit: 10,
        token_budget: 4096,
        min_score: 0.001,
        vector_weight: 0.5,
        bm25_weight: 0.5,
        ..Default::default()
    };

    let bundle = recall(&kb, "When did Jon lose his job?", &config)
        .await
        .unwrap();
    eprintln!(
        "Recall items: {}, coverage: {:.3}",
        bundle.items.len(),
        bundle.coverage
    );
    for item in &bundle.items {
        let kind = format!("{:?}", item.kind);
        eprintln!(
            "  [{:.3}] {kind} (tier {}): content_len={} '{}'",
            item.score,
            item.kind.tier() as u8,
            item.content.len(),
            &item.content[..item.content.len().min(100)]
        );
    }

    // Debug: check similar_to scores directly
    // Flush to ensure indexes are up to date
    kb.db().flush().await.unwrap();

    eprintln!("\n--- similar_to fulltext scores ---");
    let session = kb.db().session();
    let result = session
        .query_with(
            "MATCH (m:Message) \
             RETURN m.content AS content, \
                    similar_to(m.content, $q) AS fts_score \
             ORDER BY fts_score DESC LIMIT 3",
        )
        .param("q", "job banker")
        .fetch_all()
        .await
        .unwrap();
    for row in result.rows() {
        let content: String = row.get("content").unwrap_or_default();
        let score: f64 = row.get("fts_score").unwrap_or(-1.0);
        eprintln!(
            "  score={score:.4} content={}",
            &content[..content.len().min(80)]
        );
    }

    // Debug: try CALL uni.fts.query directly
    eprintln!("\n--- CALL uni.fts.query directly ---");
    let result = session
        .query_with(
            "CALL uni.fts.query('Message', 'content', $q, 3) \
             YIELD node, score \
             RETURN node.content AS content, score",
        )
        .param("q", "job banker")
        .fetch_all()
        .await
        .unwrap();
    for row in result.rows() {
        let content: String = row.get("content").unwrap_or_default();
        let score: f64 = row.get("score").unwrap_or(-1.0);
        eprintln!(
            "  fts_score={score:.4} content={}",
            &content[..content.len().min(80)]
        );
    }

    // Debug: check what properties search results contain
    eprintln!("\n--- Search result properties ---");
    let ft_results = kb
        .fulltext_search("Jon banker job", "Message", "content", 1)
        .await
        .unwrap();
    if let Some(r) = ft_results.first() {
        for (k, v) in &r.properties {
            let v_str = format!("{v:?}");
            eprintln!("  {k} = {}", &v_str[..v_str.len().min(80)]);
        }
    }

    // Debug: check what's in the graph
    eprintln!("\n--- Graph state ---");
    let session = kb.db().session();
    let result = session
        .query("MATCH (m:Message) RETURN m.message_id, m.content, m.embedding IS NOT NULL AS has_embed")
        .await
        .unwrap();
    for row in result.rows() {
        eprintln!(
            "  msg={}, has_embed={}, content={}",
            row.get::<String>("m.message_id").unwrap_or_default(),
            row.get::<bool>("has_embed").unwrap_or(false),
            &row.get::<String>("m.content").unwrap_or_default()
                [..50.min(row.get::<String>("m.content").unwrap_or_default().len())],
        );
    }

    // Try direct fulltext search
    eprintln!("\n--- Direct fulltext search ---");
    match kb
        .fulltext_search("Jon banker job", "Message", "content", 5)
        .await
    {
        Ok(results) => {
            eprintln!("  fulltext results: {}", results.len());
            for r in &results {
                eprintln!("    [{:.3}] {}", r.score, r.node_type);
            }
        }
        Err(e) => eprintln!("  fulltext error: {e}"),
    }

    // Try direct vector search
    eprintln!("\n--- Direct vector search ---");
    let intent_vec = uniko_extract::embedding::embed_query(&kb, "When did Jon lose his job?")
        .await
        .unwrap_or_default();
    eprintln!("  intent_vec len: {}", intent_vec.len());
    if !intent_vec.is_empty() {
        match kb
            .vector_search(&intent_vec, "Message", "embedding", 5)
            .await
        {
            Ok(results) => {
                eprintln!("  vector results: {}", results.len());
                for r in &results {
                    eprintln!("    [{:.3}] {}", r.score, r.node_type);
                }
            }
            Err(e) => eprintln!("  vector error: {e}"),
        }
    }

    assert!(
        !bundle.items.is_empty(),
        "Recall should return at least one item for 'When did Jon lose his job?'"
    );
}
