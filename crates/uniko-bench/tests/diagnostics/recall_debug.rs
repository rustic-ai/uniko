//! Debug retrieval ranking for missed single-hop questions.
//!
//! Run: cargo nextest run -p uniko-bench --test diagnostics recall_debug --nocapture --run-ignored all

use crate::common::load_kb;
use uniko_memory::recall::{RecallConfig, recall};

/// libtest gives each test thread 2 MiB of stack. The recall cascade composes
/// deeply nested async state machines, and in the *dev* profile — no inlining,
/// no state-machine layout optimization — a single `recall()` call overflows
/// that and aborts the whole process with `fatal runtime error: stack
/// overflow`, which reads as a crash rather than a test failure. The release
/// build is unaffected (the bench drives the same path over 105 questions), so
/// this is a debug-build stack cost, not runaway recursion.
///
/// Run the body on a thread with an explicit 32 MiB stack — a virtual
/// reservation, not committed memory. Kept here rather than in
/// `.config/nextest.toml` so the requirement travels with the test: this
/// nextest (0.9.124) ignores an `[env]` table, and a `RUST_MIN_STACK` that
/// only lives in runner config silently stops applying if the test is run any
/// other way.
#[test]
#[ignore]
fn debug_chunk_existence() {
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime")
                .block_on(debug_chunk_existence_inner());
        })
        .expect("spawn big-stack thread")
        .join()
        .expect("diagnostic thread panicked");
}

async fn debug_chunk_existence_inner() {
    let kb = load_kb("data/kb/conv-30").await;
    let session = kb.db().session();

    // Count nodes by type
    for label in &[
        "Message",
        "Chunk",
        "Observation",
        "Session",
        "Entity",
        "Participant",
    ] {
        let cypher = format!("MATCH (m:{label}) RETURN count(m) AS cnt");
        let result = session.query_with(&cypher).fetch_all().await.unwrap();
        let cnt: i64 = result
            .rows()
            .first()
            .and_then(|r| r.get("cnt").ok())
            .unwrap_or(0);
        eprintln!("{label:15}: {cnt} nodes");
    }

    // Check chunk types
    for ct in &["session", "observation", "text", ""] {
        let cypher = if ct.is_empty() {
            "MATCH (c:Chunk) WHERE c.chunk_type IS NULL RETURN count(c) AS cnt".to_string()
        } else {
            format!("MATCH (c:Chunk {{chunk_type: '{ct}'}}) RETURN count(c) AS cnt")
        };
        let result = session.query_with(&cypher).fetch_all().await.unwrap();
        let cnt: i64 = result
            .rows()
            .first()
            .and_then(|r| r.get("cnt").ok())
            .unwrap_or(0);
        let label = if ct.is_empty() { "null" } else { ct };
        eprintln!("  chunk_type={label}: {cnt}");
    }

    // Check if chunks have embeddings
    let result = session
        .query_with("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS cnt")
        .fetch_all()
        .await
        .unwrap();
    let with_embed: i64 = result
        .rows()
        .first()
        .and_then(|r| r.get("cnt").ok())
        .unwrap_or(0);

    let result = session
        .query_with("MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c) AS cnt")
        .fetch_all()
        .await
        .unwrap();
    let without_embed: i64 = result
        .rows()
        .first()
        .and_then(|r| r.get("cnt").ok())
        .unwrap_or(0);
    eprintln!("\nChunks with embedding: {with_embed}");
    eprintln!("Chunks without embedding: {without_embed}");

    // Show a sample chunk
    let result = session
        .query_with("MATCH (c:Chunk {chunk_type: 'session'}) RETURN c.text AS text, c.chunk_type AS ct LIMIT 1")
        .fetch_all()
        .await
        .unwrap();
    if let Some(row) = result.rows().first() {
        let text: String = row.get("text").unwrap_or_default();
        eprintln!(
            "\nSample session chunk ({} chars): {}...",
            text.len(),
            &text[..text.len().min(200)]
        );
    }

    // Check participants
    let result = session
        .query_with("MATCH (p:Participant) RETURN p.name AS name")
        .fetch_all()
        .await
        .unwrap();
    eprintln!("\nParticipants:");
    for row in result.rows() {
        let name: String = row.get("name").unwrap_or_default();
        eprintln!("  {name}");
    }

    // Check entity names
    let result = session
        .query_with("MATCH (e:Entity) RETURN e.name AS name, e.entity_type AS et LIMIT 10")
        .fetch_all()
        .await
        .unwrap();
    eprintln!("\nEntities (first 10):");
    for row in result.rows() {
        let name: String = row.get("name").unwrap_or_default();
        let et: String = row.get("et").unwrap_or_default();
        eprintln!("  [{et}] {name}");
    }

    // Check ABOUT edges from Chunks
    let result = session.query_with("MATCH (c:Chunk)-[:ABOUT]->(e:Entity) RETURN c.chunk_type AS ct, e.name AS ename LIMIT 10").fetch_all().await;
    match result {
        Ok(r) => {
            eprintln!("\nChunk ABOUT edges ({}):", r.len());
            for row in r.rows() {
                let ct: String = row.get("ct").unwrap_or_default();
                let ename: String = row.get("ename").unwrap_or_default();
                eprintln!("  {ct} -> {ename}");
            }
        }
        Err(e) => eprintln!("\nChunk ABOUT query failed: {e}"),
    }

    // Check IN_SESSION edges
    let result = session
        .query_with("MATCH (m:Message)-[:IN_SESSION]->(s) RETURN labels(s)[0] AS slbl LIMIT 3")
        .fetch_all()
        .await;
    match result {
        Ok(r) => {
            eprintln!("\nIN_SESSION target labels:");
            for row in r.rows() {
                let slbl: String = row.get("slbl").unwrap_or_default();
                eprintln!("  {slbl}");
            }
            if r.is_empty() {
                eprintln!("  (no IN_SESSION edges found)");
            }
        }
        Err(e) => eprintln!("\nIN_SESSION query failed: {e}"),
    }

    // Check PARTICIPATED_IN edges
    let result = session.query_with("MATCH (p:Participant)-[:PARTICIPATED_IN]->(s) RETURN p.name AS pname, labels(s)[0] AS slbl LIMIT 5").fetch_all().await;
    match result {
        Ok(r) => {
            eprintln!("\nPARTICIPATED_IN edges ({}):", r.len());
            for row in r.rows() {
                let pname: String = row.get("pname").unwrap_or_default();
                let slbl: String = row.get("slbl").unwrap_or_default();
                eprintln!("  {pname} -> {slbl}");
            }
        }
        Err(e) => eprintln!("\nPARTICIPATED_IN query failed: {e}"),
    }

    // Check HAS_CHUNK edges
    let result = session.query_with("MATCH (s)-[:HAS_CHUNK]->(c:Chunk) RETURN labels(s)[0] AS slbl, c.chunk_type AS ct LIMIT 5").fetch_all().await;
    match result {
        Ok(r) => {
            eprintln!("\nHAS_CHUNK edges ({}):", r.len());
            for row in r.rows() {
                let slbl: String = row.get("slbl").unwrap_or_default();
                let ct: String = row.get("ct").unwrap_or_default();
                eprintln!("  {slbl} -> Chunk({ct})");
            }
        }
        Err(e) => eprintln!("\nHAS_CHUNK query failed: {e}"),
    }

    // Check: where do HAS_CHUNK edges come from?
    let result = session
        .query_with("MATCH ()-[e:HAS_CHUNK]->() RETURN count(e) AS cnt")
        .fetch_all()
        .await
        .unwrap();
    let hc_cnt: i64 = result
        .rows()
        .first()
        .and_then(|r| r.get("cnt").ok())
        .unwrap_or(0);
    eprintln!("\nTotal HAS_CHUNK edges (any direction): {hc_cnt}");

    // Check session node details
    let result = session
        .query_with(
            "MATCH (m:Message)-[:IN_SESSION]->(s) RETURN id(s) AS sid, labels(s) AS lbls LIMIT 3",
        )
        .fetch_all()
        .await;
    match result {
        Ok(r) => {
            eprintln!("\nIN_SESSION targets (with id and labels):");
            for row in r.rows() {
                let sid: i64 = row.get("sid").unwrap_or(0);
                let lbls: String = row.get::<String>("lbls").unwrap_or_else(|_| "?".into());
                eprintln!("  id={sid} labels={lbls}");
            }
        }
        Err(e) => eprintln!("\nIN_SESSION detail query failed: {e}"),
    }

    // Check: do any nodes have session_id property?
    let result = session.query_with("MATCH (n) WHERE n.session_id IS NOT NULL RETURN labels(n)[0] AS lbl, n.session_id AS sid LIMIT 5").fetch_all().await;
    match result {
        Ok(r) => {
            eprintln!("\nNodes with session_id property:");
            for row in r.rows() {
                let lbl: String = row.get("lbl").unwrap_or_default();
                let sid: String = row.get("sid").unwrap_or_default();
                eprintln!("  [{lbl}] session_id={sid}");
            }
        }
        Err(e) => eprintln!("\nsession_id query failed: {e}"),
    }

    // Test BM25 on Chunk.text with a raw string
    let test_queries = &[
        "favorite style of dance",
        "contemporary",
        "dance",
        "lost job banker",
        "store doing great",
    ];
    for q in test_queries {
        let result = session
            .query_with("MATCH (c:Chunk) RETURN c.text AS text, similar_to(c.text, $q) AS score ORDER BY score DESC LIMIT 3")
            .param("q", *q)
            .fetch_all()
            .await;
        match result {
            Ok(r) => {
                let top_score: f64 = r
                    .rows()
                    .first()
                    .and_then(|row| row.get("score").ok())
                    .unwrap_or(0.0);
                eprintln!(
                    "BM25 Chunk '{q}': top_score={top_score:.4} ({} results)",
                    r.len()
                );
            }
            Err(e) => eprintln!("BM25 Chunk '{q}': ERROR {e}"),
        }
    }

    // Test BM25 on Message.content with same queries for comparison
    eprintln!();
    for q in test_queries {
        let result = session
            .query_with("MATCH (m:Message) RETURN m.content AS text, similar_to(m.content, $q) AS score ORDER BY score DESC LIMIT 3")
            .param("q", *q)
            .fetch_all()
            .await;
        match result {
            Ok(r) => {
                let top_score: f64 = r
                    .rows()
                    .first()
                    .and_then(|row| row.get("score").ok())
                    .unwrap_or(0.0);
                eprintln!(
                    "BM25 Message '{q}': top_score={top_score:.4} ({} results)",
                    r.len()
                );
            }
            Err(e) => eprintln!("BM25 Message '{q}': ERROR {e}"),
        }
    }

    // Test vector search on Chunk separately
    eprintln!();
    let intent = uniko_memory::recall::build_intent(&kb, "favorite style of dance", &[])
        .await
        .unwrap();
    eprintln!("Intent keywords: '{}'", intent.keywords());
    eprintln!("Intent vec len: {}", intent.intent_vec().len());

    if !intent.intent_vec().is_empty() {
        let result = session
            .query_with("MATCH (c:Chunk) RETURN c.text AS text, similar_to(c.embedding, $qvec) AS score ORDER BY score DESC LIMIT 3")
            .param("qvec", uni_db::Value::Vector(intent.intent_vec().to_vec()))
            .fetch_all()
            .await;
        match result {
            Ok(r) => {
                eprintln!("\nVector search on Chunk.embedding for 'favorite style of dance':");
                for row in r.rows() {
                    let text: String = row.get("text").unwrap_or_default();
                    let score: f64 = row.get("score").unwrap_or(0.0);
                    eprintln!("  score={score:.4} | {}...", &text[..text.len().min(100)]);
                }
            }
            Err(e) => eprintln!("Vector Chunk: ERROR {e}"),
        }
    }

    // Test full recall for a missed question
    let config = RecallConfig {
        limit: 15,
        ..Default::default()
    };
    let bundle = recall(&kb, "What is Gina's favorite style of dance?", &config)
        .await
        .unwrap();
    eprintln!("\nFull recall for 'What is Gina's favorite style of dance?':");
    for (i, item) in bundle.items.iter().enumerate() {
        let preview = item.content.replace('\n', " ");
        eprintln!(
            "  #{:2} [{:12}] score={:.4} | {}...",
            i + 1,
            format!("{:?}", item.kind),
            item.score,
            &preview[..preview.len().min(100)]
        );
    }

    // Test hybrid similar_to on Chunk (same as recall does)
    if !intent.intent_vec().is_empty() {
        let result = session
            .query_with(
                "MATCH (m:Chunk) \
                 RETURN id(m) AS nid, m.text AS text, \
                        similar_to([m.embedding, m.text], [$qvec, $qtxt], {method: 'weighted', weights: [0.5, 0.5]}) AS score \
                 ORDER BY score DESC LIMIT 5"
            )
            .param("qvec", uni_db::Value::Vector(intent.intent_vec().to_vec()))
            .param("qtxt", intent.keywords())
            .fetch_all()
            .await;
        match result {
            Ok(r) => {
                eprintln!("\nHybrid similar_to on Chunk (vec+bm25):");
                for row in r.rows() {
                    let text: String = row.get("text").unwrap_or_default();
                    let score: f64 = row.get("score").unwrap_or(0.0);
                    eprintln!("  score={score:.4} | {}...", &text[..text.len().min(100)]);
                }
            }
            Err(e) => eprintln!("Hybrid Chunk: ERROR {e}"),
        }
    }
}
