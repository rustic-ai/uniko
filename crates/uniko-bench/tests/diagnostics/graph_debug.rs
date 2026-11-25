//! Diagnose broken graph structure in conv-30 KB.
//!
//! Run: cargo nextest run -p uniko-bench --test integration diagnostics::graph_debug --nocapture --run-ignored all

use crate::common::{load_kb, load_kb_no_schema};
use uniko_store::KnowledgeBase;

#[tokio::test]
#[ignore]
async fn diagnose_session_nodes() {
    eprintln!("\n=== With schema.json ===");
    let kb = load_kb("data/kb/conv-30").await;
    run_diagnostics(&kb).await;

    drop(kb);

    eprintln!("\n=== Without schema.json (programmatic schema) ===");
    let kb2 = load_kb_no_schema("data/kb/conv-30").await;
    run_diagnostics(&kb2).await;
}

async fn run_diagnostics(kb: &KnowledgeBase) {
    let s = kb.db().session();

    // 1. What is node id=1?
    match kb.get_node(1).await {
        Ok(Some((label, props))) => {
            eprintln!("Node id=1: label={label}");
            for (k, v) in &props {
                let vs = format!("{v:?}");
                eprintln!("  {k} = {}", &vs[..vs.len().min(80)]);
            }
        }
        Ok(None) => eprintln!("Node id=1: NOT FOUND"),
        Err(e) => eprintln!("Node id=1: ERROR {e}"),
    }

    // 2. Count by label
    for label in &[
        "Session",
        "Message",
        "Chunk",
        "Observation",
        "Participant",
        "Entity",
    ] {
        let q = format!("MATCH (n:{label}) RETURN count(n) AS cnt");
        let cnt: i64 = s
            .query_with(&q)
            .fetch_all()
            .await
            .ok()
            .and_then(|r| r.rows().first().and_then(|row| row.get("cnt").ok()))
            .unwrap_or(-1);
        eprintln!("{label:15}: {cnt}");
    }

    // 3. Try get_node_by_ext_id for sessions
    for sid in &["session_1", "session_2", "D1"] {
        match kb.get_node_by_ext_id("Session", "session_id", sid).await {
            Ok(Some((nid, _))) => {
                eprintln!("get_node_by_ext_id(Session, session_id, {sid}) = {nid}")
            }
            Ok(None) => eprintln!("get_node_by_ext_id(Session, session_id, {sid}) = NOT FOUND"),
            Err(e) => eprintln!("get_node_by_ext_id(Session, session_id, {sid}) = ERROR: {e}"),
        }
    }

    // 4. Find nodes with session_id property using raw Cypher
    let r = s.query_with("MATCH (n) WHERE n.session_id IS NOT NULL RETURN id(n) AS nid, labels(n)[0] AS lbl, n.session_id AS sid LIMIT 5")
        .fetch_all().await;
    match r {
        Ok(result) => {
            eprintln!("Nodes with session_id ({}):", result.len());
            for row in result.rows() {
                let nid: i64 = row.get("nid").unwrap_or(0);
                let lbl: String = row.get("lbl").unwrap_or_default();
                let sid: String = row.get("sid").unwrap_or_default();
                eprintln!("  id={nid} [{lbl}] session_id={sid}");
            }
        }
        Err(e) => eprintln!("session_id query error: {e}"),
    }

    // 5. Count all edges by type
    for etype in &[
        "IN_SESSION",
        "HAS_CHUNK",
        "PARTICIPATED_IN",
        "SENT_BY",
        "OBSERVED_IN",
        "ABOUT",
    ] {
        let q = format!("MATCH ()-[e:{etype}]->() RETURN count(e) AS cnt");
        let cnt: i64 = s
            .query_with(&q)
            .fetch_all()
            .await
            .ok()
            .and_then(|r| r.rows().first().and_then(|row| row.get("cnt").ok()))
            .unwrap_or(-1);
        eprintln!("{etype:20}: {cnt} edges");
    }

    // 6. Trace one IN_SESSION edge to see what the target looks like
    let r = s.query_with("MATCH (m:Message)-[e:IN_SESSION]->(s) RETURN id(m) AS mid, id(s) AS sid, labels(s)[0] AS slbl LIMIT 1")
        .fetch_all().await;
    match r {
        Ok(result) => {
            if let Some(row) = result.rows().first() {
                let mid: i64 = row.get("mid").unwrap_or(0);
                let sid: i64 = row.get("sid").unwrap_or(0);
                let slbl: String = row.get("slbl").unwrap_or_default();
                eprintln!("Sample IN_SESSION: Message({mid}) -> node({sid}) label={slbl}");
            }
        }
        Err(e) => eprintln!("IN_SESSION trace error: {e}"),
    }
}
