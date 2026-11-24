//! Shutting a store down when its final flush fails.

use std::collections::HashMap;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use uni_db::Value;
use uniko_store::PrefetchPolicy;
use uniko_store::UnikoError;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;

async fn open(path: &Path) -> KnowledgeBase {
    KnowledgeBase::open_with_prefetch(
        path,
        UnikoConfig::default(),
        Vec::new(),
        PrefetchPolicy::Lazy,
    )
    .await
    .expect("open")
}

async fn participants(kb: &KnowledgeBase) -> i64 {
    let rows = kb
        .query_cypher(
            "MATCH (p:Participant) RETURN count(p) AS c",
            &HashMap::new(),
        )
        .await
        .expect("count");
    match rows.first().and_then(|r| r.get("c")) {
        Some(Value::Int(n)) => *n,
        other => panic!("{other:?}"),
    }
}

/// Make every directory under `dir` read-only (`writable = false`) or
/// writable again, so nothing new can be written into the store's tables.
fn set_writable(dir: &Path, writable: bool) {
    let mode = if writable { 0o755 } else { 0o555 };
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            set_writable(&path, writable);
        }
    }
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode)).unwrap();
}

/// A final flush that fails is returned by shutdown, not just logged, and
/// the commit it held is replayed from the write-ahead log at the next
/// open. The flush is made to fail by making the store's table directory
/// read-only; the write-ahead log is outside it.
#[tokio::test]
async fn a_failed_final_flush_is_returned_and_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    let kb = open(&path).await;
    let props: HashMap<String, Value> = [
        ("participant_id".to_string(), Value::String("p-1".into())),
        ("kind".to_string(), Value::String("agent".into())),
    ]
    .into_iter()
    .collect();
    kb.create_node("Participant", &props).await.expect("create");

    let tables = path.join("storage");
    set_writable(&tables, false);
    let shut = kb.shutdown().await;
    set_writable(&tables, true);
    let err = shut.expect_err("the failed flush is returned");
    assert!(
        matches!(&err, UnikoError::Storage(m) if m.contains("final flush at shutdown failed")),
        "{err}"
    );

    let kb = open(&path).await;
    assert_eq!(
        participants(&kb).await,
        1,
        "the write-ahead log replays the commit"
    );
    kb.shutdown().await.expect("a clean shutdown");
    let kb = open(&path).await;
    assert_eq!(participants(&kb).await, 1, "flushed this time");
    kb.shutdown().await.expect("shutdown");
}
