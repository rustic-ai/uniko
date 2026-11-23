//! A store kept in S3, end to end, through [`RemoteStorage`].
//!
//! Runs against an S3-compatible service: LocalStack or the AWS Emulator,
//! started by `scripts/s3/start.sh`, which prints the environment these tests
//! read. They return early unless `UNIKO_S3_ENDPOINT` is set:
//!
//! - `UNIKO_S3_ENDPOINT`: the service, e.g. `http://127.0.0.1:4566`;
//! - `UNIKO_S3_CA`: its CA certificate, for an HTTPS service with a private
//!   CA (the client itself trusts it through `SSL_CERT_FILE`);
//! - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION`: optional,
//!   `test` / `test` / `us-east-1` when unset.
//!
//! Each test makes its own bucket, and keeps its store under a prefix.
//!
//! ```sh
//! set -a; eval "$(scripts/s3/start.sh localstack)"; set +a
//! cargo nextest run -p uniko-memory --test integration cloud_e2e::
//! ```

use std::collections::HashMap;
use std::path::Path;

use uniko_memory::{
    AssertFactParams, CreateGoalParams, CreateTaskParams, IngestOutcome, IngestSource,
    RecordEpisodeParams, RemoteStorage, SearchOptions, Turn, Uniko, UnikoConfig, UnikoError, Value,
};

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// The S3 service under test, or `None` to skip.
fn endpoint() -> Option<String> {
    let e = env("UNIKO_S3_ENDPOINT");
    if e.is_none() {
        eprintln!("skipping: UNIKO_S3_ENDPOINT is unset (see scripts/s3/start.sh)");
        return None;
    }
    // Lance reaches S3 through a client that, on macOS, verifies TLS against
    // the system keychain and ignores SSL_CERT_FILE, so it can't trust a
    // private CA there. On Linux it reads SSL_CERT_FILE.
    if cfg!(target_os = "macos") && env("UNIKO_S3_CA").is_some() {
        eprintln!(
            "skipping: on macOS the store can't trust the service's private CA; \
             use LocalStack (plain HTTP) locally"
        );
        return None;
    }
    e
}

fn credentials() -> (String, String, String) {
    (
        env("AWS_ACCESS_KEY_ID").unwrap_or_else(|| "test".into()),
        env("AWS_SECRET_ACCESS_KEY").unwrap_or_else(|| "test".into()),
        env("AWS_REGION").unwrap_or_else(|| "us-east-1".into()),
    )
}

/// Send a signed S3 request with curl, which signs SigV4 itself; returns
/// the HTTP status.
fn s3(method: &str, endpoint: &str, path: &str) -> u16 {
    s3_body(method, endpoint, path).0
}

/// [`s3`], with the response body.
fn s3_body(method: &str, endpoint: &str, path: &str) -> (u16, String) {
    let (id, secret, region) = credentials();
    let mut cmd = std::process::Command::new("curl");
    cmd.args(["-s", "-w", "\n%{http_code}", "-X", method])
        .arg("--aws-sigv4")
        .arg(format!("aws:amz:{region}:s3"))
        .arg("--user")
        .arg(format!("{id}:{secret}"));
    if let Some(ca) = env("UNIKO_S3_CA") {
        cmd.arg("--cacert").arg(ca);
    }
    let out = cmd
        .arg(format!("{endpoint}/{path}"))
        .output()
        .expect("curl runs");
    let out = String::from_utf8_lossy(&out.stdout);
    let (body, status) = out.rsplit_once('\n').unwrap_or(("", &out));
    (status.trim().parse().unwrap_or(0), body.to_string())
}

/// A fresh bucket on the service.
fn bucket(endpoint: &str, name: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let bucket = format!("uniko-{name}-{}-{nanos}", std::process::id());
    let status = s3("PUT", endpoint, &bucket);
    assert!(
        (200..300).contains(&status),
        "create bucket {bucket}: HTTP {status}"
    );
    bucket
}

/// Whether `key` exists in `bucket`.
fn exists(endpoint: &str, bucket: &str, key: &str) -> bool {
    s3("HEAD", endpoint, &format!("{bucket}/{key}")) == 200
}

/// Whether uni-db's writer lease for the store under `prefix` is held:
/// present, and not marked released.
fn lease_held(endpoint: &str, bucket: &str, prefix: &str) -> bool {
    let (status, body) = s3_body(
        "GET",
        endpoint,
        &format!("{bucket}/{prefix}/catalog/writer_lease.json"),
    );
    if status != 200 {
        return false;
    }
    let record: serde_json::Value = serde_json::from_str(&body).expect("lease record");
    record["released"] != serde_json::Value::Bool(true)
}

fn config(endpoint: &str, url: &str) -> UnikoConfig {
    let (id, secret, region) = credentials();
    UnikoConfig {
        remote_storage: Some(
            RemoteStorage::s3(url)
                .with_endpoint(endpoint)
                .with_region(region)
                .with_credentials(id, secret),
        ),
        ..UnikoConfig::default()
    }
}

async fn open(local: &Path, config: &UnikoConfig) -> Result<Uniko, UnikoError> {
    Uniko::builder()
        .path(local)
        .raw_config(config.clone())
        .build()
        .await
}

/// One `count(...) AS c` row.
async fn count(memory: &Uniko, cypher: &str) -> i64 {
    let rows = memory.agent("auditor").query(cypher).await.expect(cypher);
    match rows.first().and_then(|r| r.get("c")) {
        Some(Value::Int(n)) => *n,
        other => panic!("{cypher}: {other:?}"),
    }
}

/// What the lifecycle test writes, read back the same way after each reopen.
#[derive(Debug, PartialEq)]
struct Snapshot {
    messages: i64,
    artifacts: i64,
    facts: i64,
    episodes: i64,
    goals: usize,
    completed_goals: usize,
    rules: usize,
}

async fn snapshot(memory: &Uniko) -> Snapshot {
    let agent = memory.agent("assistant");
    Snapshot {
        messages: count(memory, "MATCH (m:Message) RETURN count(m) AS c").await,
        artifacts: count(memory, "MATCH (a:Artifact) RETURN count(a) AS c").await,
        facts: count(memory, "MATCH (f:Fact) RETURN count(f) AS c").await,
        episodes: count(memory, "MATCH (e:Episode) RETURN count(e) AS c").await,
        goals: agent.goals().all().await.expect("goals").len(),
        completed_goals: agent.goals().completed().await.expect("completed").len(),
        rules: agent.list_rules().await.expect("rules").len(),
    }
}

/// Everything a store does, on S3, across shutdowns, reopens, a migration
/// and a move to another host.
#[tokio::test(flavor = "multi_thread")]
async fn a_store_on_s3_keeps_everything_across_its_lifecycle() {
    let Some(endpoint) = endpoint() else { return };
    let bucket = bucket(&endpoint, "lifecycle");
    let url = format!("s3://{bucket}/kb");
    let cfg = config(&endpoint, &url);
    let host_a = tempfile::tempdir().unwrap();
    let host_b = tempfile::tempdir().unwrap();

    // ── Open and write ──────────────────────────────────────────────
    let memory = open(host_a.path(), &cfg).await.expect("open on S3");
    assert!(
        exists(&endpoint, &bucket, "kb/catalog/schema.json"),
        "the schema is in the bucket, under the store's prefix"
    );
    assert!(lease_held(&endpoint, &bucket, "kb"), "the lease is held");

    // Another host can't open it while it is open.
    let err = open(host_b.path(), &cfg)
        .await
        .err()
        .expect("second open refused");
    assert!(
        matches!(&err, UnikoError::Config(m) if m.contains("open for writing elsewhere")),
        "a live lease refuses another opener: {err}"
    );

    let agent = memory.agent("assistant");
    let mut chat = agent.session("chat-1");
    for (who, text) in [
        ("alice", "I'm moving to Lisbon in March for the new job."),
        ("alice", "My sister Carla lives in Porto and has two cats."),
        (
            "assistant",
            "Noted: Lisbon in March, and Carla is in Porto.",
        ),
        ("alice", "I prefer window seats when I fly to Porto."),
    ] {
        chat.observe(Turn::new(who, text)).await.expect("observe");
    }
    let doc = match chat
        .ingest(IngestSource::text(
            "# Relocation checklist\n\nRegister with the Lisbon city hall within 90 days. \
             Transfer the car insurance before March 15.",
        ))
        .await
        .expect("ingest")
    {
        IngestOutcome::Artifact(a) => a,
        other => panic!("expected an artifact, got {other:?}"),
    };
    agent
        .assert_fact(AssertFactParams {
            subject: "alice".into(),
            predicate: "moving_to".into(),
            object: Some("Lisbon".into()),
            ..AssertFactParams::default()
        })
        .await
        .expect("assert fact");
    agent
        .record_episode(RecordEpisodeParams {
            action_type: "booked_flight".into(),
            outcome: Some("success".into()),
            ..RecordEpisodeParams::default()
        })
        .await
        .expect("record episode");
    agent
        .define_rule(
            "cloud_e2e_facts",
            "CREATE RULE cloud_e2e_facts AS MATCH (f:Fact) YIELD KEY f",
        )
        .await
        .expect("define rule");
    let rows = agent
        .run_rule("cloud_e2e_facts", &["f"], HashMap::new())
        .await
        .expect("run rule");
    assert!(!rows.is_empty(), "the rule sees the fact");
    let goals = agent.goals();
    goals
        .create(CreateGoalParams {
            goal_id: Some("relocate".into()),
            title: "Relocate to Lisbon".into(),
            ..CreateGoalParams::default()
        })
        .await
        .expect("create goal");
    goals
        .create_task(CreateTaskParams {
            task_id: Some("insurance".into()),
            title: "Transfer car insurance".into(),
            goal_id: Some("relocate".into()),
            ..CreateTaskParams::default()
        })
        .await
        .expect("create task");
    assert!(
        goals
            .complete_task("insurance")
            .await
            .expect("complete task")
    );
    goals
        .create(CreateGoalParams {
            goal_id: Some("visit".into()),
            title: "Visit Carla".into(),
            ..CreateGoalParams::default()
        })
        .await
        .expect("create goal");
    assert!(goals.complete("visit", None).await.expect("complete goal"));
    chat.finalize().await.expect("finalize");
    agent.consolidate().await.expect("consolidate");

    // ── Read it back ────────────────────────────────────────────────
    let recalled = agent.recall("where is alice moving").await.expect("recall");
    assert!(
        recalled.items.iter().any(|i| i.content.contains("Lisbon")),
        "recall finds the move: {:?}",
        recalled
            .items
            .iter()
            .map(|i| &i.content)
            .collect::<Vec<_>>()
    );
    let hits = agent
        .sources()
        .search("city hall", &SearchOptions::default())
        .await
        .expect("search");
    assert!(!hits.is_empty(), "source search finds the checklist");
    let bytes = agent
        .data()
        .artifact_bytes(&doc.artifact_id)
        .await
        .expect("artifact bytes")
        .expect("the artifact has bytes");
    assert!(String::from_utf8_lossy(&bytes).contains("city hall"));
    let before = snapshot(&memory).await;
    assert_eq!(before.messages, 4, "{before:?}");
    assert_eq!(before.artifacts, 1, "{before:?}");
    assert!(before.facts >= 1, "{before:?}");
    assert_eq!(before.episodes, 1, "{before:?}");
    assert_eq!((before.goals, before.completed_goals), (2, 1), "{before:?}");
    assert!(before.rules >= 1, "{before:?}");
    let info = memory.store_info().await.expect("store info");
    assert!(info.exists, "{info:?}");

    drop(goals);
    drop((chat, agent));
    memory.shutdown().await.expect("shutdown");
    assert!(
        !lease_held(&endpoint, &bucket, "kb"),
        "shutting down releases the lease"
    );

    // ── Reopen on the same host ─────────────────────────────────────
    let memory = open(host_a.path(), &cfg).await.expect("reopen");
    assert_eq!(snapshot(&memory).await, before, "reopened as it was");
    let agent = memory.agent("assistant");
    assert!(
        agent
            .recall("Carla's cats")
            .await
            .expect("recall")
            .items
            .iter()
            .any(|i| i.content.contains("Carla")),
        "recall after reopening"
    );

    // Deletes reach the bucket too.
    let deleted = agent
        .session("chat-1")
        .delete_document(&doc.artifact_id)
        .await
        .expect("delete document");
    assert!(deleted.root_existed, "{deleted:?}");
    assert_eq!(
        count(&memory, "MATCH (a:Artifact) RETURN count(a) AS c").await,
        0
    );
    drop(agent);
    memory.shutdown().await.expect("shutdown");

    // ── Inspect and migrate in place ────────────────────────────────
    let report = Uniko::inspect_store(host_a.path(), &cfg)
        .await
        .expect("inspect");
    assert!(report.info.exists, "{report:?}");
    assert!(report.compatibility.is_compatible(), "{report:?}");
    let mut dot = cfg.clone();
    dot.vector_metric = uniko_store::config::VectorMetricChoice::Dot;
    let migrated = Uniko::migrate_store(
        host_a.path(),
        dot.clone(),
        uniko_memory::Migration::RebuildVectorIndexes,
        |_| {},
    )
    .await
    .expect("migrate");
    assert!(migrated.completed, "{migrated:?}");

    // ── Move to another host ────────────────────────────────────────
    // A new local directory: the write-ahead log and ids start empty, and
    // everything else comes from the bucket.
    let memory = open(host_b.path(), &dot)
        .await
        .expect("open on another host");
    let moved = snapshot(&memory).await;
    assert_eq!(
        moved,
        Snapshot {
            artifacts: 0,
            ..before
        },
        "another host sees the same store"
    );
    // And it can keep writing without colliding with what is there.
    let agent = memory.agent("assistant");
    agent
        .session("chat-2")
        .observe(Turn::new("alice", "The movers come on March 3rd."))
        .await
        .expect("observe on the new host");
    assert_eq!(
        count(&memory, "MATCH (m:Message) RETURN count(m) AS c").await,
        before.messages + 1
    );
    let ids = memory
        .agent("auditor")
        .query("MATCH (m:Message) RETURN id(m) AS i")
        .await
        .expect("ids");
    let mut seen = std::collections::HashSet::new();
    for row in &ids {
        assert!(
            seen.insert(format!("{:?}", row.get("i"))),
            "duplicate id: {ids:?}"
        );
    }
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// A remote-only store: the path is the store's URL, and nothing is kept
/// locally, so any host opens it as it is.
#[tokio::test(flavor = "multi_thread")]
async fn a_remote_only_store_needs_nothing_local() {
    let Some(endpoint) = endpoint() else { return };
    let bucket = bucket(&endpoint, "remote-only");
    let url = format!("s3://{bucket}/kb");
    let cfg = config(&endpoint, &url);
    let memory = open(Path::new(&url), &cfg).await.expect("open remote-only");
    assert!(exists(&endpoint, &bucket, "kb/catalog/schema.json"));
    assert!(lease_held(&endpoint, &bucket, "kb"), "the lease is held");
    let err = open(Path::new(&url), &cfg)
        .await
        .err()
        .expect("a second open in this process is refused");
    assert!(err.to_string().contains("already open"), "{err}");

    memory
        .agent("assistant")
        .session("chat")
        .observe(Turn::new("alice", "My sister Carla lives in Porto."))
        .await
        .expect("observe");
    memory.shutdown().await.expect("shutdown");
    assert!(
        !lease_held(&endpoint, &bucket, "kb"),
        "released at shutdown"
    );

    let report = Uniko::inspect_store(Path::new(&url), &cfg)
        .await
        .expect("inspect a remote-only store");
    assert!(report.info.exists, "{report:?}");
    assert!(report.compatibility.is_compatible(), "{report:?}");

    let memory = open(Path::new(&url), &cfg).await.expect("reopen");
    assert_eq!(
        count(&memory, "MATCH (m:Message) RETURN count(m) AS c").await,
        1,
        "the commit was in the bucket, not on this host"
    );
    memory.shutdown().await.expect("shutdown");
}

/// With `remote_storage` set, a URI path must be that store's own URL.
#[tokio::test(flavor = "multi_thread")]
async fn remote_storage_rejects_a_uri_for_another_store() {
    let Some(endpoint) = endpoint() else { return };
    let bucket = bucket(&endpoint, "uri-path");
    let cfg = config(&endpoint, &format!("s3://{bucket}/a"));
    let err = open(Path::new(&format!("s3://{bucket}/b")), &cfg)
        .await
        .err()
        .expect("refused");
    assert!(
        err.to_string().contains("different stores"),
        "explains the mismatch: {err}"
    );
}

/// Two stores under different prefixes of one bucket stay apart.
#[tokio::test(flavor = "multi_thread")]
async fn two_prefixes_in_one_bucket_are_two_stores() {
    let Some(endpoint) = endpoint() else { return };
    let bucket = bucket(&endpoint, "prefixes");
    let (a_dir, b_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let a_cfg = config(&endpoint, &format!("s3://{bucket}/a"));
    let b_cfg = config(&endpoint, &format!("s3://{bucket}/b"));
    let a = open(a_dir.path(), &a_cfg).await.expect("open a");
    a.agent("assistant")
        .session("chat")
        .observe(Turn::new("alice", "Only store a hears this."))
        .await
        .expect("observe in a");
    // Both open at once: each holds its own lease.
    let b = open(b_dir.path(), &b_cfg)
        .await
        .expect("open b alongside a");
    assert_eq!(
        count(&b, "MATCH (m:Message) RETURN count(m) AS c").await,
        0,
        "b doesn't see a's writes"
    );
    a.shutdown().await.expect("shutdown a");
    b.shutdown().await.expect("shutdown b");
    assert!(exists(&endpoint, &bucket, "a/catalog/schema.json"));
    assert!(exists(&endpoint, &bucket, "b/catalog/schema.json"));
    assert!(
        !exists(&endpoint, &bucket, "catalog/schema.json"),
        "nothing at the bucket root"
    );
    let report = Uniko::inspect_store(b_dir.path(), &b_cfg)
        .await
        .expect("inspect b");
    assert!(report.info.exists, "b is a store of its own: {report:?}");
}
