//! A store kept in an object store, end to end, through [`RemoteStorage`].
//!
//! Each test runs once per configured provider, and returns early when none
//! is:
//!
//! - **S3**, against an S3-compatible service: LocalStack or the AWS
//!   Emulator, started by `scripts/s3/start.sh`, which prints the
//!   environment these tests read.
//!   - `UNIKO_S3_ENDPOINT`: the service, e.g. `http://127.0.0.1:4566`;
//!   - `UNIKO_S3_CA`: its CA certificate, for an HTTPS service with a
//!     private CA (the client itself trusts it through `SSL_CERT_FILE`);
//!   - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION`:
//!     optional, `test` / `test` / `us-east-1` when unset.
//! - **Azure Blob Storage**, against a real storage account:
//!   - `UNIKO_AZURE_ACCOUNT` and `UNIKO_AZURE_CONTAINER`: an existing
//!     container;
//!   - `UNIKO_AZURE_KEY`: the account key the stores open with;
//!   - `UNIKO_AZURE_SAS`: a SAS token with read and list rights on the
//!     container, which the tests check the container's contents with.
//!
//! Each test keeps its stores under a root of its own: a fresh bucket on
//! S3, a fresh prefix in the container on Azure.
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

/// An object store the tests run against.
enum Cloud {
    S3 {
        endpoint: String,
    },
    Azure {
        account: String,
        container: String,
        key: String,
        sas: String,
    },
}

/// The provider and where it is, never its credentials.
impl std::fmt::Debug for Cloud {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cloud::S3 { endpoint } => write!(f, "S3 at {endpoint}"),
            Cloud::Azure {
                account, container, ..
            } => write!(f, "Azure {account}/{container}"),
        }
    }
}

impl Cloud {
    /// Every provider the environment configures.
    fn all() -> Vec<Cloud> {
        let mut clouds = Vec::new();
        if let Some(endpoint) = s3_endpoint() {
            clouds.push(Cloud::S3 { endpoint });
        }
        match (
            env("UNIKO_AZURE_ACCOUNT"),
            env("UNIKO_AZURE_CONTAINER"),
            env("UNIKO_AZURE_KEY"),
            env("UNIKO_AZURE_SAS"),
        ) {
            (Some(account), Some(container), Some(key), Some(sas)) => clouds.push(Cloud::Azure {
                account,
                container,
                key,
                sas: sas.trim_start_matches('?').to_string(),
            }),
            _ => eprintln!(
                "skipping Azure: UNIKO_AZURE_ACCOUNT, UNIKO_AZURE_CONTAINER, UNIKO_AZURE_KEY and \
                 UNIKO_AZURE_SAS are not all set"
            ),
        }
        clouds
    }

    /// A fresh root for one test's stores: a new bucket on S3, a new
    /// prefix in the container on Azure.
    fn root(&self, name: &str) -> Root<'_> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let unique = format!("uniko-{name}-{}-{nanos}", std::process::id());
        match self {
            Cloud::S3 { endpoint } => {
                let status = s3("PUT", endpoint, &unique).0;
                assert!(
                    (200..300).contains(&status),
                    "create bucket {unique}: HTTP {status}"
                );
                Root {
                    cloud: self,
                    url: format!("s3://{unique}"),
                    path: unique,
                }
            }
            Cloud::Azure { container, .. } => Root {
                cloud: self,
                url: format!("az://{container}/{unique}"),
                path: format!("{container}/{unique}"),
            },
        }
    }

    /// A config whose `remote_storage` reaches `url` on this provider.
    fn config(&self, url: &str) -> UnikoConfig {
        let remote = match self {
            Cloud::S3 { endpoint } => {
                let (id, secret, region) = s3_credentials();
                RemoteStorage::s3(url)
                    .with_endpoint(endpoint)
                    .with_region(region)
                    .with_credentials(id, secret)
            }
            Cloud::Azure { account, key, .. } => {
                RemoteStorage::azure(url, account).with_access_key(key)
            }
        };
        UnikoConfig {
            remote_storage: Some(remote),
            ..UnikoConfig::default()
        }
    }
}

/// One test's root in a provider: every store it opens is under it.
struct Root<'a> {
    cloud: &'a Cloud,
    /// The root's URL, e.g. `s3://bucket` or `az://container/prefix`.
    url: String,
    /// The root as an HTTP path: the bucket, or the container and prefix.
    path: String,
}

impl Root<'_> {
    /// The URL of the store at `prefix` under the root.
    fn store(&self, prefix: &str) -> String {
        format!("{}/{prefix}", self.url)
    }

    /// The config for the store at `prefix`.
    fn config(&self, prefix: &str) -> UnikoConfig {
        self.cloud.config(&self.store(prefix))
    }

    /// `GET` (or `HEAD`) `key` under the root: the status and body.
    fn request(&self, method: &str, key: &str) -> (u16, String) {
        match self.cloud {
            Cloud::S3 { endpoint } => s3(method, endpoint, &format!("{}/{key}", self.path)),
            Cloud::Azure { account, sas, .. } => {
                let mut cmd = std::process::Command::new("curl");
                cmd.args(["-s", "-w", "\n%{http_code}"]);
                if method == "HEAD" {
                    cmd.arg("--head");
                }
                let out = cmd
                    .arg(format!(
                        "https://{account}.blob.core.windows.net/{}/{key}?{sas}",
                        self.path
                    ))
                    .output()
                    .expect("curl runs");
                split_status(&out.stdout)
            }
        }
    }

    /// Whether `key` exists under the root.
    fn exists(&self, key: &str) -> bool {
        self.request("HEAD", key).0 == 200
    }

    /// Whether uni-db's writer lease for the store at `prefix` is held:
    /// present, and not marked released.
    fn lease_held(&self, prefix: &str) -> bool {
        let (status, body) = self.request("GET", &format!("{prefix}/catalog/writer_lease.json"));
        if status != 200 {
            return false;
        }
        let record: serde_json::Value = serde_json::from_str(&body).expect("lease record");
        record["released"] != serde_json::Value::Bool(true)
    }
}

/// The S3 service under test, or `None` to skip S3.
fn s3_endpoint() -> Option<String> {
    let e = env("UNIKO_S3_ENDPOINT");
    if e.is_none() {
        eprintln!("skipping S3: UNIKO_S3_ENDPOINT is unset (see scripts/s3/start.sh)");
        return None;
    }
    // Lance reaches S3 through a client that, on macOS, verifies TLS against
    // the system keychain and ignores SSL_CERT_FILE, so it can't trust a
    // private CA there. On Linux it reads SSL_CERT_FILE.
    if cfg!(target_os = "macos") && env("UNIKO_S3_CA").is_some() {
        eprintln!(
            "skipping S3: on macOS the store can't trust the service's private CA; \
             use LocalStack (plain HTTP) locally"
        );
        return None;
    }
    e
}

fn s3_credentials() -> (String, String, String) {
    (
        env("AWS_ACCESS_KEY_ID").unwrap_or_else(|| "test".into()),
        env("AWS_SECRET_ACCESS_KEY").unwrap_or_else(|| "test".into()),
        env("AWS_REGION").unwrap_or_else(|| "us-east-1".into()),
    )
}

/// Send a signed S3 request with curl, which signs SigV4 itself; returns
/// the HTTP status and body.
fn s3(method: &str, endpoint: &str, path: &str) -> (u16, String) {
    let (id, secret, region) = s3_credentials();
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
    split_status(&out.stdout)
}

/// curl's output with `-w '\n%{http_code}'`: the body, then the status.
fn split_status(stdout: &[u8]) -> (u16, String) {
    let out = String::from_utf8_lossy(stdout);
    let (body, status) = out.rsplit_once('\n').unwrap_or(("", &out));
    (status.trim().parse().unwrap_or(0), body.to_string())
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

/// Everything a store does, in a bucket, across shutdowns, reopens, a migration
/// and a move to another host.
#[tokio::test(flavor = "multi_thread")]
async fn a_store_in_a_bucket_keeps_everything_across_its_lifecycle() {
    for cloud in Cloud::all() {
        eprintln!("a_store_in_a_bucket_keeps_everything_across_its_lifecycle: {cloud:?}");
        lifecycle(&cloud).await;
    }
}

async fn lifecycle(cloud: &Cloud) {
    let root = cloud.root("lifecycle");
    let cfg = root.config("kb");
    let host_a = tempfile::tempdir().unwrap();
    let host_b = tempfile::tempdir().unwrap();

    // ── Open and write ──────────────────────────────────────────────
    let memory = open(host_a.path(), &cfg).await.expect("open in the bucket");
    assert!(
        root.exists("kb/catalog/schema.json"),
        "the schema is in the bucket, under the store's prefix"
    );
    assert!(root.lease_held("kb"), "the lease is held");

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
    assert!(!root.lease_held("kb"), "shutting down releases the lease");

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
    for cloud in Cloud::all() {
        eprintln!("a_remote_only_store_needs_nothing_local: {cloud:?}");
        remote_only(&cloud).await;
    }
}

async fn remote_only(cloud: &Cloud) {
    let root = cloud.root("remote-only");
    let url = root.store("kb");
    let cfg = root.config("kb");
    let memory = open(Path::new(&url), &cfg).await.expect("open remote-only");
    assert!(root.exists("kb/catalog/schema.json"));
    assert!(root.lease_held("kb"), "the lease is held");
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
    assert!(!root.lease_held("kb"), "released at shutdown");

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
    for cloud in Cloud::all() {
        eprintln!("remote_storage_rejects_a_uri_for_another_store: {cloud:?}");
        uri_for_another_store(&cloud).await;
    }
}

async fn uri_for_another_store(cloud: &Cloud) {
    let root = cloud.root("uri-path");
    let cfg = root.config("a");
    let err = open(Path::new(&root.store("b")), &cfg)
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
    for cloud in Cloud::all() {
        eprintln!("two_prefixes_in_one_bucket_are_two_stores: {cloud:?}");
        two_prefixes(&cloud).await;
    }
}

async fn two_prefixes(cloud: &Cloud) {
    let root = cloud.root("prefixes");
    let (a_dir, b_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let a_cfg = root.config("a");
    let b_cfg = root.config("b");
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
    assert!(root.exists("a/catalog/schema.json"));
    assert!(root.exists("b/catalog/schema.json"));
    assert!(!root.exists("catalog/schema.json"), "nothing at the root");
    let report = Uniko::inspect_store(b_dir.path(), &b_cfg)
        .await
        .expect("inspect b");
    assert!(report.info.exists, "b is a store of its own: {report:?}");
}
