//! Plain uni-db, no uniko code: what uniko's object-store support relies
//! on from uni-db.
//!
//! Run against an S3-compatible service (`scripts/s3/start.sh localstack`);
//! skipped unless `UNIKO_S3_ENDPOINT` is set:
//!
//! ```sh
//! set -a; eval "$(scripts/s3/start.sh localstack)"; set +a
//! cargo nextest run -p uniko-store --test integration unidb_object_store_repro::
//! ```
//!
//! 1. A store keeps everything under its URL's prefix, so two prefixes of
//!    one bucket are two stores (once not so: rustic-ai/uni-db#349).
//! 2. A remote-only store (`Uni::open("s3://...")`) reads its endpoint and
//!    keys from the environment, or from `cloud_storage` (once neither:
//!    rustic-ai/uni-db#350).
//! 3. A second writer on a store in an object store is refused by uni-db's
//!    lease (rustic-ai/uni-db#348).

use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjPath;
use object_store::{ObjectStore, ObjectStoreExt};
use uni_db::common::CloudStorageConfig;
use uni_db::{DataType, Uni, UniConfig, UniError};

fn endpoint() -> Option<String> {
    std::env::var("UNIKO_S3_ENDPOINT").ok()
}

fn cloud(endpoint: &str, bucket: &str) -> CloudStorageConfig {
    CloudStorageConfig::S3 {
        bucket: bucket.into(),
        region: Some("us-east-1".into()),
        endpoint: Some(endpoint.into()),
        access_key_id: Some("test".into()),
        secret_access_key: Some("test".into()),
        session_token: None,
        virtual_hosted_style: false,
    }
}

fn client(endpoint: &str, bucket: &str) -> impl ObjectStore {
    AmazonS3Builder::new()
        .with_endpoint(endpoint)
        .with_bucket_name(bucket)
        .with_region("us-east-1")
        .with_access_key_id("test")
        .with_secret_access_key("test")
        .with_allow_http(true)
        .build()
        .unwrap()
}

/// A fresh bucket: a signed PUT through curl.
fn make_bucket(endpoint: &str, name: &str) -> String {
    let bucket = format!(
        "unidb-repro-{name}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let ok = std::process::Command::new("curl")
        .args(["-sf", "-X", "PUT", "--aws-sigv4", "aws:amz:us-east-1:s3"])
        .args(["--user", "test:test"])
        .arg(format!("{endpoint}/{bucket}"))
        .status()
        .unwrap()
        .success();
    assert!(ok, "create bucket {bucket}");
    bucket
}

async fn exists(store: &impl ObjectStore, key: &str) -> bool {
    store.head(&ObjPath::from(key)).await.is_ok()
}

/// Whether the writer lease under `prefix` is marked released.
async fn lease_released(store: &impl ObjectStore, prefix: &str) -> bool {
    let key = ObjPath::from(format!("{prefix}/catalog/writer_lease.json"));
    let bytes = store.get(&key).await.unwrap().bytes().await.unwrap();
    let record: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    record["released"] == serde_json::Value::Bool(true)
}

async fn labels(db: &Uni) -> Vec<String> {
    let mut l: Vec<String> = db.schema().current().labels.keys().cloned().collect();
    l.sort();
    l
}

#[tokio::test(flavor = "multi_thread")]
async fn two_prefixes_in_one_bucket_are_two_stores() {
    let Some(endpoint) = endpoint() else {
        eprintln!("skipping: UNIKO_S3_ENDPOINT is unset");
        return;
    };
    let bucket = make_bucket(&endpoint, "prefix");
    let (dir_a, dir_b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());

    let a = Uni::open(dir_a.path().to_string_lossy())
        .hybrid(&format!("s3://{bucket}/a"))
        .cloud_storage(cloud(&endpoint, &bucket))
        .build()
        .await
        .unwrap();
    a.schema()
        .label("OnlyInA")
        .property("n", DataType::Int64)
        .done()
        .apply()
        .await
        .unwrap();
    a.flush().await.unwrap();

    let b = Uni::open(dir_b.path().to_string_lossy())
        .hybrid(&format!("s3://{bucket}/b"))
        .cloud_storage(cloud(&endpoint, &bucket))
        .build()
        .await
        .unwrap();
    let b_labels = labels(&b).await;

    let store = client(&endpoint, &bucket);
    assert!(exists(&store, "a/catalog/schema.json").await);
    assert!(
        !exists(&store, "catalog/schema.json").await,
        "nothing at the bucket root"
    );
    assert!(
        !b_labels.contains(&"OnlyInA".to_string()),
        "store b has its own schema: {b_labels:?}"
    );
    a.shutdown().await.unwrap();
    b.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_remote_only_store_takes_its_endpoint_and_keys() {
    let Some(endpoint) = endpoint() else {
        eprintln!("skipping: UNIKO_S3_ENDPOINT is unset");
        return;
    };
    let bucket = make_bucket(&endpoint, "remote-only");
    let url = format!("s3://{bucket}/kb");
    let db = Uni::open(url.as_str())
        .cloud_storage(cloud(&endpoint, &bucket))
        .build()
        .await
        .expect("a remote-only store opens on the configured endpoint");
    db.schema()
        .label("Doc")
        .property("n", DataType::Int64)
        .done()
        .apply()
        .await
        .unwrap();
    db.flush().await.unwrap();

    // A second writer, from anywhere, is refused while this one holds it.
    let mut quick = UniConfig::default();
    quick.writer_lease.acquire_grace = std::time::Duration::from_millis(500);
    let second = Uni::open(url.as_str())
        .cloud_storage(cloud(&endpoint, &bucket))
        .config(quick)
        .build()
        .await;
    assert!(
        matches!(second, Err(UniError::StoreLocked { .. })),
        "a second writer is refused: {:?}",
        second.err()
    );
    db.shutdown().await.unwrap();

    let store = client(&endpoint, &bucket);
    assert!(exists(&store, "kb/catalog/schema.json").await);
    assert!(
        lease_released(&store, "kb").await,
        "shutting down released the lease"
    );
    let reopened = Uni::open_existing(url.as_str())
        .cloud_storage(cloud(&endpoint, &bucket))
        .build()
        .await
        .expect("reopens once the writer closed");
    assert!(labels(&reopened).await.contains(&"Doc".to_string()));
    reopened.shutdown().await.unwrap();
}
