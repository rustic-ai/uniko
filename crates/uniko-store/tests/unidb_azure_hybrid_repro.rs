//! Plain uni-db, no uniko code: a store in Azure Blob Storage.
//!
//! Skipped unless `UNIKO_AZURE_ACCOUNT`, `UNIKO_AZURE_CONTAINER` and
//! `UNIKO_AZURE_KEY` name a real storage account and container.
//!
//! A hybrid store on Azure: one row is committed, flushed and the store shut
//! down; a host with an empty local directory then opens it from the
//! container. A remote-only store does the same, for comparison.

use uni_db::common::CloudStorageConfig;
use uni_db::{DataType, Uni};

struct Azure {
    account: String,
    container: String,
    key: String,
}

fn azure() -> Option<Azure> {
    let get = |n: &str| std::env::var(n).ok().filter(|v| !v.is_empty());
    Some(Azure {
        account: get("UNIKO_AZURE_ACCOUNT")?,
        container: get("UNIKO_AZURE_CONTAINER")?,
        key: get("UNIKO_AZURE_KEY")?,
    })
}

impl Azure {
    fn cloud(&self) -> CloudStorageConfig {
        CloudStorageConfig::Azure {
            container: self.container.clone(),
            account: self.account.clone(),
            access_key: Some(self.key.clone()),
            sas_token: None,
        }
    }

    fn url(&self, name: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("az://{}/unidb-repro-{name}-{nanos}/kb", self.container)
    }
}

async fn write_one(db: &Uni) {
    db.schema()
        .label("Note")
        .property("text", DataType::String)
        .done()
        .apply()
        .await
        .unwrap();
    let tx = db.session().tx().await.unwrap();
    tx.query("CREATE (:Note {text: 'kept'})").await.unwrap();
    tx.commit().await.unwrap();
}

async fn notes(db: &Uni) -> i64 {
    let rows = db
        .session()
        .query("MATCH (n:Note) RETURN count(n) AS c")
        .await
        .unwrap();
    rows.rows()
        .first()
        .and_then(|r| r.get::<i64>("c").ok())
        .unwrap_or(-1)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_hybrid_store_on_azure_is_in_the_container_after_flush() {
    let Some(az) = azure() else {
        eprintln!("skipping: UNIKO_AZURE_* is unset");
        return;
    };
    let url = az.url("hybrid");
    let (host_a, host_b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let db = Uni::open(host_a.path().to_string_lossy())
        .hybrid(&url)
        .cloud_storage(az.cloud())
        .build()
        .await
        .unwrap();
    write_one(&db).await;
    let flushed = db.flush().await;
    eprintln!("hybrid flush: {flushed:?}");
    db.shutdown().await.unwrap();

    let db = Uni::open_existing(host_b.path().to_string_lossy())
        .hybrid(&url)
        .cloud_storage(az.cloud())
        .build()
        .await
        .unwrap();
    let n = notes(&db).await;
    db.shutdown().await.unwrap();
    flushed.expect("the flush succeeds");
    assert_eq!(n, 1, "another host sees the flushed row");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_remote_only_store_on_azure_is_in_the_container_after_flush() {
    let Some(az) = azure() else {
        eprintln!("skipping: UNIKO_AZURE_* is unset");
        return;
    };
    let url = az.url("remote-only");
    let db = Uni::open(url.as_str())
        .cloud_storage(az.cloud())
        .build()
        .await
        .unwrap();
    write_one(&db).await;
    let flushed = db.flush().await;
    eprintln!("remote-only flush: {flushed:?}");
    db.shutdown().await.unwrap();

    let db = Uni::open_existing(url.as_str())
        .cloud_storage(az.cloud())
        .build()
        .await
        .unwrap();
    let n = notes(&db).await;
    db.shutdown().await.unwrap();
    flushed.expect("the flush succeeds");
    assert_eq!(n, 1, "the reopened store sees the flushed row");
}

/// The same with no explicit flush: shutdown's own flush has to carry it.
#[tokio::test(flavor = "multi_thread")]
async fn a_hybrid_store_on_azure_is_in_the_container_after_shutdown() {
    let Some(az) = azure() else {
        eprintln!("skipping: UNIKO_AZURE_* is unset");
        return;
    };
    let url = az.url("hybrid-shutdown");
    let (host_a, host_b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let db = Uni::open(host_a.path().to_string_lossy())
        .hybrid(&url)
        .cloud_storage(az.cloud())
        .build()
        .await
        .unwrap();
    write_one(&db).await;
    db.shutdown().await.unwrap();

    let db = Uni::open_existing(host_b.path().to_string_lossy())
        .hybrid(&url)
        .cloud_storage(az.cloud())
        .build()
        .await
        .unwrap();
    let n = notes(&db).await;
    db.shutdown().await.unwrap();
    assert_eq!(n, 1, "another host sees the row committed before shutdown");
}
