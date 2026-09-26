//! Isolated repro for a uni-db 4.0.0 regression: re-applying an unchanged
//! schema to a populated label is rejected, so a persistent store cannot be
//! reopened once it holds a row.
//!
//! Symptom on the second `Uni::open` of any store whose schema declares a
//! NOT NULL property on a label that has rows:
//!
//! ```text
//! Schema error: Property 'name' on 'Person' is declared as String
//! (nullable: false); cannot re-declare as String (nullable: true).
//! Property types are immutable — use a new property name or migrate the data
//! ```
//!
//! Root cause: `uni-db-4.0.0/src/api/schema.rs:195-220` coerces a NOT NULL
//! declaration to nullable when the label already has rows — reasonable for
//! genuinely *adding* a column to populated data, since existing rows have no
//! value for it. But the coercion runs before
//! `SchemaManager::declare_property`'s idempotent-re-declaration check
//! (`uni-common-4.0.0/src/core/schema.rs:1823`), which returns `Ok(false)`
//! only when `data_type` AND `nullable` both match. So the re-declaration
//! arrives as `nullable: true`, compares against the stored `nullable: false`,
//! and hard-errors.
//!
//! The coercion should not apply to a property that already exists with the
//! same type and nullability: that is a no-op re-declaration, not an add.
//!
//! Impact: any embedder that registers its schema on every open — the
//! documented idempotent pattern, and what uniko does — cannot reopen a
//! persistent store after the first write. uniko trips it on
//! `:KnowledgeBaseStats`, whose singleton row is written during the first
//! open, but every populated label with a NOT NULL property is affected.
//!
//! Bisected by pinning the same source: `uni-db = "=3.4.0"` passes both
//! cases, `uni-db = "4"` fails the populated one. 3/3 deterministic on
//! 4.0.0. The coercion does not exist on 3.4.0, where the re-declaration is
//! an idempotent no-op.
//!
//! Depends on `uni_db` alone — no uniko types — so it lifts into the upstream
//! repo unchanged. A fully standalone crate version (its own Cargo.toml,
//! outside any workspace, `cargo run` printing PASS/FAIL and exiting 1 on the
//! bug) was used to confirm the bisect independently of uniko's feature set.

use uni_db::{DataType, Uni};

/// Register a label with one NOT NULL property. Identical on both opens —
/// this is the "idempotent re-registration" pattern.
async fn apply_schema(db: &Uni) -> Result<(), uni_db::UniError> {
    db.schema()
        .label("Person")
        .property("name", DataType::String)
        .done()
        .apply()
        .await
        .map(|_| ())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reapplying_an_unchanged_schema_to_a_populated_label_succeeds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = dir.path().join("store");

    // First open: declare the schema and write one row.
    {
        let db = Uni::open(store.to_string_lossy())
            .build()
            .await
            .expect("first open");
        apply_schema(&db).await.expect("first schema apply");

        let tx = db.session().tx().await.expect("begin tx");
        tx.query_with("CREATE (p:Person {name: 'ada'})")
            .fetch_all()
            .await
            .expect("create");
        tx.commit().await.expect("commit");
        db.shutdown().await.expect("shutdown");
    }

    // Reopen and re-apply the SAME schema. The label now has one row.
    let db = Uni::open(store.to_string_lossy())
        .build()
        .await
        .expect("reopen");

    // EXPECTED: an unchanged re-declaration is a no-op.
    // ACTUAL on 4.0.0: the NOT NULL declaration is coerced to nullable
    // because the label is populated, then rejected for disagreeing with the
    // stored nullability.
    apply_schema(&db)
        .await
        .expect("re-applying an unchanged schema to a populated label");

    let rows = db
        .session()
        .query_with("MATCH (p:Person) RETURN p.name AS name")
        .fetch_all()
        .await
        .expect("query");
    let name: String = rows
        .rows()
        .first()
        .expect("one row")
        .get("name")
        .expect("name");
    assert_eq!(name, "ada");
}

/// Control: the same re-application against an EMPTY label is fine, which
/// isolates "the label has rows" as the trigger rather than re-declaration
/// itself.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reapplying_an_unchanged_schema_to_an_empty_label_succeeds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = dir.path().join("store");

    {
        let db = Uni::open(store.to_string_lossy())
            .build()
            .await
            .expect("first open");
        apply_schema(&db).await.expect("first schema apply");
        db.shutdown().await.expect("shutdown");
    }

    let db = Uni::open(store.to_string_lossy())
        .build()
        .await
        .expect("reopen");
    apply_schema(&db)
        .await
        .expect("re-applying an unchanged schema to an empty label");
}
