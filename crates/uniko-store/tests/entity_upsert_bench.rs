//! RC3 decision bench: batched `UNWIND … MERGE` vs the 4-phase entity
//! upsert (`extract/ner/dedup.rs`).
//!
//! The 4-phase split exists because, pre-uni-db-2.0, batched `MERGE` ran a
//! per-row query plan (issue #69) and was no faster than a per-entity
//! loop. uni-db 2.0 (`ab405408a`) broadened the single-node/single-label
//! MERGE fast path so the entity-upsert shape — `MERGE (e:Entity
//! {entity_id: …})` on a Hash-indexed key — now skips per-row planning.
//! This bench measures whether a single `UNWIND … MERGE … ON CREATE/ON
//! MATCH` is now competitive with the hand-split 4-phase path, so we can
//! decide whether to simplify `dedup.rs`.
//!
//! Scope: structural upsert cost only. The production Entity also carries
//! an auto-embedded `embedding` vector; that cost lands on both paths
//! about equally (new entities embed once on create in either path;
//! matched entities don't re-embed), so it's omitted here to isolate the
//! MERGE-vs-manual difference the #69 fix is about.
//!
//! Run: cargo nextest run -p uniko-store --test integration entity_upsert_bench:: \
//!        --run-ignored all --no-capture

use std::collections::HashMap;
use std::time::Instant;

use uni_db::{Uni, Value};

/// Minimal Entity-like schema: the Hash-indexed `entity_id` merge key
/// plus the scalar columns the real upsert writes.
async fn fresh_db() -> Uni {
    let db = Uni::in_memory().build().await.unwrap();
    db.schema()
        .label("Entity")
        .property("entity_id", uni_db::DataType::String)
        // NOTE: production Entity declares `name` NOT NULL. We make it
        // nullable HERE only so the MERGE path can run at all: on the real
        // schema, `MERGE (e {entity_id}) ON CREATE SET e.name=…` fails with
        // "Property 'name' cannot be null" — MERGE validates NOT NULL before
        // ON CREATE SET (RC4, still unfixed on 2.0). So the perf numbers
        // below describe the *best case* for MERGE; on the real schema MERGE
        // is not even viable for entity upsert.
        .property_nullable("name", uni_db::DataType::String)
        .property_nullable("entity_type", uni_db::DataType::String)
        .property_nullable("frequency", uni_db::DataType::Int64)
        .property_nullable("confidence", uni_db::DataType::Float64)
        .index(
            "entity_id",
            uni_db::IndexType::Scalar(uni_db::ScalarType::Hash),
        )
        .done()
        .apply()
        .await
        .unwrap();
    db
}

/// One input row of the batch.
#[derive(Clone)]
struct Row {
    eid: String,
    name: String,
}

/// Build a batch of `n` rows where the first `n_existing` reuse ids that
/// are already in the graph (→ MATCH/ON MATCH), the rest are new.
fn make_batch(n: usize, n_existing: usize, generation: usize) -> Vec<Row> {
    (0..n)
        .map(|i| {
            let eid = if i < n_existing {
                format!("ent-{i:05}") // collides with pre-seeded ids
            } else {
                format!("ent-new-{generation}-{i:05}") // unique per generation
            };
            Row {
                eid,
                name: format!("name {i}"),
            }
        })
        .collect()
}

/// Pre-seed `n` entities with ids `ent-00000..ent-0000N`.
async fn seed(db: &Uni, n: usize) {
    let rows: Vec<Value> = (0..n)
        .map(|i| {
            let mut m = HashMap::new();
            m.insert("eid".to_string(), Value::String(format!("ent-{i:05}")));
            m.insert("name".to_string(), Value::String(format!("seed {i}")));
            Value::Map(m)
        })
        .collect();
    let tx = db.session().tx().await.unwrap();
    tx.query_with(
        "UNWIND $rows AS r \
         CREATE (n:Entity {entity_id: r.eid, name: r.name, frequency: 1, confidence: 0.5})",
    )
    .param("rows", Value::List(rows))
    .fetch_all()
    .await
    .unwrap();
    tx.commit().await.unwrap();
}

fn rows_to_values(rows: &[Row]) -> Vec<Value> {
    rows.iter()
        .map(|r| {
            let mut m = HashMap::new();
            m.insert("eid".to_string(), Value::String(r.eid.clone()));
            m.insert("name".to_string(), Value::String(r.name.clone()));
            Value::Map(m)
        })
        .collect()
}

/// The 4-phase path: (1) read existing by entity_id, (2) bulk CREATE new,
/// (3) bulk UPDATE existing by internal id. (MENTIONS edges, phase 4, are
/// identical for both strategies and omitted.)
async fn four_phase(db: &Uni, rows: &[Row]) {
    let tx = db.session().tx().await.unwrap();

    // Phase 1: find which ids already exist.
    let existing = tx
        .query_with(
            "UNWIND $rows AS r MATCH (n:Entity {entity_id: r.eid}) \
             RETURN r.eid AS eid, id(n) AS nid",
        )
        .param("rows", Value::List(rows_to_values(rows)))
        .fetch_all()
        .await
        .unwrap();
    let mut existing_nid: HashMap<String, i64> = HashMap::new();
    for row in existing.rows() {
        existing_nid.insert(
            row.get::<String>("eid").unwrap(),
            row.get::<i64>("nid").unwrap(),
        );
    }

    // Partition into new vs existing.
    let mut new_rows = Vec::new();
    let mut upd_rows = Vec::new();
    for r in rows {
        if let Some(&nid) = existing_nid.get(&r.eid) {
            let mut m = HashMap::new();
            m.insert("nid".to_string(), Value::Int(nid));
            upd_rows.push(Value::Map(m));
        } else {
            let mut m = HashMap::new();
            m.insert("eid".to_string(), Value::String(r.eid.clone()));
            m.insert("name".to_string(), Value::String(r.name.clone()));
            new_rows.push(Value::Map(m));
        }
    }

    // Phase 2: bulk create new.
    if !new_rows.is_empty() {
        tx.query_with(
            "UNWIND $rows AS r \
             CREATE (n:Entity {entity_id: r.eid, name: r.name, frequency: 1, confidence: 0.9})",
        )
        .param("rows", Value::List(new_rows))
        .fetch_all()
        .await
        .unwrap();
    }

    // Phase 3: bulk update existing (no label hint — RC7/#53/#54).
    if !upd_rows.is_empty() {
        tx.query_with(
            "UNWIND $rows AS u MATCH (n) WHERE id(n) = u.nid \
             SET n.frequency = coalesce(n.frequency, 0) + 1, n.confidence = 0.9",
        )
        .param("rows", Value::List(upd_rows))
        .fetch_all()
        .await
        .unwrap();
    }

    tx.commit().await.unwrap();
}

/// The single-statement path: `UNWIND … MERGE … ON CREATE/ON MATCH`.
async fn merge_path(db: &Uni, rows: &[Row]) {
    let tx = db.session().tx().await.unwrap();
    tx.query_with(
        "UNWIND $rows AS r \
         MERGE (n:Entity {entity_id: r.eid}) \
         ON CREATE SET n.name = r.name, n.frequency = 1, n.confidence = 0.9 \
         ON MATCH SET n.frequency = coalesce(n.frequency, 0) + 1, n.confidence = 0.9",
    )
    .param("rows", Value::List(rows_to_values(rows)))
    .fetch_all()
    .await
    .unwrap();
    tx.commit().await.unwrap();
}

#[tokio::test]
#[ignore = "perf bench; run with --run-ignored all"]
async fn bench_entity_upsert_merge_vs_four_phase() {
    const REPS: usize = 3;
    // Kept small: the 4-phase update leg is super-linear when run in the
    // same tx as the Phase-2 creates (it scans the uncommitted L0 buffer),
    // so larger batches blow up runtime. 200 is enough to show the trend.
    let configs = [(200usize, 0.0f64), (200, 0.5), (200, 1.0)];

    eprintln!("\n=== RC3: entity upsert — 4-phase vs UNWIND-MERGE ===");
    eprintln!("(median of {REPS} reps, fresh DB per rep, ms)\n");
    eprintln!(
        "{:>6}  {:>9}  {:>10}  {:>10}  {:>7}",
        "batch", "existing", "4-phase", "merge", "merge/4p"
    );

    for (batch, ratio) in configs {
        let n_existing = (batch as f64 * ratio) as usize;

        let mut four = Vec::with_capacity(REPS);
        let mut merge = Vec::with_capacity(REPS);

        for rep in 0..REPS {
            // 4-phase rep.
            {
                let db = fresh_db().await;
                if n_existing > 0 {
                    seed(&db, n_existing).await;
                }
                let rows = make_batch(batch, n_existing, rep);
                let t = Instant::now();
                four_phase(&db, &rows).await;
                four.push(t.elapsed().as_secs_f64() * 1000.0);
                db.shutdown().await.unwrap();
            }
            // merge rep.
            {
                let db = fresh_db().await;
                if n_existing > 0 {
                    seed(&db, n_existing).await;
                }
                let rows = make_batch(batch, n_existing, rep);
                let t = Instant::now();
                merge_path(&db, &rows).await;
                merge.push(t.elapsed().as_secs_f64() * 1000.0);
                db.shutdown().await.unwrap();
            }
        }

        four.sort_by(|a, b| a.partial_cmp(b).unwrap());
        merge.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let f = four[REPS / 2];
        let m = merge[REPS / 2];
        eprintln!(
            "{batch:>6}  {:>8.0}%  {f:>9.2}  {m:>9.2}  {:>7.2}x",
            ratio * 100.0,
            m / f
        );
    }
    eprintln!("\nmerge/4p < ~1.2 → MERGE competitive, dedup.rs can be simplified.");
    eprintln!("merge/4p >> 1   → keep the 4-phase split.\n");
}

/// RC7 verification: does dropping the `:Entity` label from
/// `UNWIND … MATCH (n) WHERE id(n)=u.nid SET …` regress on a MULTI-label
/// schema? (The main bench above is single-label and can't see it.)
#[tokio::test]
#[ignore = "perf bench; run with --run-ignored all"]
async fn bench_phase3_label_vs_nolabel_multilabel() {
    const N_LABELS: usize = 6;
    const ENTITIES: usize = 300;
    const PER_OTHER: usize = 300;
    const REPS: usize = 5;

    // Schema: Entity + 5 other labels, each Hash-indexed on its own id.
    async fn db_multi() -> Uni {
        let mut b = Uni::in_memory().build().await.unwrap();
        let _ = &mut b;
        let db = Uni::in_memory().build().await.unwrap();
        let mut s = db.schema();
        for i in 0..N_LABELS {
            let lbl = format!("L{i}");
            s = s
                .label(Box::leak(lbl.into_boxed_str()))
                .property("k", uni_db::DataType::String)
                .property_nullable("freq", uni_db::DataType::Int64)
                .done();
        }
        s.label("Entity")
            .property("entity_id", uni_db::DataType::String)
            .property_nullable("frequency", uni_db::DataType::Int64)
            .index(
                "entity_id",
                uni_db::IndexType::Scalar(uni_db::ScalarType::Hash),
            )
            .done()
            .apply()
            .await
            .unwrap();
        db
    }

    let mut with_label = Vec::new();
    let mut no_label = Vec::new();

    for _rep in 0..REPS {
        let db = db_multi().await;
        // Populate the 5 "other" labels (graph bulk) + Entity nodes.
        for i in 0..N_LABELS {
            let rows: Vec<Value> = (0..PER_OTHER)
                .map(|j| {
                    let mut m = HashMap::new();
                    m.insert("k".to_string(), Value::String(format!("l{i}-{j}")));
                    Value::Map(m)
                })
                .collect();
            let tx = db.session().tx().await.unwrap();
            tx.query_with(&format!("UNWIND $r AS x CREATE (n:L{i} {{k: x.k}})"))
                .param("r", Value::List(rows))
                .fetch_all()
                .await
                .unwrap();
            tx.commit().await.unwrap();
        }
        // Create Entity nodes, collect their ids.
        let erows: Vec<Value> = (0..ENTITIES)
            .map(|j| {
                let mut m = HashMap::new();
                m.insert("eid".to_string(), Value::String(format!("e{j}")));
                Value::Map(m)
            })
            .collect();
        let tx = db.session().tx().await.unwrap();
        let created = tx.query_with("UNWIND $r AS x CREATE (n:Entity {entity_id: x.eid, frequency: 1}) RETURN id(n) AS nid")
            .param("r", Value::List(erows)).fetch_all().await.unwrap();
        let nids: Vec<i64> = created
            .rows()
            .iter()
            .map(|r| r.get::<i64>("nid").unwrap())
            .collect();
        tx.commit().await.unwrap();

        let upd: Vec<Value> = nids
            .iter()
            .map(|&nid| {
                let mut m = HashMap::new();
                m.insert("nid".to_string(), Value::Int(nid));
                Value::Map(m)
            })
            .collect();

        // WITH label.
        {
            let tx = db.session().tx().await.unwrap();
            let t = Instant::now();
            tx.query_with(
                "UNWIND $u AS u MATCH (n:Entity) WHERE id(n) = u.nid SET n.frequency = 2",
            )
            .param("u", Value::List(upd.clone()))
            .fetch_all()
            .await
            .unwrap();
            tx.commit().await.unwrap();
            with_label.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        // WITHOUT label (current RC7 code).
        {
            let tx = db.session().tx().await.unwrap();
            let t = Instant::now();
            tx.query_with("UNWIND $u AS u MATCH (n) WHERE id(n) = u.nid SET n.frequency = 3")
                .param("u", Value::List(upd.clone()))
                .fetch_all()
                .await
                .unwrap();
            tx.commit().await.unwrap();
            no_label.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        db.shutdown().await.unwrap();
    }

    with_label.sort_by(|a, b| a.partial_cmp(b).unwrap());
    no_label.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("\n=== RC7: Phase-3 UPDATE-by-id, {N_LABELS} labels, {ENTITIES} updates ===");
    eprintln!("WITH :Entity : {:.1} ms (median)", with_label[REPS / 2]);
    eprintln!("NO label     : {:.1} ms (median)", no_label[REPS / 2]);
    eprintln!(
        "ratio nolabel/label: {:.2}x",
        no_label[REPS / 2] / with_label[REPS / 2]
    );
    eprintln!("(>>1 → RC7 regressed; dropping the label hurt)\n");
}
