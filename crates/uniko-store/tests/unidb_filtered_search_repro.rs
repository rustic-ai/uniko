//! Isolated uni-db repros for filtered multi-vector and sparse search.
//!
//! Plain uni-db, no models and no uniko code: every vector is written by hand.
//!
//! 1. `multivector_filter_*`: `uni.vector.query` over a `List(Vector)` column
//!    with a `_vid IN (…)` filter. On unflushed rows it returns a vid that is
//!    NOT in the filter and drops one that is. After `flush()` it is correct.
//! 2. `sparse_filter_*`: `uni.sparse.query` rejects any non-null filter, and
//!    the sparse arm of `uni.search` drops the filter without saying so.
//!
//! The failing cases are `#[ignore]`d until uni-db fixes them
//! (rustic-ai/uni-db#343, #344); the flushed and dense controls run always.
//!
//! Run:
//!   cargo nextest run -p uniko-store --test unidb_filtered_search_repro \
//!       --run-ignored all --no-capture

use std::collections::HashMap;

use uni_db::{DataType, IndexType, Uni, Value, VectorAlgo, VectorIndexCfg, VectorMetric};

/// Unit 4-d vector along `axis`, with a small `off` on axis 3 so every
/// document scores differently.
fn token(axis: usize, off: f32) -> Value {
    let mut v = vec![0.0_f32; 4];
    v[axis] = 1.0;
    v[3] = off;
    Value::Vector(v)
}

/// Seven documents. Doc 6 matches the query best; the filter excludes it.
fn docs() -> Vec<Value> {
    (0..7)
        .map(|i| {
            let off = if i == 6 { 0.0 } else { 0.1 * (i as f32 + 1.0) };
            Value::List(vec![token(0, off), token(1, off)])
        })
        .collect()
}

async fn multivector_db() -> (Uni, Vec<i64>) {
    let db = Uni::in_memory().build().await.unwrap();
    db.schema()
        .label("Doc")
        .property("n", DataType::Int64)
        .property_nullable(
            "mv",
            DataType::List(Box::new(DataType::Vector { dimensions: 4 })),
        )
        .index(
            "mv",
            IndexType::Vector(VectorIndexCfg {
                algorithm: VectorAlgo::Flat,
                metric: VectorMetric::Cosine,
                embedding: None,
            }),
        )
        .done()
        .apply()
        .await
        .unwrap();
    let rows = docs()
        .into_iter()
        .enumerate()
        .map(|(i, mv)| {
            HashMap::from([
                ("n".to_string(), Value::Int(i as i64)),
                ("mv".to_string(), mv),
            ])
        })
        .collect();
    let tx = db.session().tx().await.unwrap();
    tx.bulk_insert_vertices("Doc", rows).await.unwrap();
    tx.commit().await.unwrap();
    let res = db
        .session()
        .query("MATCH (d:Doc) RETURN id(d) AS vid ORDER BY d.n")
        .await
        .unwrap();
    let vids = res
        .rows()
        .iter()
        .map(|r| r.get::<i64>("vid").unwrap())
        .collect();
    (db, vids)
}

/// Ask for the six allowed vids with `k = 6` and return the vids that came back.
async fn filtered_maxsim(db: &Uni, allowed: &[i64]) -> Vec<i64> {
    let filter = format!(
        "_vid IN ({})",
        allowed
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let q = Value::List(vec![token(0, 0.0), token(1, 0.0)]);
    let res = db
        .session()
        .query_with(
            "CALL uni.vector.query('Doc', 'mv', $q, $k, $filter, null, {}) \
             YIELD vid, score RETURN vid, score ORDER BY score DESC",
        )
        .param("q", q)
        .param("k", allowed.len() as i64)
        .param("filter", filter)
        .fetch_all()
        .await
        .unwrap();
    for r in res.rows() {
        println!(
            "  vid={} score={:.4}",
            r.get::<i64>("vid").unwrap(),
            r.get::<f64>("score").unwrap()
        );
    }
    let mut got: Vec<i64> = res
        .rows()
        .iter()
        .map(|r| r.get::<i64>("vid").unwrap())
        .collect();
    got.sort_unstable();
    got
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "fails until rustic-ai/uni-db#343 is fixed; run with --run-ignored all"]
async fn multivector_filter_on_unflushed_rows() {
    let (db, vids) = multivector_db().await;
    let allowed: Vec<i64> = vids[..6].to_vec();
    println!(
        "all vids {vids:?}; filter allows {allowed:?}; excluded {}",
        vids[6]
    );
    let got = filtered_maxsim(&db, &allowed).await;
    println!("unflushed: got {got:?}");
    assert_eq!(
        got, allowed,
        "a filtered query must return only, and all of, the allowed vids"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn multivector_filter_after_flush() {
    let (db, vids) = multivector_db().await;
    db.flush().await.unwrap();
    let allowed: Vec<i64> = vids[..6].to_vec();
    let got = filtered_maxsim(&db, &allowed).await;
    println!("flushed: got {got:?}");
    assert_eq!(got, allowed);
}

async fn sparse_db() -> Uni {
    let db = Uni::in_memory().build().await.unwrap();
    db.schema()
        .label("Doc")
        .property("n", DataType::Int64)
        .property_nullable("sp", DataType::SparseVector { dimensions: 16 })
        .index("sp", IndexType::sparse(16))
        .done()
        .apply()
        .await
        .unwrap();
    // Doc 0 is the best match; docs 1-3 share term 1 with smaller weights.
    let weights = [0.9_f32, 0.6, 0.4, 0.2];
    let rows = weights
        .iter()
        .enumerate()
        .map(|(i, w)| {
            HashMap::from([
                ("n".to_string(), Value::Int(i as i64)),
                (
                    "sp".to_string(),
                    Value::SparseVector {
                        indices: vec![1, 7],
                        values: vec![*w, 0.1],
                    },
                ),
            ])
        })
        .collect();
    let tx = db.session().tx().await.unwrap();
    tx.bulk_insert_vertices("Doc", rows).await.unwrap();
    tx.commit().await.unwrap();
    db.flush().await.unwrap();
    db
}

fn sparse_query() -> Value {
    Value::SparseVector {
        indices: vec![1],
        values: vec![1.0],
    }
}

async fn doc_vids(db: &Uni) -> Vec<i64> {
    let res = db
        .session()
        .query("MATCH (d:Doc) RETURN id(d) AS vid ORDER BY d.n")
        .await
        .unwrap();
    res.rows()
        .iter()
        .map(|r| r.get::<i64>("vid").unwrap())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "fails until rustic-ai/uni-db#344 is fixed; run with --run-ignored all"]
async fn sparse_filter_is_rejected() {
    let db = sparse_db().await;
    let vids = doc_vids(&db).await;
    let filter = format!("_vid IN ({},{})", vids[2], vids[3]);
    let unfiltered = db
        .session()
        .query_with(
            "CALL uni.sparse.query('Doc', 'sp', $q, 2, null, null, {}) \
             YIELD vid, score RETURN vid, score ORDER BY score DESC",
        )
        .param("q", sparse_query())
        .fetch_all()
        .await
        .unwrap();
    println!(
        "unfiltered top-2: {:?}",
        unfiltered
            .rows()
            .iter()
            .map(|r| r.get::<i64>("vid").unwrap())
            .collect::<Vec<_>>()
    );
    let filtered = db
        .session()
        .query_with(
            "CALL uni.sparse.query('Doc', 'sp', $q, 2, $filter, null, {}) \
             YIELD vid, score RETURN vid, score ORDER BY score DESC",
        )
        .param("q", sparse_query())
        .param("filter", filter.clone())
        .fetch_all()
        .await;
    match &filtered {
        Ok(r) => println!(
            "filtered ({filter}): {:?}",
            r.rows()
                .iter()
                .map(|r| r.get::<i64>("vid").unwrap())
                .collect::<Vec<_>>()
        ),
        Err(e) => println!("filtered ({filter}): ERROR {e}"),
    }
    let got: Vec<i64> = filtered
        .expect("a filtered sparse query should run")
        .rows()
        .iter()
        .map(|r| r.get::<i64>("vid").unwrap())
        .collect();
    assert_eq!(got, vec![vids[2], vids[3]]);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "fails until rustic-ai/uni-db#344 is fixed; run with --run-ignored all"]
async fn sparse_arm_of_search_ignores_the_filter() {
    let db = sparse_db().await;
    let vids = doc_vids(&db).await;
    let filter = format!("_vid IN ({},{})", vids[2], vids[3]);
    let res = db
        .session()
        .query_with(
            "CALL uni.search('Doc', {sparse: 'sp'}, '', null, 2, $filter, \
                             {sparse_query: $q}) \
             YIELD vid, score RETURN vid, score ORDER BY score DESC",
        )
        .param("q", sparse_query())
        .param("filter", filter.clone())
        .fetch_all()
        .await;
    let got: Vec<i64> = match res {
        Ok(r) => r
            .rows()
            .iter()
            .map(|r| r.get::<i64>("vid").unwrap())
            .collect(),
        Err(e) => panic!("uni.search failed: {e}"),
    };
    println!("uni.search sparse arm with {filter}: got {got:?}");
    for v in &got {
        assert!(
            *v == vids[2] || *v == vids[3],
            "vid {v} is outside the filter {filter}"
        );
    }
}

/// Control: the same unflushed filtered query over a plain dense column.
#[tokio::test(flavor = "multi_thread")]
async fn dense_filter_on_unflushed_rows() {
    let db = Uni::in_memory().build().await.unwrap();
    db.schema()
        .label("Doc")
        .property("n", DataType::Int64)
        .property_nullable("v", DataType::Vector { dimensions: 4 })
        .index(
            "v",
            IndexType::Vector(VectorIndexCfg {
                algorithm: VectorAlgo::Flat,
                metric: VectorMetric::Cosine,
                embedding: None,
            }),
        )
        .done()
        .apply()
        .await
        .unwrap();
    let rows = (0..7)
        .map(|i| {
            let off = if i == 6 { 0.0 } else { 0.1 * (i as f32 + 1.0) };
            HashMap::from([
                ("n".to_string(), Value::Int(i)),
                ("v".to_string(), Value::Vector(vec![1.0, 0.0, 0.0, off])),
            ])
        })
        .collect();
    let tx = db.session().tx().await.unwrap();
    tx.bulk_insert_vertices("Doc", rows).await.unwrap();
    tx.commit().await.unwrap();
    let vids = doc_vids(&db).await;
    let allowed = &vids[..6];
    let filter = format!(
        "_vid IN ({})",
        allowed
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let res = db
        .session()
        .query_with(
            "CALL uni.vector.query('Doc', 'v', $q, 6, $filter, null, {}) \
             YIELD vid RETURN vid",
        )
        .param("q", Value::Vector(vec![1.0, 0.0, 0.0, 0.0]))
        .param("filter", filter)
        .fetch_all()
        .await
        .unwrap();
    let mut got: Vec<i64> = res
        .rows()
        .iter()
        .map(|r| r.get::<i64>("vid").unwrap())
        .collect();
    got.sort_unstable();
    println!("dense unflushed: got {got:?}");
    assert_eq!(got, allowed);
}
