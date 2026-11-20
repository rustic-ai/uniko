//! One-shot quality review of Observation nodes from conv-26 KB.
//!
//! Run: cargo nextest run -p uniko-bench --test integration diagnostics::review_observations \
//!   --no-capture --run-ignored all

use crate::common::load_kb;

#[tokio::test]
#[ignore]
async fn review_observations_quality() {
    let kb = load_kb("data/kb_llm/conv-26").await;
    let session = kb.db().session();

    // 1. Counts.
    let total: i64 = session
        .query_with("MATCH (o:Observation) RETURN count(o) AS n")
        .fetch_all()
        .await
        .unwrap()
        .rows()
        .first()
        .and_then(|r| r.get::<i64>("n").ok())
        .unwrap_or(0);
    eprintln!("=== Observation totals ===");
    eprintln!("Total observations: {total}");

    // 2. ABOUT edge coverage.
    let with_about: i64 = session
        .query_with("MATCH (o:Observation)-[:ABOUT]->(t) RETURN count(DISTINCT o) AS n")
        .fetch_all()
        .await
        .unwrap()
        .rows()
        .first()
        .and_then(|r| r.get::<i64>("n").ok())
        .unwrap_or(0);
    eprintln!(
        "Observations with at least one ABOUT edge: {with_about} ({:.1}%)",
        100.0 * with_about as f64 / total.max(1) as f64
    );

    // 3a. ABOUT to Participant (force-named).
    let to_p: i64 = session
        .query_with("MATCH (o:Observation)-[:ABOUT]->(p:Participant) RETURN count(*) AS n")
        .fetch_all()
        .await
        .unwrap()
        .rows()
        .first()
        .and_then(|r| r.get::<i64>("n").ok())
        .unwrap_or(0);
    eprintln!("ABOUT -> Participant edges: {to_p}");
    let to_e: i64 = session
        .query_with("MATCH (o:Observation)-[:ABOUT]->(e:Entity) RETURN count(*) AS n")
        .fetch_all()
        .await
        .unwrap()
        .rows()
        .first()
        .and_then(|r| r.get::<i64>("n").ok())
        .unwrap_or(0);
    eprintln!("ABOUT -> Entity edges: {to_e}");

    // 3b. List the actual target nodes' labels.
    let target_label_combos = session
        .query_with(
            "MATCH (o:Observation)-[:ABOUT]->(t) \
             RETURN labels(t) AS lbls, t.name AS name, count(*) AS c \
             ORDER BY c DESC LIMIT 10",
        )
        .fetch_all()
        .await
        .unwrap();
    eprintln!("Top 10 target nodes (with ALL labels):");
    for row in target_label_combos.rows() {
        let lbls = row.get::<Vec<String>>("lbls").unwrap_or_default();
        let name: String = row.get("name").unwrap_or_default();
        let c: i64 = row.get("c").unwrap_or(0);
        eprintln!("  {lbls:?}  name={name:25}  edges={c}");
    }

    // 3c. Find the empty-name Entity and dump its full property bag.
    let empty_node = session
        .query_with(
            "MATCH (e:Entity) WHERE e.name = '' OR e.name IS NULL \
             RETURN id(e) AS nid, properties(e) AS props LIMIT 5",
        )
        .fetch_all()
        .await
        .unwrap();
    eprintln!("Empty-named Entity nodes:");
    for row in empty_node.rows() {
        let nid: i64 = row.get("nid").unwrap_or(0);
        eprintln!("  nid={nid} (props omitted)");
    }

    // 3d. What labels does Caroline-the-Participant carry?
    let caroline = session
        .query_with(
            "MATCH (n) WHERE n.name = 'Caroline' RETURN id(n) AS nid, labels(n) AS lbls, properties(n) AS props",
        )
        .fetch_all()
        .await
        .unwrap();
    eprintln!("Nodes named 'Caroline':");
    for row in caroline.rows() {
        let nid: i64 = row.get("nid").unwrap_or(0);
        let lbls = row.get::<Vec<String>>("lbls").unwrap_or_default();
        eprintln!("  nid={nid} labels={lbls:?}");
    }

    // 3e. The actual top targets: id + labels + name as fetched by direct query.
    let target_ids = session
        .query_with(
            "MATCH (o:Observation)-[:ABOUT]->(t) \
             RETURN id(t) AS tid, count(*) AS c \
             ORDER BY c DESC LIMIT 5",
        )
        .fetch_all()
        .await
        .unwrap();
    eprintln!("Top 5 ABOUT target node IDs:");
    for row in target_ids.rows() {
        let tid: i64 = row.get("tid").unwrap_or(0);
        let c: i64 = row.get("c").unwrap_or(0);
        // Look up the node properties directly.
        let info = session
            .query_with(&format!(
                "MATCH (n) WHERE id(n) = {tid} \
                 RETURN labels(n) AS lbls, n.name AS name, n.participant_id AS pid"
            ))
            .fetch_all()
            .await
            .unwrap();
        for r in info.rows() {
            let lbls = r.get::<Vec<String>>("lbls").unwrap_or_default();
            let name: String = r.get("name").unwrap_or_default();
            let pid: String = r.get("pid").unwrap_or_default();
            eprintln!("  tid={tid:5} edges={c:3} labels={lbls:?} name='{name}' pid='{pid}'");
        }
    }

    // 3. ABOUT target distribution by label.
    eprintln!("\n=== ABOUT target label distribution ===");
    let by_label = session
        .query_with(
            "MATCH (o:Observation)-[:ABOUT]->(t) \
             RETURN labels(t)[0] AS lbl, count(*) AS c ORDER BY c DESC",
        )
        .fetch_all()
        .await
        .unwrap();
    for row in by_label.rows() {
        let lbl: String = row.get("lbl").unwrap_or_default();
        let c: i64 = row.get("c").unwrap_or(0);
        eprintln!("  {lbl:20} {c}");
    }

    // 4. Subject distribution (most common subjects).
    eprintln!("\n=== Top 15 subjects ===");
    let by_subj = session
        .query_with(
            "MATCH (o:Observation) \
             RETURN o.subject AS subj, count(*) AS c ORDER BY c DESC LIMIT 15",
        )
        .fetch_all()
        .await
        .unwrap();
    for row in by_subj.rows() {
        let subj: String = row.get("subj").unwrap_or_default();
        let c: i64 = row.get("c").unwrap_or(0);
        eprintln!("  {subj:30}  {c}");
    }

    // 5. Confidence distribution.
    eprintln!("\n=== Confidence buckets ===");
    let conf = session
        .query_with(
            "MATCH (o:Observation) \
             RETURN o.confidence AS c",
        )
        .fetch_all()
        .await
        .unwrap();
    let mut buckets = [0; 11];
    let mut none_conf = 0;
    for row in conf.rows() {
        match row.get::<f64>("c") {
            Ok(v) => {
                let b = ((v * 10.0).round() as usize).min(10);
                buckets[b] += 1;
            }
            Err(_) => none_conf += 1,
        }
    }
    for (i, n) in buckets.iter().enumerate() {
        if *n > 0 {
            eprintln!("  conf~{:.1}: {}", i as f64 / 10.0, n);
        }
    }
    eprintln!("  conf=NULL: {none_conf}");

    // 6. Empty / very short content detection.
    eprintln!("\n=== Quality flags ===");
    let empty = session
        .query_with(
            "MATCH (o:Observation) WHERE o.content IS NULL OR o.content = '' \
             RETURN count(o) AS n",
        )
        .fetch_all()
        .await
        .unwrap();
    let n_empty: i64 = empty
        .rows()
        .first()
        .and_then(|r| r.get::<i64>("n").ok())
        .unwrap_or(0);
    eprintln!("Empty content: {n_empty}");

    // 7. Sample 25 observations + their ABOUT entities.
    eprintln!("\n=== Sample 25 observations ===");
    let samples = session
        .query_with(
            "MATCH (o:Observation) \
             OPTIONAL MATCH (o)-[:ABOUT]->(t) \
             RETURN o.content AS content, o.subject AS subj, \
                    o.confidence AS conf, \
                    collect({label: labels(t)[0], name: t.name}) AS abouts \
             LIMIT 25",
        )
        .fetch_all()
        .await
        .unwrap();
    for (i, row) in samples.rows().iter().enumerate() {
        let content: String = row.get("content").unwrap_or_default();
        let subj: String = row.get("subj").unwrap_or_default();
        let conf: f64 = row.get("conf").unwrap_or(0.0);
        eprintln!(
            "[{i:2}] subj={subj:20} conf={conf:.2}  \"{}\"",
            content.chars().take(120).collect::<String>(),
        );
    }

    // 8. Per-session counts.
    eprintln!("\n=== Per-session counts ===");
    let per_sess = session
        .query_with(
            "MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message)-[:IN_SESSION]->(s:Session) \
             RETURN s.session_id AS sid, count(DISTINCT o) AS n \
             ORDER BY sid",
        )
        .fetch_all()
        .await
        .unwrap();
    for row in per_sess.rows() {
        let sid: String = row.get("sid").unwrap_or_default();
        let n: i64 = row.get("n").unwrap_or(0);
        eprintln!("  {sid:30} {n}");
    }
}
