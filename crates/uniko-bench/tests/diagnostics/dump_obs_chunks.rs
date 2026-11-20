//! Dump all observation chunks to a file for analysis.
//!
//! Run: cargo nextest run -p uniko-bench --test integration diagnostics::dump_obs_chunks --nocapture --run-ignored all

use crate::common::load_kb;

#[tokio::test]
#[ignore]
async fn dump_observation_chunks() {
    let kb = load_kb("data/kb/conv-30").await;
    let session = kb.db().session();

    let cypher = "\
        MATCH (s:Session)-[:HAS_CHUNK]->(c:Chunk {chunk_type: 'observation'}) \
        OPTIONAL MATCH (c)-[:ABOUT]->(e:Entity) \
        RETURN s.session_id AS session_id, \
               id(c) AS chunk_id, \
               c.text AS text, \
               c.chunk_type AS chunk_type, \
               c.symbol_name AS subjects, \
               collect(e.name) AS entities \
        ORDER BY s.session_id";

    let result = session.query_with(cypher).fetch_all().await.unwrap();

    let mut output = String::new();
    for row in result.rows() {
        let session_id: String = row.get("session_id").unwrap_or_default();
        let chunk_id: i64 = row.get("chunk_id").unwrap_or(0);
        let text: String = row.get("text").unwrap_or_default();
        let subjects: String = row.get("subjects").unwrap_or_default();
        let entities: String = row.get::<String>("entities").unwrap_or_default();

        output.push_str("======================================================================\n");
        output.push_str(&format!("Session: {session_id}\n"));
        output.push_str(&format!("Chunk ID: {chunk_id}\n"));
        output.push_str(&format!("Subjects: {subjects}\n"));
        output.push_str(&format!("ABOUT entities: {entities}\n"));
        output.push_str(&format!("Text ({} chars):\n", text.len()));
        output.push_str(&text);
        output.push_str("\n\n");
    }

    let path = "data/observation_chunks_dump.txt";
    std::fs::write(path, &output).unwrap();
    eprintln!("Wrote {} chunks to {path}", result.len());
    eprintln!("{output}");
}
