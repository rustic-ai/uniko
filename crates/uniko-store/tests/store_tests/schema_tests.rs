//! Integration tests for schema registration, CRUD operations, and edge creation.

use uni_db::Uni;
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::{edges, labels};
use uniko_store::schema::register_schema;
use uniko_store::storage::embed_catalog;

async fn test_db() -> Uni {
    let config = UnikoConfig::default();
    let db = Uni::in_memory()
        .xervo_catalog(embed_catalog(&config))
        .build()
        .await
        .expect("in-memory db");
    register_schema(&db, &config)
        .await
        .expect("schema registration");
    db
}

// ── Registration ──

#[tokio::test]
async fn test_register_schema_succeeds() {
    let config = UnikoConfig::default();
    let db = Uni::in_memory()
        .xervo_catalog(embed_catalog(&config))
        .build()
        .await
        .expect("in-memory db");
    register_schema(&db, &config)
        .await
        .expect("schema registration");
    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_register_schema_idempotent() {
    let config = UnikoConfig::default();
    let db = Uni::in_memory()
        .xervo_catalog(embed_catalog(&config))
        .build()
        .await
        .expect("in-memory db");
    register_schema(&db, &config).await.expect("first call");
    register_schema(&db, &config)
        .await
        .expect("second call must not error");
    db.shutdown().await.unwrap();
}

// ── Label existence ──

#[tokio::test]
async fn test_all_labels_registered() {
    let db = test_db().await;
    for label in labels::ALL {
        assert!(
            db.label_exists(label).await.unwrap(),
            "missing label: {label}"
        );
    }
    db.shutdown().await.unwrap();
}

// ── Edge type existence ──

#[tokio::test]
async fn test_all_edge_types_registered() {
    let db = test_db().await;
    for edge in edges::ALL {
        assert!(
            db.edge_type_exists(edge).await.unwrap(),
            "missing edge type: {edge}"
        );
    }
    db.shutdown().await.unwrap();
}

// ── CRUD: Participant (Layer 0) ──

#[tokio::test]
async fn test_participant_crud() {
    let db = test_db().await;
    let session = db.session();

    // Create
    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Participant {participant_id: 'p-1', kind: 'agent', name: 'TestBot'})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    // Read
    let result = session
        .query("MATCH (p:Participant {participant_id: 'p-1'}) RETURN p.name, p.kind")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);
    let row = &result.rows()[0];
    assert_eq!(row.get::<String>("p.name").unwrap(), "TestBot");
    assert_eq!(row.get::<String>("p.kind").unwrap(), "agent");

    // Update
    let tx = session.tx().await.unwrap();
    tx.execute("MATCH (p:Participant {participant_id: 'p-1'}) SET p.name = 'UpdatedBot'")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (p:Participant {participant_id: 'p-1'}) RETURN p.name")
        .await
        .unwrap();
    assert_eq!(
        result.rows()[0].get::<String>("p.name").unwrap(),
        "UpdatedBot"
    );

    // Delete
    let tx = session.tx().await.unwrap();
    tx.execute("MATCH (p:Participant {participant_id: 'p-1'}) DETACH DELETE p")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (p:Participant {participant_id: 'p-1'}) RETURN p")
        .await
        .unwrap();
    assert_eq!(result.len(), 0);

    db.shutdown().await.unwrap();
}

// ── CRUD: Goal (Layer 1) ──

#[tokio::test]
async fn test_goal_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Goal {goal_id: 'g-1', title: 'Reduce latency', status: 'active'})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (g:Goal {goal_id: 'g-1'}) RETURN g.title, g.status")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(
        result.rows()[0].get::<String>("g.title").unwrap(),
        "Reduce latency"
    );

    db.shutdown().await.unwrap();
}

// ── CRUD: Task (Layer 1) ──

#[tokio::test]
async fn test_task_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Task {task_id: 't-1', title: 'Profile endpoints', status: 'pending'})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (t:Task {task_id: 't-1'}) RETURN t.title")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: Session (Layer 1) ──

#[tokio::test]
async fn test_session_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Session {session_id: 's-1', started_at: datetime(), topic: 'debugging'})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (s:Session {session_id: 's-1'}) RETURN s.topic")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: Message (Layer 2) ──

#[tokio::test]
async fn test_message_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute(
        "CREATE (:Message {message_id: 'm-1', content: 'Hello world', timestamp: datetime()})",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (m:Message {message_id: 'm-1'}) RETURN m.content")
        .await
        .unwrap();
    assert_eq!(
        result.rows()[0].get::<String>("m.content").unwrap(),
        "Hello world"
    );

    db.shutdown().await.unwrap();
}

// ── CRUD: Action (Layer 2) ──

#[tokio::test]
async fn test_action_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Action {action_id: 'a-1', action_type: 'tool_call', status: 'success'})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (a:Action {action_id: 'a-1'}) RETURN a.action_type")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: Episode (Layer 2) ──

#[tokio::test]
async fn test_episode_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute(
        "CREATE (:Episode {episode_id: 'e-1', action_type: 'investigate', timestamp: datetime()})",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (e:Episode {episode_id: 'e-1'}) RETURN e.action_type")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: Artifact (Layer 3) ──

#[tokio::test]
async fn test_artifact_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute(
        "CREATE (:Artifact {artifact_id: 'art-1', kind: 'file', path: '/tmp/test.rs', language: 'rust'})",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (a:Artifact {artifact_id: 'art-1'}) RETURN a.kind, a.language")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result.rows()[0].get::<String>("a.kind").unwrap(), "file");

    db.shutdown().await.unwrap();
}

// ── CRUD: Chunk (Layer 3) ──

#[tokio::test]
async fn test_chunk_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute(
        "CREATE (:Chunk {chunk_id: 'art-1:0', text: 'fn main() {}', chunk_type: 'function', language: 'rust'})",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (c:Chunk {chunk_id: 'art-1:0'}) RETURN c.text")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: Entity (Layer 4) ──

#[tokio::test]
async fn test_entity_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Entity {entity_id: 'ent-1', name: 'Rust', entity_type: 'concept'})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (e:Entity {entity_id: 'ent-1'}) RETURN e.name, e.entity_type")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result.rows()[0].get::<String>("e.name").unwrap(), "Rust");

    db.shutdown().await.unwrap();
}

// ── CRUD: Observation (Layer 4) ──

#[tokio::test]
async fn test_observation_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute(
        "CREATE (:Observation {observation_id: 'obs-1', content: 'User prefers Rust', subject: 'user'})",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (o:Observation {observation_id: 'obs-1'}) RETURN o.content")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: Fact (Layer 4, with BTIC) ──

#[tokio::test]
async fn test_fact_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute(
        "CREATE (:Fact {fact_id: 'f-1', subject: 'user', predicate: 'prefers', object: 'Rust', confidence: 0.85})",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (f:Fact {fact_id: 'f-1'}) RETURN f.subject, f.predicate, f.confidence")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(
        result.rows()[0].get::<String>("f.predicate").unwrap(),
        "prefers"
    );

    db.shutdown().await.unwrap();
}

// ── CRUD: Topic (Layer 4) ──

#[tokio::test]
async fn test_topic_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Topic {topic_id: 'top-1', name: 'Programming Languages'})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (t:Topic {topic_id: 'top-1'}) RETURN t.name")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: Summary (Layer 4) ──

#[tokio::test]
async fn test_summary_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Summary {summary_id: 'sum-1', text: 'Session covered debugging topics', level: 'session'})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (s:Summary {summary_id: 'sum-1'}) RETURN s.level")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: Procedure (Layer 5) ──

#[tokio::test]
async fn test_procedure_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute(
        "CREATE (:Procedure {procedure_id: 'proc-1', name: 'Debug workflow', status: 'active'})",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (p:Procedure {procedure_id: 'proc-1'}) RETURN p.name")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: Rule (Layer 5) ──

#[tokio::test]
async fn test_rule_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute(
        "CREATE (:Rule {rule_id: 'r-1', name: 'contradiction_detector', source_type: 'stdlib', status: 'active'})",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (r:Rule {rule_id: 'r-1'}) RETURN r.name, r.source_type")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: ConsolidationCycle (Layer 6) ──

#[tokio::test]
async fn test_consolidation_cycle_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute(
        "CREATE (:ConsolidationCycle {cycle_id: 'cc-1', agent_id: 'agent-1', facts_created: 5})",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (c:ConsolidationCycle {cycle_id: 'cc-1'}) RETURN c.agent_id")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: DeadLetter (Layer 6) ──

#[tokio::test]
async fn test_dead_letter_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:DeadLetter {step: 'ner', error: 'timeout', retry_count: 1})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (d:DeadLetter) RETURN d.step, d.error")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: Organization (Layer 7) ──

#[tokio::test]
async fn test_organization_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Organization {org_id: 'org-1', name: 'Acme Corp'})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (o:Organization {org_id: 'org-1'}) RETURN o.name")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── CRUD: Team (Layer 7) ──

#[tokio::test]
async fn test_team_crud() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Team {team_id: 'team-1', name: 'Backend', purpose: 'API development'})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (t:Team {team_id: 'team-1'}) RETURN t.name")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── MERGE idempotency ──

#[tokio::test]
async fn test_merge_idempotent() {
    let db = test_db().await;
    let session = db.session();

    // First MERGE creates the node — all required properties in the pattern
    let tx = session.tx().await.unwrap();
    tx.execute(
        "MERGE (p:Participant {participant_id: 'p-merge', kind: 'agent'}) ON CREATE SET p.name = 'Bot'",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    // Second MERGE with same identity — should match, not create
    let tx = session.tx().await.unwrap();
    tx.execute(
        "MERGE (p:Participant {participant_id: 'p-merge', kind: 'agent'}) ON MATCH SET p.name = 'UpdatedBot'",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (p:Participant {participant_id: 'p-merge'}) RETURN p.name")
        .await
        .unwrap();
    assert_eq!(result.len(), 1, "MERGE should not create duplicates");
    assert_eq!(
        result.rows()[0].get::<String>("p.name").unwrap(),
        "UpdatedBot"
    );

    db.shutdown().await.unwrap();
}

// ── Edge creation ──

#[tokio::test]
async fn test_edge_owned_by() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Participant {participant_id: 'p-1', kind: 'human', name: 'Alice'})")
        .await
        .unwrap();
    tx.execute("CREATE (:Goal {goal_id: 'g-1', title: 'Ship v2'})")
        .await
        .unwrap();
    tx.execute(
        "MATCH (g:Goal {goal_id: 'g-1'}), (p:Participant {participant_id: 'p-1'}) CREATE (g)-[:OWNED_BY]->(p)",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (g:Goal)-[:OWNED_BY]->(p:Participant) RETURN g.title, p.name")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_edge_in_session_multi_source() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Session {session_id: 's-1', started_at: datetime()})")
        .await
        .unwrap();
    tx.execute("CREATE (:Message {message_id: 'm-1', content: 'hi', timestamp: datetime()})")
        .await
        .unwrap();
    tx.execute("CREATE (:Action {action_id: 'a-1', action_type: 'tool_call'})")
        .await
        .unwrap();
    // Both Message and Action can link to Session via IN_SESSION
    tx.execute(
        "MATCH (m:Message {message_id: 'm-1'}), (s:Session {session_id: 's-1'}) CREATE (m)-[:IN_SESSION]->(s)",
    )
    .await
    .unwrap();
    tx.execute(
        "MATCH (a:Action {action_id: 'a-1'}), (s:Session {session_id: 's-1'}) CREATE (a)-[:IN_SESSION]->(s)",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH ()-[:IN_SESSION]->(s:Session {session_id: 's-1'}) RETURN count(*) AS cnt")
        .await
        .unwrap();
    assert_eq!(result.rows()[0].get::<i64>("cnt").unwrap(), 2);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_edge_with_properties() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Message {message_id: 'm-1', content: 'hello', timestamp: datetime()})")
        .await
        .unwrap();
    tx.execute("CREATE (:Message {message_id: 'm-2', content: 'world', timestamp: datetime()})")
        .await
        .unwrap();
    tx.execute(
        "MATCH (a:Message {message_id: 'm-1'}), (b:Message {message_id: 'm-2'}) CREATE (a)-[:NEXT {gap_ms: 500}]->(b)",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (:Message)-[r:NEXT]->(:Message) RETURN r.gap_ms")
        .await
        .unwrap();
    assert_eq!(result.rows()[0].get::<i64>("r.gap_ms").unwrap(), 500);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_edge_has_chunk() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Artifact {artifact_id: 'art-1', kind: 'file'})")
        .await
        .unwrap();
    tx.execute("CREATE (:Chunk {chunk_id: 'art-1:0', text: 'first chunk'})")
        .await
        .unwrap();
    tx.execute(
        "MATCH (a:Artifact {artifact_id: 'art-1'}), (c:Chunk {chunk_id: 'art-1:0'}) CREATE (a)-[:HAS_CHUNK {index: 0}]->(c)",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (a:Artifact)-[r:HAS_CHUNK]->(c:Chunk) RETURN r.index, c.chunk_id")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result.rows()[0].get::<i64>("r.index").unwrap(), 0);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_edge_mentions() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Entity {entity_id: 'ent-1', name: 'Rust'})")
        .await
        .unwrap();
    tx.execute(
        "CREATE (:Message {message_id: 'm-1', content: 'I love Rust', timestamp: datetime()})",
    )
    .await
    .unwrap();
    tx.execute(
        "MATCH (m:Message {message_id: 'm-1'}), (e:Entity {entity_id: 'ent-1'}) CREATE (m)-[:MENTIONS {count: 3}]->(e)",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (:Message)-[r:MENTIONS]->(e:Entity) RETURN r.count, e.name")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result.rows()[0].get::<i64>("r.count").unwrap(), 3);

    db.shutdown().await.unwrap();
}

// ── Cross-layer edges ──

#[tokio::test]
async fn test_episode_mentions_entity() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute(
        "CREATE (:Episode {episode_id: 'ep-1', action_type: 'investigate', timestamp: datetime()})",
    )
    .await
    .unwrap();
    tx.execute("CREATE (:Entity {entity_id: 'ent-1', name: 'Tokio'})")
        .await
        .unwrap();
    tx.execute(
        "MATCH (ep:Episode {episode_id: 'ep-1'}), (e:Entity {entity_id: 'ent-1'}) CREATE (ep)-[:MENTIONS]->(e)",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (:Episode)-[:MENTIONS]->(e:Entity) RETURN e.name")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_fact_supported_by_observation() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Observation {observation_id: 'obs-1', content: 'User uses Rust daily'})")
        .await
        .unwrap();
    tx.execute(
        "CREATE (:Fact {fact_id: 'f-1', subject: 'user', predicate: 'uses', object: 'Rust'})",
    )
    .await
    .unwrap();
    tx.execute(
        "MATCH (f:Fact {fact_id: 'f-1'}), (o:Observation {observation_id: 'obs-1'}) CREATE (f)-[:SUPPORTED_BY {weight: 0.9}]->(o)",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (f:Fact)-[r:SUPPORTED_BY]->(o:Observation) RETURN r.weight")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_summary_summarizes_session() {
    let db = test_db().await;
    let session = db.session();

    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Session {session_id: 's-1', started_at: datetime()})")
        .await
        .unwrap();
    tx.execute(
        "CREATE (:Summary {summary_id: 'sum-1', text: 'Covered debugging', level: 'session'})",
    )
    .await
    .unwrap();
    tx.execute(
        "MATCH (sum:Summary {summary_id: 'sum-1'}), (s:Session {session_id: 's-1'}) CREATE (sum)-[:SUMMARIZES]->(s)",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (:Summary)-[:SUMMARIZES]->(s:Session) RETURN s.session_id")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

/// The NLP alias carries the option keys of the configured generation only:
/// uni-xervo rejects a v1 key on a v7 model and the reverse.
#[test]
fn nlp_catalog_options_follow_the_generation() {
    use uniko_store::config::NlpConfig;
    use uniko_store::schema::NLP_ALIAS;

    let options = |nlp: NlpConfig| {
        let config = UnikoConfig {
            nlp,
            ..UnikoConfig::default()
        };
        embed_catalog(&config)
            .into_iter()
            .find(|spec| spec.alias == NLP_ALIAS)
            .expect("nlp alias")
    };

    let v1 = options(NlpConfig::default());
    assert_eq!(v1.options["generation"], "v1");
    assert_eq!(v1.options["onnx_path"], "onnx/cascade-int8.onnx");
    assert!(v1.options.get("variant").is_none());

    let v7 = options(NlpConfig::preset("v7-small").unwrap());
    assert_eq!(v7.model_id, "dragonscale-ai/kniv-v7-small-en");
    assert_eq!(v7.options["generation"], "v7");
    assert_eq!(v7.options["variant"], "int8");
    for v1_only in [
        "onnx_path",
        "tokenizer_path",
        "label_maps_path",
        "max_seq_len",
    ] {
        assert!(v7.options.get(v1_only).is_none(), "{v1_only}");
    }
}
