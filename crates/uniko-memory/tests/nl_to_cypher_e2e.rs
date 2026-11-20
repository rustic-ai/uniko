//! Integration tests for [`uniko_memory::nl_to_cypher::translate`].
//!
//! These tests register `chat/default` → gpt-5-nano via OpenAI and
//! make real LLM calls.  They fail loudly when `OPENAI_API_KEY` is
//! not set — there's no graceful-skip path here.  The point of an
//! integration test is to exercise the integration; if the runtime
//! can't talk to the model, the test must report that, not pretend
//! to pass.

use uniko_memory::nl_to_cypher::{is_safe_read_only, translate};
use uniko_store::UnikoError;

use crate::common::{CHAT_ALIAS, kb, kb_with_chat, require_openai_key};

#[tokio::test]
async fn translate_rejects_empty_query() {
    // No LLM needed — this path errors out before contacting any
    // model.  Use the bare KB constructor (no chat alias) to prove
    // the early-rejection path doesn't depend on a registered model.
    let kb = kb().await;
    let err = translate(&kb, "   ", CHAT_ALIAS)
        .await
        .expect_err("empty query must fail before any LLM call");
    match err {
        UnikoError::Locy(msg) => assert!(msg.contains("empty"), "msg = {msg}"),
        other => panic!("expected Locy error for empty query, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY (real gpt-5-nano call); run via --run-ignored in a trusted job, never on public/fork PR CI"]
async fn translate_returns_safe_read_only_cypher() {
    require_openai_key();
    let kb = kb_with_chat().await;

    let cypher = translate(
        &kb,
        "How many messages did Caroline send last week?",
        CHAT_ALIAS,
    )
    .await
    .expect("gpt-5-nano must produce a Cypher query");

    assert!(
        !cypher.trim().is_empty(),
        "translator returned an empty query"
    );
    assert!(
        is_safe_read_only(&cypher),
        "LLM returned mutating Cypher (post-filter): {cypher}"
    );
    // The translator must always RETURN something; otherwise the
    // result is structurally not a query the caller can use.
    let upper = cypher.to_uppercase();
    assert!(upper.contains("RETURN"), "no RETURN in Cypher: {cypher}");
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY (real gpt-5-nano call); run via --run-ignored in a trusted job, never on public/fork PR CI"]
async fn translate_grounds_against_schema_labels() {
    require_openai_key();
    let kb = kb_with_chat().await;

    // A question that should bind against the Goal label.  We don't
    // pin the exact Cypher — the LLM is free to phrase it different
    // ways — but the output must mention `Goal` somewhere because
    // the schema prompt lists it as the relevant node type.
    let cypher = translate(&kb, "List the titles of every active goal", CHAT_ALIAS)
        .await
        .expect("translate");
    let upper = cypher.to_uppercase();
    assert!(
        upper.contains("GOAL"),
        "expected Goal label in Cypher for a goal-query, got: {cypher}"
    );
    assert!(is_safe_read_only(&cypher), "got mutating Cypher: {cypher}");
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY (real gpt-5-nano call); run via --run-ignored in a trusted job, never on public/fork PR CI"]
async fn translate_caches_repeat_queries() {
    require_openai_key();
    let kb = kb_with_chat().await;

    let q = "What entities did Melanie mention?";
    let first = translate(&kb, q, CHAT_ALIAS)
        .await
        .expect("first translate");

    // Even at temperature 0.0 the LLM can drift between distinct
    // requests, but the cache pins the first result regardless.
    let second = translate(&kb, q, CHAT_ALIAS)
        .await
        .expect("second translate must hit cache");
    assert_eq!(
        first, second,
        "repeat call should return the cached Cypher byte-for-byte"
    );

    // Normalisation should collapse case + whitespace into the same key.
    let third = translate(&kb, "  WHAT entities did MELANIE mention?  ", CHAT_ALIAS)
        .await
        .expect("normalised variant should hit cache");
    assert_eq!(
        first, third,
        "normalised variant should share the cache slot"
    );
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY (real gpt-5-nano call); run via --run-ignored in a trusted job, never on public/fork PR CI"]
async fn translate_executes_cypher_against_kb() {
    // End-to-end: the LLM's output must be syntactically valid Cypher
    // that the store can run.  We translate, execute, and require a
    // successful (possibly empty) result set.
    require_openai_key();
    let kb = kb_with_chat().await;

    let cypher = translate(&kb, "Count all participants", CHAT_ALIAS)
        .await
        .expect("translate");

    let session = kb.db().session();
    session
        .query_with(&cypher)
        .fetch_all()
        .await
        .unwrap_or_else(|e| panic!("LLM-produced Cypher did not parse: {cypher}\nerror: {e}"));
}

/// Proves the Kniv NER pre-extraction reaches the LLM prompt.  We
/// can't read the prompt directly (it's internal), but the test asks
/// a question with a clear PERSON ("Caroline") and asserts the
/// resulting Cypher binds that exact name as a literal.  Without the
/// Kniv hint the LLM frequently picks a generic property name; with
/// the hint it should consistently use the surface form the NLP
/// cascade saw.
#[cfg(feature = "onnx")]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY (real gpt-5-nano call); run via --run-ignored in a trusted job, never on public/fork PR CI"]
async fn translate_uses_kniv_entity_hints() {
    require_openai_key();
    let kb = kb_with_chat().await;

    let cypher = translate(
        &kb,
        "Who did Caroline talk to in the last month?",
        CHAT_ALIAS,
    )
    .await
    .expect("translate");

    assert!(is_safe_read_only(&cypher), "mutating output: {cypher}");
    assert!(
        cypher.contains("Caroline") || cypher.to_lowercase().contains("caroline"),
        "expected 'Caroline' (from Kniv NER hint) in Cypher, got: {cypher}"
    );
}

#[tokio::test]
async fn is_safe_read_only_rejects_classic_mutations() {
    // Pure logic — no LLM needed, but kept here so the public surface
    // (the only thing callers rely on) stays under integration-test
    // coverage.
    assert!(is_safe_read_only("MATCH (n) RETURN n LIMIT 5"));
    assert!(!is_safe_read_only("CREATE (x:Person)"));
    assert!(!is_safe_read_only(
        "MATCH (n) MERGE (m) ON CREATE SET m.x=1"
    ));
    assert!(!is_safe_read_only("MATCH (n) DETACH DELETE n"));
    assert!(!is_safe_read_only("MATCH (n) SET n.x = 1"));
    assert!(!is_safe_read_only("DROP INDEX foo"));
}
