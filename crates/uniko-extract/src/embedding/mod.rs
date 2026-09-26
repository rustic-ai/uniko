//! Embedding computation helpers via uni-db's Xervo runtime.
//!
//! Auto-embed (Message, Chunk, Observation, Summary) is handled
//! automatically by uni-db when the schema configures
//! the embedding config on the vector
//! index.  No application code is needed for those nodes.
//!
//! This module provides helpers for *computed* embeddings — nodes
//! whose embedding text is synthesized from multiple properties
//! (e.g. Entity: `"name (type)"`).

use std::collections::HashMap;

use uniko_store::Value;
use uniko_store::schema::{EMBED_ALIAS, HYBRID_EMBED_ALIAS};
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

/// Embed a single text for a **document** (indexing/storage).
///
/// Prepends the model's `document_prefix` (e.g., `"search_document: "`
/// for Nomic) before embedding. Use this for computed embeddings that
/// represent stored content (entities, facts, etc.).
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if the Xervo runtime is not
/// available or the model fails.
pub async fn embed_document(kb: &KnowledgeBase, text: &str) -> Result<Vec<f32>, UnikoError> {
    let prefixed = apply_prefix(text, kb.config().embedding.document_prefix.as_deref());
    embed_raw(kb, &prefixed).await
}

/// Embed a single text for a **query** (search/retrieval).
///
/// Prepends the model's `query_prefix` (e.g., `"search_query: "` for
/// Nomic) before embedding. Use this when embedding a user query for
/// similarity search against computed-embedding nodes.
///
/// For auto-embed nodes (Message, Chunk, Summary), uni-db applies
/// `query_prefix` automatically in `similar_to()` — this function is
/// only needed for computed-embed similarity searches.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if the runtime is unavailable.
pub async fn embed_query(kb: &KnowledgeBase, text: &str) -> Result<Vec<f32>, UnikoError> {
    let prefixed = apply_prefix(text, kb.config().embedding.query_prefix.as_deref());
    embed_raw(kb, &prefixed).await
}

/// Embed a query into ColBERT per-token vectors for MaxSim reranking.
///
/// Returns the query's multi-vector (`Vec<Vec<f32>>`). Applies the model's
/// `query_prefix` like [`embed_query`] (BGE-M3 uses none). Requires a
/// hybrid embedder whose embed alias exposes a multi-vector head.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if the runtime is unavailable, the
/// alias has no multi-vector head, or the result is empty.
pub async fn embed_multivector_query(
    kb: &KnowledgeBase,
    text: &str,
) -> Result<Vec<Vec<f32>>, UnikoError> {
    let prefixed = apply_prefix(text, kb.config().embedding.query_prefix.as_deref());
    // ColBERT rides the hybrid alias. uni-xervo 0.18.1 adapts a
    // `HybridEmbeddingModel` handle to the narrow `MultiVectorEmbeddingModel`
    // facade (`hybrid_adapter::HybridAsMultiVector`), so one alias serves both
    // the document pass and the query. Before 0.18.1 this lookup failed and
    // the reranker silently kept RRF order — see `rustic-ai/uni-xervo#49`.
    let results = kb
        .embed_multivector(HYBRID_EMBED_ALIAS, &[prefixed.as_str()])
        .await?;
    results
        .into_iter()
        .next()
        .ok_or_else(|| UnikoError::Embedding("empty multivector embedding result".into()))
}

/// Embed a single text string without any prefix.
///
/// Low-level function — prefer [`embed_document`] or [`embed_query`]
/// which apply the correct model-specific prefix.
pub async fn embed_raw(kb: &KnowledgeBase, text: &str) -> Result<Vec<f32>, UnikoError> {
    let results = kb.embed(EMBED_ALIAS, &[text]).await?;
    results
        .into_iter()
        .next()
        .ok_or_else(|| UnikoError::Embedding("empty embedding result".into()))
}

/// Embed multiple texts in a single batch call with a prefix.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if the runtime is unavailable.
pub async fn embed_batch(
    kb: &KnowledgeBase,
    texts: &[&str],
    prefix: Option<&str>,
) -> Result<Vec<Vec<f32>>, UnikoError> {
    let prefixed: Vec<String> = texts.iter().map(|t| apply_prefix(t, prefix)).collect();
    let refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
    kb.embed(EMBED_ALIAS, &refs).await
}

/// Embed many texts via repeated [`embed_batch`] calls of at most `chunk_size`.
///
/// The underlying ONNX runtime's BFC arena can OOM on a single batched
/// forward when input count × seq_len × hidden_dim grows large — the
/// 384-d BGE-small model has crashed in the consolidation cycle at
/// ~6 000 inputs requesting a 1.3 GB activation buffer. Issuing the
/// same total work in fixed-size chunks (default 64) keeps each forward
/// well under VRAM while preserving throughput, since per-call overhead
/// is amortized across the chunk.
///
/// Order is preserved: result `i` corresponds to input `i`.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if any chunk fails. The caller
/// receives no partial results — chunks are issued sequentially and the
/// first failure short-circuits.
pub async fn embed_batch_chunked(
    kb: &KnowledgeBase,
    texts: &[&str],
    prefix: Option<&str>,
    chunk_size: usize,
) -> Result<Vec<Vec<f32>>, UnikoError> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = chunk_size.max(1);
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(chunk_size) {
        let mut got = embed_batch(kb, chunk, prefix).await?;
        out.append(&mut got);
    }
    Ok(out)
}

/// Compute and store an embedding for an Entity node.
///
/// Formula: `"name (entity_type)"` or just `"name"` if the type is
/// unknown.  Uses [`embed_document`] (with document prefix) since
/// entity embeddings are stored content.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if embedding fails, or
/// [`UnikoError::Storage`] if the node update fails.
pub async fn embed_entity(
    kb: &KnowledgeBase,
    node_id: NodeId,
    name: &str,
    entity_type: Option<&str>,
) -> Result<Vec<f32>, UnikoError> {
    let text = match entity_type {
        Some(t) if !t.is_empty() => format!("{name} ({t})"),
        _ => name.to_string(),
    };

    let vec = embed_document(kb, &text).await?;

    // Store the computed embedding on the node.
    let mut props = HashMap::new();
    props.insert("embedding".into(), Value::Vector(vec.clone()));
    kb.update_node(node_id, &props).await?;

    Ok(vec)
}

/// Prepend prefix to text if set.
fn apply_prefix(text: &str, prefix: Option<&str>) -> String {
    match prefix {
        Some(p) => format!("{p}{text}"),
        None => text.to_string(),
    }
}

/// Extract the topic text used to embed an Episode.
///
/// Spec Part VIII names this as the v5 fix: Episodes must embed the
/// *topic* from their `state` JSON, not a generic action/outcome string.
/// Priority order (per the embedding analysis notes):
///
/// 1. `state.topic`
/// 2. `state.question`
/// 3. `state.description`
/// 4. `state.summary`
/// 5. `state.input` (truncated to 200 chars)
/// 6. Fallback: `"{action_type}: {outcome}"`
///
/// Returns the chosen text. `state` may be `Value::Map` or any other
/// shape — non-Map values fall through to the fallback.
pub fn episode_topic_text(
    state: Option<&Value>,
    action_type: &str,
    outcome: Option<&str>,
) -> String {
    const PRIORITY_KEYS: &[&str] = &["topic", "question", "description", "summary"];
    const INPUT_PREFIX_LEN: usize = 200;

    if let Some(Value::Map(map)) = state {
        for key in PRIORITY_KEYS {
            if let Some(Value::String(s)) = map.get(*key)
                && !s.trim().is_empty()
            {
                return s.clone();
            }
        }
        if let Some(Value::String(s)) = map.get("input")
            && !s.trim().is_empty()
        {
            return s.chars().take(INPUT_PREFIX_LEN).collect();
        }
    }

    match outcome {
        Some(o) if !o.is_empty() => format!("{action_type}: {o}"),
        _ => action_type.to_string(),
    }
}

/// Compute and store an embedding for an Episode node.
///
/// Extracts topic text from `state` per [`episode_topic_text`], embeds
/// it with the document prefix, and persists the result on the node.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if embedding fails, or
/// [`UnikoError::Storage`] if the node update fails.
pub async fn embed_episode(
    kb: &KnowledgeBase,
    node_id: NodeId,
    state: Option<&Value>,
    action_type: &str,
    outcome: Option<&str>,
) -> Result<Vec<f32>, UnikoError> {
    let text = episode_topic_text(state, action_type, outcome);
    let vec = embed_document(kb, &text).await?;

    let mut props = HashMap::new();
    props.insert("embedding".into(), Value::Vector(vec.clone()));
    kb.update_node(node_id, &props).await?;

    Ok(vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdMap;

    fn map_state(pairs: &[(&str, &str)]) -> Value {
        let mut m = StdMap::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), Value::String((*v).to_string()));
        }
        Value::Map(m)
    }

    #[test]
    fn topic_takes_priority_over_all_others() {
        let state = map_state(&[
            ("topic", "career plans"),
            ("question", "what role?"),
            ("description", "long form"),
        ]);
        let t = episode_topic_text(Some(&state), "retrieve", Some("success"));
        assert_eq!(t, "career plans");
    }

    #[test]
    fn question_used_when_topic_missing() {
        let state = map_state(&[("question", "what role?"), ("description", "long form")]);
        let t = episode_topic_text(Some(&state), "retrieve", Some("success"));
        assert_eq!(t, "what role?");
    }

    #[test]
    fn description_then_summary_in_priority_order() {
        let state = map_state(&[("description", "the desc"), ("summary", "the summary")]);
        assert_eq!(episode_topic_text(Some(&state), "x", None), "the desc");
        let state = map_state(&[("summary", "the summary")]);
        assert_eq!(episode_topic_text(Some(&state), "x", None), "the summary");
    }

    #[test]
    fn input_truncates_to_200_chars() {
        let long: String = "x".repeat(500);
        let state = map_state(&[("input", long.as_str())]);
        let t = episode_topic_text(Some(&state), "x", None);
        assert_eq!(t.chars().count(), 200);
    }

    #[test]
    fn empty_strings_fall_through() {
        let state = map_state(&[("topic", ""), ("question", "real question")]);
        assert_eq!(episode_topic_text(Some(&state), "x", None), "real question");
    }

    #[test]
    fn fallback_to_action_outcome() {
        assert_eq!(
            episode_topic_text(None, "retrieve", Some("success")),
            "retrieve: success"
        );
    }

    #[test]
    fn fallback_to_action_only_when_no_outcome() {
        assert_eq!(episode_topic_text(None, "retrieve", None), "retrieve");
    }

    #[test]
    fn non_map_state_falls_through() {
        let state = Value::String("not a map".into());
        assert_eq!(
            episode_topic_text(Some(&state), "act", Some("ok")),
            "act: ok"
        );
    }
}
