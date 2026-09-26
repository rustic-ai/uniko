//! Pipeline 1 — Ingest for Messages and Artifacts.
//!
//! [`IngestStep`] implements the [`Step`](uniko_pipes::Step) trait and
//! dispatches to [`atomic::ingest_message_atomic`] or
//! [`artifact::ingest_artifact`] based on the `ingest_type` metadata
//! key in the [`PipelineContext`](uniko_pipes::PipelineContext).

pub mod artifact;
pub mod atomic;
pub mod chunking;
pub mod context;
pub mod message;
pub mod modality;
pub mod pdf;
pub mod session;
pub mod session_chunk;
pub mod source;
mod unit;

pub use artifact::ArtifactIngestResult;
pub use artifact::{ArtifactContextNids, ArtifactIdentity, ArtifactPrep, UnitArtifactSeen};
pub use atomic::{AtomicIngestResult, AtomicTimings, ingest_message_atomic};
pub use chunking::{ChunkConfig, ChunkData, Chunker, count_tokens, select_chunker};
pub use modality::{ModalityExtractor, ModalityPrepared, ModalityRegistry};
pub use source::{PreparedSource, prepare_source};
pub use unit::{UnitIngestResult, UnitTurn, ingest_turns_atomic};
// IngestSource/IngestData live in uniko-pipes (wire types) so IngestTask can
// carry them; re-exported here so existing `uniko_extract::ingest` imports
// and the facade are unchanged.
pub use pdf::{
    PdfExtractCrate, PdfExtractError, PdfIngestOptions, PdfIngestResult, PdfInput,
    PdfTextExtractor, ingest_pdf,
};
pub use source::{IngestContext, IngestOutcome, ingest_source, resolve_mime};
pub use uniko_pipes::types::{IngestData, IngestSource};

use async_trait::async_trait;

use uniko_pipes::step::PipelineContext;
use uniko_pipes::types::{IngestMessage, StepOutcome};
use uniko_store::UnikoError;

/// Pipeline step that ingests messages and artifacts into the graph.
///
/// Dispatch is based on `ctx.metadata["ingest_type"]`:
/// - `"message"`: deserializes [`IngestMessage`] from `"ingest_payload"`
///   and calls [`atomic::ingest_message_atomic`].
/// - `"artifact"`: deserializes [`IngestArtifact`](uniko_pipes::IngestArtifact)
///   and calls [`artifact::ingest_artifact`].
#[derive(Debug)]
pub struct IngestStep;

#[async_trait]
impl uniko_pipes::Step for IngestStep {
    fn name(&self) -> &str {
        "ingest"
    }

    fn should_run(&self, _ctx: &PipelineContext) -> bool {
        true
    }

    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepOutcome, UnikoError> {
        let ingest_type = ctx
            .metadata
            .get("ingest_type")
            .and_then(|v| v.as_str())
            .unwrap_or("message");

        match ingest_type {
            "message" => {
                let msg: IngestMessage = deserialize_payload(&ctx.metadata, "IngestMessage")?;
                // Use SessionContext from metadata if available, else create a fresh one.
                let mut session_ctx = ctx
                    .metadata
                    .get("session_context")
                    .and_then(|v| serde_json::from_value::<context::SessionContext>(v.clone()).ok())
                    .unwrap_or_else(|| context::SessionContext::new(msg.session_id.clone(), 0));
                let result = atomic::ingest_message_atomic(&ctx.kb, &msg, &mut session_ctx).await?;
                ctx.node_id = result.message_node_id;
                // Forward chunk IDs to downstream steps (embedding).
                ctx.metadata.insert(
                    "chunk_node_ids".to_string(),
                    serde_json::to_value(&result.chunk_node_ids).unwrap_or_default(),
                );
                Ok(StepOutcome::Completed)
            }
            "artifact" => {
                let art: uniko_pipes::IngestArtifact =
                    deserialize_payload(&ctx.metadata, "IngestArtifact")?;
                let result = artifact::ingest_artifact(&ctx.kb, &art).await?;
                ctx.node_id = result.artifact_node_id;
                if result.was_deduplicated {
                    return Ok(StepOutcome::Skipped {
                        reason: "artifact deduplicated by content hash".into(),
                    });
                }
                ctx.metadata.insert(
                    "chunk_node_ids".to_string(),
                    serde_json::to_value(&result.chunk_node_ids).unwrap_or_default(),
                );
                Ok(StepOutcome::Completed)
            }
            "pdf" => {
                let task: uniko_pipes::IngestPdf = deserialize_payload(&ctx.metadata, "IngestPdf")?;
                let input = match task.input {
                    uniko_pipes::PdfInput::Bytes(b) => pdf::PdfInput::Bytes(b),
                    uniko_pipes::PdfInput::Path(p) => pdf::PdfInput::Path(p),
                };
                let options = pdf::PdfIngestOptions {
                    artifact_id: task.artifact_id,
                    // `IngestPdf.artifact_id` is non-optional and documented
                    // caller-provided, so the id is always the identity here.
                    caller_supplied_id: true,
                    extractor: None,
                    source_path: task.source_path,
                    // Streamed PDFs aren't session/message-linked.
                    session_id: None,
                    triggered_by_message_id: None,
                    category: None,
                    source_id: None,
                    revision_id: None,
                };
                let result = pdf::ingest_pdf(&ctx.kb, input, options).await?;
                ctx.node_id = result.artifact_node_id;
                Ok(StepOutcome::Completed)
            }
            "source" => {
                let src: uniko_pipes::IngestSource =
                    deserialize_payload(&ctx.metadata, "IngestSource")?;
                // Streamed sources aren't session-linked (same caveat as
                // streamed turns); no registered extractors on this path.
                let outcome = source::ingest_source(
                    &ctx.kb,
                    &modality::ModalityRegistry::default(),
                    src,
                    source::IngestContext::default(),
                )
                .await?;
                ctx.node_id = match outcome {
                    source::IngestOutcome::Artifact(r) => r.artifact_node_id,
                    source::IngestOutcome::Pdf(r) => r.artifact_node_id,
                };
                Ok(StepOutcome::Completed)
            }
            other => Ok(StepOutcome::Skipped {
                reason: format!("unknown ingest type: {other}"),
            }),
        }
    }
}

/// Deserialize an `ingest_payload` from pipeline metadata.
///
/// `kind_label` is included in the error message when deserialization
/// fails (e.g. `"IngestMessage"`, `"IngestArtifact"`).
fn deserialize_payload<T: serde::de::DeserializeOwned>(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
    kind_label: &str,
) -> Result<T, UnikoError> {
    let payload = metadata
        .get("ingest_payload")
        .ok_or_else(|| UnikoError::Pipeline("missing ingest_payload in metadata".into()))?;
    serde_json::from_value(payload.clone())
        .map_err(|e| UnikoError::Pipeline(format!("invalid {kind_label}: {e}")))
}
