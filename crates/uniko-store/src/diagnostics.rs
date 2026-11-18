//! Per-operation stage diagnostics: what ran, what was skipped, and what
//! fell back or failed.
//!
//! Recall and ingest are best-effort: a reranker that errors keeps the
//! fused order, a missing NLP model falls back to rules, and the call
//! still succeeds. A [`Diagnostic`] records each such outcome so the
//! caller can see it on the result instead of in a log.
//!
//! Code deep in the call records with [`record`]; the operation's entry
//! point wraps the work in [`collect`] and returns what was recorded. The
//! collector is a tokio task-local, so it sees every stage that runs on
//! the calling task (including futures joined with `join_all`), but not
//! work moved to another task with `tokio::spawn`. Outside a [`collect`]
//! scope, [`record`] does nothing.
//!
//! Diagnostics carry no free text: a stage, a status, a stable code, and
//! at most a label or alias as `target`. The full error stays in the log
//! line next to each recording site.

use std::cell::RefCell;

use serde::Serialize;

/// The stage a [`Diagnostic`] is about. A stable, snake_case identifier.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Embedding the query (and its variants) for vector search.
    QueryEmbedding,
    /// NLP analysis of the query (entities, keywords).
    QueryAnalysis,
    /// The learned-sparse retrieval channel.
    Sparse,
    /// ColBERT late-interaction reranking.
    Colbert,
    /// The cross-encoder reranker.
    Rerank,
    /// One candidate source query (vector, full-text, graph) of the cascade.
    Source,
    /// Resolving each item's category and source attribution.
    Provenance,
    /// The ingest NLP cascade (entities, sentence analysis) on a message.
    Nlp,
    /// Observation extraction from a message: the model path, or rules.
    ObservationModel,
    /// Loading the custom observation rules file.
    ObservationRules,
    /// Extracting text from a PDF.
    PdfExtraction,
    /// Pooling an artifact's chunk embeddings into its own vector.
    ArtifactEmbedding,
    /// A modality extractor's post-commit step.
    ExtractorPostCommit,
}

/// What happened to a stage.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    /// The stage ran as requested.
    Applied,
    /// The stage was requested but had nothing to do.
    Skipped,
    /// The stage failed and a weaker substitute ran in its place.
    Fallback,
    /// The stage failed and its contribution is missing.
    Failed,
}

impl StageStatus {
    /// Whether this status means the result is weaker than requested.
    #[must_use]
    pub fn is_degraded(self) -> bool {
        matches!(self, Self::Fallback | Self::Failed)
    }
}

/// Why a stage was skipped, fell back or failed. A stable, snake_case
/// identifier.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticCode {
    /// The embedding model call failed.
    EmbedFailed,
    /// The NLP pipeline could not be started.
    NlpUnavailable,
    /// The NLP pipeline started but analysis failed.
    NlpFailed,
    /// A store query for the stage failed.
    QueryFailed,
    /// Embedding the query as multi-vectors (for ColBERT) failed.
    QueryEmbedFailed,
    /// The ColBERT MaxSim query failed.
    MaxsimFailed,
    /// No candidate carried what the stage needs.
    NoCandidates,
    /// The reranker model call failed.
    RerankFailed,
    /// Item provenance could not be resolved.
    ProvenanceUnresolved,
    /// The recall cascade returned before reaching the stage.
    NotReached,
    /// The stage needs a Cargo feature this build lacks.
    NotCompiled,
    /// Rule-based extraction ran because the model path was unavailable.
    RuleBased,
    /// The custom rules file could not be loaded; the bundled rules ran.
    CustomRulesUnloadable,
    /// Tiered (layout and OCR) PDF extraction failed.
    TieredFailed,
    /// PDF text extraction failed; the artifact has no chunks.
    ExtractionFailed,
    /// Pooling failed; the artifact vector is left for backfill.
    PoolFailed,
    /// The extractor's post-commit step failed.
    PostCommitFailed,
}

impl DiagnosticCode {
    /// The snake_case name, as serialized.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EmbedFailed => "embed_failed",
            Self::NlpUnavailable => "nlp_unavailable",
            Self::NlpFailed => "nlp_failed",
            Self::QueryFailed => "query_failed",
            Self::QueryEmbedFailed => "query_embed_failed",
            Self::MaxsimFailed => "maxsim_failed",
            Self::NoCandidates => "no_candidates",
            Self::RerankFailed => "rerank_failed",
            Self::ProvenanceUnresolved => "provenance_unresolved",
            Self::NotReached => "not_reached",
            Self::NotCompiled => "not_compiled",
            Self::RuleBased => "rule_based",
            Self::CustomRulesUnloadable => "custom_rules_unloadable",
            Self::TieredFailed => "tiered_failed",
            Self::ExtractionFailed => "extraction_failed",
            Self::PoolFailed => "pool_failed",
            Self::PostCommitFailed => "post_commit_failed",
        }
    }
}

/// One stage outcome, safe to log or show.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Diagnostic {
    /// The stage.
    pub stage: Stage,
    /// What happened to it.
    pub status: StageStatus,
    /// Why, when the status is not `Applied`.
    pub code: Option<DiagnosticCode>,
    /// The node label, model alias or source the stage ran on, if any.
    pub target: Option<String>,
    /// How many items the stage contributed, when it counts them.
    pub count: Option<u32>,
}

impl Diagnostic {
    /// A stage that ran as requested.
    #[must_use]
    pub fn applied(stage: Stage) -> Self {
        Self {
            stage,
            status: StageStatus::Applied,
            code: None,
            target: None,
            count: None,
        }
    }

    /// A stage that was skipped, fell back or failed, with its code.
    #[must_use]
    pub fn new(stage: Stage, status: StageStatus, code: DiagnosticCode) -> Self {
        Self {
            stage,
            status,
            code: Some(code),
            target: None,
            count: None,
        }
    }

    /// Name the label, alias or source the stage ran on.
    #[must_use]
    pub fn target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Record how many items the stage contributed.
    #[must_use]
    pub fn count(mut self, count: usize) -> Self {
        self.count = Some(u32::try_from(count).unwrap_or(u32::MAX));
        self
    }

    fn same_as(&self, other: &Self) -> bool {
        self.stage == other.stage
            && self.status == other.status
            && self.code == other.code
            && self.target == other.target
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", name(&self.stage), name(&self.status))?;
        if let Some(code) = self.code {
            write!(f, " {}", code.as_str())?;
        }
        if let Some(target) = &self.target {
            write!(f, " ({target})")?;
        }
        Ok(())
    }
}

/// The serialized (snake_case) name of a unit enum variant.
fn name(value: &impl Serialize) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

/// Render diagnostics on one line, for [`UnikoError::Degraded`](crate::UnikoError::Degraded).
pub(crate) fn describe(diagnostics: &[Diagnostic]) -> String {
    diagnostics
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

tokio::task_local! {
    static COLLECTOR: RefCell<Vec<Diagnostic>>;
}

/// Run `fut`, returning its output and every diagnostic recorded while it
/// ran on this task.
///
/// A nested `collect` keeps its own diagnostics; the outer scope does not
/// see them.
pub async fn collect<F: std::future::Future>(fut: F) -> (F::Output, Vec<Diagnostic>) {
    COLLECTOR
        .scope(RefCell::new(Vec::new()), async move {
            let out = fut.await;
            let recorded = COLLECTOR.with(RefCell::take);
            (out, recorded)
        })
        .await
}

/// Record `diagnostic` in the enclosing [`collect`] scope, if any.
///
/// A repeat of the same stage, status, code and target is merged into the
/// first (adding their counts), so a retried or per-variant stage is
/// reported once.
pub fn record(diagnostic: Diagnostic) {
    let _ = COLLECTOR.try_with(|cell| {
        let mut list = cell.borrow_mut();
        if let Some(existing) = list.iter_mut().find(|d| d.same_as(&diagnostic)) {
            existing.count = match (existing.count, diagnostic.count) {
                (Some(a), Some(b)) => Some(a.saturating_add(b)),
                (a, b) => a.or(b),
            };
        } else {
            list.push(diagnostic);
        }
    });
}

/// Record that a candidate source of a recall failed and contributed
/// nothing. `name` is a stable source id such as `phase3:Chunk:Message`;
/// never pass query text or content.
pub fn record_source_failure(name: &str) {
    record(
        Diagnostic::new(
            Stage::Source,
            StageStatus::Failed,
            DiagnosticCode::QueryFailed,
        )
        .target(name),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn collect_gathers_and_merges_records() {
        let ((), recorded) = collect(async {
            record(Diagnostic::applied(Stage::Sparse).target("Chunk").count(2));
            tokio::task::yield_now().await;
            record(Diagnostic::applied(Stage::Sparse).target("Chunk").count(3));
            record(
                Diagnostic::new(
                    Stage::Rerank,
                    StageStatus::Fallback,
                    DiagnosticCode::RerankFailed,
                )
                .target("rerank/default"),
            );
        })
        .await;
        assert_eq!(recorded.len(), 2, "{recorded:?}");
        assert_eq!(recorded[0].count, Some(5));
        assert_eq!(
            recorded[1].to_string(),
            "rerank fallback rerank_failed (rerank/default)"
        );
        assert!(recorded[1].status.is_degraded());
    }

    #[tokio::test]
    async fn joined_futures_record_into_the_same_scope_and_nesting_isolates() {
        let ((), outer) = collect(async {
            let one = || async {
                record(Diagnostic::applied(Stage::QueryEmbedding).count(1));
            };
            tokio::join!(one(), one(), one());
            let ((), inner) = collect(async {
                record(Diagnostic::applied(Stage::Rerank));
            })
            .await;
            assert_eq!(inner.len(), 1);
        })
        .await;
        assert_eq!(outer.len(), 1, "{outer:?}");
        assert_eq!(outer[0].count, Some(3));
    }

    #[test]
    fn record_outside_a_scope_is_a_no_op() {
        record(Diagnostic::applied(Stage::Colbert));
    }

    #[test]
    fn diagnostics_serialize_as_snake_case() {
        let d = Diagnostic::new(
            Stage::QueryEmbedding,
            StageStatus::Failed,
            DiagnosticCode::EmbedFailed,
        );
        assert_eq!(
            serde_json::to_value(&d).unwrap(),
            serde_json::json!({
                "stage": "query_embedding",
                "status": "failed",
                "code": "embed_failed",
                "target": null,
                "count": null,
            })
        );
    }
}
