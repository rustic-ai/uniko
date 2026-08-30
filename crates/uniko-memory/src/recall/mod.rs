//! Basic recall API — searches the memory graph and returns ranked
//! results.
//!
//! Phase 1 implements only Phase 3 (Broaden) of the 3-phase recall
//! cascade.  At cold start (no Facts, no Procedures) this is the
//! expected behavior.  Compact (Phase 1) and Expand (Phase 2)
//! activate in execution Phase 2 when consolidation creates Facts.

pub mod intent;
pub mod mmr;
pub mod modality;

pub use intent::{IntentProfile, build_intent, build_intent_at};

use std::collections::HashMap;

use futures::future::join_all;
use serde::{Deserialize, Serialize};

use uniko_store::{KnowledgeBase, NodeId, UnikoError, Value};

// ── Types ───────────────────────────────────────────────────────────

/// Tier classification for recall scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RecallTier {
    /// Facts, Topics (Phase 2+).
    Semantic,
    /// Procedures (Phase 3+).
    Procedural,
    /// Episodes, Observations.
    Episodic,
    /// Chunks, Artifacts.
    KnowledgeBase,
    /// Actions, Messages.
    Provenance,
}

impl RecallTier {
    /// Scoring weight for this tier.
    ///
    /// Matches the values published in the v6 spec (Part IX,
    /// "Hybrid Scoring").  The previous local tuning (Semantic 0.9,
    /// Procedural 0.8) was reverted when Phase 1 (Compact) shipped:
    /// suppressing the top tiers had been a workaround for the
    /// missing Fact retrieval surface.
    pub fn weight(self) -> f64 {
        match self {
            Self::Semantic => 1.0,
            Self::Procedural => 0.9,
            Self::Episodic => 0.7,
            Self::KnowledgeBase => 0.5,
            Self::Provenance => 0.4,
        }
    }
}

/// What a recalled item *is* — its place in the cognitive stack.
///
/// Distinguishes raw input ([`Chunk`](RecallKind::Chunk),
/// [`Message`](RecallKind::Message)) from derived memory (Fact / Procedure /
/// Topic / Observation). Replaces the former stringly `node_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallKind {
    /// Raw content chunk — from a message turn or an attachment/document.
    Chunk,
    /// An atomic extracted statement.
    Observation,
    /// Derived, consolidated knowledge (multi-evidence).
    Fact,
    /// A learned procedure / routine.
    Procedure,
    /// A clustered topic.
    Topic,
    /// A recorded episode / event.
    Episode,
    /// A conversation message itself.
    Message,
    /// Any other node type (Goal/Task/Entity/…).
    Other,
}

impl RecallKind {
    /// Classify a node-type label.
    #[must_use]
    pub fn from_label(label: &str) -> Self {
        match label {
            "Chunk" => Self::Chunk,
            "Observation" => Self::Observation,
            "Fact" => Self::Fact,
            "Procedure" => Self::Procedure,
            "Topic" => Self::Topic,
            "Episode" => Self::Episode,
            "Message" => Self::Message,
            _ => Self::Other,
        }
    }

    /// The scoring tier for this kind (mirrors the v6 hybrid-scoring map).
    #[must_use]
    pub fn tier(self) -> RecallTier {
        match self {
            Self::Fact | Self::Topic => RecallTier::Semantic,
            Self::Procedure => RecallTier::Procedural,
            Self::Episode => RecallTier::Episodic,
            // Observations index `content` directly and compete with Chunks
            // on raw similarity, so they share the KnowledgeBase tier.
            Self::Observation | Self::Chunk => RecallTier::KnowledgeBase,
            Self::Message | Self::Other => RecallTier::Provenance,
        }
    }

    /// `true` for synthesized memory (Fact/Procedure/Topic/Observation);
    /// `false` for raw input (Chunk/Message). Answers "did the system
    /// derive this, or is it verbatim?"
    #[must_use]
    pub fn is_derived(self) -> bool {
        matches!(
            self,
            Self::Fact | Self::Procedure | Self::Topic | Self::Observation
        )
    }
}

/// Where a recalled item's content originated — its lineage.
///
/// A [`RecallKind::Chunk`] has one source; a derived
/// [`RecallKind::Fact`] lists the messages/attachments its supporting
/// observations came from. Aggregates with no clean message lineage
/// (Topic/Procedure) carry an empty list.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallSource {
    /// The content is a message's own text.
    Message {
        /// The message it traces to.
        message_id: String,
        /// The specific evidence chunk, when content-bearing.
        chunk_id: Option<String>,
    },
    /// The content is from a document attached to a message.
    Attachment {
        /// The message the document was shared in.
        message_id: String,
        /// The attached document.
        artifact_id: String,
        /// The specific evidence chunk, when content-bearing.
        chunk_id: Option<String>,
    },
    /// The content is from a standalone-loaded document (no message).
    Document {
        /// The document.
        artifact_id: String,
        /// The specific evidence chunk, when content-bearing.
        chunk_id: Option<String>,
    },
}

/// A single recalled item.
#[derive(Debug, Clone, Serialize)]
pub struct RecallItem {
    /// Node ID.
    pub node_id: NodeId,
    /// What this item is (raw vs derived). Use [`RecallKind::is_derived`].
    pub kind: RecallKind,
    /// Fused score after RRF and tier weighting.
    pub score: f64,
    /// Display text.
    pub content: String,
    /// Lineage — the messages/attachments this content came from. One entry
    /// for a chunk, many for a Fact, empty for aggregates.
    pub sources: Vec<RecallSource>,
}

impl RecallItem {
    /// The first source, for the common single-origin case.
    #[must_use]
    pub fn primary_source(&self) -> Option<&RecallSource> {
        self.sources.first()
    }
}

/// Ranked result bundle from a recall query.
#[derive(Debug, Clone, Serialize)]
pub struct ContextBundle {
    /// Ranked items.
    pub items: Vec<RecallItem>,
    /// Estimated total tokens.
    pub total_tokens: usize,
    /// Whether Compact (Phase 1) alone satisfied the coverage gate
    /// and the cascade exited before Phase 2.  Tracked as the spec's
    /// primary scaling signal (`phase1_only_pct`).
    pub phase1_only: bool,
    /// Whether the cascade exited after Phase 2 (Expand) — i.e. Phase
    /// 1 failed its gate but Phase 2's vector + fulltext over Episode/
    /// Observation/Message cleared the 0.65 coverage gate.  Tracks
    /// `phase2_only_pct` alongside `phase1_only_pct`.
    pub phase2_only: bool,
    /// Coverage score (0.0–1.0).
    pub coverage: f64,
}

/// How Phase 1 (Compact) results contribute to the final bundle.
///
/// Tested on conv-26/conv-30 ablations and documented in the
/// `rfe-p4-recall-evolution` RFE.  `Merge` is the legacy default
/// (cap=3 interleave by score); `Boost` is the architectural v2 where
/// Facts/Observations only influence chunk ranking and never occupy
/// bundle slots; `Off` disables Phase 1 contributions entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Phase1Strategy {
    /// Merge top-N Phase 1 Facts into the Phase 3 bundle by score
    /// (default — current best-known stack: conv-26 → 0.750).
    #[default]
    Merge,
    /// Use Phase 1 Facts (and top Observation hits) as a session-level
    /// boost signal on Chunk scores.  Bundle stays 100% Chunks.
    Boost,
    /// Skip Phase 1 entirely — no merge and no boost.
    Off,
}

/// Access-control scope for a recall.
///
/// Modelled as an explicit enum rather than `Option<Viewer>` so that
/// bypassing the access-control filter is a *named*, greppable choice
/// (`ViewerScope::Unrestricted`) instead of a silent `None` — the recall
/// path is otherwise fail-open (see issue #5). [`Default`] is
/// `Unrestricted` to preserve behaviour for internal/bench callers that
/// have no viewer; production callers that serve a specific participant
/// should pass [`ViewerScope::As`].
#[derive(Debug, Clone, Default)]
pub enum ViewerScope {
    /// No access-control filtering — the caller is trusted (internal
    /// consolidation, benchmarks, admin). Fail-open by explicit choice.
    #[default]
    Unrestricted,
    /// Filter the returned bundle to what this [`crate::policy::Viewer`]
    /// is allowed to see (private/team/org visibility).
    As(crate::policy::Viewer),
}

/// Dimensional hard-filters applied during recall candidate generation.
///
/// Mirrors mem0's `search(query, filters={...})`: each `Some` narrows the
/// candidate set; `None` leaves that dimension unconstrained. An empty
/// `Vec` matches nothing — `sessions: Some(vec![])` returns no results,
/// distinct from `None`. Time bounds are half-open `[since, until)`.
///
/// The all-`None` [`Default`] imposes no constraint, so an unscoped recall
/// generates byte-identical Cypher to before this type existed. Nodes that
/// cannot anchor a dimension (e.g. Facts have no session) are *excluded*
/// by a filter on that dimension — a session filter is "only these
/// sessions", so session-less tiers go dark under it.
#[derive(Debug, Clone, Default)]
pub struct Dimensions {
    /// Restrict to items anchored to one of these `Session.session_id`s.
    pub sessions: Option<Vec<String>>,
    /// Restrict to items whose sender *or* subject is one of these
    /// `Participant.name`s (matches `SENT_BY`/`PARTICIPATED_IN` or
    /// `ABOUT`).
    pub participants: Option<Vec<String>>,
    /// Lower time bound, inclusive.
    pub since: Option<chrono::DateTime<chrono::Utc>>,
    /// Upper time bound, exclusive.
    pub until: Option<chrono::DateTime<chrono::Utc>>,
}

impl Dimensions {
    /// `true` when no dimension constrains the candidate set.
    ///
    /// The recall push-down emits no predicate in this case, preserving
    /// the unscoped query path exactly.
    #[must_use]
    pub fn is_unconstrained(&self) -> bool {
        self.sessions.is_none()
            && self.participants.is_none()
            && self.since.is_none()
            && self.until.is_none()
    }
}

/// An optional, per-call recall scope: visibility plus dimensional filters.
///
/// Pass to [`Agent::recall_in`](crate::Agent::recall_in) /
/// [`Agent::answer_in`](crate::Agent::answer_in) to override the
/// instance-global scope for a single call — the way mem0 and Zep accept
/// per-call `user_id` / `session` filters. The [`Default`] is empty: empty
/// dimensions plus [`ViewerScope::Unrestricted`], which the facade reads as
/// "fall back to the instance default visibility".
///
/// # Examples
///
/// ```
/// use uniko_memory::Scope;
///
/// let scope = Scope::default()
///     .sessions(["chat-1"])
///     .since(chrono::Utc::now() - chrono::Duration::days(7));
/// # let _ = scope;
/// ```
#[derive(Debug, Clone, Default)]
pub struct Scope {
    /// Visibility scope — who may see the results.
    pub viewer: ViewerScope,
    /// Dimensional hard-filters — session / participant / time.
    pub dims: Dimensions,
}

impl Scope {
    /// Restrict recall to these session ids.
    #[must_use]
    pub fn sessions(mut self, sessions: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.dims.sessions = Some(sessions.into_iter().map(Into::into).collect());
        self
    }

    /// Restrict recall to these participant names (sender or subject).
    #[must_use]
    pub fn participants(
        mut self,
        participants: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.dims.participants = Some(participants.into_iter().map(Into::into).collect());
        self
    }

    /// Set the inclusive lower time bound.
    #[must_use]
    pub fn since(mut self, since: chrono::DateTime<chrono::Utc>) -> Self {
        self.dims.since = Some(since);
        self
    }

    /// Set the exclusive upper time bound.
    #[must_use]
    pub fn until(mut self, until: chrono::DateTime<chrono::Utc>) -> Self {
        self.dims.until = Some(until);
        self
    }

    /// Scope visibility to an explicit [`Viewer`](crate::policy::Viewer).
    #[must_use]
    pub fn as_viewer(mut self, viewer: crate::policy::Viewer) -> Self {
        self.viewer = ViewerScope::As(viewer);
        self
    }
}

/// Recall query configuration.
#[derive(Debug, Clone)]
pub struct RecallConfig {
    /// Maximum items to return.
    pub limit: usize,
    /// Maximum total tokens.
    pub token_budget: usize,
    /// Minimum fused score for inclusion.
    pub min_score: f64,
    /// Vector similarity weight in hybrid fusion.
    pub vector_weight: f64,
    /// BM25 fulltext weight in hybrid fusion.
    pub bm25_weight: f64,
    /// When true, re-score the top `reranker_top_n` RRF candidates with
    /// the cross-encoder registered at `rerank/default`.
    pub reranker_enabled: bool,
    /// Number of RRF candidates to send to the reranker.
    pub reranker_top_n: usize,
    /// Apply sigmoid to raw cross-encoder logits.
    pub reranker_apply_sigmoid: bool,
    /// Reranker code path: `"cross-encoder"` / `"generative"` (xervo model
    /// at `rerank/default`) or `"colbert"` (in-process MaxSim over the
    /// `colbert_embedding` column — no model alias). Mirrors
    /// [`RerankerConfig::style`](uniko_store::config::RerankerConfig::style).
    pub reranker_style: String,
    /// Add a learned-sparse retrieval channel for Chunk and Observation,
    /// fused into the RRF pool. Requires a hybrid embedder (the
    /// `sparse_embedding` column). Default `false`.
    pub recall_sparse_enabled: bool,
    /// When > 1.0, multiplies the score of any RecallItem whose
    /// connected entities include one of `IntentProfile.expected_answer_type`.
    /// `1.0` is a no-op. Only triggers when `predict_answer_type` returns
    /// a label.
    pub answer_type_boost: f64,
    /// Cap on how many top items are checked for the boost (one Cypher
    /// lookup per item — avoid scaling badly when limit is large).
    pub answer_type_top_n: usize,
    /// Variant labels to enable for multi-query reformulation. Empty
    /// vec (the default) means the **single** `keywords` variant, not all
    /// of them; opt into multi-query explicitly with
    /// `vec!["keywords", "original", "declarative", "type_anchored"]`.
    pub query_variants: Vec<String>,
    /// `k` constant for reciprocal rank fusion across variants. Higher
    /// values flatten the weight given to top ranks (k=60 is the
    /// canonical default).
    pub rrf_k: f64,
    /// LIMIT applied to each per-variant Cypher query. Larger values
    /// keep more candidates per variant in the fusion pool at the
    /// cost of latency. Defaults to `limit` when constructed via
    /// `from_uniko_config`.
    pub per_variant_limit: usize,
    /// Phase 1 (Compact) contribution strategy.  See [`Phase1Strategy`].
    pub phase1_strategy: Phase1Strategy,
    /// Multiplicative weight applied to Fact scores when computing the
    /// session-chunk boost under [`Phase1Strategy::Boost`].  Small (0.1-0.3)
    /// — large enough to nudge a chunk by ~one rank position, small
    /// enough not to overwhelm the cross-encoder ranking.
    pub phase1_boost_alpha: f64,
    /// Coverage gate for Phase 2 (Expand) early exit.  When Phase 2
    /// fuses vector + fulltext hits across Episode/Observation/Message
    /// and `coverage >= phase2_coverage_gate`, the cascade skips Phase
    /// 3 (Broaden).  Spec §IX default: 0.65.
    pub phase2_coverage_gate: f64,
    /// MMR `lambda` for Phase 2 deduplication.  Spec default 0.7.
    pub phase2_mmr_lambda: f64,
    /// Token-overlap (Jaccard) threshold above which a Phase 2 hit is
    /// dropped as a hard duplicate.  Spec default 0.85.
    pub phase2_mmr_duplicate_threshold: f64,
    /// Enable the temporal-interval channel in Phase 2.  When the query
    /// has a parsed [`IntentProfile::temporal_window`], fans out
    /// BTIC-overlap + BTree-range queries across Fact / Observation /
    /// Episode and folds the hits into the RRF pool.  Default `true`.
    pub phase2_temporal_enabled: bool,
    /// Reference instant for resolving *relative* temporal phrases in the
    /// query ("last May", "two weeks ago"). `None` falls back to
    /// `Utc::now()`. Set this to the conversation/question anchor when
    /// recalling against a historical corpus — otherwise the Phase-2
    /// temporal channel computes a window around *now* that never overlaps
    /// old data, silently disabling the channel for the primary benchmark
    /// case. Threaded into [`crate::recall::intent::build_intent_at`].
    pub reference_ts: Option<chrono::DateTime<chrono::Utc>>,
    /// Enable the graph-spreading-activation channel in Phase 2.  When
    /// the query has any entity refs, runs weighted PPR over the entity
    /// graph and folds the activated nodes into the RRF pool.  Default
    /// `true`.
    pub phase2_graph_enabled: bool,
    /// PPR damping factor for the graph channel.  Standard 0.85.
    pub phase2_graph_damping: f64,
    /// PPR power-iteration cap.  Convergence is usually reached within
    /// 10-20 iterations; 30 is conservative.
    pub phase2_graph_max_iter: usize,
    /// Per-edge-type weight multipliers for the graph channel
    /// (Hindsight's `μ(ℓ)`).  Unmapped edge types default to 1.0.  See
    /// [`default_phase2_graph_edge_weights`].
    pub phase2_graph_edge_weights: std::collections::HashMap<String, f64>,
    /// Enable the cross-modal image channel in Phase 2 / Phase 3
    /// recall. Lazy-gated by `:KnowledgeBaseStats.modality_presence` —
    /// even when this is `true`, the channel stays dormant in a
    /// text-only corpus. Default `false`; Track B flips this on once
    /// per-modality embeddings populate. See [`crate::recall::modality`].
    pub enable_image_channel: bool,
    /// Mirror of [`Self::enable_image_channel`] for audio.
    pub enable_audio_channel: bool,
    /// Mirror of [`Self::enable_image_channel`] for video.
    pub enable_video_channel: bool,
    /// Mirror of [`Self::enable_image_channel`] for the multimodal
    /// joint-space column (Cohere v4 / Gemini Embed 2).
    pub enable_multimodal_channel: bool,
    /// Access-control scope. When [`ViewerScope::As`], `recall()` filters
    /// the returned bundle through [`crate::policy::filter_bundle`] so the
    /// viewer only sees Facts/Observations their visibility admits. Default
    /// [`ViewerScope::Unrestricted`] (no filtering) — set this for any
    /// caller serving a specific participant. See issue #5.
    pub viewer: ViewerScope,
    /// Dimensional hard-filters (session / participant / time) pushed into
    /// candidate generation. [`Default`] is unconstrained — an unscoped
    /// recall behaves exactly as before. Set via
    /// [`Agent::recall_in`](crate::Agent::recall_in) with a [`Scope`].
    pub dimensions: Dimensions,
    /// Internally-computed allow-set resolved from [`Self::dimensions`].
    /// Populated by [`recall`]; callers set `dimensions`, not this.
    pub dimensions_allow: Option<Vec<NodeId>>,
    /// F58 drift override. When `true` (default) and any query entity ref
    /// resolves to an Entity flagged `unstable` (F39 drift), the Phase-1
    /// early exit is suppressed and the cascade always runs Phase 2+ so
    /// queries about volatile entities check recent episodic evidence.
    pub drift_override_enabled: bool,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            limit: 15,
            token_budget: 8192,
            min_score: 0.001,
            vector_weight: 0.5,
            bm25_weight: 0.5,
            reranker_enabled: false,
            reranker_top_n: 50,
            reranker_apply_sigmoid: true,
            reranker_style: "cross-encoder".to_string(),
            recall_sparse_enabled: false,
            // Off by default. The naive "any connected entity matches
            // predicted type → boost" rule swamps top-K with off-target
            // hits when the predicted type is common in the corpus
            // (especially `measurement` for "how many" questions).
            // Measured −0.149 R@5 / −0.186 NDCG@5 on a 24-question
            // LongMemEval slice (2026-05-03). Set to a small value like
            // 1.05 if used as a tiebreaker, or leave at 1.0.
            answer_type_boost: 1.0,
            answer_type_top_n: 50,
            query_variants: Vec::new(),
            rrf_k: 60.0,
            per_variant_limit: 15,
            phase1_strategy: Phase1Strategy::Merge,
            phase1_boost_alpha: 0.3,
            phase2_coverage_gate: 0.65,
            phase2_mmr_lambda: 0.7,
            phase2_mmr_duplicate_threshold: 0.85,
            phase2_temporal_enabled: true,
            reference_ts: None,
            phase2_graph_enabled: true,
            phase2_graph_damping: 0.85,
            phase2_graph_max_iter: 30,
            phase2_graph_edge_weights: default_phase2_graph_edge_weights(),
            // Cross-modal channels stay off in Phase 3. Track B (xervo
            // PR 1) is what flips these on, since image/audio/video
            // embeddings only exist once those ingest paths land.
            enable_image_channel: false,
            enable_audio_channel: false,
            enable_video_channel: false,
            enable_multimodal_channel: false,
            viewer: ViewerScope::Unrestricted,
            dimensions: Dimensions::default(),
            dimensions_allow: None,
            drift_override_enabled: true,
        }
    }
}

/// Default edge-type weights for the graph spreading-activation
/// channel.  Higher weights mean activation propagates more readily
/// along that relation type.  Unmapped edge types default to 1.0.
///
/// Tuned so that semantic edges (entity links, fact support) dominate
/// over structural ones (session containment, chronological chaining).
pub fn default_phase2_graph_edge_weights() -> std::collections::HashMap<String, f64> {
    [
        ("ABOUT", 1.0),
        ("MENTIONS", 1.0),
        ("SUPPORTED_BY", 0.9),
        ("OBSERVED_IN", 0.7),
        ("HAS_CHUNK", 0.5),
        ("IN_SESSION", 0.3),
        ("RECORDED_BY", 0.3),
        ("FOLLOWED_BY", 0.3),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

impl RecallConfig {
    /// The cross-modal channels currently enabled, as shared
    /// [`Modality`](uniko_pipes::content::Modality) values.
    ///
    /// Ties the recall channel toggles to the same modality vocabulary the
    /// ingest router uses, so the two ends of the pipeline never diverge.
    /// (The multimodal joint-space column is a fusion channel, not a single
    /// modality, so it is not represented here.)
    #[must_use]
    pub fn enabled_modalities(&self) -> std::collections::HashSet<uniko_pipes::content::Modality> {
        use uniko_pipes::content::Modality;
        let mut set = std::collections::HashSet::new();
        if self.enable_image_channel {
            set.insert(Modality::Image);
        }
        if self.enable_audio_channel {
            set.insert(Modality::Audio);
        }
        if self.enable_video_channel {
            set.insert(Modality::Video);
        }
        set
    }

    /// Build from [`UnikoConfig`](uniko_store::config::UnikoConfig).
    pub fn from_uniko_config(cfg: &uniko_store::config::UnikoConfig) -> Self {
        Self {
            limit: cfg.recall_limit,
            token_budget: cfg.recall_token_budget,
            min_score: cfg.recall_min_score,
            vector_weight: cfg.recall_vector_weight,
            bm25_weight: cfg.recall_bm25_weight,
            reranker_enabled: cfg.reranker.enabled,
            reranker_top_n: cfg.reranker.top_n,
            reranker_apply_sigmoid: cfg.reranker.apply_sigmoid,
            reranker_style: cfg.reranker.style.clone(),
            recall_sparse_enabled: cfg.recall_sparse_enabled,
            // Rationale for the 1.0 default: see `Default for RecallConfig`.
            answer_type_boost: 1.0,
            answer_type_top_n: 50,
            query_variants: cfg.query_variants.clone(),
            rrf_k: cfg.rrf_k,
            per_variant_limit: cfg.recall_per_variant_limit.unwrap_or(cfg.recall_limit),
            phase1_strategy: parse_phase1_strategy(&cfg.phase1_strategy),
            phase1_boost_alpha: cfg.phase1_boost_alpha,
            phase2_coverage_gate: cfg.phase2_coverage_threshold,
            phase2_mmr_lambda: cfg.phase2_mmr_lambda,
            phase2_mmr_duplicate_threshold: cfg.phase2_mmr_duplicate_threshold,
            phase2_temporal_enabled: cfg.phase2_temporal_enabled,
            // Per-query anchor, not a KB-level setting; callers set it on
            // the RecallConfig they pass to `recall()`.
            reference_ts: None,
            phase2_graph_enabled: cfg.phase2_graph_enabled,
            phase2_graph_damping: cfg.phase2_graph_damping,
            phase2_graph_max_iter: cfg.phase2_graph_max_iter,
            phase2_graph_edge_weights: if cfg.phase2_graph_edge_weights.is_empty() {
                default_phase2_graph_edge_weights()
            } else {
                cfg.phase2_graph_edge_weights.clone()
            },
            // Cross-modal channel toggles default off here too; they
            // need both this flag and the per-KB modality_presence to
            // fire. UnikoConfig has no corresponding fields yet —
            // Track B will add them when image/audio ingest lands.
            enable_image_channel: false,
            enable_audio_channel: false,
            enable_video_channel: false,
            enable_multimodal_channel: false,
            viewer: ViewerScope::Unrestricted,
            dimensions: Dimensions::default(),
            dimensions_allow: None,
            drift_override_enabled: true,
        }
    }
}

/// Parse the string form stored in `UnikoConfig` into [`Phase1Strategy`].
///
/// Unknown values fall back to `Merge` with a warning — keeps the
/// recall path live on malformed config rather than panicking.
fn parse_phase1_strategy(s: &str) -> Phase1Strategy {
    match s.to_ascii_lowercase().as_str() {
        "merge" => Phase1Strategy::Merge,
        "boost" => Phase1Strategy::Boost,
        "off" | "none" | "disabled" => Phase1Strategy::Off,
        other => {
            tracing::warn!(
                value = other,
                "unknown phase1_strategy, defaulting to 'merge'",
            );
            Phase1Strategy::Merge
        }
    }
}

// ── RRF constant ────────────────────────────────────────────────────

/// Estimated tokens per recall item.
const TOKENS_PER_ITEM: usize = 50;

/// Sort `items` by score descending, truncate to `limit`, then walk in
/// rank order summing [`TOKENS_PER_ITEM`] until the budget would be
/// exceeded.  Returns `(kept_items, total_tokens)` for the caller to
/// wrap in a [`ContextBundle`] with phase-specific flags.
fn finalize_bundle(
    mut items: Vec<RecallItem>,
    limit: usize,
    token_budget: usize,
) -> (Vec<RecallItem>, usize) {
    crate::sort_by_score_desc(&mut items, |x| x.score);
    items.truncate(limit);
    let mut total_tokens = 0usize;
    let mut final_items = Vec::with_capacity(items.len());
    for item in items {
        total_tokens += TOKENS_PER_ITEM;
        if total_tokens > token_budget {
            break;
        }
        final_items.push(item);
    }
    (final_items, total_tokens)
}

// ── Main recall function ────────────────────────────────────────────

/// Recall relevant context from the memory graph.
///
/// Phase 1 implementation: executes Phase 3 (Broaden) only.
/// Searches across Messages, Chunks, Observations, and Entities via
/// fulltext BM25, vector similarity, and graph traversal, then fuses
/// results via RRF with tier weights.
///
/// # Errors
///
/// Returns [`UnikoError::Search`] on search failure.
pub async fn recall(
    kb: &KnowledgeBase,
    query: &str,
    config: &RecallConfig,
) -> Result<ContextBundle, UnikoError> {
    let mut bundle = recall_unfiltered(kb, query, config).await?;
    // Redaction gate: soft-forgotten content is dropped for every recall,
    // regardless of viewer scope (forgetting is content redaction, not
    // access control). Runs before the access-control filter.
    crate::policy::filter_redacted(kb, &mut bundle).await?;
    // Access-control gate (issue #5). Fail-open is an explicit, named
    // choice (`ViewerScope::Unrestricted`), not a silent default omission.
    match &config.viewer {
        ViewerScope::As(viewer) => {
            crate::policy::filter_bundle(kb, &mut bundle, viewer).await?;
        }
        ViewerScope::Unrestricted => {
            if bundle
                .items
                .iter()
                .any(|i| matches!(i.kind, RecallKind::Fact | RecallKind::Observation))
            {
                tracing::warn!(
                    "recall() returning policy-scoped items UNFILTERED \
                     (ViewerScope::Unrestricted) — set RecallConfig.viewer to scope visibility"
                );
            }
        }
    }
    // Lineage: stamp each surviving item with the messages/attachments it
    // came from. Runs once over the (small) final bundle — off the hot
    // candidate-generation path. Best-effort: a resolution failure leaves
    // `sources` empty rather than failing the recall.
    if let Err(e) = populate_sources(kb, &mut bundle).await {
        tracing::debug!(error = %e, "source lineage resolution failed; items keep empty sources");
    }
    Ok(bundle)
}

/// Stamp every item in `bundle` with its source lineage — the
/// messages/attachments its content derives from.
///
/// Resolves over the final item ids in a handful of batched queries, one per
/// kind, then groups the flat rows in Rust (no aggregation-in-Cypher).
/// Chunks resolve to their owning message, or to the attachment/document the
/// chunk belongs to; Observations and Facts to the messages of their
/// evidence; Messages to themselves. Aggregates (Topic/Procedure/Episode/
/// Other) keep an empty list.
///
/// # Errors
///
/// Returns [`UnikoError`] when a graph query fails.
async fn populate_sources(
    kb: &KnowledgeBase,
    bundle: &mut ContextBundle,
) -> Result<(), UnikoError> {
    let ids_of = |kind: RecallKind| -> Vec<NodeId> {
        bundle
            .items
            .iter()
            .filter(|i| i.kind == kind)
            .map(|i| i.node_id)
            .collect()
    };
    let chunk_ids = ids_of(RecallKind::Chunk);
    let obs_ids = ids_of(RecallKind::Observation);
    let fact_ids = ids_of(RecallKind::Fact);
    let msg_ids = ids_of(RecallKind::Message);

    let mut sources: HashMap<NodeId, Vec<RecallSource>> = HashMap::new();

    // Chunks: owning Message, or Artifact (direct text/markdown ingest or a
    // tiered-PDF page→block chain). An Artifact that is `ATTACHED_TO` a
    // message is an Attachment; otherwise a standalone Document.
    if !chunk_ids.is_empty() {
        // Keep the ownership alternatives as separate MATCHes. Besides being
        // easier to extend, this avoids a chain of OPTIONAL MATCHes turning a
        // present Artifact owner into NULL when another ownership arm misses.
        let message_rows = kb
            .query_cypher(
                "MATCH (m:Message)-[:HAS_CHUNK]->(c:Chunk) WHERE id(c) IN $ids \
                 RETURN id(c) AS cid, m.message_id AS mid, c.chunk_id AS chid",
                &ids_param(&chunk_ids),
            )
            .await?;
        let mut artifact_rows = kb
            .query_cypher(
                "MATCH (a:Artifact)-[:HAS_CHUNK]->(c:Chunk) WHERE id(c) IN $ids \
                 RETURN id(c) AS cid, a.artifact_id AS aid, c.chunk_id AS chid",
                &ids_param(&chunk_ids),
            )
            .await?;
        artifact_rows.extend(
            kb.query_cypher(
                "MATCH (a:Artifact)-[:HAS_PAGE]->(:Page)-[:CONTAINS]->(:Block)-[:HAS_CHUNK]->(c:Chunk) \
                 WHERE id(c) IN $ids \
                 RETURN id(c) AS cid, a.artifact_id AS aid, c.chunk_id AS chid",
                &ids_param(&chunk_ids),
            )
            .await?,
        );

        for r in &message_rows {
            let (Some(cid), Some(message_id)) = (row_int(r, "cid"), row_str(r, "mid")) else {
                continue;
            };
            let src = RecallSource::Message {
                message_id,
                chunk_id: row_str(r, "chid"),
            };
            let entry = sources.entry(cid).or_default();
            if !entry.contains(&src) {
                entry.push(src);
            }
        }

        let artifact_ids: Vec<String> = artifact_rows
            .iter()
            .filter_map(|r| row_str(r, "aid"))
            .collect();
        let attached = attachment_messages(kb, &artifact_ids).await?;
        for r in &artifact_rows {
            let (Some(cid), Some(artifact_id)) = (row_int(r, "cid"), row_str(r, "aid")) else {
                continue;
            };
            let chunk_id = row_str(r, "chid");
            let src = match attached.get(&artifact_id) {
                Some(message_id) => RecallSource::Attachment {
                    message_id: message_id.clone(),
                    artifact_id,
                    chunk_id,
                },
                None => RecallSource::Document {
                    artifact_id,
                    chunk_id,
                },
            };
            let entry = sources.entry(cid).or_default();
            if !entry.contains(&src) {
                entry.push(src);
            }
        }
    }

    // Observations: the message they were observed in.
    if !obs_ids.is_empty() {
        let rows = kb
            .query_cypher(
                "MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message) WHERE id(o) IN $ids \
                 RETURN id(o) AS nid, m.message_id AS mid",
                &ids_param(&obs_ids),
            )
            .await?;
        collect_message_sources(&rows, &mut sources);
    }

    // Facts: the messages of their supporting observations (distinct).
    if !fact_ids.is_empty() {
        let rows = kb
            .query_cypher(
                "MATCH (f:Fact)-[:SUPPORTED_BY]->(:Observation)-[:OBSERVED_IN]->(m:Message) \
                 WHERE id(f) IN $ids RETURN DISTINCT id(f) AS nid, m.message_id AS mid",
                &ids_param(&fact_ids),
            )
            .await?;
        collect_message_sources(&rows, &mut sources);
    }

    // Messages: themselves.
    if !msg_ids.is_empty() {
        let rows = kb
            .query_cypher(
                "MATCH (m:Message) WHERE id(m) IN $ids RETURN id(m) AS nid, m.message_id AS mid",
                &ids_param(&msg_ids),
            )
            .await?;
        collect_message_sources(&rows, &mut sources);
    }

    for item in &mut bundle.items {
        if let Some(s) = sources.remove(&item.node_id) {
            item.sources = s;
        }
    }
    Ok(())
}

/// Push a `Message` source per `(nid, mid)` row into `sources`.
fn collect_message_sources(
    rows: &[HashMap<String, Value>],
    sources: &mut HashMap<NodeId, Vec<RecallSource>>,
) {
    for r in rows {
        if let (Some(nid), Some(message_id)) = (row_int(r, "nid"), row_str(r, "mid")) {
            sources.entry(nid).or_default().push(RecallSource::Message {
                message_id,
                chunk_id: None,
            });
        }
    }
}

/// Map `artifact_id` → `message_id` for artifacts `ATTACHED_TO` a message.
async fn attachment_messages(
    kb: &KnowledgeBase,
    artifact_ids: &[String],
) -> Result<HashMap<String, String>, UnikoError> {
    if artifact_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let params = HashMap::from([(
        "aids".to_string(),
        Value::List(
            artifact_ids
                .iter()
                .map(|a| Value::String(a.clone()))
                .collect(),
        ),
    )]);
    let rows = kb
        .query_cypher(
            "MATCH (a:Artifact)-[:ATTACHED_TO]->(m:Message) WHERE a.artifact_id IN $aids \
             RETURN a.artifact_id AS aid, m.message_id AS mid",
            &params,
        )
        .await?;
    let mut out = HashMap::new();
    for r in &rows {
        if let (Some(aid), Some(mid)) = (row_str(r, "aid"), row_str(r, "mid")) {
            out.insert(aid, mid);
        }
    }
    Ok(out)
}

/// Build a `$ids` list parameter from node ids.
fn ids_param(ids: &[NodeId]) -> HashMap<String, Value> {
    HashMap::from([(
        "ids".to_string(),
        Value::List(ids.iter().map(|&n| Value::Int(n)).collect()),
    )])
}

/// Read an integer cell from a decoded row.
fn row_int(row: &HashMap<String, Value>, key: &str) -> Option<NodeId> {
    match row.get(key) {
        Some(Value::Int(i)) => Some(*i),
        _ => None,
    }
}

/// Read a string cell from a decoded row.
fn row_str(row: &HashMap<String, Value>, key: &str) -> Option<String> {
    match row.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

/// The recall cascade proper, before any access-control filtering.
/// `recall()` wraps this and applies [`ViewerScope`].
async fn recall_unfiltered(
    kb: &KnowledgeBase,
    query: &str,
    config: &RecallConfig,
) -> Result<ContextBundle, UnikoError> {
    if query.is_empty() {
        return Ok(empty_bundle());
    }

    // Resolve the dimensional scope into an allow-set once, threaded to the
    // candidate generators via `config.dimensions_allow`. When unconstrained
    // (the default) this is skipped and the cascade runs exactly as before.
    let owned_config;
    let config = if !config.dimensions.is_unconstrained() && config.dimensions_allow.is_none() {
        let filter = uniko_store::repository::recall::ScopeFilter {
            sessions: config.dimensions.sessions.clone(),
            participants: config.dimensions.participants.clone(),
            since: config.dimensions.since,
            until: config.dimensions.until,
        };
        let allow = kb.resolve_scope_allow_set(&filter).await?;
        let mut scoped = config.clone();
        scoped.dimensions_allow = Some(allow);
        owned_config = scoped;
        &owned_config
    } else {
        config
    };

    let intent =
        intent::build_intent_at(kb, query, &config.query_variants, config.reference_ts).await?;
    tracing::debug!(
        variant_count = intent.variants.len(),
        entity_refs = ?intent.entity_refs,
        "recall intent built",
    );

    // ── Phase 1: Compact ────────────────────────────────────────────
    // Vector search over the consolidated Semantic / Procedural tier
    // (Facts, Procedures, Topics).  When the corpus has been
    // consolidated and the top hits cover the question's facets, this
    // alone suffices and we skip the heavier Phase 3 broaden.
    let phase1_items = phase1_compact(kb, &intent, config).await;
    let phase1_coverage = compute_coverage(&phase1_items, intent.facet_count);
    let phase1_sufficient = phase1_coverage >= COVERAGE_GATE_PHASE1 && phase1_items.len() >= 3;

    // F58 drift override: when the query references an entity flagged
    // `unstable` (F39), do not trust the compiled Phase-1 view — force
    // Phase 2+ so recent episodic evidence is consulted. Only worth the
    // lookup when Phase 1 would otherwise have exited early.
    let drift_forced = phase1_sufficient
        && config.drift_override_enabled
        && kb.any_unstable_entities(&intent.entity_refs).await;
    if drift_forced {
        tracing::info!(
            entity_refs = ?intent.entity_refs,
            "drift override: unstable entity referenced — forcing phase 2+"
        );
    }

    if phase1_sufficient && !drift_forced {
        tracing::info!(
            phase1_items = phase1_items.len(),
            coverage = phase1_coverage,
            "phase 1 (compact) sufficient — skipping phase 3"
        );
        let (final_items, total_tokens) =
            finalize_bundle(phase1_items, config.limit, config.token_budget);
        return Ok(ContextBundle {
            total_tokens,
            items: final_items,
            phase1_only: true,
            phase2_only: false,
            coverage: phase1_coverage,
        });
    }

    // ── Phase 2: Expand ─────────────────────────────────────────────
    // RRF-fuse vector + fulltext hits over Episode / Observation /
    // Message, apply MMR deduplication, and check the 0.65 coverage
    // gate.  When met, we return the Phase 2 bundle (merged with
    // Phase 1 hits) without paying for Phase 3.
    let phase2_items_raw = phase2_expand(kb, &intent, config, None).await;
    let phase2_items = mmr::mmr_dedup(
        phase2_items_raw,
        config.phase2_mmr_lambda,
        config.phase2_mmr_duplicate_threshold,
        None,
    );
    let phase2_coverage = compute_coverage(&phase2_items, intent.facet_count);
    let phase2_sufficient =
        phase2_coverage >= config.phase2_coverage_gate && phase2_items.len() >= 3;

    if phase2_sufficient {
        tracing::info!(
            phase2_items = phase2_items.len(),
            coverage = phase2_coverage,
            "phase 2 (expand) sufficient — skipping phase 3"
        );
        // Cap on Phase 1 items merged into the Phase 2 bundle.  Same
        // value used at the Phase 3 merge site (see `Phase1Strategy::Merge`
        // branch below).
        const PHASE1_MERGE_CAP: usize = 3;
        let mut combined: Vec<RecallItem> = phase2_items;
        if matches!(config.phase1_strategy, Phase1Strategy::Merge) {
            let mut p1 = phase1_items;
            crate::sort_by_score_desc(&mut p1, |x| x.score);
            p1.truncate(PHASE1_MERGE_CAP);
            combined.extend(p1);
        }

        let (final_items, total_tokens) =
            finalize_bundle(combined, config.limit, config.token_budget);
        return Ok(ContextBundle {
            total_tokens,
            items: final_items,
            phase1_only: false,
            phase2_only: true,
            coverage: phase2_coverage,
        });
    }

    // Per-variant fan-out: each query variant runs the full hybrid +
    // entity-scoped pipeline; the resulting per-variant ranked lists
    // are RRF-fused to produce the final candidate set. Variants run
    // concurrently — each future is independent I/O against lance.
    let variant_futures: Vec<_> = intent
        .variants
        .iter()
        .map(|variant| {
            let entity_refs = intent.entity_refs.clone();
            async move {
                let ranked = run_recall_for_variant(
                    kb,
                    variant.vector.as_slice(),
                    variant.text.as_str(),
                    &entity_refs,
                    config,
                )
                .await;
                (variant.label, ranked)
            }
        })
        .collect();
    let variant_results: Vec<(&'static str, Vec<RankedHit>)> = join_all(variant_futures).await;

    for (label, hits) in &variant_results {
        tracing::info!(variant = label, hits = hits.len(), "variant complete");
    }

    // Optional learned-sparse Chunk channel: one auto-embedded
    // `uni.sparse.query` per variant, fused as extra ranked lists in the
    // same RRF pool (a chunk matched by both dense and sparse accumulates
    // score). Observation's sparse channel rides in Phase 2. Gated on a
    // hybrid embedder via `recall_sparse_enabled`.
    let sparse_results: Vec<Vec<RankedHit>> = if config.recall_sparse_enabled {
        let sparse_futures: Vec<_> = intent
            .variants
            .iter()
            .map(|variant| {
                run_sparse_source(
                    kb,
                    "Chunk",
                    "text",
                    variant.text.clone(),
                    config.per_variant_limit as i64,
                    config.dimensions_allow.as_deref(),
                )
            })
            .collect();
        join_all(sparse_futures).await
    } else {
        Vec::new()
    };

    // RRF fusion across variants (+ sparse channels). Pure function so it
    // can be unit tested with synthetic ranked lists.
    let scored = rrf_fuse(
        variant_results
            .iter()
            .map(|(_, hits)| hits.as_slice())
            .chain(sparse_results.iter().map(Vec::as_slice)),
        config.rrf_k,
    );

    tracing::debug!(candidates = scored.len(), "recall candidates");

    if scored.is_empty() {
        // Phase 3 (Broaden) found nothing — but Phase 2 (Expand) may
        // have populated `phase2_items` that we'd otherwise silently
        // discard.  Fall back to those rather than returning an empty
        // bundle.  Common on tiny KBs (no Chunks consolidated) and on
        // queries whose facet count under-counts coverage.
        if !phase2_items.is_empty() {
            tracing::info!(
                phase2_items = phase2_items.len(),
                "phase 3 empty — returning phase 2 fallback bundle"
            );
            let (final_items, total_tokens) =
                finalize_bundle(phase2_items, config.limit, config.token_budget);
            let coverage = compute_coverage(&final_items, intent.facet_count);
            return Ok(ContextBundle {
                total_tokens,
                items: final_items,
                phase1_only: false,
                phase2_only: true,
                coverage,
            });
        }
        return Ok(empty_bundle());
    }

    // ── Build items with tier weights ──────────────────────────
    let mut items: Vec<RecallItem> = scored
        .into_iter()
        .filter(|(_, (_, content, _))| !content.is_empty())
        .map(|(nid, (label, content, score))| {
            let kind = RecallKind::from_label(&label);
            RecallItem {
                node_id: nid,
                kind,
                score: score * kind.tier().weight(),
                content,
                sources: Vec::new(),
            }
        })
        .collect();

    crate::sort_by_score_desc(&mut items, |x| x.score);

    // ── Optional reranker stage ────────────────────────────────────
    // Re-score the top RRF candidates before truncating to the recall
    // limit. Disabled by default — when off, this is a no-op. The
    // `"colbert"` style re-scores via late-interaction MaxSim in-process
    // (no reranker model); all other styles call the xervo cross-encoder.
    if config.reranker_enabled && !items.is_empty() && config.reranker_style == "colbert" {
        colbert_rerank(kb, query, &mut items, config).await;
    } else if config.reranker_enabled && !items.is_empty() {
        let top_n = config.reranker_top_n.min(items.len());
        let docs: Vec<&str> = items[..top_n].iter().map(|i| i.content.as_str()).collect();
        match kb
            .rerank(uniko_store::schema::RERANK_ALIAS, query, &docs)
            .await
        {
            Ok(scored) => {
                let mut head: Vec<RecallItem> = scored
                    .into_iter()
                    .filter_map(|(index, score)| {
                        items.get(index).cloned().map(|mut it| {
                            it.score = if config.reranker_apply_sigmoid {
                                1.0 / (1.0 + f64::exp(-(score as f64)))
                            } else {
                                score as f64
                            };
                            it
                        })
                    })
                    .collect();
                let tail = items.split_off(top_n);
                head.extend(tail);
                items = head;
                crate::sort_by_score_desc(&mut items, |x| x.score);
            }
            Err(e) => tracing::warn!(error = %e, "reranker call failed, falling back to RRF order"),
        }
    }

    // ── Answer-type boost ────────────────────────────────────────────
    // When the question's surface form predicts an entity_type
    // ("where" → location, "how many" → measurement, …), boost any
    // top-N item whose connected entities include one of that type.
    // No-op when no rule fired or the boost is 1.0.
    if config.answer_type_boost > 1.0
        && let Some(target_type) = intent.expected_answer_type
        && !items.is_empty()
    {
        let n = config.answer_type_top_n.min(items.len());
        let mut matched = 0usize;
        for item in items.iter_mut().take(n) {
            if entity_type_match(kb, item.node_id, target_type).await {
                item.score *= config.answer_type_boost;
                matched += 1;
            }
        }
        if matched > 0 {
            crate::sort_by_score_desc(&mut items, |x| x.score);
        }
        tracing::info!(
            target_type,
            checked = n,
            matched,
            boost = config.answer_type_boost,
            "answer-type reweight applied",
        );
    }

    // Phase 1 contribution: one of three strategies (Merge / Boost / Off).
    //
    // - Merge: cap=3 interleave by score (best on conv-26 in v1
    //   ablations; conv-26 0.750 / conv-30 0.802 with the rest of the
    //   stack).
    // - Boost: Facts/Obs influence the Phase-3 chunks' scores via a
    //   session-level walk (Phase C, RFE rfe-p4-recall-evolution).
    //   Bundle remains 100% chunks; gold-bearing text always present.
    // - Off: skip Phase 1 contributions entirely.
    match config.phase1_strategy {
        Phase1Strategy::Merge => {
            const PHASE1_FALLBACK_CAP: usize = 3;
            let mut phase1_top = phase1_items;
            crate::sort_by_score_desc(&mut phase1_top, |x| x.score);
            phase1_top.truncate(PHASE1_FALLBACK_CAP);

            let mut merged: HashMap<NodeId, RecallItem> = HashMap::new();
            for item in items.drain(..).chain(phase1_top) {
                merged
                    .entry(item.node_id)
                    .and_modify(|existing| {
                        if item.score > existing.score {
                            *existing = item.clone();
                        }
                    })
                    .or_insert(item);
            }
            items = merged.into_values().collect();
            crate::sort_by_score_desc(&mut items, |x| x.score);
        }
        Phase1Strategy::Boost => {
            let boost_map =
                session_boost_signals(kb, &phase1_items, config.phase1_boost_alpha).await;
            let mut boosted = 0usize;
            for item in &mut items {
                if let Some(delta) = boost_map.get(&item.node_id) {
                    item.score += delta;
                    boosted += 1;
                }
            }
            if boosted > 0 {
                crate::sort_by_score_desc(&mut items, |x| x.score);
            }
            tracing::info!(
                phase1_facts = phase1_items.len(),
                boost_targets = boost_map.len(),
                boosted_items = boosted,
                "phase 1 session boost applied",
            );
        }
        Phase1Strategy::Off => {
            // No Phase 1 contribution.  Items list stays as Phase 3 +
            // reranker output untouched.
        }
    }

    let (final_items, total_tokens) = finalize_bundle(items, config.limit, config.token_budget);

    let coverage = compute_coverage(&final_items, intent.facet_count);

    Ok(ContextBundle {
        total_tokens,
        items: final_items,
        phase1_only: false,
        phase2_only: false,
        coverage,
    })
}

/// Re-rank the top candidates by ColBERT late-interaction MaxSim.
///
/// Only Chunk and Observation carry a `colbert_embedding` column, so this
/// re-scores those candidates within the top `reranker_top_n` window and
/// leaves the rest at their fused score (partial rerank). MaxSim scores are
/// rescaled into the window's existing score band so the re-scored and
/// untouched candidates interleave without a scale jump. Best-effort: on any
/// embed/query failure it logs and leaves RRF order intact.
async fn colbert_rerank(
    kb: &KnowledgeBase,
    query: &str,
    items: &mut [RecallItem],
    config: &RecallConfig,
) {
    let top = config.reranker_top_n.min(items.len());
    if top == 0 {
        return;
    }
    // Query-side per-token embedding; document-side vectors are auto-embedded
    // into `colbert_embedding`.
    let qmulti = match uniko_extract::embedding::embed_multivector_query(kb, query).await {
        Ok(q) if !q.is_empty() => q,
        Ok(_) => return,
        Err(e) => {
            tracing::warn!(error = %e, "colbert query embed failed, keeping RRF order");
            return;
        }
    };

    // Candidate ids per colbert-bearing label, taken from the rerank window.
    let ids_of = |kind: RecallKind| -> Vec<NodeId> {
        items[..top]
            .iter()
            .filter(|i| i.kind == kind)
            .map(|i| i.node_id)
            .collect()
    };
    let chunk_ids = ids_of(RecallKind::Chunk);
    let obs_ids = ids_of(RecallKind::Observation);
    if chunk_ids.is_empty() && obs_ids.is_empty() {
        return;
    }

    let mut maxsim: HashMap<NodeId, f64> = HashMap::new();
    for (label, content_field, ids) in [
        ("Chunk", "text", &chunk_ids),
        ("Observation", "content", &obs_ids),
    ] {
        if ids.is_empty() {
            continue;
        }
        match kb
            .recall_colbert_maxsim(label, "colbert_embedding", content_field, &qmulti, ids)
            .await
        {
            Ok(rows) => {
                for r in rows {
                    maxsim.insert(r.node_id, r.score);
                }
            }
            Err(e) => tracing::warn!(label, error = %e, "colbert maxsim query failed"),
        }
    }
    if maxsim.is_empty() {
        return;
    }

    // Rescale MaxSim into the current top-window band so re-scored items
    // interleave with the untouched (non-colbert) candidates instead of
    // jumping to a different score scale.
    let band_ceiling = items[..top]
        .iter()
        .map(|i| i.score)
        .fold(f64::MIN, f64::max)
        .max(f64::MIN_POSITIVE);
    let max_ms = maxsim.values().copied().fold(f64::MIN, f64::max);
    if max_ms <= 0.0 {
        return;
    }
    for item in items[..top].iter_mut() {
        if let Some(&ms) = maxsim.get(&item.node_id) {
            item.score = ms / max_ms * band_ceiling;
        }
    }
    crate::sort_by_score_desc(items, |x| x.score);
}

/// Phase 1 (Compact) coverage threshold from spec §IX.
///
/// When Phase 1 alone clears this, we skip the heavier Phase 3
/// (Broaden) search entirely — a Mem0-style "extracted fact" hit is
/// dense enough on its own.
const COVERAGE_GATE_PHASE1: f64 = 0.75;

/// Phase 1 (Compact): vector search over consolidated knowledge tiers.
///
/// Searches `Fact.embedding` (top-20), `Procedure.embedding` (top-10),
/// and `Topic.embedding` (top-5) using the intent's primary embedding.
/// Until P5/P6 ship the Procedure and Topic queries return zero rows;
/// the implementation runs them anyway so activating those tiers is a
/// pure data-side change with no recall code modifications required.
///
/// Returns an empty vec on any error — Phase 1 is opportunistic and
/// must never break the cascade.
async fn phase1_compact(
    kb: &KnowledgeBase,
    intent: &IntentProfile,
    config: &RecallConfig,
) -> Vec<RecallItem> {
    let qvec = intent.intent_vec();
    if qvec.is_empty() {
        return Vec::new();
    }

    let targets: &[(&str, &str, &str, i64)] = &[
        // (label, embedding field, content field, top_k)
        ("Fact", "embedding", "object", 20),
        ("Procedure", "embedding", "name", 10),
        ("Topic", "embedding", "name", 5),
    ];

    let mut out: HashMap<NodeId, RecallItem> = HashMap::new();
    for &(label, embed_field, content_field, top_k) in targets {
        let rows = match kb
            .recall_vector_search(
                label,
                embed_field,
                content_field,
                qvec,
                top_k,
                config.dimensions_allow.as_deref(),
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(label, error = %e, "phase1 compact query failed");
                continue;
            }
        };
        for row in rows {
            if row.score < config.min_score {
                continue;
            }
            let kind = RecallKind::from_label(&row.label);
            let weighted = row.score * kind.tier().weight();
            // Same node could appear from multiple targets if labels
            // overlap (they don't today, but be safe): keep the higher
            // weighted score.
            out.entry(row.node_id)
                .and_modify(|existing| {
                    if weighted > existing.score {
                        existing.score = weighted;
                    }
                })
                .or_insert(RecallItem {
                    node_id: row.node_id,
                    kind,
                    score: weighted,
                    content: row.content,
                    sources: Vec::new(),
                });
        }
    }
    out.into_values().collect()
}

/// Phase 2 (Expand): RRF-fuse vector + fulltext hits over the
/// episodic tier (Episode, Observation, Message).
///
/// Sources and top-k from spec §IX-A retrieval contract:
/// - Vector top-20 on `Episode.embedding`
/// - Vector top-20 on `Observation.embedding`
/// - Vector top-10 on `Message.embedding`
/// - Fulltext top-20 on `Observation.content`
/// - Fulltext top-10 on `Message.content`
///
/// RRF (`k = 60`) fuses the five ranked lists; per-source min-max
/// normalisation is applied before fusion so heterogeneous scoring
/// ranges (cosine, BM25) compare like-for-like.  Tier weight is
/// applied (Episodic = 0.7).  Returns an empty `Vec` on any internal
/// error — Phase 2 is opportunistic.
pub async fn phase2_expand(
    kb: &KnowledgeBase,
    intent: &IntentProfile,
    config: &RecallConfig,
    counters: Option<crate::recall::modality::RecallCounters>,
) -> Vec<RecallItem> {
    let Some(act) = phase2_activation(kb, intent, config).await else {
        return Vec::new();
    };
    let qvec = act.qvec;
    let qtxt = act.qtxt;
    let qm = &intent.query_modalities;
    let (has_vec, has_txt, has_temporal, has_graph) =
        (act.has_vec, act.has_txt, act.has_temporal, act.has_graph);
    let (fire_image, fire_audio, fire_multimodal) =
        (act.fire_image, act.fire_audio, act.fire_multimodal);

    // (label, mode, content_field, top_k).  mode ∈ {"vector","fulltext"}.
    let sources: &[(&str, &str, &str, i64)] = &[
        ("Episode", "vector", "action_type", 20),
        ("Observation", "vector", "content", 20),
        ("Message", "vector", "content", 10),
        ("Observation", "fulltext", "content", 20),
        ("Message", "fulltext", "content", 10),
    ];

    // Fire all sources in parallel.  Each task returns `Vec::new()` on
    // internal failure (consistent with the previous per-source
    // `continue` semantics), so no source can fail the phase.
    let mut futs: Vec<futures::future::BoxFuture<'_, Vec<RankedHit>>> =
        Vec::with_capacity(sources.len() + 5);
    for &(label, mode, content_field, top_k) in sources {
        let skip = match mode {
            "vector" => !has_vec,
            "fulltext" => !has_txt,
            _ => true,
        };
        if skip {
            continue;
        }
        let qvec_ref = qvec;
        let qtxt_owned = qtxt.to_string();
        futs.push(Box::pin(run_phase2_source(
            kb,
            label,
            mode,
            content_field,
            top_k,
            qvec_ref,
            qtxt_owned,
            "embedding",
            config.dimensions_allow.as_deref(),
        )));
    }
    // Learned-sparse Observation channel, fused into the same Phase 2 RRF
    // pool. Gated on a hybrid embedder; auto-embeds the query text.
    eprintln!(
        "RECALL_PROF gate recall_sparse_enabled={}",
        config.recall_sparse_enabled
    );
    if config.recall_sparse_enabled {
        futs.push(Box::pin(run_sparse_source(
            kb,
            "Observation",
            "content",
            qtxt.to_string(),
            20,
            config.dimensions_allow.as_deref(),
        )));
    }
    // Cross-modal channels query :Artifact.{image,audio,multimodal}_embedding
    // via the same `run_phase2_source` helper. `content_field = "kind"`
    // surfaces the artifact kind into the RecallItem.content slot (so
    // an image artifact reads as e.g. `"image"`); downstream consumers
    // can resolve URI via the parent edge if needed.
    if fire_image && let Some(v) = qm.image_vec.as_deref() {
        if let Some(c) = counters.as_ref() {
            c.bump_image();
        }
        futs.push(Box::pin(run_phase2_source(
            kb,
            "Artifact",
            "vector",
            "kind",
            20,
            v,
            String::new(),
            "image_embedding",
            config.dimensions_allow.as_deref(),
        )));
    }
    if fire_audio && let Some(v) = qm.audio_vec.as_deref() {
        if let Some(c) = counters.as_ref() {
            c.bump_audio();
        }
        futs.push(Box::pin(run_phase2_source(
            kb,
            "Artifact",
            "vector",
            "kind",
            20,
            v,
            String::new(),
            "audio_embedding",
            config.dimensions_allow.as_deref(),
        )));
    }
    if fire_multimodal {
        // Multimodal joint space: prefer image vec, fall back to audio.
        let v = qm.image_vec.as_deref().or(qm.audio_vec.as_deref());
        if let Some(v) = v {
            if let Some(c) = counters.as_ref() {
                c.bump_multimodal();
            }
            futs.push(Box::pin(run_phase2_source(
                kb,
                "Artifact",
                "vector",
                "kind",
                20,
                v,
                String::new(),
                "multimodal_embedding",
                config.dimensions_allow.as_deref(),
            )));
        }
    }
    if has_temporal {
        futs.push(Box::pin(phase2_temporal(kb, intent, config)));
    }
    if has_graph {
        futs.push(Box::pin(phase2_graph_activation(kb, intent, config)));
    }

    let per_source: Vec<Vec<RankedHit>> = futures::future::join_all(futs)
        .await
        .into_iter()
        .filter(|v| !v.is_empty())
        .collect();

    if per_source.is_empty() {
        return Vec::new();
    }

    fuse_and_score_phase2(per_source, config)
}

/// Activation signals + cached query text/vec for phase-2 expansion.
///
/// `None` means no channel will fire and the phase can short-circuit.
struct Phase2Activation<'a> {
    qvec: &'a [f32],
    qtxt: &'a str,
    has_vec: bool,
    has_txt: bool,
    has_temporal: bool,
    has_graph: bool,
    fire_image: bool,
    fire_audio: bool,
    fire_multimodal: bool,
}

/// Compute which phase-2 channels will fire for this query.
async fn phase2_activation<'a>(
    kb: &KnowledgeBase,
    intent: &'a IntentProfile,
    config: &RecallConfig,
) -> Option<Phase2Activation<'a>> {
    let qvec = intent.intent_vec();
    let qtxt = intent
        .variants
        .iter()
        .find(|v| v.label == "keywords")
        .map(|v| v.text.as_str())
        .or_else(|| intent.variants.first().map(|v| v.text.as_str()))
        .unwrap_or("");
    let has_vec = !qvec.is_empty();
    let has_txt = !qtxt.is_empty();
    let has_temporal = intent.temporal_window.is_some() && config.phase2_temporal_enabled;
    let has_graph = !intent.entity_refs.is_empty() && config.phase2_graph_enabled;
    // Cross-modal channels: each gated on (presence flag ∧ toggle ∧
    // query has a per-modality vec).
    let presence = kb.read_modality_presence().await.unwrap_or_default();
    let qm = &intent.query_modalities;
    let fire_image =
        crate::recall::modality::image_channel_active(&presence, config.enable_image_channel)
            && qm.image_vec.is_some();
    let fire_audio =
        crate::recall::modality::audio_channel_active(&presence, config.enable_audio_channel)
            && qm.audio_vec.is_some();
    let fire_multimodal = crate::recall::modality::multimodal_channel_active(
        &presence,
        config.enable_multimodal_channel,
    ) && (qm.image_vec.is_some() || qm.audio_vec.is_some());
    if !has_vec
        && !has_txt
        && !has_temporal
        && !has_graph
        && !fire_image
        && !fire_audio
        && !fire_multimodal
    {
        return None;
    }
    Some(Phase2Activation {
        qvec,
        qtxt,
        has_vec,
        has_txt,
        has_temporal,
        has_graph,
        fire_image,
        fire_audio,
        fire_multimodal,
    })
}

/// RRF-fuse the per-source ranked lists, apply tier weights, and sort
/// the final phase-2 bundle by score descending.
fn fuse_and_score_phase2(
    per_source: Vec<Vec<RankedHit>>,
    config: &RecallConfig,
) -> Vec<RecallItem> {
    let fused = rrf_fuse(per_source.iter().map(|v| v.as_slice()), config.rrf_k);
    let mut items: Vec<RecallItem> = fused
        .into_iter()
        .filter(|(_, (_, content, _))| !content.is_empty())
        .map(|(nid, (label, content, score))| {
            let kind = RecallKind::from_label(&label);
            RecallItem {
                node_id: nid,
                kind,
                score: score * kind.tier().weight(),
                content,
                sources: Vec::new(),
            }
        })
        .filter(|item| item.score >= config.min_score)
        .collect();
    crate::sort_by_score_desc(&mut items, |x| x.score);
    items
}

/// Run a learned-sparse channel for one label, returning RRF-ready hits.
///
/// `qtxt` is auto-embedded into a sparse query vector by the hybrid embed
/// alias (no precomputed vector needed). Returns `Vec::new()` on failure so
/// the fan-out tolerates per-source errors. The sparse companion to
/// [`run_phase2_source`]; used by both Phase 2 (Observation) and the Phase 3
/// variant fan-out (Chunk).
async fn run_sparse_source(
    kb: &KnowledgeBase,
    label: &str,
    content_field: &str,
    qtxt: String,
    top_k: i64,
    allow: Option<&[NodeId]>,
) -> Vec<RankedHit> {
    let start = std::time::Instant::now();
    let rows = match kb
        .recall_sparse_search(
            label,
            "sparse_embedding",
            content_field,
            &qtxt,
            top_k,
            allow,
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(label, error = %e, "sparse query failed");
            return Vec::new();
        }
    };
    eprintln!(
        "RECALL_PROF sparse_source label={label} rows={} elapsed_ms={}",
        rows.len(),
        start.elapsed().as_millis() as u64
    );
    let mut hits: Vec<RankedHit> = rows
        .into_iter()
        .filter(|r| !r.content.is_empty())
        .map(|r| RankedHit {
            node_id: r.node_id,
            label: if r.label.is_empty() {
                label.to_string()
            } else {
                r.label
            },
            content: r.content,
            raw_score: r.score,
        })
        .collect();
    crate::sort_by_score_desc(&mut hits, |x| x.raw_score);
    hits
}

/// Run a single Phase 2 vector-or-fulltext source.
///
/// Returns `Vec::new()` on internal failure so the calling fan-out can
/// tolerate per-source errors without poisoning the whole phase.  This
/// is the parallel-callable building block fed into
/// `futures::future::join_all` inside [`phase2_expand`].
#[allow(clippy::too_many_arguments)]
async fn run_phase2_source(
    kb: &KnowledgeBase,
    label: &str,
    mode: &str,
    content_field: &str,
    top_k: i64,
    qvec: &[f32],
    qtxt: String,
    vector_field: &str,
    allow: Option<&[NodeId]>,
) -> Vec<RankedHit> {
    let start = std::time::Instant::now();

    let rows = match mode {
        "vector" => {
            kb.recall_vector_search(label, vector_field, content_field, qvec, top_k, allow)
                .await
        }
        "fulltext" => {
            kb.recall_fulltext_search(label, content_field, qtxt.as_str(), top_k, allow)
                .await
        }
        _ => return Vec::new(),
    };

    let hits = match rows {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(label, mode, error = %e, "phase2 query failed");
            return Vec::new();
        }
    };

    let mut ranked: Vec<RankedHit> = Vec::with_capacity(hits.len());
    for row in hits {
        if row.content.is_empty() {
            continue;
        }
        // The query always projects `labels(m)[0]`; fall back to the
        // requested label only on the (defensive) empty case.
        let lbl = if row.label.is_empty() {
            label.to_string()
        } else {
            row.label
        };
        ranked.push(RankedHit {
            node_id: row.node_id,
            label: lbl,
            content: row.content,
            raw_score: row.score,
        });
    }
    normalize_scores_in_place(&mut ranked);
    eprintln!(
        "RECALL_PROF phase2_source label={label} mode={mode} hits={} elapsed_ms={}",
        ranked.len(),
        start.elapsed().as_millis() as u64
    );
    ranked
}

/// Phase 2 temporal-interval channel.
///
/// When the query has a parsed `[lo, hi)` temporal window
/// (`IntentProfile.temporal_window`), fans out three queries in
/// parallel:
///
/// - **Fact**: BTIC overlap on `valid_at` via the `btic_overlaps()`
///   Cypher UDF.  Score: flat `1.0` (overlap is binary).
/// - **Observation**: BTree range scan on `temporal_anchor`.
///   Score: `1.0 / (1.0 + days_from_window_center)` to favour hits
///   closer to the middle of the window.
/// - **Episode**: BTree range scan on `timestamp`.  Same proximity
///   scoring as Observation.
///
/// Returns hits in a single ranked list — RRF treats them as one
/// channel.  Empty Vec when the window is None or all queries fail.
async fn phase2_temporal(
    kb: &KnowledgeBase,
    intent: &IntentProfile,
    config: &RecallConfig,
) -> Vec<RankedHit> {
    let Some((lo, hi)) = intent.temporal_window else {
        return Vec::new();
    };
    let start = std::time::Instant::now();

    // Per-arm budget 20/20/10 (Fact/Observation/Episode). Flat score 1.0
    // for every hit — the window is the discriminator, not proximity-
    // within-window. RRF rank order is what matters downstream, and a
    // narrow query window already restricts the candidate set tightly.
    let mut ranked: Vec<RankedHit> = Vec::new();
    match kb
        .temporal_window_hits(lo, hi, 20, 20, 10, config.dimensions_allow.as_deref())
        .await
    {
        Ok(rows) => {
            for row in rows {
                if row.content.trim().is_empty() {
                    continue;
                }
                ranked.push(RankedHit {
                    node_id: row.node_id,
                    label: row.label,
                    content: row.content,
                    raw_score: 1.0,
                });
            }
        }
        Err(e) => {
            tracing::debug!(error = %e, "phase2_temporal unified query failed");
        }
    }

    normalize_scores_in_place(&mut ranked);
    tracing::debug!(
        hits = ranked.len(),
        elapsed_ms = start.elapsed().as_millis() as u64,
        lo = %lo,
        hi = %hi,
        "phase2_temporal complete",
    );
    ranked
}

/// Phase 2 graph spreading-activation channel.
///
/// Resolves the query's `entity_refs` to graph seed NodeIds, runs
/// edge-weight-aware personalized PageRank from those seeds, then
/// converts the top activated nodes into `RankedHit`s.  Edge-type
/// weights bias propagation toward semantic relations (ABOUT,
/// MENTIONS, SUPPORTED_BY) and away from structural ones (IN_SESSION,
/// FOLLOWED_BY).  See [`default_phase2_graph_edge_weights`].
///
/// Excludes the seed nodes themselves from output — they already
/// dominate by construction and the recall bundle gains nothing from
/// echoing the query's entity names back.
///
/// Returns empty Vec when seed resolution yields nothing, PPR fails,
/// or no activated nodes have non-empty content.
async fn phase2_graph_activation(
    kb: &KnowledgeBase,
    intent: &IntentProfile,
    config: &RecallConfig,
) -> Vec<RankedHit> {
    let start = std::time::Instant::now();
    let seeds = intent.resolve_seeds(kb).await;
    if seeds.is_empty() {
        return Vec::new();
    }

    let top_k = (config.per_variant_limit.max(15)).saturating_mul(2);
    let ppr_result = match kb
        .personalized_pagerank_weighted(
            &seeds,
            config.phase2_graph_damping,
            config.phase2_graph_max_iter,
            top_k,
            Some(&config.phase2_graph_edge_weights),
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(error = %e, "phase2_graph_activation PPR failed");
            return Vec::new();
        }
    };

    if ppr_result.scores.is_empty() {
        return Vec::new();
    }

    // Drop seed nodes from output and collect the rest.  We need their
    // label + a content string suitable for downstream RRF / rerank.
    let seed_set: std::collections::HashSet<NodeId> = seeds.iter().copied().collect();
    let vids: Vec<NodeId> = ppr_result
        .scores
        .iter()
        .filter(|(nid, _)| !seed_set.contains(nid))
        .map(|(nid, _)| *nid)
        .collect();
    if vids.is_empty() {
        return Vec::new();
    }
    let score_by_nid: HashMap<NodeId, f64> = ppr_result.scores.into_iter().collect();

    // Pull labels + a best-effort content field for each activated node
    // in a single round-trip.
    let r = match kb.fetch_node_contents(&vids).await {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(error = %e, "phase2_graph_activation node fetch failed");
            return Vec::new();
        }
    };

    let mut ranked: Vec<RankedHit> = Vec::with_capacity(r.len());
    for row in r {
        if row.content.trim().is_empty() {
            continue;
        }
        // Skip entities themselves — they're recall anchors, not items
        // we want to show in the bundle.  Same exclusion downstream
        // code in Phase 3 entity fan-out applies.
        if row.label == "Entity" || row.label == "Participant" {
            continue;
        }
        let score = score_by_nid.get(&row.node_id).copied().unwrap_or(0.0);
        ranked.push(RankedHit {
            node_id: row.node_id,
            label: row.label,
            content: row.content,
            raw_score: score,
        });
    }
    // PPR's natural ordering is already top-down by score, but the
    // UNWIND/MATCH round-trip doesn't preserve order — re-sort here.
    crate::sort_by_score_desc(&mut ranked, |x| x.raw_score);
    normalize_scores_in_place(&mut ranked);
    tracing::debug!(
        seeds = seeds.len(),
        hits = ranked.len(),
        iterations = ppr_result.iterations,
        converged = ppr_result.converged,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "phase2_graph_activation complete",
    );
    ranked
}

/// Min-max normalise a `RankedHit` list's `raw_score` field in-place
/// to `[0,1]`.  No-op when all scores are equal or the list is empty.
fn normalize_scores_in_place(hits: &mut [RankedHit]) {
    if hits.is_empty() {
        return;
    }
    let (min, max) = hits
        .iter()
        .map(|h| h.raw_score)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s), hi.max(s))
        });
    let range = max - min;
    if range <= f64::EPSILON {
        return;
    }
    for h in hits {
        h.raw_score = (h.raw_score - min) / range;
    }
}

/// Phase C session-boost: walk each Phase 1 Fact to its containing
/// session chunks and aggregate boost signals per chunk node id.
///
/// Edge walk: `Fact -[:SUPPORTED_BY]-> Observation -[:OBSERVED_IN]->
/// Message -[:IN_SESSION]-> Session -[:HAS_CHUNK]-> Chunk`
///
/// The final hop needs the session-level chunks that
/// [`Session::finalize`](crate::Session::finalize) builds; a session that
/// was never finalized contributes no boost.
///
/// Returns a map from chunk node id → score delta (`alpha · fact_score`,
/// summed across all Facts whose evidence touches that session).  Empty
/// when no Facts or when the walks return nothing.
async fn session_boost_signals(
    kb: &KnowledgeBase,
    phase1_items: &[RecallItem],
    alpha: f64,
) -> HashMap<NodeId, f64> {
    let mut boosts: HashMap<NodeId, f64> = HashMap::new();
    if phase1_items.is_empty() || alpha <= 0.0 {
        return boosts;
    }

    for fact in phase1_items {
        if !matches!(
            fact.kind.tier(),
            RecallTier::Semantic | RecallTier::Procedural
        ) {
            continue;
        }
        // One round-trip per Fact.  Cheap relative to the LLM call that
        // follows; could be batched if profiling shows it.
        let chunk_ids = match kb.fact_session_chunk_ids(fact.node_id).await {
            Ok(ids) => ids,
            Err(e) => {
                tracing::debug!(error = %e, fact_id = fact.node_id, "session_boost walk failed");
                continue;
            }
        };
        let delta = fact.score * alpha;
        for cid in chunk_ids {
            *boosts.entry(cid).or_insert(0.0) += delta;
        }
    }
    boosts
}

/// Spec §IX coverage formula:
/// `0.4 · facet_coverage + 0.3 · mean_score + 0.3 · diversity`.
///
/// `facet_coverage` is the share of the intent's named entities that
/// appear in any retrieved Semantic/Procedural item via the
/// (subject|predicate|content) text — proxied here as
/// `min(items_in_top_tiers, facet_count) / facet_count` to avoid a
/// per-item Cypher round-trip.  `diversity` counts distinct tiers in
/// the bundle (max 5).
fn compute_coverage(items: &[RecallItem], facet_count: usize) -> f64 {
    if items.is_empty() {
        return 0.0;
    }
    let mean_score = items.iter().map(|i| i.score).sum::<f64>() / items.len() as f64;
    let distinct_tiers = items
        .iter()
        .map(|i| std::mem::discriminant(&i.kind.tier()))
        .collect::<std::collections::HashSet<_>>()
        .len();
    let diversity = (distinct_tiers as f64) / 5.0;
    let semantic_or_procedural = items
        .iter()
        .filter(|i| matches!(i.kind.tier(), RecallTier::Semantic | RecallTier::Procedural))
        .count();
    let facets = facet_count.max(1) as f64;
    let facet_coverage = (semantic_or_procedural as f64).min(facets) / facets;
    0.4 * facet_coverage + 0.3 * mean_score + 0.3 * diversity
}

/// Does any Entity reachable from `node_id` (via the relationships
/// the recall path actually walks — `MENTIONS` from Message, `ABOUT`
/// from Observation/Chunk) carry `entity_type = target_type`? Also
/// matches when the node *is* an Entity of that type.
///
/// Single Cypher query per call; intended to be invoked on at most
/// `answer_type_top_n` items (default 50). Returns `false` on any
/// query error so a transient failure can't accidentally re-rank
/// items downward.
async fn entity_type_match(
    kb: &uniko_store::KnowledgeBase,
    node_id: uniko_store::NodeId,
    target_type: &str,
) -> bool {
    // The store binds both the node id and the target type as parameters
    // (issue #2). Treat a query error as "no match" so a transient
    // failure can't accidentally re-rank items downward.
    match kb.entity_type_matches(node_id, target_type).await {
        Ok(hit) => hit,
        Err(e) => {
            tracing::debug!(error = %e, node_id, "entity_type_match: query failed");
            false
        }
    }
}

/// One scored hit returned by [`run_recall_for_variant`]. The order
/// of the returned `Vec<RankedHit>` is `score`-descending; downstream
/// RRF fusion uses the *index* (= rank) rather than `raw_score`.
#[derive(Debug, Clone)]
struct RankedHit {
    node_id: NodeId,
    label: String,
    content: String,
    raw_score: f64,
}

/// Run the full hybrid + entity-scoped recall pipeline for a single
/// query variant. Returns a deduped, score-descending list ready for
/// RRF fusion against the other variants' lists.
///
/// Per-node deduplication uses `score.max()` *within* a single
/// variant's queries (same semantics as the legacy single-query code).
/// Cross-variant fusion is done by the caller via reciprocal rank.
async fn run_recall_for_variant(
    kb: &KnowledgeBase,
    qvec: &[f32],
    qtxt: &str,
    entity_refs: &[String],
    config: &RecallConfig,
) -> Vec<RankedHit> {
    // Hybrid chunk search (vector+BM25 over session/observation chunks)
    // plus the entity-scoped fan-out, max-merged per node id by the store
    // (which binds every value and interpolates only structural lists).
    let scored = kb
        .recall_chunk_and_entity_scoped(
            qvec,
            qtxt,
            entity_refs,
            config.per_variant_limit as i64,
            config.vector_weight,
            config.bm25_weight,
            config.dimensions_allow.as_deref(),
        )
        .await;

    let mut hits: Vec<RankedHit> = scored
        .into_iter()
        .filter(|r| !r.content.is_empty())
        .map(|r| RankedHit {
            node_id: r.node_id,
            label: r.label,
            content: r.content,
            raw_score: r.score,
        })
        .collect();
    crate::sort_by_score_desc(&mut hits, |x| x.raw_score);
    hits
}

/// Reciprocal-rank fusion across a set of per-variant ranked lists.
///
/// Each variant contributes `1 / (k + rank)` to a per-node accumulator.
/// Nodes appearing across multiple variants accumulate score; no
/// single variant dominates because the contribution is rank-based,
/// not raw-score based. The first variant that mentions a node
/// supplies the `(label, content)` returned to the caller.
fn rrf_fuse<'a, I>(per_variant: I, k: f64) -> HashMap<NodeId, (String, String, f64)>
where
    I: IntoIterator<Item = &'a [RankedHit]>,
{
    let mut fused: HashMap<NodeId, (String, String, f64)> = HashMap::new();
    for ranked in per_variant {
        for (rank, hit) in ranked.iter().enumerate() {
            let contribution = 1.0 / (k + rank as f64);
            fused
                .entry(hit.node_id)
                .and_modify(|(_, _, s)| *s += contribution)
                .or_insert((hit.label.clone(), hit.content.clone(), contribution));
        }
    }
    fused
}

fn empty_bundle() -> ContextBundle {
    ContextBundle {
        items: Vec::new(),
        total_tokens: 0,
        phase1_only: false,
        phase2_only: false,
        coverage: 0.0,
    }
}

#[cfg(test)]
mod rrf_tests {
    use super::*;

    fn hit(node_id: i64, score: f64) -> RankedHit {
        RankedHit {
            node_id,
            label: "Chunk".to_string(),
            content: format!("doc {node_id}"),
            raw_score: score,
        }
    }

    fn fused_sorted(per_variant: Vec<Vec<RankedHit>>, k: f64) -> Vec<(NodeId, f64)> {
        let slices: Vec<&[RankedHit]> = per_variant.iter().map(|v| v.as_slice()).collect();
        let mut pairs: Vec<(NodeId, f64)> = rrf_fuse(slices, k)
            .into_iter()
            .map(|(nid, (_, _, s))| (nid, s))
            .collect();
        crate::sort_by_score_desc(&mut pairs, |x| x.1);
        pairs
    }

    #[test]
    fn rrf_one_variant_orders_by_input_rank() {
        // Single variant — fused order should match the input's
        // descending raw_score order (i.e. input rank).
        let v0 = vec![hit(1, 0.9), hit(2, 0.5), hit(3, 0.1)];
        let pairs = fused_sorted(vec![v0], 60.0);
        let nids: Vec<i64> = pairs.iter().map(|(n, _)| *n).collect();
        assert_eq!(nids, vec![1, 2, 3]);
    }

    #[test]
    fn rrf_fuses_three_variants() {
        // Item 1 appears in variants 0 and 2 (rank 0 in both).
        // Item 2 appears only in variant 1 (rank 0).
        // Item 1 should outrank item 2 in fused order because it
        // accumulates two contributions (1/60 + 1/60) vs one (1/60).
        let v0 = vec![hit(1, 0.9), hit(3, 0.5)];
        let v1 = vec![hit(2, 0.9), hit(4, 0.5)];
        let v2 = vec![hit(1, 0.9), hit(5, 0.5)];
        let pairs = fused_sorted(vec![v0, v1, v2], 60.0);
        let nids: Vec<i64> = pairs.iter().map(|(n, _)| *n).collect();
        assert_eq!(nids[0], 1, "item 1 hit by 2 variants must rank first");
        // The remaining four items each got one rank-0 or rank-1
        // contribution; their order between them doesn't matter for
        // this assertion.
    }

    #[test]
    fn rrf_robust_to_noisy_variant() {
        // Two variants put gold (id=42) at ranks 0, 1.
        // A third variant returns junk and gold is absent there.
        // Gold must still win because its accumulated 1/60 + 1/61
        // exceeds any junk item's 1/60 from one variant.
        let v0 = vec![hit(42, 1.0), hit(7, 0.5)];
        let v1 = vec![hit(8, 1.0), hit(42, 0.6)];
        let v2 = vec![hit(99, 1.0), hit(98, 0.5)];
        let pairs = fused_sorted(vec![v0, v1, v2], 60.0);
        assert_eq!(
            pairs[0].0, 42,
            "gold should rank first across noisy variants"
        );
    }

    #[test]
    fn rrf_empty_input_returns_empty_map() {
        let empty: Vec<Vec<RankedHit>> = vec![];
        let pairs = fused_sorted(empty, 60.0);
        assert!(pairs.is_empty());
    }
}
