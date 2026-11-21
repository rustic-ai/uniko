//! What a store was built with, and whether a config can open it unchanged.
//!
//! uni-db keeps every vector column's type and every vector index's config
//! in its schema catalog; uniko records the embedding model on the
//! `KnowledgeBaseStats` row. [`inspect_db`] reads both **without applying a
//! schema**, and [`check`] compares them with a config. Opening a store runs
//! the same check before the schema is applied, so an incompatible config is
//! refused before anything on disk changes or any model loads.
//!
//! A difference is either:
//!
//! - **incompatible**: the stored vectors can't serve the config (another
//!   model, other dimensions, a hybrid store opened dense-only). uni-db
//!   can't change a column's type, so the fix is a fresh store; or
//! - **a migration**: the store can be brought in line in place with
//!   [`KnowledgeBase::migrate`](crate::KnowledgeBase::migrate).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uni_db::common::core::schema::{DistanceMetric, IndexDefinition, VectorIndexType};
use uni_db::common::vector_index_opts::{AUTO_SUB_VECTORS, default_sub_vectors};
use uni_db::{DataType, Uni, Value};

use crate::config::{UnikoConfig, VectorAlgorithm, VectorMetricChoice};
use crate::schema::constants::labels;

/// The label whose columns and indexes stand for the whole store: every
/// dense column has the configured dimensions, and Chunk is one of the two
/// labels carrying the hybrid columns.
const PROBE_LABEL: &str = labels::CHUNK;

/// The marker `phase` written once a migration's schema step (and any index
/// rebuild) is done, so only its backfill is left.
pub(crate) const MARKER_BACKFILL: &str = "backfill";
/// The marker `phase` written before a migration's schema step.
pub(crate) const MARKER_SCHEMA: &str = "schema";

/// An in-place change [`KnowledgeBase::migrate`](crate::KnowledgeBase::migrate)
/// can make.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Migration {
    /// Add the sparse and ColBERT columns to a store built with the same
    /// model and dimensions, and fill them for existing chunks and
    /// observations. Also applies any index change.
    EnableHybrid,
    /// Rebuild the vector indexes with a new algorithm, metric, embed
    /// prefix or batch size. No stored vector is rewritten.
    RebuildVectorIndexes,
}

impl Migration {
    /// The stable lowercase name: `enable_hybrid` or `rebuild_vector_indexes`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EnableHybrid => "enable_hybrid",
            Self::RebuildVectorIndexes => "rebuild_vector_indexes",
        }
    }

    /// Whether running `self` also takes care of a difference that needs
    /// `other`: enabling hybrid applies the whole schema, index changes
    /// included.
    #[must_use]
    pub fn covers(self, other: Migration) -> bool {
        self == other || (self == Self::EnableHybrid && other == Self::RebuildVectorIndexes)
    }
}

impl std::str::FromStr for Migration {
    type Err = crate::UnikoError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "enable_hybrid" => Ok(Self::EnableHybrid),
            "rebuild_vector_indexes" => Ok(Self::RebuildVectorIndexes),
            other => Err(crate::UnikoError::Config(format!(
                "unknown migration '{other}'; expected enable_hybrid or rebuild_vector_indexes"
            ))),
        }
    }
}

impl std::fmt::Display for Migration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a difference between a store and a config can be resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "migration")]
#[non_exhaustive]
pub enum Verdict {
    /// The stored vectors can't serve this config: use a fresh store.
    Incompatible,
    /// Run this migration first.
    NeedsMigration(Migration),
}

/// One way a store differs from a config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StoreDifference {
    /// What differs, e.g. `dimensions`, `model_id`, `vector_metric`.
    pub field: String,
    /// What the store has.
    pub stored: String,
    /// What the config asks for.
    pub requested: String,
    /// How to resolve it.
    pub verdict: Verdict,
}

impl std::fmt::Display for StoreDifference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fix = match self.verdict {
            Verdict::Incompatible => {
                "stored vectors can't serve it; open with the original config or use a fresh store"
                    .to_string()
            }
            Verdict::NeedsMigration(m) => format!("run the {m} migration first"),
        };
        write!(
            f,
            "{}: store has {}, config asks for {} ({fix})",
            self.field, self.stored, self.requested
        )
    }
}

/// How a store compares with a config.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Compatibility {
    /// Every difference found; empty when the config opens the store as is.
    pub differences: Vec<StoreDifference>,
}

impl Compatibility {
    /// The config opens the store unchanged.
    #[must_use]
    pub fn is_compatible(&self) -> bool {
        self.differences.is_empty()
    }

    /// Some difference can only be fixed with a fresh store.
    #[must_use]
    pub fn has_incompatible(&self) -> bool {
        self.differences
            .iter()
            .any(|d| d.verdict == Verdict::Incompatible)
    }

    /// The migrations the differences call for. Empty when some difference
    /// needs a fresh store anyway: no migration would make it open.
    #[must_use]
    pub fn migrations(&self) -> Vec<Migration> {
        let mut out: Vec<Migration> = Vec::new();
        if self.has_incompatible() {
            return out;
        }
        for d in &self.differences {
            if let Verdict::NeedsMigration(m) = d.verdict
                && !out.contains(&m)
            {
                out.push(m);
            }
        }
        // A migration another one listed covers is redundant: `enable_hybrid`
        // applies index changes too, so it alone is what to run.
        let all = out.clone();
        out.retain(|m| !all.iter().any(|n| n != m && n.covers(*m)));
        out
    }

    /// Whether `migration` resolves every difference.
    #[must_use]
    pub fn resolved_by(&self, migration: Migration) -> bool {
        self.differences.iter().all(|d| match d.verdict {
            Verdict::Incompatible => false,
            Verdict::NeedsMigration(m) => migration.covers(m),
        })
    }
}

impl std::fmt::Display for Compatibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lines: Vec<String> = self.differences.iter().map(ToString::to_string).collect();
        f.write_str(&lines.join("; "))
    }
}

/// The embedding setup a store was built with, as stored.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StoredEmbedding {
    /// The model recorded on the stats row; `None` on a store that predates
    /// the record.
    pub model_id: Option<String>,
    /// Dense vector dimensions.
    pub dimensions: Option<usize>,
    /// Sparse vocabulary size, when the store has the sparse column.
    pub sparse_dimensions: Option<usize>,
    /// ColBERT token-vector dimensions, when the store has that column.
    pub multivector_dimensions: Option<usize>,
    /// The dense index's algorithm, e.g. `hnsw_sq(m=16, ef_construction=100)`.
    pub algorithm: Option<String>,
    /// The dense index's metric: `cosine`, `l2` or `dot`.
    pub metric: Option<String>,
    /// The model alias the dense index embeds with: `embed/default`, or
    /// `embed/hybrid` on a hybrid store.
    pub embed_alias: Option<String>,
    /// Prefix added to text before it is embedded.
    pub document_prefix: Option<String>,
    /// Prefix added to queries before they are embedded.
    pub query_prefix: Option<String>,
    /// Texts embedded per batch.
    pub batch_size: Option<usize>,
}

impl StoredEmbedding {
    /// The store has the sparse or ColBERT column.
    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        self.sparse_dimensions.is_some() || self.multivector_dimensions.is_some()
    }
}

/// What a store at a path was built with.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StoreInfo {
    /// A uniko store exists there.
    pub exists: bool,
    /// Its embedding setup.
    pub embedding: StoredEmbedding,
    /// Chunks and observations whose hybrid columns aren't filled yet, on a
    /// hybrid store.
    pub hybrid_backfill_pending: Option<u64>,
    /// A migration that started and has not finished.
    pub migration_in_progress: Option<Migration>,
    /// Whether that migration was cut off in its schema step (index builds
    /// included), as opposed to its backfill. A marker with no phase, from
    /// a run that set it before the phase was recorded, counts as the schema
    /// step: the index state is unknown.
    #[serde(skip)]
    pub(crate) migration_in_schema: bool,
}

/// Read what the store behind `db` was built with, without applying a
/// schema or loading a model.
///
/// `with_counts` also counts the rows a hybrid backfill still has to fill,
/// which scans Chunk and Observation: the open-time check leaves it off,
/// [`KnowledgeBase::inspect`](crate::KnowledgeBase::inspect) and
/// `store_info` turn it on.
pub(crate) async fn inspect_db(db: &Uni, with_counts: bool) -> crate::Result<StoreInfo> {
    let schema = db.schema().current();
    let Some(props) = schema.properties.get(PROBE_LABEL) else {
        return Ok(StoreInfo::default());
    };
    let mut embedding = StoredEmbedding {
        dimensions: match props.get("embedding").map(|p| &p.r#type) {
            Some(DataType::Vector { dimensions }) => Some(*dimensions),
            _ => None,
        },
        sparse_dimensions: match props.get("sparse_embedding").map(|p| &p.r#type) {
            Some(DataType::SparseVector { dimensions }) => Some(*dimensions),
            _ => None,
        },
        multivector_dimensions: match props.get("colbert_embedding").map(|p| &p.r#type) {
            Some(DataType::List(inner)) => match inner.as_ref() {
                DataType::Vector { dimensions } => Some(*dimensions),
                _ => None,
            },
            _ => None,
        },
        ..StoredEmbedding::default()
    };
    if let Some(IndexDefinition::Vector(ix)) = schema
        .indexes
        .iter()
        .find(|i| matches!(i, IndexDefinition::Vector(v) if v.label == PROBE_LABEL && v.property == "embedding"))
    {
        embedding.algorithm = Some(describe_algorithm(&ix.index_type));
        embedding.metric = Some(describe_metric(&ix.metric));
        if let Some(e) = &ix.embedding_config {
            embedding.embed_alias = Some(e.alias.clone());
            embedding.document_prefix.clone_from(&e.document_prefix);
            embedding.query_prefix.clone_from(&e.query_prefix);
            embedding.batch_size = Some(e.batch_size);
        }
    }

    let mut info = StoreInfo {
        exists: true,
        embedding,
        ..StoreInfo::default()
    };
    if schema.labels.contains_key(labels::KNOWLEDGE_BASE_STATS) {
        let rows = db
            .session()
            .query("MATCH (s:KnowledgeBaseStats) RETURN s.embedding AS embedding, s.migration AS migration")
            .await?;
        if let Some(row) = rows.rows().first() {
            if let Some(Value::Map(m)) = row.value("embedding")
                && let Some(Value::String(model)) = m.get("model_id")
            {
                info.embedding.model_id = Some(model.clone());
            }
            if let Some(Value::Map(m)) = row.value("migration")
                && let Some(Value::String(kind)) = m.get("kind")
            {
                info.migration_in_progress = kind.parse().ok();
                info.migration_in_schema = info.migration_in_progress.is_some()
                    && !matches!(m.get("phase"), Some(Value::String(p)) if p == MARKER_BACKFILL);
            }
        }
    }
    if with_counts && info.embedding.is_hybrid() {
        info.hybrid_backfill_pending = Some(
            hybrid_backfill_pending(
                db,
                info.embedding.sparse_dimensions.is_some(),
                info.embedding.multivector_dimensions.is_some(),
            )
            .await?,
        );
    }
    Ok(info)
}

/// The condition for a row of `label` whose hybrid columns aren't all
/// filled yet: it has text, and the sparse and/or ColBERT column (whichever
/// the store has) is empty. `None` when neither column exists.
pub(crate) fn pending_predicate(
    var: &str,
    text: &str,
    sparse: bool,
    colbert: bool,
) -> Option<String> {
    let missing: Vec<String> = [
        sparse.then(|| format!("{var}.sparse_embedding IS NULL")),
        colbert.then(|| format!("{var}.colbert_embedding IS NULL")),
    ]
    .into_iter()
    .flatten()
    .collect();
    (!missing.is_empty())
        .then(|| format!("{var}.{text} IS NOT NULL AND ({})", missing.join(" OR ")))
}

/// Chunks and observations with text whose hybrid columns aren't all
/// filled yet.
pub(crate) async fn hybrid_backfill_pending(
    db: &Uni,
    sparse: bool,
    colbert: bool,
) -> crate::Result<u64> {
    let mut total = 0;
    for (label, text) in HYBRID_LABELS {
        let Some(pending) = pending_predicate("n", text, sparse, colbert) else {
            continue;
        };
        let rows = db
            .session()
            .query(&format!(
                "MATCH (n:{label}) WHERE {pending} RETURN count(n) AS c"
            ))
            .await?;
        if let Some(Ok(n)) = rows.rows().first().map(|r| r.get::<i64>("c")) {
            total += u64::try_from(n).unwrap_or(0);
        }
    }
    Ok(total)
}

/// The labels carrying the hybrid columns, with the text property their
/// embeddings are computed from.
pub(crate) const HYBRID_LABELS: [(&str, &str); 2] =
    [(labels::CHUNK, "text"), (labels::OBSERVATION, "content")];

/// Compare a store with `config`. Pure: dimensions come from the config,
/// so this never needs a model.
#[must_use]
pub fn check(stored: &StoreInfo, config: &UnikoConfig) -> Compatibility {
    let mut out = Compatibility::default();
    if !stored.exists {
        return out;
    }
    let s = &stored.embedding;
    let want = &config.embedding;
    let mut push = |field: &str, stored: String, requested: String, verdict: Verdict| {
        out.differences.push(StoreDifference {
            field: field.to_string(),
            stored,
            requested,
            verdict,
        });
    };
    let show = |v: Option<usize>| v.map_or_else(|| "none".to_string(), |n| n.to_string());

    // A migration cut off in its schema step may have left uni-db with the
    // new index definitions but indexes not fully built, which would quietly
    // miss results; rerunning it rebuilds them outright. One cut off in its
    // backfill is left out: the store is correct, some rows just lack the
    // new vectors, and `hybrid_backfill_pending` says how many.
    if let Some(m) = stored.migration_in_progress
        && stored.migration_in_schema
    {
        push(
            "migration_in_progress",
            m.to_string(),
            "finished".to_string(),
            Verdict::NeedsMigration(m),
        );
    }

    let mut same_vectors = true;
    if let Some(d) = s.dimensions
        && d != want.dimensions
    {
        same_vectors = false;
        push(
            "dimensions",
            d.to_string(),
            want.dimensions.to_string(),
            Verdict::Incompatible,
        );
    }
    match &s.model_id {
        Some(m) if *m != want.model_id => {
            same_vectors = false;
            push(
                "model_id",
                m.clone(),
                want.model_id.clone(),
                Verdict::Incompatible,
            );
        }
        Some(_) => {}
        // A store from before the model was recorded: its shape may match,
        // but nothing proves its vectors came from this model, so hybrid
        // columns can't be added over them. Reopening it once with its
        // original config records the model.
        None => {
            let adds_hybrid = (s.sparse_dimensions.is_none() && want.sparse_dimensions.is_some())
                || (s.multivector_dimensions.is_none() && want.multivector_dimensions.is_some());
            if adds_hybrid {
                same_vectors = false;
                push(
                    "model_id",
                    "unrecorded (reopen once with the original config to record it)".to_string(),
                    want.model_id.clone(),
                    Verdict::Incompatible,
                );
            }
        }
    }

    for (field, have, wanted) in [
        (
            "sparse_dimensions",
            s.sparse_dimensions,
            want.sparse_dimensions,
        ),
        (
            "multivector_dimensions",
            s.multivector_dimensions,
            want.multivector_dimensions,
        ),
    ] {
        let verdict = match (have, wanted) {
            (a, b) if a == b => continue,
            // Adding the hybrid columns is in place, but only over vectors
            // the hybrid model would also have produced.
            (None, Some(_)) if same_vectors => Verdict::NeedsMigration(Migration::EnableHybrid),
            _ => Verdict::Incompatible,
        };
        push(field, show(have), show(wanted), verdict);
    }

    let rebuild = Verdict::NeedsMigration(Migration::RebuildVectorIndexes);
    if let Some(a) = &s.algorithm {
        let wanted = describe_algorithm(&config.vector_algorithm.to_index_type(want.dimensions));
        if *a != wanted {
            push("vector_algorithm", a.clone(), wanted, rebuild);
        }
    }
    if let Some(m) = &s.metric {
        let wanted = describe_metric(&config.vector_metric.to_distance_metric());
        if *m != wanted {
            push("vector_metric", m.clone(), wanted, rebuild);
        }
    }
    if s.embed_alias.is_some() {
        let opt = |v: &Option<String>| v.clone().unwrap_or_else(|| "none".to_string());
        // The stored vectors were embedded with the stored prefix: a new one
        // would mix two embedding spaces, which no index rebuild undoes.
        if s.document_prefix != want.document_prefix {
            push(
                "document_prefix",
                opt(&s.document_prefix),
                opt(&want.document_prefix),
                Verdict::Incompatible,
            );
        }
        if s.query_prefix != want.query_prefix {
            push(
                "query_prefix",
                opt(&s.query_prefix),
                opt(&want.query_prefix),
                rebuild,
            );
        }
        if s.batch_size.is_some_and(|b| b != want.batch_size) {
            push(
                "batch_size",
                show(s.batch_size),
                want.batch_size.to_string(),
                rebuild,
            );
        }
    }
    out
}

fn describe_metric(m: &DistanceMetric) -> String {
    format!("{m:?}").to_lowercase()
}

fn describe_algorithm(t: &VectorIndexType) -> String {
    match t {
        VectorIndexType::HnswSq {
            m, ef_construction, ..
        } => format!("hnsw_sq(m={m}, ef_construction={ef_construction})"),
        VectorIndexType::HnswFlat {
            m, ef_construction, ..
        } => format!("hnsw_flat(m={m}, ef_construction={ef_construction})"),
        VectorIndexType::HnswPq {
            m,
            ef_construction,
            num_sub_vectors,
            ..
        } => format!(
            "hnsw_pq(m={m}, ef_construction={ef_construction}, sub_vectors={num_sub_vectors})"
        ),
        VectorIndexType::IvfSq { num_partitions } => format!("ivf_sq(partitions={num_partitions})"),
        VectorIndexType::IvfPq {
            num_partitions,
            num_sub_vectors,
            ..
        } => format!("ivf_pq(partitions={num_partitions}, sub_vectors={num_sub_vectors})"),
        VectorIndexType::IvfRq {
            num_partitions,
            num_bits,
        } => format!("ivf_rq(partitions={num_partitions}, bits={num_bits:?})"),
        VectorIndexType::Flat => "flat".to_string(),
        other => format!("{other:?}"),
    }
}

impl VectorAlgorithm {
    /// The stored index type uni-db makes from this algorithm on a column of
    /// `dimensions`, mirroring its builder conversion and the defaults its
    /// `apply` resolves (an automatic PQ `sub_vectors` of 0 becomes
    /// `default_sub_vectors(dimensions)`), so a stored index can be compared
    /// with a config.
    #[must_use]
    pub(crate) fn to_index_type(&self, dimensions: usize) -> VectorIndexType {
        let sub = |s: u32| {
            if s == AUTO_SUB_VECTORS {
                default_sub_vectors(Some(dimensions))
            } else {
                s
            }
        };
        match *self {
            Self::HnswSq { m, ef_construction } => VectorIndexType::HnswSq {
                m,
                ef_construction,
                num_partitions: None,
            },
            Self::HnswFlat { m, ef_construction } => VectorIndexType::HnswFlat {
                m,
                ef_construction,
                num_partitions: None,
            },
            Self::HnswPq {
                m,
                ef_construction,
                sub_vectors,
            } => VectorIndexType::HnswPq {
                m,
                ef_construction,
                num_sub_vectors: sub(sub_vectors),
                num_partitions: None,
            },
            Self::IvfSq { partitions } => VectorIndexType::IvfSq {
                num_partitions: partitions,
            },
            Self::IvfPq {
                partitions,
                sub_vectors,
            } => VectorIndexType::IvfPq {
                num_partitions: partitions,
                num_sub_vectors: sub(sub_vectors),
                bits_per_subvector: 8,
            },
            Self::IvfRq {
                partitions,
                num_bits,
            } => VectorIndexType::IvfRq {
                num_partitions: partitions,
                num_bits,
            },
        }
    }
}

impl VectorMetricChoice {
    /// The stored metric uni-db makes from this choice.
    #[must_use]
    pub(crate) fn to_distance_metric(self) -> DistanceMetric {
        match self {
            Self::Cosine => DistanceMetric::Cosine,
            Self::L2 => DistanceMetric::L2,
            Self::Dot => DistanceMetric::Dot,
        }
    }
}

/// The embedding identity written to the stats row: the model, its dense,
/// sparse and multi-vector dimensions, and the index algorithm and metric.
/// The prefixes and batch size are read from the stored index config
/// instead.
pub(crate) fn identity_map(config: &UnikoConfig) -> Value {
    let e = &config.embedding;
    let int = |n: usize| Value::Int(i64::try_from(n).unwrap_or(i64::MAX));
    let mut m: HashMap<String, Value> = HashMap::new();
    m.insert("model_id".into(), Value::String(e.model_id.clone()));
    m.insert("dimensions".into(), int(e.dimensions));
    m.insert(
        "sparse_dimensions".into(),
        e.sparse_dimensions.map_or(Value::Null, int),
    );
    m.insert(
        "multivector_dimensions".into(),
        e.multivector_dimensions.map_or(Value::Null, int),
    );
    m.insert(
        "algorithm".into(),
        Value::String(describe_algorithm(
            &config.vector_algorithm.to_index_type(e.dimensions),
        )),
    );
    m.insert(
        "metric".into(),
        Value::String(describe_metric(&config.vector_metric.to_distance_metric())),
    );
    Value::Map(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EmbeddingConfig;

    fn stored_for(config: &UnikoConfig) -> StoreInfo {
        let e = &config.embedding;
        StoreInfo {
            exists: true,
            embedding: StoredEmbedding {
                model_id: Some(e.model_id.clone()),
                dimensions: Some(e.dimensions),
                sparse_dimensions: e.sparse_dimensions,
                multivector_dimensions: e.multivector_dimensions,
                algorithm: Some(describe_algorithm(
                    &config.vector_algorithm.to_index_type(e.dimensions),
                )),
                metric: Some(describe_metric(&config.vector_metric.to_distance_metric())),
                embed_alias: Some(
                    if e.sparse_dimensions.is_some() || e.multivector_dimensions.is_some() {
                        "embed/hybrid"
                    } else {
                        "embed/default"
                    }
                    .to_string(),
                ),
                document_prefix: e.document_prefix.clone(),
                query_prefix: e.query_prefix.clone(),
                batch_size: Some(e.batch_size),
            },
            ..StoreInfo::default()
        }
    }

    fn bge_m3_dense_only() -> UnikoConfig {
        let mut c = UnikoConfig {
            embedding: EmbeddingConfig::bge_m3(),
            ..UnikoConfig::default()
        };
        c.embedding.sparse_dimensions = None;
        c.embedding.multivector_dimensions = None;
        c
    }

    #[test]
    fn a_store_matches_its_own_config() {
        for c in [
            UnikoConfig::default(),
            UnikoConfig {
                embedding: EmbeddingConfig::bge_m3(),
                ..UnikoConfig::default()
            },
        ] {
            assert!(check(&stored_for(&c), &c).is_compatible());
        }
        assert!(check(&StoreInfo::default(), &UnikoConfig::default()).is_compatible());
    }

    #[test]
    fn another_model_or_size_is_incompatible() {
        let small = UnikoConfig::default();
        let m3 = UnikoConfig {
            embedding: EmbeddingConfig::bge_m3(),
            ..UnikoConfig::default()
        };
        let c = check(&stored_for(&small), &m3);
        assert!(c.has_incompatible());
        let fields: Vec<&str> = c.differences.iter().map(|d| d.field.as_str()).collect();
        assert!(fields.contains(&"dimensions") && fields.contains(&"model_id"));
        // Hybrid columns over a different model can't be added in place.
        assert!(
            c.differences
                .iter()
                .filter(|d| d.field == "sparse_dimensions")
                .all(|d| d.verdict == Verdict::Incompatible)
        );
    }

    #[test]
    fn same_model_dense_to_hybrid_needs_enable_hybrid() {
        let dense = bge_m3_dense_only();
        let hybrid = UnikoConfig {
            embedding: EmbeddingConfig::bge_m3(),
            ..UnikoConfig::default()
        };
        let c = check(&stored_for(&dense), &hybrid);
        assert!(!c.has_incompatible());
        assert_eq!(c.migrations(), vec![Migration::EnableHybrid]);
        assert!(c.resolved_by(Migration::EnableHybrid));
        assert!(!c.resolved_by(Migration::RebuildVectorIndexes));

        // The reverse drops columns the store relies on.
        assert!(check(&stored_for(&hybrid), &dense).has_incompatible());
    }

    #[test]
    fn index_settings_need_a_rebuild() {
        let base = UnikoConfig::default();
        let changed = UnikoConfig {
            vector_metric: VectorMetricChoice::Dot,
            vector_algorithm: VectorAlgorithm::HnswFlat {
                m: 32,
                ef_construction: 200,
            },
            ..UnikoConfig::default()
        };
        let c = check(&stored_for(&base), &changed);
        assert_eq!(c.migrations(), vec![Migration::RebuildVectorIndexes]);
        assert!(c.resolved_by(Migration::RebuildVectorIndexes));
        assert!(
            c.resolved_by(Migration::EnableHybrid),
            "enable_hybrid covers it"
        );
        let mut prefix = base.clone();
        prefix.embedding.query_prefix = Some("query: ".into());
        assert_eq!(
            check(&stored_for(&base), &prefix).migrations(),
            vec![Migration::RebuildVectorIndexes]
        );
        // A document prefix is baked into the stored vectors.
        let mut doc = base.clone();
        doc.embedding.document_prefix = Some("passage: ".into());
        let c = check(&stored_for(&base), &doc);
        assert!(c.has_incompatible() && c.migrations().is_empty());
    }

    #[test]
    fn an_incompatible_store_suggests_no_migration() {
        let small = UnikoConfig::default();
        let m3 = UnikoConfig {
            embedding: EmbeddingConfig::bge_m3(),
            ..UnikoConfig::default()
        };
        let c = check(&stored_for(&small), &m3);
        assert!(c.has_incompatible());
        assert!(c.migrations().is_empty(), "{:?}", c.migrations());
    }

    #[test]
    fn auto_sub_vectors_matches_what_uni_db_stores() {
        for algo in [
            VectorAlgorithm::HnswPq {
                m: 16,
                ef_construction: 100,
                sub_vectors: 0,
            },
            VectorAlgorithm::IvfPq {
                partitions: 4,
                sub_vectors: 0,
            },
        ] {
            let resolved = describe_algorithm(&algo.to_index_type(384));
            assert!(resolved.contains("sub_vectors=48"), "{resolved}");
        }
    }

    #[test]
    fn an_older_store_without_a_model_record_is_checked_on_shape() {
        let base = UnikoConfig::default();
        let mut stored = stored_for(&base);
        stored.embedding.model_id = None;
        assert!(check(&stored, &base).is_compatible());

        // But hybrid columns can't be added until the model is recorded.
        let dense = bge_m3_dense_only();
        let mut stored = stored_for(&dense);
        stored.embedding.model_id = None;
        let hybrid = UnikoConfig {
            embedding: EmbeddingConfig::bge_m3(),
            ..UnikoConfig::default()
        };
        let c = check(&stored, &hybrid);
        assert!(c.has_incompatible());
        assert!(c.differences.iter().any(|d| d.field == "model_id"));
    }

    #[test]
    fn pending_covers_whichever_hybrid_columns_exist() {
        assert_eq!(pending_predicate("n", "text", false, false), None);
        let both = pending_predicate("n", "text", true, true).unwrap();
        assert!(both.contains("sparse_embedding IS NULL OR n.colbert_embedding IS NULL"));
        let colbert = pending_predicate("n", "text", false, true).unwrap();
        assert!(colbert.contains("colbert_embedding") && !colbert.contains("sparse"));
    }

    #[test]
    fn migration_names_round_trip() {
        for m in [Migration::EnableHybrid, Migration::RebuildVectorIndexes] {
            assert_eq!(m.as_str().parse::<Migration>().unwrap(), m);
        }
        assert!("reindex".parse::<Migration>().is_err());
    }
}
