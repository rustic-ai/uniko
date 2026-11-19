//! Store compatibility and in-place migrations (#66).

use std::panic::AssertUnwindSafe;

use pyo3::exceptions::PySystemExit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use uniko_api::tools::{
    Compatibility, MigrationReport, StoreDifference, StoreInfo, StoreReport, StoredEmbedding,
    Verdict,
};
use uniko_api::tools::{UnikoConfig, UnikoError as CoreError};

use crate::config::{config_from_py, embedding_from_py};

/// The config to check or migrate against: `config` (a partial config dict)
/// over the defaults, with `embedding` (a preset name or a partial
/// embedding section) applied on top.
pub(crate) fn target_config(
    config: Option<&Bound<'_, PyAny>>,
    embedding: Option<&Bound<'_, PyAny>>,
) -> PyResult<UnikoConfig> {
    let mut c = match config {
        Some(obj) => config_from_py(obj)?,
        None => UnikoConfig::default(),
    };
    if let Some(e) = embedding {
        c.embedding = embedding_from_py(e)?;
    }
    Ok(c)
}

/// The embedding setup a store was built with.
#[pyclass(name = "StoredEmbedding", module = "uniko", frozen)]
pub struct PyStoredEmbedding {
    /// The recorded model, or ``None`` on a store from before it was kept.
    #[pyo3(get)]
    model_id: Option<String>,
    /// Dense vector dimensions.
    #[pyo3(get)]
    dimensions: Option<usize>,
    /// Sparse vocabulary size, on a hybrid store.
    #[pyo3(get)]
    sparse_dimensions: Option<usize>,
    /// ColBERT token-vector dimensions, on a hybrid store.
    #[pyo3(get)]
    multivector_dimensions: Option<usize>,
    /// The dense index's algorithm, e.g. ``"hnsw_sq(m=16, ef_construction=100)"``.
    #[pyo3(get)]
    algorithm: Option<String>,
    /// ``"cosine"``, ``"l2"`` or ``"dot"``.
    #[pyo3(get)]
    metric: Option<String>,
    /// ``"embed/default"``, or ``"embed/hybrid"`` on a hybrid store.
    #[pyo3(get)]
    embed_alias: Option<String>,
    /// Prefix added to text before embedding.
    #[pyo3(get)]
    document_prefix: Option<String>,
    /// Prefix added to queries before embedding.
    #[pyo3(get)]
    query_prefix: Option<String>,
    /// Texts embedded per batch.
    #[pyo3(get)]
    batch_size: Option<usize>,
}

impl PyStoredEmbedding {
    fn from_rust(py: Python<'_>, e: &StoredEmbedding) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                model_id: e.model_id.clone(),
                dimensions: e.dimensions,
                sparse_dimensions: e.sparse_dimensions,
                multivector_dimensions: e.multivector_dimensions,
                algorithm: e.algorithm.clone(),
                metric: e.metric.clone(),
                embed_alias: e.embed_alias.clone(),
                document_prefix: e.document_prefix.clone(),
                query_prefix: e.query_prefix.clone(),
                batch_size: e.batch_size,
            },
        )
    }
}

#[pymethods]
impl PyStoredEmbedding {
    /// The store has the sparse or ColBERT column.
    #[getter]
    fn is_hybrid(&self) -> bool {
        self.sparse_dimensions.is_some() || self.multivector_dimensions.is_some()
    }

    fn __repr__(&self) -> String {
        format!(
            "StoredEmbedding(model_id={:?}, dimensions={:?}, hybrid={})",
            self.model_id,
            self.dimensions,
            self.is_hybrid()
        )
    }
}

/// What a store was built with.
#[pyclass(name = "StoreInfo", module = "uniko", frozen)]
pub struct PyStoreInfo {
    /// A uniko store exists at the path.
    #[pyo3(get)]
    exists: bool,
    /// Its embedding setup.
    #[pyo3(get)]
    embedding: Py<PyStoredEmbedding>,
    /// Chunks and observations whose hybrid columns aren't filled yet.
    #[pyo3(get)]
    hybrid_backfill_pending: Option<u64>,
    /// ``"enable_hybrid"`` or ``"rebuild_vector_indexes"`` while one is
    /// unfinished.
    #[pyo3(get)]
    migration_in_progress: Option<&'static str>,
}

impl PyStoreInfo {
    pub(crate) fn from_rust(py: Python<'_>, i: &StoreInfo) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                exists: i.exists,
                embedding: PyStoredEmbedding::from_rust(py, &i.embedding)?,
                hybrid_backfill_pending: i.hybrid_backfill_pending,
                migration_in_progress: i.migration_in_progress.map(|m| m.as_str()),
            },
        )
    }
}

#[pymethods]
impl PyStoreInfo {
    fn __repr__(&self) -> String {
        format!("StoreInfo(exists={})", self.exists)
    }
}

/// One way a store differs from a config.
#[pyclass(name = "StoreDifference", module = "uniko", frozen)]
pub struct PyStoreDifference {
    /// What differs, e.g. ``"dimensions"`` or ``"vector_metric"``.
    #[pyo3(get)]
    field: String,
    /// What the store has.
    #[pyo3(get)]
    stored: String,
    /// What the config asks for.
    #[pyo3(get)]
    requested: String,
    /// ``"incompatible"`` (use a fresh store) or ``"needs_migration"``;
    /// ``"unknown"`` for a verdict this build doesn't know.
    #[pyo3(get)]
    verdict: &'static str,
    /// The migration that fixes it, when ``verdict`` is ``"needs_migration"``.
    #[pyo3(get)]
    migration: Option<&'static str>,
}

/// How a difference's verdict reads in Python: ``"incompatible"``,
/// ``"needs_migration"`` with the migration's name, or ``"unknown"`` (no
/// migration) for a verdict this binding was built without.
fn verdict_parts(v: Verdict) -> (&'static str, Option<&'static str>) {
    match v {
        Verdict::Incompatible => ("incompatible", None),
        Verdict::NeedsMigration(m) => ("needs_migration", Some(m.as_str())),
        _ => ("unknown", None),
    }
}

/// A [`StoreDifference`] in its Python shape. The one place that shape is
/// built: [`PyStoreDifference`] and the dicts on `IncompatibleStoreError`
/// both come from here, so they can't drift apart.
struct DifferenceParts {
    field: String,
    stored: String,
    requested: String,
    verdict: &'static str,
    migration: Option<&'static str>,
}

impl DifferenceParts {
    fn new(d: &StoreDifference) -> Self {
        let (verdict, migration) = verdict_parts(d.verdict);
        Self {
            field: d.field.clone(),
            stored: d.stored.clone(),
            requested: d.requested.clone(),
            verdict,
            migration,
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("field", &self.field)?;
        d.set_item("stored", &self.stored)?;
        d.set_item("requested", &self.requested)?;
        d.set_item("verdict", self.verdict)?;
        d.set_item("migration", self.migration)?;
        Ok(d)
    }
}

impl PyStoreDifference {
    fn from_rust(py: Python<'_>, d: &StoreDifference) -> PyResult<Py<Self>> {
        let p = DifferenceParts::new(d);
        Py::new(
            py,
            Self {
                field: p.field,
                stored: p.stored,
                requested: p.requested,
                verdict: p.verdict,
                migration: p.migration,
            },
        )
    }
}

/// The differences as plain dicts, in the same shape as
/// [`PyStoreDifference`], for `IncompatibleStoreError.differences`.
pub(crate) fn difference_dicts<'py>(
    py: Python<'py>,
    c: &Compatibility,
) -> PyResult<Bound<'py, PyList>> {
    let dicts = c
        .differences
        .iter()
        .map(|d| DifferenceParts::new(d).to_dict(py))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, dicts)
}

/// The names of the migrations `c` calls for; empty when any difference
/// is incompatible.
pub(crate) fn migration_names(c: &Compatibility) -> Vec<&'static str> {
    c.migrations().into_iter().map(|m| m.as_str()).collect()
}

#[pymethods]
impl PyStoreDifference {
    fn __repr__(&self) -> String {
        format!(
            "StoreDifference(field='{}', stored='{}', requested='{}', verdict='{}')",
            self.field, self.stored, self.requested, self.verdict
        )
    }
}

fn differences(py: Python<'_>, c: &Compatibility) -> PyResult<Vec<Py<PyStoreDifference>>> {
    c.differences
        .iter()
        .map(|d| PyStoreDifference::from_rust(py, d))
        .collect()
}

/// What ``Uniko.inspect_store`` found.
#[pyclass(name = "StoreReport", module = "uniko", frozen)]
pub struct PyStoreReport {
    /// What the store was built with.
    #[pyo3(get)]
    info: Py<PyStoreInfo>,
    /// How it differs from the config; empty when it opens as is.
    #[pyo3(get)]
    differences: Vec<Py<PyStoreDifference>>,
    /// The config opens the store unchanged.
    #[pyo3(get)]
    compatible: bool,
    /// The migrations the differences call for.
    #[pyo3(get)]
    migrations: Vec<&'static str>,
}

impl PyStoreReport {
    pub(crate) fn from_rust(py: Python<'_>, r: &StoreReport) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                info: PyStoreInfo::from_rust(py, &r.info)?,
                differences: differences(py, &r.compatibility)?,
                compatible: r.compatibility.is_compatible(),
                migrations: migration_names(&r.compatibility),
            },
        )
    }
}

#[pymethods]
impl PyStoreReport {
    fn __repr__(&self) -> String {
        format!(
            "StoreReport(compatible={}, differences={})",
            self.compatible,
            self.differences.len()
        )
    }
}

/// What ``Uniko.migrate_store`` did.
#[pyclass(name = "MigrationReport", module = "uniko", frozen)]
pub struct PyMigrationReport {
    /// ``"enable_hybrid"`` or ``"rebuild_vector_indexes"``.
    #[pyo3(get)]
    migration: &'static str,
    /// Existing rows whose new columns were filled.
    #[pyo3(get)]
    rows_backfilled: u64,
    /// Rows re-embedded whose new columns still came back empty (e.g. a
    /// text with no sparse terms). They are left as they are.
    #[pyo3(get)]
    rows_unfilled: u64,
    /// The store needs nothing further for this config.
    #[pyo3(get)]
    completed: bool,
    /// Seconds it took.
    #[pyo3(get)]
    elapsed_secs: f64,
    /// ``"enable_hybrid"`` or ``"rebuild_vector_indexes"`` when a migration
    /// is still marked unfinished after the run (one reason ``completed``
    /// can be false); ``None`` otherwise.
    #[pyo3(get)]
    migration_in_progress: Option<&'static str>,
}

impl PyMigrationReport {
    pub(crate) fn from_rust(py: Python<'_>, r: &MigrationReport) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                migration: r.migration.as_str(),
                rows_backfilled: r.rows_backfilled,
                rows_unfilled: r.rows_unfilled,
                completed: r.completed,
                elapsed_secs: r.elapsed.as_secs_f64(),
                migration_in_progress: r.migration_in_progress.map(|m| m.as_str()),
            },
        )
    }
}

#[pymethods]
impl PyMigrationReport {
    fn __repr__(&self) -> String {
        format!(
            "MigrationReport(migration='{}', rows_backfilled={}, rows_unfilled={}, completed={}, migration_in_progress={})",
            self.migration,
            self.rows_backfilled,
            self.rows_unfilled,
            self.completed,
            self.migration_in_progress
                .map_or("None".into(), |m| format!("'{m}'"))
        )
    }
}

/// Call the Python `progress` callable, if any, with a dict per event.
///
/// Nothing the callback does can stop a migration half way: an exception it
/// raises (`KeyboardInterrupt` and `SystemExit` included) is printed to
/// stderr and ignored, and so is a Rust panic it sets off (e.g. calling a
/// `_sync` uniko method from inside the callback). PyO3 resumes a
/// `PanicException` as a Rust panic when it fetches the error after the
/// call (it prints the Python traceback first), so the call runs under
/// `catch_unwind` and the panic stops there instead of unwinding through
/// the migration and skipping the store's shutdown.
pub(crate) fn progress_fn(
    progress: Option<Py<PyAny>>,
) -> impl FnMut(&uniko_api::tools::MigrationProgress) + Send {
    move |p| {
        let Some(cb) = &progress else {
            return;
        };
        Python::attach(|py| {
            let call = || -> PyResult<()> {
                let d = PyDict::new(py);
                d.set_item("migration", p.migration.as_str())?;
                d.set_item("phase", &p.phase)?;
                d.set_item("label", &p.label)?;
                d.set_item("done", p.done)?;
                d.set_item("total", p.total)?;
                cb.call1(py, (d,))?;
                Ok(())
            };
            // On `Err(panic)` there is nothing left to print: PyO3 printed
            // the traceback of a `PanicException` before resuming it, and
            // the panic hook printed any other panic.
            if let Ok(Err(e)) = std::panic::catch_unwind(AssertUnwindSafe(call)) {
                print_ignored(py, &e);
            }
        });
    }
}

/// Print an exception the progress callback raised, and drop it.
///
/// `PyErr::print` goes through `PyErr_PrintEx`, which exits the process on
/// `SystemExit`, so that one is printed with `traceback.print_exception`.
fn print_ignored(py: Python<'_>, e: &PyErr) {
    if e.is_instance_of::<PySystemExit>(py) {
        let printed = py
            .import("traceback")
            .and_then(|tb| tb.call_method1("print_exception", (e.value(py),)));
        if printed.is_err() {
            eprintln!("progress callback raised SystemExit (ignored): {e}");
        }
    } else {
        e.print(py);
    }
}

/// Parse a migration name for Python callers.
pub(crate) fn migration_from_str(name: &str) -> PyResult<uniko_api::tools::Migration> {
    name.parse()
        .map_err(|e: CoreError| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStoreReport>()?;
    m.add_class::<PyStoreInfo>()?;
    m.add_class::<PyStoredEmbedding>()?;
    m.add_class::<PyStoreDifference>()?;
    m.add_class::<PyMigrationReport>()?;
    Ok(())
}
