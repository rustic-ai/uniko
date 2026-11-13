// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The [`PyData`] handle — addressed (by-id) retrieval of messages, artifacts,
//! and raw artifact bytes.
//!
//! The facade's `Data<'_>` borrows the agent's store. A `'static` pyclass can't
//! hold that borrow, so `PyData` holds an Arc-clone of the `Agent` and rebuilds
//! `agent.data()` inside each async block — the borrow lives and dies within
//! one `await`.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use uniko_api::tools::Agent;

use crate::errors::to_pyerr;
use crate::macros::pymethods_with_sync;
use crate::outputs::{PyArtifactView, PyMessageView};

/// Addressed (by-id) retrieval over an agent's store.
///
/// Obtained from [`Agent.data`](crate::agent::PyAgent). All lookups resolve to
/// `None` when the id is unknown rather than raising.
#[pyclass(name = "Data", module = "uniko")]
pub struct PyData {
    inner: Agent,
}

impl PyData {
    /// Wrap an Agent clone.
    pub(crate) fn wrap(agent: Agent) -> Self {
        Self { inner: agent }
    }
}

pymethods_with_sync! {
    PyData;
    plain {
        fn __repr__(&self) -> String {
            format!("Data(agent_id='{}')", self.inner.agent_id())
        }

    }
    bridged {
        /// Fetch a message by its external id, or `None`.
        fn message / message_sync(&self, py, message_id: String) -> Py<PyAny> {
            prepare {}
            run(agent = self.inner.clone()) {
                let view = agent.data().message(&message_id).await.map_err(to_pyerr)?;
                Python::attach(|py| match view {
                    Some(v) => PyMessageView::from_rust(py, &v).map(|p| p.into_any()),
                    None => Ok(py.None()),
                })

            }
        }

        /// Fetch an artifact by its external id, or `None`.
        fn artifact / artifact_sync(&self, py, artifact_id: String) -> Py<PyAny> {
            prepare {}
            run(agent = self.inner.clone()) {
                let view = agent
                    .data()
                    .artifact(&artifact_id)
                    .await
                    .map_err(to_pyerr)?;
                Python::attach(|py| match view {
                    Some(v) => PyArtifactView::from_rust(py, &v).map(|p| p.into_any()),
                    None => Ok(py.None()),
                })

            }
        }

        /// Fetch an artifact's raw bytes from the blob store, or `None`.
        ///
        /// This is the supported path for binary payloads — `Value::Bytes` cannot
        /// be read back through a Cypher `RETURN` (a uni-db limitation).
        fn artifact_bytes / artifact_bytes_sync(&self, py, artifact_id: String) -> Py<PyAny> {
            prepare {}
            run(agent = self.inner.clone()) {
                let bytes = agent
                    .data()
                    .artifact_bytes(&artifact_id)
                    .await
                    .map_err(to_pyerr)?;
                Python::attach(|py| match bytes {
                    Some(b) => Ok(PyBytes::new(py, &b).into_any().unbind()),
                    None => Ok(py.None()),
                })

            }
        }

    }
}
