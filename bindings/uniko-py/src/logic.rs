// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The [`PyAssumeBuilder`] — hypothetical (`ASSUME`) reasoning.
//!
//! The facade's `AssumeBuilder<'_>` borrows the store only inside `run().await`,
//! so this Python builder accumulates the assume block, query, and params, and
//! does the borrow + execute in a single async block at the terminal `run()`.

use std::collections::HashMap;

use pyo3::prelude::*;
use uniko_api::tools::{Agent, Value};

use crate::convert::{py_to_value, records_to_py};
use crate::errors::to_pyerr;
use crate::macros::pymethods_with_sync;

/// The mutable, accumulated state of an in-progress assume query.
struct AssumeState {
    query: Option<String>,
    params: HashMap<String, Value>,
}

/// A hypothetical-reasoning query builder: fork state, mutate, query, roll back.
///
/// Obtained from [`Agent.assume`](crate::agent::PyAgent). Set the query with
/// `then_query`, inject params with `param`, then execute with `run`. The real
/// graph is never modified.
#[pyclass(name = "AssumeBuilder", module = "uniko")]
pub struct PyAssumeBuilder {
    agent: Agent,
    assume_block: String,
    state: std::sync::Mutex<AssumeState>,
}

impl PyAssumeBuilder {
    /// Begin a builder over `agent` with the given assume block.
    pub(crate) fn new(agent: Agent, assume_block: String) -> Self {
        Self {
            agent,
            assume_block,
            state: std::sync::Mutex::new(AssumeState {
                query: None,
                params: HashMap::new(),
            }),
        }
    }
}

pymethods_with_sync! {
    PyAssumeBuilder;
    plain {
        /// Set the query to run against the hypothetical state.
        fn then_query<'py>(slf: PyRef<'py, Self>, query: String) -> PyRef<'py, Self> {
            slf.state.lock().expect("Assume mutex poisoned").query = Some(query);
            slf
        }

        /// Inject a parameter into the Locy program.
        fn param<'py>(
            slf: PyRef<'py, Self>,
            key: String,
            value: &Bound<'_, PyAny>,
        ) -> PyResult<PyRef<'py, Self>> {
            let value = py_to_value(value)?;
            slf.state
                .lock()
                .expect("Assume mutex poisoned")
                .params
                .insert(key, value);
            Ok(slf)
        }

        fn __repr__(&self) -> String {
            "AssumeBuilder(...)".to_string()
        }

    }
    bridged {
        /// Execute: fork state, apply mutations, query, roll back. Returns the
        /// query rows as `list[dict]`.
        fn run / run_sync(&self, py) -> Py<PyAny> {
            prepare {
                let agent = self.agent.clone();
                let block = self.assume_block.clone();
                let (query, params) = {
                    let state = self.state.lock().expect("Assume mutex poisoned");
                    (state.query.clone(), state.params.clone())
                };

            }
            run(agent = agent) {
                let mut builder = agent.assume(&block);
                if let Some(q) = &query {
                    builder = builder.then_query(q);
                }
                for (k, v) in params {
                    builder = builder.param(k, v);
                }
                let rows = builder.run().await.map_err(to_pyerr)?;
                Python::attach(|py| records_to_py(py, &rows))

            }
        }

    }
}
