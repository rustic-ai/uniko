//! Hypothetical reasoning via the ASSUME builder.
//!
//! Forks graph state, applies mutations, runs a query, then rolls back
//! — the original graph is never modified.

use std::collections::HashMap;

use uni_db::Value;
use uni_db::locy::CommandResult;

use crate::error::{Result, UnikoError};
use crate::locy::Record;
use crate::storage::KnowledgeBase;

/// Builder for hypothetical reasoning queries.
///
/// # Examples
///
/// ```no_run
/// # async fn example(kb: &uniko_store::storage::KnowledgeBase) -> uniko_store::Result<()> {
/// let results = kb.assume("ASSUME { CREATE (:Fact {subject: 'server', predicate: 'port', object: '9090'}) }")
///     .then_query("MATCH (f:Fact {subject: 'server'}) RETURN f")
///     .run()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct AssumeBuilder<'a> {
    kb: &'a KnowledgeBase,
    assume_block: String,
    query: Option<String>,
    params: HashMap<String, Value>,
}

impl<'a> AssumeBuilder<'a> {
    pub(crate) fn new(kb: &'a KnowledgeBase, assume_block: &str) -> Self {
        Self {
            kb,
            assume_block: assume_block.to_string(),
            query: None,
            params: HashMap::new(),
        }
    }

    /// Set the query to execute against the hypothetical state.
    pub fn then_query(mut self, query: &str) -> Self {
        self.query = Some(query.to_string());
        self
    }

    /// Inject a parameter into the Locy program.
    pub fn param(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }

    /// Execute: fork state, apply mutations, query, rollback.
    ///
    /// Target: < 200 ms (NF9).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Locy`] on evaluation failure.
    pub async fn run(self) -> Result<Vec<Record>> {
        // Locy grammar is `ASSUME { mutations } THEN <body>` where the body is a
        // braced block (e.g. `THEN { MATCH (n) RETURN n }`). Splice the query in
        // as that body, wrapping it in braces unless the caller already did.
        let program = if let Some(ref q) = self.query {
            let body = q.trim();
            if body.starts_with('{') {
                format!("{} THEN {}", self.assume_block, body)
            } else {
                format!("{} THEN {{ {body} }}", self.assume_block)
            }
        } else {
            self.assume_block.clone()
        };

        let session = self.kb.db.session();
        let mut builder = session.locy_with(&program);
        for (k, v) in &self.params {
            builder = builder.param(k.as_str(), v.clone());
        }
        let result = builder
            .run()
            .await
            .map_err(|e| UnikoError::Locy(e.to_string()))?;

        // The `THEN` body's rows surface under a `CommandResult::Assume`, not
        // via `LocyResult::rows()` (which finds only the `Query` goal command).
        // Pull the assume (or plain query) command's rows directly.
        //
        // `CommandResult::Assume` carries one result per body command since
        // uni-db 5.0 (it was the body's rows itself before), so the body's rows
        // sit one level down. `CommandResult::rows()` is upstream's depth-first
        // search for them, and it is what the body needs: a spliced `THEN`
        // block reports as a nested `Cypher` result, which this used to miss.
        let rows = result
            .command_results()
            .iter()
            .find_map(CommandResult::rows)
            .cloned()
            .unwrap_or_default();
        Ok(rows)
    }
}

impl KnowledgeBase {
    /// Begin building an ASSUME (hypothetical reasoning) query.
    ///
    /// Target: < 200 ms (NF9).
    pub fn assume(&self, assume_block: &str) -> AssumeBuilder<'_> {
        AssumeBuilder::new(self, assume_block)
    }
}
