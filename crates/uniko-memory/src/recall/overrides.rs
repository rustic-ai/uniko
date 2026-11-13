//! Per-call overrides for [`Agent::recall_with`](crate::Agent::recall_with).

use chrono::{DateTime, Utc};
use uniko_store::UnikoError;

use super::RecallConfig;

/// Recall settings to change for one call, leaving the rest of the
/// instance's configuration as it is.
///
/// Every field is optional; `None` keeps the configured value. Build one
/// with the chainable setters:
///
/// ```
/// use uniko_memory::RecallOverrides;
///
/// let overrides = RecallOverrides::default().limit(5).min_score(0.2);
/// # let _ = overrides;
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct RecallOverrides {
    /// Maximum items to return.
    pub limit: Option<usize>,
    /// Maximum total tokens across the returned items.
    pub token_budget: Option<usize>,
    /// Minimum fused score for inclusion.
    pub min_score: Option<f64>,
    /// Turn the cross-encoder reranker on or off for this call. Turning it
    /// on requires a reranker enabled at build time, since the model is
    /// only registered then.
    pub reranker: Option<bool>,
    /// Anchor relative time expressions ("yesterday") to this instant
    /// instead of now.
    pub reference_ts: Option<DateTime<Utc>>,
}

impl RecallOverrides {
    /// Set the maximum number of items.
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set the token budget.
    #[must_use]
    pub fn token_budget(mut self, token_budget: usize) -> Self {
        self.token_budget = Some(token_budget);
        self
    }

    /// Set the minimum fused score.
    #[must_use]
    pub fn min_score(mut self, min_score: f64) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Turn the reranker on or off for this call.
    #[must_use]
    pub fn reranker(mut self, on: bool) -> Self {
        self.reranker = Some(on);
        self
    }

    /// Anchor relative time expressions to `ts`.
    #[must_use]
    pub fn reference_ts(mut self, ts: DateTime<Utc>) -> Self {
        self.reference_ts = Some(ts);
        self
    }

    /// Apply these overrides to `config`.
    ///
    /// `reranker_available` says whether a reranker model was registered
    /// at build time.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] when the result is invalid: a zero
    /// limit or token budget, a non-finite score, the reranker turned on
    /// without a registered model, or a limit above the reranker's
    /// `top_n` (the same rule `UnikoConfig::validate` applies).
    pub(crate) fn apply(
        &self,
        config: &mut RecallConfig,
        reranker_available: bool,
    ) -> Result<(), UnikoError> {
        if let Some(limit) = self.limit {
            if limit == 0 {
                return Err(UnikoError::Config("recall limit must be positive".into()));
            }
            config.limit = limit;
        }
        if let Some(budget) = self.token_budget {
            if budget == 0 {
                return Err(UnikoError::Config(
                    "recall token_budget must be positive".into(),
                ));
            }
            config.token_budget = budget;
        }
        if let Some(min_score) = self.min_score {
            if !min_score.is_finite() {
                return Err(UnikoError::Config(format!(
                    "recall min_score ({min_score}) must be finite"
                )));
            }
            config.min_score = min_score;
        }
        if let Some(on) = self.reranker {
            if on && !reranker_available {
                return Err(UnikoError::Config(
                    "reranker override needs a reranker enabled at build time".into(),
                ));
            }
            config.reranker_enabled = on;
        }
        if let Some(ts) = self.reference_ts {
            config.reference_ts = Some(ts);
        }
        if config.reranker_enabled && config.reranker_top_n < config.limit {
            return Err(UnikoError::Config(format!(
                "recall limit ({}) must be <= reranker.top_n ({}) while the reranker is on",
                config.limit, config.reranker_top_n,
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniko_store::config::UnikoConfig;

    fn base() -> RecallConfig {
        RecallConfig::from_uniko_config(&UnikoConfig::default())
    }

    #[test]
    fn empty_overrides_change_nothing() {
        let mut config = base();
        let before = format!("{config:?}");
        RecallOverrides::default().apply(&mut config, true).unwrap();
        assert_eq!(format!("{config:?}"), before);
    }

    #[test]
    fn overrides_replace_the_named_fields() {
        let mut config = base();
        let ts = Utc::now();
        RecallOverrides::default()
            .limit(3)
            .token_budget(100)
            .min_score(0.5)
            .reranker(false)
            .reference_ts(ts)
            .apply(&mut config, true)
            .unwrap();
        assert_eq!(config.limit, 3);
        assert_eq!(config.token_budget, 100);
        assert!((config.min_score - 0.5).abs() < f64::EPSILON);
        assert!(!config.reranker_enabled);
        assert_eq!(config.reference_ts, Some(ts));
    }

    #[test]
    fn a_limit_above_reranker_top_n_is_rejected() {
        let mut config = base();
        config.reranker_enabled = true;
        let too_many = config.reranker_top_n + 1;
        let err = RecallOverrides::default()
            .limit(too_many)
            .apply(&mut config, true)
            .unwrap_err();
        assert!(err.to_string().contains("reranker.top_n"), "got: {err}");

        // The same limit is fine with the reranker turned off.
        let mut config = base();
        RecallOverrides::default()
            .limit(too_many)
            .reranker(false)
            .apply(&mut config, true)
            .unwrap();
        assert_eq!(config.limit, too_many);
    }

    #[test]
    fn enabling_an_unregistered_reranker_is_rejected() {
        let mut config = base();
        let err = RecallOverrides::default()
            .reranker(true)
            .apply(&mut config, false)
            .unwrap_err();
        assert!(err.to_string().contains("build time"), "got: {err}");
    }

    #[test]
    fn invalid_values_are_rejected() {
        for overrides in [
            RecallOverrides::default().limit(0),
            RecallOverrides::default().token_budget(0),
            RecallOverrides::default().min_score(f64::NAN),
        ] {
            assert!(overrides.apply(&mut base(), true).is_err(), "{overrides:?}");
        }
    }
}
