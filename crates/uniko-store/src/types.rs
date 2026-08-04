//! Shared type aliases used across all uniko layers.
//!
//! These are domain-level types. `NodeId` and `EdgeId` are `i64` aliases for
//! uniko's own tracking — conversion to/from uni-db's `Vid(u64)` / `Eid(u64)`
//! happens at the storage boundary in the `storage` module.

/// Internal node identifier (matches uni-db's sequential ID range).
pub type NodeId = i64;

/// Internal edge identifier (matches uni-db's sequential ID range).
pub type EdgeId = i64;

/// UTC timestamp used for all temporal fields.
pub type Timestamp = chrono::DateTime<chrono::Utc>;

/// Embedding vector (f32 components, dimensionality depends on model).
pub type EmbeddingVec = Vec<f32>;

/// Convert a UTC datetime to uni-db's wire-form temporal value.
///
/// This is the canonical, type-safe way to write a `DataType::DateTime`
/// property: it carries full nanosecond precision and a normalized UTC
/// offset. Prefer it over `Value::String(dt.to_rfc3339())` + uni-db's
/// write-time string→DateTime coercion (added in uni-db 2.0, issue #68),
/// which round-trips through the `datetime()` parser and is not
/// guaranteed to preserve sub-second precision.
///
/// Historically this helper was *mandatory*: pre-2.0, a String written
/// to a DateTime column committed but was silently dropped at flush
/// (row omitted from the per-label table). #68 fixed that — strings are
/// now coerced or loudly rejected — but the typed path here remains the
/// preferred one.
#[must_use]
pub fn datetime_value(dt: chrono::DateTime<chrono::Utc>) -> uni_db::Value {
    uni_db::Value::Temporal(uni_db::common::TemporalValue::DateTime {
        nanos_since_epoch: dt.timestamp_nanos_opt().unwrap_or(0),
        offset_seconds: 0,
        timezone_name: None,
    })
}

/// Epoch milliseconds for a datetime-shaped temporal value.
///
/// Stands in for `TemporalValue::epoch_millis`, which uni-db removed in
/// 3.2.0 as a dead accessor (rustic-ai/uni-db `c75b8384a`) while uniko was
/// still a live caller. The original semantics are reproduced exactly: the
/// `nanos_since_epoch` field of the two datetime variants divided by 1e6
/// (truncating, so pre-1970 instants round toward the epoch), and `None`
/// for every variant that pins no instant — a bare date, a time of day, a
/// duration, or a BTIC interval.
///
/// Lives here rather than at each call site because this crate is the sole
/// uni-db boundary: `uniko-bench` and the Python bindings decode the same
/// `Value::Temporal` and must not each re-derive the conversion.
///
/// # Examples
///
/// ```
/// use uniko_store::temporal::TemporalValue;
/// use uniko_store::temporal_epoch_millis;
///
/// let dt = TemporalValue::LocalDateTime { nanos_since_epoch: 1_500_000_000 };
/// assert_eq!(temporal_epoch_millis(&dt), Some(1_500));
///
/// // A date carries no time-of-day, so it pins no instant.
/// assert_eq!(temporal_epoch_millis(&TemporalValue::Date { days_since_epoch: 0 }), None);
/// ```
#[must_use]
pub fn temporal_epoch_millis(value: &uni_db::common::TemporalValue) -> Option<i64> {
    use uni_db::common::TemporalValue;
    match value {
        TemporalValue::DateTime {
            nanos_since_epoch, ..
        }
        | TemporalValue::LocalDateTime {
            nanos_since_epoch, ..
        } => Some(nanos_since_epoch / 1_000_000),
        _ => None,
    }
}

/// Decode a `Value::Temporal` into a UTC datetime, surfacing a precise
/// [`UnikoError::Storage`] when it does not yield a valid timestamp.
///
/// The inverse of [`datetime_value`] for the read path. Use this when an
/// absent / malformed timestamp is genuinely an error (e.g. an Episode
/// that *must* carry a write time); use [`optional_datetime_from_row`]
/// when absence is acceptable. uni-db (>= 2.0) returns DateTime properties
/// as typed `Value::Temporal`, so no RFC-3339 string branch is needed.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) if the
/// value is not a Temporal or the millis fall outside the supported range.
pub fn datetime_from_value(
    value: &uni_db::Value,
    context: &str,
) -> crate::error::Result<chrono::DateTime<chrono::Utc>> {
    use chrono::{DateTime, Utc};
    match value {
        uni_db::Value::Temporal(t) => {
            let millis = temporal_epoch_millis(t)
                .ok_or_else(|| crate::UnikoError::Storage(format!("{context} has no epoch")))?;
            DateTime::<Utc>::from_timestamp_millis(millis).ok_or_else(|| {
                crate::UnikoError::Storage(format!("epoch millis {millis} out of range"))
            })
        }
        other => Err(crate::UnikoError::Storage(format!(
            "{context} unexpected type: {other:?}"
        ))),
    }
}

/// Pull an optional UTC datetime out of a uni-db row `column`.
///
/// uni-db (>= 2.0) serialises DateTime properties as `Value::Temporal`.
/// Returns `None` when the column is missing, null, or not a valid
/// Temporal — callers treat absence as "not yet set" rather than an error.
#[must_use]
pub fn optional_datetime_from_row(
    row: &uni_db::Row,
    column: &str,
) -> Option<chrono::DateTime<chrono::Utc>> {
    use chrono::{DateTime, Utc};
    let idx = row.columns().iter().position(|c| c == column)?;
    let value = row.values().get(idx)?;
    match value {
        uni_db::Value::Temporal(t) => {
            let millis = temporal_epoch_millis(t)?;
            DateTime::<Utc>::from_timestamp_millis(millis)
        }
        _ => None,
    }
}
