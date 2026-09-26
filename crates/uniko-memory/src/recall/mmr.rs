//! Maximal Marginal Relevance deduplication for Phase 2 bundles.
//!
//! Spec §IX: applied to fused Phase 2 candidates with `lambda = 0.7`
//! (diversity weight) and a hard duplicate threshold over content
//! similarity (cosine when embeddings are available; Jaccard
//! word-overlap as fallback).  We carry token-content rather than
//! embeddings on [`RecallItem`] so this implementation uses Jaccard
//! exclusively — that's sufficient to filter the near-duplicate
//! observation/message pairs the cascade tends to surface.
//!
//! MMR objective at each selection step:
//!     `lambda * rel(i) - (1 - lambda) * max_sim(i, selected)`

use std::collections::HashSet;

use super::RecallItem;

/// Apply MMR deduplication to a score-descending candidate list.
///
/// `lambda` weights pure relevance (`1.0` keeps the original order,
/// `0.0` selects purely for diversity).  Items whose Jaccard overlap
/// with any already-selected item exceeds `duplicate_threshold` are
/// dropped entirely as hard duplicates — this models the spec's
/// `cosine > 0.85` skip rule using token overlap.
///
/// `limit` caps the number of selected items; when `None`, all items
/// are processed (still de-duplicated).
///
/// Items already arrive scored; their `score` is interpreted as the
/// relevance term.
pub fn mmr_dedup(
    candidates: Vec<RecallItem>,
    lambda: f64,
    duplicate_threshold: f64,
    limit: Option<usize>,
) -> Vec<RecallItem> {
    if candidates.is_empty() {
        return candidates;
    }
    let target = limit.unwrap_or(candidates.len()).min(candidates.len());
    let lambda = lambda.clamp(0.0, 1.0);

    let token_sets: Vec<HashSet<String>> =
        candidates.iter().map(|c| tokenize(&c.content)).collect();

    let mut remaining: Vec<usize> = (0..candidates.len()).collect();
    let mut selected_indices: Vec<usize> = Vec::with_capacity(target);

    while !remaining.is_empty() && selected_indices.len() < target {
        // Single pass: compute max-similarity once per candidate, prune
        // hard duplicates, and track the best non-duplicate MMR score.
        let mut best: Option<(usize, f64)> = None;
        let mut keep: Vec<usize> = Vec::with_capacity(remaining.len());
        for &i in &remaining {
            let max_sim = selected_indices
                .iter()
                .map(|&j| jaccard(&token_sets[i], &token_sets[j]))
                .fold(0.0_f64, f64::max);
            if max_sim > duplicate_threshold {
                continue;
            }
            let mmr = lambda * candidates[i].score - (1.0 - lambda) * max_sim;
            if best.is_none_or(|(_, s)| mmr > s) {
                best = Some((i, mmr));
            }
            keep.push(i);
        }
        remaining = keep;
        let Some((chosen, _)) = best else {
            // All remaining were skipped as duplicates above.
            break;
        };
        selected_indices.push(chosen);
        if let Some(pos) = remaining.iter().position(|&i| i == chosen) {
            remaining.remove(pos);
        }
    }

    selected_indices
        .into_iter()
        .map(|i| candidates[i].clone())
        .collect()
}

/// Lowercase, alphanumeric token set.
///
/// Splits on non-alphanumeric characters; strips empty tokens; drops
/// single-character tokens (high noise / low signal).
fn tokenize(text: &str) -> HashSet<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_string())
        .collect()
}

/// Jaccard similarity `|A ∩ B| / |A ∪ B|`.
///
/// Returns 0.0 when both sets are empty (defensive — could otherwise
/// surface as NaN through floating-point arithmetic upstream).
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recall::RecallKind;

    fn item(id: i64, content: &str, score: f64) -> RecallItem {
        RecallItem {
            node_id: id,
            kind: RecallKind::Observation,
            score,
            content: content.into(),
            sources: Vec::new(),
            category: None,
            source_id: None,
            revision_id: None,
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(mmr_dedup(vec![], 0.7, 0.85, None).is_empty());
    }

    #[test]
    fn keeps_order_when_no_overlap() {
        let items = vec![
            item(1, "alpha beta gamma", 0.9),
            item(2, "delta epsilon zeta", 0.8),
            item(3, "eta theta iota", 0.7),
        ];
        let out = mmr_dedup(items, 0.7, 0.85, None);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].node_id, 1);
        assert_eq!(out[1].node_id, 2);
        assert_eq!(out[2].node_id, 3);
    }

    #[test]
    fn near_duplicate_filtered_out() {
        let items = vec![
            item(1, "caroline researches adoption agencies", 0.9),
            // Same exact token set — Jaccard = 1.0.
            item(2, "researches adoption caroline agencies", 0.85),
            item(3, "melanie paints sunrises", 0.8),
        ];
        let out = mmr_dedup(items, 0.7, 0.85, None);
        let ids: Vec<i64> = out.iter().map(|i| i.node_id).collect();
        assert!(
            !ids.contains(&2),
            "near-duplicate should be removed: {ids:?}"
        );
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }

    #[test]
    fn lambda_zero_prioritizes_diversity() {
        // Three items: top by relevance is item 1, but items 2 and 3
        // share words with it; item 4 is fully diverse.  With λ=0,
        // selection should drive toward maximally different items.
        let items = vec![
            item(1, "alpha beta gamma delta", 0.95),
            item(2, "alpha epsilon zeta", 0.90),
            item(3, "beta epsilon kappa", 0.85),
            item(4, "omega psi chi phi", 0.80),
        ];
        let out = mmr_dedup(items, 0.0, 0.99, Some(2));
        let ids: Vec<i64> = out.iter().map(|i| i.node_id).collect();
        // With λ=0, second selection should be 4 (no token overlap).
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], 1);
        assert_eq!(ids[1], 4, "λ=0 should prefer the diverse item: got {ids:?}");
    }

    #[test]
    fn limit_caps_selection() {
        // Each item is fully disjoint to keep dedup from interfering.
        // Token filter drops single-char and underscore-separated parts.
        let pools = [
            "alpha beta",
            "gamma delta",
            "epsilon zeta",
            "eta theta",
            "iota kappa",
            "lambda mu",
            "nu xi",
            "omicron pi",
            "rho sigma",
            "tau upsilon",
        ];
        let items: Vec<RecallItem> = pools
            .iter()
            .enumerate()
            .map(|(i, t)| item(i as i64, t, 1.0 - 0.05 * i as f64))
            .collect();
        let out = mmr_dedup(items, 0.7, 0.85, Some(3));
        assert_eq!(out.len(), 3);
    }
}
