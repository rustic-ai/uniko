//! Access-control policy for Facts and Observations (F66).
//!
//! Visibility format on the node's `visibility` property:
//! - `null` or `""` or `"public"` — visible to every viewer.
//! - `"private:{participant_id}"` — visible only when the viewer's
//!   `participant_id` matches.
//! - `"team:{team_id}"` — visible when the viewer is a `PART_OF_TEAM`
//!   member of `team_id`.
//! - `"org:{org_id}"` — visible when the viewer is a `MEMBER_OF`
//!   member of `org_id` (directly or via team membership).
//!
//! The policy is applied as a *filter* over the recall cascade's
//! [`crate::recall::ContextBundle`] just before it's handed to the
//! caller. [`Visibility`] is the typed form callers write with
//! `Agent::set_visibility`; deletion writes `redacted:` values, which
//! hide an item for good.
//!
//! Facts and Observations carry visibility, and so do Messages and their
//! Chunks when the turn was scoped with `Turn::visibility`; the filter
//! covers all four. Session-level transcript and observation chunks
//! aggregate many turns, so they are built from public turns only.

use std::collections::HashSet;

use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use crate::recall::{ContextBundle, RecallItem, RecallKind};

/// Viewer identity used for filtering.
///
/// Construct via [`Viewer::new`] from an external `participant_id`.
/// The recall filter resolves the participant's team and org memberships
/// once per call and caches them in this struct so per-item checks
/// stay O(1).
#[derive(Debug, Clone)]
pub struct Viewer {
    /// External `participant_id` for the requesting participant.
    pub participant_id: String,
    /// `team_id` values the viewer belongs to.
    pub teams: HashSet<String>,
    /// `org_id` values the viewer belongs to (directly or via team).
    pub orgs: HashSet<String>,
}

impl Viewer {
    /// Build a viewer for `participant_id` by resolving its
    /// memberships against the store.
    ///
    /// Membership is the union of:
    /// - Direct `(:Participant)-[:MEMBER_OF]->(:Organization)` edges.
    /// - Teams reached via `(:Participant)-[:PART_OF_TEAM]->(:Team)`
    ///   plus the orgs reached from those teams via
    ///   `(:Team)-[:TEAM_IN_ORG]->(:Organization)`.
    ///
    /// An unknown `participant_id` returns a viewer with empty
    /// memberships — they can still see anything `"public"` or
    /// `"private:{their_id}"`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the membership lookup fails.
    pub async fn new(
        kb: &KnowledgeBase,
        participant_id: impl Into<String>,
    ) -> Result<Self, UnikoError> {
        let pid = participant_id.into();
        let m = kb.resolve_participant_memberships(&pid).await?;
        Ok(Self {
            participant_id: pid,
            teams: m.teams.into_iter().collect(),
            orgs: m.orgs.into_iter().collect(),
        })
    }

    /// Construct a viewer from already-resolved memberships.
    ///
    /// Useful for tests and for callers that maintain their own
    /// participant directory and don't want a graph round-trip.
    #[must_use]
    pub fn from_parts(
        participant_id: impl Into<String>,
        teams: impl IntoIterator<Item = String>,
        orgs: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            participant_id: participant_id.into(),
            teams: teams.into_iter().collect(),
            orgs: orgs.into_iter().collect(),
        }
    }
}

/// Who may see a Fact or Observation.
///
/// Written with [`Agent::set_visibility`](crate::Agent::set_visibility);
/// recall then drops the item for any viewer it does not admit. `Team`
/// and `Org` resolve through the memberships recorded with
/// [`Uniko::add_team_member`](crate::Uniko::add_team_member) and its
/// siblings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Visibility {
    /// Every viewer. The default for an item that was never scoped.
    Public,
    /// Only the participant with this id.
    Private(String),
    /// Members of this team.
    Team(String),
    /// Members of this org, directly or through one of its teams.
    Org(String),
}

impl Visibility {
    /// The stored form: `public`, `private:{id}`, `team:{id}`, `org:{id}`.
    #[must_use]
    pub fn as_property(&self) -> String {
        match self {
            Self::Public => "public".to_string(),
            Self::Private(id) => format!("private:{id}"),
            Self::Team(id) => format!("team:{id}"),
            Self::Org(id) => format!("org:{id}"),
        }
    }

    /// Parse the stored form back. An empty value is `Public`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] for an unknown scheme or an empty id.
    pub fn parse(raw: &str) -> Result<Self, UnikoError> {
        let raw = raw.trim();
        if raw.is_empty() || raw == "public" {
            return Ok(Self::Public);
        }
        let (scheme, id) = raw.split_once(':').ok_or_else(|| {
            UnikoError::Config(format!(
                "visibility '{raw}': expected public, private:<id>, team:<id> or org:<id>"
            ))
        })?;
        let id = id.trim();
        if id.is_empty() {
            return Err(UnikoError::Config(format!(
                "visibility '{raw}' has an empty id"
            )));
        }
        match scheme {
            "private" => Ok(Self::Private(id.to_string())),
            "team" => Ok(Self::Team(id.to_string())),
            "org" => Ok(Self::Org(id.to_string())),
            _ => Err(UnikoError::Config(format!(
                "visibility '{raw}': unknown scheme '{scheme}'"
            ))),
        }
    }

    fn validate(&self) -> Result<(), UnikoError> {
        match self {
            Self::Public => Ok(()),
            Self::Private(id) | Self::Team(id) | Self::Org(id) if id.trim().is_empty() => Err(
                UnikoError::Config("visibility id must not be empty".to_string()),
            ),
            _ => Ok(()),
        }
    }
}

/// Set `node_id`'s visibility; see [`Agent::set_visibility`](crate::Agent::set_visibility).
pub(crate) async fn set_visibility(
    kb: &KnowledgeBase,
    node_id: NodeId,
    visibility: &Visibility,
) -> Result<bool, UnikoError> {
    visibility.validate()?;
    kb.set_visibility(node_id, &visibility.as_property()).await
}

/// Decide whether a single visibility string admits `viewer`.
///
/// Exposed for callers that want to enforce visibility outside the
/// recall cascade (e.g. on direct Fact lookups).
#[must_use]
pub fn visibility_admits(visibility: Option<&str>, viewer: &Viewer) -> bool {
    let raw = visibility.unwrap_or("").trim();
    if raw.is_empty() || raw == "public" {
        return true;
    }
    if let Some(pid) = raw.strip_prefix("private:") {
        return pid.trim() == viewer.participant_id;
    }
    if let Some(tid) = raw.strip_prefix("team:") {
        return viewer.teams.contains(tid.trim());
    }
    if let Some(oid) = raw.strip_prefix("org:") {
        return viewer.orgs.contains(oid.trim());
    }
    // Unknown scheme: fail closed so a malformed visibility tag
    // doesn't accidentally leak data.
    false
}

/// Filter a [`ContextBundle`] in-place to remove items the viewer
/// cannot see.
///
/// Only Fact and Observation items are policy-checked — other node
/// types have no visibility scope in spec v6.  Items are looked up
/// in batch (one Cypher query per call) so the filter stays O(items).
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] when the visibility lookup fails.
pub async fn filter_bundle(
    kb: &KnowledgeBase,
    bundle: &mut ContextBundle,
    viewer: &Viewer,
) -> Result<(), UnikoError> {
    let policy_node_ids: Vec<NodeId> = bundle
        .items
        .iter()
        .filter(|i| carries_visibility(i.kind))
        .map(|i| i.node_id)
        .collect();
    if policy_node_ids.is_empty() {
        return Ok(());
    }

    let visibilities = kb.fetch_visibilities(&policy_node_ids).await?;
    bundle.items.retain(|item| {
        visibility_for(item, &visibilities)
            .is_none_or(|v| visibility_admits(Some(v.as_str()), viewer))
    });
    // Recompute token count after filtering — same heuristic the
    // recall cascade and working memory use.
    bundle.total_tokens = bundle.items.len() * 50;
    Ok(())
}

/// Drop soft-forgotten items from a [`ContextBundle`], unconditionally.
///
/// Unlike [`filter_bundle`], this runs for *every* recall regardless of
/// viewer scope, because forgetting is content redaction, not access
/// control. Forgotten Facts/Observations carry a `redacted:`-scheme
/// `visibility`; forgotten Messages/Chunks carry a `redacted = true` flag.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] when a lookup fails.
pub async fn filter_redacted(
    kb: &KnowledgeBase,
    bundle: &mut ContextBundle,
) -> Result<(), UnikoError> {
    let flag_ids: Vec<NodeId> = bundle
        .items
        .iter()
        .filter(|i| matches!(i.kind, RecallKind::Message | RecallKind::Chunk))
        .map(|i| i.node_id)
        .collect();
    let vis_ids: Vec<NodeId> = bundle
        .items
        .iter()
        .filter(|i| matches!(i.kind, RecallKind::Fact | RecallKind::Observation))
        .map(|i| i.node_id)
        .collect();
    if flag_ids.is_empty() && vis_ids.is_empty() {
        return Ok(());
    }

    let redacted = kb.fetch_redacted(&flag_ids).await?;
    let visibilities = if vis_ids.is_empty() {
        std::collections::HashMap::new()
    } else {
        kb.fetch_visibilities(&vis_ids).await?
    };

    let before = bundle.items.len();
    bundle.items.retain(|item| match item.kind {
        RecallKind::Message | RecallKind::Chunk => !redacted.contains(&item.node_id),
        RecallKind::Fact | RecallKind::Observation => !visibilities
            .get(&item.node_id)
            .is_some_and(|v| v.starts_with("redacted:")),
        _ => true,
    });
    if bundle.items.len() != before {
        bundle.total_tokens = bundle.items.len() * 50;
    }
    Ok(())
}

fn visibility_for<'a>(
    item: &RecallItem,
    visibilities: &'a std::collections::HashMap<NodeId, String>,
) -> Option<&'a String> {
    if !carries_visibility(item.kind) {
        return None;
    }
    visibilities.get(&item.node_id)
}

/// The recall kinds whose nodes carry a `visibility` property: Facts and
/// Observations, and Messages and their Chunks scoped by
/// [`Turn::visibility`](crate::Turn::visibility).
pub(crate) fn carries_visibility(kind: RecallKind) -> bool {
    matches!(
        kind,
        RecallKind::Fact | RecallKind::Observation | RecallKind::Message | RecallKind::Chunk
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn viewer(pid: &str, teams: &[&str], orgs: &[&str]) -> Viewer {
        Viewer::from_parts(
            pid,
            teams.iter().map(|s| s.to_string()),
            orgs.iter().map(|s| s.to_string()),
        )
    }

    #[test]
    fn public_admits_everyone() {
        let v = viewer("alice", &[], &[]);
        assert!(visibility_admits(None, &v));
        assert!(visibility_admits(Some(""), &v));
        assert!(visibility_admits(Some("public"), &v));
    }

    #[test]
    fn private_matches_only_owner() {
        let alice = viewer("alice", &[], &[]);
        let bob = viewer("bob", &[], &[]);
        assert!(visibility_admits(Some("private:alice"), &alice));
        assert!(!visibility_admits(Some("private:alice"), &bob));
    }

    #[test]
    fn team_visibility_checks_membership() {
        let alice = viewer("alice", &["red"], &[]);
        let bob = viewer("bob", &["blue"], &[]);
        assert!(visibility_admits(Some("team:red"), &alice));
        assert!(!visibility_admits(Some("team:red"), &bob));
    }

    #[test]
    fn org_visibility_checks_membership() {
        let alice = viewer("alice", &[], &["acme"]);
        let bob = viewer("bob", &[], &["other"]);
        assert!(visibility_admits(Some("org:acme"), &alice));
        assert!(!visibility_admits(Some("org:acme"), &bob));
    }

    #[test]
    fn unknown_scheme_fails_closed() {
        let v = viewer("alice", &[], &[]);
        assert!(!visibility_admits(Some("secret:42"), &v));
        assert!(!visibility_admits(Some("hunter2"), &v));
    }
}
