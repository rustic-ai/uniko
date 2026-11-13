//! Access control: participant membership resolution and writes, and
//! node-visibility reads and writes. Backs `uniko_memory::policy`.

use std::collections::HashMap;

use uni_db::Value;

use crate::error::{Result, UnikoError};
use crate::schema::constants::{edges, labels};
use crate::storage::KnowledgeBase;
use crate::types::NodeId;

/// A participant's resolved group memberships, used to decide which
/// `team:`/`org:`-scoped items they may see.
///
/// `teams` and `orgs` are non-empty membership ids; `orgs` is the union
/// of direct `MEMBER_OF` orgs and orgs reached transitively through the
/// participant's teams. Neither list is deduplicated across the two org
/// sources — callers typically collect into a set.
#[derive(Debug, Clone, Default)]
pub struct ParticipantMemberships {
    /// Team ids reached via `(:Participant)-[:PART_OF_TEAM]->(:Team)`.
    pub teams: Vec<String>,
    /// Org ids: direct `MEMBER_OF` plus `TEAM_IN_ORG` from the teams.
    pub orgs: Vec<String>,
}

impl KnowledgeBase {
    /// Resolve the group memberships for `participant_id`.
    ///
    /// An unknown participant yields empty membership lists (not an
    /// error) — they can still see `public` and their own `private:`
    /// items.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) when
    /// the lookup fails.
    pub async fn resolve_participant_memberships(
        &self,
        participant_id: &str,
    ) -> Result<ParticipantMemberships> {
        let session = self.db.session();
        let cypher = "MATCH (p:Participant) WHERE p.participant_id = $pid \
                      OPTIONAL MATCH (p)-[:PART_OF_TEAM]->(t:Team) \
                      OPTIONAL MATCH (t)-[:TEAM_IN_ORG]->(o1:Organization) \
                      OPTIONAL MATCH (p)-[:MEMBER_OF]->(o2:Organization) \
                      RETURN collect(DISTINCT t.team_id) AS teams, \
                             collect(DISTINCT o1.org_id) AS orgs1, \
                             collect(DISTINCT o2.org_id) AS orgs2";
        let result = session
            .query_with(cypher)
            .param("pid", participant_id)
            .fetch_all()
            .await?;

        let mut memberships = ParticipantMemberships::default();
        if let Some(row) = result.rows().first() {
            let team_ids: Vec<String> = row.get("teams").unwrap_or_default();
            let org_ids_a: Vec<String> = row.get("orgs1").unwrap_or_default();
            let org_ids_b: Vec<String> = row.get("orgs2").unwrap_or_default();
            memberships
                .teams
                .extend(team_ids.into_iter().filter(|t| !t.is_empty()));
            memberships.orgs.extend(
                org_ids_a
                    .into_iter()
                    .chain(org_ids_b)
                    .filter(|o| !o.is_empty()),
            );
        }
        Ok(memberships)
    }

    /// Batch-fetch the `visibility` property for `node_ids`.
    ///
    /// Returns a map from node id to its visibility string (`""` when the
    /// property is absent). Nodes that don't exist are simply absent from
    /// the map.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) when
    /// the lookup fails.
    pub async fn fetch_visibilities(&self, node_ids: &[NodeId]) -> Result<HashMap<NodeId, String>> {
        let session = self.db.session();
        let ids: Vec<Value> = node_ids.iter().map(|n| Value::Int(*n)).collect();
        let cypher = "MATCH (n) WHERE id(n) IN $ids \
                      RETURN id(n) AS nid, coalesce(n.visibility, '') AS vis";
        let result = session
            .query_with(cypher)
            .param("ids", Value::List(ids))
            .fetch_all()
            .await?;
        let mut out = HashMap::new();
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            let vis: String = row.get("vis").unwrap_or_default();
            out.insert(nid, vis);
        }
        Ok(out)
    }
}

/// A group membership that `team:` / `org:` visibility resolves through.
///
/// Each variant names the edge [`resolve_participant_memberships`]
/// follows: a participant in a team (`PART_OF_TEAM`), a participant in an
/// org directly (`MEMBER_OF`), or a team in an org (`TEAM_IN_ORG`), which
/// gives the team's members that org too.
///
/// [`resolve_participant_memberships`]: KnowledgeBase::resolve_participant_memberships
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Membership {
    /// `(:Participant)-[:PART_OF_TEAM]->(:Team)`.
    ParticipantInTeam,
    /// `(:Participant)-[:MEMBER_OF]->(:Organization)`.
    ParticipantInOrg,
    /// `(:Team)-[:TEAM_IN_ORG]->(:Organization)`.
    TeamInOrg,
}

impl Membership {
    /// `(member label, member id field, group label, group id field, edge)`.
    fn shape(
        self,
    ) -> (
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    ) {
        match self {
            Self::ParticipantInTeam => (
                labels::PARTICIPANT,
                "participant_id",
                labels::TEAM,
                "team_id",
                edges::PART_OF_TEAM,
            ),
            Self::ParticipantInOrg => (
                labels::PARTICIPANT,
                "participant_id",
                labels::ORGANIZATION,
                "org_id",
                edges::MEMBER_OF,
            ),
            Self::TeamInOrg => (
                labels::TEAM,
                "team_id",
                labels::ORGANIZATION,
                "org_id",
                edges::TEAM_IN_ORG,
            ),
        }
    }
}

/// Visibility values written by deletion. They hide an item for good, so
/// [`set_visibility`](KnowledgeBase::set_visibility) refuses to overwrite
/// them.
const REDACTED_PREFIX: &str = "redacted:";

impl KnowledgeBase {
    /// Record a membership, creating the member and group nodes when
    /// absent. Idempotent: returns `false` when it was already recorded.
    ///
    /// A Participant created here gets `kind = "unknown"`; ingest fills in
    /// the real kind when the participant first speaks, and an existing
    /// Participant is left as it is.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] for an empty id, or
    /// [`UnikoError::Storage`] when a write fails.
    pub async fn add_membership(
        &self,
        membership: Membership,
        member_id: &str,
        group_id: &str,
    ) -> Result<bool> {
        let (member_label, member_field, group_label, group_field, edge) = membership.shape();
        let member = self
            .ensure_group_node(member_label, member_field, member_id)
            .await?;
        let group = self
            .ensure_group_node(group_label, group_field, group_id)
            .await?;

        // Serialize the check-then-create so two concurrent calls cannot
        // both record the edge.
        let lock_key = format!("membership:{edge}:{member}:{group}").into_bytes();
        let _guard = self.rmw_locks.lock(&lock_key).await;
        if self.count_edges(edge, member, group).await? > 0 {
            return Ok(false);
        }
        self.create_edge(edge, member, group, &HashMap::new())
            .await?;
        Ok(true)
    }

    /// Remove a membership. Returns `false` when it was not recorded. The
    /// member and group nodes stay.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the delete fails.
    pub async fn remove_membership(
        &self,
        membership: Membership,
        member_id: &str,
        group_id: &str,
    ) -> Result<bool> {
        let (member_label, member_field, group_label, group_field, edge) = membership.shape();
        let cypher = format!(
            "MATCH (a:{member_label} {{{member_field}: $a}})-[r:{edge}]->\
             (b:{group_label} {{{group_field}: $b}}) DELETE r"
        );
        let before = self
            .count_memberships(membership, member_id, group_id)
            .await?;
        if before == 0 {
            return Ok(false);
        }
        let session = self.db.session();
        let tx = session.tx().await?;
        tx.query_with(&cypher)
            .param("a", member_id)
            .param("b", group_id)
            .fetch_all()
            .await?;
        tx.commit().await?;
        Ok(true)
    }

    /// Set the `visibility` of a Fact or Observation, the two kinds of
    /// item recall filters by visibility.
    ///
    /// Returns `false` when no node has `node_id`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] when the node is not a Fact or
    /// Observation, or was forgotten (its `redacted:` visibility must stay),
    /// or [`UnikoError::Storage`] when the write fails.
    pub async fn set_visibility(&self, node_id: NodeId, visibility: &str) -> Result<bool> {
        let Some((label, props)) = self.get_node(node_id).await? else {
            return Ok(false);
        };
        if label != labels::FACT && label != labels::OBSERVATION {
            return Err(UnikoError::Config(format!(
                "visibility applies to Fact and Observation nodes; node {node_id} is a {label}"
            )));
        }
        if let Some(Value::String(current)) = props.get("visibility")
            && current.starts_with(REDACTED_PREFIX)
        {
            return Err(UnikoError::Config(format!(
                "node {node_id} was forgotten; its visibility cannot change"
            )));
        }
        let mut update = HashMap::new();
        update.insert(
            "visibility".to_string(),
            Value::String(visibility.to_string()),
        );
        self.update_node(node_id, &update).await?;
        Ok(true)
    }

    /// The node with `id_field = id` under `label`, created when absent.
    async fn ensure_group_node(&self, label: &str, id_field: &str, id: &str) -> Result<NodeId> {
        if id.trim().is_empty() {
            return Err(UnikoError::Config(format!("{id_field} must not be empty")));
        }
        if let Some((nid, _)) = self.get_node_by_ext_id(label, id_field, id).await? {
            return Ok(nid);
        }
        let mut props = HashMap::new();
        if label == labels::PARTICIPANT {
            props.insert("kind".to_string(), Value::String("unknown".to_string()));
        }
        self.merge_node(label, id_field, id, &props).await
    }

    async fn count_edges(&self, edge: &str, from: NodeId, to: NodeId) -> Result<i64> {
        let session = self.db.session();
        let cypher = format!(
            "MATCH (a)-[r:{edge}]->(b) WHERE id(a) = $a AND id(b) = $b RETURN count(r) AS n"
        );
        let result = session
            .query_with(&cypher)
            .param("a", from)
            .param("b", to)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .first()
            .and_then(|r| r.get::<i64>("n").ok())
            .unwrap_or(0))
    }

    async fn count_memberships(
        &self,
        membership: Membership,
        member_id: &str,
        group_id: &str,
    ) -> Result<i64> {
        let (member_label, member_field, group_label, group_field, edge) = membership.shape();
        let session = self.db.session();
        let cypher = format!(
            "MATCH (a:{member_label} {{{member_field}: $a}})-[r:{edge}]->\
             (b:{group_label} {{{group_field}: $b}}) RETURN count(r) AS n"
        );
        let result = session
            .query_with(&cypher)
            .param("a", member_id)
            .param("b", group_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .first()
            .and_then(|r| r.get::<i64>("n").ok())
            .unwrap_or(0))
    }
}
