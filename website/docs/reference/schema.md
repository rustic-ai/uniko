# Schema Reference

Everything an agent learns lives as a single property graph inside uni-db — one store for the
graph, the vectors, and the full-text index, with nothing to keep in sync. Only `Message`s and
`Action`s are observed directly; every other node is derived from them and linked back to its
evidence, so the schema *is* the provenance. This page is the flat catalog of every node and edge
type that `register_schema` installs, drawn straight from the source in `uniko-store`.

For the narrative behind these shapes — why knowledge is derived rather than
stored, how the memory layers map onto cognitive memory — read
[Concepts > Data Model](../concepts/data-model.md) first. This page is the flat
reference you come back to.

!!! note "Always in sync with the code"
    Every label and edge name below is drawn directly from the schema source in
    `crates/uniko-store/src/schema`, so this catalog always matches what
    `register_schema` installs. The canonical name lists live in
    `schema/constants.rs` (`labels::ALL`, `edges::ALL`); the per-node property
    and index definitions live in the sibling files (`facts.rs`, `entities.rs`,
    …). `register_schema` is idempotent — running it again on an existing
    database is a no-op beyond the first call.

## Node types

Node labels are organized by memory layer. Each carries a stable string id
property (`fact_id`, `entity_id`, …), and most carry an auto-computed or
application-computed `embedding` vector for similarity recall.

| Node | Memory family | Purpose |
|---|---|---|
| `Participant` | Foundation | A human, agent, or service. Humans and agents are the same type — uniform actors that send messages and perform actions. |
| `Goal` | Working | A long-running objective that context is assembled around. |
| `Task` | Working | A unit of work toward a `Goal`. |
| `Session` | Episodic | A bounded interaction with a start and (optional) end. |
| `Message` | Episodic | The atomic unit of communication — what someone said, when. |
| `Action` | Episodic | Something a participant did beyond talking (tool call, file write, search). |
| `Episode` | Episodic | A structured learning experience with an outcome — the unit procedure promotion operates on. |
| `Artifact` | Artifacts | A thing in the world: file, document, url, snippet, image, audio, video. |
| `ArtifactContent` | Artifacts | Content-addressed blob metadata, deduplicated by `content_id` (SHA-256). |
| `Chunk` | Artifacts | A segment of an `Artifact` or long `Message`, independently embedded and searchable. |
| `Page` | Artifacts | A rendered/parsed page of a PDF artifact in the document-IR graph (`page_id`, `page_number`, `produced_by`, `plain_markdown`, `block_count`, `escalations`). No vector index — the embeddable text lives on its child `Chunk`s. |
| `Block` | Artifacts | An atomic content block within a `Page` (`kind`, `text`, `reading_order`, `produced_by`, `confidence_kind`/`score`, `bbox_x0..y1`, `confidence_signals`). No vector index — the text rides on a child `Chunk` with `chunk_type="block"`. |
| `Entity` | Semantic | A named thing mentioned across messages, chunks, actions, and artifacts. |
| `Observation` | Semantic | A direct statement perceived from a message or chunk — "Caroline attended an LGBTQ support group". |
| `Fact` | Semantic | A claim consolidated from multiple observations, with temporal validity. |
| `Topic` | Semantic | A cluster of related entities and facts. |
| `Summary` | Semantic | Generated text summarizing a session, task, goal, artifact, entity, or topic. |
| `Procedure` | Procedural | A proven action sequence with effectiveness statistics. |
| `Rule` | Procedural | A Locy rule used for formal reasoning and fact derivation. |
| `Pattern` | Procedural | A recurring episode pattern surfaced by the `episode_pattern_detector` stdlib Locy rule, grouped by `(action_type, outcome)` with `support` and `mean_importance`. Upserted on a deterministic `pattern_id`. |
| `ConsolidationCycle` | Meta-memory | A record of one consolidation run — what it processed, created, and invalidated. |
| `DeadLetter` | Meta-memory | A failed pipeline task held for retry. |
| `Organization` | Organization | A grouping that participants belong to. |
| `Team` | Organization | A sub-grouping within an organization. |
| `KnowledgeBaseStats` | KB metadata | Singleton node holding modality-presence cache and persisted blob-storage config. |

!!! tip "Working memory computes on demand"
    There is no stored `WorkingMemory` node. Working memory is a live view assembled from a
    `Goal` — its tasks, recent sessions and messages, derived facts, involved entities, and
    proven procedures — reflecting the graph at query time. See
    [Concepts > Memory Model](../concepts/memory-model.md).

### Key fields on the most important nodes

A few fields control uniko's load-bearing behaviour and are worth knowing by
name: `Fact.valid_at` gates contradiction detection and bitemporal recall,
`Entity.unstable` triggers the recall drift override, and
`Observation.temporal_anchor` powers temporal recall.

=== "Fact"

    `Fact` is `(subject, predicate, object)` plus the metadata that makes it
    *temporal* and *traceable* (`facts.rs`).

    | Field | Type | Meaning |
    |---|---|---|
    | `subject`, `predicate`, `object` | String | The claim triple. `object` is nullable. |
    | `confidence` | Float64 | Belief in the claim. |
    | `observation_count` | Int64 | How many observations support it. |
    | `valid_at` | **Btic** | Temporal validity as a half-open interval `[lo, hi)` with per-bound certainty and granularity. |
    | `source_rule` | String | Which `Rule` derived this fact. |
    | `visibility` | String | Access scope — `null`/`""`/`"public"` visible to all; `"private:{participant_id}"`, `"team:{team_id}"`, `"org:{org_id}"` restrict the audience. |

    `valid_at` is a single BTIC property, not a pair of DateTime columns. An
    active fact is `[observed_at, +∞)`; an invalidated fact has its `hi` bound
    closed to the moment it stopped being true. Certainty is `approximate`
    below `CERTAINTY_THRESHOLD` (10) supporting observations and upgrades to
    `definite` at or above it (`btic.rs`). See
    [Concepts > Facts and Drift](../concepts/facts-and-drift.md).

=== "Entity"

    `Entity` carries the drift-tracking fields consolidation uses to flag
    unstable knowledge (`entities.rs`).

    | Field | Type | Meaning |
    |---|---|---|
    | `name`, `entity_type` | String | The named thing and its kind. |
    | `frequency` | Int64 | How often it has been mentioned. |
    | `confidence` | Float64 | Belief in the entity's resolution. |
    | `invalidation_count` | Int64 | Total times any `Fact` about this entity has been invalidated. |
    | `last_invalidation_at` | DateTime | When the most recent invalidation happened. |
    | `unstable` | Bool | Flips to `true` when invalidations cross the drift threshold within a rolling window. |

    When `unstable` is set, recall can force deeper retrieval phases for any
    query that references the entity — the *drift override*. The `unstable`
    column is hash-indexed so this check stays cheap.

=== "Observation"

    `Observation` is the first-class, retrievable claim form (`observations.rs`).

    | Field | Type | Meaning |
    |---|---|---|
    | `content` | String | The perceived statement (auto-embedded, fulltext-indexed). |
    | `subject`, `predicate`, `object` | String | Structured triple surfaced from the parser, when available. |
    | `temporal_phrase` | String | Surface time expression, e.g. "two weekends ago". |
    | `temporal_anchor` | DateTime | Its resolved absolute date (BTree-indexed for range filters). |
    | `observed_at` | DateTime | When the observed event happened in the real world. |
    | `visibility` | String | Optional scope: `null`/`"public"` visible to all; `"private:<pid>"`, `"team:<tid>"`, `"org:<oid>"` restrict the audience. |

## Edge types

Edges connect the graph. Several are **polymorphic** — the same edge name is
registered with multiple source or target labels (for example `MENTIONS` runs
from five different node types into `Entity`). Those are noted inline.

### Working memory and tasks

| Edge | From → To | Meaning |
|---|---|---|
| `OWNED_BY` | `Goal` → `Participant` | Who is responsible for the goal. |
| `PARENT_GOAL` | `Goal` → `Goal` | Goal decomposition. |
| `PART_OF` | `Task` → `Goal` | The task belongs to this goal. |
| `ASSIGNED_TO` | `Task` → `Participant` | Who the task is assigned to. |
| `DEPENDS_ON` | `Task` → `Task` | Task dependency. |
| `SUBTASK_OF` | `Task` → `Task` | Task hierarchy. |
| `FOR_TASK` | `Session`, `Episode` → `Task` | Which task a session or episode works on. |
| `FOR_GOAL` | `Session` → `Goal` | Direct link when there is no intermediate task. |
| `PARTICIPATED_IN` | `Participant` → `Session` | Who took part in a session. |

### Episodic

| Edge | From → To | Meaning |
|---|---|---|
| `SENT_BY` | `Message` → `Participant` | The author of a message (carries `role`). |
| `ADDRESSED_TO` | `Message` → `Participant` | The intended recipient. |
| `IN_SESSION` | `Message`, `Action`, `Episode` → `Session` | Places episodic nodes in their session. |
| `NEXT` | `Message` → `Message` | Conversational order (carries `gap_ms`). |
| `PERFORMED_BY` | `Action` → `Participant` | Who performed an action. |
| `TRIGGERED_BY` | `Action`, `Episode` → `Message` | The message that caused this action or episode. |
| `PRODUCED` | `Action` → `Artifact` | The artifact an action produced. |
| `NEXT_ACTION` | `Action` → `Action` | Action sequence within a session (provenance only). |
| `RECORDED_BY` | `Episode` → `Participant` | Who the episode was recorded for. |
| `INVOLVES` | `Episode` → `Action` | The actions taken during an episode. |
| `FOLLOWED_BY` | `Episode` → `Episode` | Temporal chain of experiences (carries `gap_ms`). |

### Artifacts and chunks

| Edge | From → To | Meaning |
|---|---|---|
| `HAS_CHUNK` | `Artifact`, `Message`, `Session`, `Block` → `Chunk` | Splits content into independently searchable chunks (carries `index`). A `Block` owns a child `Chunk` with `chunk_type="block"` carrying the embeddable text. |
| `HAS_PAGE` | `Artifact` → `Page` | Attaches a PDF artifact to one of its pages (carries `index`). |
| `HAS_CONTENT` | `Artifact` → `ArtifactContent` | Links an artifact to its deduplicated blob (carries `role`). |
| `CREATED_BY` | `Artifact` → `Action` | Which action created the artifact. |
| `MODIFIED_BY` | `Artifact` → `Action` | Which action modified it (carries `diff_summary`). |
| `ATTACHED_TO` | `Artifact` → `Session`, `Message` | Where a file was dropped into a conversation, when no producing action exists (carries `attached_at`). |
| `CONTAINS` | `Page` → `Block` | Attaches a page to one of its blocks (carries `reading_order`). |
| `NEXT_IN_READING_ORDER` | `Block` → `Block` | Chains blocks in page reading order. |

### Semantic

| Edge | From → To | Meaning |
|---|---|---|
| `MENTIONS` | `Message`, `Chunk`, `Action`, `Artifact`, `Episode` → `Entity` | Surfaces a named entity (carries `count`). |
| `OBSERVED_IN` | `Observation` → `Message`, `Chunk` | The source an observation came from. |
| `OBSERVED_DURING` | `Observation` → `Episode` | Which episode it was observed in. |
| `ABOUT` | `Observation`, `Fact`, `Chunk` → `Entity`, `Participant` | What an observation, fact, or chunk is about. |
| `SUPPORTED_BY` | `Fact` → `Observation` | Evidence backing a fact (carries `weight`). |
| `DERIVED_BY` | `Fact` → `Rule` | Which rule derived the fact. |
| `DERIVED_FROM` | `Fact`, `Procedure`, `Artifact` → `Episode`, `Action`, `Artifact` | Provenance: the episodes/actions a fact or procedure came from, or the artifact a derived artifact came from (carries `derivation_kind`, `derived_at`). |
| `INVALIDATES` | `Fact` → `Fact` | A newer fact closes an older one (carries `reason`, `invalidated_at`). |
| `SHARED_FROM` | `Fact` → `Fact` | A fact shared across agents (carries `shared_by`, `shared_at`). |
| `CONTRADICTED_BY` | `Fact` → `Episode` | Provenance for a contradiction surfaced by the `contradiction_detector` rule; the fact is also BTIC-invalidated (carries `detected_at`). |
| `BELONGS_TO` | `Entity`, `Fact` → `Topic` | Membership in a topic cluster. |
| `SUMMARIZES` | `Summary` → `Session`, `Task`, `Goal`, `Artifact`, `Entity`, `Topic` | What a summary covers. |

### Procedural

| Edge | From → To | Meaning |
|---|---|---|
| `OPERATES_ON` | `Procedure` → `Entity` | The entity a procedure acts on. |
| `USED_IN` | `Procedure` → `Task` | Tasks where the procedure was applied. |
| `SUPERSEDES` | `Rule` → `Rule` | A newer rule replaces an older one. |
| `COVERS` | `Rule` → `Episode` | Episodes a rule applies to (carries `correct`). |

### Meta-memory and organization

| Edge | From → To | Meaning |
|---|---|---|
| `PROCESSED` | `ConsolidationCycle` → `Observation` | Observations a cycle consumed. |
| `INVOLVED` | `ConsolidationCycle` → `Episode` | Episodes a cycle involved. |
| `CREATED` | `ConsolidationCycle` → `Fact` | Facts a cycle newly created. |
| `REINFORCED` | `ConsolidationCycle` → `Fact` | Facts a cycle re-observed rather than created. |
| `INVALIDATED` | `ConsolidationCycle` → `Fact` | Facts a cycle closed. |
| `PROMOTED` | `ConsolidationCycle` → `Procedure` | Procedures a cycle promoted. |
| `APPLIED_RULE` | `ConsolidationCycle` → `Rule` | Rules a cycle evaluated. |
| `MEMBER_OF` | `Participant` → `Organization` | Org membership (carries `role`, `joined_at`). |
| `PART_OF_TEAM` | `Participant` → `Team` | Team membership. |
| `TEAM_IN_ORG` | `Team` → `Organization` | A team's parent organization. |

!!! note "DeadLetter has no edges"
    `DeadLetter` is a standalone node — it records a failed pipeline step
    (`step`, `error`, `node_ref`, `retry_count`, `max_retries`,
    `next_retry_at`) and is not wired into the graph by an edge type.

## Indexes at a glance

Most id properties carry a `Hash` scalar index for O(1) lookup — the exceptions
are `Chunk.chunk_id`, which is unindexed because chunks are reached through
`HAS_CHUNK` / vector / BM25 rather than by id, and `DeadLetter`, which has no id
property at all. Beyond that, the schema leans on three index families:

- **Hash** — exact-match columns like `Entity.name`, `Fact.subject`,
  `Fact.predicate`, `Episode.action_type`, `Entity.unstable`.
- **BTree** — range columns like `Message.timestamp`, `Session.started_at`,
  `Fact.confidence`, `Observation.temporal_anchor`, `Artifact.page_count`.
- **FullText** — BM25 columns like `Message.content`, `Chunk.text`,
  `Observation.content`, `Summary.text`, `Fact.subject`.

Most embedding-bearing nodes also carry a **Vector** index. `Message`, `Chunk`,
`Observation`, and `Summary` auto-embed from their text property (uni-db
computes the vector on write); `Entity`, `Fact`, `Episode`, `Goal`, `Task`,
`Session`, `Topic`, and `Procedure` use application-computed embeddings.

!!! note "This catalog reflects the installed schema"
    Because it is drawn from `crates/uniko-store/src/schema`, this page is
    authoritative — it lists exactly what `register_schema` installs. Some
    earlier design notes sketch additional multimodal and cross-agent fields that
    are not part of the registered schema; this catalog is the one to build on.
