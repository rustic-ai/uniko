# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Visibility and memberships, viewer scopes, rule management, and the
facts / actions / episodes / observations methods."""

from __future__ import annotations

import pytest

import uniko
from uniko.models import ConsolidationReport, FactUpsert, RuleView
from uniko.models.typed import TypedUniko

QUERY = "zeppelin launch"


def recall_ids(agent: uniko.Agent, scope: uniko.Scope | None = None) -> set[int]:
    if scope is None:
        bundle = agent.recall_sync(QUERY)
    else:
        bundle = agent.recall_in_sync(QUERY, scope)
    return {item.node_id for item in bundle.items}


def seeded(memory: uniko.Uniko) -> tuple[int, int]:
    """A team-scoped and a private observation, written by `writer`."""
    writer = memory.agent("writer")
    writer.session("planning").observe_sync(
        uniko.Turn("writer", "notes from the planning call").id("plan-1")
    )
    team = writer.add_observation_sync(
        "plan-1",
        "the zeppelin launch is scheduled for march",
        "launch",
        visibility="team:eng",
    )
    private = writer.add_observation_sync(
        "plan-1", "the zeppelin launch slips to april for carol", "launch"
    )
    assert writer.set_visibility_sync(private, "private:carol")
    return team, private


def test_visibility_filters_scoped_reads_by_membership() -> None:
    memory = uniko.Uniko.builder().in_memory().scope_to_agent().build_sync()
    assert memory.add_team_member_sync("alice", "eng") is True
    assert memory.add_team_member_sync("alice", "eng") is False, "idempotent"
    team, private = seeded(memory)

    alice = recall_ids(memory.agent("alice"))
    assert team in alice and private not in alice
    carol = recall_ids(memory.agent("carol"))
    assert private in carol and team not in carol
    bob = recall_ids(memory.agent("bob"))
    assert team not in bob and private not in bob

    assert memory.remove_team_member_sync("alice", "eng") is True
    assert team not in recall_ids(memory.agent("alice")), "a fresh handle sees it"


def test_viewer_scopes_one_call_or_the_whole_instance() -> None:
    memory = uniko.Uniko.in_memory_sync()
    team, private = seeded(memory)
    # Unscoped reads see everything.
    assert {team, private} <= recall_ids(memory.agent("anyone"))

    viewer = uniko.Viewer("bob", teams=["eng"])
    assert viewer.participant_id == "bob" and viewer.teams == ["eng"]
    scoped = recall_ids(memory.agent("anyone"), uniko.Scope().as_viewer(viewer))
    assert team in scoped and private not in scoped

    # Builder scope applies to every agent. Reopening the same store is not
    # possible in memory, so check it on a fresh instance.
    outsider = (
        uniko.Uniko.builder().in_memory().scope(uniko.Viewer("nobody")).build_sync()
    )
    team2, private2 = seeded(outsider)
    assert not {team2, private2} & recall_ids(outsider.agent("x"))


def test_team_in_org_reaches_its_members() -> None:
    memory = uniko.Uniko.builder().in_memory().scope_to_agent().build_sync()
    writer = memory.agent("writer")
    writer.session("s").observe_sync(uniko.Turn("writer", "org news").id("m-1"))
    item = writer.add_observation_sync(
        "m-1",
        "the zeppelin launch is an acme-wide event",
        "launch",
        visibility="org:acme",
    )
    memory.add_team_member_sync("dana", "infra")
    assert item not in recall_ids(memory.agent("dana"))
    assert memory.add_team_to_org_sync("infra", "acme") is True
    assert item in recall_ids(memory.agent("dana"))
    assert memory.add_org_member_sync("erin", "acme") is True
    assert item in recall_ids(memory.agent("erin"))
    assert memory.remove_org_member_sync("erin", "acme") is True
    assert memory.remove_team_from_org_sync("infra", "acme") is True
    assert memory.remove_team_from_org_sync("infra", "acme") is False


def test_set_visibility_errors() -> None:
    memory = uniko.Uniko.in_memory_sync()
    agent = memory.agent("a")
    with pytest.raises(uniko.ConfigError, match="unknown scheme"):
        agent.set_visibility_sync(1, "group:x")
    assert agent.set_visibility_sync(2**62, "public") is False
    with pytest.raises(uniko.ConfigError, match="must not be empty"):
        memory.add_team_member_sync("", "eng")


def test_assert_and_invalidate_a_fact() -> None:
    agent = uniko.Uniko.in_memory_sync().agent("a")
    upsert = agent.assert_fact_sync(
        "release", "ships_on", "friday", visibility="team:eng"
    )
    assert isinstance(upsert, uniko.FactUpsert)
    assert upsert.was_created and upsert.fact_id
    again = agent.assert_fact_sync(
        "release", "ships_on", "friday", visibility="team:eng"
    )
    assert again.node_id == upsert.node_id and not again.was_created
    agent.invalidate_fact_sync(upsert.fact_id, reason="moved to monday")
    with pytest.raises(uniko.UnikoError):
        agent.invalidate_fact_sync("no-such-fact")


async def test_actions_and_episodes() -> None:
    memory = await uniko.Uniko.in_memory()
    agent = memory.agent("fresh-agent")
    action = await agent.record_action(
        "tool_call", input={"query": "weather"}, output={"temp": 21}, status="success"
    )
    assert isinstance(action, uniko.ActionResult) and action.action_node > 0
    episode = await agent.record_episode(
        "tool_call", outcome="success", involved_action_ids=[]
    )
    assert episode > 0

    session = agent.session("ops")
    await session.observe(uniko.Turn("user", "deploy please").id("ops-1"))
    in_session = await session.record_action("deploy")
    assert in_session.action_node > 0


def test_rules_list_explain_delete() -> None:
    agent = uniko.Uniko.in_memory_sync().agent("a")
    source = "CREATE RULE py_rule AS MATCH (n:Episode) YIELD KEY n"
    agent.define_rule_sync("py_rule", source)

    rules = agent.list_rules_sync()
    names = {r.name for r in rules}
    assert "py_rule" in names
    stdlib = [r for r in rules if r.source_type == "stdlib"]
    assert stdlib, "stdlib rules are listed"

    explained = agent.explain_rule_sync("py_rule")
    assert isinstance(explained, uniko.RuleView)
    assert explained.source == source
    assert agent.explain_rule_sync("missing") is None

    assert agent.delete_rule_sync("py_rule") is True
    deleted = agent.explain_rule_sync("py_rule")
    assert deleted is not None and deleted.status == "pruned"
    assert agent.delete_rule_sync("missing") is False
    with pytest.raises(uniko.ConfigError, match="stdlib"):
        agent.delete_rule_sync(stdlib[0].name)


def test_consolidate_reports_the_cortex_sweeps() -> None:
    memory = uniko.Uniko.in_memory_sync()
    agent = memory.agent("a")
    agent.session("s").observe_sync(uniko.Turn("alice", "I love hiking in the Alps"))
    report = agent.consolidate_sync()
    assert isinstance(report, uniko.ConsolidationReport)
    assert isinstance(report.cycle, uniko.CycleStats)
    assert report.procedures.created >= 0 and report.topics.created >= 0
    assert isinstance(ConsolidationReport.from_native(report).cycle.facts_created, int)


def test_typed_wrappers_return_models() -> None:
    uni = TypedUniko.in_memory_sync()
    agent = uni.agent("a")
    assert isinstance(agent.assert_fact_sync("x", "is", "y"), FactUpsert)
    assert all(isinstance(r, RuleView) for r in agent.list_rules_sync())
    assert isinstance(agent.consolidate_sync(), ConsolidationReport)
    assert uni.add_team_member_sync("p", "t") is True


# ── Turn visibility, extractors, procedures ──────────────────────────────


def test_turn_visibility_scopes_the_message() -> None:
    memory = uniko.Uniko.builder().in_memory().scope_to_agent().build_sync()
    memory.add_team_member_sync("alice", "eng")
    session = memory.agent("writer").session("s")
    result = session.observe_sync(
        uniko.Turn("writer", "The zeppelin hangar opens in Lisbon next quarter")
        .id("scoped")
        .visibility("team:eng")
    )
    hidden = {
        result.message_node_id,
        *result.chunk_node_ids,
        *result.extracted_observations,
    }
    query = "zeppelin hangar Lisbon"
    alice = {i.node_id for i in memory.agent("alice").recall_sync(query).items}
    bob = {i.node_id for i in memory.agent("bob").recall_sync(query).items}
    assert alice & hidden, "alice is in eng"
    assert not bob & hidden, "bob is not"
    with pytest.raises(uniko.ConfigError):
        uniko.Turn("a", "b").visibility("crowd:x")


def test_scoped_fact_is_apart_from_the_public_one() -> None:
    agent = uniko.Uniko.in_memory_sync().agent("a")
    public = agent.assert_fact_sync("budget", "capped_at", "ten thousand")
    scoped = agent.assert_fact_sync(
        "budget", "capped_at", "ten thousand", visibility="private:cfo"
    )
    assert public.node_id != scoped.node_id
    assert scoped.fact_id.endswith("|private:cfo")
    agent.invalidate_fact_sync(scoped.fact_id)


def test_python_extractor_indexes_an_image() -> None:
    seen: list[uniko.ExtractRequest] = []

    def caption(request: uniko.ExtractRequest) -> str:
        seen.append(request)
        return f"a photo of a red zeppelin ({len(request.data)} bytes)"

    memory = uniko.Uniko.builder().in_memory().extractor("image", caption).build_sync()
    agent = memory.agent("viewer")
    png = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0])
    outcome = agent.session("photos").ingest_sync(uniko.IngestSource.from_bytes(png))
    assert outcome.kind == "artifact" and outcome.chunk_node_ids
    assert seen and seen[0].modality == "image" and seen[0].mime == "image/png"
    view = agent.data.artifact_sync(outcome.artifact_id)
    assert view is not None and "red zeppelin" in view.text


def test_extractor_errors() -> None:
    with pytest.raises(uniko.ConfigError, match="image, audio or video"):
        uniko.Uniko.builder().extractor("smell", lambda r: "")
    with pytest.raises(uniko.ConfigError, match="callable"):
        uniko.Uniko.builder().extractor("image", "not a function")  # ty: ignore[invalid-argument-type]

    def broken(request: uniko.ExtractRequest) -> str:
        raise RuntimeError("model offline")

    memory = uniko.Uniko.builder().in_memory().extractor("image", broken).build_sync()
    png = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    with pytest.raises(uniko.UnikoError, match="model offline"):
        memory.agent("a").session("s").ingest_sync(uniko.IngestSource.from_bytes(png))


def test_learned_procedures_are_matched_and_scored() -> None:
    import datetime

    agent = uniko.Uniko.in_memory_sync().agent("planner")
    base = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)
    for i in range(3):
        for offset, action in ((0, "scan"), (30, "report")):
            agent.record_episode_sync(
                action,
                outcome="success",
                timestamp=base + datetime.timedelta(minutes=5 * i, seconds=offset),
                state={"topic": action},
            )
    assert agent.consolidate_sync().procedures.created >= 1
    matched = agent.match_procedures_sync({"last_action_type": "scan"})
    procedure = next(m for m in matched if m.name == "scan → report")
    assert isinstance(procedure, uniko.MatchedProcedure)
    agent.record_procedure_use_sync(procedure.procedure_id, True)
    assert agent.match_procedures_sync({"last_action_type": "other"}) == [] or all(
        m.name != "scan → report"
        for m in agent.match_procedures_sync({"last_action_type": "other"})
    )
