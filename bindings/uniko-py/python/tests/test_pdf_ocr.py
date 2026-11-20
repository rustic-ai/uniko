"""PDF OCR in the published wheels (issue #55)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import uniko

SCANNED_PDF = (
    Path(__file__).resolve().parents[4]
    / "crates/uniko-extract/tests/fixtures/scanned.pdf"
)


def test_the_wheel_compiles_ocr_in() -> None:
    caps = uniko.Uniko.in_memory_sync().capabilities_sync()
    assert caps["compiled_features"]["pdf-ocr"] is True
    assert caps["ocr_compiled"] is True
    # Compiled in, but still off until the config turns it on.
    assert caps["ocr_configured"] is False


@pytest.mark.skipif(
    os.environ.get("EXPENSIVE_TESTS") != "1",
    reason="set EXPENSIVE_TESTS=1 to download the OCR models",
)
def test_a_scanned_pdf_becomes_searchable_text() -> None:
    # Needs uni-xervo 0.20: 0.19.2 dropped the spaces between words and cut
    # long lines short, so these words were not found.
    uni = (
        uniko.Uniko.builder()
        .config({"ocr": {"enabled": True}, "reranker": {"enabled": False}})
        .build_sync()
    )
    agent = uni.agent("a")
    outcome = agent.session("s").ingest_sync(
        uniko.IngestSource.from_bytes(SCANNED_PDF.read_bytes()).with_mime(
            "application/pdf"
        )
    )
    assert not outcome.extraction_failed, outcome.extraction_error
    assert outcome.page_count == 1
    assert outcome.block_node_ids
    rows = agent.query_sync(
        "MATCH (:Artifact)-[:HAS_PAGE]->(p:Page)-[:CONTAINS]->(b:Block) "
        "RETURN p.page_number AS page, b.produced_by AS tier, b.text AS text"
    )
    assert rows
    assert {(r["page"], r["tier"]) for r in rows} == {(1, "ocr")}
    text = " ".join(r["text"] for r in rows).lower()
    assert "pump" in text and "okafor" in text, text
    bundle = agent.recall_sync("when was the north pump replaced")
    assert any("pump" in item.content.lower() for item in bundle.items)
