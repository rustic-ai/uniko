"""A store kept in S3 through ``remote_storage``, from Python.

Skipped unless ``UNIKO_S3_ENDPOINT`` is set; start a local S3 with
``scripts/s3/start.sh localstack`` (see ``crates/uniko-memory/tests/cloud_e2e.rs``
for the full Rust lifecycle test).
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

import pytest

import uniko

ENDPOINT = os.environ.get("UNIKO_S3_ENDPOINT")

pytestmark = pytest.mark.skipif(
    not ENDPOINT or bool(os.environ.get("UNIKO_S3_CA")),
    reason="needs UNIKO_S3_ENDPOINT, over plain HTTP (scripts/s3/start.sh localstack)",
)


def _bucket() -> str:
    name = f"uniko-py-{uuid.uuid4().hex[:12]}"
    status = subprocess.run(
        [
            "curl",
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "-X",
            "PUT",
            "--aws-sigv4",
            "aws:amz:us-east-1:s3",
            "--user",
            "test:test",
            f"{ENDPOINT}/{name}",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert status.startswith("2"), status
    return name


def _config(bucket: str) -> dict:
    return {
        "remote_storage": {
            "url": f"s3://{bucket}/kb",
            "endpoint": ENDPOINT,
            "region": os.environ.get("AWS_REGION", "us-east-1"),
            "access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", "test"),
            "secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        }
    }


def _open(path: Path, config: dict) -> uniko.Uniko:
    return uniko.Uniko.builder().path(str(path)).config(config).build_sync()


def test_a_store_on_s3_survives_a_reopen_and_a_move(tmp_path: Path) -> None:
    config = _config(_bucket())
    memory = _open(tmp_path / "host-a", config)

    shown = memory.config()["remote_storage"]
    assert shown["secret_access_key"] == "<redacted>"
    assert shown["access_key_id"] == config["remote_storage"]["access_key_id"]

    with pytest.raises(uniko.ConfigError, match="open for writing elsewhere"):
        _open(tmp_path / "host-b", config)

    session = memory.agent("assistant").session("chat")
    session.observe_sync(uniko.Turn("alice", "My favourite city is Kyoto."))
    del session
    memory.shutdown_sync()

    # The same host, then another one: a fresh local directory, with
    # everything else read from the bucket.
    for host in ("host-a", "host-b"):
        reopened = _open(tmp_path / host, config)
        agent = reopened.agent("assistant")
        bundle = agent.recall_sync("favourite city")
        assert any("Kyoto" in item.content for item in bundle.items), (host, bundle)
        del agent
        reopened.shutdown_sync()
