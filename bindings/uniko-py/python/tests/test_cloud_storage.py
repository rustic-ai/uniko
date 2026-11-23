"""A store kept in an object store through ``remote_storage``, from Python.

Runs once per configured provider, and is skipped when none is:

- S3, when ``UNIKO_S3_ENDPOINT`` is set; start a local S3 with
  ``scripts/s3/start.sh localstack``;
- Azure Blob Storage, when ``UNIKO_AZURE_ACCOUNT``, ``UNIKO_AZURE_CONTAINER``
  and ``UNIKO_AZURE_SAS`` are set. The store opens with the SAS token, which
  needs read, add, create, write, delete and list rights on the container.

See ``crates/uniko-memory/tests/cloud_e2e.rs`` for the full Rust lifecycle
test.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

import pytest

import uniko

ENDPOINT = os.environ.get("UNIKO_S3_ENDPOINT")
AZURE_ACCOUNT = os.environ.get("UNIKO_AZURE_ACCOUNT")
AZURE_CONTAINER = os.environ.get("UNIKO_AZURE_CONTAINER")
AZURE_SAS = os.environ.get("UNIKO_AZURE_SAS")

PROVIDERS = []
if ENDPOINT and not os.environ.get("UNIKO_S3_CA"):
    PROVIDERS.append("s3")
if AZURE_ACCOUNT and AZURE_CONTAINER and AZURE_SAS:
    PROVIDERS.append("azure")

pytestmark = pytest.mark.skipif(
    not PROVIDERS,
    reason="needs UNIKO_S3_ENDPOINT over plain HTTP (scripts/s3/start.sh localstack), "
    "or UNIKO_AZURE_ACCOUNT, UNIKO_AZURE_CONTAINER and UNIKO_AZURE_SAS",
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


def _s3_config(bucket: str) -> dict:
    return {
        "remote_storage": {
            "url": f"s3://{bucket}/kb",
            "endpoint": ENDPOINT,
            "region": os.environ.get("AWS_REGION", "us-east-1"),
            "access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", "test"),
            "secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        }
    }


def _azure_config() -> dict:
    return {
        "remote_storage": {
            "url": f"az://{AZURE_CONTAINER}/uniko-py-{uuid.uuid4().hex[:12]}/kb",
            "account": AZURE_ACCOUNT,
            "sas_token": AZURE_SAS,
        }
    }


@pytest.fixture(params=PROVIDERS or ["none"])
def provider(request: pytest.FixtureRequest) -> tuple[dict, str]:
    """The config for a fresh store, and the secret its read-back hides."""
    if request.param == "s3":
        return _s3_config(_bucket()), "secret_access_key"
    return _azure_config(), "sas_token"


def _open(path: Path, config: dict) -> uniko.Uniko:
    return uniko.Uniko.builder().path(str(path)).config(config).build_sync()


def test_a_store_in_a_bucket_survives_a_reopen_and_a_move(
    tmp_path: Path, provider: tuple[dict, str]
) -> None:
    config, secret = provider
    memory = _open(tmp_path / "host-a", config)

    shown = memory.config()["remote_storage"]
    assert shown[secret] == "<redacted>"
    assert shown["url"] == config["remote_storage"]["url"]

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
