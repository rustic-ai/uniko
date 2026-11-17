"""Several generation models and the answer alias (issue #58)."""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterator

import pytest

import uniko
from uniko.models.inputs import LlmSpecAdapter


class _Chat(http.server.BaseHTTPRequestHandler):
    reply = ""
    hits = 0

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        length = int(self.headers.get("content-length", 0))
        self.rfile.read(length)
        type(self).hits += 1
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": self.reply}}]}
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _serve(reply: str) -> tuple[str, type[_Chat], http.server.HTTPServer]:
    handler = type("Chat", (_Chat,), {"reply": reply, "hits": 0})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}/v1", handler, server


@pytest.fixture
def two_stubs() -> Iterator[tuple[str, type[_Chat], str, type[_Chat]]]:
    url_a, a, server_a = _serve("from the first model")
    url_b, b, server_b = _serve("from the second model")
    yield url_a, a, url_b, b
    server_a.shutdown()
    server_b.shutdown()


async def test_answer_alias_picks_among_several_llms(two_stubs) -> None:
    url_a, a, url_b, b = two_stubs
    uni = await (
        uniko.Uniko.builder()
        .in_memory()
        .llm(uniko.LlmSpec.llamacpp("generate/a", "stub", url_a))
        .llm(uniko.LlmSpec.llamacpp("generate/b", "stub", url_b))
        .answer_alias("generate/b")
        .build()
    )
    answer = await uni.agent("assistant").answer("what happened?")
    assert answer.text == "from the second model"
    assert (a.hits, b.hits) == (0, 1)


def test_answer_defaults_to_the_first_llm_sync(two_stubs) -> None:
    url_a, a, url_b, b = two_stubs
    uni = (
        uniko.Uniko.builder()
        .in_memory()
        .llm(uniko.LlmSpec.llamacpp("generate/a", "stub", url_a))
        .llm(uniko.LlmSpec.llamacpp("generate/b", "stub", url_b))
        .build_sync()
    )
    assert uni.agent("assistant").answer_sync("q").text == "from the first model"
    assert (a.hits, b.hits) == (1, 0)


def test_llm_spec_new_takes_any_provider() -> None:
    spec = uniko.LlmSpec.new(
        "generate/claude",
        "remote/anthropic",
        "claude-x",
        options={"anthropic_version": "2023-06-01"},
        api_key_env="ANTHROPIC_API_KEY",
    )
    assert spec.alias == "generate/claude"
    assert "ANTHROPIC" not in repr(spec)


@pytest.mark.parametrize(
    ("spec", "needle"),
    [
        (uniko.LlmSpec.new("generate/x", "remote/voyageai", "m"), "remote/voyageai"),
        (
            uniko.LlmSpec.new(
                "generate/x", "remote/mistral", "m", options={"api_key": "sk-1"}
            ),
            "api_key",
        ),
        (
            uniko.LlmSpec.new(
                "generate/x", "remote/anthropic", "m", options={"bad": 1}
            ),
            "bad",
        ),
    ],
)
async def test_build_rejects_an_invalid_llm_spec(spec, needle: str) -> None:
    with pytest.raises(uniko.ConfigError) as info:
        await uniko.Uniko.builder().in_memory().llm(spec).build()
    assert needle in str(info.value)
    assert "sk-1" not in str(info.value)


async def test_build_rejects_an_answer_alias_that_cannot_answer() -> None:
    with pytest.raises(uniko.ConfigError) as info:
        await uniko.Uniko.builder().in_memory().answer_alias("embed/default").build()
    assert "not a generation model" in str(info.value)


def test_llm_spec_model_generic_variant() -> None:
    model = LlmSpecAdapter.validate_python(
        {
            "provider": "generic",
            "alias": "generate/m",
            "provider_id": "remote/mistral",
            "model_id": "mistral-small",
            "api_key_env": "MISTRAL_API_KEY",
        }
    )
    assert model.to_native().alias == "generate/m"
