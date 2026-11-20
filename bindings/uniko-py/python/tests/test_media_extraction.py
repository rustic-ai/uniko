"""Structured media extraction, retention and model-backed extractors (#61)."""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

import uniko
from uniko.models import MediaSourceView, TypedUniko

PNG = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0])


class _Chat(http.server.BaseHTTPRequestHandler):
    reply = ""
    bodies: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        length = int(self.headers.get("content-length", 0))
        type(self).bodies.append(self.rfile.read(length).decode())
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


@pytest.fixture
def vision() -> Iterator[tuple[str, type[_Chat]]]:
    handler = type("Chat", (_Chat,), {"reply": "a lighthouse at dusk", "bodies": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", handler
    server.shutdown()


def test_a_structured_extraction_is_indexed_and_traced_to_its_source() -> None:
    def transcribe(request: uniko.ExtractRequest) -> uniko.Extraction:
        return uniko.Extraction(
            [
                uniko.Segment(
                    "the budget review moves to thursday",
                    "transcript",
                    start_ms=0,
                    end_ms=2400,
                    confidence=0.9,
                    speaker="alice",
                ),
                uniko.Segment(
                    "bring the q3 forecast", "transcript", start_ms=2400, end_ms=4100
                ),
            ],
            uniko.ModelProvenance("asr/host", "whisper-large", "transcribe"),
        )

    memory = (
        uniko.Uniko.builder()
        .in_memory()
        .extractor("audio", transcribe, retention="bytes")
        .build_sync()
    )
    agent = memory.agent("viewer")
    wav = b"RIFF" + b"\x00" * 40
    outcome = agent.session("memo").ingest_sync(
        uniko.IngestSource.from_bytes(wav).with_mime("audio/wav").with_id("memo-1")
    )
    assert outcome.artifact_id == "memo-1"
    view = agent.data.artifact_sync("memo-1")
    assert view is not None and "q3 forecast" in view.text

    sources = agent.data.media_sources_sync("memo-1")
    assert len(sources) == 1
    source = sources[0]
    assert source.modality == "audio"
    assert source.hash == hashlib.sha256(wav).hexdigest()
    assert source.derivation_kind == "transcript"
    assert source.bytes_retained
    assert agent.data.artifact_bytes_sync(source.artifact_id) == wav


def test_identity_retention_keeps_no_bytes_and_dedups() -> None:
    memory = (
        uniko.Uniko.builder()
        .in_memory()
        .extractor("image", lambda r: f"photo {r.path}", retention="identity")
        .build_sync()
    )
    agent = memory.agent("viewer")
    session = agent.session("photos")
    for id, path in [("p1", "a.png"), ("p2", "b.png")]:
        session.ingest_sync(
            uniko.IngestSource.from_bytes(PNG).with_path(path).with_id(id)
        )
    s1 = agent.data.media_sources_sync("p1")
    s2 = agent.data.media_sources_sync("p2")
    assert [s.artifact_id for s in s1] == [s.artifact_id for s in s2]
    assert not s1[0].bytes_retained
    assert agent.data.artifact_bytes_sync(s1[0].artifact_id) is None


def test_plain_text_extractors_record_no_source() -> None:
    memory = (
        uniko.Uniko.builder()
        .in_memory()
        .extractor("image", lambda r: "a cat")
        .build_sync()
    )
    agent = memory.agent("viewer")
    outcome = agent.session("s").ingest_sync(uniko.IngestSource.from_bytes(PNG))
    assert agent.data.media_sources_sync(outcome.artifact_id) == []


def test_typed_layer_returns_media_source_models() -> None:
    memory = TypedUniko(
        uniko.Uniko.builder()
        .in_memory()
        .extractor("image", lambda r: "a cat", retention="identity")
        .build_sync()
    )
    agent = memory.agent("viewer")
    agent.session("s").ingest_sync(uniko.IngestSource.from_bytes(PNG).with_id("cat"))
    sources = agent.data.media_sources_sync("cat")
    assert len(sources) == 1 and isinstance(sources[0], MediaSourceView)
    assert sources[0].mime == "image/png"


def test_bad_extractor_results_and_arguments() -> None:
    with pytest.raises(uniko.ConfigError, match="segment kind"):
        uniko.Segment("x", "smell")
    with pytest.raises(uniko.ConfigError, match="before it starts"):
        uniko.Segment("x", "transcript", start_ms=10, end_ms=5)
    with pytest.raises(uniko.ConfigError, match="retention"):
        uniko.Uniko.builder().extractor("image", lambda r: "x", retention="forever")

    wrong: Any = lambda r: 42  # noqa: E731 - returns neither a str nor an Extraction
    memory = uniko.Uniko.builder().in_memory().extractor("image", wrong).build_sync()
    with pytest.raises(uniko.UnikoError, match="a str, an Extraction or a list"):
        memory.agent("v").session("s").ingest_sync(uniko.IngestSource.from_bytes(PNG))


def test_unsupported_model_extractors_fail_at_build() -> None:
    with pytest.raises(uniko.UnikoError, match="no video task"):
        uniko.Uniko.builder().in_memory().model_extractor(
            "video", ocr="ocr/default"
        ).build_sync()
    with pytest.raises(uniko.UnikoError, match="cannot be transcribed"):
        uniko.Uniko.builder().in_memory().model_extractor(
            "image", transcribe="asr/default"
        ).build_sync()
    with pytest.raises(uniko.UnikoError, match="not in the model catalog"):
        uniko.Uniko.builder().in_memory().model_extractor(
            "image", ocr="ocr/nowhere"
        ).build_sync()
    with pytest.raises(uniko.UnikoError, match="does not pass images"):
        (
            uniko.Uniko.builder()
            .in_memory()
            .llm(
                uniko.LlmSpec.new("vision/gemini", "remote/gemini", "gemini-2.5-flash")
            )
            .model_extractor("image", caption="vision/gemini")
            .build_sync()
        )
    with pytest.raises(uniko.UnikoError, match="video only"):
        uniko.Uniko.builder().in_memory().model_extractor(
            "image", ocr="ocr/default", composer=lambda r: uniko.VideoParts([])
        ).build_sync()


def test_a_caption_model_receives_the_image(vision) -> None:
    url, chat = vision
    memory = (
        uniko.Uniko.builder()
        .in_memory()
        .llm(uniko.LlmSpec.llamacpp("vision/stub", "stub-vlm", url))
        .model_extractor(
            "image",
            caption="vision/stub",
            caption_prompt="Describe it.",
            retention="identity",
        )
        .build_sync()
    )
    agent = memory.agent("viewer")
    agent.session("photos").ingest_sync(
        uniko.IngestSource.from_bytes(PNG).with_id("coast")
    )
    assert len(chat.bodies) == 1
    assert "data:image/png;base64," in chat.bodies[0]
    assert "Describe it." in chat.bodies[0]
    view = agent.data.artifact_sync("coast")
    assert view is not None and "lighthouse" in view.text
    assert agent.data.media_sources_sync("coast")[0].derivation_kind == "caption"


def test_a_video_composer_feeds_every_frame_to_the_captioner(vision) -> None:
    url, chat = vision
    calls: list[str] = []

    def compose(request: uniko.ExtractRequest) -> uniko.VideoParts:
        calls.append(request.modality)
        return uniko.VideoParts(
            [
                uniko.VideoFrame(0, PNG, "image/png"),
                uniko.VideoFrame(5000, PNG, "image/png"),
            ]
        )

    memory = (
        uniko.Uniko.builder()
        .in_memory()
        .llm(uniko.LlmSpec.llamacpp("vision/stub", "stub-vlm", url))
        .model_extractor("video", caption="vision/stub", composer=compose)
        .build_sync()
    )
    agent = memory.agent("viewer")
    agent.session("standup").ingest_sync(
        uniko.IngestSource.from_bytes(b"\x00" * 32).with_mime("video/mp4").with_id("v")
    )
    assert calls == ["video"]
    assert len(chat.bodies) == 2
    view = agent.data.artifact_sync("v")
    assert view is not None and "lighthouse" in view.text


def test_several_models_output_is_kept_apart() -> None:
    def both(request: uniko.ExtractRequest) -> list[uniko.Extraction]:
        return [
            uniko.Extraction(
                [uniko.Segment("a stop sign at a crossing", "caption")],
                uniko.ModelProvenance("vision/host", "vlm", "generate"),
            ),
            uniko.Extraction(
                [uniko.Segment("STOP", "ocr_region", bbox=(1.0, 2.0, 30.0, 12.0))],
                uniko.ModelProvenance("ocr/host", "ppocr", "ocr"),
            ),
        ]

    memory = (
        uniko.Uniko.builder()
        .in_memory()
        .extractor("image", both, retention="identity")
        .build_sync()
    )
    agent = memory.agent("viewer")
    agent.session("s").ingest_sync(uniko.IngestSource.from_bytes(PNG).with_id("sign"))
    view = agent.data.artifact_sync("sign")
    assert view is not None and "stop sign" in view.text and "STOP" in view.text
    assert agent.data.media_sources_sync("sign")[0].derivation_kind == "multimodal"


def test_max_caption_tokens_reaches_the_request(vision) -> None:
    url, chat = vision
    memory = (
        uniko.Uniko.builder()
        .in_memory()
        .llm(uniko.LlmSpec.llamacpp("vision/stub", "stub-vlm", url))
        .model_extractor("image", caption="vision/stub", max_caption_tokens=77)
        .build_sync()
    )
    memory.agent("v").session("s").ingest_sync(uniko.IngestSource.from_bytes(PNG))
    assert '"max_tokens":77' in chat.bodies[0].replace(" ", "")


def test_transcription_has_no_usable_provider_yet() -> None:
    with pytest.raises(
        uniko.UnikoError, match="cannot be transcribed|not in the model catalog"
    ):
        uniko.Uniko.builder().in_memory().model_extractor(
            "audio", transcribe="asr/default"
        ).build_sync()
