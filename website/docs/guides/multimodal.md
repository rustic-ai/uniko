# Images, Audio and Video

Text, code, markup, structured data and PDFs are ingested natively. Images, audio and video are
indexed through the text derived from them: a caption, a transcript, OCR output, a page's blocks.
That text is an ordinary artifact. It's chunked, embedded and recalled like a document, and entity
and observation extraction runs on it.

There are two ways to derive it:

- **A function you supply**, for a model or service uniko doesn't run
  ([`TextExtractor`](#your-own-function)).
- **Models in the catalog**, called by alias
  ([`ModelExtractor`](#models-from-the-catalog)).

Either way, the result can be plain text or structured segments that keep where each piece came
from. The original media can be recorded alongside the text it was derived from, so you can trace
an answer back to it.

## Your own function

```rust
use std::sync::Arc;
use uniko_memory::{Modality, TextExtractor, Uniko};

let captioner = TextExtractor::new(Modality::Image, |req| {
    Ok(my_vision_model.describe(&req.bytes)?)        // -> String
});
let memory = Uniko::builder().extractor(Arc::new(captioner)).build().await?;
```

```python
memory = uniko.Uniko.builder().extractor("image", lambda r: my_model.describe(r.data)).build_sync()
```

The function receives an `ExtractRequest` (`modality`, the bytes, the resolved `mime`, `path` and
the source's `metadata`). It runs on a blocking thread, so it can be a slow model call or a call
back into Python, which holds the GIL only while it runs. An error fails that ingest.

### Structured segments

Return an `Extraction` instead of a string to keep each piece's position and provenance:

```rust
use uniko_memory::{Extraction, ModelProvenance, Segment, SegmentKind, TextExtractor};

let transcriber = TextExtractor::structured(Modality::Audio, |req| {
    let segments = my_asr(&req.bytes)?
        .into_iter()
        .map(|s| Segment::new(SegmentKind::Transcript, s.text)
            .with_time(s.start_ms, s.end_ms)
            .with_confidence(s.confidence)
            .with_speaker(s.speaker))
        .collect();
    Ok(Extraction::new(segments, Some(ModelProvenance::new("asr/host", "whisper-large", "transcribe"))))
});
```

```python
def transcribe(request: uniko.ExtractRequest) -> uniko.Extraction:
    return uniko.Extraction(
        [uniko.Segment("bring the q3 forecast", "transcript", start_ms=2400, end_ms=4100,
                       confidence=0.81, speaker="alice")],
        uniko.ModelProvenance("asr/host", "whisper-large", "transcribe"),
    )
```

A Python function may return either a `str` or an `Extraction`, decided per payload. In Rust,
`TextExtractor::new` takes a function returning `String`, `structured` one returning `Extraction`,
and `with_output` one returning either (`ExtractOutput`).

A segment's `kind` is `caption`, `transcript`, `ocr_region`, `page_block` or `text`. Its `page`,
`start_ms`/`end_ms`, `bbox` (`[x0, y0, x1, y1]` in image coordinates), `confidence` (0 to 1) and
`speaker` are all optional.

How segments become chunks (`Extraction::chunks`,
`crates/uniko-extract/src/ingest/derivation.rs:322`):

- Consecutive segments from the same model, of the same kind, on the same page and from the same
  speaker are grouped up to `max_chunk_tokens`, so a transcript isn't stored one sentence per
  chunk. A new speaker starts a new chunk, and so does a change between a labelled and an unlabelled
  segment: a diarizer that labels only a turn's first utterance, or a fast back-and-forth, gives
  more and smaller chunks. A segment longer than the budget is split, each piece
  keeping the segment's position.
- `chunk_type` is the kind's chunk name: `caption`, `transcript_segment`, `ocr_region`,
  `page_block` or `text`.
- The chunk's `modality` column is `image`, `audio` or `video`. Its `source_model_version` column
  is `alias@model_id` when the model is known
  (`crates/uniko-extract/src/ingest/message.rs:452`).
- The chunk's `metadata` holds:
  - `page_number`;
  - `time_start_ms`/`time_end_ms`, the range the group covers;
  - `bbox`, for a chunk of one region;
  - `confidence`, the lowest in the group;
  - `speaker`, which a group always shares;
  - `segments`, for a group of several positioned segments: each one's `start`/`end` byte range
    in the chunk text, with its own time range, page, `bbox`, `confidence` and `speaker`. So a
    chunk of five utterances or OCR lines still says where each one is.
- The newline joining two grouped segments counts toward the token budget.

### Where a hit came from

A recall or search hit on a chunk derived from media through a `ModelExtractor` or a structured
callback carries a `position`: a `MediaPosition` with the modality, model, page, time range, box,
confidence, speaker and the per-segment list above. It's `None` for every other item, including text
from a plain-string callback, which has no position to give (`RecallItem.position`,
`SearchHit.position`). A segment's `start`/`end` are UTF-8 byte offsets into the chunk text.

```rust
let bundle = agent.recall("when does the budget review move").await?;
for item in &bundle.items {
    if let Some(p) = &item.position {
        // e.g. audio 0..4100 ms, then each utterance in p.segments
        println!("{} {:?}..{:?} {:?}", p.modality, p.time_start_ms, p.time_end_ms, p.speaker);
    }
}
```

```python
item = agent.recall_sync("when does the budget review move").items[0]
if item.position:
    text = item.content.encode()           # offsets are UTF-8 bytes
    for seg in item.position.segments:   # each utterance, frame or line
        print(seg.time_start_ms, seg.speaker, text[seg.start:seg.end].decode())
```

Pair it with [`media_sources`](#keeping-the-original) to find the original file.

An `Extraction` with no text in any segment fails the ingest with `Unsupported` ("no text was
found"): there's nothing to index. That includes a video whose frames and audio yield nothing.
A plain string is indexed as before, even when it's empty.

A Python function can also return a list of `Extraction`s, one per model: a caption and OCR of
the same image, each with its own provenance. They are grouped separately, and the source's
`derivation_kind` is `multimodal`.

## Models from the catalog

`ModelExtractor` calls uni-xervo models by alias. Each mode maps to a model task:

| Mode | Task | Providers that serve it | Chunks |
|---|---|---|---|
| `ocr(alias)` | `ocr` | `local/onnx` (`ocr/default` when `ocr.enabled`) | `ocr_region` chunks of grouped lines, each line's box and confidence in `segments` (or `bbox` for a chunk of one line) |
| `caption(alias)` | `generate`, with the image attached | `remote/openai`, `remote/azure-openai`, `remote/llamacpp` | One `caption` |
| `document(alias)` | `document_extract` | `remote/llamacpp` with a document VLM (olmOCR, MinerU, Granite-Docling) | `page_block` chunks in reading order, each block's box in `segments` (or `bbox` for a chunk of one block) |
| `transcribe(alias)` | `transcribe` | None yet; see below | `transcript_segment` chunks with time ranges and word confidence |

```rust
use uniko_memory::{LlmSpec, MediaRetention, ModelExtractor, Uniko};

let mut config = uniko_memory::UnikoConfig::default();
config.ocr.enabled = true;                       // registers ocr/default
let memory = Uniko::builder()
    .raw_config(config)
    .llm(LlmSpec::llamacpp("vision/qwen", "qwen2.5-vl", "http://localhost:8080/v1"))
    .model_extractor(
        ModelExtractor::image()
            .ocr("ocr/default")
            .caption("vision/qwen")
            .retention(MediaRetention::Bytes),
    )
    .build()
    .await?;
```

```python
memory = (
    uniko.Uniko.builder()
    .config({"ocr": {"enabled": True}})
    .llm(uniko.LlmSpec.llamacpp("vision/qwen", "qwen2.5-vl", "http://localhost:8080/v1"))
    .model_extractor("image", ocr="ocr/default", caption="vision/qwen", retention="bytes")
    .build_sync()
)
```

Images take any mix of `ocr`, `caption` and `document`. Each runs, and its segments are grouped
separately. Audio takes `transcribe`. The caption prompt defaults to `DEFAULT_CAPTION_PROMPT`; set
your own with `caption_prompt`, and the caption's length with `max_caption_tokens` (default 512). A
model call that fails returns `UnikoError::Llm`, naming the alias.

Some details:

- **OCR boxes.** Without a text detector, uni-xervo's recognizer reports the whole image as
  `[0, 0, 1, 1]`, which locates nothing, so uniko drops that box
  (`crates/uniko-extract/src/ingest/model_extractor.rs:714`). `ocr/default` has a detector.
- **Captions.** Only providers that pass an attached image to the model can caption (`IMAGE_CAPABLE_PROVIDERS`,
  `model_extractor.rs:58`). Gemini, Vertex AI, Anthropic, Cohere and Mistral drop images without
  saying so, so a caption alias on one of them is refused at build, rather than captioning
  nothing.
- **Audio** must be PCM WAV: 16-bit, 16 kHz, mono or stereo. uni-xervo's whisper.cpp provider
  accepts nothing else, and uniko doesn't decode or resample. Anything else is refused with
  `Unsupported` and the conversion to run (`check_wav`, `model_extractor.rs:797`):

  ```sh
  ffmpeg -i in.mp3 -ar 16000 -ac 1 -c:a pcm_s16le out.wav
  ```

- **No transcription provider yet.** uni-xervo's only transcriber is `local/whisper-cpp`, and
  uni-db builds the store's model runtime without registering it, so an alias on it can't load.
  `build()` refuses one with `Unsupported` (`WHISPER_CPP_PROVIDER`, `model_extractor.rs:55`).
  Until uni-db registers it, transcribe with your own function and return
  `transcript` segments through `TextExtractor::structured`, as shown above.

### Video

uni-xervo has no video task, so a video extractor takes a **composer**: your function that splits
a video into sampled frames and an audio track, usually by running `ffmpeg`. The image modes then
run on every frame, and `transcribe` on the track. Frames are sorted by time. Each frame's
segments cover the time from that frame to the next, and the last frame's cover only its own
instant (`time_start_ms` equals `time_end_ms`).

```rust
use uniko_memory::{ModelExtractor, VideoFrame, VideoParts};

let video = ModelExtractor::video(|req| {
    let frames = sample_frames_with_ffmpeg(&req.bytes, 5_000)?   // a frame every 5 s
        .into_iter()
        .map(|(ms, jpeg)| VideoFrame::new(ms, jpeg, "image/jpeg"))
        .collect();
    Ok(VideoParts::new(frames, None))       // no audio: no transcription provider yet
})
.caption("vision/qwen")
.ocr("ocr/default");
```

Once a transcription provider is available, return the audio track as 16 kHz mono PCM WAV
(`VideoParts::new(frames, Some(wav))`) and add `.transcribe(alias)`.

```python
def compose(request: uniko.ExtractRequest) -> uniko.VideoParts:
    frames = [uniko.VideoFrame(ms, jpeg, "image/jpeg") for ms, jpeg in sample(request.data)]
    return uniko.VideoParts(frames)

builder.model_extractor("video", caption="vision/qwen", ocr="ocr/default", composer=compose)
```

The composer runs on a blocking thread. Its output is checked against the modes: frames with no
image mode set to read them are refused with `Unsupported`, and no audio track while `transcribe` is
set is reported as a `media_extraction` / `no_audio_track` diagnostic, with ingest going on (no
transcription provider exists yet, so this can't happen through `build()` today). An audio track
returned with no `transcribe` set is ignored. A typical recipe:

```sh
ffmpeg -i in.mp4 -vf fps=1/5 -q:v 3 frame-%04d.jpg          # a frame every 5 seconds
ffmpeg -i in.mp4 -vn -ar 16000 -ac 1 -c:a pcm_s16le audio.wav
```

### What is refused at build

`build()` checks every model extractor against the catalog before the store opens
(`ModelExtractor::validate`, `model_extractor.rs:328`, called from
`crates/uniko-memory/src/facade/uniko.rs:758`):

| Configuration | Error |
|---|---|
| An image extractor with `transcribe`; an audio one with `ocr`, `caption` or `document` | `Unsupported` |
| A video extractor without a composer; a composer on any other modality | `Unsupported` |
| A caption alias whose provider drops images | `Unsupported` |
| A `local/whisper-cpp` transcribe alias, which the runtime doesn't register | `Unsupported` |
| An extractor with no mode | `Config` |
| An alias that isn't in the catalog, or serves a different task | `Config` |

With `.require([Requirement::Extractor(modality)])`, readiness also loads the extractor's aliases
and fails if one doesn't load (`crates/uniko-memory/src/facade/readiness.rs:234`).

## Keeping the original

Both kinds of extractor take a retention (`with_retention` / `retention(..)`; in Python
`retention=`):

| Retention | What is kept |
|---|---|
| `None` (default) | The derived text only, as before. |
| `Identity` | A **media source** artifact (`kind = "media"`) with the original's SHA-256, MIME type, size, the path of the first ingest that recorded it, and for WAV its duration, sample rate and channels. No bytes. |
| `Bytes` | The same, with the original bytes in the blob store (`HAS_CONTENT`). |

The derived text artifact links to its source with
`DERIVED_FROM {derivation_kind, derived_at}`. `derivation_kind` is `ocr`, `caption`,
`transcript`, `document_extract`, `multimodal` (segments of several kinds) or `extract` (a
function's plain text). The same bytes always map to one source, whichever artifacts are derived
from it (`merge_media_source_in_tx`, `crates/uniko-store/src/repository/media.rs:161`).

With a retention, the derived artifact is identified by its media, not by its text:

- Without a caller id, its id is `<modality>-<sha256 of the media>`. Re-ingesting the same bytes
  finds it, and two different images that happen to get the same caption stay two artifacts.
- Re-ingesting the same media under the same id is a replay. It keeps the first text, even if
  the model words it differently, and attaches the bytes if this run retains them. The same id
  over different media is an `IdConflict`.
- A plain document never deduplicates onto text derived from media, even when the two read the
  same.

Trace a recall hit or an ingest result back to its source with `data().media_sources`:

```rust
for source in agent.data().media_sources(&hit_artifact_id).await? {
    println!("{} {:?} {}", source.artifact_id, source.mime, source.derivation_kind.unwrap_or_default());
    if source.bytes_retained {
        let original = agent.data().artifact_bytes(&source.artifact_id).await?;
    }
}
```

```python
for source in agent.data.media_sources_sync(artifact_id):
    if source.bytes_retained:
        original = agent.data.artifact_bytes_sync(source.artifact_id)
```

Visibility and deletion follow the derived text:

- A source is readable only when every artifact derived from it is wholly readable to the viewer.
  Once any of that text is forgotten or hidden, `artifact`, `artifact_bytes` and `media_sources`
  return nothing for the source, since its bytes would reveal what the text withholds
  (`crates/uniko-memory/src/facade/data.rs:237`).
- Deleting the last artifact derived from a source deletes the source and its stored bytes too
  (`orphaned_media_ids`, `crates/uniko-store/src/storage/deletion.rs:561`). While another derived
  artifact remains, the source stays. This applies to `delete_document`, to deleting the session
  the media was ingested into, and to deleting the turn it was attached to (including forgetting
  the participant who sent it). A turn takes its media-derived text with it unless that text is
  also attached to another message or another session; the same bytes ingested elsewhere replay
  onto the same artifact, so they keep it alive.
