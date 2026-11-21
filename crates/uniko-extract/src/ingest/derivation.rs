//! Structured text derived from media: segments with their position and
//! provenance, and how the original media is kept.
//!
//! An extractor turns an image, audio or video payload into text. The text
//! alone is enough for search, but it loses where each piece came from: the
//! page, the time range, the region of the image, how confident the model
//! was, and which model produced it. [`Extraction`] carries that, and
//! [`MediaRetention`] says whether the original media is recorded as a
//! source artifact that the derived text can be traced back to.
//!
//! Every field is optional. A callback that returns a plain `String` keeps
//! working exactly as before ([`ExtractOutput::Text`]).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as JsonValue, json};
use uniko_pipes::content::Modality;

use super::chunking::{ChunkConfig, ChunkData, Chunker, count_tokens, text::TextChunker};

/// What a piece of derived text is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SegmentKind {
    /// A description of an image or a video frame.
    Caption,
    /// Speech from an audio track.
    Transcript,
    /// Text recognized in a region of an image.
    OcrRegion,
    /// A block of a page, from document extraction.
    PageBlock,
    /// Any other text.
    Text,
}

impl SegmentKind {
    /// The `:Chunk.chunk_type` a segment of this kind is stored under.
    #[must_use]
    pub fn chunk_type(self) -> &'static str {
        match self {
            Self::Caption => "caption",
            Self::Transcript => "transcript_segment",
            Self::OcrRegion => "ocr_region",
            Self::PageBlock => "page_block",
            Self::Text => "text",
        }
    }

    /// The `DERIVED_FROM.derivation_kind` an artifact made of this kind is
    /// recorded under.
    #[must_use]
    pub fn derivation_kind(self) -> &'static str {
        match self {
            Self::Caption => "caption",
            Self::Transcript => "transcript",
            Self::OcrRegion => "ocr",
            Self::PageBlock => "document_extract",
            Self::Text => "extract",
        }
    }

    /// Parse a kind from its snake_case name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "caption" => Self::Caption,
            "transcript" => Self::Transcript,
            "ocr_region" => Self::OcrRegion,
            "page_block" => Self::PageBlock,
            "text" => Self::Text,
            _ => return None,
        })
    }
}

/// One piece of derived text and where it came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Segment {
    /// The text.
    pub text: String,
    /// What the text is.
    pub kind: SegmentKind,
    /// 1-based page number, for a paged source.
    pub page: Option<u32>,
    /// Start of the segment in milliseconds from the start of the media.
    pub start_ms: Option<u64>,
    /// End of the segment in milliseconds (exclusive).
    pub end_ms: Option<u64>,
    /// Region as `[x0, y0, x1, y1]` in the source image's coordinates.
    pub bbox: Option<[f32; 4]>,
    /// The model's confidence, from 0 to 1.
    pub confidence: Option<f32>,
    /// Speaker label, for speech.
    pub speaker: Option<String>,
}

impl Segment {
    /// A segment of `kind` with no position.
    #[must_use]
    pub fn new(kind: SegmentKind, text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            kind,
            page: None,
            start_ms: None,
            end_ms: None,
            bbox: None,
            confidence: None,
            speaker: None,
        }
    }

    /// Set the page.
    #[must_use]
    pub fn with_page(mut self, page: u32) -> Self {
        self.page = Some(page);
        self
    }

    /// Set the time range, in milliseconds.
    #[must_use]
    pub fn with_time(mut self, start_ms: u64, end_ms: u64) -> Self {
        self.start_ms = Some(start_ms);
        self.end_ms = Some(end_ms);
        self
    }

    /// Set the region.
    #[must_use]
    pub fn with_bbox(mut self, bbox: [f32; 4]) -> Self {
        self.bbox = Some(bbox);
        self
    }

    /// Set the confidence.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Set the speaker.
    #[must_use]
    pub fn with_speaker(mut self, speaker: impl Into<String>) -> Self {
        self.speaker = Some(speaker.into());
        self
    }

    /// Move the segment later by `offset_ms` (a frame or a clip that starts
    /// partway through the media).
    #[must_use]
    pub fn offset_by(mut self, offset_ms: u64) -> Self {
        self.start_ms = self.start_ms.map(|t| t + offset_ms);
        self.end_ms = self.end_ms.map(|t| t + offset_ms);
        self
    }
}

/// The model that derived a set of segments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelProvenance {
    /// The catalog alias the model was called through.
    pub alias: String,
    /// The provider's model id.
    pub model_id: String,
    /// The uni-xervo task: `ocr`, `transcribe`, `generate`, `document_extract`.
    pub task: String,
}

impl ModelProvenance {
    /// Provenance for `alias`, serving `model_id`, for `task`.
    #[must_use]
    pub fn new(
        alias: impl Into<String>,
        model_id: impl Into<String>,
        task: impl Into<String>,
    ) -> Self {
        Self {
            alias: alias.into(),
            model_id: model_id.into(),
            task: task.into(),
        }
    }

    /// The `:Chunk.source_model_version` value: `alias@model_id`.
    #[must_use]
    pub fn version(&self) -> String {
        format!("{}@{}", self.alias, self.model_id)
    }
}

/// Segments derived by one model (or by a callback, with no model).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Derivation {
    /// The segments, in source order.
    pub segments: Vec<Segment>,
    /// The model that produced them, when known.
    pub model: Option<ModelProvenance>,
}

impl Derivation {
    /// Segments with an optional model.
    #[must_use]
    pub fn new(segments: Vec<Segment>, model: Option<ModelProvenance>) -> Self {
        Self { segments, model }
    }
}

/// Structured text extracted from one media payload.
///
/// Several derivations can describe the same media: an image can be
/// captioned by one model and read by an OCR model, and a video combines
/// the captions of its frames with the transcript of its audio.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Extraction {
    /// Each model's output, in the order they ran.
    pub derivations: Vec<Derivation>,
}

impl Extraction {
    /// An extraction of `segments` from `model` (or from a callback).
    #[must_use]
    pub fn new(segments: Vec<Segment>, model: Option<ModelProvenance>) -> Self {
        Self {
            derivations: vec![Derivation::new(segments, model)],
        }
    }

    /// Add another model's output.
    pub fn push(&mut self, derivation: Derivation) {
        self.derivations.push(derivation);
    }

    /// Every non-empty segment.
    pub fn segments(&self) -> impl Iterator<Item = &Segment> {
        self.derivations
            .iter()
            .flat_map(|d| d.segments.iter())
            .filter(|s| !s.text.trim().is_empty())
    }

    /// True when no segment has any text.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments().next().is_none()
    }

    /// The derived text as one document: each segment on its own line,
    /// with a blank line between derivations. This is the artifact's text.
    #[must_use]
    pub fn text(&self) -> String {
        self.layout().0
    }

    /// The `DERIVED_FROM.derivation_kind` for the whole extraction: the
    /// kind every segment shares, or `"multimodal"` when they differ.
    #[must_use]
    pub fn derivation_kind(&self) -> &'static str {
        let mut kinds = self.segments().map(|s| s.kind.derivation_kind());
        match kinds.next() {
            None => "extract",
            Some(first) if kinds.all(|k| k == first) => first,
            Some(_) => "multimodal",
        }
    }

    /// The document text, and each segment's byte range in it, tagged with
    /// the derivation it came from.
    fn layout(&self) -> (String, Vec<(usize, &Segment, usize, usize)>) {
        let mut text = String::new();
        let mut spans = Vec::new();
        for (d, derivation) in self.derivations.iter().enumerate() {
            let segments: Vec<&Segment> = derivation
                .segments
                .iter()
                .filter(|s| !s.text.trim().is_empty())
                .collect();
            if segments.is_empty() {
                continue;
            }
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            for (i, segment) in segments.into_iter().enumerate() {
                if i > 0 {
                    text.push('\n');
                }
                let piece = segment.text.trim();
                let start = text.len();
                text.push_str(piece);
                spans.push((d, segment, start, text.len()));
            }
        }
        (text, spans)
    }

    /// Chunk the extraction for `modality`.
    ///
    /// Consecutive segments from the same derivation, of the same kind, on
    /// the same page and from the same speaker, are grouped up to `config.max_chunk_tokens`, so a
    /// transcript is not stored one sentence per chunk. A segment longer
    /// than that is split with the text chunker, each piece keeping the
    /// segment's position. Each chunk carries:
    ///
    /// - `chunk_type`: the segments' kind;
    /// - `metadata.modality` and, when a model is known,
    ///   `metadata.source_model_version`; the store copies both into their
    ///   own `:Chunk` columns;
    /// - `metadata`: `page_number`, `time_start_ms` / `time_end_ms` (the
    ///   range the group covers), `bbox` (only for a single region),
    ///   `confidence` (the lowest in the group), `speaker` (a group never
    ///   spans two speakers), and for a group of several positioned
    ///   segments, `segments`: each one's `start`/`end` byte range in the
    ///   chunk text with its own time, page, `bbox`, `confidence` and
    ///   `speaker`.
    #[must_use]
    pub fn chunks(&self, modality: Modality, config: &ChunkConfig) -> Vec<ChunkData> {
        let (text, spans) = self.layout();
        let max = config.max_chunk_tokens.max(1);
        let mut out: Vec<ChunkData> = Vec::new();
        let mut group: Vec<(&Segment, usize, usize)> = Vec::new();
        let mut group_derivation = usize::MAX;
        let mut group_tokens = 0usize;

        let flush = |group: &mut Vec<(&Segment, usize, usize)>,
                     derivation: usize,
                     out: &mut Vec<ChunkData>| {
            if group.is_empty() {
                return;
            }
            let model = self.derivations[derivation].model.as_ref();
            let start = group[0].1;
            let end = group[group.len() - 1].2;
            let segs: Vec<&Segment> = group.iter().map(|(s, _, _)| *s).collect();
            let piece = &text[start..end];
            let mut c = chunk(piece, start, end, &segs, modality, model, out.len());
            // A group keeps each segment's own position (time, speaker,
            // region, confidence), with its byte range in the chunk text, so
            // a hit can point at the exact utterance, frame or line.
            if group.len() > 1 && group.iter().any(|(seg, _, _)| has_position(seg)) {
                let segments: Vec<JsonValue> = group
                    .iter()
                    .map(|(seg, s, e)| segment_json(seg, s - start, e - start))
                    .collect();
                if let Some(JsonValue::Object(m)) = c.metadata.as_mut() {
                    m.insert("segments".into(), JsonValue::Array(segments));
                }
            }
            out.push(c);
            group.clear();
        };

        for (d, segment, start, end) in spans {
            let tokens = count_tokens(&text[start..end]);
            if tokens > max {
                flush(&mut group, group_derivation, &mut out);
                group_tokens = 0;
                let model = self.derivations[d].model.as_ref();
                for piece in TextChunker.chunk(&text[start..end], config) {
                    out.push(chunk(
                        &piece.text,
                        start + piece.start,
                        start + piece.end,
                        &[segment],
                        modality,
                        model,
                        out.len(),
                    ));
                }
                continue;
            }
            let same_group = group.last().is_some_and(|(last, _, _)| {
                group_derivation == d
                    && last.kind == segment.kind
                    && last.page == segment.page
                    // A new speaker starts a new chunk, so a chunk's speaker
                    // is exact.
                    && last.speaker == segment.speaker
                    // +1 for the newline that joins it to the group.
                    && group_tokens + tokens < max
            });
            if !same_group {
                flush(&mut group, group_derivation, &mut out);
                group_tokens = 0;
                group_derivation = d;
            }
            group_tokens += tokens + usize::from(!group.is_empty());
            group.push((segment, start, end));
        }
        flush(&mut group, group_derivation, &mut out);
        out
    }
}

/// Whether a segment carries any position worth keeping per segment.
fn has_position(seg: &Segment) -> bool {
    seg.start_ms.is_some()
        || seg.end_ms.is_some()
        || seg.bbox.is_some()
        || seg.confidence.is_some()
        || seg.speaker.is_some()
}

/// One entry of a chunk's `metadata.segments`.
fn segment_json(seg: &Segment, start: usize, end: usize) -> JsonValue {
    let mut r = Map::new();
    r.insert("start".into(), json!(start));
    r.insert("end".into(), json!(end));
    if let Some(p) = seg.page {
        r.insert("page_number".into(), json!(p));
    }
    if let Some(t) = seg.start_ms {
        r.insert("time_start_ms".into(), json!(t));
    }
    if let Some(t) = seg.end_ms {
        r.insert("time_end_ms".into(), json!(t));
    }
    if let Some(b) = seg.bbox {
        r.insert("bbox".into(), json!(b));
    }
    if let Some(c) = seg.confidence {
        r.insert("confidence".into(), json!(c));
    }
    if let Some(sp) = &seg.speaker {
        r.insert("speaker".into(), JsonValue::String(sp.clone()));
    }
    JsonValue::Object(r)
}

/// Build one chunk over `segments`, which all share a kind, a page and a
/// speaker.
fn chunk(
    text: &str,
    start: usize,
    end: usize,
    segments: &[&Segment],
    modality: Modality,
    model: Option<&ModelProvenance>,
    index: usize,
) -> ChunkData {
    let first = segments[0];
    let mut meta = Map::new();
    meta.insert(
        "modality".into(),
        JsonValue::String(modality_name(modality).to_string()),
    );
    if let Some(model) = model {
        meta.insert(
            "source_model_version".into(),
            JsonValue::String(model.version()),
        );
    }
    if let Some(page) = first.page {
        meta.insert("page_number".into(), json!(page));
    }
    if let Some(t) = segments.iter().filter_map(|s| s.start_ms).min() {
        meta.insert("time_start_ms".into(), json!(t));
    }
    if let Some(t) = segments.iter().filter_map(|s| s.end_ms).max() {
        meta.insert("time_end_ms".into(), json!(t));
    }
    if let [only] = segments
        && let Some(bbox) = only.bbox
    {
        meta.insert("bbox".into(), json!(bbox));
    }
    let confidence = segments
        .iter()
        .filter_map(|s| s.confidence)
        .fold(None, |acc: Option<f32>, c| {
            Some(acc.map_or(c, |a| a.min(c)))
        });
    if let Some(c) = confidence {
        meta.insert("confidence".into(), json!(c));
    }
    if let Some(speaker) = first.speaker.as_ref()
        && segments.iter().all(|s| s.speaker.as_ref() == Some(speaker))
    {
        meta.insert("speaker".into(), JsonValue::String(speaker.clone()));
    }
    ChunkData {
        text: text.to_string(),
        index,
        start,
        end,
        token_count: count_tokens(text),
        chunk_type: first.kind.chunk_type().to_string(),
        language: None,
        symbol_name: None,
        heading: None,
        metadata: Some(JsonValue::Object(meta)),
    }
}

/// The name a modality is stored under.
#[must_use]
pub fn modality_name(modality: Modality) -> &'static str {
    match modality {
        Modality::Image => "image",
        Modality::Audio => "audio",
        Modality::Video => "video",
        _ => "media",
    }
}

/// What an extractor returns: plain text, or structured segments.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ExtractOutput {
    /// Plain text, chunked like a document. The behavior before structured
    /// extraction existed.
    Text(String),
    /// Segments with their position and provenance.
    Structured(Extraction),
}

impl From<String> for ExtractOutput {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for ExtractOutput {
    fn from(text: &str) -> Self {
        Self::Text(text.to_string())
    }
}

impl From<Extraction> for ExtractOutput {
    fn from(extraction: Extraction) -> Self {
        Self::Structured(extraction)
    }
}

/// Whether the original media is recorded alongside the text derived from
/// it.
///
/// With anything but [`None`](Self::None), the media becomes a source
/// artifact (`kind = "media"`), and the derived text artifact links to it
/// with `DERIVED_FROM {derivation_kind, derived_at}`. The same bytes always
/// map to the same source artifact.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MediaRetention {
    /// Keep only the derived text. The original bytes are dropped and no
    /// source artifact is recorded.
    #[default]
    None,
    /// Record the media's identity (hash, MIME type, size, path and what
    /// the header says about it) but not its bytes.
    Identity,
    /// Record the identity and store the bytes in the blob store, so they
    /// can be read back.
    Bytes,
}

impl MediaRetention {
    /// Parse a retention from its snake_case name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "none" => Self::None,
            "identity" => Self::Identity,
            "bytes" => Self::Bytes,
            _ => return None,
        })
    }

    /// The snake_case name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Identity => "identity",
            Self::Bytes => "bytes",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max: usize) -> ChunkConfig {
        ChunkConfig {
            max_chunk_tokens: max,
            min_chunk_tokens: 1,
            overlap_tokens: 0,
        }
    }

    fn whisper() -> ModelProvenance {
        ModelProvenance::new("asr/default", "ggml-base", "transcribe")
    }

    #[test]
    fn transcript_segments_group_into_one_chunk_with_the_whole_time_range() {
        let ex = Extraction::new(
            vec![
                Segment::new(SegmentKind::Transcript, "hello there")
                    .with_time(0, 1200)
                    .with_confidence(0.9)
                    .with_speaker("A"),
                Segment::new(SegmentKind::Transcript, "general kenobi")
                    .with_time(1200, 2500)
                    .with_confidence(0.7)
                    .with_speaker("A"),
            ],
            Some(whisper()),
        );
        let chunks = ex.chunks(Modality::Audio, &cfg(512));
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert_eq!(c.chunk_type, "transcript_segment");
        assert_eq!(c.text, "hello there\ngeneral kenobi");
        let m = c.metadata.as_ref().unwrap();
        assert_eq!(m["modality"], "audio");
        assert_eq!(m["source_model_version"], "asr/default@ggml-base");
        assert_eq!(m["time_start_ms"], 0);
        assert_eq!(m["time_end_ms"], 2500);
        assert!((m["confidence"].as_f64().unwrap() - 0.7).abs() < 1e-6);
        assert_eq!(m["speaker"], "A");
        assert!(m.get("bbox").is_none(), "a group has no single region");
        assert_eq!(&ex.text()[c.start..c.end], c.text);
    }

    #[test]
    fn groups_break_on_kind_page_derivation_and_size() {
        let mut ex = Extraction::new(
            vec![
                Segment::new(SegmentKind::PageBlock, "page one").with_page(1),
                Segment::new(SegmentKind::PageBlock, "page two").with_page(2),
            ],
            None,
        );
        ex.push(Derivation::new(
            vec![Segment::new(SegmentKind::OcrRegion, "STOP").with_bbox([1.0, 2.0, 3.0, 4.0])],
            Some(ModelProvenance::new("ocr/default", "ppocr", "ocr")),
        ));
        let chunks = ex.chunks(Modality::Image, &cfg(512));
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].metadata.as_ref().unwrap()["page_number"], 1);
        assert_eq!(chunks[1].metadata.as_ref().unwrap()["page_number"], 2);
        let ocr = chunks[2].metadata.as_ref().unwrap();
        assert_eq!(chunks[2].chunk_type, "ocr_region");
        assert_eq!(ocr["bbox"], json!([1.0, 2.0, 3.0, 4.0]));
        assert!(
            chunks[0]
                .metadata
                .as_ref()
                .unwrap()
                .get("source_model_version")
                .is_none()
        );
        assert_eq!(ex.derivation_kind(), "multimodal");
        let text = ex.text();
        for c in &chunks {
            assert_eq!(&text[c.start..c.end], c.text);
        }
        assert_eq!(
            chunks.iter().map(|c| c.index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );

        let many: Vec<Segment> = (0..10)
            .map(|i| {
                Segment::new(SegmentKind::Transcript, format!("word{i} word word word"))
                    .with_time(i * 10, i * 10 + 10)
            })
            .collect();
        let ex = Extraction::new(many, None);
        let chunks = ex.chunks(Modality::Audio, &cfg(12));
        assert!(chunks.len() > 1, "a small budget splits the transcript");
        for c in &chunks {
            assert!(c.token_count <= 12, "{} tokens", c.token_count);
        }
    }

    #[test]
    fn grouped_ocr_lines_keep_every_region() {
        let ex = Extraction::new(
            vec![
                Segment::new(SegmentKind::OcrRegion, "PUMP STATION")
                    .with_bbox([10.0, 10.0, 200.0, 40.0])
                    .with_confidence(0.95),
                Segment::new(SegmentKind::OcrRegion, "replaced on march 12")
                    .with_bbox([10.0, 50.0, 300.0, 80.0])
                    .with_confidence(0.88),
            ],
            Some(ModelProvenance::new("ocr/default", "ppocr", "ocr")),
        );
        let chunks = ex.chunks(Modality::Image, &cfg(512));
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        let m = c.metadata.as_ref().unwrap();
        assert!(m.get("bbox").is_none());
        let regions = m["segments"].as_array().unwrap();
        assert_eq!(regions.len(), 2);
        for (r, line) in regions.iter().zip(["PUMP STATION", "replaced on march 12"]) {
            let (s, e) = (
                r["start"].as_u64().unwrap() as usize,
                r["end"].as_u64().unwrap() as usize,
            );
            assert_eq!(&c.text[s..e], line);
        }
        assert_eq!(regions[1]["bbox"], json!([10.0, 50.0, 300.0, 80.0]));
    }

    #[test]
    fn a_transcript_keeps_each_utterance_and_breaks_on_a_new_speaker() {
        let ex = Extraction::new(
            vec![
                Segment::new(SegmentKind::Transcript, "budget moves to thursday")
                    .with_time(0, 2400)
                    .with_speaker("alice"),
                Segment::new(SegmentKind::Transcript, "bring the forecast")
                    .with_time(2400, 4100)
                    .with_speaker("alice"),
                Segment::new(SegmentKind::Transcript, "will do")
                    .with_time(4100, 5000)
                    .with_speaker("bob"),
            ],
            None,
        );
        let chunks = ex.chunks(Modality::Audio, &cfg(512));
        assert_eq!(chunks.len(), 2, "bob starts a new chunk");
        let alice = chunks[0].metadata.as_ref().unwrap();
        assert_eq!(alice["speaker"], "alice");
        let segs = alice["segments"].as_array().unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[1]["time_start_ms"], 2400);
        assert_eq!(segs[1]["time_end_ms"], 4100);
        let (s, e) = (
            segs[1]["start"].as_u64().unwrap() as usize,
            segs[1]["end"].as_u64().unwrap() as usize,
        );
        assert_eq!(&chunks[0].text[s..e], "bring the forecast");
        let bob = chunks[1].metadata.as_ref().unwrap();
        assert_eq!(bob["speaker"], "bob");
        assert!(bob.get("segments").is_none(), "one segment needs no list");
    }

    #[test]
    fn segment_offsets_are_byte_ranges_on_char_boundaries() {
        let ex = Extraction::new(
            vec![
                Segment::new(SegmentKind::Transcript, "café crème").with_time(0, 900),
                Segment::new(SegmentKind::Transcript, "東京へ行く").with_time(900, 2000),
            ],
            None,
        );
        let chunks = ex.chunks(Modality::Audio, &cfg(512));
        let c = &chunks[0];
        let segs = c.metadata.as_ref().unwrap()["segments"].as_array().unwrap();
        let texts: Vec<&str> = segs
            .iter()
            .map(|s| {
                let (a, b) = (
                    s["start"].as_u64().unwrap() as usize,
                    s["end"].as_u64().unwrap() as usize,
                );
                &c.text[a..b]
            })
            .collect();
        assert_eq!(texts, vec!["café crème", "東京へ行く"]);
    }

    #[test]
    fn a_group_stays_within_the_budget_counting_its_newlines() {
        let budget = 2 * count_tokens("alpha beta gamma") + 1;
        let segs = (0..6)
            .map(|_| Segment::new(SegmentKind::Transcript, "alpha beta gamma"))
            .collect();
        for c in Extraction::new(segs, None).chunks(Modality::Audio, &cfg(budget)) {
            assert!(c.token_count <= budget, "{} > {budget}", c.token_count);
        }
    }

    #[test]
    fn a_long_segment_is_split_and_keeps_its_position() {
        let long = "lorem ipsum dolor sit amet. ".repeat(80);
        let ex = Extraction::new(
            vec![Segment::new(SegmentKind::Caption, long).with_time(5000, 9000)],
            None,
        );
        let chunks = ex.chunks(Modality::Video, &cfg(64));
        assert!(chunks.len() > 1);
        for c in &chunks {
            let m = c.metadata.as_ref().unwrap();
            assert_eq!(m["time_start_ms"], 5000);
            assert_eq!(m["modality"], "video");
            assert_eq!(c.chunk_type, "caption");
        }
    }

    #[test]
    fn empty_segments_are_skipped() {
        let ex = Extraction::new(
            vec![
                Segment::new(SegmentKind::Caption, "  "),
                Segment::new(SegmentKind::Caption, "a cat"),
            ],
            None,
        );
        assert!(!ex.is_empty());
        assert_eq!(ex.text(), "a cat");
        assert_eq!(ex.derivation_kind(), "caption");
        assert!(Extraction::default().is_empty());
        assert_eq!(
            Segment::new(SegmentKind::Text, "x")
                .with_time(1, 2)
                .offset_by(10)
                .start_ms,
            Some(11)
        );
    }
}
