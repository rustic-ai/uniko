// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! Media extraction types (issue #61): the structured result an extractor
//! may return ([`PySegment`], [`PyExtraction`], [`PyModelProvenance`]), the
//! frames a video composer returns ([`PyVideoFrame`], [`PyVideoParts`]), and
//! the recorded original ([`PyMediaSourceView`]).

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use uniko_api::tools::{
    Extraction, MediaRetention, MediaSourceView, ModelProvenance, Segment, SegmentKind, UnikoError,
    VideoFrame, VideoParts,
};

use crate::errors::to_pyerr;

/// Parse a retention name: `"none"`, `"identity"` or `"bytes"`.
pub(crate) fn parse_retention(name: &str) -> PyResult<MediaRetention> {
    MediaRetention::parse(name).ok_or_else(|| {
        to_pyerr(UnikoError::Config(format!(
            "retention '{name}': expected none, identity or bytes"
        )))
    })
}

/// One piece of text derived from media, and where it came from.
///
/// `kind` is `"caption"`, `"transcript"`, `"ocr_region"`, `"page_block"` or
/// `"text"`. Every position field is optional.
#[pyclass(name = "Segment", module = "uniko", frozen)]
pub struct PySegment {
    /// The text.
    #[pyo3(get)]
    text: String,
    /// What the text is.
    #[pyo3(get)]
    kind: String,
    /// 1-based page number.
    #[pyo3(get)]
    page: Option<u32>,
    /// Start, in milliseconds from the start of the media.
    #[pyo3(get)]
    start_ms: Option<u64>,
    /// End, in milliseconds (exclusive).
    #[pyo3(get)]
    end_ms: Option<u64>,
    /// Region as `(x0, y0, x1, y1)` in image coordinates.
    #[pyo3(get)]
    bbox: Option<(f32, f32, f32, f32)>,
    /// The model's confidence, from 0 to 1.
    #[pyo3(get)]
    confidence: Option<f32>,
    /// Speaker label.
    #[pyo3(get)]
    speaker: Option<String>,
}

#[pymethods]
impl PySegment {
    #[new]
    #[pyo3(signature = (text, kind = "text", *, page = None, start_ms = None, end_ms = None, bbox = None, confidence = None, speaker = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        text: String,
        kind: &str,
        page: Option<u32>,
        start_ms: Option<u64>,
        end_ms: Option<u64>,
        bbox: Option<(f32, f32, f32, f32)>,
        confidence: Option<f32>,
        speaker: Option<String>,
    ) -> PyResult<Self> {
        if SegmentKind::parse(kind).is_none() {
            return Err(to_pyerr(UnikoError::Config(format!(
                "segment kind '{kind}': expected caption, transcript, ocr_region, page_block \
                 or text"
            ))));
        }
        if let (Some(s), Some(e)) = (start_ms, end_ms)
            && e < s
        {
            return Err(to_pyerr(UnikoError::Config(format!(
                "segment ends ({e} ms) before it starts ({s} ms)"
            ))));
        }
        Ok(Self {
            text,
            kind: kind.to_string(),
            page,
            start_ms,
            end_ms,
            bbox,
            confidence,
            speaker,
        })
    }

    fn __repr__(&self) -> String {
        format!("Segment(kind='{}', text={:?})", self.kind, self.text)
    }
}

impl PySegment {
    fn to_rust(&self) -> Segment {
        let kind = SegmentKind::parse(&self.kind).unwrap_or(SegmentKind::Text);
        let mut s = Segment::new(kind, self.text.clone());
        s.page = self.page;
        s.start_ms = self.start_ms;
        s.end_ms = self.end_ms;
        s.bbox = self.bbox.map(|(a, b, c, d)| [a, b, c, d]);
        s.confidence = self.confidence;
        s.speaker = self.speaker.clone();
        s
    }
}

/// The model that derived a set of segments.
#[pyclass(name = "ModelProvenance", module = "uniko", frozen)]
pub struct PyModelProvenance {
    /// The catalog alias (or any name the host uses for its model).
    #[pyo3(get)]
    alias: String,
    /// The model id.
    #[pyo3(get)]
    model_id: String,
    /// The task: `ocr`, `transcribe`, `generate`, `document_extract`, ...
    #[pyo3(get)]
    task: String,
}

#[pymethods]
impl PyModelProvenance {
    #[new]
    fn new(alias: String, model_id: String, task: String) -> Self {
        Self {
            alias,
            model_id,
            task,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelProvenance(alias='{}', model_id='{}', task='{}')",
            self.alias, self.model_id, self.task
        )
    }
}

/// Structured text an extractor returns instead of a `str`: segments with
/// their position, and the model that produced them.
#[pyclass(name = "Extraction", module = "uniko", frozen)]
pub struct PyExtraction {
    /// The segments, in source order.
    #[pyo3(get)]
    segments: Vec<Py<PySegment>>,
    /// The model, when known.
    #[pyo3(get)]
    model: Option<Py<PyModelProvenance>>,
}

#[pymethods]
impl PyExtraction {
    #[new]
    #[pyo3(signature = (segments, model = None))]
    fn new(segments: Vec<Py<PySegment>>, model: Option<Py<PyModelProvenance>>) -> Self {
        Self { segments, model }
    }

    fn __repr__(&self) -> String {
        format!("Extraction(segments={})", self.segments.len())
    }
}

impl PyExtraction {
    /// The Rust extraction.
    pub(crate) fn to_rust(&self, py: Python<'_>) -> Extraction {
        let segments = self
            .segments
            .iter()
            .map(|s| s.borrow(py).to_rust())
            .collect();
        let model = self.model.as_ref().map(|m| {
            let m = m.borrow(py);
            ModelProvenance::new(m.alias.clone(), m.model_id.clone(), m.task.clone())
        });
        Extraction::new(segments, model)
    }
}

/// One frame of a video, as a composer returns it.
#[pyclass(name = "VideoFrame", module = "uniko", frozen)]
pub struct PyVideoFrame {
    /// Where the frame sits in the video, in milliseconds.
    #[pyo3(get)]
    timestamp_ms: u64,
    /// The encoded image.
    #[pyo3(get)]
    data: Py<PyBytes>,
    /// Its MIME type, e.g. `image/jpeg`.
    #[pyo3(get)]
    mime: String,
}

#[pymethods]
impl PyVideoFrame {
    #[new]
    fn new(timestamp_ms: u64, data: Py<PyBytes>, mime: String) -> Self {
        Self {
            timestamp_ms,
            data,
            mime,
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "VideoFrame(timestamp_ms={}, mime='{}', bytes={})",
            self.timestamp_ms,
            self.mime,
            self.data.bind(py).as_bytes().len()
        )
    }
}

/// What a video composer splits a video into: sampled frames, and the audio
/// track as 16 kHz, 16-bit PCM WAV.
#[pyclass(name = "VideoParts", module = "uniko", frozen)]
pub struct PyVideoParts {
    /// The frames.
    #[pyo3(get)]
    frames: Vec<Py<PyVideoFrame>>,
    /// The audio track, as WAV bytes.
    #[pyo3(get)]
    audio: Option<Py<PyBytes>>,
}

#[pymethods]
impl PyVideoParts {
    #[new]
    #[pyo3(signature = (frames, audio = None))]
    fn new(frames: Vec<Py<PyVideoFrame>>, audio: Option<Py<PyBytes>>) -> Self {
        Self { frames, audio }
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoParts(frames={}, audio={})",
            self.frames.len(),
            self.audio.is_some()
        )
    }
}

impl PyVideoParts {
    /// The Rust parts.
    pub(crate) fn to_rust(&self, py: Python<'_>) -> VideoParts {
        let frames = self
            .frames
            .iter()
            .map(|f| {
                let f = f.borrow(py);
                VideoFrame::new(
                    f.timestamp_ms,
                    f.data.bind(py).as_bytes().to_vec(),
                    f.mime.clone(),
                )
            })
            .collect();
        let audio = self.audio.as_ref().map(|a| a.bind(py).as_bytes().to_vec());
        VideoParts::new(frames, audio)
    }
}

/// The original media a text artifact was derived from.
#[pyclass(name = "MediaSourceView", module = "uniko", frozen)]
pub struct PyMediaSourceView {
    /// The source artifact's id. With `bytes_retained`, pass it to
    /// `Data.artifact_bytes` for the original bytes.
    #[pyo3(get)]
    artifact_id: String,
    /// `image`, `audio` or `video`.
    #[pyo3(get)]
    modality: Option<String>,
    /// MIME type.
    #[pyo3(get)]
    mime: Option<String>,
    /// SHA-256 hex of the original bytes.
    #[pyo3(get)]
    hash: String,
    /// Byte length.
    #[pyo3(get)]
    size: Option<i64>,
    /// Source path, when known.
    #[pyo3(get)]
    path: Option<String>,
    /// Audio duration in milliseconds, from a WAV header.
    #[pyo3(get)]
    duration_ms: Option<i64>,
    /// Audio sample rate in Hz.
    #[pyo3(get)]
    sample_rate: Option<i32>,
    /// Audio channel count.
    #[pyo3(get)]
    channels: Option<i32>,
    /// How the text was derived: `ocr`, `caption`, `transcript`,
    /// `document_extract`, `multimodal` or `extract`.
    #[pyo3(get)]
    derivation_kind: Option<String>,
    /// Whether the original bytes are stored.
    #[pyo3(get)]
    bytes_retained: bool,
}

impl PyMediaSourceView {
    /// Build the Python wrapper from a Rust [`MediaSourceView`].
    pub fn from_rust(py: Python<'_>, v: &MediaSourceView) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                artifact_id: v.artifact_id.clone(),
                modality: v.modality.clone(),
                mime: v.mime.clone(),
                hash: v.hash.clone(),
                size: v.size,
                path: v.path.clone(),
                duration_ms: v.duration_ms,
                sample_rate: v.sample_rate,
                channels: v.channels,
                derivation_kind: v.derivation_kind.clone(),
                bytes_retained: v.bytes_retained,
            },
        )
    }
}

#[pymethods]
impl PyMediaSourceView {
    fn __repr__(&self) -> String {
        format!(
            "MediaSourceView(artifact_id='{}', modality={:?}, bytes_retained={})",
            self.artifact_id, self.modality, self.bytes_retained
        )
    }
}
