//! [`ModelExtractor`]: text from images, audio and video through the model
//! runtime, chosen by catalog alias.
//!
//! uni-xervo serves four tasks that turn media into text, and each maps to
//! a mode here:
//!
//! | Mode | Task | Providers that serve it |
//! |---|---|---|
//! | [`ocr`](ModelExtractor::ocr) | `ocr` | `local/onnx` |
//! | [`caption`](ModelExtractor::caption) | `generate`, with the image attached | `remote/openai`, `remote/azure-openai`, `remote/llamacpp` |
//! | [`document`](ModelExtractor::document) | `document_extract` | `remote/llamacpp` with a document VLM |
//! | [`transcribe`](ModelExtractor::transcribe) | `transcribe` | none yet: uni-db does not register `local/whisper-cpp` |
//!
//! Other generation providers (Gemini, Vertex AI, Anthropic, Cohere,
//! Mistral) drop images without saying so, so a caption alias on one of
//! them is refused when the instance is built rather than producing
//! captions of nothing.
//!
//! uni-xervo has no video task. A video extractor takes a
//! [`VideoComposer`]: a host function (usually `ffmpeg`) that splits the
//! video into frames and an audio track. The image modes then run on every
//! frame and the transcriber on the track, with each segment's time placed
//! on the video's timeline: a frame's segments run from it to the next
//! frame, and the last frame's cover its own instant.
//!
//! A model call that fails returns [`UnikoError::Llm`] naming the alias.
//!
//! Every mismatch (a mode the modality cannot use, an alias that is not in
//! the catalog or serves the wrong task, a missing composer) is an
//! [`UnikoError::Unsupported`] or [`UnikoError::Config`] at build time,
//! never at ingest.

use std::sync::Arc;

use uni_xervo::traits::{
    AudioInput, ContentBlock, DocExtractOptions, DocOutputFormat, GenerationOptions, ImageInput,
    Message, MessageRole, TranscribeOptions,
};
use uniko_pipes::content::Modality;
use uniko_store::xervo::ModelTask;
use uniko_store::{KnowledgeBase, ModelEntry, UnikoError};

use super::derivation::{
    Derivation, Extraction, MediaRetention, ModelProvenance, Segment, SegmentKind, modality_name,
};
use super::text_extractor::{ExtractRequest, TextExtractor};

/// The caption prompt used when none is set.
pub const DEFAULT_CAPTION_PROMPT: &str = "Describe this image so it can be found by search. \
     Name the people, objects, places and actions you see, and transcribe any visible text. \
     Reply with the description only.";

/// uni-xervo's speech-transcription provider. uni-db builds the store's
/// model runtime without it, so a transcription alias on it cannot load.
pub const WHISPER_CPP_PROVIDER: &str = "local/whisper-cpp";

/// Generation providers that pass an attached image to the model.
pub const IMAGE_CAPABLE_PROVIDERS: &[&str] =
    &["remote/openai", "remote/azure-openai", "remote/llamacpp"];

/// One frame of a video, as a [`VideoComposer`] returns it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VideoFrame {
    /// Where the frame sits in the video, in milliseconds.
    pub timestamp_ms: u64,
    /// The encoded image.
    pub data: Vec<u8>,
    /// Its MIME type, e.g. `image/jpeg`.
    pub mime: String,
}

impl VideoFrame {
    /// A frame at `timestamp_ms`.
    #[must_use]
    pub fn new(timestamp_ms: u64, data: Vec<u8>, mime: impl Into<String>) -> Self {
        Self {
            timestamp_ms,
            data,
            mime: mime.into(),
        }
    }
}

/// What a [`VideoComposer`] splits a video into.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct VideoParts {
    /// Sampled frames, in time order.
    pub frames: Vec<VideoFrame>,
    /// The audio track as 16 kHz, 16-bit PCM WAV, mono or stereo.
    pub audio: Option<Vec<u8>>,
}

impl VideoParts {
    /// Frames and an optional WAV audio track.
    #[must_use]
    pub fn new(frames: Vec<VideoFrame>, audio: Option<Vec<u8>>) -> Self {
        Self { frames, audio }
    }
}

/// Splits a video into frames and an audio track. Runs on a blocking
/// thread, so it can shell out to `ffmpeg` or call back into a host
/// language.
pub type VideoComposer =
    Arc<dyn Fn(&ExtractRequest) -> Result<VideoParts, UnikoError> + Send + Sync>;

/// A caption mode: a generation alias and the prompt sent with the image.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Caption {
    alias: String,
    prompt: String,
}

/// A transcription mode: a transcription alias and an optional language.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Transcribe {
    alias: String,
    language: Option<String>,
}

/// A model-backed extractor for one modality.
///
/// ```no_run
/// use uniko_extract::ingest::{MediaRetention, ModelExtractor};
/// let image = ModelExtractor::image()
///     .ocr("ocr/default")
///     .caption("vision/default")
///     .retention(MediaRetention::Bytes);
/// let audio = ModelExtractor::audio().transcribe("asr/default");
/// ```
///
/// Register it with `UnikoBuilder::model_extractor`; the builder checks it
/// against the catalog before the store opens.
#[derive(Clone)]
pub struct ModelExtractor {
    modality: Modality,
    ocr: Option<String>,
    caption: Option<Caption>,
    document: Option<String>,
    transcribe: Option<Transcribe>,
    composer: Option<VideoComposer>,
    retention: MediaRetention,
    max_caption_tokens: usize,
}

impl std::fmt::Debug for ModelExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelExtractor")
            .field("modality", &self.modality)
            .field("ocr", &self.ocr)
            .field("caption", &self.caption)
            .field("document", &self.document)
            .field("transcribe", &self.transcribe)
            .field("composer", &self.composer.as_ref().map(|_| "<fn>"))
            .field("retention", &self.retention)
            .finish()
    }
}

impl ModelExtractor {
    /// An extractor for `modality`, with no mode yet.
    #[must_use]
    pub fn new(modality: Modality) -> Self {
        Self {
            modality,
            ocr: None,
            caption: None,
            document: None,
            transcribe: None,
            composer: None,
            retention: MediaRetention::None,
            max_caption_tokens: 512,
        }
    }

    /// An image extractor. Add at least one of [`ocr`](Self::ocr),
    /// [`caption`](Self::caption) or [`document`](Self::document).
    #[must_use]
    pub fn image() -> Self {
        Self::new(Modality::Image)
    }

    /// An audio extractor. Add [`transcribe`](Self::transcribe).
    #[must_use]
    pub fn audio() -> Self {
        Self::new(Modality::Audio)
    }

    /// A video extractor that splits each video with `composer`. Add an
    /// image mode for the frames, [`transcribe`](Self::transcribe) for the
    /// audio track, or both.
    #[must_use]
    pub fn video(
        composer: impl Fn(&ExtractRequest) -> Result<VideoParts, UnikoError> + Send + Sync + 'static,
    ) -> Self {
        let mut me = Self::new(Modality::Video);
        me.composer = Some(Arc::new(composer));
        me
    }

    /// Split each video with `composer`. Only a video extractor may have
    /// one; [`validate`](Self::validate) refuses it on any other modality.
    #[must_use]
    pub fn composer(
        mut self,
        composer: impl Fn(&ExtractRequest) -> Result<VideoParts, UnikoError> + Send + Sync + 'static,
    ) -> Self {
        self.composer = Some(Arc::new(composer));
        self
    }

    /// Read text in the image with the OCR alias `alias`. Recognized lines
    /// are grouped into `ocr_region` chunks; each line's box and confidence
    /// are kept in the chunk's `metadata.regions` (a box only when the alias
    /// has a text detector).
    #[must_use]
    pub fn ocr(mut self, alias: impl Into<String>) -> Self {
        self.ocr = Some(alias.into());
        self
    }

    /// Describe the image with the generation alias `alias`, using
    /// [`DEFAULT_CAPTION_PROMPT`]. The description becomes a `caption`
    /// chunk.
    #[must_use]
    pub fn caption(mut self, alias: impl Into<String>) -> Self {
        self.caption = Some(Caption {
            alias: alias.into(),
            prompt: DEFAULT_CAPTION_PROMPT.to_string(),
        });
        self
    }

    /// Replace the caption prompt. Has no effect without
    /// [`caption`](Self::caption).
    #[must_use]
    pub fn caption_prompt(mut self, prompt: impl Into<String>) -> Self {
        if let Some(c) = self.caption.as_mut() {
            c.prompt = prompt.into();
        }
        self
    }

    /// The most tokens a caption may use (default 512).
    #[must_use]
    pub fn max_caption_tokens(mut self, tokens: usize) -> Self {
        self.max_caption_tokens = tokens.max(1);
        self
    }

    /// Extract the image as a document page with the document-extraction
    /// alias `alias`. Blocks are grouped into `page_block` chunks in reading
    /// order, each block's box kept in `metadata.regions`.
    #[must_use]
    pub fn document(mut self, alias: impl Into<String>) -> Self {
        self.document = Some(alias.into());
        self
    }

    /// Transcribe speech with the transcription alias `alias`. Segments
    /// become `transcript_segment` chunks with their time range.
    ///
    /// No provider can serve this yet: uni-db builds the store's runtime
    /// without uni-xervo's `local/whisper-cpp` provider, and
    /// [`validate`](Self::validate) refuses an alias on it. Until it does,
    /// transcribe with your own function through
    /// [`TextExtractor::structured`].
    #[must_use]
    pub fn transcribe(mut self, alias: impl Into<String>) -> Self {
        self.transcribe = Some(Transcribe {
            alias: alias.into(),
            language: None,
        });
        self
    }

    /// The language to transcribe in (e.g. `"en"`), instead of detecting
    /// it. Has no effect without [`transcribe`](Self::transcribe).
    #[must_use]
    pub fn language(mut self, language: impl Into<String>) -> Self {
        if let Some(t) = self.transcribe.as_mut() {
            t.language = Some(language.into());
        }
        self
    }

    /// Whether to record the original media; see [`MediaRetention`].
    #[must_use]
    pub fn retention(mut self, retention: MediaRetention) -> Self {
        self.retention = retention;
        self
    }

    /// The modality this extractor handles.
    #[must_use]
    pub fn modality(&self) -> Modality {
        self.modality
    }

    /// Every alias this extractor calls, in mode order.
    #[must_use]
    pub fn aliases(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.extend(self.ocr.clone());
        out.extend(self.caption.as_ref().map(|c| c.alias.clone()));
        out.extend(self.document.clone());
        out.extend(self.transcribe.as_ref().map(|t| t.alias.clone()));
        out
    }

    fn has_image_mode(&self) -> bool {
        self.ocr.is_some() || self.caption.is_some() || self.document.is_some()
    }

    /// Check the modes against the modality and the catalog `entries`.
    ///
    /// # Errors
    ///
    /// - [`UnikoError::Unsupported`] for a mode the modality cannot use
    ///   (OCR on audio, transcription of an image, any video without a
    ///   composer), a caption alias whose provider drops images, or a
    ///   transcription alias on `local/whisper-cpp`, which the runtime does
    ///   not register.
    /// - [`UnikoError::Config`] for an extractor with no mode, or an alias
    ///   that is not in the catalog or serves a different task.
    pub fn validate(&self, entries: &[ModelEntry]) -> Result<(), UnikoError> {
        let name = modality_name(self.modality);
        match self.modality {
            Modality::Image => {
                if self.transcribe.is_some() {
                    return Err(UnikoError::Unsupported(
                        "image extractor: an image cannot be transcribed".into(),
                    ));
                }
                if !self.has_image_mode() {
                    return Err(UnikoError::Config(
                        "image extractor: set ocr, caption or document".into(),
                    ));
                }
            }
            Modality::Audio => {
                if self.has_image_mode() {
                    return Err(UnikoError::Unsupported(
                        "audio extractor: ocr, caption and document apply to images only".into(),
                    ));
                }
                if self.transcribe.is_none() {
                    return Err(UnikoError::Config("audio extractor: set transcribe".into()));
                }
            }
            Modality::Video => {
                if self.composer.is_none() {
                    return Err(UnikoError::Unsupported(
                        "video extractor: the model runtime has no video task; build it with \
                         ModelExtractor::video(composer), which splits the video into frames \
                         and an audio track"
                            .into(),
                    ));
                }
                if !self.has_image_mode() && self.transcribe.is_none() {
                    return Err(UnikoError::Config(
                        "video extractor: set an image mode for the frames, transcribe for the \
                         audio, or both"
                            .into(),
                    ));
                }
            }
            other => {
                return Err(UnikoError::Unsupported(format!(
                    "model extractor: {other:?} is not image, audio or video"
                )));
            }
        }
        if self.composer.is_some() && self.modality != Modality::Video {
            return Err(UnikoError::Unsupported(format!(
                "{name} extractor: a composer applies to video only"
            )));
        }
        let entry = |alias: &str, task: ModelTask, mode: &str| -> Result<&ModelEntry, UnikoError> {
            let e = entries.iter().find(|e| e.alias == alias).ok_or_else(|| {
                UnikoError::Config(format!(
                    "{name} extractor: {mode} alias '{alias}' is not in the model catalog"
                ))
            })?;
            if e.task != task {
                return Err(UnikoError::Config(format!(
                    "{name} extractor: {mode} alias '{alias}' serves {:?}, not {task:?}",
                    e.task
                )));
            }
            Ok(e)
        };
        if let Some(alias) = &self.ocr {
            entry(alias, ModelTask::Ocr, "ocr")?;
        }
        if let Some(c) = &self.caption {
            let e = entry(&c.alias, ModelTask::Generate, "caption")?;
            if !IMAGE_CAPABLE_PROVIDERS.contains(&e.provider_id.as_str()) {
                return Err(UnikoError::Unsupported(format!(
                    "{name} extractor: caption alias '{}' uses {}, which does not pass images to \
                     the model; use one of {}",
                    c.alias,
                    e.provider_id,
                    IMAGE_CAPABLE_PROVIDERS.join(", ")
                )));
            }
        }
        if let Some(alias) = &self.document {
            entry(alias, ModelTask::DocumentExtract, "document")?;
        }
        if let Some(t) = &self.transcribe {
            let e = entry(&t.alias, ModelTask::Transcribe, "transcribe")?;
            if e.provider_id == WHISPER_CPP_PROVIDER {
                return Err(UnikoError::Unsupported(format!(
                    "{name} extractor: transcribe alias '{}' uses {WHISPER_CPP_PROVIDER}, which \
                     the store's model runtime does not register (uni-db builds it without the \
                     whisper.cpp provider); transcribe with your own function through \
                     TextExtractor::structured instead",
                    t.alias
                )));
            }
        }
        Ok(())
    }

    /// Turn this into a registrable extractor.
    #[must_use]
    pub fn into_extractor(self) -> TextExtractor {
        let modality = self.modality;
        let retention = self.retention;
        TextExtractor::from_models(modality, Arc::new(self)).with_retention(retention)
    }

    /// Derive the text of `request` through the model runtime.
    pub(crate) async fn derive(
        &self,
        kb: &KnowledgeBase,
        request: &ExtractRequest,
    ) -> Result<Extraction, UnikoError> {
        let runtime = kb.model_runtime().ok_or_else(|| {
            UnikoError::Unsupported(format!(
                "{} extractor: this store has no model runtime",
                modality_name(self.modality)
            ))
        })?;
        let mut out = Extraction::default();
        match self.modality {
            Modality::Image => {
                let image = vec![(None, image_input(&request.bytes, request.mime.as_ref()))];
                self.derive_images(&runtime, image, &mut out).await?;
            }
            Modality::Audio => {
                check_wav(&request.bytes)?;
                self.derive_audio(&runtime, request.bytes.clone(), 0, &mut out)
                    .await?;
            }
            Modality::Video => {
                let composer = self.composer.clone().ok_or_else(|| {
                    UnikoError::Unsupported("video extractor: no composer".into())
                })?;
                let req = request.clone();
                let parts = tokio::task::spawn_blocking(move || composer(&req))
                    .await
                    .map_err(|e| {
                        UnikoError::Internal(format!("composer did not complete: {e}"))
                    })??;
                if self.has_image_mode() && !parts.frames.is_empty() {
                    let mut frames: Vec<VideoFrame> = parts.frames;
                    frames.sort_by_key(|f| f.timestamp_ms);
                    let ends: Vec<u64> = frames
                        .iter()
                        .enumerate()
                        .map(|(i, f)| frames.get(i + 1).map_or(f.timestamp_ms, |n| n.timestamp_ms))
                        .collect();
                    let images = frames
                        .iter()
                        .zip(ends)
                        .map(|(f, end)| {
                            (Some((f.timestamp_ms, end)), image_input(&f.data, &f.mime))
                        })
                        .collect();
                    self.derive_images(&runtime, images, &mut out).await?;
                }
                if self.transcribe.is_some()
                    && let Some(audio) = parts.audio
                {
                    check_wav(&audio)?;
                    self.derive_audio(&runtime, audio, 0, &mut out).await?;
                }
            }
            _ => {}
        }
        Ok(out)
    }

    /// Run every image mode over `images`, each tagged with the time range
    /// it covers (for video frames).
    async fn derive_images(
        &self,
        runtime: &uniko_store::ModelRuntime,
        images: Vec<(Option<(u64, u64)>, ImageInput)>,
        out: &mut Extraction,
    ) -> Result<(), UnikoError> {
        let place = |seg: Segment, time: Option<(u64, u64)>| match time {
            Some((start, end)) => seg.with_time(start, end),
            None => seg,
        };
        if let Some(alias) = &self.caption.as_ref().map(|c| c.alias.clone()) {
            let caption = self.caption.as_ref().expect("checked");
            let model = runtime.generator(alias).await.map_err(model_err(alias))?;
            let mut segments = Vec::new();
            for (time, image) in &images {
                let message = Message {
                    role: MessageRole::User,
                    content: vec![
                        ContentBlock::Text(caption.prompt.clone()),
                        ContentBlock::Image(image.clone()),
                    ],
                };
                let result = model
                    .generate(
                        &[message],
                        GenerationOptions {
                            max_tokens: Some(self.max_caption_tokens),
                            temperature: Some(0.0),
                            ..GenerationOptions::default()
                        },
                    )
                    .await
                    .map_err(model_err(alias))?;
                segments.push(place(
                    Segment::new(SegmentKind::Caption, result.text.trim()),
                    *time,
                ));
            }
            out.push(Derivation::new(
                segments,
                Some(ModelProvenance::new(alias, model.model_id(), "generate")),
            ));
        }
        if let Some(alias) = &self.ocr {
            let model = runtime.ocr_model(alias).await.map_err(model_err(alias))?;
            let results = model
                .recognize(images.iter().map(|(_, i)| i.clone()).collect())
                .await
                .map_err(model_err(alias))?;
            let mut segments = Vec::new();
            for ((time, _), result) in images.iter().zip(results) {
                segments.extend(
                    ocr_segments(result)
                        .into_iter()
                        .map(|seg| place(seg, *time)),
                );
            }
            out.push(Derivation::new(
                segments,
                Some(ModelProvenance::new(alias, model.model_id(), "ocr")),
            ));
        }
        if let Some(alias) = &self.document {
            let model = runtime
                .document_extractor(alias)
                .await
                .map_err(model_err(alias))?;
            let results = model
                .extract(
                    images.iter().map(|(_, i)| i.clone()).collect(),
                    DocExtractOptions {
                        output: DocOutputFormat::Markdown,
                        include_tables: true,
                        include_formulas: true,
                        include_bboxes: true,
                    },
                )
                .await
                .map_err(model_err(alias))?;
            let mut segments = Vec::new();
            for ((time, _), mut result) in images.iter().zip(results) {
                result.blocks.sort_by_key(|b| b.reading_order);
                if result.blocks.is_empty() && !result.plain_markdown.trim().is_empty() {
                    segments.push(place(
                        Segment::new(SegmentKind::PageBlock, result.plain_markdown.trim()),
                        *time,
                    ));
                }
                for block in result.blocks {
                    let mut seg = Segment::new(SegmentKind::PageBlock, block.content.trim());
                    if let Some(bbox) = block.bbox {
                        seg = seg.with_bbox(bbox);
                    }
                    segments.push(place(seg, *time));
                }
            }
            out.push(Derivation::new(
                segments,
                Some(ModelProvenance::new(
                    alias,
                    model.model_id(),
                    "document_extract",
                )),
            ));
        }
        Ok(())
    }

    /// Transcribe a WAV track, placing its segments `offset_ms` into the
    /// media.
    async fn derive_audio(
        &self,
        runtime: &uniko_store::ModelRuntime,
        wav: Vec<u8>,
        offset_ms: u64,
        out: &mut Extraction,
    ) -> Result<(), UnikoError> {
        let Some(t) = &self.transcribe else {
            return Ok(());
        };
        let model = runtime
            .transcriber(&t.alias)
            .await
            .map_err(model_err(&t.alias))?;
        let result = model
            .transcribe_one(
                AudioInput::Bytes {
                    data: wav,
                    media_type: "audio/wav".into(),
                },
                TranscribeOptions {
                    language: t.language.clone(),
                    word_timestamps: true,
                    ..TranscribeOptions::default()
                },
            )
            .await
            .map_err(model_err(&t.alias))?;
        let segments = result
            .segments
            .into_iter()
            .map(|s| {
                let mut seg = Segment::new(SegmentKind::Transcript, s.text.trim())
                    .with_time(s.start_ms, s.end_ms)
                    .offset_by(offset_ms);
                if let Some(c) = s
                    .words
                    .iter()
                    .filter_map(|w| w.confidence)
                    .fold(None, |a: Option<f32>, c| Some(a.map_or(c, |a| a.min(c))))
                {
                    seg = seg.with_confidence(c);
                }
                if let Some(speaker) = s.speaker {
                    seg = seg.with_speaker(speaker);
                }
                seg
            })
            .collect();
        out.push(Derivation::new(
            segments,
            Some(ModelProvenance::new(
                &t.alias,
                model.model_id(),
                "transcribe",
            )),
        ));
        Ok(())
    }
}

/// One `ocr_region` segment per recognized line.
fn ocr_segments(result: uni_xervo::traits::OcrResult) -> Vec<Segment> {
    result
        .blocks
        .into_iter()
        .map(|block| {
            let seg = Segment::new(SegmentKind::OcrRegion, block.text.trim())
                .with_confidence(block.confidence);
            // Without a detector the recognizer reports the whole image as
            // `[0, 0, 1, 1]`, which locates nothing.
            if block.bbox == [0.0, 0.0, 1.0, 1.0] {
                seg
            } else {
                seg.with_bbox(block.bbox)
            }
        })
        .collect()
}

fn image_input(bytes: &[u8], mime: &str) -> ImageInput {
    ImageInput::Bytes {
        data: bytes.to_vec(),
        media_type: mime.to_string(),
    }
}

fn model_err(alias: &str) -> impl Fn(uni_xervo::error::RuntimeError) -> UnikoError + '_ {
    // `Llm`: a model provider call failed. Not `Embedding`, which callers
    // read as "the embedder is unavailable".
    move |e| UnikoError::Llm(format!("model alias '{alias}': {e}"))
}

/// What a WAV header says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavInfo {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u16,
    /// Bits per sample.
    pub bits_per_sample: u16,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// Read a RIFF/WAVE header. `None` when `bytes` is not a WAV file.
#[must_use]
pub fn wav_info(bytes: &[u8]) -> Option<(u16, WavInfo)> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut pos = 12;
    let mut fmt: Option<(u16, u16, u32, u32, u16)> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?) as usize;
        let body = pos + 8;
        if id == b"fmt " && body + 16 <= bytes.len() {
            let b = &bytes[body..body + 16];
            fmt = Some((
                u16::from_le_bytes([b[0], b[1]]),
                u16::from_le_bytes([b[2], b[3]]),
                u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
                u32::from_le_bytes([b[8], b[9], b[10], b[11]]),
                u16::from_le_bytes([b[14], b[15]]),
            ));
        } else if id == b"data" {
            let (format, channels, sample_rate, byte_rate, bits) = fmt?;
            let data_len = len.min(bytes.len().saturating_sub(body)) as u64;
            return Some((
                format,
                WavInfo {
                    sample_rate,
                    channels,
                    bits_per_sample: bits,
                    duration_ms: (data_len * 1000)
                        .checked_div(u64::from(byte_rate))
                        .unwrap_or(0),
                },
            ));
        }
        pos = body + len + (len & 1);
    }
    None
}

/// Check that `bytes` is audio the transcriber accepts: PCM WAV, 16-bit,
/// 16 kHz, mono or stereo.
///
/// # Errors
///
/// [`UnikoError::Unsupported`] naming what to convert, with the `ffmpeg`
/// command that does it.
pub fn check_wav(bytes: &[u8]) -> Result<WavInfo, UnikoError> {
    const FIX: &str =
        "convert it first, e.g. `ffmpeg -i in -ar 16000 -ac 1 -c:a pcm_s16le out.wav`";
    let Some((format, info)) = wav_info(bytes) else {
        return Err(UnikoError::Unsupported(format!(
            "audio extractor: only PCM WAV audio is accepted; {FIX}"
        )));
    };
    if format != 1 || info.bits_per_sample != 16 {
        return Err(UnikoError::Unsupported(format!(
            "audio extractor: WAV must be 16-bit PCM (format {format}, {} bits); {FIX}",
            info.bits_per_sample
        )));
    }
    if info.sample_rate != 16_000 {
        return Err(UnikoError::Unsupported(format!(
            "audio extractor: WAV must be 16000 Hz (got {}); {FIX}",
            info.sample_rate
        )));
    }
    if !matches!(info.channels, 1 | 2) {
        return Err(UnikoError::Unsupported(format!(
            "audio extractor: WAV must be mono or stereo (got {} channels); {FIX}",
            info.channels
        )));
    }
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(alias: &str, task: ModelTask, provider: &str) -> ModelEntry {
        let spec: uniko_store::xervo::ModelAliasSpec = serde_json::from_value(serde_json::json!({
            "alias": alias,
            "task": task,
            "provider_id": provider,
            "model_id": "m",
        }))
        .expect("a valid alias spec");
        ModelEntry::from(&spec)
    }

    fn catalog() -> Vec<ModelEntry> {
        vec![
            entry("ocr/default", ModelTask::Ocr, "local/onnx"),
            entry("vision/openai", ModelTask::Generate, "remote/openai"),
            entry("llm/gemini", ModelTask::Generate, "remote/gemini"),
            entry("doc/vlm", ModelTask::DocumentExtract, "remote/llamacpp"),
            entry("asr/default", ModelTask::Transcribe, "local/whisper-cpp"),
            entry("embed/default", ModelTask::Embed, "local/onnx"),
        ]
    }

    pub(crate) fn wav(sample_rate: u32, channels: u16, bits: u16, ms: u32) -> Vec<u8> {
        let bytes_per_sample = u32::from(bits / 8);
        let data_len = sample_rate * u32::from(channels) * bytes_per_sample * ms / 1000;
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data_len).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&channels.to_le_bytes());
        v.extend_from_slice(&sample_rate.to_le_bytes());
        v.extend_from_slice(&(sample_rate * u32::from(channels) * bytes_per_sample).to_le_bytes());
        v.extend_from_slice(&(channels * bits / 8).to_le_bytes());
        v.extend_from_slice(&bits.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data_len.to_le_bytes());
        v.resize(v.len() + data_len as usize, 0);
        v
    }

    #[test]
    fn a_valid_extractor_per_modality_passes() {
        let c = catalog();
        ModelExtractor::image()
            .ocr("ocr/default")
            .caption("vision/openai")
            .document("doc/vlm")
            .validate(&c)
            .unwrap();
        ModelExtractor::video(|_| Ok(VideoParts::default()))
            .ocr("ocr/default")
            .validate(&c)
            .unwrap();
        let audio = ModelExtractor::audio()
            .transcribe("asr/default")
            .validate(&c);
        assert!(
            matches!(&audio, Err(UnikoError::Unsupported(m)) if m.contains("does not register")),
            "{audio:?}"
        );
        // Any other provider serving `transcribe` is accepted.
        let mut other = c.clone();
        other.push(entry("asr/remote", ModelTask::Transcribe, "remote/someday"));
        ModelExtractor::audio()
            .transcribe("asr/remote")
            .validate(&other)
            .unwrap();
    }

    #[test]
    fn unsupported_modes_are_refused() {
        let c = catalog();
        let unsupported = |e: ModelExtractor, needle: &str| match e.validate(&c) {
            Err(UnikoError::Unsupported(m)) => assert!(m.contains(needle), "{m}"),
            other => panic!("expected Unsupported containing {needle:?}, got {other:?}"),
        };
        unsupported(
            ModelExtractor::image().transcribe("asr/default"),
            "cannot be transcribed",
        );
        unsupported(ModelExtractor::audio().ocr("ocr/default"), "images only");
        unsupported(
            ModelExtractor::new(Modality::Video).ocr("ocr/default"),
            "no video task",
        );
        unsupported(
            ModelExtractor::image().caption("llm/gemini"),
            "does not pass images",
        );
        unsupported(
            ModelExtractor::image()
                .composer(|_| Ok(VideoParts::default()))
                .ocr("ocr/default"),
            "video only",
        );
    }

    #[test]
    fn misconfigured_aliases_are_config_errors() {
        let c = catalog();
        let config = |e: ModelExtractor, needle: &str| match e.validate(&c) {
            Err(UnikoError::Config(m)) => assert!(m.contains(needle), "{m}"),
            other => panic!("expected Config containing {needle:?}, got {other:?}"),
        };
        config(ModelExtractor::image(), "set ocr, caption or document");
        config(ModelExtractor::audio(), "set transcribe");
        config(
            ModelExtractor::video(|_| Ok(VideoParts::default())),
            "or both",
        );
        config(
            ModelExtractor::image().ocr("ocr/missing"),
            "not in the model catalog",
        );
        config(ModelExtractor::image().ocr("embed/default"), "not Ocr");
        assert_eq!(
            ModelExtractor::image()
                .ocr("a")
                .caption("b")
                .document("c")
                .aliases(),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn ocr_blocks_keep_real_boxes_only() {
        use uni_xervo::traits::{OcrBlock, OcrResult};
        let block = |text: &str, bbox: [f32; 4], confidence: f32| OcrBlock {
            text: text.into(),
            bbox,
            confidence,
        };
        let result = OcrResult {
            blocks: vec![
                block(" whole image ", [0.0, 0.0, 1.0, 1.0], 0.5),
                block("a line", [4.0, 5.0, 60.0, 20.0], 0.9),
            ],
            plain_text: String::new(),
        };
        let segs = ocr_segments(result);
        assert_eq!(segs[0].text, "whole image");
        assert_eq!(segs[0].bbox, None, "the placeholder box locates nothing");
        assert_eq!(segs[1].bbox, Some([4.0, 5.0, 60.0, 20.0]));
        assert_eq!(segs[1].confidence, Some(0.9));
    }

    #[test]
    fn wav_headers_are_checked() {
        assert_eq!(
            wav_info(&wav(44_100, 2, 16, 1500)).unwrap().1.duration_ms,
            1500
        );
        assert_eq!(wav_info(&wav(8_000, 1, 8, 250)).unwrap().1.duration_ms, 250);
        let ok = check_wav(&wav(16_000, 1, 16, 1500)).unwrap();
        assert_eq!(ok.duration_ms, 1500);
        assert_eq!(ok.channels, 1);
        check_wav(&wav(16_000, 2, 16, 10)).unwrap();
        for (bytes, needle) in [
            (b"ID3 not a wav".to_vec(), "only PCM WAV"),
            (wav(44_100, 1, 16, 10), "16000 Hz"),
            (wav(16_000, 1, 8, 10), "16-bit"),
            (wav(16_000, 6, 16, 10), "mono or stereo"),
        ] {
            match check_wav(&bytes) {
                Err(UnikoError::Unsupported(m)) => {
                    assert!(m.contains(needle), "{m}");
                    assert!(m.contains("ffmpeg"), "{m}");
                }
                other => panic!("{needle}: {other:?}"),
            }
        }
    }
}
