//! Deterministic read tools over what an agent has stored.
//!
//! Recall ranks; these read. An agent that has recalled a chunk of a tool's
//! result, or wants an exact identifier, uses them to look around it:
//!
//! - [`search`](Sources::search): lexical search that matches identifiers
//!   and figures whole, every hit citing its source and address;
//! - [`outline`](Sources::outline): the shape of a source — a JSON
//!   payload's keys and arrays, a table's chunks, a log's line ranges;
//! - [`read`](Sources::read): the verbatim text at an address, a line or
//!   byte range, or a run of chunks, with the range to read next;
//! - [`tool_call`](Sources::tool_call) / [`tool_calls`](Sources::tool_calls):
//!   what was called, with which arguments, and where its result is.
//!
//! Every call is a pure function of the store and its arguments, so a
//! sequence of them can be replayed, and every read applies the same policy
//! as recall and [`Data`](crate::facade::Data): nothing forgotten, nothing
//! the agent's viewer may not see.

use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;
use uniko_extract::ingest::count_tokens;
use uniko_extract::ingest::structure::json::{self as sjson, JsonKind, JsonNode};
use uniko_extract::ingest::structure::normalize::normalize_for_search;
use uniko_extract::ingest::structure::{Format, detect_format};
use uniko_store::repository::sources::{ActionRow, ChunkRow};
use uniko_store::{NodeId, UnikoError};

use crate::Agent;
use crate::facade::Data;
use crate::policy::admitted_nodes;
use crate::recall::RecallKind;
use crate::recall::intent::extract_literals;

/// A source the read tools work on, by external id.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SourceRef {
    /// An artifact: a document, an attachment or a tool call's result.
    Artifact(String),
    /// A message (a turn).
    Message(String),
    /// A session's transcript chunks.
    Session(String),
}

impl SourceRef {
    fn from_owner(label: &str, id: &str) -> Self {
        match label {
            "Message" => Self::Message(id.to_string()),
            "Session" => Self::Session(id.to_string()),
            _ => Self::Artifact(id.to_string()),
        }
    }
}

/// Filters for [`Sources::search`].
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Most hits to return (default 10).
    pub limit: usize,
    /// Only hits from this session.
    pub session_id: Option<String>,
    /// Only hits from this tool's results.
    pub tool_name: Option<String>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            session_id: None,
            tool_name: None,
        }
    }
}

impl SearchOptions {
    /// At most `limit` hits.
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Only hits from session `session_id`.
    #[must_use]
    pub fn session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Only hits from results of `tool_name`.
    #[must_use]
    pub fn tool(mut self, tool_name: impl Into<String>) -> Self {
        self.tool_name = Some(tool_name.into());
        self
    }
}

/// One [`Sources::search`] hit.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// Where the hit lives.
    pub source: SourceRef,
    /// The chunk's id.
    pub chunk_id: String,
    /// The chunk's position in its source.
    pub chunk_index: usize,
    /// Its address in the source, for a structure-aware chunk.
    pub address: Option<String>,
    /// Byte span in the source.
    pub start: usize,
    /// Byte span end.
    pub end: usize,
    /// The chunk's verbatim text.
    pub text: String,
    /// What the chunk inherits from outside itself, for a tool output.
    pub context: Option<String>,
    /// Lexical score.
    pub score: f64,
    /// The session, when known.
    pub session_id: Option<String>,
    /// The tool call whose result this is, when it is one.
    pub call_id: Option<String>,
    /// That call's tool.
    pub tool_name: Option<String>,
}

/// One entry of an [`Outline`].
#[derive(Debug, Clone)]
pub struct OutlineEntry {
    /// Where it is: a JSON pointer, `rows 10-19`, `lines 1-40`, `chunk 3`.
    pub address: String,
    /// `object`, `array`, `scalar`, or `chunk`.
    pub kind: String,
    /// A one-line description: keys, element count, a preview.
    pub summary: String,
    /// Byte span in the source.
    pub start: usize,
    /// Byte span end.
    pub end: usize,
    /// Nesting depth (0 for the top).
    pub depth: usize,
}

/// The shape of a source, from [`Sources::outline`].
#[derive(Debug, Clone)]
pub struct Outline {
    /// The source outlined.
    pub source: SourceRef,
    /// Detected format (`json`, `delimited`, `log`, `text`, …).
    pub format: String,
    /// Size of the source text in bytes.
    pub total_bytes: usize,
    /// Number of lines in it.
    pub total_lines: usize,
    /// Entries in source order.
    pub entries: Vec<OutlineEntry>,
    /// More entries existed than were returned.
    pub truncated: bool,
}

/// What to read with [`Sources::read`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadRange {
    /// An address as search hits and outlines give them: a JSON pointer
    /// (`/data/items/3`, `/data/items[3-9]`), `rows a-b`, `lines a-b`,
    /// `records a-b`.
    Address(String),
    /// Bytes `[start, end)` of the source.
    Bytes(usize, usize),
    /// Lines `first..=last`, 1-based.
    Lines(usize, usize),
    /// Chunks `first..=last` by index.
    Chunks(usize, usize),
}

/// The text a [`Sources::read`] returned.
#[derive(Debug, Clone)]
pub struct ReadResult {
    /// The source read.
    pub source: SourceRef,
    /// Verbatim text.
    pub text: String,
    /// Byte span of `text` in the source.
    pub start: usize,
    /// Byte span end.
    pub end: usize,
    /// First line of `text`, 1-based.
    pub first_line: usize,
    /// Last line of `text`.
    pub last_line: usize,
    /// The range was cut at the token limit.
    pub truncated: bool,
    /// The rest of the range, when it was cut.
    pub next: Option<ReadRange>,
}

/// A recorded tool call, from [`Sources::tool_call`].
#[derive(Debug, Clone)]
pub struct ToolCallView {
    /// The call's id.
    pub call_id: String,
    /// The tool, for a call recorded as a tool call.
    pub tool_name: Option<String>,
    /// `"tool_call"`, or the kind given to `record_action`.
    pub action_type: String,
    /// The arguments it was called with.
    pub arguments: Option<JsonValue>,
    /// Its outcome.
    pub status: Option<String>,
    /// Error text for a failed call.
    pub error: Option<String>,
    /// When it started.
    pub started_at: Option<DateTime<Utc>>,
    /// When it ended.
    pub ended_at: Option<DateTime<Utc>>,
    /// The message that made the call.
    pub triggered_by: Option<String>,
    /// Its session.
    pub session_id: Option<String>,
    /// Its result, readable with [`Sources::read`].
    pub artifact_id: Option<String>,
}

/// Deterministic read tools over an agent's store; obtain with
/// [`Agent::sources`](crate::Agent::sources). See the module docs.
#[derive(Debug)]
pub struct Sources<'a> {
    agent: &'a Agent,
}

/// A source's readable text and its chunks, after policy.
struct Loaded {
    /// The full verbatim text, when the reader may see all of it.
    text: Option<String>,
    /// The chunks the reader may see, in index order.
    chunks: Vec<ChunkRow>,
}

impl<'a> Sources<'a> {
    pub(crate) fn new(agent: &'a Agent) -> Self {
        Self { agent }
    }

    fn data(&self) -> Data<'a> {
        Data::new(self.agent)
    }

    /// Search stored chunks lexically. Identifiers and figures in `query`
    /// (`ORD-48213`, `req-9f3a`, `2026-07`) are matched whole; a query with
    /// none is matched word by word. Hits are ordered by score, then chunk
    /// id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when a store read fails.
    pub async fn search(
        &self,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<SearchHit>, UnikoError> {
        let literals = extract_literals(query);
        let lex = if literals.is_empty() {
            normalize_for_search(query)
        } else {
            normalize_for_search(&literals.join(" "))
        };
        if lex.is_empty() || options.limit == 0 {
            return Ok(Vec::new());
        }
        let kb = self.agent.kb();
        let fetch = i64::try_from(options.limit.saturating_mul(5).max(25)).unwrap_or(i64::MAX);
        let rows = kb.search_chunks_lexical(&lex, fetch).await?;
        let viewer = self.data().viewer().await?;
        let pairs: Vec<(NodeId, RecallKind)> = rows
            .iter()
            .map(|r| (r.node_id, RecallKind::Chunk))
            .collect();
        let admitted = admitted_nodes(kb, viewer.as_ref(), &pairs).await?;
        Ok(rows
            .into_iter()
            .filter(|r| admitted.contains(&r.node_id) && !r.owner_id.is_empty())
            .filter(|r| {
                options
                    .session_id
                    .as_deref()
                    .is_none_or(|s| r.session_id.as_deref() == Some(s))
            })
            .filter(|r| {
                options
                    .tool_name
                    .as_deref()
                    .is_none_or(|t| r.tool_name.as_deref() == Some(t))
            })
            .take(options.limit)
            .map(|r| SearchHit {
                source: SourceRef::from_owner(&r.owner_label, &r.owner_id),
                chunk_id: r.chunk_id,
                chunk_index: usize::try_from(r.index).unwrap_or(0),
                address: r.address,
                start: usize::try_from(r.start).unwrap_or(0),
                end: usize::try_from(r.end).unwrap_or(0),
                text: r.text,
                context: r.context,
                score: r.score,
                session_id: r.session_id,
                call_id: r.call_id,
                tool_name: r.tool_name,
            })
            .collect())
    }

    /// Load `source`'s text and chunks as far as the reader may see them.
    /// `None` when the source is unknown or not readable at all.
    async fn load(&self, source: &SourceRef) -> Result<Option<Loaded>, UnikoError> {
        let kb = self.agent.kb();
        let data = self.data();
        let (label, id) = match source {
            SourceRef::Artifact(id) => ("Artifact", id),
            SourceRef::Message(id) => ("Message", id),
            SourceRef::Session(id) => ("Session", id),
        };
        let chunks = kb.chunk_range(label, id, 0, i64::MAX).await?;
        let viewer = data.viewer().await?;
        let pairs: Vec<(NodeId, RecallKind)> = chunks
            .iter()
            .map(|c| (c.node_id, RecallKind::Chunk))
            .collect();
        let admitted = admitted_nodes(kb, viewer.as_ref(), &pairs).await?;
        let visible: Vec<ChunkRow> = chunks
            .iter()
            .filter(|c| admitted.contains(&c.node_id))
            .cloned()
            .collect();
        let whole = visible.len() == chunks.len();
        match source {
            SourceRef::Artifact(id) => {
                let Some((nid, content_id)) = kb.artifact_content_id(id).await? else {
                    return Ok(None);
                };
                let all: Vec<NodeId> = chunks.iter().map(|c| c.node_id).collect();
                let access = data.artifact_access(nid, &all).await?;
                if !access.readable {
                    return Ok(None);
                }
                let text = if whole {
                    let bytes = kb.fetch_blob(&content_id).await?;
                    String::from_utf8(bytes).ok()
                } else {
                    None
                };
                Ok(Some(Loaded {
                    text,
                    chunks: visible,
                }))
            }
            SourceRef::Message(id) => Ok(data.message(id).await?.map(|m| Loaded {
                text: whole.then_some(m.content),
                chunks: visible,
            })),
            SourceRef::Session(_) => {
                if chunks.is_empty() {
                    return Ok(None);
                }
                Ok(Some(Loaded {
                    text: None,
                    chunks: visible,
                }))
            }
        }
    }

    /// The shape of `source`: for JSON its objects and arrays down to
    /// `depth` levels, otherwise its chunks with their addresses. At most
    /// `max_entries` entries. `None` when the source is unknown or the
    /// reader may not see it.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when a store read fails.
    pub async fn outline(
        &self,
        source: &SourceRef,
        depth: usize,
        max_entries: usize,
    ) -> Result<Option<Outline>, UnikoError> {
        let Some(loaded) = self.load(source).await? else {
            return Ok(None);
        };
        let max_entries = if max_entries == 0 { 200 } else { max_entries };
        let mut entries = Vec::new();
        let (format, total_bytes, total_lines) = match &loaded.text {
            Some(text) => {
                let format = detect_format(text, None);
                if format == Format::Json
                    && let Some(root) = sjson::parse(text)
                {
                    json_outline(text, &root, &mut Vec::new(), 0, depth.max(1), &mut entries);
                }
                (format.as_str(), text.len(), text.lines().count())
            }
            None => ("chunks", 0, 0),
        };
        if entries.is_empty() {
            for c in &loaded.chunks {
                let preview: String = c
                    .text
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(80)
                    .collect();
                entries.push(OutlineEntry {
                    address: c
                        .address
                        .clone()
                        .unwrap_or_else(|| format!("chunk {}", c.index)),
                    kind: "chunk".into(),
                    summary: format!("{} tokens: {preview}", count_tokens(&c.text)),
                    start: usize::try_from(c.start).unwrap_or(0),
                    end: usize::try_from(c.end).unwrap_or(0),
                    depth: 0,
                });
            }
        }
        let truncated = entries.len() > max_entries;
        entries.truncate(max_entries);
        Ok(Some(Outline {
            source: source.clone(),
            format: format.to_string(),
            total_bytes,
            total_lines,
            entries,
            truncated,
        }))
    }

    /// Read `range` of `source` verbatim, at most `max_tokens` tokens (0 =
    /// 2,000). A cut range reports the remainder in `next`. Byte, line and
    /// address ranges need the whole source to be readable; a chunk range
    /// returns only the chunks the reader may see. `None` when the source
    /// is unknown, unreadable, or the range resolves to nothing.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] for an address that cannot be
    /// resolved, or [`UnikoError`] when a store read fails.
    pub async fn read(
        &self,
        source: &SourceRef,
        range: &ReadRange,
        max_tokens: usize,
    ) -> Result<Option<ReadResult>, UnikoError> {
        let max_tokens = if max_tokens == 0 { 2_000 } else { max_tokens };
        let Some(loaded) = self.load(source).await? else {
            return Ok(None);
        };
        let Some(text) = loaded.text.as_deref() else {
            // Only chunk reads are possible without the whole text.
            let ReadRange::Chunks(first, last) = range else {
                return Ok(None);
            };
            let picked: Vec<&ChunkRow> = loaded
                .chunks
                .iter()
                .filter(|c| (*first as i64..=*last as i64).contains(&c.index))
                .collect();
            if picked.is_empty() {
                return Ok(None);
            }
            let joined = picked
                .iter()
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let (body, cut) = cut_to_tokens(&joined, max_tokens);
            return Ok(Some(ReadResult {
                source: source.clone(),
                text: body.to_string(),
                start: usize::try_from(picked[0].start).unwrap_or(0),
                end: usize::try_from(picked[picked.len() - 1].end).unwrap_or(0),
                first_line: 0,
                last_line: 0,
                truncated: cut.is_some(),
                next: None,
            }));
        };
        let Some((start, end)) = resolve_range(text, range, &loaded.chunks)? else {
            return Ok(None);
        };
        let slice = &text[start..end];
        let (body, cut) = cut_to_tokens(slice, max_tokens);
        let body_end = start + body.len();
        Ok(Some(ReadResult {
            source: source.clone(),
            text: body.to_string(),
            start,
            end: body_end,
            first_line: line_of(text, start),
            last_line: line_of(text, body_end.saturating_sub(1).max(start)),
            truncated: cut.is_some(),
            next: cut.map(|_| ReadRange::Bytes(body_end, end)),
        }))
    }

    /// A recorded tool call by id, or `None` when unknown or not visible to
    /// the reader (its triggering turn or its result is hidden).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when a store read fails.
    pub async fn tool_call(&self, call_id: &str) -> Result<Option<ToolCallView>, UnikoError> {
        let rows = self
            .agent
            .kb()
            .actions(Some(call_id), None, None, 1)
            .await?;
        Ok(self.visible_calls(rows).await?.into_iter().next())
    }

    /// Recorded tool calls, newest first, optionally limited to one session
    /// and one tool; at most `limit` (0 = 50).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when a store read fails.
    pub async fn tool_calls(
        &self,
        session_id: Option<&str>,
        tool_name: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ToolCallView>, UnikoError> {
        let limit = if limit == 0 { 50 } else { limit };
        let rows = self
            .agent
            .kb()
            .actions(
                None,
                session_id,
                tool_name,
                i64::try_from(limit).unwrap_or(50),
            )
            .await?;
        self.visible_calls(rows).await
    }

    async fn visible_calls(&self, rows: Vec<ActionRow>) -> Result<Vec<ToolCallView>, UnikoError> {
        let kb = self.agent.kb();
        let data = self.data();
        let viewer = data.viewer().await?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            if let Some((mnid, _)) = &r.triggered_by
                && admitted_nodes(kb, viewer.as_ref(), &[(*mnid, RecallKind::Message)])
                    .await?
                    .is_empty()
            {
                continue;
            }
            if let Some((anid, _)) = &r.artifact {
                let chunks = kb.owned_chunk_ids(*anid).await?;
                let access = data.artifact_access(*anid, &chunks).await?;
                if !access.readable || (!chunks.is_empty() && access.chunks.is_empty()) {
                    continue;
                }
            }
            out.push(ToolCallView {
                call_id: r.action_id,
                tool_name: r.tool_name,
                action_type: r.action_type,
                arguments: r.input.as_ref().map(crate::value_convert::value_to_json),
                status: r.status,
                error: r.error,
                started_at: r.started_at.as_ref().and_then(|v| {
                    uniko_store::types::datetime_from_value(v, "Action.started_at").ok()
                }),
                ended_at: r.ended_at.as_ref().and_then(|v| {
                    uniko_store::types::datetime_from_value(v, "Action.ended_at").ok()
                }),
                triggered_by: r.triggered_by.map(|(_, id)| id),
                session_id: r.session_id,
                artifact_id: r.artifact.map(|(_, id)| id),
            });
        }
        Ok(out)
    }
}

/// 1-based line number of byte `at` in `text`.
fn line_of(text: &str, at: usize) -> usize {
    text[..at.min(text.len())]
        .bytes()
        .filter(|b| *b == b'\n')
        .count()
        + 1
}

/// Byte offset where 1-based line `n` starts.
fn line_start(text: &str, n: usize) -> Option<usize> {
    if n <= 1 {
        return Some(0);
    }
    text.match_indices('\n').nth(n - 2).map(|(i, _)| i + 1)
}

/// Byte offset where 1-based line `n` ends (before its newline).
fn line_end(text: &str, n: usize) -> usize {
    text.match_indices('\n')
        .nth(n.saturating_sub(1))
        .map_or(text.len(), |(i, _)| i)
}

/// Cut `text` to at most `max_tokens`, at the last line break that fits,
/// or the last character that does. Returns the kept prefix and, when cut,
/// its length.
fn cut_to_tokens(text: &str, max_tokens: usize) -> (&str, Option<usize>) {
    if count_tokens(text) <= max_tokens {
        return (text, None);
    }
    let mut best = 0;
    for (i, _) in text.match_indices('\n') {
        if count_tokens(&text[..i]) > max_tokens {
            break;
        }
        best = i;
    }
    if best == 0 {
        // No line fits: cut at characters (binary search on boundaries).
        let bounds: Vec<usize> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain([text.len()])
            .collect();
        let (mut lo, mut hi) = (0, bounds.len() - 1);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if count_tokens(&text[..bounds[mid]]) <= max_tokens {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        best = bounds[lo];
    }
    (&text[..best], Some(best))
}

/// Resolve `range` to a byte span of `text`.
fn resolve_range(
    text: &str,
    range: &ReadRange,
    chunks: &[ChunkRow],
) -> Result<Option<(usize, usize)>, UnikoError> {
    let clamp = |s: usize, e: usize| -> Option<(usize, usize)> {
        let mut s = s.min(text.len());
        let mut e = e.min(text.len());
        while !text.is_char_boundary(s) {
            s -= 1;
        }
        while !text.is_char_boundary(e) {
            e += 1;
        }
        (s < e).then_some((s, e))
    };
    Ok(match range {
        ReadRange::Bytes(s, e) => clamp(*s, *e),
        ReadRange::Lines(a, b) => {
            line_start(text, *a).and_then(|s| clamp(s, line_end(text, (*b).max(*a))))
        }
        ReadRange::Chunks(a, b) => {
            let picked: Vec<&ChunkRow> = chunks
                .iter()
                .filter(|c| (*a as i64..=*b as i64).contains(&c.index))
                .collect();
            match (picked.first(), picked.last()) {
                (Some(f), Some(l)) => clamp(
                    usize::try_from(f.start).unwrap_or(0),
                    usize::try_from(l.end).unwrap_or(0),
                ),
                _ => None,
            }
        }
        ReadRange::Address(addr) => resolve_address(text, addr.trim())?,
    })
}

/// Parse `"a-b"` (or `"a"`) into an inclusive pair.
fn parse_span(s: &str) -> Option<(usize, usize)> {
    let s = s.trim();
    match s.split_once('-') {
        Some((a, b)) => Some((a.trim().parse().ok()?, b.trim().parse().ok()?)),
        None => s.parse().ok().map(|n| (n, n)),
    }
}

fn resolve_address(text: &str, addr: &str) -> Result<Option<(usize, usize)>, UnikoError> {
    // A diff hunk address carries its line range after the comma.
    let addr = addr.rsplit_once(", ").map_or(addr, |(_, rest)| rest);
    if let Some(span) = addr.strip_prefix("lines ") {
        let (a, b) = parse_span(span).ok_or_else(|| bad_address(addr))?;
        return Ok(line_start(text, a).map(|s| (s, line_end(text, b.max(a)))));
    }
    if let Some(span) = addr.strip_prefix("records ") {
        let (a, b) = parse_span(span).ok_or_else(|| bad_address(addr))?;
        return Ok(nth_nonblank_lines(text, a, b));
    }
    if let Some(span) = addr.strip_prefix("rows ") {
        // Row 1 is the first line after the header.
        let (a, b) = parse_span(span).ok_or_else(|| bad_address(addr))?;
        return Ok(nth_nonblank_lines(text, a + 1, b + 1));
    }
    if addr.starts_with('/') {
        let root = sjson::parse(text).ok_or_else(|| {
            UnikoError::Config("address is a JSON pointer but the source is not JSON".into())
        })?;
        return Ok(resolve_pointer(&root, addr));
    }
    Err(bad_address(addr))
}

fn bad_address(addr: &str) -> UnikoError {
    UnikoError::Config(format!(
        "cannot read address '{addr}': expected a JSON pointer, 'rows a-b', 'lines a-b' or 'records a-b'"
    ))
}

/// Span of non-blank lines `a..=b` (1-based, counting only non-blank).
fn nth_nonblank_lines(text: &str, a: usize, b: usize) -> Option<(usize, usize)> {
    let mut offset = 0;
    let mut n = 0;
    let (mut start, mut end) = (None, None);
    for line in text.split_inclusive('\n') {
        let body = line.trim_end();
        if !body.trim().is_empty() {
            n += 1;
            if n == a {
                start = Some(offset);
            }
            if n >= a && n <= b {
                end = Some(offset + body.len());
            }
        }
        offset += line.len();
    }
    start.zip(end)
}

/// Resolve a JSON pointer, with an optional trailing `[a-b]` element range
/// on the last array (`/data/items[3-9]`) or a trailing `{…}` member list.
fn resolve_pointer(root: &JsonNode, addr: &str) -> Option<(usize, usize)> {
    let addr = addr.split(" {").next().unwrap_or(addr);
    let (path, range) = match addr.rfind('[') {
        Some(i) if addr.ends_with(']') => (&addr[..i], parse_span(&addr[i + 1..addr.len() - 1])),
        _ => (addr, None),
    };
    let mut node = root;
    for token in path.split('/').skip(1).filter(|t| !t.is_empty()) {
        let token = token.replace("~1", "/").replace("~0", "~");
        node = match &node.kind {
            JsonKind::Object(m) => &m.iter().find(|(k, _)| *k == token)?.1,
            JsonKind::Array(items) => items.get(token.parse::<usize>().ok()?)?,
            JsonKind::Scalar => return None,
        };
    }
    match (range, &node.kind) {
        (Some((a, b)), JsonKind::Array(items)) => {
            let first = items.get(a)?;
            let last = items.get(b.min(items.len().saturating_sub(1)))?;
            Some((first.start, last.end))
        }
        (Some(_), _) => None,
        (None, _) => Some((node.start, node.end)),
    }
}

fn json_outline(
    text: &str,
    node: &JsonNode,
    path: &mut Vec<String>,
    depth: usize,
    max_depth: usize,
    out: &mut Vec<OutlineEntry>,
) {
    let address = if path.is_empty() {
        "/".to_string()
    } else {
        path.iter().map(|t| format!("/{t}")).collect()
    };
    let (kind, summary) = match &node.kind {
        JsonKind::Object(m) => {
            let keys: Vec<&str> = m.iter().take(12).map(|(k, _)| k.as_str()).collect();
            let more = if m.len() > 12 { ", …" } else { "" };
            (
                "object",
                format!("{} keys: {}{more}", m.len(), keys.join(", ")),
            )
        }
        JsonKind::Array(items) => {
            let objects = items
                .iter()
                .filter(|i| matches!(i.kind, JsonKind::Object(_)))
                .count();
            let what = if objects == items.len() && !items.is_empty() {
                "objects"
            } else {
                "values"
            };
            ("array", format!("{} {what}", items.len()))
        }
        JsonKind::Scalar => {
            let v: String = text[node.start..node.end].chars().take(60).collect();
            ("scalar", v)
        }
    };
    out.push(OutlineEntry {
        address,
        kind: kind.into(),
        summary,
        start: node.start,
        end: node.end,
        depth,
    });
    if depth + 1 >= max_depth {
        return;
    }
    match &node.kind {
        JsonKind::Object(m) => {
            for (k, child) in m {
                if matches!(child.kind, JsonKind::Scalar) {
                    continue;
                }
                path.push(sjson::pointer_token(k));
                json_outline(text, child, path, depth + 1, max_depth, out);
                path.pop();
            }
        }
        JsonKind::Array(items) => {
            // An array of records is described by its first element.
            if let Some(first) = items.first()
                && !matches!(first.kind, JsonKind::Scalar)
            {
                path.push("0".into());
                json_outline(text, first, path, depth + 1, max_depth, out);
                path.pop();
            }
        }
        JsonKind::Scalar => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "{\n  \"meta\": {\"currency\": \"EUR\"},\n  \"data\": {\"items\": [\n    {\"id\": 1},\n    {\"id\": 2},\n    {\"id\": 3}\n  ]}\n}";

    #[test]
    fn pointers_and_element_ranges_resolve_to_exact_spans() {
        let root = sjson::parse(DOC).unwrap();
        let (s, e) = resolve_pointer(&root, "/data/items/1").unwrap();
        assert_eq!(&DOC[s..e], "{\"id\": 2}");
        let (s, e) = resolve_pointer(&root, "/data/items[1-2]").unwrap();
        assert_eq!(&DOC[s..e], "{\"id\": 2},\n    {\"id\": 3}");
        let (s, e) = resolve_pointer(&root, "/meta {currency}").unwrap();
        assert_eq!(&DOC[s..e], "{\"currency\": \"EUR\"}");
        assert!(resolve_pointer(&root, "/data/nope").is_none());
    }

    #[test]
    fn rows_lines_and_records_resolve() {
        let csv = "a,b\n1,2\n3,4\n\n5,6\n";
        let (s, e) = resolve_address(csv, "rows 2-3").unwrap().unwrap();
        assert_eq!(&csv[s..e], "3,4\n\n5,6");
        let (s, e) = resolve_address(csv, "lines 2-2").unwrap().unwrap();
        assert_eq!(&csv[s..e], "1,2");
        assert!(resolve_address(csv, "somewhere").is_err());
    }

    #[test]
    fn cutting_keeps_whole_lines_and_reports_the_rest() {
        let text: String = (0..200).map(|i| format!("line number {i}\n")).collect();
        let (body, cut) = cut_to_tokens(&text, 50);
        assert!(cut.is_some());
        assert!(body.ends_with(char::is_numeric));
        assert!(count_tokens(body) <= 50);
    }

    #[test]
    fn outline_walks_objects_and_first_record() {
        let root = sjson::parse(DOC).unwrap();
        let mut out = Vec::new();
        json_outline(DOC, &root, &mut Vec::new(), 0, 4, &mut out);
        let addrs: Vec<&str> = out.iter().map(|e| e.address.as_str()).collect();
        assert_eq!(
            addrs,
            ["/", "/meta", "/data", "/data/items", "/data/items/0"][..addrs.len()]
        );
        assert!(out[3].summary.starts_with("3 objects"));
    }
}
