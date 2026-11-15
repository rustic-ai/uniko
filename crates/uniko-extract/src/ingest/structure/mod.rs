//! Structure-aware chunking for tool output and other structured text.
//!
//! A record in a tool's result rarely means much on its own: an order's
//! `amount` needs the `currency` declared once in `meta`, a CSV row needs
//! its header, a failing test line needs the commit the suite ran on —
//! which may only be in the tool's arguments. Fixed-size chunking keeps
//! the record and drops that context.
//!
//! This module detects the payload's format ([`detect_format`]), splits it
//! along its own structure — JSON values by pointer, CSV and table rows,
//! log and output lines, diff hunks — and gives every chunk:
//!
//! - `text`: an exact slice of the payload, never rewritten;
//! - an `address` locating it (`/data/items[3-9]`, `rows 10-19`,
//!   `lines 120-180`);
//! - a `context` header: the tool and its arguments, the address, and the
//!   values the slice inherits from outside itself — the scalar fields of
//!   every enclosing JSON object and their small sibling objects, a table's
//!   header row — capped at a token budget.
//!
//! The context header is embedded and indexed with the chunk; the text a
//! reader is handed stays verbatim.

pub mod json;
pub mod normalize;

use serde_json::Value as JsonValue;

use self::json::{JsonKind, JsonNode, pointer_token, scalar_text};
use super::chunking::{ChunkConfig, ChunkData, count_tokens, text::chunk_text};

/// What a payload looks like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// One JSON value.
    Json,
    /// One JSON value per line.
    Ndjson,
    /// Delimiter-separated rows with a header row.
    Delimited(u8),
    /// A Markdown pipe table, possibly with text around it.
    MarkdownTable,
    /// A unified diff.
    Diff,
    /// Timestamped or levelled log lines.
    Log,
    /// Anything else: prose, test output, code.
    Text,
}

impl Format {
    /// Stable name, as stored in chunk metadata.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Ndjson => "ndjson",
            Self::Delimited(_) => "delimited",
            Self::MarkdownTable => "markdown_table",
            Self::Diff => "diff",
            Self::Log => "log",
            Self::Text => "text",
        }
    }
}

/// Fill `Chunk.search_text` for chunks written before it existed, in
/// batches of `batch` (0 = 500). Idempotent; returns how many were filled.
///
/// Like the store's other migrations this is manual: run it once after
/// upgrading a store, or the new lexical channel only sees new chunks.
///
/// # Errors
///
/// Returns [`uniko_store::UnikoError::Storage`] on database failure.
pub async fn backfill_chunk_search_text(
    kb: &uniko_store::KnowledgeBase,
    batch: usize,
) -> uniko_store::Result<usize> {
    let batch = if batch == 0 { 500 } else { batch };
    let mut total = 0;
    loop {
        let rows = kb.chunks_missing_search_text(batch).await?;
        if rows.is_empty() {
            return Ok(total);
        }
        let updates: Vec<(uniko_store::NodeId, String)> = rows
            .into_iter()
            .map(|(nid, text, ctx)| {
                let searchable = match ctx {
                    Some(c) => format!("{c}\n{text}"),
                    None => text,
                };
                (nid, normalize::normalize_for_search(&searchable))
            })
            .collect();
        total += kb.set_chunk_search_text(&updates).await?;
    }
}

/// Largest payload parsed as JSON; larger ones are chunked as text.
const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;

/// Detect `content`'s format. `declared` is a MIME type or legacy content
/// type the caller supplied (`"application/json"`, `"csv"`, …): a hint, and
/// never trusted over what the content actually parses as.
#[must_use]
pub fn detect_format(content: &str, declared: Option<&str>) -> Format {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Format::Text;
    }
    if matches!(trimmed.as_bytes()[0], b'{' | b'[')
        && trimmed.len() <= MAX_JSON_BYTES
        && json::parse(trimmed).is_some()
    {
        return Format::Json;
    }
    let lines: Vec<&str> = trimmed
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(50)
        .collect();
    if lines.len() >= 2
        && lines
            .iter()
            .take(20)
            .all(|l| l.trim_start().starts_with('{') && json::parse(l.trim()).is_some())
    {
        return Format::Ndjson;
    }
    if lines
        .iter()
        .any(|l| l.starts_with("diff --git ") || l.starts_with("@@ "))
        && lines
            .iter()
            .any(|l| l.starts_with("--- ") || l.starts_with("+++ "))
    {
        return Format::Diff;
    }
    if markdown_table_at(&lines).is_some() {
        return Format::MarkdownTable;
    }
    let declared_csv = declared.is_some_and(|d| {
        let d = d.to_ascii_lowercase();
        d.contains("csv") || d.contains("tab-separated") || d.contains("tsv")
    });
    if let Some(delim) = sniff_delimiter(&lines, declared_csv) {
        return Format::Delimited(delim);
    }
    let logish = lines.iter().filter(|l| looks_like_log_line(l)).count();
    if lines.len() >= 3 && logish * 10 >= lines.len() * 6 {
        return Format::Log;
    }
    Format::Text
}

fn looks_like_log_line(line: &str) -> bool {
    let l = line.trim_start_matches('[').as_bytes();
    let date = l.len() >= 16
        && l[..4].iter().all(u8::is_ascii_digit)
        && l[4] == b'-'
        && l[7] == b'-'
        && matches!(l[10], b'T' | b' ')
        && l[13] == b':';
    let time =
        l.len() >= 8 && l[..2].iter().all(u8::is_ascii_digit) && l[2] == b':' && l[5] == b':';
    let level = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
        .iter()
        .any(|lv| line.starts_with(lv));
    date || time || level
}

/// A delimiter that splits every one of the first lines into the same
/// number (≥ 2) of fields, ignoring quoted fields.
fn sniff_delimiter(lines: &[&str], declared: bool) -> Option<u8> {
    if lines.len() < 2 && !declared {
        return None;
    }
    let sample: Vec<&str> = lines.iter().take(20).copied().collect();
    b",\t;|".iter().copied().find(|&d| {
        let counts: Vec<usize> = sample.iter().map(|l| field_count(l, d)).collect();
        let first = counts[0];
        first >= 2 && counts.iter().all(|c| *c == first)
    })
}

fn field_count(line: &str, delim: u8) -> usize {
    let mut n = 1;
    let mut quoted = false;
    for b in line.bytes() {
        match b {
            b'"' => quoted = !quoted,
            d if d == delim && !quoted => n += 1,
            _ => {}
        }
    }
    n
}

/// Index of a Markdown table's header line: a `|`-led line followed by a
/// `|---|` separator line.
fn markdown_table_at(lines: &[&str]) -> Option<usize> {
    lines.windows(2).position(|w| {
        let sep = w[1].trim();
        w[0].trim_start().starts_with('|')
            && sep.starts_with('|')
            && sep.contains('-')
            && sep.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
    })
}

/// Where a payload came from, for its chunks' context header.
#[derive(Debug, Clone, Default)]
pub struct PayloadContext {
    /// The tool that produced it.
    pub tool_name: Option<String>,
    /// The arguments it was called with.
    pub tool_arguments: Option<JsonValue>,
    /// Token budget for one chunk's context header.
    pub max_context_tokens: usize,
}

impl PayloadContext {
    /// Read `tool_name` / `tool_arguments` from ingest metadata.
    #[must_use]
    pub fn from_metadata(
        metadata: &std::collections::HashMap<String, JsonValue>,
        max_context_tokens: usize,
    ) -> Self {
        Self {
            tool_name: metadata
                .get("tool_name")
                .and_then(JsonValue::as_str)
                .map(str::to_string),
            tool_arguments: metadata.get("tool_arguments").cloned(),
            max_context_tokens,
        }
    }

    fn tool_line(&self) -> Option<String> {
        let name = self.tool_name.as_deref()?;
        let args = self
            .tool_arguments
            .as_ref()
            .map(|a| {
                let s = a.to_string();
                if s.chars().count() > 240 {
                    format!("{}…", s.chars().take(240).collect::<String>())
                } else {
                    s
                }
            })
            .unwrap_or_default();
        Some(if args.is_empty() {
            format!("tool: {name}")
        } else {
            format!("tool: {name} {args}")
        })
    }
}

/// One chunk before it becomes [`ChunkData`].
struct Piece {
    start: usize,
    end: usize,
    address: String,
    /// `key=value` pairs the slice inherits, nearest first.
    inherited: Vec<String>,
    /// Lines that always lead the header after the tool line (a table's
    /// header row).
    lead: Vec<String>,
}

/// Chunk `content` along its structure. Every chunk's `text` is
/// `content[start..end]`; `metadata` carries `address`, `context` and
/// `format`.
#[must_use]
pub fn chunk_structured(
    content: &str,
    format: Format,
    ctx: &PayloadContext,
    config: &ChunkConfig,
) -> Vec<ChunkData> {
    let max = config.max_chunk_tokens.max(1);
    let pieces = match format {
        Format::Json => json::parse(content).map_or_else(
            || line_pieces(content, config, "lines"),
            |root| {
                let mut out = Vec::new();
                json_pieces(
                    content,
                    &root,
                    &mut Vec::new(),
                    &mut Vec::new(),
                    max,
                    &mut out,
                );
                out
            },
        ),
        Format::Ndjson => ndjson_pieces(content, max),
        Format::Delimited(d) => delimited_pieces(content, d, max),
        Format::MarkdownTable => markdown_pieces(content, config),
        Format::Diff => diff_pieces(content, config),
        Format::Log | Format::Text => line_pieces(content, config, "lines"),
    };
    pieces
        .into_iter()
        .enumerate()
        .map(|(index, p)| {
            let text = &content[p.start..p.end];
            let context = header(ctx, &p);
            ChunkData {
                text: text.to_string(),
                index,
                start: p.start,
                end: p.end,
                token_count: count_tokens(text),
                chunk_type: format!("tool_{}", format.as_str()),
                language: None,
                symbol_name: None,
                heading: None,
                metadata: Some(serde_json::json!({
                    "format": format.as_str(),
                    "address": p.address,
                    "context": context,
                })),
            }
        })
        .collect()
}

/// Build a piece's context header within the token budget: the tool line,
/// the address, lead lines, then inherited values nearest first.
fn header(ctx: &PayloadContext, p: &Piece) -> String {
    let budget = if ctx.max_context_tokens == 0 {
        64
    } else {
        ctx.max_context_tokens
    };
    let mut lines: Vec<String> = Vec::new();
    if let Some(tool) = ctx.tool_line() {
        lines.push(tool);
    }
    lines.push(format!("at: {}", p.address));
    lines.extend(p.lead.iter().cloned());
    let mut out = lines.join("\n");
    if count_tokens(&out) > budget {
        // Even the fixed part is over budget: keep it, truncated by chars.
        let keep: String = out.chars().take(budget * 4).collect();
        return keep;
    }
    let mut inherited = String::new();
    for kv in &p.inherited {
        let candidate = if inherited.is_empty() {
            kv.clone()
        } else {
            format!("{inherited}; {kv}")
        };
        if count_tokens(&format!("{out}\n{candidate}")) > budget {
            break;
        }
        inherited = candidate;
    }
    if !inherited.is_empty() {
        out.push('\n');
        out.push_str(&inherited);
    }
    out
}

// ── JSON ────────────────────────────────────────────────────────────────

fn pointer(path: &[String]) -> String {
    if path.is_empty() {
        "/".to_string()
    } else {
        path.iter().map(|t| format!("/{t}")).collect()
    }
}

/// Scalars an object contributes to the context of its children: its own
/// scalar members, and the scalar members of its small object members, by
/// dotted key. `skip` is the member being descended into.
fn object_scalars(
    content: &str,
    members: &[(String, JsonNode)],
    prefix: &str,
    skip: Option<&str>,
) -> Vec<String> {
    const MAX_VALUE_CHARS: usize = 80;
    let render = |n: &JsonNode| {
        let v = scalar_text(&content[n.start..n.end]);
        if v.chars().count() > MAX_VALUE_CHARS {
            format!("{}…", v.chars().take(MAX_VALUE_CHARS).collect::<String>())
        } else {
            v
        }
    };
    let mut out = Vec::new();
    for (key, node) in members {
        if Some(key.as_str()) == skip {
            continue;
        }
        let name = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match &node.kind {
            JsonKind::Scalar => out.push(format!("{name}={}", render(node))),
            JsonKind::Object(sub) if sub.len() <= 16 => {
                for (k, n) in sub {
                    if matches!(n.kind, JsonKind::Scalar) {
                        out.push(format!("{name}.{k}={}", render(n)));
                    }
                }
            }
            _ => {}
        }
    }
    out
}

fn json_pieces(
    content: &str,
    node: &JsonNode,
    path: &mut Vec<String>,
    inherited: &mut Vec<String>,
    max: usize,
    out: &mut Vec<Piece>,
) {
    let fits = |s: usize, e: usize| count_tokens(&content[s..e]) <= max;
    if fits(node.start, node.end) {
        out.push(Piece {
            start: node.start,
            end: node.end,
            address: pointer(path),
            inherited: inherited.clone(),
            lead: Vec::new(),
        });
        return;
    }
    match &node.kind {
        JsonKind::Object(members) => {
            // Small members are grouped in source order; a member too big
            // for one chunk is descended into, inheriting this object's
            // other scalars.
            let mut group: Option<(usize, usize, Vec<&str>)> = None;
            let flush = |group: &mut Option<(usize, usize, Vec<&str>)>,
                         out: &mut Vec<Piece>,
                         path: &[String],
                         inherited: &[String]| {
                if let Some((s, e, keys)) = group.take() {
                    out.push(Piece {
                        start: s,
                        end: e,
                        address: format!("{} {{{}}}", pointer(path), keys.join(", ")),
                        inherited: inherited.to_vec(),
                        lead: Vec::new(),
                    });
                }
            };
            for (key, child) in members {
                if fits(child.start, child.end) {
                    let joined = group.as_ref().is_some_and(|(s, _, _)| fits(*s, child.end));
                    if !joined {
                        flush(&mut group, out, path, inherited);
                    }
                    match group.as_mut() {
                        Some((_, e, keys)) => {
                            *e = child.end;
                            keys.push(key);
                        }
                        None => group = Some((child.start, child.end, vec![key])),
                    }
                } else {
                    flush(&mut group, out, path, inherited);
                    let mut child_inherited = object_scalars(
                        content,
                        members,
                        &pointer(path).trim_start_matches('/').replace('/', "."),
                        Some(key),
                    );
                    child_inherited.extend(inherited.iter().cloned());
                    path.push(pointer_token(key));
                    json_pieces(content, child, path, &mut child_inherited, max, out);
                    path.pop();
                }
            }
            flush(&mut group, out, path, inherited);
        }
        JsonKind::Array(items) => {
            let mut i = 0;
            while i < items.len() {
                if !fits(items[i].start, items[i].end) {
                    path.push(i.to_string());
                    json_pieces(content, &items[i], path, inherited, max, out);
                    path.pop();
                    i += 1;
                    continue;
                }
                let first = i;
                let mut last = i;
                while last + 1 < items.len() && fits(items[first].start, items[last + 1].end) {
                    last += 1;
                }
                out.push(Piece {
                    start: items[first].start,
                    end: items[last].end,
                    address: if first == last {
                        format!("{}/{first}", pointer(path).trim_end_matches('/'))
                    } else {
                        format!("{}[{first}-{last}]", pointer(path))
                    },
                    inherited: inherited.clone(),
                    lead: Vec::new(),
                });
                i = last + 1;
            }
        }
        JsonKind::Scalar => {
            // One enormous string: split its text, keeping offsets.
            let cfg = ChunkConfig {
                max_chunk_tokens: max,
                min_chunk_tokens: 0,
                overlap_tokens: 0,
            };
            for c in chunk_text(&content[node.start..node.end], &cfg) {
                out.push(Piece {
                    start: node.start + c.start,
                    end: node.start + c.end,
                    address: pointer(path),
                    inherited: inherited.clone(),
                    lead: Vec::new(),
                });
            }
        }
    }
}

fn ndjson_pieces(content: &str, max: usize) -> Vec<Piece> {
    let lines = line_spans(content);
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let first = i;
        let mut last = i;
        while last + 1 < lines.len()
            && count_tokens(&content[lines[first].0..lines[last + 1].1]) <= max
        {
            last += 1;
        }
        out.push(Piece {
            start: lines[first].0,
            end: lines[last].1,
            address: format!("records {}-{}", first + 1, last + 1),
            inherited: Vec::new(),
            lead: Vec::new(),
        });
        i = last + 1;
    }
    out
}

// ── Rows and lines ──────────────────────────────────────────────────────

/// `(start, end)` of every non-blank line, trailing whitespace excluded.
fn line_spans(content: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut offset = 0;
    for line in content.split_inclusive('\n') {
        let body = line.trim_end();
        let lead = body.len() - body.trim_start().len();
        if !body.trim().is_empty() {
            out.push((offset + lead, offset + body.len()));
        }
        offset += line.len();
    }
    out
}

/// 1-based line number of byte `at`.
fn line_of(content: &str, at: usize) -> usize {
    content[..at].bytes().filter(|b| *b == b'\n').count() + 1
}

fn delimited_pieces(content: &str, delim: u8, max: usize) -> Vec<Piece> {
    let lines = line_spans(content);
    let Some(&(hs, he)) = lines.first() else {
        return Vec::new();
    };
    let header_row = &content[hs..he];
    let columns: Vec<String> = header_row
        .split(char::from(delim))
        .map(|c| c.trim().trim_matches('"').to_string())
        .collect();
    let lead = vec![format!("columns: {}", columns.join(", "))];
    let rows = &lines[1..];
    let mut out = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        let first = i;
        let mut last = i;
        while last + 1 < rows.len()
            && count_tokens(&content[rows[first].0..rows[last + 1].1]) <= max
        {
            last += 1;
        }
        out.push(Piece {
            start: rows[first].0,
            end: rows[last].1,
            address: format!("rows {}-{}", first + 1, last + 1),
            inherited: Vec::new(),
            lead: lead.clone(),
        });
        i = last + 1;
    }
    out
}

fn markdown_pieces(content: &str, config: &ChunkConfig) -> Vec<Piece> {
    let lines = line_spans(content);
    let texts: Vec<&str> = lines.iter().map(|(s, e)| &content[*s..*e]).collect();
    let Some(h) = markdown_table_at(&texts) else {
        return line_pieces(content, config, "lines");
    };
    let header_row = texts[h].trim().to_string();
    // Rows run until the first line that is not a table row.
    let body_end = (h + 2..lines.len())
        .find(|&i| !texts[i].trim_start().starts_with('|'))
        .unwrap_or(lines.len());
    let mut out = Vec::new();
    if h > 0 {
        out.extend(span_line_pieces(
            content,
            lines[0].0,
            lines[h - 1].1,
            config,
        ));
    }
    let max = config.max_chunk_tokens.max(1);
    let rows = &lines[h + 2..body_end];
    let mut i = 0;
    while i < rows.len() {
        let first = i;
        let mut last = i;
        while last + 1 < rows.len()
            && count_tokens(&content[rows[first].0..rows[last + 1].1]) <= max
        {
            last += 1;
        }
        out.push(Piece {
            start: rows[first].0,
            end: rows[last].1,
            address: format!("rows {}-{}", first + 1, last + 1),
            inherited: Vec::new(),
            lead: vec![format!("columns: {header_row}")],
        });
        i = last + 1;
    }
    if body_end < lines.len() {
        out.extend(span_line_pieces(
            content,
            lines[body_end].0,
            lines[lines.len() - 1].1,
            config,
        ));
    }
    out
}

/// Text-chunk `[start, end)` and address each piece by its line range.
fn span_line_pieces(content: &str, start: usize, end: usize, config: &ChunkConfig) -> Vec<Piece> {
    let cfg = ChunkConfig {
        overlap_tokens: 0,
        ..config.clone()
    };
    chunk_text(&content[start..end], &cfg)
        .into_iter()
        .map(|c| {
            let (s, e) = (start + c.start, start + c.end);
            Piece {
                start: s,
                end: e,
                address: format!("lines {}-{}", line_of(content, s), line_of(content, e)),
                inherited: Vec::new(),
                lead: Vec::new(),
            }
        })
        .collect()
}

fn line_pieces(content: &str, config: &ChunkConfig, _kind: &str) -> Vec<Piece> {
    let Some((first, last)) = line_spans(content)
        .first()
        .copied()
        .zip(line_spans(content).last().copied())
    else {
        return Vec::new();
    };
    span_line_pieces(content, first.0, last.1, config)
}

fn diff_pieces(content: &str, config: &ChunkConfig) -> Vec<Piece> {
    // Hunk boundaries: each `@@` line starts a hunk; the `---`/`+++` pair
    // before it names the file and leads the hunk's header.
    let lines = line_spans(content);
    let mut out = Vec::new();
    let mut file = String::new();
    let mut hunk_start: Option<usize> = None;
    let mut hunk_no = 0;
    let max = config.max_chunk_tokens.max(1);
    let flush = |from: usize, to: usize, file: &str, hunk_no: usize, out: &mut Vec<Piece>| {
        for p in span_line_pieces(
            content,
            from,
            to,
            &ChunkConfig {
                max_chunk_tokens: max,
                ..config.clone()
            },
        ) {
            out.push(Piece {
                address: format!("hunk {hunk_no}, {}", p.address),
                lead: if file.is_empty() {
                    Vec::new()
                } else {
                    vec![format!("file: {file}")]
                },
                ..p
            });
        }
    };
    for (i, &(s, e)) in lines.iter().enumerate() {
        let text = &content[s..e];
        let starts_file = text.starts_with("diff --git ") || text.starts_with("--- ");
        let starts_hunk = text.starts_with("@@ ");
        if (starts_file || starts_hunk)
            && let Some(hs) = hunk_start.take()
        {
            flush(hs, lines[i - 1].1, &file, hunk_no, &mut out);
        }
        if let Some(name) = text.strip_prefix("+++ ") {
            file = name.trim_start_matches("b/").to_string();
        }
        if starts_hunk {
            hunk_no += 1;
            hunk_start = Some(s);
        }
    }
    if let Some(hs) = hunk_start {
        flush(hs, lines[lines.len() - 1].1, &file, hunk_no, &mut out);
    }
    if out.is_empty() {
        return line_pieces(content, config, "lines");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max: usize) -> ChunkConfig {
        ChunkConfig {
            max_chunk_tokens: max,
            min_chunk_tokens: 0,
            overlap_tokens: 0,
        }
    }

    fn ctx() -> PayloadContext {
        PayloadContext {
            tool_name: Some("list_orders".into()),
            tool_arguments: Some(serde_json::json!({"customer_id": "C-1042"})),
            max_context_tokens: 64,
        }
    }

    fn meta<'a>(c: &'a ChunkData, key: &str) -> &'a str {
        c.metadata.as_ref().unwrap()[key].as_str().unwrap()
    }

    fn orders_json(n: usize) -> String {
        let items: Vec<JsonValue> = (0..n)
            .map(|i| serde_json::json!({"order_id": format!("ORD-{}", 10_000 + i), "status": "shipped", "amount": 100 + i}))
            .collect();
        serde_json::to_string_pretty(&serde_json::json!({
            "meta": {"currency": "EUR", "amount_unit": "cents", "page": 2},
            "data": {"items": items}
        }))
        .unwrap()
    }

    #[test]
    fn detects_formats() {
        assert_eq!(detect_format(&orders_json(2), None), Format::Json);
        assert_eq!(
            detect_format("{\"a\":1}\n{\"a\":2}\n", None),
            Format::Ndjson
        );
        assert_eq!(
            detect_format(
                "region,month,revenue\nAPAC,2026-01,10\nNA,2026-01,12\n",
                None
            ),
            Format::Delimited(b',')
        );
        assert_eq!(
            detect_format("a\tb\n1\t2\n", None),
            Format::Delimited(b'\t')
        );
        assert_eq!(
            detect_format("| a | b |\n|---|---|\n| 1 | 2 |\n", None),
            Format::MarkdownTable
        );
        assert_eq!(
            detect_format(
                "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
                None
            ),
            Format::Diff
        );
        let log = "2026-10-01T12:00:00Z INFO start\n2026-10-01T12:00:01Z ERROR boom\n2026-10-01T12:00:02Z INFO done\n";
        assert_eq!(detect_format(log, None), Format::Log);
        assert_eq!(
            detect_format("Just a sentence. And another one.", None),
            Format::Text
        );
        assert_eq!(detect_format("{ not json", None), Format::Text);
    }

    #[test]
    fn json_records_are_verbatim_and_inherit_meta() {
        let src = orders_json(40);
        let chunks = chunk_structured(&src, Format::Json, &ctx(), &cfg(120));
        assert!(chunks.len() > 2, "expected the items array to be split");
        for c in &chunks {
            assert_eq!(c.text, &src[c.start..c.end], "verbatim");
        }
        let item_chunk = chunks
            .iter()
            .find(|c| meta(c, "address").starts_with("/data/items["))
            .expect("an items range");
        let context = meta(item_chunk, "context");
        assert!(
            context.starts_with("tool: list_orders {\"customer_id\":\"C-1042\"}"),
            "{context}"
        );
        assert!(context.contains("meta.currency=EUR"), "{context}");
        assert!(context.contains("meta.amount_unit=cents"), "{context}");
        assert!(count_tokens(context) <= 64, "{context}");
    }

    #[test]
    fn csv_rows_carry_their_header() {
        let mut src = String::from("region,month,revenue\n");
        for i in 0..200 {
            src.push_str(&format!("APAC,2026-{:02},{}\n", i % 12 + 1, 1000 + i));
        }
        let chunks = chunk_structured(&src, Format::Delimited(b','), &ctx(), &cfg(60));
        assert!(chunks.len() > 3);
        for c in &chunks {
            assert_eq!(c.text, &src[c.start..c.end]);
            assert!(
                !c.text.contains("region,month"),
                "header is context, not text"
            );
            assert!(meta(c, "context").contains("columns: region, month, revenue"));
            assert!(meta(c, "address").starts_with("rows "));
        }
    }

    #[test]
    fn log_lines_are_addressed_by_line() {
        let src: String = (0..300)
            .map(|i| {
                format!(
                    "2026-10-01T12:00:{:02}Z INFO payments request_id=r{i} status=200\n",
                    i % 60
                )
            })
            .collect();
        let chunks = chunk_structured(&src, Format::Log, &ctx(), &cfg(80));
        assert!(chunks.len() > 3);
        assert!(meta(&chunks[0], "address").starts_with("lines 1-"));
        for c in &chunks {
            assert_eq!(c.text, &src[c.start..c.end]);
            assert!(c.text.ends_with("status=200"));
        }
    }

    #[test]
    fn diff_hunks_name_their_file() {
        let src = "diff --git a/src/x.rs b/src/x.rs\n--- a/src/x.rs\n+++ b/src/x.rs\n@@ -1,2 +1,2 @@\n-old\n+new\n@@ -10 +10 @@\n-a\n+b\n";
        let chunks = chunk_structured(src, Format::Diff, &ctx(), &cfg(200));
        assert_eq!(chunks.len(), 2);
        assert!(meta(&chunks[0], "context").contains("file: src/x.rs"));
        assert!(meta(&chunks[1], "address").starts_with("hunk 2"));
    }

    #[test]
    fn header_respects_budget() {
        let mut members = String::from("{\"meta\":{");
        for i in 0..16 {
            members.push_str(&format!(
                "\"field_{i}\":\"value number {i} with some words\","
            ));
        }
        members.push_str("\"z\":1},\"items\":[");
        members.push_str(
            &(0..60)
                .map(|i| format!("{{\"id\":{i},\"v\":\"x\"}}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        members.push_str("]}");
        let c = PayloadContext {
            max_context_tokens: 40,
            ..ctx()
        };
        for chunk in chunk_structured(&members, Format::Json, &c, &cfg(60)) {
            assert!(count_tokens(meta(&chunk, "context")) <= 40);
        }
    }
}
