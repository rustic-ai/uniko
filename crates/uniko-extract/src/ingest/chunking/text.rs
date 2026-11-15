//! Span-based text chunker.
//!
//! Every chunk is a verbatim slice of its source: `text ==
//! content[start..end]` always holds, so a chunk's offsets locate it exactly
//! and a reader can reassemble or extend it from the source.
//!
//! A span over `max_chunk_tokens` is split at the strongest boundary it
//! contains, recursively: blank lines (paragraphs), then line breaks, then
//! sentence ends, then whitespace, then characters. A sentence ends at `.`,
//! `!` or `?` only when whitespace follows, so `3.14`, `foo.rs` and
//! `v1.2.3` are never cut. The pieces are then packed greedily back into
//! chunks of at most `max_chunk_tokens`, a chunk under `min_chunk_tokens`
//! is merged into a neighbour, and overlap is taken by starting a chunk a
//! little before the previous one ends, never by copying text.

use super::{ChunkConfig, ChunkData, Chunker, count_tokens};

/// Recursive paragraph/line/sentence/word splitter for plain text and
/// markdown.
pub struct TextChunker;

impl Chunker for TextChunker {
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
        chunk_text(content, config)
    }
}

/// Chunk text content. See the module docs for the guarantees.
pub fn chunk_text(content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
    let Some((start, end)) = trimmed_span(content, 0, content.len()) else {
        return Vec::new();
    };
    let max = config.max_chunk_tokens.max(1);
    let headings = heading_lines(content);

    let total_tokens = count_tokens(content);
    let spans = if total_tokens <= max {
        vec![(start, end)]
    } else {
        let mut pieces = Vec::new();
        split_span(content, start, end, Level::Paragraph, max, &mut pieces);
        let mut spans = pack(content, &pieces, max);
        merge_small(content, &mut spans, config.min_chunk_tokens, max);
        spans
    };
    let overlapped = overlap_starts(content, &spans, config.overlap_tokens);

    spans
        .iter()
        .zip(overlapped)
        .enumerate()
        .map(|(index, (&(own_start, end), start))| {
            let text = &content[start..end];
            ChunkData {
                text: text.to_string(),
                index,
                start,
                end,
                token_count: count_tokens(text),
                chunk_type: "text".into(),
                language: None,
                symbol_name: None,
                heading: heading_for(&headings, own_start, end),
                metadata: None,
            }
        })
        .collect()
}

// ── Splitting ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Paragraph,
    Line,
    Sentence,
    Word,
    Char,
}

impl Level {
    fn next(self) -> Self {
        match self {
            Self::Paragraph => Self::Line,
            Self::Line => Self::Sentence,
            Self::Sentence => Self::Word,
            Self::Word | Self::Char => Self::Char,
        }
    }
}

/// `[start, end)` shrunk past leading and trailing whitespace, or `None`
/// when nothing but whitespace is left.
fn trimmed_span(content: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    let s = &content[start..end];
    let lead = s.len() - s.trim_start().len();
    let body = s.trim();
    (!body.is_empty()).then(|| (start + lead, start + lead + body.len()))
}

/// Split `[start, end)` into pieces of at most `max` tokens, preferring
/// the strongest boundary available at `level` and below.
fn split_span(
    content: &str,
    start: usize,
    end: usize,
    level: Level,
    max: usize,
    out: &mut Vec<(usize, usize)>,
) {
    if count_tokens(&content[start..end]) <= max {
        out.push((start, end));
        return;
    }
    if level == Level::Char {
        split_chars(content, start, end, max, out);
        return;
    }
    let parts = boundaries(content, start, end, level);
    if parts.len() <= 1 {
        split_span(content, start, end, level.next(), max, out);
        return;
    }
    for (s, e) in parts {
        split_span(content, s, e, level.next(), max, out);
    }
}

/// The non-blank sub-spans of `[start, end)` separated at `level`.
fn boundaries(content: &str, start: usize, end: usize, level: Level) -> Vec<(usize, usize)> {
    let text = &content[start..end];
    let mut cuts = Vec::new(); // byte offsets (relative) where a new part begins
    let bytes = text.as_bytes();
    match level {
        Level::Paragraph => {
            // A blank line: '\n', optional spaces/tabs/'\r', '\n'.
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'\n' {
                    let mut j = i + 1;
                    while j < bytes.len() && matches!(bytes[j], b' ' | b'\t' | b'\r') {
                        j += 1;
                    }
                    if j < bytes.len() && bytes[j] == b'\n' {
                        cuts.push(j + 1);
                        i = j + 1;
                        continue;
                    }
                }
                i += 1;
            }
        }
        Level::Line => {
            for (i, b) in bytes.iter().enumerate() {
                if *b == b'\n' {
                    cuts.push(i + 1);
                }
            }
        }
        Level::Sentence => {
            let mut prev_end = false;
            for (i, c) in text.char_indices() {
                if prev_end && c.is_whitespace() {
                    cuts.push(i);
                }
                prev_end = matches!(c, '.' | '!' | '?');
            }
        }
        Level::Word => {
            let mut prev_ws = false;
            for (i, c) in text.char_indices() {
                if !c.is_whitespace() && prev_ws {
                    cuts.push(i);
                }
                prev_ws = c.is_whitespace();
            }
        }
        Level::Char => {}
    }
    let mut parts = Vec::with_capacity(cuts.len() + 1);
    let mut from = 0;
    for cut in cuts.into_iter().chain([text.len()]) {
        if let Some(span) = trimmed_span(content, start + from, start + cut) {
            parts.push(span);
        }
        from = cut;
    }
    parts
}

/// Last resort for a run with no whitespace at all (minified JSON, a
/// base64 blob): cut at character boundaries into pieces of `max` tokens.
fn split_chars(content: &str, start: usize, end: usize, max: usize, out: &mut Vec<(usize, usize)>) {
    let mut from = start;
    while from < end {
        // Binary-search the longest prefix that fits, within a window no
        // token can exceed (a token is at most a few bytes of text per
        // character, so 16 bytes per token is generous).
        let mut window = (from + max * 16).min(end);
        while !content.is_char_boundary(window) {
            window += 1;
        }
        let chars: Vec<usize> = content[from..window]
            .char_indices()
            .map(|(i, _)| from + i)
            .chain([window])
            .collect();
        let (mut lo, mut hi) = (1, chars.len() - 1);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if count_tokens(&content[from..chars[mid]]) <= max {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let to = chars[lo];
        out.push((from, to));
        from = to;
    }
}

// ── Packing ─────────────────────────────────────────────────────────

/// Greedily join consecutive pieces into spans of at most `max` tokens.
fn pack(content: &str, pieces: &[(usize, usize)], max: usize) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut cur: Option<(usize, usize, usize)> = None; // start, end, tokens
    for &(s, e) in pieces {
        let t = count_tokens(&content[s..e]);
        cur = match cur {
            Some((cs, ce, ct)) if ct + t <= max => {
                // Joining may count differently at the seam; check exactly.
                let joined = count_tokens(&content[cs..e]);
                if joined <= max {
                    Some((cs, e, joined))
                } else {
                    spans.push((cs, ce));
                    Some((s, e, t))
                }
            }
            Some((cs, ce, _)) => {
                spans.push((cs, ce));
                Some((s, e, t))
            }
            None => Some((s, e, t)),
        };
    }
    if let Some((s, e, _)) = cur {
        spans.push((s, e));
    }
    spans
}

/// Merge a span under `min` tokens into its neighbour while the result
/// stays within `max` — the next one, or the previous one for the last.
fn merge_small(content: &str, spans: &mut Vec<(usize, usize)>, min: usize, max: usize) {
    let tokens = |s: (usize, usize)| count_tokens(&content[s.0..s.1]);
    let mut i = 0;
    while spans.len() > 1 && i < spans.len() {
        if tokens(spans[i]) >= min {
            i += 1;
            continue;
        }
        if i + 1 < spans.len() && tokens((spans[i].0, spans[i + 1].1)) <= max {
            spans[i].1 = spans[i + 1].1;
            spans.remove(i + 1);
        } else if i > 0 && tokens((spans[i - 1].0, spans[i].1)) <= max {
            spans[i - 1].1 = spans[i].1;
            spans.remove(i);
            i -= 1;
        } else {
            i += 1;
        }
    }
}

/// Each chunk's start after overlap: up to `overlap_tokens` of the
/// previous chunk's tail, beginning at a word boundary, never reaching
/// back before the previous chunk's own start.
fn overlap_starts(content: &str, spans: &[(usize, usize)], overlap_tokens: usize) -> Vec<usize> {
    spans
        .iter()
        .enumerate()
        .map(|(i, &(start, _))| {
            if i == 0 || overlap_tokens == 0 {
                return start;
            }
            let (prev_start, prev_end) = spans[i - 1];
            let tail = &content[prev_start..prev_end.min(start)];
            // Word starts in the previous chunk, nearest the seam first.
            let mut best = start;
            let mut prev_ws = true;
            let starts: Vec<usize> = tail
                .char_indices()
                .filter_map(|(j, c)| {
                    let at = (!c.is_whitespace() && prev_ws).then_some(prev_start + j);
                    prev_ws = c.is_whitespace();
                    at
                })
                .collect();
            for &w in starts.iter().rev() {
                if w == prev_start {
                    break;
                }
                if count_tokens(&content[w..prev_end.min(start)]) > overlap_tokens {
                    break;
                }
                best = w;
            }
            best
        })
        .collect()
}

// ── Headings ────────────────────────────────────────────────────────

/// `(byte offset, text)` of every markdown heading line.
fn heading_lines(content: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut offset = 0;
    for line in content.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            let h = trimmed.trim_start_matches('#').trim();
            if !h.is_empty() {
                out.push((offset, h.to_string()));
            }
        }
        offset += line.len();
    }
    out
}

/// The first heading inside `[start, end)`, else the last one before it.
fn heading_for(headings: &[(usize, String)], start: usize, end: usize) -> Option<String> {
    headings
        .iter()
        .find(|(at, _)| *at >= start && *at < end)
        .or_else(|| headings.iter().rev().find(|(at, _)| *at < start))
        .map(|(_, h)| h.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ChunkConfig {
        ChunkConfig {
            max_chunk_tokens: 50,
            min_chunk_tokens: 10,
            overlap_tokens: 5,
        }
    }

    fn assert_verbatim(content: &str, chunks: &[ChunkData]) {
        for c in chunks {
            assert_eq!(c.text, &content[c.start..c.end], "chunk {}", c.index);
            assert_eq!(c.token_count, count_tokens(&c.text));
        }
    }

    #[test]
    fn test_short_text_single_chunk() {
        let chunks = chunk_text("Hello world.", &cfg());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].index, 0);
        assert!(chunks[0].text.contains("Hello"));
    }

    #[test]
    fn test_empty_text() {
        assert!(chunk_text("", &cfg()).is_empty());
        assert!(chunk_text(" \n\n\t ", &cfg()).is_empty());
    }

    #[test]
    fn test_paragraph_split() {
        let para1 = "The quick brown fox jumps over the lazy dog. ".repeat(3);
        let para2 = "A second paragraph with different content here. ".repeat(3);
        let content = format!("{para1}\n\n{para2}");
        let chunks = chunk_text(&content, &cfg());
        assert!(chunks.len() >= 2, "should split into multiple chunks");
        assert_verbatim(&content, &chunks);
    }

    #[test]
    fn test_markdown_heading_extraction() {
        let content = "## Introduction\n\nSome text here about the topic.";
        let chunks = chunk_text(content, &cfg());
        assert_eq!(chunks[0].heading.as_deref(), Some("Introduction"));
    }

    #[test]
    fn test_chunks_have_sequential_indices() {
        let long = "Word. ".repeat(200);
        let chunks = chunk_text(&long, &cfg());
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
        assert_verbatim(&long, &chunks);
    }

    #[test]
    fn test_chunk_type_is_text() {
        let chunks = chunk_text("Some content here.", &cfg());
        assert_eq!(chunks[0].chunk_type, "text");
    }

    #[test]
    fn decimals_paths_and_versions_are_never_cut() {
        let line = "Pi is 3.14159 and the file is src/foo.rs at v1.2.3 today ok ";
        let content = line.repeat(30);
        let chunks = chunk_text(&content, &cfg());
        assert!(chunks.len() > 1);
        assert_verbatim(&content, &chunks);
        for c in &chunks {
            for word in c.text.split_whitespace() {
                assert!(
                    !matches!(word, "3." | "14159" | "src/foo." | "rs" | "v1." | "2.3"),
                    "cut inside a token: {word:?} in chunk {}",
                    c.index
                );
            }
        }
    }

    #[test]
    fn crlf_blank_lines_are_paragraph_breaks() {
        let content = format!(
            "{}\r\n\r\n{}",
            "First paragraph words go here. ".repeat(6),
            "Second paragraph words go here. ".repeat(6)
        );
        let chunks = chunk_text(&content, &cfg());
        assert_verbatim(&content, &chunks);
        // The first paragraph's own text, trailing space trimmed.
        let brk = content[..content.find("\r\n\r\n").unwrap()]
            .trim_end()
            .len();
        assert!(
            chunks.iter().any(|c| c.end == brk),
            "no chunk ends at the break"
        );
    }

    #[test]
    fn lines_split_before_sentences() {
        // Log-like content: no blank lines, one record per line.
        let content: String = (0..80)
            .map(|i| format!("2026-10-01T12:00:{i:02}Z INFO request_id=r{i} status=200\n"))
            .collect();
        let chunks = chunk_text(&content, &cfg());
        assert_verbatim(&content, &chunks);
        for c in &chunks {
            // Overlap may start mid-line; the chunk's own text ends on a line.
            assert!(c.text.ends_with("status=200"), "{:?}", c.text);
        }
    }

    #[test]
    fn oversized_run_without_whitespace_is_split_and_bounded() {
        let content = "x".repeat(4000);
        let chunks = chunk_text(&content, &cfg());
        assert!(chunks.len() > 1);
        assert_verbatim(&content, &chunks);
        for c in &chunks {
            assert!(c.token_count <= cfg().max_chunk_tokens + cfg().overlap_tokens);
        }
    }

    #[test]
    fn overlap_reaches_back_into_previous_chunk_only() {
        let content = "alpha beta gamma delta epsilon zeta eta theta. ".repeat(40);
        let chunks = chunk_text(&content, &cfg());
        assert_verbatim(&content, &chunks);
        for w in chunks.windows(2) {
            assert!(w[1].start < w[0].end, "overlap expected");
            assert!(w[1].start > w[0].start);
            assert!(w[1].end > w[0].end);
        }
    }

    #[test]
    fn multibyte_text_keeps_char_boundaries() {
        let content = "naïve café — 東京 🚀 ".repeat(200);
        let chunks = chunk_text(&content, &cfg());
        assert_verbatim(&content, &chunks);
        assert_eq!(chunks.last().unwrap().end, content.trim_end().len());
    }
}
