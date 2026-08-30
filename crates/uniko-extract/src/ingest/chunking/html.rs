//! HTML chunker — extracts readable content via `dom_smoothie`
//! (Readability.js port), then walks the cleaned subtree to emit
//! heading + body sections.
//!
//! Pipeline:
//! 1. `dom_smoothie::Readability::parse` runs the same scoring used by
//!    Firefox Reader View to strip nav / footer / sidebar / ad boilerplate
//!    and isolate the article body. `article.title` becomes the lead
//!    heading; `article.content` is cleaned HTML.
//! 2. If Readability fails (too short, unrecognized structure, etc.) we
//!    fall back to walking the raw HTML so we never silently drop input.
//! 3. The walker is `&str`-slice-based (uses `str::find`, which always
//!    returns char-boundary byte offsets because `<` and `>` are
//!    ASCII), so UTF-8 content survives intact.
//! 4. `<script>`, `<style>`, `<noscript>`, and `<!-- ... -->` blocks are
//!    dropped wholesale before tag walking.
//! 5. Each `<h1>..<h6>` becomes a `chunk_type="heading"` section; the
//!    surrounding body becomes `chunk_type="text"` with `Chunk.heading`
//!    set to the nearest enclosing heading.
//
use dom_smoothie::Readability;

use super::text::TextChunker;
use super::{ChunkConfig, ChunkData, Chunker};

/// DOM-aware chunker for `text/html`.
pub struct HtmlChunker;

impl Chunker for HtmlChunker {
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
        let (preprocessed, lead_heading) = preprocess_with_readability(content);
        let sections = extract_sections(&preprocessed, lead_heading);
        if sections.is_empty() {
            // Degenerate input — fall back to TextChunker on the raw
            // bytes so we never silently drop content.
            return TextChunker.chunk(content, config);
        }

        let mut out: Vec<ChunkData> = Vec::new();
        let mut next_index: usize = 0;
        for section in sections {
            let pieces = TextChunker.chunk(&section.text, config);
            for mut piece in pieces {
                piece.index = next_index;
                next_index += 1;
                piece.chunk_type = section.chunk_type.clone();
                piece.heading = section.heading.clone();
                out.push(piece);
            }
        }
        if out.is_empty() {
            // Sections were all empty after stripping — fall back so
            // downstream tests don't see an empty chunk vector when
            // the input had any text at all.
            return TextChunker.chunk(content, config);
        }
        out
    }
}

#[derive(Debug)]
struct Section {
    text: String,
    chunk_type: String,
    heading: Option<String>,
}

/// Run `dom_smoothie::Readability` to isolate the article body. Returns
/// `(preprocessed_html, lead_heading)`. On any error (short input,
/// no scorable elements, parse failure) falls back to the raw HTML and
/// no lead heading, so the walker never sees less content than it would
/// have without Readability in the pipeline.
fn preprocess_with_readability(html: &str) -> (String, Option<String>) {
    let readability = match Readability::new(html, None, None) {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(error = %e, "dom_smoothie::Readability::new failed; using raw HTML");
            return (html.to_string(), None);
        }
    };
    let mut readability = readability;
    match readability.parse() {
        Ok(article) => {
            let title = if article.title.trim().is_empty() {
                None
            } else {
                Some(article.title.trim().to_string())
            };
            let content_str: String = article.content.to_string();
            // If Readability produced an empty body, prefer raw input so
            // we don't lose all content for inputs that don't look like
            // articles (snippets, fragments).
            if content_str.trim().is_empty() {
                (html.to_string(), title)
            } else {
                (content_str, title)
            }
        }
        Err(e) => {
            tracing::debug!(error = %e, "dom_smoothie parse failed; using raw HTML");
            (html.to_string(), None)
        }
    }
}

/// Walk the HTML in source order, producing a stream of `Section`s:
/// one `chunk_type="heading"` per `<h1>..<h6>`, then one
/// `chunk_type="text"` for the body that follows it.
fn extract_sections(html: &str, lead_heading: Option<String>) -> Vec<Section> {
    let cleaned = strip_uninteresting(html);
    let mut sections: Vec<Section> = Vec::new();
    let mut buf = String::new();
    let mut current_heading: Option<String> = lead_heading.clone();

    // Emit the lead heading as a section so a recall search anchored on
    // the page title still hits.
    if let Some(ref h) = lead_heading {
        sections.push(Section {
            text: h.clone(),
            chunk_type: "heading".into(),
            heading: Some(h.clone()),
        });
    }

    let mut i: usize = 0;
    while i < cleaned.len() {
        // Find the next `<` (start of a tag or comment). All bytes
        // before it are literal text — append in one go.
        let rest = &cleaned[i..];
        let Some(lt) = rest.find('<') else {
            buf.push_str(rest);
            break;
        };
        if lt > 0 {
            buf.push_str(&rest[..lt]);
        }
        i += lt;

        // Find the matching `>` for the tag we're about to parse.
        let Some(rel) = cleaned[i + 1..].find('>') else {
            // Unmatched `<` — treat literally and move past it.
            buf.push('<');
            i += 1;
            continue;
        };
        let tag_end = i + 1 + rel; // index of '>'
        let tag_raw = &cleaned[i + 1..tag_end];
        let tag_name = tag_raw
            .trim_start_matches('/')
            .trim_end_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();

        // Headings: flush body, then capture inner text as its own
        // section + update the active heading for subsequent body text.
        if matches!(tag_name.as_str(), "h1" | "h2" | "h3" | "h4" | "h5" | "h6")
            && !tag_raw.starts_with('/')
        {
            flush_text(&mut buf, &mut sections, &current_heading);
            let close = format!("</{tag_name}");
            let after = tag_end + 1;
            let close_start = cleaned[after..]
                .find(&close)
                .map(|p| p + after)
                .unwrap_or(cleaned.len());
            let inner = strip_inline_tags(&cleaned[after..close_start]);
            let heading_text = decode_entities(&inner).trim().to_string();
            if !heading_text.is_empty() {
                sections.push(Section {
                    text: heading_text.clone(),
                    chunk_type: "heading".into(),
                    heading: Some(heading_text.clone()),
                });
                current_heading = Some(heading_text);
            }
            // Advance past `</hN>`.
            i = cleaned[close_start..]
                .find('>')
                .map(|p| close_start + p + 1)
                .unwrap_or(cleaned.len());
            continue;
        }

        // Block-level boundary: insert a paragraph break so the
        // TextChunker can split on `\n\n`.
        if matches!(
            tag_name.as_str(),
            "p" | "div" | "section" | "article" | "li" | "br" | "tr" | "td" | "th" | "blockquote"
        ) && !buf.ends_with("\n\n")
        {
            buf.push_str("\n\n");
        }

        // Skip past `>`.
        i = tag_end + 1;
    }
    flush_text(&mut buf, &mut sections, &current_heading);
    sections
}

fn flush_text(buf: &mut String, out: &mut Vec<Section>, heading: &Option<String>) {
    let s = decode_entities(buf.as_str());
    let collapsed = collapse_whitespace(&s);
    if !collapsed.is_empty() {
        out.push(Section {
            text: collapsed,
            chunk_type: "text".into(),
            heading: heading.clone(),
        });
    }
    buf.clear();
}

/// Drop the contents of `<script>` / `<style>` / `<noscript>` blocks
/// and HTML comments before any further tag walking. Operates on
/// `&str` slices via `str::find`, so UTF-8 bytes between markers pass
/// through untouched.
fn strip_uninteresting(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let lower = html.to_ascii_lowercase();
    let mut i = 0;
    while i < html.len() {
        // HTML comment — must be checked before tag matching since
        // `<!--` shares a `<` prefix with real tags.
        if html[i..].starts_with("<!--") {
            match html[i + 4..].find("-->") {
                Some(end_rel) => {
                    i = i + 4 + end_rel + 3;
                    continue;
                }
                None => break,
            }
        }

        // Opaque tag bodies: script / style / noscript.
        let rest_lower = &lower[i..];
        if let Some(open_tag) = ["<script", "<style", "<noscript"]
            .iter()
            .find(|t| rest_lower.starts_with(*t))
        {
            let close = format!("</{}", &open_tag[1..]);
            let end_rel = lower[i..].find(&close).unwrap_or(lower.len() - i);
            let close_at = i + end_rel;
            i = lower[close_at..]
                .find('>')
                .map(|p| close_at + p + 1)
                .unwrap_or(lower.len());
            continue;
        }

        // Find the next interesting marker so we can copy a contiguous
        // `&str` slice in one shot — this is the UTF-8-safe path.
        let candidates = [
            html[i..].find("<!--"),
            rest_lower.find("<script"),
            rest_lower.find("<style"),
            rest_lower.find("<noscript"),
        ];
        let next_skip = candidates.iter().filter_map(|x| *x).min();
        match next_skip {
            None => {
                out.push_str(&html[i..]);
                break;
            }
            Some(skip) => {
                out.push_str(&html[i..i + skip]);
                i += skip;
            }
        }
    }
    out
}

/// Remove HTML tags from `s` while preserving textual content.
/// Iterates over `chars()` to stay UTF-8-safe.
fn strip_inline_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

/// Decode a small but practically-relevant set of HTML entities. Named
/// entities cover the most common cases produced by docs / blogs;
/// numeric entities (`&#NNNN;` / `&#xHHHH;`) round-trip any character.
fn decode_entities(s: &str) -> String {
    // Fast path for ASCII-only inputs without any '&'.
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < s.len() {
        if bytes[i] != b'&' {
            // Copy a UTF-8-safe slice up to the next '&' or end.
            let rest = &s[i..];
            let next = rest.find('&').unwrap_or(rest.len());
            out.push_str(&rest[..next]);
            i += next;
            continue;
        }
        // We're at '&'. Find the terminating ';' within a sane window.
        let rest = &s[i..];
        let Some(semi) = rest
            .as_bytes()
            .iter()
            .take(10)
            .position(|&byte| byte == b';')
        else {
            out.push('&');
            i += 1;
            continue;
        };
        let entity = &rest[1..semi];
        let decoded = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" | "#39" => Some('\''),
            "nbsp" => Some(' '),
            "mdash" => Some('—'),
            "ndash" => Some('–'),
            "hellip" => Some('…'),
            "copy" => Some('©'),
            "reg" => Some('®'),
            "trade" => Some('™'),
            e if e.starts_with("#x") || e.starts_with("#X") => u32::from_str_radix(&e[2..], 16)
                .ok()
                .and_then(char::from_u32),
            e if e.starts_with('#') => e[1..].parse::<u32>().ok().and_then(char::from_u32),
            _ => None,
        };
        match decoded {
            Some(ch) => {
                out.push(ch);
                i += semi + 1;
            }
            None => {
                // Unknown entity — pass through literally.
                out.push('&');
                i += 1;
            }
        }
    }
    out
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blank_line = false;
    for line in s.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !blank_line && !out.is_empty() {
                out.push_str("\n\n");
                blank_line = true;
            }
        } else {
            out.push_str(trimmed);
            out.push('\n');
            blank_line = false;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strips_script_and_style() {
        let html = "<html><head><style>.x{color:red}</style><script>alert(1)</script></head>\
                    <body><h1>Title</h1><p>Hello world this is some prose worth keeping</p></body></html>";
        let chunks = HtmlChunker.chunk(html, &ChunkConfig::default());
        let combined: String = chunks
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !combined.contains("alert"),
            "script body leaked: {combined}"
        );
        assert!(
            !combined.contains("color:red"),
            "style body leaked: {combined}"
        );
        assert!(combined.contains("Hello world"));
    }

    #[test]
    fn test_heading_propagates_to_body() {
        let html = "<h2>Greeting</h2><p>Hi there friend, this body must carry the heading context for retrieval.</p>";
        let chunks = HtmlChunker.chunk(html, &ChunkConfig::default());
        let body = chunks
            .iter()
            .find(|c| c.chunk_type == "text")
            .expect("body text chunk");
        assert_eq!(body.heading.as_deref(), Some("Greeting"));
    }

    #[test]
    fn test_heading_emits_heading_chunk_type() {
        let html = "<h1>Title</h1><p>Body with sufficient prose to survive any minimum-length filter that Readability or the chunker might apply.</p>";
        let chunks = HtmlChunker.chunk(html, &ChunkConfig::default());
        assert!(chunks.iter().any(|c| c.chunk_type == "heading"));
    }

    #[test]
    fn test_degenerate_html_falls_back() {
        // No tags at all — still produces a chunk via TextChunker.
        let chunks = HtmlChunker.chunk("just some plain text", &ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("plain text"));
    }

    #[test]
    fn test_utf8_content_preserved() {
        // The pre-fix walker did `bytes[i] as char`, corrupting any
        // multi-byte UTF-8 sequence. This test pins the fix in place.
        let html = "<p>café — héllo 你好 🌍 — Olá!</p>";
        let chunks = HtmlChunker.chunk(html, &ChunkConfig::default());
        let combined: String = chunks
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        for needle in ["café", "héllo", "你好", "🌍", "Olá"] {
            assert!(
                combined.contains(needle),
                "expected {needle:?} preserved, got {combined:?}"
            );
        }
    }

    #[test]
    fn test_html_comments_stripped() {
        let html = "<p>before <!-- secret > leak --> after</p>";
        let chunks = HtmlChunker.chunk(html, &ChunkConfig::default());
        let combined: String = chunks
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(combined.contains("before"));
        assert!(combined.contains("after"));
        assert!(
            !combined.contains("secret"),
            "comment body leaked: {combined:?}"
        );
    }

    #[test]
    fn test_numeric_entities_decode() {
        // &#8217; is the right single quote (’), &#x2014; is em-dash (—).
        let html = "<p>It&#8217;s a test &#x2014; with entities.</p>";
        let chunks = HtmlChunker.chunk(html, &ChunkConfig::default());
        let combined: String = chunks
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            combined.contains("It’s"),
            "numeric entity not decoded: {combined:?}"
        );
        assert!(
            combined.contains("—"),
            "hex entity not decoded: {combined:?}"
        );
    }

    #[test]
    fn test_entity_scan_without_terminator_is_utf8_safe() {
        for input in ["&12345678–", "&12345678¹", "&12345678×"] {
            assert_eq!(decode_entities(input), input);
        }
    }

    #[test]
    fn test_readability_strips_nav_and_footer() {
        // Real-world-shaped article with nav + footer boilerplate that
        // Readability scores as low-density.
        let html = r#"<html><body>
            <nav><a href="/x">Home</a><a href="/y">About</a></nav>
            <article>
                <h1>The Rise of Vector Databases</h1>
                <p>Vector databases have emerged as a central piece of modern AI infrastructure. They store high-dimensional embeddings and serve approximate nearest-neighbor queries at scale. This has unlocked use cases ranging from semantic search to multimodal retrieval.</p>
                <p>The key trick is the index structure. HNSW remains the default for in-memory deployments; IVF-PQ for very large collections that must stay on disk. Both make different trade-offs between recall, latency, and memory footprint.</p>
                <p>Quantization is the other major axis. Scalar quantization is cheap and lossless enough for most workloads; product quantization delivers higher compression at the cost of recall.</p>
            </article>
            <footer>Copyright 2026 Example Corp. All rights reserved. <a href="/privacy">Privacy</a></footer>
            </body></html>"#;
        let chunks = HtmlChunker.chunk(html, &ChunkConfig::default());
        let combined: String = chunks
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        // Article body must survive.
        assert!(
            combined.contains("Vector databases"),
            "article missing: {combined:?}"
        );
        // Boilerplate should be gone (Readability scores nav/footer below threshold).
        assert!(
            !combined.contains("Copyright 2026"),
            "footer boilerplate leaked: {combined:?}"
        );
    }
}
