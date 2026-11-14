//! Normalised search text: the form of a chunk that the lexical index
//! sees, and that a query is put in before matching it.
//!
//! The default analyzer splits on punctuation, so `1,234` is indexed as
//! `1` and `234`, `3.14` as `3` and `14`, and `ORD-48213` as `ORD` and
//! `48213` — an exact figure or identifier in a query then matches every
//! chunk that shares any of the pieces. Normalised text keeps them whole
//! and adds their parts:
//!
//! - digit groups lose their thousands separators (`1,234,567` →
//!   `1234567`) and decimals stay one token (`3.14`);
//! - an identifier is kept whole and also split at separators and case
//!   changes (`request_id` → `request_id request id`, `orderId` → `orderId
//!   order Id`, `src/foo.rs` → `src/foo.rs src foo rs`);
//! - JSON and markup punctuation become spaces.
//!
//! Tokens are separated by single spaces, so the index's whitespace
//! tokenizer takes them as they are.

/// Characters that may sit inside one token.
fn is_token_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':' | '@' | '#' | '+' | '%' | '$')
}

/// Characters a token may not start or end with.
fn is_edge_trim(c: char) -> bool {
    matches!(c, '.' | ':' | '-' | '/' | '+' | '#' | '@')
}

/// Remove thousands separators inside digit groups: `1,234,567.89` →
/// `1234567.89`. A comma between digits that are not a group of three is
/// left alone (`1,2` stays a list).
fn strip_digit_group_commas(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let mut last = 0;
    while i < b.len() {
        if b[i] == b','
            && i > 0
            && b[i - 1].is_ascii_digit()
            && i + 3 < b.len() + 1
            && b.get(i + 1..i + 4)
                .is_some_and(|g| g.iter().all(u8::is_ascii_digit))
            && !b.get(i + 4).is_some_and(u8::is_ascii_digit)
        {
            out.push_str(&text[last..i]);
            last = i + 1;
        }
        i += 1;
    }
    out.push_str(&text[last..]);
    out
}

/// The parts of an identifier: split at `_ - . / : @ # +` and at a
/// lower-to-upper case change. Single-letter parts are dropped.
fn parts(token: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for piece in token.split(['_', '-', '.', '/', ':', '@', '#', '+']) {
        let mut start = 0;
        let mut prev_lower = false;
        for (i, c) in piece.char_indices() {
            if c.is_uppercase() && prev_lower {
                out.push(&piece[start..i]);
                start = i;
            }
            prev_lower = c.is_lowercase() || c.is_ascii_digit();
        }
        out.push(&piece[start..]);
    }
    out.retain(|p| p.chars().count() > 1 || p.chars().all(|c| c.is_ascii_digit()));
    out
}

/// Normalise `text` into search tokens. See the module docs.
#[must_use]
pub fn normalize_for_search(text: &str) -> String {
    let text = strip_digit_group_commas(text);
    let mut out = String::with_capacity(text.len() + text.len() / 4);
    let mut push = |t: &str| {
        if !t.is_empty() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(t);
        }
    };
    for raw in text.split(|c: char| !is_token_char(c)) {
        let token = raw.trim_matches(is_edge_trim);
        if token.is_empty() {
            continue;
        }
        push(token);
        let p = parts(token);
        if p.len() > 1 {
            for part in p {
                push(part);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(s: &str) -> Vec<String> {
        normalize_for_search(s)
            .split(' ')
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn figures_stay_whole() {
        let t = tokens("revenue 1,234,567.89 and pi 3.14, list 1,2");
        assert!(t.contains(&"1234567.89".to_string()), "{t:?}");
        assert!(t.contains(&"3.14".to_string()), "{t:?}");
        assert!(
            t.contains(&"1".to_string()) && t.contains(&"2".to_string()),
            "{t:?}"
        );
    }

    #[test]
    fn identifiers_are_kept_and_split() {
        let t = tokens(r#"{"order_id": "ORD-48213", "orderId": 7} src/foo.rs v1.2.3"#);
        for want in [
            "order_id",
            "order",
            "id",
            "ORD-48213",
            "ORD",
            "48213",
            "orderId",
            "Id",
            "src/foo.rs",
            "foo",
            "rs",
            "v1.2.3",
        ] {
            assert!(t.contains(&want.to_string()), "missing {want}: {t:?}");
        }
        assert!(
            !t.iter().any(|x| x.contains('"') || x.contains('{')),
            "{t:?}"
        );
    }

    #[test]
    fn sentence_punctuation_is_trimmed() {
        let t = tokens("It failed at step 3. Then: retry.");
        assert!(
            t.contains(&"3".to_string()) && t.contains(&"retry".to_string()),
            "{t:?}"
        );
        assert!(
            !t.iter().any(|x| x.ends_with('.') || x.ends_with(':')),
            "{t:?}"
        );
    }

    #[test]
    fn query_and_document_agree() {
        let doc = normalize_for_search(r#""request_id": "req-9f3a2b1c", "error": "card_declined""#);
        let q = normalize_for_search("Which error did request req-9f3a2b1c fail with?");
        assert!(q.split(' ').any(|t| t == "req-9f3a2b1c"));
        assert!(doc.split(' ').any(|t| t == "req-9f3a2b1c"));
    }
}
