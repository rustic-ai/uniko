//! A JSON parser that keeps every value's byte span.
//!
//! `serde_json` decodes values but forgets where they were. Chunking a
//! tool's JSON result verbatim needs the opposite: each object, array and
//! scalar's `[start, end)` in the source, so a chunk can be an exact slice
//! and its address a JSON pointer. Values are not decoded beyond object
//! keys; a scalar's text is its slice.

/// One parsed value and its span in the source.
#[derive(Debug, Clone)]
pub struct JsonNode {
    /// Byte offset of the value's first character.
    pub start: usize,
    /// Byte offset just past its last character.
    pub end: usize,
    /// Its shape.
    pub kind: JsonKind,
}

/// The shape of a [`JsonNode`].
#[derive(Debug, Clone)]
pub enum JsonKind {
    /// An object: its members in source order, keys decoded.
    Object(Vec<(String, JsonNode)>),
    /// An array: its elements in order.
    Array(Vec<JsonNode>),
    /// A string, number, boolean or null; its text is the node's slice.
    Scalar,
}

/// Deepest nesting accepted; deeper input is treated as not-JSON rather
/// than risking the stack.
const MAX_DEPTH: usize = 256;

/// Parse `text` as exactly one JSON value, surrounded by optional
/// whitespace. `None` when it is not valid JSON.
#[must_use]
pub fn parse(text: &str) -> Option<JsonNode> {
    let mut p = Parser {
        s: text.as_bytes(),
        src: text,
        i: 0,
    };
    p.ws();
    let node = p.value(0)?;
    p.ws();
    (p.i == p.s.len()).then_some(node)
}

/// Decoded text of a JSON string scalar `raw` (with its quotes), or the raw
/// text for any other scalar.
#[must_use]
pub fn scalar_text(raw: &str) -> String {
    if raw.len() >= 2 && raw.starts_with('"') && raw.ends_with('"') {
        decode_string(&raw[1..raw.len() - 1])
    } else {
        raw.to_string()
    }
}

/// Escape one JSON-pointer reference token (RFC 6901).
#[must_use]
pub fn pointer_token(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

struct Parser<'a> {
    s: &'a [u8],
    src: &'a str,
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn value(&mut self, depth: usize) -> Option<JsonNode> {
        if depth > MAX_DEPTH {
            return None;
        }
        let start = self.i;
        let kind = match *self.s.get(self.i)? {
            b'{' => self.object(depth)?,
            b'[' => self.array(depth)?,
            b'"' => {
                self.string()?;
                JsonKind::Scalar
            }
            b't' => self.literal(b"true")?,
            b'f' => self.literal(b"false")?,
            b'n' => self.literal(b"null")?,
            b'-' | b'0'..=b'9' => self.number()?,
            _ => return None,
        };
        Some(JsonNode {
            start,
            end: self.i,
            kind,
        })
    }

    fn object(&mut self, depth: usize) -> Option<JsonKind> {
        self.i += 1; // '{'
        let mut members = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Some(JsonKind::Object(members));
        }
        loop {
            self.ws();
            let ks = self.i;
            self.string()?;
            let key = decode_string(&self.src[ks + 1..self.i - 1]);
            self.ws();
            if self.s.get(self.i) != Some(&b':') {
                return None;
            }
            self.i += 1;
            self.ws();
            let v = self.value(depth + 1)?;
            members.push((key, v));
            self.ws();
            match self.s.get(self.i)? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Some(JsonKind::Object(members));
                }
                _ => return None,
            }
        }
    }

    fn array(&mut self, depth: usize) -> Option<JsonKind> {
        self.i += 1; // '['
        let mut items = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Some(JsonKind::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value(depth + 1)?);
            self.ws();
            match self.s.get(self.i)? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Some(JsonKind::Array(items));
                }
                _ => return None,
            }
        }
    }

    /// Consume a string, leaving `i` just past the closing quote.
    fn string(&mut self) -> Option<()> {
        if self.s.get(self.i) != Some(&b'"') {
            return None;
        }
        self.i += 1;
        while self.i < self.s.len() {
            match self.s[self.i] {
                b'"' => {
                    self.i += 1;
                    return Some(());
                }
                b'\\' => self.i += 2,
                c if c < 0x20 => return None,
                _ => self.i += 1,
            }
        }
        None
    }

    fn literal(&mut self, word: &[u8]) -> Option<JsonKind> {
        if self.s.get(self.i..self.i + word.len()) == Some(word) {
            self.i += word.len();
            Some(JsonKind::Scalar)
        } else {
            None
        }
    }

    fn number(&mut self) -> Option<JsonKind> {
        let start = self.i;
        if self.s.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        let digits = |p: &mut Self| {
            let d = p.i;
            while p.i < p.s.len() && p.s[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i > d
        };
        if !digits(self) {
            return None;
        }
        if self.s.get(self.i) == Some(&b'.') {
            self.i += 1;
            if !digits(self) {
                return None;
            }
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if !digits(self) {
                return None;
            }
        }
        (self.i > start).then_some(JsonKind::Scalar)
    }
}

/// Decode the body of a JSON string (no quotes). Malformed escapes are kept
/// as written.
fn decode_string(body: &str) -> String {
    if !body.contains('\\') {
        return body.to_string();
    }
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('b') => out.push('\u{8}'),
            Some('f') => out.push('\u{c}'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    Some(ch) => out.push(ch),
                    None => {
                        out.push_str("\\u");
                        out.push_str(&hex);
                    }
                }
            }
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_cover_exact_values() {
        let src = r#" { "meta": {"currency": "EUR"}, "items": [1, 2.5e3, "a\"b", null] } "#;
        let root = parse(src).expect("valid");
        assert_eq!(&src[root.start..root.end], src.trim());
        let JsonKind::Object(members) = &root.kind else {
            panic!("object expected")
        };
        assert_eq!(members[0].0, "meta");
        assert_eq!(
            &src[members[0].1.start..members[0].1.end],
            r#"{"currency": "EUR"}"#
        );
        let JsonKind::Array(items) = &members[1].1.kind else {
            panic!("array expected")
        };
        let raw: Vec<&str> = items.iter().map(|n| &src[n.start..n.end]).collect();
        assert_eq!(raw, ["1", "2.5e3", r#""a\"b""#, "null"]);
        assert_eq!(scalar_text(raw[2]), "a\"b");
    }

    #[test]
    fn rejects_invalid_and_trailing_input() {
        for bad in ["", "{", "[1,]", "{\"a\" 1}", "01x", "[1] 2", "{\"a\":tru}"] {
            assert!(parse(bad).is_none(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn decodes_escaped_keys_and_escapes_pointers() {
        let root = parse(r#"{"a/b~c": 1, "é": 2}"#).unwrap();
        let JsonKind::Object(m) = root.kind else {
            panic!()
        };
        assert_eq!(m[0].0, "a/b~c");
        assert_eq!(pointer_token(&m[0].0), "a~1b~0c");
        assert_eq!(m[1].0, "é");
    }

    #[test]
    fn deep_nesting_is_refused_not_overflowed() {
        let deep = format!("{}{}", "[".repeat(10_000), "]".repeat(10_000));
        assert!(parse(&deep).is_none());
    }
}
