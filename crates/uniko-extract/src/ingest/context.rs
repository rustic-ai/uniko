//! Session-level ingestion context.
//!
//! [`SessionContext`] persists across messages within a session,
//! tracking participants, message chain, and the pronoun resolution
//! window ([`SentenceContext`]).
//!
//! Created once per session by the caller (bench or production engine),
//! passed to `ingest_message()` and pipeline steps via metadata.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use uniko_store::NodeId;

/// Session-level state that persists across messages.
///
/// Owns the sentence context window for pronoun resolution,
/// participant cache, and message chain tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContext {
    /// External session identifier.
    pub session_id: String,
    /// Internal node ID of the Session node.
    pub session_nid: NodeId,
    /// Participant name → node ID cache (avoids re-querying).
    pub participants: HashMap<String, NodeId>,
    /// Current message speaker.
    pub current_speaker: String,
    /// Other participants in the session.
    pub other_speakers: Vec<String>,
    /// Previous message node ID for NEXT edge linking.
    pub prev_message_nid: Option<NodeId>,
    /// Timestamp of the previous message, for computing `NEXT.gap_ms`.
    pub prev_message_ts: Option<DateTime<Utc>>,
    /// Pronoun resolution context window.
    pub sentence_ctx: SentenceContext,
}

impl SessionContext {
    /// Create a new session context.
    pub fn new(session_id: String, session_nid: NodeId) -> Self {
        Self {
            session_id,
            session_nid,
            participants: HashMap::new(),
            current_speaker: String::new(),
            other_speakers: Vec::new(),
            prev_message_nid: None,
            prev_message_ts: None,
            sentence_ctx: SentenceContext::default(),
        }
    }

    /// Set the current speaker and update other_speakers list.
    pub fn set_current_speaker(&mut self, speaker: &str) {
        self.other_speakers = advance_speaker(&mut self.sentence_ctx, &self.participants, speaker);
        self.current_speaker = speaker.to_string();
    }

    /// Register a participant (caches name → nid).
    pub fn register_participant(&mut self, name: &str, nid: NodeId) {
        self.participants.insert(name.to_string(), nid);
    }

    /// Get a participant node ID by name.
    pub fn participant_nid(&self, name: &str) -> Option<NodeId> {
        self.participants.get(name).copied()
    }
}

/// Point a sentence-context window at `speaker`, rebuilding
/// `other_speakers` from `participants`. Returns the rebuilt list.
///
/// A free function rather than only a [`SessionContext`] method because a
/// multi-turn atomic unit must advance the speaker on a **local**
/// `SentenceContext` inside the transaction body: the real `SessionContext`
/// must not be mutated until the unit commits, or a retriable conflict on a
/// later turn would re-seed pronoun resolution from an already-advanced
/// window. [`SessionContext::set_current_speaker`] is implemented on top of
/// this, so there is one rule for both paths.
pub fn advance_speaker(
    sentence_ctx: &mut SentenceContext,
    participants: &HashMap<String, NodeId>,
    speaker: &str,
) -> Vec<String> {
    sentence_ctx.speaker = speaker.to_string();
    let others: Vec<String> = participants
        .keys()
        .filter(|name| name.as_str() != speaker)
        .cloned()
        .collect();
    sentence_ctx.other_speakers = others.clone();
    others
}

/// Pronoun resolution context window.
///
/// Tracks the most recent noun phrases (subject and object) across
/// sentences within a session. Used to resolve pronouns like "it",
/// "this", "that", "you" to their antecedents.
///
/// Updated after each sentence by [`update_sentence_context`](crate::nlp::decode::update_sentence_context).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SentenceContext {
    /// Current speaker name (for first-person resolution).
    pub speaker: String,
    /// Other participants (for second-person resolution).
    pub other_speakers: Vec<String>,
    /// Last NOUN/PROPN subject seen (for "it"/"this"/"that" resolution).
    pub last_noun_subject: Option<String>,
    /// Last NOUN/PROPN object seen (for "it"/"this"/"that" resolution).
    pub last_noun_object: Option<String>,
}

impl SentenceContext {
    /// Create a context with speaker info.
    pub fn new(speaker: &str, other_speakers: Vec<String>) -> Self {
        Self {
            speaker: speaker.to_string(),
            other_speakers,
            last_noun_subject: None,
            last_noun_object: None,
        }
    }
}
