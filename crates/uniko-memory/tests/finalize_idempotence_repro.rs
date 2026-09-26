//! Isolated repro for a spurious session-chunk rebuild.
//!
//! `finalize()` on an unchanged session must report `rebuilt == false`: the
//! transcript is unchanged, so the chunk surface is current and nothing
//! should be re-embedded. Under the full workspace suite that assertion
//! trips in roughly a third of runs, which makes it useless to iterate on —
//! each attempt costs a four-minute suite and a coin flip.
//!
//! This reproduces it on its own, with the load generated locally, so the
//! failure can be bisected in seconds instead of minutes.
//!
//! Run:
//! ```sh
//! cargo nextest run -p uniko-memory --test finalize_idempotence_repro \
//!   --run-ignored all --no-capture
//! ```
//!
//! `REPRO_ROUNDS` sets the sessions per task (default 12) and
//! `REPRO_TASKS` the concurrent tasks (default 8) — the race needs
//! concurrency, so a single-threaded run proves nothing.

use std::sync::Arc;

use uniko_memory::{Turn, Uniko};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// One session: observe a turn, finalize twice, report whether the second
/// finalize spuriously rebuilt. Returns `Err` with the captured state.
async fn one_round(memory: &Uniko, session_id: String) -> Result<(), String> {
    let agent = memory.agent("assistant");
    let mut session = agent.session(session_id.clone());

    if session
        .observe(
            Turn::new("alice", "Sourdough needs a stiff starter.").id(format!("{session_id}-m")),
        )
        .await
        .is_err()
    {
        // Model unavailable in this environment — treat as a skip.
        return Ok(());
    }

    let first = session
        .finalize()
        .await
        .map_err(|e| format!("first finalize: {e}"))?;
    let second = session
        .finalize()
        .await
        .map_err(|e| format!("second finalize: {e}"))?;

    if second.rebuilt {
        // `FinalizeReport::rebuilt` is the OR of the transcript surface and
        // the observation surface. Which one moved is the whole question, so
        // report both rather than the flag alone.
        let transcript_same = first.transcript_chunks == second.transcript_chunks;
        let observations_same = first.observation_chunks == second.observation_chunks;
        return Err(format!(
            "spurious rebuild on {session_id}\n  \
             transcript:   {:?} -> {:?}  (same: {transcript_same})\n  \
             observations: {:?} -> {:?}  (same: {observations_same})",
            first.transcript_chunks,
            second.transcript_chunks,
            first.observation_chunks,
            second.observation_chunks,
        ));
    }

    Ok(())
}

#[ignore = "repro: run explicitly, needs concurrency to trip"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn finalize_is_idempotent_under_load() {
    let rounds = env_usize("REPRO_ROUNDS", 12);
    let tasks = env_usize("REPRO_TASKS", 8);

    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("SKIP: in-memory instance unavailable (no model?)");
        return;
    };
    let memory = Arc::new(memory);

    let mut handles = Vec::with_capacity(tasks);
    for t in 0..tasks {
        let memory = Arc::clone(&memory);
        handles.push(tokio::spawn(async move {
            let mut failures = Vec::new();
            for r in 0..rounds {
                if let Err(e) = one_round(&memory, format!("repro-{t}-{r}")).await {
                    failures.push(e);
                }
            }
            failures
        }));
    }

    let mut all = Vec::new();
    for h in handles {
        all.extend(h.await.expect("task panicked"));
    }

    let total = rounds * tasks;
    eprintln!(
        "finalize repro: {} spurious rebuilds out of {total} sessions",
        all.len()
    );
    for f in all.iter().take(5) {
        eprintln!("---\n{f}");
    }
    assert!(
        all.is_empty(),
        "{} of {total} sessions rebuilt an unchanged chunk surface",
        all.len()
    );
}
