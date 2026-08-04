//! Repro for **rustic-ai/uni-db#157**: a *registered* Locy rule's parameters
//! are demanded by **every** Locy program, including programs that reference
//! no rule at all.
//!
//! Symptom:
//!
//! ```text
//! LocyRuntimeError: Execution error: Sub-plan error: Unresolved parameter: $needed
//! ```
//!
//! Register one parameterized rule, then run an unrelated `ASSUME`/`MATCH`
//! program that never names it. The run fails because the registered rule is
//! resolved as a sub-plan and its `$needed` parameter is unbound. Binding the
//! parameter on the unrelated call makes it succeed, which is the tell: the
//! rule is being compiled into a program that has nothing to do with it.
//!
//! Real-world impact on uniko: `Uniko` registers four parameterized stdlib
//! rules (`$agent_id`, `$decay_rate`, `$decay_threshold`) at construction, so
//! every `Agent::assume` / `Agent::abduce` / `Agent::run_rule` call had to
//! carry the union of all of them. Any consumer registering a rule with a
//! parameter breaks every unrelated Locy query for the life of the database.
//!
//! Expected: a program that references no rules should not require any
//! registered rule's parameters. Rules should be resolved only when the
//! program actually reaches them.
//!
//! These tests assert the *correct, post-fix* behavior, so `unrelated_program_
//! does_not_need_a_registered_rules_params` FAILS against current uni-db
//! (verified on 3.0.1 and 3.2.0). `#[ignore]`d to keep CI green; run with
//! `cargo nextest run -p uniko-store --test bug_repros --run-ignored all \
//! registered_rule_param_leak`.

use uni_db::{DataType, Uni};

async fn setup() -> Uni {
    let db = Uni::in_memory().build().await.unwrap();
    db.schema()
        .label("Item")
        .property("tag", DataType::String)
        .done()
        .apply()
        .await
        .unwrap();
    let s = db.session();
    let tx = s.tx().await.unwrap();
    tx.execute("CREATE (:Item {tag: 'a'})").await.unwrap();
    tx.commit().await.unwrap();

    // One registered rule that takes a parameter. Never referenced below.
    db.rules()
        .register("CREATE RULE needs_param AS MATCH (i:Item {tag: $needed}) YIELD KEY i")
        .await
        .unwrap();
    db
}

/// A Locy program that references no rule must run without supplying an
/// unrelated registered rule's parameters.
#[tokio::test]
#[ignore = "fails against uni-db 3.2.0 — asserts post-fix behavior"]
async fn unrelated_program_does_not_need_a_registered_rules_params() {
    let db = setup().await;
    let s = db.session();
    let result = s
        .locy_with(
            "ASSUME { CREATE (:Item {tag: 'hypothetical'}) } THEN { MATCH (i:Item) RETURN i }",
        )
        .run()
        .await;
    eprintln!("PROBE unrelated program -> {result:?}");
    assert!(
        result.is_ok(),
        "a program referencing no rule must not require $needed: {:?}",
        result.err()
    );
    db.shutdown().await.unwrap();
}

/// Control: binding the unrelated rule's parameter makes the same program
/// succeed. This is what pins the cause to rule resolution rather than to the
/// program itself.
#[tokio::test]
async fn binding_the_unrelated_rules_param_makes_it_pass() {
    let db = setup().await;
    let s = db.session();
    let result = s
        .locy_with(
            "ASSUME { CREATE (:Item {tag: 'hypothetical'}) } THEN { MATCH (i:Item) RETURN i }",
        )
        .param("needed", "a")
        .run()
        .await;
    eprintln!("PROBE with $needed bound -> {:?}", result.is_ok());
    assert!(
        result.is_ok(),
        "control must pass with the param bound: {:?}",
        result.err()
    );
    db.shutdown().await.unwrap();
}
