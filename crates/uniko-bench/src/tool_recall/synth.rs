//! Deterministic generator for agent sessions whose answers live in tool
//! output.
//!
//! Every value is drawn from a seeded `SplitMix64`, so a seed always yields
//! the same sessions, payloads and questions. Gold evidence is a set of
//! tokens that must all appear in ONE recalled item; context tokens are the
//! values a record inherits from outside itself (a currency in `meta`, a
//! column header, a commit named only in the tool's arguments).

use chrono::{DateTime, Duration, TimeZone, Utc};
use serde::Serialize;
use serde_json::{Value, json};

/// Small, dependency-free PRNG (SplitMix64).
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }

    pub fn hex(&mut self, len: usize) -> String {
        (0..len)
            .map(|_| char::from_digit(self.below(16) as u32, 16).unwrap())
            .collect()
    }
}

/// How a tool's result is rendered.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PayloadFormat {
    Json,
    Csv,
    Log,
    TestOutput,
}

/// One tool invocation and its verbatim result.
#[derive(Debug, Clone)]
pub struct ToolInvocation {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: Value,
    pub format: PayloadFormat,
    pub payload: String,
    pub at: DateTime<Utc>,
}

/// One step of a session, in order.
#[derive(Debug, Clone)]
pub enum Step {
    /// A plain conversational turn.
    Say {
        message_id: String,
        sender: String,
        text: String,
        at: DateTime<Utc>,
    },
    /// The assistant calls a tool; `message_id` is the assistant turn that
    /// announced it.
    Tool {
        message_id: String,
        invocation: ToolInvocation,
    },
}

#[derive(Debug, Clone)]
pub struct GenSession {
    pub session_id: String,
    pub steps: Vec<Step>,
}

/// Question categories, mirroring the failure modes we care about.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    /// A record found by its identifier.
    Lookup,
    /// A record whose meaning needs a value from outside the record.
    Inherited,
    /// Many records are operands.
    Aggregation,
    /// Records from two different calls.
    Join,
    /// The same record seen twice; only the later one is right.
    Latest,
    /// The identifier does not exist anywhere.
    Unanswerable,
}

/// Tokens that must all appear, after whitespace is removed, in one item.
#[derive(Debug, Clone, Serialize)]
pub struct Evidence {
    pub tokens: Vec<String>,
    /// Call that holds the record, and the record's address in it.
    pub call_id: String,
    pub address: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Question {
    pub id: String,
    pub category: Category,
    pub text: String,
    pub answer: String,
    /// Every group must be found for the question to be fully covered.
    pub evidence: Vec<Evidence>,
    /// Values the record inherits; must appear in the same item as the
    /// first evidence group for the item to be context-complete.
    pub context: Vec<String>,
    /// For `Latest`: evidence of the stale record, which should rank below.
    pub stale: Option<Evidence>,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub sessions: Vec<GenSession>,
    pub questions: Vec<Question>,
}

const STATUSES: &[&str] = &["pending", "paid", "shipped", "delivered", "refunded"];
const CURRENCIES: &[(&str, &str)] = &[("EUR", "cents"), ("USD", "cents"), ("JPY", "yen")];
const REGIONS: &[&str] = &["APAC", "EMEA", "LATAM", "NA"];
const TIERS: &[&str] = &["bronze", "silver", "gold", "platinum"];
const SERVICES: &[&str] = &["payments", "ledger", "checkout", "inventory"];
const ERRORS: &[&str] = &[
    "upstream_timeout",
    "card_declined",
    "idempotency_conflict",
    "schema_mismatch",
];

struct Customer {
    id: String,
    name: String,
    tier: &'static str,
    region: &'static str,
}

struct Order {
    id: String,
    status: &'static str,
    amount: u64,
    sku: String,
    updated_at: String,
}

struct Builder {
    rng: Rng,
    clock: DateTime<Utc>,
    msg: usize,
    call: usize,
    q: usize,
    questions: Vec<Question>,
    used_ids: std::collections::HashSet<String>,
}

impl Builder {
    fn tick(&mut self, secs: i64) -> DateTime<Utc> {
        self.clock += Duration::seconds(secs);
        self.clock
    }

    fn message_id(&mut self, session: &str) -> String {
        self.msg += 1;
        format!("{session}-m{}", self.msg)
    }

    fn call_id(&mut self) -> String {
        self.call += 1;
        format!("call-{:04}", self.call)
    }

    fn question(
        &mut self,
        category: Category,
        text: String,
        answer: String,
        evidence: Vec<Evidence>,
        context: Vec<String>,
        stale: Option<Evidence>,
    ) {
        self.q += 1;
        self.questions.push(Question {
            id: format!("q{:04}", self.q),
            category,
            text,
            answer,
            evidence,
            context,
            stale,
        });
    }

    fn say(&mut self, steps: &mut Vec<Step>, session: &str, sender: &str, text: String) {
        let message_id = self.message_id(session);
        let at = self.tick(7);
        steps.push(Step::Say {
            message_id,
            sender: sender.into(),
            text,
            at,
        });
    }

    fn tool(
        &mut self,
        steps: &mut Vec<Step>,
        session: &str,
        tool_name: &str,
        arguments: Value,
        format: PayloadFormat,
        payload: String,
    ) -> String {
        let message_id = self.message_id(session);
        let call_id = self.call_id();
        let at = self.tick(3);
        steps.push(Step::Tool {
            message_id,
            invocation: ToolInvocation {
                call_id: call_id.clone(),
                tool_name: tool_name.into(),
                arguments,
                format,
                payload,
                at,
            },
        });
        call_id
    }

    fn orders(&mut self, n: usize) -> Vec<Order> {
        (0..n)
            .map(|_| {
                let id = loop {
                    let id = format!("ORD-{}", 10_000 + self.rng.below(90_000));
                    if self.used_ids.insert(id.clone()) {
                        break id;
                    }
                };
                let status = *self.rng.pick(STATUSES);
                let amount = 500 + self.rng.below(250_000) as u64;
                let sku = format!("SKU-{}", 1000 + self.rng.below(60));
                let updated = self.clock - Duration::minutes(self.rng.below(50_000) as i64);
                Order {
                    id,
                    status,
                    amount,
                    sku,
                    updated_at: updated.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
                }
            })
            .collect()
    }
}

fn orders_payload(
    customer: &str,
    page: usize,
    currency: (&str, &str),
    as_of: &str,
    orders: &[Order],
) -> String {
    let items: Vec<Value> = orders
        .iter()
        .map(|o| {
            json!({
                "order_id": o.id,
                "customer_id": customer,
                "status": o.status,
                "amount": o.amount,
                "sku": o.sku,
                "updated_at": o.updated_at,
            })
        })
        .collect();
    let doc = json!({
        "meta": {
            "currency": currency.0,
            "amount_unit": currency.1,
            "as_of": as_of,
            "page": page,
            "page_size": orders.len(),
        },
        "data": { "items": items },
    });
    serde_json::to_string_pretty(&doc).unwrap()
}

/// Generate `n_sessions` sessions and their questions from `seed`.
pub fn generate(seed: u64, n_sessions: usize) -> Dataset {
    let mut b = Builder {
        rng: Rng::new(seed),
        clock: Utc.with_ymd_and_hms(2026, 10, 1, 9, 0, 0).unwrap(),
        msg: 0,
        call: 0,
        q: 0,
        questions: Vec::new(),
        used_ids: Default::default(),
    };
    let customers: Vec<Customer> = (0..n_sessions.max(4) / 2)
        .map(|i| Customer {
            id: format!("C-{}", 1000 + i * 37),
            name: format!("Org {:03}", i),
            tier: TIERS[i % TIERS.len()],
            region: REGIONS[i % REGIONS.len()],
        })
        .collect();

    let mut sessions = Vec::with_capacity(n_sessions);
    // Orders seen per customer, kept so a later session can re-fetch one
    // with a changed status (the `Latest` category).
    let mut seen: Vec<Option<(String, Order, Evidence)>> =
        (0..customers.len()).map(|_| None).collect();

    for s in 0..n_sessions {
        let session_id = format!("s{:03}", s);
        let mut steps = Vec::new();
        b.tick(3600);
        let ci = s % customers.len();
        let cust = &customers[ci];

        // --- Orders page with inherited currency/unit ---------------------
        let page = 1 + b.rng.below(3);
        let currency = *b.rng.pick(CURRENCIES);
        let as_of = b.clock.format("%Y-%m-%d").to_string();
        let mut orders = b.orders(28);
        b.say(
            &mut steps,
            &session_id,
            "user",
            format!(
                "Can you pull page {page} of the orders for customer {}?",
                cust.id
            ),
        );
        b.say(
            &mut steps,
            &session_id,
            "assistant",
            "Checking the orders API.".into(),
        );

        // A stale copy of an earlier order, re-fetched with a new status.
        let mut latest_q = None;
        if let Some((old_call, old, old_ev)) = seen[ci].take() {
            let new_status = STATUSES
                .iter()
                .find(|st| **st != old.status)
                .copied()
                .unwrap();
            let updated = b.clock.format("%Y-%m-%dT%H:%M:%SZ").to_string();
            let slot = b.rng.below(orders.len());
            orders[slot] = Order {
                id: old.id.clone(),
                status: new_status,
                amount: old.amount,
                sku: old.sku.clone(),
                updated_at: updated.clone(),
            };
            latest_q = Some((slot, old_call, old, old_ev, new_status, updated));
        }

        let payload = orders_payload(&cust.id, page, currency, &as_of, &orders);
        let args = json!({"customer_id": cust.id, "page": page});
        let call = b.tool(
            &mut steps,
            &session_id,
            "list_orders",
            args,
            PayloadFormat::Json,
            payload,
        );
        let ev = |i: usize, o: &Order| Evidence {
            tokens: vec![o.id.clone(), o.updated_at.clone()],
            call_id: call.clone(),
            address: format!("/data/items/{i}"),
        };

        let k = b.rng.below(orders.len());
        let o = &orders[k];
        b.question(
            Category::Lookup,
            format!("What is the status of order {}?", o.id),
            o.status.into(),
            vec![ev(k, o)],
            vec![],
            None,
        );
        let k2 = (k + 1 + b.rng.below(orders.len() - 1)) % orders.len();
        let o2 = &orders[k2];
        b.question(
            Category::Inherited,
            format!(
                "In which currency is the amount of order {} given, and in what unit?",
                o2.id
            ),
            format!("{} {}", currency.0, currency.1),
            vec![ev(k2, o2)],
            vec![currency.0.into(), currency.1.into()],
            None,
        );
        let shipped: Vec<usize> = (0..orders.len())
            .filter(|i| orders[*i].status == "shipped")
            .take(5)
            .collect();
        if shipped.len() >= 2 {
            let total: u64 = shipped.iter().map(|i| orders[*i].amount).sum();
            b.question(
                Category::Aggregation,
                format!(
                    "What is the total amount of the shipped orders on page {page} for customer {}?",
                    cust.id
                ),
                total.to_string(),
                shipped.iter().map(|i| ev(*i, &orders[*i])).collect(),
                vec![currency.0.into()],
                None,
            );
        }
        if let Some((slot, _old_call, old, old_ev, new_status, updated)) = latest_q {
            b.question(
                Category::Latest,
                format!("What is the latest status of order {}?", old.id),
                new_status.into(),
                vec![Evidence {
                    tokens: vec![old.id.clone(), updated],
                    call_id: call.clone(),
                    address: format!("/data/items/{slot}"),
                }],
                vec![],
                Some(old_ev),
            );
        }
        let keep = b.rng.below(orders.len());
        let kept = Order {
            id: orders[keep].id.clone(),
            status: orders[keep].status,
            amount: orders[keep].amount,
            sku: orders[keep].sku.clone(),
            updated_at: orders[keep].updated_at.clone(),
        };
        let kept_ev = ev(keep, &kept);
        seen[ci] = Some((call.clone(), kept, kept_ev));

        // --- Customer record, joined to an order --------------------------
        let cust_payload = serde_json::to_string_pretty(&json!({
            "customer": {
                "customer_id": cust.id,
                "name": cust.name,
                "tier": cust.tier,
                "region": cust.region,
            },
            "meta": {"source": "crm", "fetched_at": b.clock.to_rfc3339()},
        }))
        .unwrap();
        let ccall = b.tool(
            &mut steps,
            &session_id,
            "get_customer",
            json!({"customer_id": cust.id}),
            PayloadFormat::Json,
            cust_payload,
        );
        let j = b.rng.below(orders.len());
        let oj = &orders[j];
        b.question(
            Category::Join,
            format!(
                "What loyalty tier does the customer who placed order {} have?",
                oj.id
            ),
            cust.tier.into(),
            vec![
                ev(j, oj),
                Evidence {
                    tokens: vec![cust.id.clone(), cust.tier.into()],
                    call_id: ccall,
                    address: "/customer".into(),
                },
            ],
            vec![],
            None,
        );

        // --- SQL result as CSV: header row and unit live outside the row --
        let unit = if s % 2 == 0 {
            "thousands_usd"
        } else {
            "millions_jpy"
        };
        let mut csv = String::from("region,month,revenue,orders\n");
        let mut rows = Vec::new();
        for region in REGIONS {
            for m in 1..=9 {
                let rev = 100 + b.rng.below(9000);
                let n = 10 + b.rng.below(900);
                let month = format!("2026-{m:02}");
                csv.push_str(&format!("{region},{month},{rev},{n}\n"));
                rows.push((region, month, rev));
            }
        }
        let scall = b.tool(
            &mut steps,
            &session_id,
            "run_sql",
            json!({
                "query": "SELECT region, month, revenue, orders FROM monthly_sales ORDER BY region, month",
                "revenue_unit": unit,
            }),
            PayloadFormat::Csv,
            csv,
        );
        let r = b.rng.below(rows.len());
        let (region, month, rev) = &rows[r];
        b.question(
            Category::Inherited,
            format!("What was the revenue for {region} in {month}, and in what unit?"),
            format!("{rev} {unit}"),
            vec![Evidence {
                tokens: vec![format!("{region},{month},{rev}")],
                call_id: scall,
                address: format!("rows {}", r + 1),
            }],
            vec!["revenue".into(), unit.into()],
            None,
        );

        // --- Service logs: one error among many similar lines -------------
        let service = *b.rng.pick(SERVICES);
        let mut log = String::new();
        let err_line = b.rng.below(150);
        let mut err = None;
        for i in 0..150 {
            let t = b.clock + Duration::milliseconds(i as i64 * 431);
            let req = format!("req-{}", b.rng.hex(8));
            if i == err_line {
                let e = *b.rng.pick(ERRORS);
                log.push_str(&format!(
                    "{} ERROR {service} request_id={req} error={e} attempt=3\n",
                    t.format("%Y-%m-%dT%H:%M:%S%.3fZ")
                ));
                err = Some((req, e));
            } else {
                log.push_str(&format!(
                    "{} INFO {service} request_id={req} status=200 latency_ms={}\n",
                    t.format("%Y-%m-%dT%H:%M:%S%.3fZ"),
                    3 + b.rng.below(400)
                ));
            }
        }
        let lcall = b.tool(
            &mut steps,
            &session_id,
            "fetch_logs",
            json!({"service": service, "since": b.clock.to_rfc3339(), "limit": 150}),
            PayloadFormat::Log,
            log,
        );
        let (req, e) = err.unwrap();
        b.question(
            Category::Lookup,
            format!("Which error did request {req} fail with?"),
            e.into(),
            vec![Evidence {
                tokens: vec![req.clone(), format!("error={e}")],
                call_id: lcall,
                address: format!("lines {}", err_line + 1),
            }],
            vec![],
            None,
        );

        // --- Test run: the commit is only in the tool's arguments ---------
        let commit = b.rng.hex(7);
        let failing = format!(
            "tests/test_{}.py::test_{}_{}",
            service,
            *b.rng.pick(&["refund", "capture", "retry", "settle"]),
            b.rng.below(90)
        );
        let mut out = String::from(
            "============================= test session starts ==============================\n",
        );
        let fail_at = b.rng.below(40);
        for i in 0..40 {
            if i == fail_at {
                out.push_str(&format!("{failing} FAILED\n"));
            } else {
                out.push_str(&format!(
                    "tests/test_{}.py::test_case_{} PASSED\n",
                    service,
                    b.rng.below(400)
                ));
            }
        }
        out.push_str("=========================== 1 failed, 39 passed in 12.31s ===========================\n");
        let tcall = b.tool(
            &mut steps,
            &session_id,
            "run_tests",
            json!({"commit": commit, "suite": service}),
            PayloadFormat::TestOutput,
            out,
        );
        b.question(
            Category::Inherited,
            format!("Which test failed when the {service} suite ran on commit {commit}?"),
            failing.clone(),
            vec![Evidence {
                tokens: vec![format!("{failing}FAILED")],
                call_id: tcall,
                address: format!("lines {}", fail_at + 2),
            }],
            vec![commit],
            None,
        );

        b.say(
            &mut steps,
            &session_id,
            "assistant",
            "Done, I have the results.".into(),
        );

        // --- An identifier that exists nowhere ----------------------------
        if s % 4 == 3 {
            b.question(
                Category::Unanswerable,
                format!("What is the status of order ORD-{}?", 100_000 + s),
                String::new(),
                vec![],
                vec![],
                None,
            );
        }

        sessions.push(GenSession { session_id, steps });
    }

    Dataset {
        sessions,
        questions: b.questions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_dataset() {
        let a = generate(7, 6);
        let b = generate(7, 6);
        assert_eq!(a.questions.len(), b.questions.len());
        for (x, y) in a.questions.iter().zip(&b.questions) {
            assert_eq!(x.text, y.text);
            assert_eq!(x.answer, y.answer);
        }
    }

    #[test]
    fn every_evidence_token_is_in_its_payload() {
        let d = generate(11, 8);
        let payloads: std::collections::HashMap<&str, String> = d
            .sessions
            .iter()
            .flat_map(|s| &s.steps)
            .filter_map(|st| match st {
                Step::Tool { invocation, .. } => Some((
                    invocation.call_id.as_str(),
                    invocation.payload.split_whitespace().collect::<String>(),
                )),
                Step::Say { .. } => None,
            })
            .collect();
        for q in &d.questions {
            for ev in q.evidence.iter().chain(q.stale.as_ref()) {
                let p = &payloads[ev.call_id.as_str()];
                for t in &ev.tokens {
                    let t: String = t.split_whitespace().collect();
                    assert!(p.contains(&t), "{}: {t} not in {}", q.id, ev.call_id);
                }
            }
        }
    }
}
