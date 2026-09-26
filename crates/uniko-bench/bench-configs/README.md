# bench-configs

JSON profiles for `uniko-bench` runs.  Each file is a complete spec
of every model / device / recall / cost knob.  Run with:

```bash
./crates/uniko-bench/run.sh \
    --bench-config crates/uniko-bench/bench-configs/<profile>.json \
    --data data/locomo10.json \
    --conversations conv-26 \
    --output data/results.json
```

CLI carries only what changes per invocation: data file, output path,
conversation/category filters, KB reuse.  All model selection lives
in the JSON.

## Datasets

Not in the repo — `data/` is gitignored (KB stores are GB-scale, and the
source corpora are third-party). Fetch them before a run:

| File | Source |
|---|---|
| `data/locomo10.json` | `snap-research/locomo`, `data/locomo10.json` on `main`. 2,805,274 bytes, sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. 10 conversations / 5,882 turns / 1,986 questions — the figures the README quotes. From *Evaluating Very Long-Term Conversational Memory of LLM Agents* (Maharana et al., Snap Research). |
| `data/longmemeval_s_cleaned.json` | LongMemEval `_s` split. Provenance not yet recorded here — fill in when next fetched. |

The LoCoMo entry was reconstructed by matching `src/data.rs`'s expected
filename and schema against the published dataset, then confirming the
conversation / turn / question counts; it was not recorded anywhere when
the benchmark was first run. Verify the checksum rather than assuming a
re-download matches.

## Shipped profiles

| File | Generator | Judge | Embedder | Notes |
|---|---|---|---|---|
| `locomo-bge-openai.json` | gpt-4o-mini | gpt-4o-mini | bge-small (384d ONNX-CUDA) | Canonical baseline.  Last measured: 0.849 judge, $0.083/conv. |
| `locomo-bgesmall-rerank-retrieval.json` | — (retrieval-only) | — | bge-small | The shipped recall defaults (rerank-on + boost + α=0.6, per `3d3afbd`) with no LLM, so a full corpus run needs no API key. The other bge-small retrieval-only profile, `locomo-bgesmall-retrieval.json`, disables the reranker and therefore does **not** measure the default configuration. |
| `locomo-bge-gemini31.json` | gemini-3.1-flash-lite | gemini-3.1-pro-preview | bge-small | All-Gemini run via Vertex global endpoint.  Requires `VERTEXAI_PROJECT`/`VERTEXAI_LOCATION` env. |
| `locomo-embeddinggemma-openai.json` | gpt-4o-mini | gpt-4o-mini | embeddinggemma-300m (ONNX-CUDA) | A/B vs bge-small for the embedder swap.  768d vectors — fresh ingest required. |

## Sparse + ColBERT hybrid sweep

Four arms isolate one variable each so gains are read as deltas — see
`docs/design/sparse-multivector-integration-plan.md`. Tune retrieval-only
on LongMemEval `--phase1`, cross-check on LoCoMo evidence_hit, then run the
winning arm judged. **Arms A–C need a fresh ingest** (dense dim 384→1024 +
new sparse/colbert columns); KBs are reusable within an arm, never across
the bge-small→bge-m3 boundary.

| File (LoCoMo / LME twin) | Embedder | Sparse | Reranker | Isolates |
|---|---|---|---|---|
| `locomo-arm0-bge-small-baseline` / `lme-arm0-bge-small-baseline` | bge-small | off | cross-encoder | current main |
| `locomo-arm-a-bge-m3-dense` / `lme-arm-a-bge-m3-dense` | bge-m3 | off | cross-encoder | dense-model upgrade (A−0) |
| `locomo-arm-b-bge-m3-sparse` / `lme-arm-b-bge-m3-sparse` | bge-m3 | on | cross-encoder | learned-sparse (B−A) |
| `locomo-arm-c-bge-m3-colbert` / `lme-arm-c-bge-m3-colbert` | bge-m3 | on | colbert (MaxSim) | late-interaction (C−B) |

!!! note
    **Arms B and C could not measure anything before 2026-09-19.** Both the
    learned-sparse and ColBERT query paths failed on every query and were
    swallowed (`debug` and `warn` respectively), so the channels contributed
    no candidates while still costing a query each. Any B−A or C−B delta
    recorded before that date is an artifact, not a measurement. See
    `website/docs/guides/configuration.md` for the fix, the first real
    numbers, and how to verify a channel is live.

`reranker.style = "colbert"` re-scores the top candidates in-process by
ColBERT MaxSim over the `colbert_embedding` column — it registers no
reranker model (the `model_id` is a label only). Always report
`avg_recall_latency_ms` and index storage alongside the quality deltas.

## Schema reference

See the module-level doc comment on
`crates/uniko-bench/src/bench_config.rs::BenchConfig` for the full
serde schema.  Key fields:

- `models.gen` / `models.judge` — required `LlmAlias` objects: `alias`, `model_id`, `provider`, optional `base_url`, optional `use_default_options`.
- `models.extract_triples` — optional; set to an `LlmAlias` to use LLM triple extraction during P4 consolidation.
- `models.embedder` — either `{ "preset": "<name>" }` or `{ "inline": <EmbeddingConfig> }`.  Presets: `bge-small`, `bge-large`, `bge-m3`, `nomic`, `minilm`, `embeddinggemma`, `embeddinggemma-mistralrs`.  `bge-m3` is single-pass hybrid (dense + sparse + ColBERT); pair with `recall.sparse_enabled` and/or `reranker.style = "colbert"`.
- `recall.sparse_enabled` — add the learned-sparse channel (needs `bge-m3`).  `recall.vector_weight` / `recall.bm25_weight` — override hybrid fusion weights (`null` keeps the 0.5/0.5 default).
- `models.reranker` — full `RerankerConfig` shape (`enabled`, `model_id`, `style`, `top_n`, `apply_sigmoid`, optional `execution_providers`).
- `models.nlp` — `model_id`, `artifact`, `max_batch_size`, optional `execution_providers`.  Defaults to the xsmall deberta INT8 cascade.
- `recall.*` — phase strategies, recall_limit, variants, consolidation toggles.
- `cost.pricing_csv` — path to the per-model price table.  `cost.no_events` skips the JSONL event writer.

## Per-component CPU/GPU control

Every component (`models.reranker`, `models.nlp`, and the inline
embedder) accepts an `execution_providers` array, e.g.:

```json
"reranker": {
  "enabled": true,
  "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "top_n": 50,
  "apply_sigmoid": false,
  "execution_providers": ["cpu"]
}
```

Leave the field `null` to inherit the build-time default
(`["cuda","cpu"]` on `gpu-cuda` builds, `["cpu"]` otherwise).  The
NLP cascade additionally inherits the embedder's setting when its
own `execution_providers` is `null`.
