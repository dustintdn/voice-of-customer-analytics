# Voice-of-Customer (VoC) Intelligence Pipeline

> **Business question:** Across hundreds of thousands of unstructured customer
> complaints, *what are people actually upset about, is it getting better or
> worse, and what predicts an unhappy outcome* — with enough statistical rigor
> that a leadership team can act on the numbers?

This is an end-to-end data science pipeline that turns raw customer text into
quantified, statistically-defensible insights and an auto-generated executive
report. Built on the public [CFPB Consumer Complaint Database](data/README.md).

## Pipeline

```
raw CSV → 1. Ingest → 2. Embed → 3. Themes → 4. LLM extract →┬→ 5. Predict ─┐
              (vector store)   (BERTopic)   (structured JSON) └→ 6. Stats ───┴→ 7. Report
```

1. **Ingest** — clean, dedup, date-parse; emit a data-quality report.
2. **Embed** — `sentence-transformers` + vector store with `semantic_search`.
3. **Themes** — BERTopic clustering; per-theme summaries + prevalence over time.
4. **LLM extract** — free text → schema-validated JSON (sentiment / category /
   severity / actionable), with prompt-variant evaluation; checkpointed,
   cost-capped real runs and an offline `--dry-run` mock.
5. **Predict** — forecast a behavioral outcome from text-derived features
   (gradient boosting vs. logistic baseline, time-aware split).
6. **Statistics** — confidence intervals on every proportion, significance
   testing of change-over-time with multiple-comparison correction, trend
   detection. *The differentiator.*
7. **Report** — executive-readable `reports/voc_insight_report.md` with charts.

## Quickstart

```bash
make setup      # create venv + install pinned deps
make sample     # build the small committed sample (synthesizes if no raw CSV)
make run        # run the pipeline end-to-end on the sample (offline, LLM mocked)
make test       # run the test suite (offline)
```

The report lands at `reports/voc_insight_report.md`. To run on the real data,
follow [`data/README.md`](data/README.md) to download the CFPB CSV first.

## LLM extraction — run protocol (M4)

The LLM stage is offline-testable and built to spend money safely:

```bash
# Offline (mock LLM, no network, no spend):
voc extract --dry-run               # structured fields for the sample
voc evaluate --dry-run              # compare prompt variants on the gold set

# Real, paid run (needs ANTHROPIC_API_KEY; starts small per max_spend_usd):
python scripts/make_gold_stub.py --n 80      # create a labeling stub; hand-label it
voc evaluate                                  # pick the best prompt on the gold set
voc extract --yes                             # run the winning prompt on the sample
```

Recommended order (per `docs/SPEC.md` §5.4.1): run the gold-set evaluation first
(the only place you pay for multiple variants), pick the winner, then run the
larger statistically-drawn sample with **only** that prompt. The stage is
checkpointed (`record_id`-keyed JSONL) so a crash never re-bills completed
records, enforces `max_records`/`max_spend_usd`, and writes measured token/cost
figures to `reports/llm_run_summary.md`.

## Design choices that matter

These are filled in as the milestones land (see below). The pipeline is
**config-driven** (`config/config.yaml`), **dataset-agnostic** (column
mapping), and **offline-testable** (mock LLM + committed sample).

## Build status

This repo is built milestone by milestone (see `docs/SPEC.md` §8).

- [x] **M0 — Scaffold:** structure, typed config loader, Makefile, sample
      script, test stub (`make test` green).
- [x] **M1 — Ingest:** config-mapped CSV → cleaned `records.parquet` (two text
      columns, dedup, date parsing, time columns) + `reports/data_quality.md`;
      transforms unit-tested.
- [x] **M2 — Embed + vector store:** injectable embedder (sentence-transformers
      default + offline hashing backend), content-hash embedding cache, ChromaDB
      index, and `semantic_search`.
- [x] **M3 — Themes:** config-selected clustering (BERTopic default + offline
      KMeans backend), c-TF-IDF keywords, representative examples, explicit
      outlier cluster, and a theme-prevalence-over-time table; `reports/themes.md`.
- [x] **M4 — LLM extraction:** schema-validated structured fields with robust
      parsing; 3 prompt variants + offline eval harness (`reports/llm_eval.md`);
      checkpoint/resume, bounded concurrency, retry+backoff, `max_records`/
      `max_spend_usd` caps, pre-run cost estimate, measured token/cost accounting
      (`reports/llm_run_summary.md`); fully offline via `--dry-run`.
- [x] **M5 — Predictive model:** forecasts a behavioral outcome from
      text-derived features (PCA embeddings, theme, LLM fields, metadata) with a
      time-aware split; gradient boosting vs. logistic baseline; ROC-AUC/PR-AUC/
      Brier, confusion matrix, permutation importance (`reports/predictive_model.md`).
- [x] **M6 — Statistics:** sample-size design, Wilson/bootstrap CIs on every
      proportion, two-proportion z-tests of change-over-time with
      Benjamini-Hochberg correction, emergence/fading detection; findings carry
      estimate + CI + test stat + corrected p-value + plain-language reading
      (`reports/statistics.md`). Primitives unit-tested against textbook values.
- [x] **M7 — Report:** auto-generated executive `reports/voc_insight_report.md`
      (optional `.html`) — leads with the "so what", CI on every number,
      significant movers, what predicts the outcome, methodology + caveats, and
      charts (prevalence-over-time with CI bands, sentiment, feature importance).
- [ ] M8 — Polish (real result numbers land here)

## Repository layout

```
config/      config.yaml — single source of truth
data/        README (download + schema), committed sample (raw/processed gitignored)
src/voc/     config, ingest, embed, themes, llm_extract, predict, stats, report, pipeline
prompts/     versioned LLM prompt templates (M4)
scripts/     make_sample.py
tests/       pytest (offline)
reports/     generated outputs (gitignored)
docs/        SPEC.md
```
