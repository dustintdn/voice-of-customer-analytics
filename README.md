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

## Design choices that matter

These are filled in as the milestones land (see below). The pipeline is
**config-driven** (`config/config.yaml`), **dataset-agnostic** (column
mapping), and **offline-testable** (mock LLM + committed sample).

## Build status

This repo is built milestone by milestone (see `docs/SPEC.md` §8).

- [x] **M0 — Scaffold:** structure, typed config loader, Makefile, sample
      script, test stub (`make test` green).
- [ ] M1 — Ingest
- [ ] M2 — Embed + vector store
- [ ] M3 — Themes
- [ ] M4 — LLM extraction
- [ ] M5 — Predictive model
- [ ] M6 — Statistics
- [ ] M7 — Report
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
