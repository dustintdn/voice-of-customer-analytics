# Voice-of-Customer (VoC) Intelligence Pipeline

> **Business question:** Across hundreds of thousands of unstructured customer
> complaints, *what are people actually upset about, is it getting better or
> worse, and what predicts an unhappy outcome* — with enough statistical rigor
> that a leadership team can act on the numbers?

An end-to-end data science pipeline that turns raw customer text into quantified,
statistically-defensible insights and an auto-generated executive report. Built
on the public [CFPB Consumer Complaint Database](data/README.md), but
**dataset-agnostic** via a config-defined column mapping.

## Pipeline

```
raw CSV → 1. Ingest → 2. Embed → 3. Themes → 4. LLM extract →┬→ 5. Predict ─┐
              (vector store)   (BERTopic)   (structured JSON) └→ 6. Stats ───┴→ 7. Report
```

1. **Ingest** — clean, dedup, date-parse; emit a data-quality report.
2. **Embed** — `sentence-transformers` + ChromaDB vector store with `semantic_search`.
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
make setup      # venv + pinned deps + pre-fetch the embedding model (one-time network)
make sample     # build the committed sample (synthesizes a CFPB-shaped set if no raw CSV)
make run        # full pipeline end-to-end; LLM mocked (--dry-run). Report -> reports/
make test       # offline test suite (no network)
```

The report lands at **`reports/voc_insight_report.md`** (charts in `reports/figures/`).

**Two backends for every heavy stage** (`config/config.yaml`):

| Stage | Real (default) | Offline (no heavy deps, no network) |
|---|---|---|
| Embed | `sentence-transformers` (`all-mpnet-base-v2`) | `hashing` feature embedder |
| Themes | `bertopic` | `kmeans` (scikit-learn) |
| Predict | `lightgbm` | `sklearn` HistGradientBoosting |
| LLM | Anthropic API | `--dry-run` deterministic mock |

`make run` uses the real backends (so `make setup` installs them and caches the
model — the one network step). To run the **whole pipeline fully offline** with
no heavy install, flip the four backends:

```bash
sed -e 's/backend: sentence_transformers/backend: hashing/' \
    -e 's/algorithm: bertopic/algorithm: kmeans/' \
    -e 's/^  model: lightgbm.*/  model: sklearn/' \
    config/config.yaml > config.offline.yaml
python -m voc.pipeline run --config config.offline.yaml --dry-run
```

## Results

> ⚠️ **These are demo numbers from the offline run on the synthetic, CFPB-shaped
> sample** (`hashing` + `kmeans` + `sklearn` + mock LLM). They exercise the full
> pipeline but are **not** real findings. See *Reproduce with real data* below to
> generate genuine numbers from the actual CFPB corpus and a live LLM run.

**Data quality (M1).** 3,000 raw → **2,552** cleaned records (14.9% exact-duplicate
removal), spanning 36 months.

**Themes (M3).** 8 themes + an explicit outlier cluster (5.0% of docs), each with
keywords, representative examples, and a prevalence-over-time table.

**Prompt-variant evaluation (M4).** Issue-category accuracy on the 80-record gold
set rises with prompt sophistication, and cost rises with prompt length:

| Variant | issue_category acc | $/1k records |
|---|--:|--:|
| v1 (zero-shot) | 0.44 | $0.38 |
| v2 (few-shot) | 0.54 | $0.52 |
| **v3 (few-shot + rubric)** | **0.75** | $0.81 |

**Prediction (M5).** Target `not_timely_response` (base rate 12.5%), time-aware
split (train ≤ 2023-03-31, test after). Gradient boosting on text-derived
features **beats the logistic baseline: ROC-AUC 0.645 vs 0.595** (Brier 0.123 vs
0.227). Top signals: `text_length`, embedding components, `sentiment_score`.

**Statistics (M6).** Estimating a theme's prevalence within ±3% at 95% confidence
needs **1,068** labeled records; 2,552 were labeled. Top theme prevalence: 15.4%
(95% CI 14.0%–16.8%). *No theme was flagged as a significant mover* — by
construction the synthetic sample is temporally stationary, so Benjamini-Hochberg
correctly reports no trend (the emergence-detection path is verified by unit
tests). On real data a mover reads like: *"Theme X rose from 8% to 12%, a
statistically significant increase (p<0.01, BH-corrected)."*

### Reproduce with real data

```bash
# 1. Get the real corpus (instructions + URL in data/README.md)
curl -L -o data/raw/complaints.csv.zip https://files.consumerfinance.gov/ccdb/complaints.csv.zip
unzip data/raw/complaints.csv.zip -d data/raw/
make sample                                  # now slices the real CSV

# 2. Real NLP backends (defaults already point here)
make setup
python -m voc.pipeline ingest --full         # ingest the full corpus
python -m voc.pipeline embed
python -m voc.pipeline themes

# 3. Real, paid LLM run (CLAUDE.md: keep the first run cheap — max_spend_usd ≈ 2)
export ANTHROPIC_API_KEY=...
python scripts/make_gold_stub.py --n 100      # then hand-label data/sample/gold_stub.csv -> gold.csv
python -m voc.pipeline evaluate               # pick the best prompt (only place you pay per-variant)
python -m voc.pipeline extract --yes          # winning prompt on the statistically-sized sample

# 4. Downstream + report
python -m voc.pipeline predict
python -m voc.pipeline stats
python -m voc.pipeline report
```

Then replace the numbers above with the values from `reports/`.

## Design choices that signal seniority

- **Time-aware validation, not a random split.** Complaints arrive over time; a
  random split leaks future information into the past and inflates metrics. We
  train on earlier periods and test on later ones, and fit the unsupervised PCA
  on the training split only.
- **A confidence interval on every proportion.** No bare point estimates — each
  theme share and sentiment share carries a Wilson score interval, so readers see
  the precision, not just a number that could be sampling noise.
- **The LLM runs on a statistically-drawn sample.** A full LLM pass over hundreds
  of thousands of records is expensive and unnecessary: a sample sized by the
  margin-of-error calculator (M6) estimates prevalence to a stated precision at a
  fraction of the cost. The stage is checkpointed, concurrency-bounded, retried
  with backoff, and hard-capped on spend.
- **Multiple-comparison correction.** Testing every theme for change-over-time
  means dozens of simultaneous tests; without correction ~1 in 20 looks
  "significant" by chance. Benjamini-Hochberg controls the false-discovery rate so
  flagged movers are trustworthy.
- **Measured, not estimated, LLM cost.** Token counts and spend come from real API
  responses and are persisted to `reports/llm_run_summary.md`.

## Limitations

- The **committed sample is synthetic** (CFPB-shaped) so the repo runs offline;
  headline insights require the real corpus.
- **Offline backends are intentionally simpler** than the real ones — the hashing
  embedder is bag-of-words, so offline themes can cluster on shared phrasing
  rather than pure semantics. Real `sentence-transformers` + BERTopic yield
  cleaner themes.
- **Theme labels are auto-generated** keyword summaries, not curated names.
- The pipeline is a **batch analysis tool**, not a production service (no API
  server, no real-time serving — see SPEC stretch goals).
- LLM extraction quality depends on the chosen model and prompt; the gold-set
  evaluation measures it but the gold set should be sized/curated for the domain.

## Testing

```bash
make test        # 67 offline tests: ingest transforms, embedding cache + search,
                 # clustering, LLM parsing/retry/checkpoint/spend-cap, predictive
                 # split/eval, and statistical primitives vs. textbook values.
make lint        # ruff
```

All tests run offline (mock LLM + committed sample); statistical functions are
verified against known textbook values.

## Repository layout

```
config/      config.yaml — single source of truth (paths, column mapping, params)
data/        README (download + schema), committed sample (raw/processed gitignored)
src/voc/     config, schema, ingest, embed, themes, llm_extract, predict, stats, report, pipeline
prompts/     versioned LLM prompt templates (v1/v2/v3) + system prompt
scripts/     make_sample.py, make_gold_stub.py, fetch_model.py
tests/       pytest (offline)
reports/     generated outputs (gitignored)
docs/        SPEC.md
```

## Status

All milestones complete (M0–M8 per `docs/SPEC.md` §8). The full pipeline runs
end-to-end (`voc run`) and the test suite is green offline. Real result numbers
are produced by following *Reproduce with real data* above.
```
M0 Scaffold · M1 Ingest · M2 Embed · M3 Themes · M4 LLM · M5 Predict · M6 Stats · M7 Report · M8 Polish
```
