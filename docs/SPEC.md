# Project Specification: Voice-of-Customer (VoC) Intelligence Pipeline

> **Purpose of this document:** This is a build specification for Claude Code. It defines an end-to-end data science project that turns large volumes of unstructured customer text into quantified, statistically-defensible business insights. Build the project in the milestone order given. Treat the *Acceptance Criteria* at the end as the definition of done.

---

## 1. Context & Motivation

This project is a portfolio piece designed to mirror the work of a **Business Data Scientist on a "Customer Voice" / Voice-of-Customer analytics team**. The target workflow is:

> Ingest hundreds of thousands of unstructured customer messages → extract nuanced themes and signals → cluster and quantify them → predict downstream customer outcomes from text-derived features → report everything with rigorous statistical confidence → communicate the "so what" clearly.

The finished repo should read as a single coherent system, not a pile of disconnected notebooks. A senior reviewer should be able to clone it, run one command, and see the whole pipeline produce trustworthy, quantified insights.

The four capabilities this project must demonstrate, in priority order:

1. **NLP on unstructured conversational text** — embeddings, topic/theme extraction, clustering.
2. **LLM-based structured extraction** — prompt engineering to convert free text into structured fields, with a real evaluation harness.
3. **Predictive modeling from text-derived features** — using NLP outputs as features to forecast a behavioral outcome.
4. **Statistical rigor in reporting** — confidence intervals, significance testing, sampling strategy, trend-change detection. *This is the differentiator; do not treat it as an afterthought.*

---

## 2. Goals & Non-Goals

### Goals
- A reproducible, modular Python pipeline runnable end-to-end from the command line.
- Clean separation between stages (ingest → embed → cluster → extract → predict → report).
- Every reported number that could vary by sample carries an uncertainty estimate.
- A short, executive-readable insight report generated automatically as the final output.
- Code quality that holds up to review: typed function signatures, docstrings, tests on the statistical and data-transformation logic, config-driven (no hardcoded paths/magic numbers).

### Non-Goals
- Not a production service. No real-time serving, no API server, no containerized deployment (these can be listed as stretch goals only).
- Not a from-scratch model training exercise. Use pretrained embedding models and existing libraries.
- Not a front-end engineering project. A lightweight dashboard is a stretch goal; the primary deliverable is the pipeline + the generated report.

---

## 3. Dataset

**Primary dataset: CFPB Consumer Complaint Database** (U.S. Consumer Financial Protection Bureau).

Rationale for this choice:
- Hundreds of thousands of real, free-text consumer complaint narratives — closely mirrors "hundreds of thousands of customer conversations."
- Rich structured metadata: product/sub-product category, company, date received, state, and company response.
- Contains a usable **behavioral outcome** for the predictive stage: whether the consumer disputed the company's response (`Consumer disputed?`) and/or whether the response was timely. This lets the predictive model forecast a real downstream outcome from text-derived features.
- Timestamps enable genuine **time-series / trend-change** analysis, which the statistical layer needs.
- Public, free, redistributable.

Acquisition: the dataset is available as a bulk CSV download from the CFPB. The pipeline must **not** hardcode a download step that requires network access at run time — instead:
- Provide a `data/README.md` with the exact download URL and instructions.
- The ingest stage should read from a local CSV path specified in config.
- Include a `make sample` (or script) that produces a small committed sample (~2–5k rows) so the repo is runnable/testable without the full download.

**Fallback / alternative datasets** (document these in the data README so the project generalizes): Amazon product reviews, Yelp Open Dataset, or any support-ticket dataset with text + a categorical/temporal/outcome column. The pipeline should be dataset-agnostic via the config-defined column mapping (see §5.1).

---

## 4. High-Level Architecture

```
                    ┌─────────────────┐
   raw CSV  ───────▶│  1. Ingest /    │
                    │     Preprocess  │
                    └────────┬────────┘
                             │ cleaned records (parquet)
                             ▼
                    ┌─────────────────┐
                    │  2. Embed       │──▶ vector store (Chroma/FAISS)
                    └────────┬────────┘
                             │ embeddings
                             ▼
                    ┌─────────────────┐
                    │  3. Theme       │──▶ topic model + cluster labels
                    │     Extraction  │
                    └────────┬────────┘
                             │ doc→theme assignments
                             ▼
                    ┌─────────────────┐
                    │  4. LLM         │──▶ structured fields
                    │     Extraction  │    (sentiment/category/severity)
                    └────────┬────────┘
                             │ enriched feature table
                             ▼
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌─────────────────┐          ┌─────────────────────┐
   │  5. Predictive  │          │  6. Statistical     │
   │     Model       │          │     Reporting       │
   └────────┬────────┘          └──────────┬──────────┘
            │                              │
            └──────────────┬───────────────┘
                           ▼
                 ┌─────────────────┐
                 │  7. Insight     │──▶ executive report (md/html)
                 │     Report      │
                 └─────────────────┘
```

---

## 5. Module Specifications

### 5.1 Module 1 — Ingest & Preprocess
**Input:** raw CSV path (from config).
**Output:** cleaned record table written to `data/processed/records.parquet`.

Requirements:
- A **config-driven column mapping** so the pipeline isn't tied to CFPB column names. Config defines: `text_column`, `date_column`, `category_column`, `outcome_column`, `id_column`, plus any grouping columns (e.g., company, state).
- Drop records with empty/near-empty text (configurable minimum token threshold).
- Basic text normalization: whitespace, control-character stripping, lowercasing handled *only where appropriate* (preserve casing for the LLM stage; clustering can use normalized text). Document the choice.
- Deduplicate exact and near-duplicate narratives (near-dup via hashing or minhash is fine; keep it simple).
- Parse dates to a proper datetime; derive `year_month` and `week` columns for time-series work.
- Emit a short **data quality report** (row counts before/after each filter, null rates, date range, category distribution) as `reports/data_quality.md`.

### 5.2 Module 2 — Embedding
**Input:** cleaned records.
**Output:** embeddings persisted, plus a queryable vector store.

Requirements:
- Use `sentence-transformers` (default model: a strong general-purpose model such as `all-mpnet-base-v2`; make the model name a config value).
- Batch encode with progress logging; cache embeddings to disk keyed by a content hash so re-runs don't re-embed unchanged data.
- Load embeddings into a **vector database** — ChromaDB (preferred for simplicity) or FAISS. Expose a `semantic_search(query, k)` helper that returns the nearest records. This demonstrates the vector-DB / embedding-model competency explicitly.

### 5.3 Module 3 — Theme Extraction (Clustering / Topic Modeling)
**Input:** embeddings + cleaned text.
**Output:** per-document theme assignments + per-theme summaries.

Requirements:
- Use **BERTopic** (preferred) layered on the precomputed embeddings, or an HDBSCAN/UMAP + c-TF-IDF approach. Make the clustering algorithm and key hyperparameters config-driven.
- Produce for each theme: an ID, a human-readable label, top representative keywords, the count and share of documents, and 3–5 representative example snippets.
- Handle the outlier/noise cluster explicitly (don't silently drop it; report its size).
- Persist `theme_assignments.parquet` (doc_id → theme_id, probability) and `theme_summary.parquet`.
- Compute **theme prevalence over time** (theme share per `year_month`) — this feeds the statistical trend analysis in Module 6.

### 5.4 Module 4 — LLM Structured Extraction
**Input:** a representative sample of records (full pass is optional/costly — see sampling note).
**Output:** structured fields appended to the feature table.

Requirements:
- Use an LLM API to convert each free-text record into a **strict JSON object** with fields: `sentiment` (categorical: negative/neutral/positive + a -1..1 score), `issue_category` (from a controlled vocabulary you define), `severity` (low/medium/high), and `is_actionable` (bool). Prompt must instruct the model to return JSON only, no prose.
- Implement **robust parsing**: strip code fences, validate against a schema (use `pydantic`), and handle/retry malformed responses. Never let one bad response crash the batch.
- **Prompt engineering deliverable:** include at least three prompt variants (zero-shot, few-shot, and a refined few-shot with explicit rubric) and an evaluation comparing them.
- **Evaluation harness:** hand-label a small gold set (50–150 records — script the creation of a labeling stub CSV) and report accuracy / agreement per field for each prompt variant, plus mean latency and estimated cost per 1k records. This is the part that signals real LLM rigor.
- **Cost control:** because a full LLM pass over hundreds of thousands of records is expensive, the LLM stage runs on a *statistically drawn sample by default* (see Module 6 sampling), with a config flag to run the full set. Document this tradeoff in the README.

> Note on environment: the LLM call should read its API key from an environment variable and degrade gracefully (clear error message) if absent. Do not commit keys. Provide a `--dry-run` mode that uses a stub/mock LLM response so the pipeline is testable offline.

#### 5.4.1 LLM Production Run (real, paid execution)
The dry-run mock keeps the pipeline testable offline, but this project **will be run for real** against a live API to produce genuine numbers for the report and README. The LLM stage must therefore be built to spend money safely and efficiently:

- **Checkpointing / resumability (required).** Write results incrementally to a durable store (append-only JSONL or a parquet keyed by `record_id`) as each record completes. On restart, load already-processed IDs and skip them — never re-pay for completed calls. A mid-run crash on record 8,001 must not re-bill the first 8,000.
- **Bounded concurrency.** Process records concurrently with a configurable `max_concurrency` (async or a thread pool). This is the difference between a 20-minute run and a multi-hour one. Keep concurrency a config value so it can be tuned to the account's rate limits.
- **Retry with backoff.** On 429 (rate limit) and 5xx responses, retry with exponential backoff + jitter, up to a configurable max attempts. Distinguish retryable errors from permanent ones (e.g., malformed-request 400s should fail fast, not retry).
- **Hard cost controls (required).** Config exposes `max_records` and `max_spend_usd`. Before the batch starts, print a **pre-run cost estimate** ("about to process N records, est. \$X at \$Y/1k tokens — continue?") and require confirmation unless a `--yes` flag is passed. The stage must stop and checkpoint if `max_spend_usd` is reached mid-run. This is cheap insurance against a config typo billing the full dataset.
- **Measured token/cost accounting.** Read actual input/output token counts from each API response and accumulate true spend. The cost-per-1k figures in the evaluation and README must be **measured from the real run**, not estimated. Persist a run-summary (records processed, total tokens, total \$, mean/median latency, error/retry counts) to `reports/llm_run_summary.md`.
- **Recommended run protocol (document in README):**
  1. Run the gold-set evaluation first (50–150 records across all 3 prompt variants) to select the single best prompt. This is the only place you pay for multiple variants.
  2. Run the larger statistically-drawn sample (a few thousand records, sized by Module 6's sampling calculator) with **only the winning prompt** — do not run all variants at scale (3× spend, no benefit).
  3. Quote the resulting confidence-interval'd prevalence estimates and measured spend in the README.

### 5.5 Module 5 — Predictive Model
**Input:** feature table combining text-derived features.
**Output:** trained model + evaluation report.

Requirements:
- **Target:** a real behavioral outcome from the data (CFPB: `Consumer disputed?` or a timely-response flag). State the target clearly and handle class imbalance.
- **Features must be text-derived**, demonstrating the "leverage text-derived features to forecast key outcomes" requirement. Include: embedding-based features (e.g., reduced-dimension embeddings or cluster membership), theme assignment, LLM-extracted sentiment/severity, plus light metadata (category, text length). Document which features come from which upstream module.
- Model: gradient boosting (LightGBM or XGBoost) with a logistic-regression baseline for comparison.
- **Time-aware validation:** split train/test by time (train on earlier periods, test on later) to avoid leakage — *not* a random split. Justify this in the README.
- Report ROC-AUC, PR-AUC, calibration, and a confusion matrix at a chosen threshold. Include **feature importance** and a short interpretation (SHAP optional but nice).

### 5.6 Module 6 — Statistical Reporting Layer *(the differentiator — invest here)*
**Input:** theme assignments over time, LLM-extracted fields, predictive outputs.
**Output:** a set of statistically-validated findings.

Requirements:
- **Sampling strategy module:** given a target margin of error and confidence level, compute the required sample size for estimating a theme's prevalence (proportion). Provide both the design and the realized sample used by the LLM stage. Document the method.
- **Confidence intervals on every proportion:** every reported theme share / sentiment share carries a CI (Wilson interval for proportions, bootstrap where appropriate). No bare point estimates in the final report.
- **Significance testing for change over time:** for each major theme, test whether its prevalence changed significantly between two periods (two-proportion z-test or chi-square), with multiple-comparison correction (Benjamini-Hochberg) since many themes are tested at once. Flag only the statistically significant movers.
- **Trend/emergence detection:** identify themes that are *emerging* (rising share with significance) vs *fading*. A simple, well-justified method is fine; rigor and correctness matter more than sophistication.
- Every finding object should carry: estimate, CI, test statistic, p-value (corrected), and a plain-language interpretation string.

### 5.7 Module 7 — Insight Report Generation
**Input:** outputs of Modules 3–6.
**Output:** `reports/voc_insight_report.md` (and optionally `.html`).

Requirements:
- Auto-generated, executive-readable. Lead with the "so what," not the methodology.
- Structure: (1) headline findings with confidence levels, (2) top themes with prevalence + CI + trend direction, (3) statistically significant movers, (4) what predicts the negative outcome (from Module 5), (5) methodology appendix with sample sizes and caveats.
- Each quantitative claim states its uncertainty in plain language ("Theme X rose from 8% to 12%, a statistically significant increase (p<0.01, corrected)").
- Include a few charts (matplotlib/plotly): theme prevalence over time with CI bands, sentiment distribution, feature importance.

---

## 6. Tech Stack

- **Language:** Python 3.11+
- **Core:** pandas, numpy, pyarrow
- **NLP/embeddings:** sentence-transformers, BERTopic, umap-learn, hdbscan
- **Vector store:** chromadb (preferred) or faiss-cpu
- **LLM:** an LLM API client (key via env var) + pydantic for response validation
- **ML:** scikit-learn, lightgbm (and/or xgboost), optionally shap
- **Stats:** scipy, statsmodels
- **Viz:** matplotlib and/or plotly
- **Config:** pydantic-settings or a YAML config loaded into a typed config object (includes LLM run controls: `max_concurrency`, `max_records`, `max_spend_usd`, retry/backoff settings)
- **Quality:** pytest, ruff (lint), mypy (optional), pre-commit (optional)
- **Orchestration:** a simple `Makefile` and/or a `pipeline.py` CLI (argparse or typer). No heavy orchestration framework required.

Pin versions in `pyproject.toml` (preferred) or `requirements.txt`.

---

## 7. Repository Structure

```
voc-intelligence-pipeline/
├── README.md
├── pyproject.toml            # or requirements.txt
├── Makefile                  # make setup / sample / run / test / report
├── config/
│   └── config.yaml           # all paths, model names, column mapping, params
├── data/
│   ├── README.md             # download instructions + URL
│   ├── raw/                  # gitignored
│   ├── sample/               # small committed sample for tests/demos
│   └── processed/            # gitignored
├── src/voc/
│   ├── __init__.py
│   ├── config.py             # typed config loader
│   ├── ingest.py             # Module 1
│   ├── embed.py              # Module 2
│   ├── themes.py             # Module 3
│   ├── llm_extract.py        # Module 4 (+ prompts/ subpackage)
│   ├── predict.py            # Module 5
│   ├── stats.py              # Module 6
│   ├── report.py             # Module 7
│   └── pipeline.py           # CLI orchestrator
├── prompts/                  # versioned prompt templates (v1/v2/v3)
├── tests/                    # pytest: stats, ingest transforms, parsing
├── reports/                  # generated outputs (gitignored except samples)
└── notebooks/                # optional exploration, not the deliverable
```

---

## 8. Build Order (Milestones)

Build and verify each milestone before moving on. Each should leave the repo in a runnable state.

1. **M0 — Scaffold:** repo structure, config loader, Makefile, sample-data script, CI-free test stub. `make test` runs green on an empty suite.
2. **M1 — Ingest:** Module 1 + data quality report + tests on the cleaning/transform functions.
3. **M2 — Embed + vector store:** Module 2 with caching and a working `semantic_search`.
4. **M3 — Themes:** Module 3 with theme summaries and prevalence-over-time table.
5. **M4 — LLM extraction:** Module 4 with `--dry-run` mock, schema validation, prompt variants, and the evaluation harness on the gold set.
6. **M5 — Predictive model:** Module 5 with time-aware split and evaluation.
7. **M6 — Statistics:** Module 6 — sampling, CIs, significance tests with correction, trend detection. Heavily unit-tested.
8. **M7 — Report:** Module 7 auto-generates the executive report with charts.
9. **M8 — Polish:** README, end-to-end `make run`, docstrings, lint clean.

---

## 9. Testing & Validation

- **Unit tests (required) for:** ingest transforms (dedup, date parsing, filtering), LLM response parsing (including malformed inputs), and **all statistical functions** (CI bounds, z-test, multiple-comparison correction, sample-size calc) — verify these against known textbook values.
- **Smoke test:** `make run` on the committed sample completes end-to-end and produces a report.
- **No network at test time:** tests must pass offline (use the LLM `--dry-run` mock and the committed sample).

---

## 10. README Requirements

The README is part of the grade. It must include:
- One-paragraph problem statement framed as a business question.
- A diagram or description of the pipeline.
- Quickstart: setup → get data → run → where the report lands.
- A **results section with actual numbers** from a real run (top themes, a significant mover with its p-value, predictive AUC).
- Explicit notes on the design choices that signal seniority: why time-aware validation, why CIs on proportions, why the LLM runs on a sample, what the multiple-comparison correction guards against.
- Honest limitations section.

---

## 11. Acceptance Criteria (Definition of Done)

- [ ] `make setup && make sample && make run` produces `reports/voc_insight_report.md` end-to-end on the committed sample with no network access.
- [ ] Embeddings are cached; `semantic_search` returns sensible neighbors.
- [ ] Theme extraction yields labeled themes with counts, representative examples, and a prevalence-over-time table.
- [ ] LLM stage returns schema-valid JSON, has ≥3 prompt variants, and reports per-field accuracy on a gold set plus latency/cost estimates; runs offline in `--dry-run`.
- [ ] LLM stage is checkpointed/resumable (skips processed `record_id`s on restart), uses bounded concurrency with backoff on 429/5xx, enforces `max_records`/`max_spend_usd`, prints a pre-run cost estimate, and writes `reports/llm_run_summary.md` with **measured** tokens and spend.
- [ ] Predictive model uses text-derived features, a time-aware split, beats its logistic baseline, and reports AUC + feature importance.
- [ ] Every proportion in the final report carries a confidence interval; every claimed change over time has a corrected significance test behind it.
- [ ] Statistical functions are unit-tested against known values.
- [ ] README contains real result numbers and the design-rationale notes.
- [ ] `ruff` passes; functions are typed and docstringed.

---

## 12. Stretch Goals (only after Acceptance Criteria are met)

- Streamlit dashboard over the outputs (theme explorer + semantic search box).
- Fine-tune a small classifier to replace/benchmark the LLM extraction and compare cost/accuracy.
- Lightweight FastAPI endpoint for `semantic_search` and on-demand classification.
- Drift monitoring: detect when incoming theme distribution diverges from a reference period.
