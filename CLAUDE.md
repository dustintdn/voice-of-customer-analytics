# CLAUDE.md

Standing instructions for working in this repository. The full build spec is in `SPEC.md` — follow it. This file is *how to behave*; `SPEC.md` is *what to build*.

## Build discipline
- Build milestone by milestone per `SPEC.md` §8 (M0–M8). Do not jump ahead or implement future milestones early.
- Each milestone must leave the repo runnable and `make test` green before moving on.
- After each milestone, stop and show what changed; wait for review before continuing.
- Treat `SPEC.md` §11 (Acceptance Criteria) as the definition of done.

## LLM stage safety (Module 4)
- Never make a paid LLM API call until M4 is built and tested in `--dry-run`. The mock path must work end-to-end first.
- The first real LLM run uses `max_spend_usd` ≈ 2 to validate cost estimation, checkpointing, and the run summary before scaling.
- The LLM stage must be resumable: skip already-processed `record_id`s on restart. Never re-pay for completed calls.
- API keys come from environment variables only. Never hardcode or commit secrets.

## Two separate models — do not conflate
- The model used to *call* Module 4's extraction is separate from, and smaller than, a top-tier model. Default to a Haiku/Sonnet-class model for high-volume structured extraction.
- The gold-set evaluation decides whether the cheaper model is accurate enough — do not assume; measure.

## Code conventions
- Python 3.11+. Dependencies pinned in `pyproject.toml`.
- All functions are typed and have docstrings.
- Run `ruff` and ensure it passes before considering any task done.
- Config-driven: no hardcoded paths, model names, or magic numbers — everything lives in `config/config.yaml` loaded via a typed config object.
- Statistical functions (CIs, significance tests, multiple-comparison correction, sample-size calc) must be unit-tested against known values. Correctness here is non-negotiable.

## Testing
- Tests must pass offline: use the committed sample data and the LLM `--dry-run` mock. No network at test time.
- `make run` on the committed sample must complete end-to-end and produce the report.

## Git
- The user handles all commits manually. Do NOT run `git commit` unless explicitly told to commit a specific change.
- At each green milestone checkpoint, leave the working tree clean and green (tests + ruff pass), report what changed, and stop — let the user commit.
- Keep `data/raw/`, `data/processed/`, and live `reports/` outputs gitignored; commit only the small sample and the spec/docs.
