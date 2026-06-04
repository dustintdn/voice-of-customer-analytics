"""Module 4 — LLM Structured Extraction.

Converts free-text complaints into schema-validated structured fields
(``sentiment``, ``issue_category``, ``severity``, ``is_actionable``) via an LLM
API, with:

  * robust parsing (fence-stripping, JSON extraction, pydantic validation,
    controlled-vocabulary coercion) that never lets one bad response crash a batch;
  * three prompt variants (zero-shot / few-shot / few-shot+rubric) and an
    evaluation harness against a hand-labeled gold set;
  * a production-run path built to spend money safely (SPEC §5.4.1):
    checkpoint/resume, bounded concurrency, retry-with-backoff, hard
    ``max_records``/``max_spend_usd`` caps, a pre-run cost estimate, and
    measured token/cost accounting.

Offline-testable via ``--dry-run``, which swaps in a deterministic mock LLM
(no network, no spend). No paid call is made until a real run is requested.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Protocol

import pandas as pd
from pydantic import BaseModel, field_validator

from voc import schema as cols
from voc.config import Config, repo_root

PROMPT_VARIANTS = ("v1", "v2", "v3")
_EST_OUTPUT_TOKENS = 80  # rough output size for the pre-run estimate; real spend is measured
SENTIMENT_LABELS = ("negative", "neutral", "positive")
SEVERITIES = ("low", "medium", "high")


# --------------------------------------------------------------------------- #
# Schema + parsing                                                            #
# --------------------------------------------------------------------------- #
class ExtractionResult(BaseModel):
    """Validated structured fields extracted from one complaint narrative."""

    sentiment_label: str
    sentiment_score: float
    issue_category: str
    severity: str
    is_actionable: bool

    @field_validator("sentiment_label")
    @classmethod
    def _check_sentiment(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in SENTIMENT_LABELS:
            raise ValueError(f"sentiment_label must be one of {SENTIMENT_LABELS}, got {v!r}")
        return v

    @field_validator("severity")
    @classmethod
    def _check_severity(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {v!r}")
        return v

    @field_validator("sentiment_score")
    @classmethod
    def _clamp_score(cls, v: float) -> float:
        return max(-1.0, min(1.0, float(v)))


_FENCE_RE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def parse_extraction(raw: str, vocab: list[str]) -> ExtractionResult:
    """Parse and validate an LLM response into an :class:`ExtractionResult`.

    Strips code fences, extracts the first JSON object, validates against the
    schema, and coerces an out-of-vocabulary ``issue_category`` to ``"other"``
    (when available). Raises ``ValueError`` on unrecoverable malformed input.
    """
    cleaned = _strip_fences(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = _JSON_RE.search(cleaned)
        if not match:
            raise ValueError(f"No JSON object found in response: {raw[:120]!r}") from None
        obj = json.loads(match.group(0))

    result = ExtractionResult(**obj)
    if result.issue_category not in vocab:
        result.issue_category = "other" if "other" in vocab else result.issue_category
    return result


# --------------------------------------------------------------------------- #
# Prompts                                                                     #
# --------------------------------------------------------------------------- #
def _prompts_dir() -> Path:
    return repo_root() / "prompts"


def load_prompt(version: str) -> tuple[str, str]:
    """Return ``(system_prompt, user_template)`` for a prompt version."""
    system = (_prompts_dir() / "system.txt").read_text(encoding="utf-8").strip()
    template = (_prompts_dir() / f"{version}.txt").read_text(encoding="utf-8")
    return system, template


def build_user_prompt(template: str, vocab: list[str], narrative: str) -> str:
    """Fill a user-prompt template with the controlled vocabulary and narrative."""
    categories = ", ".join(f'"{c}"' for c in vocab)
    return template.replace("{{CATEGORIES}}", categories).replace("{{NARRATIVE}}", narrative)


# --------------------------------------------------------------------------- #
# LLM clients                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class LLMResponse:
    """A single LLM completion plus measured token usage."""

    text: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    """Protocol shared by the mock and real LLM clients."""

    def complete(self, system: str, user: str, *, variant: str | None = None) -> LLMResponse:
        """Return a completion for the given system/user prompt."""
        ...


class RetryableLLMError(Exception):
    """Transient error (rate limit / 5xx) — safe to retry with backoff."""


class PermanentLLMError(Exception):
    """Non-retryable error (e.g. malformed request / auth) — fail fast."""


_CATEGORY_KEYWORDS = [
    ("unauthorized_activity", ("unauthorized", "fraud", "fraudulent", "identity", "stolen")),
    ("credit_reporting_error", ("credit report", "bureau", "inquiry", "inaccurate", "score")),
    ("debt_collection_harassment", ("collector", "collection", "garnish", "harass", "validation")),
    ("loan_servicing", ("mortgage", "escrow", "servicer", "modification", "forbearance", "loan")),
    ("account_access", ("frozen", "access", "closed my account", "locked", "withdrawn")),
    ("fees_and_charges", ("fee", "overdraft", "charged", "charge", "interest", "apr")),
    ("billing_dispute", ("dispute", "billing", "statement")),
]
_ESCALATION_WORDS = ("legal action", "attorney", "third complaint", "lawsuit", "foreclosure", "garnish")


class MockLLMClient:
    """Deterministic offline LLM mock (no network, no spend).

    Classifies the narrative with simple keyword heuristics so the full pipeline
    — parsing, cost accounting, checkpointing — is exercised offline. Accepts a
    ``variant`` hint and degrades accuracy for weaker prompts (v1 > v2 > v3 noise)
    so the evaluation harness shows meaningful variant differences in dry-run.
    """

    _DEGRADE = {"v1": 0.40, "v2": 0.20, "v3": 0.0}

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab

    def _classify(self, narrative: str, variant: str | None) -> dict:
        text = narrative.lower()
        escalated = any(w in text for w in _ESCALATION_WORDS)

        category = "other"
        for cat, kws in _CATEGORY_KEYWORDS:
            if cat in self.vocab and any(k in text for k in kws):
                category = cat
                break

        # Deterministic per-(record, variant) degradation simulates prompt quality.
        if variant in self._DEGRADE:
            seed = int(hashlib.blake2b((narrative + variant).encode(), digest_size=4).hexdigest(), 16)
            if (seed % 100) < self._DEGRADE[variant] * 100:
                category = "other"

        return {
            "sentiment_label": "negative",
            "sentiment_score": -0.9 if escalated else -0.6,
            "issue_category": category if category in self.vocab else "other",
            "severity": "high" if escalated else "medium",
            "is_actionable": True,
        }

    def complete(self, system: str, user: str, *, variant: str | None = None) -> LLMResponse:
        """Return a deterministic mock completion for the record in ``user``."""
        narrative = user.rsplit("RECORD:", 1)[-1].strip()
        payload = json.dumps(self._classify(narrative, variant))
        # Approximate token usage by characters/4 (mock has no real tokenizer).
        return LLMResponse(text=payload, input_tokens=len(user) // 4, output_tokens=len(payload) // 4)


class AnthropicLLMClient:
    """Real backend using the Anthropic SDK (lazy import; key from env var)."""

    def __init__(self, model: str, api_key_env: str, max_tokens: int, temperature: float) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        self._api_key_env = api_key_env

    def _ensure_client(self):
        if self._client is None:
            import os

            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - optional dep
                raise PermanentLLMError(
                    "anthropic is not installed; run `make setup` or use --dry-run."
                ) from exc
            key = os.environ.get(self._api_key_env)
            if not key:
                raise PermanentLLMError(
                    f"{self._api_key_env} is not set. Export your API key or use --dry-run."
                )
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def complete(self, system: str, user: str, *, variant: str | None = None) -> LLMResponse:
        """Call the Anthropic API; map transient errors to RetryableLLMError."""
        client = self._ensure_client()
        import anthropic

        try:
            msg = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APITimeoutError) as exc:
            raise RetryableLLMError(str(exc)) from exc
        except anthropic.APIStatusError as exc:  # 4xx other than rate limit -> permanent
            if exc.status_code >= 500:
                raise RetryableLLMError(str(exc)) from exc
            raise PermanentLLMError(str(exc)) from exc

        text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
        return LLMResponse(text=text, input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)


def get_llm_client(config: Config, dry_run: bool) -> LLMClient:
    """Construct the mock (dry-run) or real Anthropic client."""
    if dry_run:
        return MockLLMClient(config.llm.controlled_vocab)
    return AnthropicLLMClient(
        model=config.llm.model,
        api_key_env=config.llm.api_key_env,
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
    )


# --------------------------------------------------------------------------- #
# Cost + retry helpers                                                        #
# --------------------------------------------------------------------------- #
def response_cost(resp: LLMResponse, config: Config) -> float:
    """USD cost of a single response from measured token counts."""
    return (
        resp.input_tokens / 1000 * config.llm.input_cost_per_1k
        + resp.output_tokens / 1000 * config.llm.output_cost_per_1k
    )


def estimate_cost(n_records: int, avg_input_tokens: float, config: Config) -> float:
    """Rough pre-run cost estimate (USD) for ``n_records``."""
    per_record = (
        avg_input_tokens / 1000 * config.llm.input_cost_per_1k
        + _EST_OUTPUT_TOKENS / 1000 * config.llm.output_cost_per_1k
    )
    return n_records * per_record


def _backoff_seconds(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with full jitter."""
    return min(cap, base * (2 ** (attempt - 1))) + random.uniform(0, base)


# --------------------------------------------------------------------------- #
# Per-record extraction                                                       #
# --------------------------------------------------------------------------- #
def extract_one(
    client: LLMClient,
    system: str,
    template: str,
    record_id: object,
    narrative: str,
    config: Config,
    variant: str,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Extract one record with retry/backoff; never raises (errors are recorded).

    Returns a checkpoint row dict with the fields (on success) or an ``error``
    (on failure), plus measured tokens, latency, and attempt count.
    """
    vocab = config.llm.controlled_vocab
    user = build_user_prompt(template, vocab, narrative)
    last_err: str | None = None

    for attempt in range(1, config.llm.max_attempts + 1):
        try:
            t0 = time.perf_counter()
            resp = client.complete(system, user, variant=variant)
            latency = time.perf_counter() - t0
        except RetryableLLMError as exc:
            last_err = f"retryable: {exc}"
            sleep(_backoff_seconds(attempt, config.llm.backoff_base_seconds, config.llm.backoff_max_seconds))
            continue
        except PermanentLLMError as exc:
            return {"record_id": record_id, "ok": False, "error": f"permanent: {exc}", "attempts": attempt}

        try:
            result = parse_extraction(resp.text, vocab)
        except (ValueError, json.JSONDecodeError) as exc:
            last_err = f"parse: {exc}"
            sleep(_backoff_seconds(attempt, config.llm.backoff_base_seconds, config.llm.backoff_max_seconds))
            continue

        return {
            "record_id": record_id,
            "ok": True,
            **result.model_dump(),
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": response_cost(resp, config),
            "latency_s": latency,
            "attempts": attempt,
            "variant": variant,
        }

    return {"record_id": record_id, "ok": False, "error": last_err or "unknown", "attempts": config.llm.max_attempts}


# --------------------------------------------------------------------------- #
# Checkpointing                                                               #
# --------------------------------------------------------------------------- #
def load_done_ids(checkpoint_path: Path) -> set[str]:
    """Return the set of already-processed ``record_id``s (as strings)."""
    if not checkpoint_path.exists():
        return set()
    done: set[str] = set()
    with checkpoint_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                done.add(str(json.loads(line)["record_id"]))
    return done


class _CheckpointWriter:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, row: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps(row, default=str) + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------- #
# Sampling                                                                    #
# --------------------------------------------------------------------------- #
def sample_records(records: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw a reproducible random sample of ``n`` records (all if n >= len)."""
    if n >= len(records):
        return records.reset_index(drop=True)
    return records.sample(n=n, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Production batch run                                                        #
# --------------------------------------------------------------------------- #
def _confirm_run(n: int, est: float, dry_run: bool, yes: bool) -> bool:
    """Print a pre-run cost estimate and require confirmation unless safe/skipped."""
    print(f"[extract] About to process {n:,} records (est. ${est:.2f}; real spend is measured).")
    if dry_run:
        print("[extract] dry-run: mock LLM, no network, no spend.")
        return True
    if yes:
        return True
    reply = input("[extract] Continue with a PAID run? [y/N] ").strip().lower()
    return reply in {"y", "yes"}


def run_batch(
    client: LLMClient,
    todo: pd.DataFrame,
    config: Config,
    checkpoint_path: Path,
    system: str,
    template: str,
    variant: str,
    enforce_spend_cap: bool = True,
) -> float:
    """Run extraction over ``todo`` with bounded concurrency and a spend cap.

    Returns the total measured spend (USD). When ``enforce_spend_cap`` is set,
    stops scheduling new work once ``max_spend_usd`` is reached. The cap guards
    paid runs; dry-runs (no real spend) disable it so the offline pipeline
    processes every record.

    Submission is bounded to ``max_concurrency`` in-flight calls, so the cap can
    actually halt the run (at most ``max_concurrency`` already-started calls
    finish after it trips) and we never queue the entire dataset at once.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    writer = _CheckpointWriter(checkpoint_path)
    spend = 0.0
    capped = False
    rows_iter = ((row[cols.RECORD_ID], row[cols.TEXT]) for _, row in todo.iterrows())

    try:
        with ThreadPoolExecutor(max_workers=config.llm.max_concurrency) as pool:
            pending: set = set()

            def submit_next() -> bool:
                try:
                    rid, narrative = next(rows_iter)
                except StopIteration:
                    return False
                pending.add(
                    pool.submit(extract_one, client, system, template, rid, narrative, config, variant)
                )
                return True

            for _ in range(config.llm.max_concurrency):
                if not submit_next():
                    break

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    row = fut.result()
                    writer.write(row)
                    spend += float(row.get("cost_usd", 0.0))
                    if enforce_spend_cap and not capped and spend >= config.llm.max_spend_usd:
                        capped = True
                        print(f"[extract] Spend cap ${config.llm.max_spend_usd:.2f} reached — stopping.")
                if not capped:
                    while len(pending) < config.llm.max_concurrency and submit_next():
                        pass
    finally:
        writer.close()
    return spend


def write_run_summary(checkpoint_path: Path, config: Config, out_path: Path, dry_run: bool) -> None:
    """Summarize the checkpoint into ``reports/llm_run_summary.md`` (measured)."""
    rows = [json.loads(line) for line in checkpoint_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [r for r in rows if r.get("ok")]
    errors = [r for r in rows if not r.get("ok")]
    in_tok = sum(r.get("input_tokens", 0) for r in ok)
    out_tok = sum(r.get("output_tokens", 0) for r in ok)
    spend = sum(r.get("cost_usd", 0.0) for r in ok)
    lats = [r["latency_s"] for r in ok if "latency_s" in r]
    retries = sum(max(0, r.get("attempts", 1) - 1) for r in rows)
    per_1k = (spend / len(ok) * 1000) if ok else 0.0

    lines = [
        "# LLM Run Summary",
        "",
        f"- **Mode:** {'dry-run (mock LLM)' if dry_run else 'live API'}",
        f"- **Model:** {config.llm.model}",
        f"- **Prompt version:** {config.llm.prompt_version}",
        f"- **Records processed:** {len(ok):,} ok, {len(errors):,} errors",
        f"- **Input tokens:** {in_tok:,}  |  **Output tokens:** {out_tok:,}",
        f"- **Total spend:** ${spend:.4f}  |  **Cost per 1k records:** ${per_1k:.4f}",
        f"- **Latency (s):** mean {mean(lats):.4f}, median {median(lats):.4f}" if lats else "- **Latency:** n/a",
        f"- **Retries:** {retries:,}",
        "",
    ]
    if dry_run:
        lines.append("> Dry-run numbers come from the mock LLM; real token/cost/latency figures "
                     "require a live run.\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_extract(config: Config, dry_run: bool = False, yes: bool = False) -> None:
    """Run LLM structured extraction over the sampled records (Module 4).

    Args:
        config: Loaded pipeline configuration.
        dry_run: If True, use the deterministic mock LLM (offline, no spend).
        yes: Skip the paid-run confirmation prompt.
    """
    records_path = config.paths.records_parquet
    if not records_path.exists():
        raise FileNotFoundError(f"Cleaned records not found: {records_path}. Run ingest first.")
    records = pd.read_parquet(records_path)

    sample = sample_records(records, config.llm.max_records, config.project.seed)
    system, template = load_prompt(config.llm.prompt_version)

    # Skip already-processed records (resume).
    checkpoint = config.paths.llm_extractions
    done = load_done_ids(checkpoint)
    todo = sample[~sample[cols.RECORD_ID].astype(str).isin(done)].reset_index(drop=True)
    if done:
        print(f"[extract] Resuming: {len(done):,} already done, {len(todo):,} remaining.")

    # Pre-run cost estimate + confirmation.
    avg_input_tokens = (
        mean(len(build_user_prompt(template, config.llm.controlled_vocab, t)) / 4 for t in sample[cols.TEXT].head(50))
        if len(sample) else 0.0
    )
    est = estimate_cost(len(todo), avg_input_tokens, config)
    if not _confirm_run(len(todo), est, dry_run, yes):
        print("[extract] Aborted by user.")
        return

    client = get_llm_client(config, dry_run)
    spend = run_batch(
        client, todo, config, checkpoint, system, template, config.llm.prompt_version,
        enforce_spend_cap=not dry_run,
    )
    print(f"[extract] Measured spend this run: ${spend:.4f}")

    write_run_summary(checkpoint, config, config.paths.reports_dir / "llm_run_summary.md", dry_run)
    print(f"[extract] Wrote run summary to {config.paths.reports_dir / 'llm_run_summary.md'}")


# --------------------------------------------------------------------------- #
# Evaluation harness                                                          #
# --------------------------------------------------------------------------- #
_CATEGORICAL_FIELDS = ("sentiment_label", "issue_category", "severity", "is_actionable")


def evaluate_variant(
    client: LLMClient, gold: pd.DataFrame, version: str, config: Config
) -> dict:
    """Run one prompt variant over the gold set; return per-field metrics."""
    system, template = load_prompt(version)
    vocab = config.llm.controlled_vocab
    preds, latencies, in_tok, out_tok = [], [], 0, 0

    for _, row in gold.iterrows():
        user = build_user_prompt(template, vocab, row["text"])
        t0 = time.perf_counter()
        resp = client.complete(system, user, variant=version)
        latencies.append(time.perf_counter() - t0)
        in_tok += resp.input_tokens
        out_tok += resp.output_tokens
        preds.append(parse_extraction(resp.text, vocab))

    metrics: dict = {"variant": version}
    for field in _CATEGORICAL_FIELDS:
        correct = sum(str(getattr(p, field)) == str(g) for p, g in zip(preds, gold[field], strict=False))
        metrics[f"acc_{field}"] = correct / len(gold) if len(gold) else 0.0
    metrics["mae_sentiment_score"] = (
        mean(abs(p.sentiment_score - float(g)) for p, g in zip(preds, gold["sentiment_score"], strict=False))
        if len(gold) else 0.0
    )
    metrics["mean_latency_s"] = mean(latencies) if latencies else 0.0
    per_record_cost = (
        (in_tok / 1000 * config.llm.input_cost_per_1k + out_tok / 1000 * config.llm.output_cost_per_1k)
        / len(gold)
        if len(gold) else 0.0
    )
    metrics["cost_per_1k"] = per_record_cost * 1000
    return metrics


def run_evaluation(config: Config, dry_run: bool = False, gold_path: Path | None = None) -> dict:
    """Evaluate all prompt variants on the gold set; write ``reports/llm_eval.md``.

    Returns the winning variant's metrics. The winner maximizes issue_category
    accuracy (the hardest field), tie-broken by mean categorical accuracy.
    """
    gold_path = gold_path or (config.paths.sample_csv.parent / "gold.csv")
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold set not found: {gold_path}. Create it with scripts/make_gold_stub.py."
        )
    gold = pd.read_csv(gold_path)
    client = get_llm_client(config, dry_run)

    results = [evaluate_variant(client, gold, v, config) for v in PROMPT_VARIANTS]

    def score(m: dict) -> tuple[float, float]:
        mean_acc = mean(m[f"acc_{f}"] for f in _CATEGORICAL_FIELDS)
        return (m["acc_issue_category"], mean_acc)

    best = max(results, key=score)

    lines = [
        "# LLM Prompt-Variant Evaluation",
        "",
        f"- **Gold set:** {len(gold):,} records (`{gold_path.name}`)",
        f"- **Mode:** {'dry-run (mock LLM)' if dry_run else 'live API'}",
        f"- **Winner:** **{best['variant']}** (highest issue_category accuracy)",
        "",
        "| Variant | sentiment | issue_category | severity | actionable | score MAE | latency (s) | $/1k |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for m in results:
        lines.append(
            f"| {m['variant']} | {m['acc_sentiment_label']:.2f} | {m['acc_issue_category']:.2f} | "
            f"{m['acc_severity']:.2f} | {m['acc_is_actionable']:.2f} | {m['mae_sentiment_score']:.3f} | "
            f"{m['mean_latency_s']:.4f} | ${m['cost_per_1k']:.4f} |"
        )
    if dry_run:
        lines += ["", "> Dry-run: predictions are from the mock LLM; accuracy/cost figures are "
                  "illustrative. Real numbers require a live run against the gold set."]
    out_path = config.paths.reports_dir / "llm_eval.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[evaluate] Winner: {best['variant']} | wrote {out_path}")
    return best
