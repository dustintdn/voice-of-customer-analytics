"""M4 tests: parsing, mock client, retry/backoff, checkpointing, spend cap, eval.

All offline: uses the deterministic MockLLMClient and fakes; no network, no spend.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from voc import schema as cols
from voc.config import load_config
from voc.llm_extract import (
    LLMResponse,
    MockLLMClient,
    PermanentLLMError,
    RetryableLLMError,
    _backoff_seconds,
    build_user_prompt,
    estimate_cost,
    extract_one,
    load_done_ids,
    load_prompt,
    parse_extraction,
    response_cost,
    run_batch,
    run_evaluation,
    sample_records,
)

VOCAB = ["fees_and_charges", "debt_collection_harassment", "credit_reporting_error",
         "unauthorized_activity", "loan_servicing", "account_access", "other"]


# --- parsing --------------------------------------------------------------- #
def _valid_json(category="fees_and_charges") -> str:
    return json.dumps(
        {"sentiment_label": "negative", "sentiment_score": -0.6, "issue_category": category,
         "severity": "medium", "is_actionable": True}
    )


def test_parse_valid_json() -> None:
    r = parse_extraction(_valid_json(), VOCAB)
    assert r.sentiment_label == "negative"
    assert r.issue_category == "fees_and_charges"


def test_parse_strips_code_fences() -> None:
    raw = f"```json\n{_valid_json()}\n```"
    assert parse_extraction(raw, VOCAB).severity == "medium"


def test_parse_extracts_json_amid_prose() -> None:
    raw = f"Sure! Here is the result:\n{_valid_json()}\nHope that helps."
    assert parse_extraction(raw, VOCAB).is_actionable is True


def test_parse_coerces_unknown_category_to_other() -> None:
    r = parse_extraction(_valid_json(category="totally_made_up"), VOCAB)
    assert r.issue_category == "other"


def test_parse_clamps_sentiment_score() -> None:
    raw = json.dumps({"sentiment_label": "negative", "sentiment_score": -5.0,
                      "issue_category": "other", "severity": "low", "is_actionable": False})
    assert parse_extraction(raw, VOCAB).sentiment_score == -1.0


def test_parse_rejects_invalid_label() -> None:
    raw = json.dumps({"sentiment_label": "furious", "sentiment_score": 0.0,
                      "issue_category": "other", "severity": "low", "is_actionable": True})
    with pytest.raises(ValueError):
        parse_extraction(raw, VOCAB)


def test_parse_rejects_non_json() -> None:
    with pytest.raises(ValueError):
        parse_extraction("the model refused to answer", VOCAB)


# --- prompts --------------------------------------------------------------- #
def test_prompts_load_and_fill() -> None:
    for v in ("v1", "v2", "v3"):
        system, template = load_prompt(v)
        assert system
        user = build_user_prompt(template, VOCAB, "my narrative text")
        assert "my narrative text" in user
        assert "fees_and_charges" in user
        assert "{{NARRATIVE}}" not in user and "{{CATEGORIES}}" not in user


# --- mock client ----------------------------------------------------------- #
def test_mock_client_returns_parseable_json() -> None:
    client = MockLLMClient(VOCAB)
    resp = client.complete("sys", "RECORD:\nThe bank charged me overdraft fees")
    r = parse_extraction(resp.text, VOCAB)
    assert r.issue_category == "fees_and_charges"
    assert resp.input_tokens > 0


def test_mock_variant_degradation_orders_v1_worse_than_v3() -> None:
    client = MockLLMClient(VOCAB)
    texts = [f"RECORD:\nThe bank charged me overdraft fees number {i}" for i in range(50)]
    def n_correct(variant: str) -> int:
        return sum(
            parse_extraction(client.complete("s", t, variant=variant).text, VOCAB).issue_category
            == "fees_and_charges"
            for t in texts
        )
    assert n_correct("v3") > n_correct("v1")


# --- cost + backoff -------------------------------------------------------- #
def test_response_cost_and_estimate() -> None:
    config = load_config("config/config.yaml")
    resp = LLMResponse(text="x", input_tokens=1000, output_tokens=1000)
    expected = config.llm.input_cost_per_1k + config.llm.output_cost_per_1k
    assert abs(response_cost(resp, config) - expected) < 1e-9
    assert estimate_cost(100, 500, config) > 0


def test_backoff_is_bounded_and_increases() -> None:
    base, cap = 1.0, 30.0
    assert _backoff_seconds(1, base, cap) <= base + base
    assert _backoff_seconds(10, base, cap) <= cap + base


# --- extract_one: retry / permanent / malformed ---------------------------- #
class _FlakyClient:
    def __init__(self, fail_times: int, exc: type[Exception]) -> None:
        self.calls = 0
        self.fail_times = fail_times
        self.exc = exc

    def complete(self, system, user, *, variant=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc("transient")
        return LLMResponse(_valid_json(), 10, 5)


class _MalformedThenValid:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system, user, *, variant=None):
        self.calls += 1
        return LLMResponse("not json" if self.calls == 1 else _valid_json(), 10, 5)


def _cfg():
    c = load_config("config/config.yaml")
    c.llm.controlled_vocab = VOCAB
    return c


def test_extract_one_retries_then_succeeds() -> None:
    client = _FlakyClient(fail_times=2, exc=RetryableLLMError)
    row = extract_one(client, "sys", "{{NARRATIVE}}", 1, "text", _cfg(), "v3", sleep=lambda _: None)
    assert row["ok"] is True
    assert row["attempts"] == 3


def test_extract_one_permanent_fails_fast() -> None:
    client = _FlakyClient(fail_times=1, exc=PermanentLLMError)
    row = extract_one(client, "sys", "{{NARRATIVE}}", 1, "text", _cfg(), "v3", sleep=lambda _: None)
    assert row["ok"] is False
    assert row["attempts"] == 1
    assert "permanent" in row["error"]


def test_extract_one_retries_malformed_then_parses() -> None:
    client = _MalformedThenValid()
    row = extract_one(client, "sys", "{{NARRATIVE}}", 1, "text", _cfg(), "v3", sleep=lambda _: None)
    assert row["ok"] is True
    assert row["attempts"] == 2


# --- sampling -------------------------------------------------------------- #
def test_sample_records_deterministic_and_sized() -> None:
    df = pd.DataFrame({cols.RECORD_ID: range(100), cols.TEXT: ["t"] * 100})
    a = sample_records(df, 10, seed=42)
    b = sample_records(df, 10, seed=42)
    assert len(a) == 10
    assert list(a[cols.RECORD_ID]) == list(b[cols.RECORD_ID])
    assert len(sample_records(df, 999, seed=42)) == 100  # n >= len returns all


# --- checkpoint + spend cap ------------------------------------------------ #
def _todo(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {cols.RECORD_ID: list(range(n)),
         cols.TEXT: [f"the bank charged me overdraft fees record {i}" for i in range(n)]}
    )


def test_checkpoint_and_resume(tmp_path) -> None:
    config = _cfg()
    config.llm.max_concurrency = 2
    ckpt = tmp_path / "ck.jsonl"
    client = MockLLMClient(VOCAB)
    system, template = load_prompt("v3")

    run_batch(client, _todo(8), config, ckpt, system, template, "v3", enforce_spend_cap=False)
    done = load_done_ids(ckpt)
    assert len(done) == 8
    assert done == {str(i) for i in range(8)}


def test_spend_cap_halts_paid_run(tmp_path) -> None:
    config = _cfg()
    config.llm.max_concurrency = 1
    config.llm.max_spend_usd = 0.001  # tiny: cap trips almost immediately
    ckpt = tmp_path / "ck.jsonl"
    client = MockLLMClient(VOCAB)
    system, template = load_prompt("v3")

    run_batch(client, _todo(50), config, ckpt, system, template, "v3", enforce_spend_cap=True)
    rows = [json.loads(line) for line in ckpt.read_text().splitlines()]
    n_ok = sum(r.get("ok") for r in rows)
    assert 1 <= n_ok < 50  # some processed, then halted by the cap


# --- evaluation ------------------------------------------------------------ #
def test_run_evaluation_ranks_v3_over_v1(tmp_path) -> None:
    config = _cfg()
    config.paths.reports_dir = tmp_path
    rows = []
    keywords = {
        "fees_and_charges": "the bank charged me overdraft fees",
        "debt_collection_harassment": "a debt collector keeps calling about a collection",
        "credit_reporting_error": "an inaccurate item on my credit report from the bureau",
        "loan_servicing": "my mortgage servicer mishandled my escrow payment",
    }
    for i in range(24):
        cat = list(keywords)[i % len(keywords)]
        rows.append(
            {"record_id": i, "text": f"{keywords[cat]} number {i}",
             "sentiment_label": "negative", "sentiment_score": -0.6,
             "issue_category": cat, "severity": "medium", "is_actionable": True}
        )
    gold = tmp_path / "gold.csv"
    pd.DataFrame(rows).to_csv(gold, index=False)

    best = run_evaluation(config, dry_run=True, gold_path=gold)
    assert best["variant"] == "v3"
    assert (tmp_path / "llm_eval.md").exists()
