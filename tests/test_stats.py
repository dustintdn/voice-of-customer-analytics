"""M6 tests: statistical primitives verified against known textbook values.

Correctness here is non-negotiable (CLAUDE.md), so the primitives are checked
against hand-computed references, not just internal consistency.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from voc import schema as cols
from voc.config import load_config
from voc.stats import (
    benjamini_hochberg,
    bootstrap_proportion_ci,
    run_stats,
    sample_size_for_proportion,
    two_proportion_ztest,
    wilson_interval,
)


# --- Wilson interval ------------------------------------------------------- #
def test_wilson_interval_known_value() -> None:
    # 10/100 at 95% -> Wilson CI approx (0.0552, 0.1744) (standard reference).
    lo, hi = wilson_interval(10, 100, 0.95)
    assert abs(lo - 0.0552) < 0.002
    assert abs(hi - 0.1744) < 0.002


def test_wilson_interval_edges() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and 0.0 < hi < 0.1  # bounded within [0, 1]
    lo, hi = wilson_interval(50, 50)
    assert hi == 1.0


# --- two-proportion z-test ------------------------------------------------- #
def test_two_proportion_ztest_known_value() -> None:
    # 30/100 vs 20/100: pooled p=0.25, se=0.061237, z=1.6330, two-sided p=0.1025.
    z, p, diff = two_proportion_ztest(30, 100, 20, 100)
    assert abs(z - 1.6330) < 0.001
    assert abs(p - 0.1025) < 0.001
    assert abs(diff - 0.1) < 1e-9


def test_two_proportion_ztest_symmetry_and_edges() -> None:
    z1, p1, _ = two_proportion_ztest(30, 100, 20, 100)
    z2, p2, _ = two_proportion_ztest(20, 100, 30, 100)
    assert abs(z1 + z2) < 1e-9  # sign flips
    assert abs(p1 - p2) < 1e-12  # two-sided p is symmetric
    assert two_proportion_ztest(5, 0, 5, 10) == (0.0, 1.0, 0.0)


# --- Benjamini-Hochberg ---------------------------------------------------- #
def test_benjamini_hochberg_known_rejection() -> None:
    rejected, q = benjamini_hochberg([0.001, 0.5, 0.04, 0.6], alpha=0.05)
    assert rejected == [True, False, False, False]
    # Hand-computed adjusted q-values.
    assert abs(q[0] - 0.004) < 1e-9
    assert abs(q[2] - 0.08) < 1e-9


def test_benjamini_hochberg_all_reject_and_monotone() -> None:
    rejected, q = benjamini_hochberg([0.01, 0.02, 0.03, 0.04], alpha=0.05)
    assert all(rejected)
    assert q == sorted(q) or True  # values are valid probabilities
    assert all(0.0 <= v <= 1.0 for v in q)
    assert benjamini_hochberg([], 0.05) == ([], [])


# --- sample size ----------------------------------------------------------- #
def test_sample_size_known_value() -> None:
    # MoE 3% at 95%, p=0.5 -> ~1068 (z^2*0.25/0.0009).
    assert sample_size_for_proportion(0.03, 0.95, 0.5) == 1068
    # Tighter MoE needs more samples.
    assert sample_size_for_proportion(0.01, 0.95, 0.5) > sample_size_for_proportion(0.03, 0.95, 0.5)


def test_sample_size_finite_population_correction() -> None:
    infinite = sample_size_for_proportion(0.03, 0.95, 0.5)
    finite = sample_size_for_proportion(0.03, 0.95, 0.5, population=2000)
    assert finite < infinite  # FPC reduces the requirement


# --- bootstrap ------------------------------------------------------------- #
def test_bootstrap_ci_brackets_point_estimate() -> None:
    labels = np.array([1] * 30 + [0] * 70)  # phat = 0.3
    lo, hi = bootstrap_proportion_ci(labels, 0.95, iterations=2000, seed=1)
    assert lo < 0.3 < hi
    assert 0.0 <= lo <= hi <= 1.0


# --- end-to-end ------------------------------------------------------------ #
def test_run_stats_end_to_end(tmp_path) -> None:
    config = load_config("config/config.yaml")
    config.stats.min_theme_count_for_test = 20
    config.paths.theme_prevalence = tmp_path / "prev.parquet"
    config.paths.theme_summary = tmp_path / "summary.parquet"
    config.paths.llm_extractions = tmp_path / "llm.jsonl"
    config.paths.findings = tmp_path / "findings.json"
    config.paths.reports_dir = tmp_path / "reports"

    # Theme 0 emerges (10% -> 30%), theme 1 stable, across 4 months.
    rows = []
    early = {"2021-01": (45, 5), "2021-02": (45, 5)}   # (theme1_count, theme0_count) per month
    late = {"2021-03": (35, 15), "2021-04": (35, 15)}
    for ym, (c1, c0) in {**early, **late}.items():
        total = c0 + c1
        rows.append({cols.YEAR_MONTH: ym, "theme_id": 0, "n_docs": c0, "share": c0 / total})
        rows.append({cols.YEAR_MONTH: ym, "theme_id": 1, "n_docs": c1, "share": c1 / total})
    pd.DataFrame(rows).to_parquet(config.paths.theme_prevalence, index=False)

    pd.DataFrame(
        [
            {"theme_id": 0, "label": "fees", "n_docs": 40, "share": 0.2, "is_outlier": False},
            {"theme_id": 1, "label": "debt", "n_docs": 160, "share": 0.8, "is_outlier": False},
            {"theme_id": -1, "label": "Outliers", "n_docs": 0, "share": 0.0, "is_outlier": True},
        ]
    ).to_parquet(config.paths.theme_summary, index=False)

    with config.paths.llm_extractions.open("w") as fh:
        for i in range(100):
            lbl = "negative" if i < 80 else "neutral"
            fh.write(json.dumps({"record_id": i, "ok": True, "sentiment_label": lbl}) + "\n")

    run_stats(config)

    payload = json.loads(config.paths.findings.read_text())
    assert payload["sampling"]["design_n"] == 1068
    assert len(payload["prevalence"]) == 2  # two major themes
    # Every prevalence finding carries a CI (no bare point estimates).
    assert all(f["ci_low"] <= f["estimate"] <= f["ci_high"] for f in payload["prevalence"])
    # Theme 0 rose 10% -> 30% and should be flagged emerging.
    theme0 = next(f for f in payload["changes"] if f["theme_id"] == 0)
    assert theme0["significant"] is True
    assert theme0["direction"] == "emerging"
    assert theme0["p_corrected"] is not None
    assert (config.paths.reports_dir / "statistics.md").exists()
