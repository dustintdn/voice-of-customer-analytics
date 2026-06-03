"""Module 6 — Statistical Reporting Layer (the differentiator).

Turns the upstream counts into statistically-defensible findings:

  * **Sampling design** — required sample size for a target margin of error.
  * **Confidence intervals on every proportion** — Wilson score interval
    (default) or bootstrap. No bare point estimates.
  * **Significance testing for change over time** — two-proportion z-test per
    theme between an early and a late period, with Benjamini-Hochberg
    multiple-comparison correction (many themes are tested at once).
  * **Trend detection** — significant rising themes are "emerging", significant
    falling themes are "fading".

Every finding carries an estimate, a CI, and (for change tests) a test
statistic, a corrected p-value, and a plain-language interpretation. The
primitives are implemented directly on ``scipy.stats`` and unit-tested against
known textbook values.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from voc import schema as cols
from voc.config import Config


# --------------------------------------------------------------------------- #
# Primitives (unit-tested against textbook values)                            #
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    More accurate than the normal approximation, especially for small ``n`` or
    extreme proportions. Returns ``(lower, upper)`` clamped to ``[0, 1]``.
    """
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_proportion_ci(
    labels: np.ndarray, confidence: float = 0.95, iterations: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI for a proportion from 0/1 ``labels``."""
    labels = np.asarray(labels, dtype=float)
    if labels.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = rng.choice(labels, size=(iterations, labels.size), replace=True).mean(axis=1)
    alpha = 1 - confidence
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def proportion_ci(
    successes: int, n: int, config: Config
) -> tuple[float, float]:
    """Confidence interval for a proportion, dispatched by config method."""
    conf = config.stats.confidence_level
    if config.stats.proportion_ci_method == "bootstrap":
        labels = np.concatenate([np.ones(successes), np.zeros(max(0, n - successes))])
        return bootstrap_proportion_ci(labels, conf, config.stats.bootstrap_iterations, config.project.seed)
    return wilson_interval(successes, n, conf)


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float, float]:
    """Pooled two-proportion z-test (two-sided).

    Returns ``(z_statistic, p_value, diff)`` where ``diff = p1 - p2``.
    """
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0, 0.0)
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0, p1 - p2)
    z = (p1 - p2) / se
    p_value = 2 * stats.norm.sf(abs(z))
    return (z, p_value, p1 - p2)


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg FDR correction.

    Returns ``(rejected, qvalues)`` aligned to the input order. ``qvalues`` are
    the monotone-adjusted p-values; ``rejected[i]`` is ``qvalues[i] <= alpha``.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return ([], [])
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    q = np.clip(q, 0.0, 1.0)
    qvals = np.empty(m)
    qvals[order] = q
    rejected = qvals <= alpha
    return (rejected.tolist(), qvals.tolist())


def sample_size_for_proportion(
    margin_of_error: float, confidence: float = 0.95, p: float = 0.5, population: int | None = None
) -> int:
    """Required sample size to estimate a proportion within ``margin_of_error``.

    Uses ``n = z^2 p(1-p) / E^2`` (conservative at ``p=0.5``), with an optional
    finite-population correction. Returns the ceiling.
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    n0 = z**2 * p * (1 - p) / margin_of_error**2
    if population:
        n0 = n0 / (1 + (n0 - 1) / population)
    return math.ceil(n0)


# --------------------------------------------------------------------------- #
# Findings                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Finding:
    """A single statistically-characterized result."""

    kind: str  # "prevalence" | "change" | "sentiment"
    label: str
    estimate: float
    ci_low: float
    ci_high: float
    interpretation: str
    theme_id: int | None = None
    test_stat: float | None = None
    p_value: float | None = None
    p_corrected: float | None = None
    significant: bool | None = None
    direction: str | None = None  # "emerging" | "fading" | "stable"


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def _load_llm_labels(path) -> list[str]:
    """Sentiment labels from successful LLM extractions."""
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            if obj.get("ok"):
                labels.append(obj["sentiment_label"])
    return labels


def _prevalence_findings(summary: pd.DataFrame, total: int, config: Config) -> list[Finding]:
    """Overall theme prevalence with a CI for each major theme."""
    findings = []
    major = summary[(~summary["is_outlier"]) & (summary["n_docs"] >= config.stats.min_theme_count_for_test)]
    for _, row in major.sort_values("n_docs", ascending=False).iterrows():
        count = int(row["n_docs"])
        lo, hi = proportion_ci(count, total, config)
        est = count / total
        findings.append(
            Finding(
                kind="prevalence",
                label=str(row["label"]),
                theme_id=int(row["theme_id"]),
                estimate=est,
                ci_low=lo,
                ci_high=hi,
                interpretation=f"Theme '{row['label']}' accounts for {est:.1%} of complaints "
                f"(95% CI {lo:.1%}–{hi:.1%}).",
            )
        )
    return findings


def _change_findings(
    prev: pd.DataFrame, summary: pd.DataFrame, config: Config
) -> list[Finding]:
    """Test each major theme's prevalence change between an early and late period."""
    months = sorted(prev[cols.YEAR_MONTH].unique())
    if len(months) < 2:
        return []
    mid = len(months) // 2
    early, late = set(months[:mid]), set(months[mid:])
    early_total = int(prev[prev[cols.YEAR_MONTH].isin(early)]["n_docs"].sum())
    late_total = int(prev[prev[cols.YEAR_MONTH].isin(late)]["n_docs"].sum())
    if early_total == 0 or late_total == 0:
        return []

    labels = dict(zip(summary["theme_id"], summary["label"], strict=False))
    major_ids = summary.loc[
        (~summary["is_outlier"]) & (summary["n_docs"] >= config.stats.min_theme_count_for_test),
        "theme_id",
    ].tolist()

    tested, pvals = [], []
    for tid in major_ids:
        e = int(prev[(prev[cols.YEAR_MONTH].isin(early)) & (prev["theme_id"] == tid)]["n_docs"].sum())
        late_count = int(prev[(prev[cols.YEAR_MONTH].isin(late)) & (prev["theme_id"] == tid)]["n_docs"].sum())
        z, p, _ = two_proportion_ztest(late_count, late_total, e, early_total)
        tested.append((tid, e, late_count, z, p))
        pvals.append(p)

    rejected, qvals = benjamini_hochberg(pvals, config.stats.fdr_alpha)

    findings = []
    for (tid, e, late_count, z, p), sig, q in zip(tested, rejected, qvals, strict=False):
        early_share = e / early_total
        late_share = late_count / late_total
        lo, hi = proportion_ci(late_count, late_total, config)
        rising = late_share > early_share
        direction = ("emerging" if rising else "fading") if sig else "stable"
        verb = "rose" if rising else "fell"
        sig_txt = "a statistically significant" if sig else "no significant"
        findings.append(
            Finding(
                kind="change",
                label=str(labels.get(tid, f"theme {tid}")),
                theme_id=int(tid),
                estimate=late_share,
                ci_low=lo,
                ci_high=hi,
                test_stat=z,
                p_value=p,
                p_corrected=q,
                significant=bool(sig),
                direction=direction,
                interpretation=f"Theme '{labels.get(tid, tid)}' {verb} from {early_share:.1%} to "
                f"{late_share:.1%} between the early and late period — {sig_txt} change "
                f"(z={z:.2f}, p={p:.3f}, BH q={q:.3f}).",
            )
        )
    return findings


def _sentiment_findings(labels: list[str], config: Config) -> list[Finding]:
    """Share of each sentiment label with a CI."""
    n = len(labels)
    if n == 0:
        return []
    findings = []
    series = pd.Series(labels)
    for label, count in series.value_counts().items():
        lo, hi = proportion_ci(int(count), n, config)
        est = count / n
        findings.append(
            Finding(
                kind="sentiment",
                label=f"sentiment: {label}",
                estimate=est,
                ci_low=lo,
                ci_high=hi,
                interpretation=f"{est:.1%} of analyzed complaints are {label} "
                f"(95% CI {lo:.1%}–{hi:.1%}).",
            )
        )
    return findings


def _write_report(out_path, payload: dict) -> None:
    s = payload["sampling"]
    lines = [
        "# Statistical Findings",
        "",
        "## Sampling design",
        "",
        f"- To estimate a theme's prevalence within ±{s['margin_of_error']:.0%} at "
        f"{s['confidence']:.0%} confidence requires **{s['design_n']:,}** labeled records "
        "(conservative at p=0.5).",
        f"- Realized LLM-labeled sample: **{s['realized_n']:,}** records.",
        "",
        "## Top theme prevalence (with 95% CI)",
        "",
        "| Theme | Prevalence | 95% CI |",
        "|---|--:|--:|",
    ]
    for f in payload["prevalence"]:
        lines.append(f"| {f['label']} | {f['estimate']:.1%} | {f['ci_low']:.1%}–{f['ci_high']:.1%} |")

    movers = [f for f in payload["changes"] if f["significant"]]
    lines += ["", "## Statistically significant movers (BH-corrected)", ""]
    if movers:
        for f in sorted(movers, key=lambda f: f["p_corrected"]):
            arrow = "📈" if f["direction"] == "emerging" else "📉"
            lines.append(f"- {arrow} **{f['label']}** ({f['direction']}): {f['interpretation']}")
    else:
        lines.append("- No themes changed significantly after multiple-comparison correction.")

    lines += ["", "## Sentiment distribution (with 95% CI)", ""]
    for f in payload["sentiment"]:
        lines.append(f"- {f['interpretation']}")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_stats(config: Config) -> None:
    """Produce statistically-validated findings from upstream outputs.

    Args:
        config: Loaded pipeline configuration.
    """
    if not config.paths.theme_prevalence.exists():
        raise FileNotFoundError(
            f"Theme prevalence not found: {config.paths.theme_prevalence}. Run the themes stage."
        )
    prev = pd.read_parquet(config.paths.theme_prevalence)
    summary = pd.read_parquet(config.paths.theme_summary)
    total = int(summary["n_docs"].sum())
    llm_labels = _load_llm_labels(config.paths.llm_extractions)

    prevalence = _prevalence_findings(summary, total, config)
    changes = _change_findings(prev, summary, config)
    sentiment = _sentiment_findings(llm_labels, config)

    n_sig = sum(1 for f in changes if f.significant)
    print(f"[stats] {len(prevalence)} themes; {n_sig} significant movers (BH-corrected); "
          f"{len(sentiment)} sentiment findings")

    payload = {
        "sampling": {
            "margin_of_error": config.stats.margin_of_error,
            "confidence": config.stats.confidence_level,
            "design_n": sample_size_for_proportion(config.stats.margin_of_error, config.stats.confidence_level),
            "realized_n": len(llm_labels),
        },
        "prevalence": [asdict(f) for f in prevalence],
        "changes": [asdict(f) for f in changes],
        "sentiment": [asdict(f) for f in sentiment],
    }
    config.paths.findings.parent.mkdir(parents=True, exist_ok=True)
    config.paths.findings.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(config.paths.reports_dir / "statistics.md", payload)
    print(f"[stats] Wrote {config.paths.findings} and reports/statistics.md")
