"""Module 7 — Insight Report Generation.

Auto-generates an executive-readable report (``reports/voc_insight_report.md``,
optionally ``.html``) from the outputs of Modules 3–6. Leads with the "so what",
states uncertainty on every quantitative claim in plain language, and embeds a
few charts (theme prevalence over time with CI bands, sentiment distribution,
predictive feature importance).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display, safe in tests/CI
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from voc import schema as cols  # noqa: E402
from voc.config import Config  # noqa: E402
from voc.stats import wilson_interval  # noqa: E402

_FIG_DIR = "figures"


# --------------------------------------------------------------------------- #
# Charts                                                                      #
# --------------------------------------------------------------------------- #
def _chart_prevalence_over_time(
    prev: pd.DataFrame, summary: pd.DataFrame, out_path: Path, top_n: int, confidence: float
) -> bool:
    """Line chart of the top themes' monthly share with Wilson CI bands."""
    if prev.empty:
        return False
    top_ids = (
        summary[~summary["is_outlier"]].sort_values("n_docs", ascending=False)["theme_id"].head(top_n).tolist()
    )
    labels = dict(zip(summary["theme_id"], summary["label"], strict=False))
    months = sorted(prev[cols.YEAR_MONTH].unique())
    month_total = prev.groupby(cols.YEAR_MONTH)["n_docs"].sum().to_dict()

    fig, ax = plt.subplots(figsize=(10, 5))
    for tid in top_ids:
        sub = prev[prev["theme_id"] == tid].set_index(cols.YEAR_MONTH)
        shares, los, his = [], [], []
        for m in months:
            count = int(sub.loc[m, "n_docs"]) if m in sub.index else 0
            total = month_total.get(m, 0)
            share = count / total if total else 0.0
            lo, hi = wilson_interval(count, total, confidence) if total else (0.0, 0.0)
            shares.append(share)
            los.append(lo)
            his.append(hi)
        label = str(labels.get(tid, f"theme {tid}"))[:32]
        line, = ax.plot(months, shares, marker="o", markersize=3, label=label)
        ax.fill_between(months, los, his, alpha=0.15, color=line.get_color())

    ax.set_title("Theme prevalence over time (95% CI bands)")
    ax.set_ylabel("Share of complaints")
    ax.set_xlabel("Month")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _chart_sentiment(findings: list[dict], out_path: Path) -> bool:
    """Bar chart of sentiment shares with CI error bars."""
    if not findings:
        return False
    labels = [f["label"].replace("sentiment: ", "") for f in findings]
    est = [f["estimate"] for f in findings]
    lower = [f["estimate"] - f["ci_low"] for f in findings]
    upper = [f["ci_high"] - f["estimate"] for f in findings]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, est, yerr=[lower, upper], capsize=5, color="#4C72B0")
    ax.set_title("Sentiment distribution (95% CI)")
    ax.set_ylabel("Share of complaints")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _chart_feature_importance(predict: dict | None, out_path: Path, top_n: int = 12) -> bool:
    """Horizontal bar chart of permutation feature importance."""
    if not predict or not predict.get("feature_importance"):
        return False
    items = predict["feature_importance"][:top_n][::-1]
    names = [it["feature"] for it in items]
    vals = [it["importance"] for it in items]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, vals, color="#55A868")
    ax.set_title("Top predictors of the adverse outcome (permutation importance)")
    ax.set_xlabel("ROC-AUC drop when shuffled")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Markdown                                                                     #
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{x:.1%}"


def _headline_lines(findings: dict, predict: dict | None) -> list[str]:
    """Lead with the 'so what': the biggest, most decision-relevant takeaways."""
    lines = []
    prevalence = findings["prevalence"]
    if prevalence:
        top = max(prevalence, key=lambda f: f["estimate"])
        lines.append(
            f"- **Biggest driver of complaints:** \"{top['label']}\" at "
            f"{_pct(top['estimate'])} (95% CI {_pct(top['ci_low'])}–{_pct(top['ci_high'])})."
        )

    movers = [f for f in findings["changes"] if f["significant"]]
    if movers:
        m = min(movers, key=lambda f: f["p_corrected"])
        lines.append(
            f"- **Biggest mover:** \"{m['label']}\" is **{m['direction']}** — "
            f"now {_pct(m['estimate'])} (corrected p={m['p_corrected']:.3f})."
        )
    else:
        lines.append(
            "- **Trends:** no theme changed significantly after multiple-comparison "
            "correction — the complaint mix is stable over the period."
        )

    if predict and predict.get("gradient_boosting", {}).get("roc_auc") is not None:
        gb = predict["gradient_boosting"]["roc_auc"]
        base = predict["baseline_logistic"]["roc_auc"]
        top_feat = predict["feature_importance"][0]["feature"] if predict["feature_importance"] else "n/a"
        lines.append(
            f"- **Predicting `{predict['target']}`** (base rate {_pct(predict['base_rate'])}): "
            f"text-derived features reach ROC-AUC {gb:.3f} (baseline {base:.3f}); "
            f"top signal: `{top_feat}`."
        )

    sentiment = findings["sentiment"]
    if sentiment:
        neg = next((f for f in sentiment if "negative" in f["label"]), None)
        if neg:
            lines.append(
                f"- **Sentiment:** {_pct(neg['estimate'])} of analyzed complaints are negative "
                f"(95% CI {_pct(neg['ci_low'])}–{_pct(neg['ci_high'])})."
            )
    return lines


def build_markdown(findings: dict, predict: dict | None, charts: dict[str, bool]) -> str:
    """Assemble the executive report markdown from the findings + predictions."""
    s = findings["sampling"]
    lines: list[str] = [
        "# Voice-of-Customer Insight Report",
        "",
        "*What customers are telling us, how it is changing, and what it predicts — "
        "with statistical confidence.*",
        "",
        "## Headline findings",
        "",
        *_headline_lines(findings, predict),
        "",
        "## Top themes",
        "",
        "| Theme | Prevalence | 95% CI | Trend |",
        "|---|--:|--:|---|",
    ]
    trend_by_id = {f["theme_id"]: f for f in findings["changes"]}
    for f in findings["prevalence"]:
        ch = trend_by_id.get(f["theme_id"])
        trend = "stable"
        if ch and ch["significant"]:
            trend = f"📈 {ch['direction']}" if ch["direction"] == "emerging" else f"📉 {ch['direction']}"
        lines.append(
            f"| {f['label']} | {_pct(f['estimate'])} | "
            f"{_pct(f['ci_low'])}–{_pct(f['ci_high'])} | {trend} |"
        )

    if charts.get("prevalence"):
        lines += ["", f"![Theme prevalence over time]({_FIG_DIR}/prevalence_over_time.png)"]

    lines += ["", "## Statistically significant movers", ""]
    movers = [f for f in findings["changes"] if f["significant"]]
    if movers:
        for f in sorted(movers, key=lambda f: f["p_corrected"]):
            lines.append(f"- {f['interpretation']}")
    else:
        lines.append(
            "No themes changed significantly between the early and late period after "
            "Benjamini-Hochberg correction. With many themes tested at once, this guards "
            "against mistaking noise for a trend."
        )

    if charts.get("sentiment"):
        lines += ["", "## Sentiment", "", f"![Sentiment distribution]({_FIG_DIR}/sentiment_distribution.png)"]

    lines += ["", "## What predicts the adverse outcome", ""]
    if predict and predict.get("gradient_boosting", {}).get("roc_auc") is not None:
        gb = predict["gradient_boosting"]
        base = predict["baseline_logistic"]
        lines += [
            f"Target: `{predict['target']}` (1 = adverse), base rate {_pct(predict['base_rate'])}. "
            f"Using a **time-aware** split (train on earlier complaints, test on later), a "
            f"gradient-boosting model on text-derived features reaches **ROC-AUC {gb['roc_auc']:.3f}** "
            f"/ PR-AUC {gb['pr_auc']:.3f}, versus a logistic baseline at ROC-AUC {base['roc_auc']:.3f}.",
            "",
            "Top predictors (permutation importance):",
            "",
            *[f"- `{it['feature']}` ({it['importance']:.4f})" for it in predict["feature_importance"][:8]],
        ]
        if charts.get("importance"):
            lines += ["", f"![Feature importance]({_FIG_DIR}/feature_importance.png)"]
    else:
        lines.append("_Predictive model outputs not available._")

    lines += [
        "",
        "## Methodology & caveats",
        "",
        f"- **Sampling:** estimating a theme's prevalence within ±{s['margin_of_error']:.0%} at "
        f"{s['confidence']:.0%} confidence needs {s['design_n']:,} labeled records; "
        f"{s['realized_n']:,} were labeled.",
        "- **Confidence intervals:** every reported proportion carries a Wilson score interval — "
        "no bare point estimates.",
        "- **Change over time:** two-proportion z-tests between an early and a late period, with "
        "Benjamini-Hochberg FDR correction because many themes are tested simultaneously.",
        "- **Prediction:** time-aware train/test split (not random) to avoid using the future to "
        "predict the past; unsupervised dimensionality reduction fit on the training split only.",
        "- **Caveats:** the LLM labels a sample of records (not the full corpus); theme labels are "
        "auto-generated keyword summaries; figures reflect the configured dataset.",
        "",
    ]
    return "\n".join(lines)


def _md_to_html(md: str, title: str) -> str:
    """Minimal markdown -> HTML for the constructs this report emits."""
    html: list[str] = []
    in_table = in_list = False

    def close_blocks() -> None:
        nonlocal in_list
        if in_list:
            html.append("</ul>")
            in_list = False

    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("|") and "---" in line:
            continue  # table separator row
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not in_table:
                html.append("<table border='1' cellpadding='4' cellspacing='0'>")
                in_table = True
            html.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            html.append("</table>")
            in_table = False
        if line.startswith(("# ", "## ", "### ")):
            close_blocks()
            level = len(line) - len(line.lstrip("#"))
            html.append(f"<h{level}>{line[level + 1:]}</h{level}>")
        elif line.startswith("!["):
            close_blocks()
            src = line[line.find("(") + 1 : line.find(")")]
            html.append(f"<img src='{src}' style='max-width:100%'>")
        elif line.startswith("- "):
            if not in_list:
                html.append("<ul>")
                in_list = True
            html.append(f"<li>{line[2:]}</li>")
        elif line:
            close_blocks()
            html.append(f"<p>{line}</p>")
    close_blocks()
    if in_table:
        html.append("</table>")
    body = "\n".join(html)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head><body>{body}</body></html>"


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run_report(config: Config) -> None:
    """Generate the executive insight report from stage outputs.

    Args:
        config: Loaded pipeline configuration.
    """
    if not config.paths.findings.exists():
        raise FileNotFoundError(
            f"Findings not found: {config.paths.findings}. Run the stats stage first."
        )
    findings = json.loads(config.paths.findings.read_text(encoding="utf-8"))
    predict = (
        json.loads(config.paths.predict_metrics.read_text(encoding="utf-8"))
        if config.paths.predict_metrics.exists()
        else None
    )
    summary = pd.read_parquet(config.paths.theme_summary)
    prev = pd.read_parquet(config.paths.theme_prevalence)

    fig_dir = config.report.output_md.parent / _FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "prevalence": _chart_prevalence_over_time(
            prev, summary, fig_dir / "prevalence_over_time.png",
            config.report.top_n_themes, config.stats.confidence_level,
        ),
        "sentiment": _chart_sentiment(findings["sentiment"], fig_dir / "sentiment_distribution.png"),
        "importance": _chart_feature_importance(predict, fig_dir / "feature_importance.png"),
    }

    md = build_markdown(findings, predict, charts)
    config.report.output_md.parent.mkdir(parents=True, exist_ok=True)
    config.report.output_md.write_text(md, encoding="utf-8")
    print(f"[report] Wrote {config.report.output_md}")

    if config.report.output_html:
        config.report.output_html.write_text(
            _md_to_html(md, "Voice-of-Customer Insight Report"), encoding="utf-8"
        )
        print(f"[report] Wrote {config.report.output_html}")
