"""M7 tests: executive report assembly, charts, and optional HTML."""

from __future__ import annotations

import json

import pandas as pd

from voc import schema as cols
from voc.config import load_config
from voc.report import _md_to_html, build_markdown, run_report


def _findings(with_mover: bool = True) -> dict:
    changes = [
        {"kind": "change", "label": "fees", "theme_id": 0, "estimate": 0.30,
         "ci_low": 0.25, "ci_high": 0.35, "test_stat": 4.1, "p_value": 0.0001,
         "p_corrected": 0.0003, "significant": with_mover, "direction": "emerging",
         "interpretation": "Theme 'fees' rose from 10.0% to 30.0% — a statistically significant change."},
        {"kind": "change", "label": "debt", "theme_id": 1, "estimate": 0.20,
         "ci_low": 0.16, "ci_high": 0.24, "test_stat": 0.3, "p_value": 0.7,
         "p_corrected": 0.7, "significant": False, "direction": "stable",
         "interpretation": "Theme 'debt' was stable."},
    ]
    return {
        "sampling": {"margin_of_error": 0.03, "confidence": 0.95, "design_n": 1068, "realized_n": 2000},
        "prevalence": [
            {"kind": "prevalence", "label": "fees", "theme_id": 0, "estimate": 0.30,
             "ci_low": 0.28, "ci_high": 0.32, "interpretation": "fees 30%"},
            {"kind": "prevalence", "label": "debt", "theme_id": 1, "estimate": 0.20,
             "ci_low": 0.18, "ci_high": 0.22, "interpretation": "debt 20%"},
        ],
        "changes": changes,
        "sentiment": [
            {"kind": "sentiment", "label": "sentiment: negative", "estimate": 0.8,
             "ci_low": 0.77, "ci_high": 0.83, "interpretation": "80% negative"},
            {"kind": "sentiment", "label": "sentiment: neutral", "estimate": 0.2,
             "ci_low": 0.17, "ci_high": 0.23, "interpretation": "20% neutral"},
        ],
    }


def _predict() -> dict:
    return {
        "target": "not_timely_response",
        "base_rate": 0.15,
        "gradient_boosting": {"roc_auc": 0.72, "pr_auc": 0.40, "brier": 0.10},
        "baseline_logistic": {"roc_auc": 0.66, "pr_auc": 0.35, "brier": 0.12},
        "feature_importance": [{"feature": "severity_ord", "importance": 0.08},
                               {"feature": "emb_2", "importance": 0.03}],
    }


def test_build_markdown_leads_with_so_what_and_marks_trends() -> None:
    md = build_markdown(_findings(with_mover=True), _predict(), charts={})
    assert md.startswith("# Voice-of-Customer Insight Report")
    assert "## Headline findings" in md
    assert "Biggest mover" in md            # significant mover surfaced in headline
    assert "📈 emerging" in md               # trend marked in the themes table
    assert "ROC-AUC 0.720" in md            # predictive section
    assert "Benjamini-Hochberg" in md       # methodology
    assert "Wilson" in md


def test_build_markdown_handles_no_movers() -> None:
    md = build_markdown(_findings(with_mover=False), _predict(), charts={})
    assert "no theme changed significantly" in md.lower()
    assert "📈" not in md and "📉" not in md


def test_md_to_html_renders_constructs() -> None:
    md = "# Title\n\n## Section\n\n- a bullet\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n![chart](figures/x.png)"
    html = _md_to_html(md, "T")
    assert "<h1>Title</h1>" in html
    assert "<li>a bullet</li>" in html
    assert "<table" in html and "<td>1</td>" in html
    assert "<img src='figures/x.png'" in html


def test_run_report_end_to_end(tmp_path) -> None:
    config = load_config("config/config.yaml")
    config.paths.findings = tmp_path / "findings.json"
    config.paths.predict_metrics = tmp_path / "predict_metrics.json"
    config.paths.theme_summary = tmp_path / "summary.parquet"
    config.paths.theme_prevalence = tmp_path / "prev.parquet"
    config.report.output_md = tmp_path / "report.md"
    config.report.output_html = tmp_path / "report.html"

    config.paths.findings.write_text(json.dumps(_findings()))
    config.paths.predict_metrics.write_text(json.dumps(_predict()))
    pd.DataFrame(
        [
            {"theme_id": 0, "label": "fees", "n_docs": 60, "share": 0.3, "is_outlier": False},
            {"theme_id": 1, "label": "debt", "n_docs": 40, "share": 0.2, "is_outlier": False},
            {"theme_id": -1, "label": "Outliers", "n_docs": 0, "share": 0.0, "is_outlier": True},
        ]
    ).to_parquet(config.paths.theme_summary, index=False)
    pd.DataFrame(
        [
            {cols.YEAR_MONTH: "2021-01", "theme_id": 0, "n_docs": 5, "share": 0.25},
            {cols.YEAR_MONTH: "2021-01", "theme_id": 1, "n_docs": 15, "share": 0.75},
            {cols.YEAR_MONTH: "2021-02", "theme_id": 0, "n_docs": 15, "share": 0.75},
            {cols.YEAR_MONTH: "2021-02", "theme_id": 1, "n_docs": 5, "share": 0.25},
        ]
    ).to_parquet(config.paths.theme_prevalence, index=False)

    run_report(config)

    assert config.report.output_md.exists()
    assert config.report.output_html.exists()
    assert "<img" in config.report.output_html.read_text()
    fig_dir = config.report.output_md.parent / "figures"
    assert (fig_dir / "prevalence_over_time.png").exists()
    assert (fig_dir / "sentiment_distribution.png").exists()
    assert (fig_dir / "feature_importance.png").exists()
