"""M5 tests: target construction, time-aware split, evaluation, end-to-end.

Offline: uses the HashingEmbedder and the sklearn gradient-boosting backend.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from voc import schema as cols
from voc.config import load_config
from voc.predict import build_target, evaluate_model, run_predict, time_split


# --- target ---------------------------------------------------------------- #
def test_build_target_timely() -> None:
    config = load_config("config/config.yaml")
    config.predict.target = "timely_response"
    records = pd.DataFrame({cols.TIMELY: ["Yes", "No", "yes", "NO"], cols.OUTCOME: ["No"] * 4})
    y, name = build_target(records, config)
    assert name == "not_timely_response"
    assert list(y) == [0, 1, 0, 1]  # "not timely" is the adverse outcome


def test_build_target_disputed() -> None:
    config = load_config("config/config.yaml")
    config.predict.target = "consumer_disputed"
    records = pd.DataFrame({cols.OUTCOME: ["Yes", "No", "YES", "no"], cols.TIMELY: ["Yes"] * 4})
    y, name = build_target(records, config)
    assert name == "consumer_disputed"
    assert list(y) == [1, 0, 1, 0]


# --- time split ------------------------------------------------------------ #
def test_time_split_is_chronological_and_disjoint() -> None:
    dates = pd.to_datetime([f"2021-{m:02d}-01" for m in range(1, 13)])
    train, test = time_split(pd.Series(dates), test_fraction=0.25)
    assert len(train) + len(test) == 12
    assert set(train).isdisjoint(set(test))
    # Every test date is strictly later than every train date (no leakage).
    assert dates[test].min() > dates[train].max()
    assert len(test) == 3  # latest 25% of 12


# --- evaluation ------------------------------------------------------------ #
def test_evaluate_model_on_separable_data() -> None:
    from sklearn.linear_model import LogisticRegression

    x = pd.DataFrame({"f": list(range(-20, 20))})
    y = pd.Series([0] * 20 + [1] * 20)
    model = LogisticRegression().fit(x, y)
    metrics = evaluate_model(model, x, y, threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    cm = metrics["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 40
    assert 0.0 <= metrics["brier"] <= 1.0


# --- end-to-end ------------------------------------------------------------ #
def _make_fixtures(tmp_path, n=160):
    rng = np.random.default_rng(0)
    categories = ["Credit reporting", "Mortgage", "Debt collection"]
    rows = []
    for i in range(n):
        cat = categories[i % len(categories)]
        escalated = rng.random() < 0.3
        text = f"complaint {i} about {cat} issue" + (" legal action attorney" if escalated else "")
        # Category-dependent escalation interaction -> learnable, non-additive.
        eff = {"Credit reporting": 0.4, "Mortgage": -0.1, "Debt collection": 0.3}[cat]
        untimely = min(0.9, max(0.02, 0.1 + (eff if escalated else 0)))
        rows.append(
            {
                cols.RECORD_ID: i,
                cols.TEXT: text,
                cols.TEXT_CLEAN: text.lower(),
                cols.DATE: pd.Timestamp("2021-01-01") + pd.Timedelta(days=i * 2),
                cols.YEAR_MONTH: "2021-01",
                cols.CATEGORY: cat,
                cols.TIMELY: "No" if rng.random() < untimely else "Yes",
                cols.OUTCOME: "No",
            }
        )
    records = pd.DataFrame(rows)
    records.to_parquet(tmp_path / "records.parquet", index=False)

    themes = pd.DataFrame(
        {cols.RECORD_ID: records[cols.RECORD_ID], "theme_id": [i % 3 for i in range(n)],
         "probability": rng.random(n)}
    )
    themes.to_parquet(tmp_path / "themes.parquet", index=False)

    with (tmp_path / "llm.jsonl").open("w") as fh:
        for _i, r in records.iterrows():
            esc = "attorney" in r[cols.TEXT]
            fh.write(json.dumps({
                "record_id": int(r[cols.RECORD_ID]), "ok": True,
                "sentiment_label": "negative", "sentiment_score": -0.9 if esc else -0.6,
                "issue_category": "other", "severity": "high" if esc else "medium",
                "is_actionable": True,
            }) + "\n")
    return records


def _configure(tmp_path):
    config = load_config("config/config.yaml")
    config.embed.backend = "hashing"
    config.predict.model = "sklearn"
    config.predict.embedding_pca_components = 8
    config.paths.records_parquet = tmp_path / "records.parquet"
    config.paths.theme_assignments = tmp_path / "themes.parquet"
    config.paths.llm_extractions = tmp_path / "llm.jsonl"
    config.paths.embeddings_dir = tmp_path / "emb"
    config.paths.feature_table = tmp_path / "features.parquet"
    config.paths.predict_metrics = tmp_path / "predict_metrics.json"
    config.paths.reports_dir = tmp_path / "reports"
    return config


def test_run_predict_end_to_end(tmp_path) -> None:
    _make_fixtures(tmp_path)
    config = _configure(tmp_path)

    run_predict(config)

    metrics = json.loads((tmp_path / "predict_metrics.json").read_text())
    assert metrics["target"] == "not_timely_response"
    assert 0.0 <= metrics["base_rate"] <= 1.0
    for key in ("baseline_logistic", "gradient_boosting"):
        assert metrics[key]["roc_auc"] is None or 0.0 <= metrics[key]["roc_auc"] <= 1.0
    assert metrics["feature_importance"]  # non-empty
    assert metrics["split"]["n_train"] > 0 and metrics["split"]["n_test"] > 0
    assert (tmp_path / "reports" / "predictive_model.md").exists()
    assert (tmp_path / "features.parquet").exists()
