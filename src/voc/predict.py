"""Module 5 — Predictive Model.

Forecasts a real behavioral outcome (CFPB ``Consumer disputed?`` or
``Timely response?``) from **text-derived** features, demonstrating that NLP
outputs can predict a downstream business outcome:

  * PCA-reduced embeddings           (Module 2)
  * theme assignment + probability   (Module 3)
  * LLM sentiment / severity / etc.  (Module 4)
  * light metadata (category, text length)

Uses a **time-aware** train/test split (train on earlier periods, test on
later) to avoid leakage, fits the unsupervised PCA and the logistic scaler on
the training split only, and compares a gradient-boosting model against a
logistic-regression baseline. Reports ROC-AUC, PR-AUC, calibration (Brier),
a confusion matrix, and model-agnostic permutation feature importance.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from voc import schema as cols
from voc.config import Config
from voc.embed import embed_corpus, get_embedder

_SEVERITY_ORD = {"low": 0, "medium": 1, "high": 2}


# --------------------------------------------------------------------------- #
# Target                                                                      #
# --------------------------------------------------------------------------- #
def build_target(records: pd.DataFrame, config: Config) -> tuple[pd.Series, str]:
    """Build the binary target (1 = the adverse outcome we predict).

    Returns ``(y, name)``. Rows whose target is missing should be dropped by
    the caller.
    """
    target = config.predict.target
    if target == "timely_response":
        if cols.TIMELY not in records.columns:
            raise ValueError("config target 'timely_response' requires a mapped timely column.")
        y = (records[cols.TIMELY].astype(str).str.strip().str.lower() == "no").astype(int)
        return y, "not_timely_response"
    if target == "consumer_disputed":
        y = (records[cols.OUTCOME].astype(str).str.strip().str.lower() == "yes").astype(int)
        return y, "consumer_disputed"
    raise ValueError(f"Unknown predict.target: {target!r}")


# --------------------------------------------------------------------------- #
# Feature assembly                                                            #
# --------------------------------------------------------------------------- #
def _load_llm_extractions(path) -> pd.DataFrame:
    """Load successful LLM extractions from the checkpoint JSONL."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("ok"):
            rows.append(obj)
    if not rows:
        raise ValueError(f"No successful LLM extractions in {path}. Run the extract stage.")
    keep = ["record_id", "sentiment_label", "sentiment_score", "issue_category",
            "severity", "is_actionable"]
    return pd.DataFrame(rows)[keep]


def assemble_features(
    config: Config, embedder=None
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Series, dict[str, str]]:
    """Join all upstream outputs into a modeling table.

    Returns ``(features, embeddings, y, dates, provenance)`` where ``features``
    is the non-embedding feature frame (numeric after one-hot encoding),
    ``embeddings`` is the aligned embedding matrix (reduced later, on train
    only), ``y`` is the target, ``dates`` drives the time split, and
    ``provenance`` documents which module each feature group came from.
    """
    records = pd.read_parquet(config.paths.records_parquet)
    themes = pd.read_parquet(config.paths.theme_assignments)
    llm = _load_llm_extractions(config.paths.llm_extractions)

    # Embeddings aligned to records (reuse the M2 cache), indexed by record_id.
    embedder = embedder or get_embedder(config)
    cache_path = config.paths.embeddings_dir / "embeddings.npz"
    emb_matrix = embed_corpus(
        embedder, records[cols.TEXT_CLEAN].tolist(), cache_path, use_cache=config.embed.cache
    )
    emb_df = pd.DataFrame(emb_matrix, index=records[cols.RECORD_ID].values)

    records = records.copy()
    records["text_length"] = records[cols.TEXT].str.len()
    y_all, _ = build_target(records, config)
    records = records.assign(_y=y_all)

    # Inner-join: model only records that have LLM features (the LLM-labeled sample).
    df = (
        records.merge(themes, on=cols.RECORD_ID, how="inner")
        .merge(llm, on=cols.RECORD_ID, how="inner")
        .dropna(subset=["_y"])
        .reset_index(drop=True)
    )

    y = df["_y"].astype(int)
    dates = pd.to_datetime(df[cols.DATE])
    embeddings = emb_df.loc[df[cols.RECORD_ID].values].to_numpy()

    # Engineered, text-derived features.
    feat = pd.DataFrame(index=df.index)
    feat["sentiment_score"] = df["sentiment_score"].astype(float)          # M4
    feat["is_actionable"] = df["is_actionable"].astype(int)                # M4
    feat["severity_ord"] = df["severity"].map(_SEVERITY_ORD).fillna(1).astype(int)  # M4
    feat["theme_probability"] = df["probability"].astype(float)            # M3
    feat["text_length"] = df["text_length"].astype(float)                  # metadata

    categoricals = pd.get_dummies(
        df[["category", "issue_category", "sentiment_label", "theme_id"]].astype(str),
        prefix={"category": "cat", "issue_category": "iss",
                "sentiment_label": "sent", "theme_id": "theme"},
    ).astype(float)
    features = pd.concat([feat, categoricals], axis=1)

    provenance = {
        "embedding PCA components": "M2 (embeddings)",
        "theme_*, theme_probability": "M3 (themes)",
        "sentiment_*, severity_ord, is_actionable, iss_*": "M4 (LLM extraction)",
        "cat_*, text_length": "metadata",
    }
    return features, embeddings, y, dates, provenance


# --------------------------------------------------------------------------- #
# Time-aware split                                                            #
# --------------------------------------------------------------------------- #
def time_split(dates: pd.Series, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split indices by time: earliest ``1 - test_fraction`` train, latest test.

    The cutoff is the ``(1 - test_fraction)`` date quantile; records on or
    before it train, later records test. This prevents using the future to
    predict the past.
    """
    cutoff = dates.quantile(1.0 - test_fraction)
    train = np.where(dates <= cutoff)[0]
    test = np.where(dates > cutoff)[0]
    return train, test


# --------------------------------------------------------------------------- #
# Models                                                                      #
# --------------------------------------------------------------------------- #
def _make_gb(config: Config):
    """Construct the configured gradient-boosting classifier.

    Falls back from a missing/unloadable optional booster to a clear, actionable
    error (the offline ``sklearn`` backend has no system dependencies).
    """
    model = config.predict.model
    if model == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "LightGBM is unavailable. Install the ML extra (`pip install -e \".[ml]\"`); "
                "on macOS it also needs OpenMP (`brew install libomp`), on Debian/Ubuntu "
                "`apt-get install libgomp1`. Or set `predict.model: sklearn` for a "
                "dependency-free gradient booster."
            ) from exc
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, class_weight="balanced",
            random_state=config.project.seed, verbose=-1,
        )
    if model == "xgboost":
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "XGBoost is unavailable. `pip install xgboost`, or set "
                "`predict.model: sklearn` for a dependency-free gradient booster."
            ) from exc
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05, eval_metric="logloss",
            random_state=config.project.seed,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, class_weight="balanced",
        random_state=config.project.seed,
    )


def _make_baseline(config: Config):
    """Logistic-regression baseline with standardized features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    """Compute ROC-AUC, PR-AUC, Brier, and a confusion matrix at ``threshold``."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        roc_auc_score,
    )

    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    single_class = y_test.nunique() < 2
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)) if not single_class else None,
        "pr_auc": float(average_precision_score(y_test, proba)) if not single_class else None,
        "brier": float(brier_score_loss(y_test, proba)),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
    }


def _permutation_importance(model, x_test: pd.DataFrame, y_test: pd.Series, seed: int, top_n: int = 15):
    """Top model-agnostic permutation importances (works for any estimator)."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, x_test, y_test, n_repeats=5, random_state=seed, scoring="roc_auc"
    )
    order = np.argsort(result.importances_mean)[::-1][:top_n]
    return [
        {"feature": str(x_test.columns[i]), "importance": float(result.importances_mean[i])}
        for i in order
    ]


def _calibration_points(model, x_test: pd.DataFrame, y_test: pd.Series) -> list[dict]:
    from sklearn.calibration import calibration_curve

    if y_test.nunique() < 2:
        return []
    proba = model.predict_proba(x_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    return [
        {"prob_pred": float(p), "prob_true": float(t)}
        for p, t in zip(prob_pred, prob_true, strict=False)
    ]


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def _write_report(out_path, metrics: dict) -> None:
    s = metrics
    cm = s["gradient_boosting"]["confusion_matrix"]
    lines = [
        "# Predictive Model",
        "",
        f"**Target:** `{s['target']}` (1 = adverse outcome). "
        f"Base rate: {s['base_rate']:.1%} positive.",
        "",
        f"**Time-aware split:** train on records through {s['split']['cutoff']} "
        f"({s['split']['n_train']:,} rows), test on later records "
        f"({s['split']['n_test']:,} rows). No random split — avoids using the future "
        "to predict the past.",
        "",
        "## Model comparison (test set)",
        "",
        "| Model | ROC-AUC | PR-AUC | Brier |",
        "|---|--:|--:|--:|",
    ]
    for key, name in [("baseline_logistic", "Logistic (baseline)"),
                      ("gradient_boosting", f"Gradient boosting ({s['gb_backend']})")]:
        m = s[key]
        roc = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "n/a"
        pr = f"{m['pr_auc']:.3f}" if m["pr_auc"] is not None else "n/a"
        lines.append(f"| {name} | {roc} | {pr} | {m['brier']:.3f} |")

    lines += [
        "",
        f"## Confusion matrix — gradient boosting @ threshold {s['threshold']}",
        "",
        "| | pred 0 | pred 1 |",
        "|---|--:|--:|",
        f"| **actual 0** | {cm['tn']:,} | {cm['fp']:,} |",
        f"| **actual 1** | {cm['fn']:,} | {cm['tp']:,} |",
        "",
        "## Top features (permutation importance, ROC-AUC drop)",
        "",
        "| Feature | Importance |",
        "|---|--:|",
    ]
    lines += [f"| {f['feature']} | {f['importance']:.4f} |" for f in s["feature_importance"]]
    lines += [
        "",
        "## Feature provenance",
        "",
        *[f"- **{group}** — {src}" for group, src in s["provenance"].items()],
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_predict(config: Config, embedder=None) -> None:
    """Train and evaluate the predictive model on text-derived features.

    Args:
        config: Loaded pipeline configuration.
        embedder: Optional embedder override (defaults to the configured backend).
    """
    from sklearn.decomposition import PCA

    features, embeddings, y, dates, provenance = assemble_features(config, embedder)
    print(f"[predict] {len(features):,} rows, {features.shape[1]} non-embedding features")

    train_idx, test_idx = time_split(dates, config.predict.test_fraction)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Time split produced an empty train or test set; check date coverage.")

    # Fit PCA on TRAIN embeddings only (no leakage), then transform all.
    n_comp = min(config.predict.embedding_pca_components, embeddings.shape[1], len(train_idx))
    pca = PCA(n_components=n_comp, random_state=config.project.seed)
    pca.fit(embeddings[train_idx])
    emb_cols = [f"emb_{i}" for i in range(n_comp)]
    emb_feat = pd.DataFrame(pca.transform(embeddings), columns=emb_cols, index=features.index)

    x = pd.concat([features, emb_feat], axis=1)
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    baseline = _make_baseline(config)
    baseline.fit(x_train, y_train)
    gb = _make_gb(config)
    gb.fit(x_train, y_train)

    metrics = {
        "target": build_target(pd.read_parquet(config.paths.records_parquet), config)[1],
        "base_rate": float(y.mean()),
        "gb_backend": config.predict.model,
        "threshold": config.predict.threshold,
        "split": {
            "cutoff": str(dates.quantile(1.0 - config.predict.test_fraction).date()),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "baseline_logistic": evaluate_model(baseline, x_test, y_test, config.predict.threshold),
        "gradient_boosting": evaluate_model(gb, x_test, y_test, config.predict.threshold),
        "feature_importance": _permutation_importance(gb, x_test, y_test, config.project.seed),
        "calibration": _calibration_points(gb, x_test, y_test),
        "provenance": provenance,
    }

    gb_auc = metrics["gradient_boosting"]["roc_auc"]
    base_auc = metrics["baseline_logistic"]["roc_auc"]
    if gb_auc is not None and base_auc is not None:
        verdict = "beats" if gb_auc > base_auc else "does not beat"
        print(f"[predict] GB ROC-AUC {gb_auc:.3f} {verdict} baseline {base_auc:.3f}")

    x.assign(_y=y).to_parquet(config.paths.feature_table, index=False)
    config.paths.predict_metrics.parent.mkdir(parents=True, exist_ok=True)
    config.paths.predict_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_report(config.paths.reports_dir / "predictive_model.md", metrics)
    print(f"[predict] Wrote {config.paths.predict_metrics} and reports/predictive_model.md")
