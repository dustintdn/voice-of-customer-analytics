"""M3 tests: clustering, c-TF-IDF keywords, prevalence-over-time, end-to-end.

All offline: uses the KMeans backend and the HashingEmbedder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from voc import schema
from voc.config import load_config
from voc.themes import (
    OUTLIER_LABEL,
    build_summary,
    cluster_embeddings,
    ctfidf_keywords,
    prevalence_over_time,
    representative_docs,
    run_themes,
)


def _kmeans_config(tmp_path, n_clusters=2, outlier_quantile=0.0):
    config = load_config("config/config.yaml")
    config.themes.algorithm = "kmeans"
    config.themes.kmeans.n_clusters = n_clusters
    config.themes.kmeans.outlier_quantile = outlier_quantile
    config.embed.backend = "hashing"
    config.paths.records_parquet = tmp_path / "records.parquet"
    config.paths.embeddings_dir = tmp_path / "emb"
    config.paths.theme_assignments = tmp_path / "assign.parquet"
    config.paths.theme_summary = tmp_path / "summary.parquet"
    config.paths.theme_prevalence = tmp_path / "prev.parquet"
    config.paths.reports_dir = tmp_path / "reports"
    return config


# --- c-TF-IDF -------------------------------------------------------------- #
def test_ctfidf_keywords_are_cluster_distinctive() -> None:
    docs = [
        "overdraft fee bank checking overdraft",
        "overdraft fee bank account overdraft",
        "mortgage escrow payment loan mortgage",
        "mortgage escrow payment servicer mortgage",
    ]
    labels = np.array([0, 0, 1, 1])
    kw = ctfidf_keywords(docs, labels, top_n=3)
    assert "overdraft" in kw[0]
    assert "mortgage" in kw[1]
    assert "overdraft" not in kw[1]


def test_ctfidf_excludes_outliers() -> None:
    docs = ["alpha alpha beta", "gamma gamma delta", "noise noise noise"]
    labels = np.array([0, 1, OUTLIER_LABEL])
    kw = ctfidf_keywords(docs, labels, top_n=2)
    assert OUTLIER_LABEL not in kw
    assert set(kw.keys()) == {0, 1}


# --- clustering + outliers ------------------------------------------------- #
def test_kmeans_clusters_and_outlier_rule(tmp_path) -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.1, size=(25, 8)) + np.array([6, 0, 0, 0, 0, 0, 0, 0])
    b = rng.normal(0.0, 0.1, size=(25, 8)) + np.array([0, 6, 0, 0, 0, 0, 0, 0])
    emb = np.vstack([a, b]).astype(np.float32)

    # No outlier rule: a clean two-way split, no noise label, valid probabilities.
    cfg0 = _kmeans_config(tmp_path, n_clusters=2, outlier_quantile=0.0)
    labels0, probs0 = cluster_embeddings(emb, ["x"] * len(emb), cfg0)
    assert (labels0 == OUTLIER_LABEL).sum() == 0
    assert len(set(labels0)) == 2
    assert len(set(labels0[:25])) == 1 and len(set(labels0[25:])) == 1
    assert labels0[0] != labels0[25]
    assert probs0.shape == (50,)
    assert (probs0 >= 0).all() and (probs0 <= 1).all()

    # With the outlier rule, the farthest-from-centroid fraction becomes noise (~10%).
    cfg1 = _kmeans_config(tmp_path, n_clusters=2, outlier_quantile=0.1)
    labels1, _ = cluster_embeddings(emb, ["x"] * len(emb), cfg1)
    n_out = int((labels1 == OUTLIER_LABEL).sum())
    assert 1 <= n_out <= 8  # ~10% of 50, with slack


# --- prevalence ------------------------------------------------------------ #
def test_prevalence_shares_sum_to_one_per_month() -> None:
    df = pd.DataFrame(
        {
            "theme_id": [0, 0, 1, 0, 1, 1],
            schema.YEAR_MONTH: ["2021-01", "2021-01", "2021-01", "2021-02", "2021-02", "2021-02"],
        }
    )
    prev = prevalence_over_time(df)
    by_month = prev.groupby(schema.YEAR_MONTH)["share"].sum()
    np.testing.assert_allclose(by_month.values, 1.0)
    # Jan: theme 0 has 2/3, theme 1 has 1/3.
    jan0 = prev[(prev[schema.YEAR_MONTH] == "2021-01") & (prev["theme_id"] == 0)]["share"].iloc[0]
    assert abs(jan0 - 2 / 3) < 1e-9


# --- representative docs + summary ----------------------------------------- #
def test_representative_docs_and_summary() -> None:
    emb = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    texts = ["a0", "a1", "b0", "b1"]
    reps = representative_docs(emb, labels, texts, n=1)
    assert reps[0][0] in {"a0", "a1"}
    summary = build_summary(labels, {0: ["x"], 1: ["y"]}, reps)
    assert set(summary["theme_id"]) == {0, 1}
    assert abs(summary["share"].sum() - 1.0) < 1e-9


# --- end-to-end ------------------------------------------------------------ #
def test_run_themes_end_to_end(tmp_path) -> None:
    # Two clearly separable themes across two months.
    overdraft = "the bank charged me overdraft fees on my checking account"
    mortgage = "my mortgage escrow payment increased without explanation from the servicer"
    records = pd.DataFrame(
        {
            schema.RECORD_ID: list(range(8)),
            schema.TEXT: [overdraft, mortgage] * 4,
            schema.TEXT_CLEAN: [overdraft, mortgage] * 4,
            schema.YEAR_MONTH: ["2021-01"] * 4 + ["2021-02"] * 4,
        }
    )
    config = _kmeans_config(tmp_path, n_clusters=2, outlier_quantile=0.0)
    records.to_parquet(config.paths.records_parquet, index=False)

    run_themes(config)

    assign = pd.read_parquet(config.paths.theme_assignments)
    summary = pd.read_parquet(config.paths.theme_summary)
    prev = pd.read_parquet(config.paths.theme_prevalence)

    assert set(assign.columns) == {schema.RECORD_ID, "theme_id", "probability"}
    assert len(assign) == 8
    assert {"theme_id", "label", "keywords", "n_docs", "share", "examples"} <= set(summary.columns)
    # keywords/examples round-trip as lists through parquet.
    assert isinstance(summary.iloc[0]["keywords"], list | np.ndarray)
    assert (config.paths.reports_dir / "themes.md").exists()
    np.testing.assert_allclose(prev.groupby(schema.YEAR_MONTH)["share"].sum().values, 1.0)
