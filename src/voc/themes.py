"""Module 3 — Theme Extraction (clustering / topic modeling).

Clusters the precomputed embeddings into themes, then (backend-agnostically)
derives c-TF-IDF keywords, representative example snippets, document
counts/shares, and theme prevalence over time. The outlier/noise cluster
(label ``-1``) is reported explicitly, never silently dropped.

Two clustering backends (``config.themes.algorithm``):
  * ``bertopic`` — the real, default backend, layered on the precomputed
    embeddings (UMAP + HDBSCAN + c-TF-IDF). Lazy-imported.
  * ``kmeans`` — a lightweight, fully-offline backend (scikit-learn KMeans with
    a distance-based outlier rule). Used by the test suite and for offline runs.

Outputs:
  * ``theme_assignments.parquet`` — record_id -> theme_id, probability
  * ``theme_summary.parquet``     — per-theme label, keywords, count, share, examples
  * ``theme_prevalence.parquet``  — theme share per year_month (feeds M6 stats)
  * ``reports/themes.md``         — human-readable summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from voc import schema
from voc.config import Config
from voc.embed import embed_corpus, get_embedder

OUTLIER_LABEL = -1


# --------------------------------------------------------------------------- #
# Clustering backends                                                         #
# --------------------------------------------------------------------------- #
def _cluster_kmeans(embeddings: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Offline KMeans clustering with a distance-based outlier rule.

    Returns ``(labels, probabilities)``. The farthest ``outlier_quantile``
    fraction of documents (by distance to their assigned centroid) are relabeled
    as noise (``-1``). Probability is a confidence in ``(0, 1]`` derived from the
    within-corpus distance.
    """
    from sklearn.cluster import KMeans

    params = config.themes.kmeans
    k = min(params.n_clusters, len(embeddings))
    km = KMeans(n_clusters=k, random_state=config.project.seed, n_init=10)
    labels = km.fit_predict(embeddings).astype(int)

    dist = np.linalg.norm(embeddings - km.cluster_centers_[labels], axis=1)
    max_dist = float(dist.max()) or 1.0
    probabilities = 1.0 - (dist / max_dist)

    if params.outlier_quantile > 0:
        threshold = np.quantile(dist, 1.0 - params.outlier_quantile)
        labels[dist > threshold] = OUTLIER_LABEL

    return labels, probabilities.astype(np.float32)


def _cluster_bertopic(
    embeddings: np.ndarray, docs: list[str], config: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Real BERTopic clustering on precomputed embeddings (lazy-imported)."""
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError(
            "BERTopic backend requires the ML extra (`pip install -e \".[ml]\"`). "
            "Or set `themes.algorithm: kmeans` for the dependency-free offline backend."
        ) from exc

    t = config.themes
    umap_model = UMAP(
        n_neighbors=t.umap.n_neighbors,
        n_components=t.umap.n_components,
        min_dist=t.umap.min_dist,
        metric="cosine",
        random_state=config.project.seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=t.hdbscan.min_cluster_size,
        min_samples=t.hdbscan.min_samples,
        prediction_data=True,
    )
    nr_topics = None if t.nr_topics == "auto" else int(t.nr_topics)
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=None,  # embeddings are supplied; never download a model
        min_topic_size=t.min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=True,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    labels = np.asarray(topics, dtype=int)
    if probs is not None and np.ndim(probs) == 2:
        probabilities = probs.max(axis=1)
    else:
        probabilities = np.asarray(probs if probs is not None else np.ones(len(labels)))
    return labels, probabilities.astype(np.float32)


def cluster_embeddings(
    embeddings: np.ndarray, docs: list[str], config: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the configured clustering backend."""
    if config.themes.algorithm == "kmeans":
        return _cluster_kmeans(embeddings, config)
    return _cluster_bertopic(embeddings, docs, config)


# --------------------------------------------------------------------------- #
# Backend-agnostic theme characterization                                     #
# --------------------------------------------------------------------------- #
def ctfidf_keywords(
    docs: list[str], labels: np.ndarray, top_n: int, stopwords: str = "english"
) -> dict[int, list[str]]:
    """Top class-based TF-IDF keywords per cluster (BERTopic-style c-TF-IDF).

    Each cluster is treated as a single class-document; terms are weighted by
    ``(t_c / w_c) * log(1 + A / f_t)`` where ``t_c`` is the term's frequency in
    class ``c``, ``w_c`` the class's token total, ``A`` the average tokens per
    class, and ``f_t`` the term's total frequency across all classes. The
    outlier cluster (``-1``) is excluded.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    mask = labels != OUTLIER_LABEL
    if not mask.any():
        return {}
    classes = sorted({int(label) for label in labels[mask]})

    vectorizer = CountVectorizer(stop_words=stopwords)
    counts = vectorizer.fit_transform([docs[i] for i in np.where(mask)[0]])
    vocab = np.array(vectorizer.get_feature_names_out())
    sub_labels = labels[mask]

    # Aggregate term counts per class -> (n_classes, vocab).
    class_counts = np.vstack(
        [np.asarray(counts[sub_labels == c].sum(axis=0)).ravel() for c in classes]
    )
    w_c = class_counts.sum(axis=1, keepdims=True)
    w_c[w_c == 0] = 1
    f_t = class_counts.sum(axis=0)
    f_t[f_t == 0] = 1
    avg_tokens = class_counts.sum() / len(classes)
    ctfidf = (class_counts / w_c) * np.log1p(avg_tokens / f_t)

    keywords: dict[int, list[str]] = {}
    for row, c in enumerate(classes):
        top_idx = np.argsort(ctfidf[row])[::-1][:top_n]
        keywords[c] = [str(vocab[i]) for i in top_idx if ctfidf[row, i] > 0]
    return keywords


def _centroids(embeddings: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Mean embedding per non-outlier cluster."""
    return {
        int(c): embeddings[labels == c].mean(axis=0)
        for c in {int(x) for x in labels}
        if c != OUTLIER_LABEL
    }


def representative_docs(
    embeddings: np.ndarray, labels: np.ndarray, texts: list[str], n: int
) -> dict[int, list[str]]:
    """The ``n`` documents nearest each cluster centroid (most representative)."""
    centroids = _centroids(embeddings, labels)
    reps: dict[int, list[str]] = {}
    for c, centroid in centroids.items():
        idx = np.where(labels == c)[0]
        dists = np.linalg.norm(embeddings[idx] - centroid, axis=1)
        nearest = idx[np.argsort(dists)[:n]]
        reps[c] = [texts[i] for i in nearest]
    return reps


def prevalence_over_time(theme_ym: pd.DataFrame) -> pd.DataFrame:
    """Theme share per ``year_month``.

    Args:
        theme_ym: DataFrame with ``theme_id`` and ``year_month`` columns.

    Returns:
        DataFrame with ``year_month``, ``theme_id``, ``n_docs``, ``share``
        (share is within-month and sums to 1 across themes for each month).
    """
    grouped = (
        theme_ym.groupby([schema.YEAR_MONTH, "theme_id"]).size().reset_index(name="n_docs")
    )
    totals = grouped.groupby(schema.YEAR_MONTH)["n_docs"].transform("sum")
    grouped["share"] = grouped["n_docs"] / totals
    return grouped.sort_values([schema.YEAR_MONTH, "theme_id"]).reset_index(drop=True)


def build_summary(
    labels: np.ndarray,
    keywords: dict[int, list[str]],
    reps: dict[int, list[str]],
) -> pd.DataFrame:
    """Per-theme summary table (label, keywords, count, share, examples)."""
    n_total = len(labels)
    rows = []
    for c in sorted({int(x) for x in labels}):
        count = int((labels == c).sum())
        is_outlier = c == OUTLIER_LABEL
        kws = keywords.get(c, [])
        label = "Outliers / unclustered" if is_outlier else ", ".join(kws[:4]) or f"theme {c}"
        rows.append(
            {
                "theme_id": c,
                "label": label,
                "keywords": kws,
                "n_docs": count,
                "share": count / n_total if n_total else 0.0,
                "is_outlier": is_outlier,
                "examples": reps.get(c, []),
            }
        )
    return pd.DataFrame(rows).sort_values("n_docs", ascending=False).reset_index(drop=True)


def _write_themes_report(summary: pd.DataFrame, out_path: str | object) -> None:
    """Render a human-readable theme summary to markdown."""
    from pathlib import Path

    lines = ["# Themes", "", f"Discovered **{(~summary['is_outlier']).sum()}** themes "
             f"(plus an outlier cluster).", ""]
    for _, r in summary.iterrows():
        head = f"## {r['label']}  \n" if not r["is_outlier"] else "## Outliers / unclustered  \n"
        lines.append(head)
        lines.append(f"- **Theme ID:** {r['theme_id']}  ")
        lines.append(f"- **Documents:** {r['n_docs']:,} ({100 * r['share']:.1f}%)  ")
        if not r["is_outlier"]:
            lines.append(f"- **Keywords:** {', '.join(r['keywords'])}  ")
            if len(r["examples"]):
                lines.append("- **Examples:**")
                lines.extend(f"  - {ex[:200]}" for ex in r["examples"][:3])
        lines.append("")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run_themes(config: Config) -> None:
    """Extract themes and write assignments, summaries, and prevalence tables.

    Args:
        config: Loaded pipeline configuration.
    """
    records_path = config.paths.records_parquet
    if not records_path.exists():
        raise FileNotFoundError(
            f"Cleaned records not found: {records_path}. Run the ingest stage first."
        )
    records = pd.read_parquet(records_path)

    # Reuse the M2 embedding cache (same configured backend).
    embedder = get_embedder(config)
    cache_path = config.paths.embeddings_dir / "embeddings.npz"
    clustering_texts = records[schema.TEXT_CLEAN].tolist()
    embeddings = embed_corpus(embedder, clustering_texts, cache_path, use_cache=config.embed.cache)

    print(f"[themes] Clustering {len(records):,} docs with '{config.themes.algorithm}'")
    labels, probabilities = cluster_embeddings(embeddings, clustering_texts, config)

    keywords = ctfidf_keywords(
        clustering_texts, labels, config.themes.top_n_keywords, config.themes.stopwords
    )
    reps = representative_docs(
        embeddings, labels, records[schema.TEXT].tolist(), config.themes.n_representative_docs
    )
    summary = build_summary(labels, keywords, reps)

    n_themes = int((~summary["is_outlier"]).sum())
    n_outliers = int(summary.loc[summary["is_outlier"], "n_docs"].sum())
    print(f"[themes] {n_themes} themes; {n_outliers:,} outlier docs "
          f"({100 * n_outliers / len(records):.1f}%)")

    # Persist assignments.
    assignments = pd.DataFrame(
        {
            schema.RECORD_ID: records[schema.RECORD_ID].values,
            "theme_id": labels,
            "probability": probabilities,
        }
    )
    config.paths.theme_assignments.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(config.paths.theme_assignments, index=False)
    summary.to_parquet(config.paths.theme_summary, index=False)

    # Prevalence over time.
    theme_ym = pd.DataFrame(
        {"theme_id": labels, schema.YEAR_MONTH: records[schema.YEAR_MONTH].values}
    )
    theme_ym = theme_ym[theme_ym["theme_id"] != OUTLIER_LABEL]  # outliers excluded from shares
    prevalence = prevalence_over_time(theme_ym)
    prevalence.to_parquet(config.paths.theme_prevalence, index=False)

    _write_themes_report(summary, config.paths.reports_dir / "themes.md")
    print("[themes] Wrote assignments, summary, prevalence, and reports/themes.md")
