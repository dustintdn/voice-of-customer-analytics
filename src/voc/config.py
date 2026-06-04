"""Typed configuration loader for the VoC pipeline.

The YAML at ``config/config.yaml`` is the single source of truth for paths,
model names, column mappings, and tunable parameters. This module parses it
into validated, typed objects so the rest of the pipeline never touches raw
dicts or hardcodes values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProjectConfig(BaseModel):
    """Top-level project metadata."""

    name: str
    seed: int = 42


class PathsConfig(BaseModel):
    """Filesystem locations for inputs, intermediates, and outputs.

    Paths are resolved to absolute paths against the repository root by
    :func:`load_config`, so downstream code can use them directly.
    """

    raw_csv: Path
    sample_csv: Path
    processed_dir: Path
    records_parquet: Path
    embeddings_dir: Path
    vector_store_dir: Path
    theme_assignments: Path
    theme_summary: Path
    theme_prevalence: Path
    llm_extractions: Path
    feature_table: Path
    predict_metrics: Path
    findings: Path
    reports_dir: Path


class ColumnsConfig(BaseModel):
    """Dataset-agnostic column mapping (SPEC §5.1)."""

    text_column: str
    date_column: str
    category_column: str
    subcategory_column: str | None = None
    outcome_column: str
    timely_column: str | None = None
    id_column: str
    group_columns: list[str] = Field(default_factory=list)


class IngestConfig(BaseModel):
    """Module 1 — ingest/preprocess parameters."""

    min_tokens: int = 5
    lowercase_for_clustering: bool = True
    dedup_exact: bool = True
    dedup_near: bool = True
    drop_null_outcome: bool = False


class UmapConfig(BaseModel):
    """UMAP dimensionality-reduction parameters for the BERTopic backend."""

    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0


class HdbscanConfig(BaseModel):
    """HDBSCAN clustering parameters for the BERTopic backend."""

    min_cluster_size: int = 50
    min_samples: int = 10


class EmbedConfig(BaseModel):
    """Module 2 — embedding parameters."""

    # ``model_name`` would otherwise collide with pydantic's ``model_`` namespace.
    model_config = ConfigDict(protected_namespaces=())

    backend: Literal["sentence_transformers", "hashing"] = "sentence_transformers"
    model_name: str
    batch_size: int = 64
    normalize: bool = True
    cache: bool = True
    hashing_dim: int = 256


class KmeansConfig(BaseModel):
    """Parameters for the offline KMeans theme-clustering backend."""

    n_clusters: int = 8
    outlier_quantile: float = 0.05


class ThemesConfig(BaseModel):
    """Module 3 — theme extraction parameters."""

    algorithm: Literal["bertopic", "kmeans"] = "bertopic"
    min_topic_size: int = 50
    nr_topics: int | Literal["auto"] = "auto"
    umap: UmapConfig = Field(default_factory=UmapConfig)
    hdbscan: HdbscanConfig = Field(default_factory=HdbscanConfig)
    kmeans: KmeansConfig = Field(default_factory=KmeansConfig)
    n_representative_docs: int = 5
    top_n_keywords: int = 10
    stopwords: str = "english"


class LlmConfig(BaseModel):
    """Module 4 — LLM extraction + cost/safety controls (SPEC §5.4.1)."""

    provider: str = "anthropic"
    model: str
    api_key_env: str = "ANTHROPIC_API_KEY"
    prompt_version: str = "v3"
    max_tokens: int = 512
    temperature: float = 0.0
    max_concurrency: int = 8
    max_records: int = 3000
    max_spend_usd: float = 2.0
    max_attempts: int = 5
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    input_cost_per_1k: float = 0.001
    output_cost_per_1k: float = 0.005
    controlled_vocab: list[str] = Field(default_factory=list)


class PredictConfig(BaseModel):
    """Module 5 — predictive model parameters."""

    target: str = "timely_response"
    positive_class_is_bad: bool = True
    test_split: Literal["time", "random"] = "time"
    test_fraction: float = 0.25
    model: Literal["lightgbm", "xgboost", "sklearn"] = "lightgbm"
    baseline: str = "logistic"
    threshold: float = 0.5
    embedding_pca_components: int = 16


class StatsConfig(BaseModel):
    """Module 6 — statistical reporting parameters."""

    confidence_level: float = 0.95
    margin_of_error: float = 0.03
    proportion_ci_method: Literal["wilson", "bootstrap"] = "wilson"
    bootstrap_iterations: int = 2000
    fdr_alpha: float = 0.05
    min_theme_count_for_test: int = 30


class ReportConfig(BaseModel):
    """Module 7 — insight report parameters."""

    output_md: Path
    output_html: Path | None = None
    top_n_themes: int = 10


class Config(BaseModel):
    """Root configuration object for the entire pipeline."""

    project: ProjectConfig
    paths: PathsConfig
    columns: ColumnsConfig
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    embed: EmbedConfig
    themes: ThemesConfig = Field(default_factory=ThemesConfig)
    llm: LlmConfig
    predict: PredictConfig = Field(default_factory=PredictConfig)
    stats: StatsConfig = Field(default_factory=StatsConfig)
    report: ReportConfig


def repo_root() -> Path:
    """Return the repository root (two levels above this file: src/voc/ -> root)."""
    return Path(__file__).resolve().parents[2]


def load_dotenv(path: str | Path | None = None) -> None:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into the environment.

    Used so secrets (e.g. ``ANTHROPIC_API_KEY``) can live in a gitignored
    ``.env`` instead of being exported each session. Already-set environment
    variables take precedence (an explicit ``export`` always wins), and a
    missing file is a no-op. Lines may be blank, ``# comments``, or
    ``export KEY=VALUE``; surrounding quotes are stripped.
    """
    env_path = Path(path) if path else (repo_root() / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, sep, value = line.partition("=")
        if not sep:
            continue
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _resolve_paths(paths: PathsConfig, root: Path) -> PathsConfig:
    """Resolve every path field to an absolute path against ``root``."""
    resolved = {
        field: (value if value.is_absolute() else (root / value))
        for field, value in paths
    }
    return PathsConfig(**resolved)


def load_config(config_path: str | Path = "config/config.yaml") -> Config:
    """Load and validate the pipeline configuration.

    Args:
        config_path: Path to the YAML config. Relative paths are resolved
            against the repository root.

    Returns:
        A validated :class:`Config`. All path fields are absolute.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    root = repo_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    config = Config(**raw)
    config.paths = _resolve_paths(config.paths, root)
    # Resolve report output paths too (they live outside PathsConfig).
    if not config.report.output_md.is_absolute():
        config.report.output_md = root / config.report.output_md
    if config.report.output_html and not config.report.output_html.is_absolute():
        config.report.output_html = root / config.report.output_html
    return config
