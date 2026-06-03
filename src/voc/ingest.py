"""Module 1 — Ingest & Preprocess.

Reads the raw CSV via the config column mapping, cleans/normalizes/dedups
narratives, parses dates, and writes ``records.parquet`` plus a data-quality
report (``reports/data_quality.md``).

Normalization policy (SPEC §5.1): we keep **two** text columns.
  * ``text``       — the original narrative with casing/punctuation preserved.
                     This is what the LLM stage (M4) sees, because casing and
                     punctuation carry signal for sentiment/severity.
  * ``text_clean`` — lowercased, control-char-stripped, whitespace-collapsed,
                     redaction-mask-stripped. This feeds embeddings/clustering,
                     where surface form is noise.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from voc import schema
from voc.config import Config

# Control characters except common whitespace (tab/newline are collapsed anyway).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
# CFPB redaction masks: standalone runs of "x" ("XXXX") and masked date/number
# groups joined by separators ("XX/XX/XXXX"). Removed wholesale for clustering.
_REDACTION_RE = re.compile(r"\bx{2,}(?:[\s/.\-]+x{2,})*\b", re.IGNORECASE)
# Non-alphanumeric (keep spaces) — used only to build the near-dup key.
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


# --------------------------------------------------------------------------- #
# Pure text transforms (unit-tested)                                          #
# --------------------------------------------------------------------------- #
def normalize_whitespace(text: str) -> str:
    """Strip control characters and collapse all whitespace runs to a single space."""
    text = _CONTROL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def count_tokens(text: str) -> int:
    """Count whitespace-delimited tokens in ``text``."""
    return len(text.split())


def make_display_text(text: object) -> str:
    """Build the LLM-facing text: original casing, only whitespace/control cleanup."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return normalize_whitespace(str(text))


def make_clustering_text(text: object) -> str:
    """Build the clustering/embedding text: lowercased, redaction-masks removed."""
    cleaned = make_display_text(text).lower()
    cleaned = _REDACTION_RE.sub(" ", cleaned)
    return normalize_whitespace(cleaned)


def canonical_form(text: object) -> str:
    """Aggressive canonical form used only as a near-duplicate key.

    Casefolds, drops all non-alphanumeric characters, and collapses whitespace,
    so variants differing only in casing/punctuation/spacing collide.
    """
    cleaned = make_display_text(text).casefold()
    cleaned = _NON_ALNUM_RE.sub(" ", cleaned)
    return normalize_whitespace(cleaned)


def parse_date_series(series: pd.Series) -> pd.Series:
    """Parse a column of date-like strings into datetime64 (unparseable -> NaT)."""
    return pd.to_datetime(series, errors="coerce")


def derive_year_month(dates: pd.Series) -> pd.Series:
    """Derive a lexically-sortable ``YYYY-MM`` string column from datetimes."""
    return dates.dt.strftime("%Y-%m")


def derive_week(dates: pd.Series) -> pd.Series:
    """Derive an ISO ``YYYY-Www`` week string column from datetimes."""
    iso = dates.dt.isocalendar()
    return iso["year"].astype("Int64").astype(str) + "-W" + iso["week"].astype("Int64").astype(
        str
    ).str.zfill(2)


# --------------------------------------------------------------------------- #
# DataFrame-level filters (unit-tested)                                       #
# --------------------------------------------------------------------------- #
def filter_min_tokens(df: pd.DataFrame, min_tokens: int) -> pd.DataFrame:
    """Drop rows whose cleaned text has fewer than ``min_tokens`` tokens."""
    keep = df[schema.TEXT_CLEAN].map(count_tokens) >= min_tokens
    return df[keep].reset_index(drop=True)


def dedup_exact(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with identical display text, keeping the first occurrence."""
    return df.drop_duplicates(subset=schema.TEXT, keep="first").reset_index(drop=True)


def dedup_near(df: pd.DataFrame) -> pd.DataFrame:
    """Drop near-duplicates (same :func:`canonical_form`), keeping the first.

    A simple O(n) hashing approach (SPEC §5.1 "keep it simple"): two narratives
    differing only by casing, punctuation, or whitespace map to one key and are
    deduplicated. Heavier MinHash/LSH near-dup is a documented future option.
    """
    key = df[schema.TEXT].map(canonical_form)
    return df[~key.duplicated(keep="first")].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def _standardize_columns(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Map source columns onto canonical internal names; carry group columns through."""
    cols = config.columns
    out = pd.DataFrame()
    out[schema.RECORD_ID] = df[cols.id_column]
    out[schema.TEXT] = df[cols.text_column].map(make_display_text)
    out[schema.TEXT_CLEAN] = df[cols.text_column].map(make_clustering_text)
    out["_raw_date"] = df[cols.date_column]
    out[schema.CATEGORY] = df[cols.category_column]
    if cols.subcategory_column and cols.subcategory_column in df.columns:
        out[schema.SUBCATEGORY] = df[cols.subcategory_column]
    if cols.outcome_column in df.columns:
        out[schema.OUTCOME] = df[cols.outcome_column]
    if cols.timely_column and cols.timely_column in df.columns:
        out[schema.TIMELY] = df[cols.timely_column]
    for gcol in cols.group_columns:
        if gcol in df.columns:
            out[gcol] = df[gcol]
    return out


class _QualityTracker:
    """Accumulates row counts across the cleaning pipeline for the QA report."""

    def __init__(self) -> None:
        self.steps: list[tuple[str, int, int]] = []  # (label, before, after)

    def record(self, label: str, before: int, after: int) -> None:
        self.steps.append((label, before, after))


def _write_quality_report(
    df: pd.DataFrame, tracker: _QualityTracker, source: Path, out_path: Path, config: Config
) -> None:
    """Render the data-quality markdown report (SPEC §5.1)."""
    lines: list[str] = ["# Data Quality Report", ""]
    lines.append(f"- **Source:** `{source}`")
    if tracker.steps:
        lines.append(f"- **Rows in:** {tracker.steps[0][1]:,}")
    lines.append(f"- **Rows out:** {len(df):,}")
    if tracker.steps and tracker.steps[0][1]:
        pct = 100.0 * len(df) / tracker.steps[0][1]
        lines.append(f"- **Retained:** {pct:.1f}%")
    lines += ["", "## Filter pipeline", "", "| Step | Rows before | Rows after | Dropped | % dropped |", "|---|---:|---:|---:|---:|"]
    for label, before, after in tracker.steps:
        dropped = before - after
        pct = 100.0 * dropped / before if before else 0.0
        lines.append(f"| {label} | {before:,} | {after:,} | {dropped:,} | {pct:.1f}% |")

    lines += ["", "## Null rates (output)", "", "| Column | Null % |", "|---|---:|"]
    for col in df.columns:
        null_pct = 100.0 * df[col].isna().mean()
        lines.append(f"| {col} | {null_pct:.1f}% |")

    if schema.DATE in df.columns and df[schema.DATE].notna().any():
        lines += [
            "",
            "## Date range",
            "",
            f"- **Earliest:** {df[schema.DATE].min().date()}",
            f"- **Latest:** {df[schema.DATE].max().date()}",
            f"- **Distinct months:** {df[schema.YEAR_MONTH].nunique()}",
        ]

    if schema.CATEGORY in df.columns:
        counts = df[schema.CATEGORY].value_counts()
        lines += ["", "## Category distribution", "", "| Category | Count | Share |", "|---|---:|---:|"]
        for cat, cnt in counts.items():
            lines.append(f"| {cat} | {cnt:,} | {100.0 * cnt / len(df):.1f}% |")

    lines += [
        "",
        "## Normalization notes",
        "",
        f"- Minimum tokens to keep a record: **{config.ingest.min_tokens}**.",
        "- `text` preserves original casing/punctuation (consumed by the LLM stage).",
        "- `text_clean` is lowercased, control-char-stripped, whitespace-collapsed, "
        "and has redaction masks (e.g. `XXXX`) removed (consumed by embeddings/clustering).",
        f"- Exact dedup on `text`: **{config.ingest.dedup_exact}**; "
        f"near-dup on canonical form: **{config.ingest.dedup_near}**.",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def clean_records(df: pd.DataFrame, config: Config, tracker: _QualityTracker) -> pd.DataFrame:
    """Run the full cleaning pipeline on a standardized DataFrame.

    Order: filter empty/short -> exact dedup -> near dedup -> parse dates ->
    drop unparseable dates -> derive time columns.
    """
    n = len(df)
    df = filter_min_tokens(df, config.ingest.min_tokens)
    tracker.record(f"min_tokens >= {config.ingest.min_tokens}", n, len(df))

    if config.ingest.dedup_exact:
        n = len(df)
        df = dedup_exact(df)
        tracker.record("exact dedup", n, len(df))

    if config.ingest.dedup_near:
        n = len(df)
        df = dedup_near(df)
        tracker.record("near dedup", n, len(df))

    df[schema.DATE] = parse_date_series(df["_raw_date"])
    n = len(df)
    df = df[df[schema.DATE].notna()].reset_index(drop=True)
    tracker.record("valid date", n, len(df))

    df[schema.YEAR_MONTH] = derive_year_month(df[schema.DATE])
    df[schema.WEEK] = derive_week(df[schema.DATE])
    return df.drop(columns=["_raw_date"])


def run_ingest(config: Config, full: bool = False) -> None:
    """Ingest and preprocess raw records into the cleaned record table.

    Args:
        config: Loaded pipeline configuration.
        full: If True, read the full raw CSV (``paths.raw_csv``); otherwise read
            the committed sample (``paths.sample_csv``).
    """
    source = config.paths.raw_csv if full else config.paths.sample_csv
    if not source.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {source}. Run `make sample` (sample) "
            "or download the CFPB CSV (see data/README.md) for --full."
        )

    print(f"[ingest] Reading {source}")
    raw = pd.read_csv(source, low_memory=False)
    tracker = _QualityTracker()
    tracker.record("raw rows", len(raw), len(raw))

    standardized = _standardize_columns(raw, config)
    cleaned = clean_records(standardized, config, tracker)

    out_parquet = config.paths.records_parquet
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out_parquet, index=False)
    print(f"[ingest] Wrote {len(cleaned):,} cleaned records to {out_parquet}")

    report_path = config.paths.reports_dir / "data_quality.md"
    _write_quality_report(cleaned, tracker, source, report_path, config)
    print(f"[ingest] Wrote data-quality report to {report_path}")
