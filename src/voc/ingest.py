"""Module 1 — Ingest & Preprocess (implemented at M1).

Reads the raw CSV via the config column mapping, cleans/normalizes/dedups
narratives, parses dates, and writes ``records.parquet`` plus a data-quality
report.
"""

from __future__ import annotations

from voc.config import Config


def run_ingest(config: Config) -> None:
    """Ingest and preprocess raw records into the cleaned record table.

    Args:
        config: Loaded pipeline configuration.
    """
    raise NotImplementedError("Module 1 (Ingest) is implemented at milestone M1.")
