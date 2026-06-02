"""Module 4 — LLM Structured Extraction (implemented at M4).

Converts free text into schema-validated structured fields via an LLM API,
with prompt variants, a gold-set evaluation harness, and a production run path
that is checkpointed, concurrency-bounded, retry-safe, and cost-capped
(SPEC §5.4 / §5.4.1). Supports a ``--dry-run`` mock for offline testing.
"""

from __future__ import annotations

from voc.config import Config


def run_extract(config: Config, dry_run: bool = False) -> None:
    """Run LLM structured extraction over the sampled records.

    Args:
        config: Loaded pipeline configuration.
        dry_run: If True, use a deterministic mock LLM (no network, no spend).
    """
    raise NotImplementedError("Module 4 (LLM Extraction) is implemented at milestone M4.")
