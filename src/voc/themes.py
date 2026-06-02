"""Module 3 — Theme Extraction (implemented at M3).

Clusters embeddings (BERTopic by default) into themes, summarizes each theme,
and computes theme prevalence over time for the statistical layer.
"""

from __future__ import annotations

from voc.config import Config


def run_themes(config: Config) -> None:
    """Extract themes and write assignments, summaries, and prevalence tables.

    Args:
        config: Loaded pipeline configuration.
    """
    raise NotImplementedError("Module 3 (Themes) is implemented at milestone M3.")
