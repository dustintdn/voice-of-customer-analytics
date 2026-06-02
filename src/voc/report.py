"""Module 7 — Insight Report Generation (implemented at M7).

Auto-generates an executive-readable report (``reports/voc_insight_report.md``)
that leads with the "so what", carries uncertainty on every number, and
includes charts.
"""

from __future__ import annotations

from voc.config import Config


def run_report(config: Config) -> None:
    """Generate the executive insight report from stage outputs.

    Args:
        config: Loaded pipeline configuration.
    """
    raise NotImplementedError("Module 7 (Report) is implemented at milestone M7.")
