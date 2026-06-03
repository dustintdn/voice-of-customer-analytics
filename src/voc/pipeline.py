"""CLI orchestrator for the VoC pipeline.

Exposes per-stage subcommands plus an end-to-end ``run``. Stages are wired up
incrementally across milestones M1–M7; until a stage is implemented its
subcommand reports the milestone that will deliver it.

Usage:
    voc run --config config/config.yaml --dry-run
    voc ingest --config config/config.yaml
    python -m voc.pipeline run --dry-run
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from voc.config import Config, load_config
from voc.embed import run_embed
from voc.ingest import run_ingest
from voc.llm_extract import run_evaluation, run_extract
from voc.predict import run_predict
from voc.report import run_report
from voc.stats import run_stats
from voc.themes import run_themes


def _run_all(config: Config, dry_run: bool, full: bool) -> None:
    """Run every stage in order on the configured dataset."""
    run_ingest(config, full=full)
    run_embed(config)
    run_themes(config)
    run_extract(config, dry_run=dry_run, yes=True)
    run_predict(config)
    run_stats(config)
    run_report(config)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI with one subcommand per pipeline stage.

    ``--config`` and ``--dry-run`` are attached to each subcommand (via a shared
    parent parser) so they may follow the subcommand, e.g.
    ``voc run --config config/config.yaml --dry-run``.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the YAML config (default: config/config.yaml).",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a mock LLM (no network, no spend) for the extraction stage.",
    )
    common.add_argument(
        "--full",
        action="store_true",
        help="Ingest the full raw CSV instead of the committed sample.",
    )
    common.add_argument(
        "--yes",
        action="store_true",
        help="Skip the paid-run confirmation prompt for the extraction stage.",
    )

    parser = argparse.ArgumentParser(prog="voc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stages = ("ingest", "embed", "themes", "extract", "evaluate", "predict", "stats", "report", "run")
    for name in stages:
        sub.add_parser(name, parents=[common], help=f"Run the {name} stage.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    dispatch = {
        "ingest": lambda: run_ingest(config, full=args.full),
        "embed": lambda: run_embed(config),
        "themes": lambda: run_themes(config),
        "extract": lambda: run_extract(config, dry_run=args.dry_run, yes=args.yes),
        "evaluate": lambda: run_evaluation(config, dry_run=args.dry_run),
        "predict": lambda: run_predict(config),
        "stats": lambda: run_stats(config),
        "report": lambda: run_report(config),
        "run": lambda: _run_all(config, dry_run=args.dry_run, full=args.full),
    }

    try:
        dispatch[args.command]()
    except NotImplementedError as exc:
        # Expected during scaffolding/early milestones — fail clearly, not with a traceback.
        print(f"[voc] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
