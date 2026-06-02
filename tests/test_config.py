"""M0 scaffold tests: the config loader and CLI wiring.

These verify the scaffold is sound (config parses into a typed object, paths
resolve absolute, the CLI parser builds). Stage logic is tested from M1 on.
"""

from __future__ import annotations

from voc.config import Config, load_config, repo_root
from voc.pipeline import build_parser


def test_load_config_returns_typed_config() -> None:
    config = load_config("config/config.yaml")
    assert isinstance(config, Config)
    assert config.project.name


def test_paths_are_absolute_and_rooted() -> None:
    config = load_config("config/config.yaml")
    root = repo_root()
    assert config.paths.raw_csv.is_absolute()
    assert config.paths.sample_csv.is_absolute()
    assert str(config.paths.records_parquet).startswith(str(root))
    assert config.report.output_md.is_absolute()


def test_column_mapping_present() -> None:
    config = load_config("config/config.yaml")
    cols = config.columns
    assert cols.text_column
    assert cols.date_column
    assert cols.category_column
    assert cols.outcome_column
    assert cols.id_column


def test_llm_cost_controls_present() -> None:
    """The cost/safety knobs (SPEC §5.4.1) must exist in config from the start."""
    llm = load_config("config/config.yaml").llm
    assert llm.max_spend_usd > 0
    assert llm.max_concurrency >= 1
    assert llm.max_attempts >= 1
    assert llm.controlled_vocab, "issue_category controlled vocabulary must be defined"


def test_cli_parser_builds_with_all_stages() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--dry-run"])
    assert args.command == "run"
    assert args.dry_run is True
