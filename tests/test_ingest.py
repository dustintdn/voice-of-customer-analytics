"""M1 tests: ingest cleaning/transform functions verified against known values."""

from __future__ import annotations

import pandas as pd

from voc import schema
from voc.config import load_config
from voc.ingest import (
    _QualityTracker,
    canonical_form,
    clean_records,
    count_tokens,
    dedup_exact,
    dedup_near,
    derive_week,
    derive_year_month,
    filter_min_tokens,
    make_clustering_text,
    make_display_text,
    normalize_whitespace,
    parse_date_series,
)


# --- text normalization --------------------------------------------------- #
def test_normalize_whitespace_collapses_and_strips() -> None:
    assert normalize_whitespace("  hello   world \n\t ") == "hello world"


def test_normalize_whitespace_strips_control_chars() -> None:
    assert normalize_whitespace("bad\x00text\x07here") == "bad text here"


def test_count_tokens() -> None:
    assert count_tokens("one two three") == 3
    assert count_tokens("   ") == 0


def test_display_text_preserves_casing_handles_nan() -> None:
    assert make_display_text("Hello WORLD") == "Hello WORLD"
    assert make_display_text(float("nan")) == ""
    assert make_display_text(None) == ""


def test_clustering_text_lowercases_and_strips_redactions() -> None:
    # CFPB-style masks (XXXX) and date masks (XX/XX/XXXX) are removed for clustering.
    assert make_clustering_text("My name is XXXX on XX/XX/XXXX") == "my name is on"
    assert make_clustering_text("Hello WORLD") == "hello world"


def test_canonical_form_collides_case_and_punctuation_variants() -> None:
    a = canonical_form("Hello, WORLD!!!")
    b = canonical_form("hello world")
    assert a == b == "hello world"


# --- filtering ------------------------------------------------------------- #
def test_filter_min_tokens() -> None:
    df = pd.DataFrame({schema.TEXT_CLEAN: ["too short", "this one is long enough now", "x"]})
    out = filter_min_tokens(df, min_tokens=3)
    assert list(out[schema.TEXT_CLEAN]) == ["this one is long enough now"]


# --- dedup ----------------------------------------------------------------- #
def test_dedup_exact_keeps_first() -> None:
    df = pd.DataFrame({schema.TEXT: ["a complaint", "a complaint", "different"]})
    out = dedup_exact(df)
    assert list(out[schema.TEXT]) == ["a complaint", "different"]


def test_dedup_near_collapses_punctuation_variants() -> None:
    df = pd.DataFrame({schema.TEXT: ["The bank charged me.", "the bank charged me!!!", "new one"]})
    out = dedup_near(df)
    assert list(out[schema.TEXT]) == ["The bank charged me.", "new one"]


# --- date parsing ---------------------------------------------------------- #
def test_parse_date_series_handles_bad_values() -> None:
    parsed = parse_date_series(pd.Series(["2021-03-15", "not a date", "2022-12-01"]))
    assert parsed.isna().sum() == 1
    assert parsed.iloc[0] == pd.Timestamp("2021-03-15")


def test_derive_year_month_and_week_known_values() -> None:
    dates = pd.Series(pd.to_datetime(["2021-03-15", "2022-01-03"]))
    assert list(derive_year_month(dates)) == ["2021-03", "2022-01"]
    # 2021-03-15 is ISO week 11 of 2021; 2022-01-03 is ISO week 1 of 2022.
    assert list(derive_week(dates)) == ["2021-W11", "2022-W01"]


# --- end-to-end on the standardized frame ---------------------------------- #
def test_clean_records_pipeline() -> None:
    config = load_config("config/config.yaml")
    df = pd.DataFrame(
        {
            schema.TEXT: ["The bank charged me a fee", "The bank charged me a fee!", "short", "ok now here we go"],
            schema.TEXT_CLEAN: ["the bank charged me a fee", "the bank charged me a fee", "short", "ok now here we go"],
            "_raw_date": ["2021-05-01", "2021-06-01", "2021-07-01", "bad-date"],
            schema.CATEGORY: ["A", "A", "B", "B"],
        }
    )
    out = clean_records(df, config, _QualityTracker())
    # "short" dropped (min_tokens), near-dup "!" dropped, bad-date row dropped.
    assert len(out) == 1
    assert out.iloc[0][schema.TEXT] == "The bank charged me a fee"
    assert out.iloc[0][schema.YEAR_MONTH] == "2021-05"
    assert schema.WEEK in out.columns
