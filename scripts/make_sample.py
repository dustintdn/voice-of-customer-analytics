#!/usr/bin/env python
"""Build the small committed sample dataset (SPEC §3).

Behavior:
  * If the full raw CFPB CSV exists at ``paths.raw_csv``, take a stratified
    slice of ~N rows (by category, preserving the date range) and write it to
    ``paths.sample_csv``.
  * Otherwise, synthesize a CFPB-shaped dataset so the repo is runnable and
    testable offline without the multi-hundred-MB download. The synthetic data
    is clearly labeled and exists only for smoke tests / demos — the real run
    (and the README's headline numbers) uses the actual CFPB download.

Run via ``make sample`` or ``python scripts/make_sample.py``.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from voc.config import Config, load_config

N_SAMPLE = 3000

# Category -> (sub-product, narrative templates). Templates are deliberately
# distinguishable so the clustering stage (M3) has separable themes to find.
_TEMPLATES: dict[str, tuple[str, list[str]]] = {
    "Credit reporting": (
        "Credit reporting",
        [
            "There is an account on my credit report that does not belong to me and the bureau will not remove it.",
            "I disputed an inaccurate late payment on my credit report months ago and nothing has been corrected.",
            "My credit score dropped because of a hard inquiry I never authorized appearing on my report.",
        ],
    ),
    "Debt collection": (
        "Other debt",
        [
            "A debt collector keeps calling me multiple times a day about a debt I already paid off.",
            "I am being harassed by a collection agency for a debt that is not mine and they refuse to validate it.",
            "The collector threatened legal action over an old debt that is past the statute of limitations.",
        ],
    ),
    "Mortgage": (
        "Conventional home mortgage",
        [
            "My mortgage servicer misapplied my payment and is now charging me late fees that are not my fault.",
            "I requested a loan modification due to hardship and the servicer lost my paperwork three times.",
            "My escrow account was miscalculated and my monthly mortgage payment jumped without explanation.",
        ],
    ),
    "Credit card": (
        "General-purpose credit card",
        [
            "I see unauthorized charges on my credit card and the bank declined to investigate the fraud.",
            "My credit card interest rate was raised without proper notice and now my balance is unmanageable.",
            "I was charged an annual fee that I was told would be waived when I opened the card.",
        ],
    ),
    "Checking or savings account": (
        "Checking account",
        [
            "The bank charged me multiple overdraft fees in a single day on small transactions.",
            "My checking account was frozen without warning and I could not access my own money.",
            "I was charged maintenance fees on an account that was supposed to be free.",
        ],
    ),
    "Student loan": (
        "Federal student loan servicing",
        [
            "My student loan servicer applied my extra payments to interest instead of principal.",
            "I was told I qualified for an income-driven repayment plan but the servicer never processed it.",
            "My student loan payments are being reported as late even though I paid on time.",
        ],
    ),
}

_COMPANIES = [
    "Equifax", "Experian", "TransUnion", "Bank of the Region", "First National",
    "Capital Trust", "Unified Servicing LLC", "Metro Collections",
]
_STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]


def synthesize(columns, n: int = N_SAMPLE, seed: int = 42) -> pd.DataFrame:
    """Generate an offline, CFPB-shaped synthetic complaint dataset.

    Args:
        columns: The column-mapping config (provides destination column names).
        n: Number of rows to generate.
        seed: RNG seed for reproducibility.

    Returns:
        A DataFrame using the configured CFPB-style column names.
    """
    rng = np.random.default_rng(seed)
    categories = list(_TEMPLATES.keys())
    # Uneven category mix so prevalence/stat tests have something to chew on.
    weights = np.array([0.28, 0.22, 0.16, 0.14, 0.12, 0.08])
    weights = weights / weights.sum()

    dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 3, size=n), unit="D"
    )

    rows = []
    for i in range(n):
        cat = rng.choice(categories, p=weights)
        subproduct, templates = _TEMPLATES[cat]
        narrative = str(rng.choice(templates))
        # Inject a little lexical variation so near-dup dedup has work to do later.
        if rng.random() < 0.15:
            narrative = narrative + " Please help me resolve this issue."
        rows.append(
            {
                columns.id_column: 1_000_000 + i,
                columns.date_column: dates[i].strftime("%Y-%m-%d"),
                columns.category_column: cat,
                (columns.subcategory_column or "Sub-product"): subproduct,
                columns.text_column: narrative,
                "Company": str(rng.choice(_COMPANIES)),
                "State": str(rng.choice(_STATES)),
                columns.outcome_column: str(rng.choice(["Yes", "No"], p=[0.2, 0.8])),
                (columns.timely_column or "Timely response?"): str(
                    rng.choice(["Yes", "No"], p=[0.85, 0.15])
                ),
            }
        )
    return pd.DataFrame(rows)


def stratified_slice(df: pd.DataFrame, category_col: str, n: int, seed: int) -> pd.DataFrame:
    """Take a category-stratified slice of ~n rows from a large DataFrame."""
    frac = min(1.0, n / max(1, len(df)))
    sampled = (
        df.groupby(category_col, group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
        .reset_index(drop=True)
    )
    return sampled


def build_sample(config: Config) -> pd.DataFrame:
    """Produce the sample DataFrame from the raw CSV if present, else synthesize."""
    raw = config.paths.raw_csv
    if raw.exists():
        print(f"[sample] Slicing stratified sample from raw CSV: {raw}")
        df = pd.read_csv(raw)
        # Keep only rows with a non-empty narrative, mirroring the eventual ingest filter.
        df = df[df[config.columns.text_column].astype(str).str.strip().ne("")]
        return stratified_slice(
            df, config.columns.category_column, N_SAMPLE, config.project.seed
        )
    print("[sample] Raw CSV not found — synthesizing an offline CFPB-shaped sample.")
    return synthesize(config.columns, N_SAMPLE, config.project.seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the committed sample dataset.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    df = build_sample(config)

    out = config.paths.sample_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[sample] Wrote {len(df):,} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
