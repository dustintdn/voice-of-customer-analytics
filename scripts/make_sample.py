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

The synthesizer composes narratives from category-specific fragment banks with
filled-in slots (amounts, counts, durations, company names) so that:
  * narratives are diverse enough to survive near-dup dedup (M1), yet
  * remain cleanly separable into themes by category (M3), and
  * carry a learnable relationship between escalation language / category and
    the behavioral outcome, so the predictive model (M5) has real signal.

Run via ``make sample`` or ``python scripts/make_sample.py``.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from voc.config import Config, load_config

N_SAMPLE = 3000

# Per-category fragment banks. {slots} are filled at generation time. Each
# category's vocabulary is distinct so themes stay separable; `dispute_rate` is
# the baseline P(consumer disputed) used to build a learnable outcome.
_CATEGORIES: dict[str, dict] = {
    "Credit reporting": {
        "subproduct": "Credit reporting",
        "dispute_rate": 0.30,
        "openers": [
            "I have been trying to fix my credit report for {months} months without success.",
            "There is an account on my credit report that I do not recognize.",
            "My credit score dropped by {count} points last month.",
            "An item on my credit report is reporting inaccurate information.",
            "I submitted a dispute to the credit bureau and heard nothing back.",
        ],
        "details": [
            "{company} listed a late payment I never made and it is hurting my ability to get a loan.",
            "I have disputed this {count} times but the bureau keeps verifying it without investigating.",
            "A hard inquiry I never authorized appears on my report and no one will remove it.",
            "The account shows a balance of {amount} that was already paid in full.",
            "Fraudulent accounts from identity theft are still showing on my report.",
            "They are reporting an account as open when it was closed {months} months ago.",
        ],
    },
    "Debt collection": {
        "subproduct": "Other debt",
        "dispute_rate": 0.22,
        "openers": [
            "A debt collector has been calling me {count} times a day.",
            "I keep receiving collection notices for a debt that is not mine.",
            "{company} is attempting to collect a debt I do not owe.",
            "I am being contacted about a debt that is past the statute of limitations.",
        ],
        "details": [
            "They refuse to send written validation of the {amount} they claim I owe.",
            "The calls continue even after I asked them in writing to stop.",
            "They threatened to garnish my wages over a {amount} balance.",
            "They are contacting my family and coworkers about this debt.",
            "I already paid this debt {months} months ago but it keeps getting resold.",
        ],
    },
    "Mortgage": {
        "subproduct": "Conventional home mortgage",
        "dispute_rate": 0.18,
        "openers": [
            "My mortgage servicer has mishandled my account for {months} months.",
            "I applied for a loan modification due to financial hardship.",
            "My monthly mortgage payment increased unexpectedly.",
            "{company} transferred my mortgage and now my payments are not being applied.",
        ],
        "details": [
            "They misapplied my payment of {amount} and charged me a late fee.",
            "My escrow was miscalculated and my payment jumped by {amount}.",
            "They lost my modification paperwork {count} separate times.",
            "I was placed in forbearance without my consent.",
            "They are threatening foreclosure even though I am current on payments.",
        ],
    },
    "Credit card": {
        "subproduct": "General-purpose credit card",
        "dispute_rate": 0.16,
        "openers": [
            "I found unauthorized charges on my credit card.",
            "My credit card company raised my interest rate without notice.",
            "I was charged a fee on my {company} credit card that I do not recognize.",
            "I disputed a transaction on my card statement.",
        ],
        "details": [
            "There are {count} fraudulent charges totaling {amount} that the bank will not refund.",
            "My APR jumped and now I owe {amount} in interest alone.",
            "They charged me an annual fee of {amount} after promising to waive it.",
            "My credit limit was cut without explanation, hurting my score.",
            "The dispute has been open for {months} months with no resolution.",
        ],
    },
    "Checking or savings account": {
        "subproduct": "Checking account",
        "dispute_rate": 0.12,
        "openers": [
            "My bank charged me multiple overdraft fees.",
            "My checking account was frozen without warning.",
            "I was charged unexpected fees on my account.",
            "{company} closed my account without notifying me.",
        ],
        "details": [
            "They charged me {count} overdraft fees of {amount} each in a single day.",
            "I could not access {amount} of my own money for {days} days.",
            "They charged a monthly maintenance fee on an account advertised as free.",
            "A deposit of {amount} was held for {days} days with no explanation.",
            "Funds were withdrawn due to an error they will not reverse.",
        ],
    },
    "Student loan": {
        "subproduct": "Federal student loan servicing",
        "dispute_rate": 0.20,
        "openers": [
            "My student loan servicer has mismanaged my payments.",
            "I was told I qualified for an income-driven repayment plan.",
            "My student loans were transferred to a new servicer.",
            "{company} is reporting my student loan payments incorrectly.",
        ],
        "details": [
            "They applied my extra payment of {amount} to interest instead of principal.",
            "My income-driven application has been pending for {months} months.",
            "They are reporting me as late even though I paid {amount} on time.",
            "My loans were placed in forbearance without my request.",
            "I cannot get a straight answer about my loan balance of {amount}.",
        ],
    },
}

_CLOSERS = [
    "Please help me resolve this issue.",
    "I have tried to resolve this directly with no success.",
    "This has caused me significant financial stress.",
    "I am requesting that this be corrected immediately.",
    "No one at the company will give me a clear answer.",
    "I would like a full investigation into this matter.",
]

# Escalation language raises the modeled probability of a bad outcome, giving
# the predictive model a learnable text-derived signal.
_ESCALATIONS = [
    "I am prepared to take legal action if this is not resolved.",
    "I have already contacted an attorney about this.",
    "This is the third complaint I have had to file.",
]

_COMPANIES = [
    "Equifax", "Experian", "TransUnion", "Bank of the Region", "First National",
    "Capital Trust", "Unified Servicing LLC", "Metro Collections",
]
_STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]

# Escalation's effect on the outcome is category-dependent (a non-linear
# interaction, with a sign flip for Mortgage). This gives the gradient-boosting
# model a genuine edge over an additive logistic baseline — the relationship is
# not captured by category + escalation main effects alone.
_ESC_DISPUTE_EFFECT = {
    "Credit reporting": 0.40,
    "Debt collection": 0.35,
    "Mortgage": -0.10,
    "Credit card": 0.10,
    "Checking or savings account": 0.05,
    "Student loan": 0.20,
}
_ESC_TIMELY_EFFECT = {  # added to P(not timely) when escalated
    "Credit reporting": 0.35,
    "Debt collection": 0.30,
    "Mortgage": 0.00,
    "Credit card": 0.05,
    "Checking or savings account": 0.05,
    "Student loan": 0.20,
}


def _fill(template: str, rng: np.random.Generator) -> str:
    """Fill slot placeholders in a fragment with random, realistic values."""
    return (
        template.replace("{company}", str(rng.choice(_COMPANIES)))
        .replace("{amount}", f"${int(rng.integers(50, 9000)):,}")
        .replace("{count}", str(int(rng.integers(2, 12))))
        .replace("{months}", str(int(rng.integers(1, 24))))
        .replace("{days}", str(int(rng.integers(2, 30))))
    )


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
    categories = list(_CATEGORIES.keys())
    # Uneven category mix so prevalence/stat tests have something to chew on.
    weights = np.array([0.28, 0.22, 0.16, 0.14, 0.12, 0.08])
    weights = weights / weights.sum()

    dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 3, size=n), unit="D"
    )

    rows = []
    for i in range(n):
        cat = str(rng.choice(categories, p=weights))
        spec = _CATEGORIES[cat]
        escalated = rng.random() < 0.22

        parts = [_fill(str(rng.choice(spec["openers"])), rng),
                 _fill(str(rng.choice(spec["details"])), rng)]
        if escalated:
            parts.append(str(rng.choice(_ESCALATIONS)))
        parts.append(str(rng.choice(_CLOSERS)))
        narrative = " ".join(parts)

        # Learnable outcome: per-category base rate plus a category-dependent
        # escalation interaction (non-additive — rewards gradient boosting).
        dispute_prob = min(0.95, max(0.02, spec["dispute_rate"] + (_ESC_DISPUTE_EFFECT[cat] if escalated else 0.0)))
        disputed = "Yes" if rng.random() < dispute_prob else "No"
        untimely_prob = min(0.90, max(0.02, 0.08 + (_ESC_TIMELY_EFFECT[cat] if escalated else 0.0)))
        timely = "No" if rng.random() < untimely_prob else "Yes"

        rows.append(
            {
                columns.id_column: 1_000_000 + i,
                columns.date_column: dates[i].strftime("%Y-%m-%d"),
                columns.category_column: cat,
                (columns.subcategory_column or "Sub-product"): spec["subproduct"],
                columns.text_column: narrative,
                "Company": str(rng.choice(_COMPANIES)),
                "State": str(rng.choice(_STATES)),
                columns.outcome_column: disputed,
                (columns.timely_column or "Timely response?"): timely,
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
