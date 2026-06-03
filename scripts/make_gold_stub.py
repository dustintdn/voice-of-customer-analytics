#!/usr/bin/env python
"""Create a gold-set labeling stub for the LLM evaluation harness (SPEC §5.4).

Default: sample N records and write a CSV with the text plus EMPTY label columns
for a human to fill in (the real gold set the README's accuracy numbers come from).

``--auto-label``: fill the labels using the synthetic sample's ground truth
(Product -> issue_category) plus simple rules, producing a committed gold set so
the evaluation harness is runnable fully offline. This is for the offline demo
only; a real run uses a human-labeled gold set.

Usage:
    python scripts/make_gold_stub.py --n 80                 # empty stub for labeling
    python scripts/make_gold_stub.py --n 80 --auto-label    # offline demo gold
"""

from __future__ import annotations

import argparse

import pandas as pd

from voc.config import load_config
from voc.llm_extract import _ESCALATION_WORDS

# Ground-truth product -> issue_category mapping for the synthetic offline gold.
_PRODUCT_TO_CATEGORY = {
    "Credit reporting": "credit_reporting_error",
    "Debt collection": "debt_collection_harassment",
    "Mortgage": "loan_servicing",
    "Credit card": "fees_and_charges",
    "Checking or savings account": "account_access",
    "Student loan": "loan_servicing",
}

_LABEL_COLUMNS = [
    "sentiment_label",
    "sentiment_score",
    "issue_category",
    "severity",
    "is_actionable",
]


def _auto_label(row: pd.Series, text_col: str, category_col: str) -> dict:
    text = str(row[text_col]).lower()
    escalated = any(w in text for w in _ESCALATION_WORDS)
    return {
        "sentiment_label": "negative",
        "sentiment_score": -0.9 if escalated else -0.6,
        "issue_category": _PRODUCT_TO_CATEGORY.get(str(row[category_col]), "other"),
        "severity": "high" if escalated else "medium",
        "is_actionable": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a gold-set labeling stub.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--n", type=int, default=80, help="Number of records to sample.")
    parser.add_argument("--auto-label", action="store_true", help="Fill labels from ground truth (offline demo).")
    parser.add_argument("--out", default=None, help="Output CSV path (defaults next to the sample).")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    src = config.paths.sample_csv
    if not src.exists():
        print(f"Sample not found: {src}. Run `make sample` first.")
        return 1

    df = pd.read_csv(src)
    text_col = config.columns.text_column
    category_col = config.columns.category_column
    id_col = config.columns.id_column

    sample = df.sample(n=min(args.n, len(df)), random_state=config.project.seed).reset_index(drop=True)
    out = pd.DataFrame({"record_id": sample[id_col], "text": sample[text_col]})

    if args.auto_label:
        labels = sample.apply(lambda r: _auto_label(r, text_col, category_col), axis=1, result_type="expand")
        out = pd.concat([out, labels], axis=1)
        default_name = "gold.csv"
    else:
        for col in _LABEL_COLUMNS:
            out[col] = ""
        default_name = "gold_stub.csv"

    out_path = args.out or str(src.parent / default_name)
    out.to_csv(out_path, index=False)
    kind = "auto-labeled offline gold" if args.auto_label else "empty labeling stub"
    print(f"[gold] Wrote {len(out):,}-row {kind} to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
