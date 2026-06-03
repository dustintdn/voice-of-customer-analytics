"""Canonical internal column names for the cleaned record table.

The ingest stage maps the source dataset's columns (named per the CFPB schema,
or whatever ``config.columns`` specifies) onto these stable internal names.
Every downstream stage reads these constants rather than dataset-specific
column names, so the rest of the pipeline is decoupled from the source schema.
"""

from __future__ import annotations

# Core standardized columns written to data/processed/records.parquet.
RECORD_ID = "record_id"
TEXT = "text"               # original narrative, casing preserved — for the LLM stage
TEXT_CLEAN = "text_clean"   # normalized/lowercased — for embeddings & clustering
DATE = "date"               # parsed datetime64
YEAR_MONTH = "year_month"   # "YYYY-MM" string, lexically sortable
WEEK = "week"               # ISO "YYYY-Www" string
CATEGORY = "category"
SUBCATEGORY = "subcategory"
OUTCOME = "outcome"         # raw outcome label (e.g. CFPB "Consumer disputed?")
TIMELY = "timely"           # raw timely-response label, if mapped

# Group/metadata columns (company, state, …) are carried through under their
# original source names; downstream code reads config.columns.group_columns.
