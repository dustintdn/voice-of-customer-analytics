# Data

## Primary dataset — CFPB Consumer Complaint Database

The pipeline is built around the U.S. Consumer Financial Protection Bureau's
public Consumer Complaint Database: hundreds of thousands of real, free-text
consumer complaint narratives with rich structured metadata (product category,
company, date, state, company response, and a `Consumer disputed?` /
`Timely response?` outcome).

### Download (bulk CSV)

The pipeline does **not** download data at run time. Fetch the bulk CSV once and
place it at the path configured in `config/config.yaml` (`paths.raw_csv`,
default `data/raw/complaints.csv`).

- Dataset homepage: https://www.consumerfinance.gov/data-research/consumer-complaints/
- Direct bulk CSV export: https://files.consumerfinance.gov/ccdb/complaints.csv.zip

```bash
mkdir -p data/raw
curl -L -o data/raw/complaints.csv.zip \
  https://files.consumerfinance.gov/ccdb/complaints.csv.zip
unzip -o data/raw/complaints.csv.zip -d data/raw/
# -> data/raw/complaints.csv
```

> Note: only complaints where the consumer opted to publish a narrative have
> text in the narrative column; the ingest stage filters out empty narratives.

## The committed sample

`make sample` builds a small (~3k row) sample at `data/sample/sample.csv`:

- **If the full raw CSV is present**, it takes a category-stratified slice.
- **If not**, it synthesizes a CFPB-shaped dataset so the repo runs and tests
  pass fully offline. This synthetic sample exists only for smoke tests and
  demos — the real run and the README's headline numbers come from the actual
  CFPB download.

Only `data/sample/` is committed. `data/raw/` and `data/processed/` are
gitignored.

## Dataset-agnostic by design

The pipeline is tied to **no** specific schema: `config/config.yaml`'s
`columns:` block maps logical roles (`text_column`, `date_column`,
`category_column`, `outcome_column`, `id_column`, grouping columns) onto
whatever the source CSV calls them. To run on a different corpus, change that
mapping.

### Alternative datasets

Any support-ticket / review corpus with **text + a categorical column + a
timestamp + an outcome column** works:

- Amazon product reviews
- Yelp Open Dataset
- Any support-ticket export with a resolution/escalation flag
