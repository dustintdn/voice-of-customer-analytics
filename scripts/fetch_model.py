#!/usr/bin/env python
"""Pre-download the embedding model so later runs are fully offline.

Run by ``make setup`` (and standalone) so the sentence-transformers weights are
cached locally before the first ``make run``. This is the one place that needs
network access; the pipeline itself never downloads at run time.

No-op when ``embed.backend`` is ``hashing`` (that backend needs no model).
"""

from __future__ import annotations

import argparse
import sys

from voc.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-fetch the embedding model.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if config.embed.backend != "sentence_transformers":
        print(f"[fetch-model] backend is '{config.embed.backend}'; no model to fetch.")
        return 0

    model_name = config.embed.model_name
    print(f"[fetch-model] Downloading/caching '{model_name}' (one-time, needs network)…")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "[fetch-model] sentence-transformers is not installed. "
            "Run `make setup` first, or set embed.backend: hashing.",
            file=sys.stderr,
        )
        return 1

    SentenceTransformer(model_name)  # triggers the download into the local cache
    print(f"[fetch-model] '{model_name}' is cached. Runs are now offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
