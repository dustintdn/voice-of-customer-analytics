"""Module 2 — Embedding (implemented at M2).

Batch-encodes cleaned narratives with a sentence-transformers model, caches
embeddings by content hash, and loads them into a vector store exposing
``semantic_search``.
"""

from __future__ import annotations

from voc.config import Config


def run_embed(config: Config) -> None:
    """Embed cleaned records and populate the vector store.

    Args:
        config: Loaded pipeline configuration.
    """
    raise NotImplementedError("Module 2 (Embed) is implemented at milestone M2.")
