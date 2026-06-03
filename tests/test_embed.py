"""M2 tests: embedder backends, content-hash caching, and semantic search.

All offline: uses the deterministic HashingEmbedder, never the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from voc import schema
from voc.config import load_config
from voc.embed import (
    HashingEmbedder,
    build_index,
    content_hash,
    embed_corpus,
    semantic_search,
)


# --- HashingEmbedder ------------------------------------------------------- #
def test_hashing_embedder_shape_and_normalized() -> None:
    emb = HashingEmbedder(dim=64)
    vecs = emb.encode(["overdraft fee charged", "credit report dispute"])
    assert vecs.shape == (2, 64)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_hashing_embedder_deterministic() -> None:
    emb = HashingEmbedder(dim=64)
    a = emb.encode(["same text here"])
    b = emb.encode(["same text here"])
    np.testing.assert_allclose(a, b)


def test_hashing_embedder_similarity_tracks_shared_vocab() -> None:
    emb = HashingEmbedder(dim=512)
    v = emb.encode(
        [
            "the bank charged me overdraft fees",
            "the bank charged me overdraft fees again",
            "my mortgage escrow was miscalculated",
        ]
    )
    sim_related = float(v[0] @ v[1])
    sim_unrelated = float(v[0] @ v[2])
    assert sim_related > sim_unrelated


# --- caching --------------------------------------------------------------- #
class _CountingEmbedder:
    """Wraps HashingEmbedder and counts how many texts it actually encodes."""

    def __init__(self) -> None:
        self.tag = "counting-1"
        self.n_encoded = 0
        self._base = HashingEmbedder(dim=32)

    def encode(self, texts: list[str]) -> np.ndarray:
        self.n_encoded += len(texts)
        return self._base.encode(texts)


def test_embed_corpus_encodes_only_unique_uncached(tmp_path) -> None:
    cache = tmp_path / "emb.npz"
    texts = ["alpha beta", "gamma delta", "alpha beta"]  # 2 unique

    emb1 = _CountingEmbedder()
    m1 = embed_corpus(emb1, texts, cache)
    assert emb1.n_encoded == 2  # duplicate not re-encoded
    np.testing.assert_allclose(m1[0], m1[2])  # identical texts share embedding
    assert cache.exists()

    # Second run with a fresh embedder: everything served from disk cache.
    emb2 = _CountingEmbedder()
    m2 = embed_corpus(emb2, texts, cache)
    assert emb2.n_encoded == 0
    np.testing.assert_allclose(m1, m2)


def test_content_hash_changes_with_backend_tag() -> None:
    assert content_hash("st-modelA", "text") != content_hash("st-modelB", "text")
    assert content_hash("tag", "a") != content_hash("tag", "b")


# --- vector store + semantic search ---------------------------------------- #
def _toy_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            schema.RECORD_ID: [1, 2, 3, 4],
            schema.TEXT: [
                "The bank charged me overdraft fees on my checking account",
                "My mortgage escrow payment increased without explanation",
                "A debt collector keeps calling me about a debt I do not owe",
                "There is an inaccurate late payment on my credit report",
            ],
            schema.TEXT_CLEAN: [
                "the bank charged me overdraft fees on my checking account",
                "my mortgage escrow payment increased without explanation",
                "a debt collector keeps calling me about a debt i do not owe",
                "there is an inaccurate late payment on my credit report",
            ],
            schema.CATEGORY: ["Checking", "Mortgage", "Debt collection", "Credit reporting"],
            schema.YEAR_MONTH: ["2021-01", "2021-02", "2021-03", "2021-04"],
        }
    )


def test_build_index_and_semantic_search(tmp_path) -> None:
    config = load_config("config/config.yaml")
    config.paths.vector_store_dir = tmp_path / "chroma"
    embedder = HashingEmbedder(dim=512)

    records = _toy_records()
    embeddings = embedder.encode(records[schema.TEXT_CLEAN].tolist())
    build_index(config, records, embeddings)

    results = semantic_search(config, "overdraft fees on my bank account", k=4, embedder=embedder)
    assert len(results) == 4
    # The overdraft complaint should be the nearest neighbor.
    assert results[0]["record_id"] == "1"
    assert results[0]["category"] == "Checking"
    # Distances are sorted nearest-first.
    assert results[0]["distance"] <= results[-1]["distance"]
