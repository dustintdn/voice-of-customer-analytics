"""Module 2 — Embedding & vector store.

Batch-encodes cleaned narratives, caches embeddings to disk keyed by a content
hash (so re-runs never re-embed unchanged data), and loads them into a ChromaDB
collection exposing :func:`semantic_search`.

Two embedder backends (selected by ``config.embed.backend``):
  * ``sentence_transformers`` — the real, default backend (``all-mpnet-base-v2``).
    Downloads model weights on first use; thereafter runs locally.
  * ``hashing`` — a deterministic, dependency-light feature-hashing embedder
    that needs no model and no network. Used by the test suite and available
    for fully-offline smoke runs. Cosine similarity still tracks shared
    vocabulary, so neighbors remain sensible.

Embeddings are always supplied to Chroma explicitly, so Chroma never loads its
own (network-dependent) embedding model.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from voc import schema
from voc.config import Config

# Disable ChromaDB telemetry before chromadb is imported (lazily, at runtime).
# The env var + Settings(anonymized_telemetry=False) disable it semantically;
# silencing the telemetry logger suppresses the noisy posthog-version error that
# chromadb 0.5.x emits while building the (disabled) capture call.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

_COLLECTION = "voc"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ADD_BATCH = 1000  # Chroma add() batch size


# --------------------------------------------------------------------------- #
# Embedder backends                                                           #
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    """Encodes a list of texts into an (n, dim) float32 matrix."""

    tag: str  # stable identity used in the cache key (changing it invalidates cache)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into an (n, dim) float32 matrix."""
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero rows are left as zeros."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic feature-hashing embedder (offline, no model, no network).

    Hashes word unigrams and bigrams into a fixed-dimension vector and
    L2-normalizes. Shared vocabulary -> higher cosine similarity, so it gives
    sensible neighbors for smoke tests without any downloaded model.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.tag = f"hashing-{dim}"

    def _hash(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into an L2-normalized (n, dim) float32 matrix."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            for tok in tokens:
                out[row, self._hash(tok)] += 1.0
            for a, b in zip(tokens, tokens[1:], strict=False):
                out[row, self._hash(f"{a}_{b}")] += 1.0
        return _l2_normalize(out)


class SentenceTransformerEmbedder:
    """Real backend wrapping ``sentence-transformers`` (lazy import)."""

    def __init__(self, model_name: str, batch_size: int = 64, normalize: bool = True) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.tag = f"st-{model_name}"
        self._model = None  # loaded lazily on first encode

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on optional heavy dep
                raise ImportError(
                    "sentence-transformers is required for the 'sentence_transformers' "
                    'embed backend. Install the ML extra (`pip install -e ".[ml]"`) or set '
                    "embed.backend: hashing for the offline backend."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts with the sentence-transformers model into a float32 matrix."""
        model = self._ensure_model()
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


def get_embedder(config: Config) -> Embedder:
    """Construct the embedder selected by ``config.embed.backend``."""
    if config.embed.backend == "hashing":
        return HashingEmbedder(dim=config.embed.hashing_dim)
    return SentenceTransformerEmbedder(
        model_name=config.embed.model_name,
        batch_size=config.embed.batch_size,
        normalize=config.embed.normalize,
    )


# --------------------------------------------------------------------------- #
# Content-hash embedding cache                                                #
# --------------------------------------------------------------------------- #
def content_hash(tag: str, text: str) -> str:
    """Stable cache key for a (embedder, text) pair."""
    h = hashlib.sha1()
    h.update(tag.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    """Load the embedding cache (hash -> vector); empty dict if absent."""
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    return dict(zip(data["hashes"].tolist(), data["vectors"], strict=False))


def _save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes = np.array(list(cache.keys()))
    vectors = np.stack(list(cache.values())) if cache else np.empty((0, 0), dtype=np.float32)
    np.savez(path, hashes=hashes, vectors=vectors)


def embed_corpus(
    embedder: Embedder, texts: list[str], cache_path: Path, use_cache: bool = True
) -> np.ndarray:
    """Embed ``texts``, reusing cached vectors and encoding only what's missing.

    Returns the embedding matrix in the original ``texts`` order.
    """
    hashes = [content_hash(embedder.tag, t) for t in texts]
    cache = _load_cache(cache_path) if use_cache else {}

    # Encode each unique uncached text exactly once.
    missing_text_by_hash: dict[str, str] = {}
    for h, t in zip(hashes, texts, strict=False):
        if h not in cache and h not in missing_text_by_hash:
            missing_text_by_hash[h] = t

    if missing_text_by_hash:
        print(
            f"[embed] Encoding {len(missing_text_by_hash):,} unique texts "
            f"({len(texts):,} total; rest served from cache)"
        )
        miss_hashes = list(missing_text_by_hash)
        new_vecs = embedder.encode([missing_text_by_hash[h] for h in miss_hashes])
        for h, vec in zip(miss_hashes, new_vecs, strict=False):
            cache[h] = vec
        if use_cache:
            # Drop stale entries from a prior embedder with a different output dim
            # (different model → different tag → different hashes → they'd never be
            # retrieved, but mixing dims in the same dict breaks np.stack on save).
            expected_dim = new_vecs.shape[1]
            cache = {h: v for h, v in cache.items() if v.shape[0] == expected_dim}
            _save_cache(cache_path, cache)
    else:
        print(f"[embed] All {len(texts):,} embeddings served from cache")

    return np.stack([cache[h] for h in hashes])


# --------------------------------------------------------------------------- #
# Vector store (ChromaDB)                                                     #
# --------------------------------------------------------------------------- #
def _client(config: Config):
    import chromadb
    from chromadb.config import Settings

    config.paths.vector_store_dir.mkdir(parents=True, exist_ok=True)
    # Disable telemetry: it attempts a network call and must stay offline.
    return chromadb.PersistentClient(
        path=str(config.paths.vector_store_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def build_index(config: Config, records: pd.DataFrame, embeddings: np.ndarray) -> None:
    """(Re)build the Chroma collection from records + their embeddings."""
    client = _client(config)
    with contextlib.suppress(Exception):
        client.delete_collection(_COLLECTION)  # rebuild fresh; avoids stale vectors
    collection = client.create_collection(
        name=_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    ids = records[schema.RECORD_ID].astype(str).tolist()
    documents = records[schema.TEXT].tolist()
    metadatas = [
        {
            schema.CATEGORY: str(row.get(schema.CATEGORY, "")),
            schema.YEAR_MONTH: str(row.get(schema.YEAR_MONTH, "")),
        }
        for _, row in records.iterrows()
    ]

    for start in range(0, len(ids), _ADD_BATCH):
        end = start + _ADD_BATCH
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    print(f"[embed] Indexed {len(ids):,} records into Chroma at {config.paths.vector_store_dir}")


def semantic_search(
    config: Config, query: str, k: int = 5, embedder: Embedder | None = None
) -> list[dict]:
    """Return the ``k`` nearest records to ``query`` from the vector store.

    Args:
        config: Loaded pipeline configuration.
        query: Free-text query.
        k: Number of neighbors to return.
        embedder: Optional embedder override (defaults to the configured backend).
            Must match the backend used to build the index.

    Returns:
        A list of dicts with ``record_id``, ``text``, ``distance``, ``category``,
        and ``year_month``, ordered nearest-first.
    """
    embedder = embedder or get_embedder(config)
    qvec = embedder.encode([query])[0]

    client = _client(config)
    collection = client.get_collection(_COLLECTION)
    res = collection.query(query_embeddings=[qvec.tolist()], n_results=k)

    results: list[dict] = []
    for rid, doc, dist, meta in zip(
        res["ids"][0], res["documents"][0], res["distances"][0], res["metadatas"][0], strict=False
    ):
        results.append(
            {
                "record_id": rid,
                "text": doc,
                "distance": float(dist),
                schema.CATEGORY: meta.get(schema.CATEGORY, ""),
                schema.YEAR_MONTH: meta.get(schema.YEAR_MONTH, ""),
            }
        )
    return results


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run_embed(config: Config, embedder: Embedder | None = None) -> None:
    """Embed cleaned records and populate the vector store.

    Args:
        config: Loaded pipeline configuration.
        embedder: Optional embedder override (defaults to the configured backend).
    """
    records_path = config.paths.records_parquet
    if not records_path.exists():
        raise FileNotFoundError(
            f"Cleaned records not found: {records_path}. Run the ingest stage first."
        )

    records = pd.read_parquet(records_path)
    embedder = embedder or get_embedder(config)
    print(f"[embed] Backend: {embedder.tag} | records: {len(records):,}")

    cache_path = config.paths.embeddings_dir / "embeddings.npz"
    texts = records[schema.TEXT_CLEAN].tolist()
    embeddings = embed_corpus(embedder, texts, cache_path, use_cache=config.embed.cache)

    build_index(config, records, embeddings)
