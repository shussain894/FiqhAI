"""
rerank.py

Reranks a list of retrieved chunks using a cross-encoder model.

Unlike the bi-encoder used for initial retrieval (which embeds query and
document separately), a cross-encoder reads the query and each document
together — much more accurate, but too slow to run over the whole index.

The pattern here is:
  1. Retrieve a larger candidate set from ChromaDB (e.g. top-10)
  2. Rerank with the cross-encoder
  3. Keep only the best top_k for generation
"""

import os

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def load_reranker(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    """
    Loads the cross-encoder reranker model.
    Call once at startup and reuse — model download is cached locally.
    """
    return CrossEncoder(model_name)


def rerank(
    query: str,
    results: list[dict],
    reranker: CrossEncoder,
    top_k: int = 5
) -> list[dict]:
    """
    Reranks retrieved chunks by cross-encoder relevance score.

    Args:
        query:    The original user query (not the rewritten retrieval query).
        results:  Chunks from ChromaDB, each with a "text" field.
        reranker: Loaded CrossEncoder model.
        top_k:    Number of chunks to return after reranking.

    Returns:
        Top-k chunks sorted by rerank_score descending.
        Each result gains a "rerank_score" field.
    """
    if not results:
        return results

    if len(results) == 1:
        results[0]["rerank_score"] = 0.0
        return results

    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = round(float(score), 4)

    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
