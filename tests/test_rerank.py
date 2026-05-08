"""
test_rerank.py

Tests for the reranking module.
"""

import pytest
from src.retrieval.rerank import load_reranker, rerank


@pytest.fixture(scope="module")
def reranker():
    return load_reranker()


def _make_results(texts: list[str]) -> list[dict]:
    return [
        {
            "chunk_id": f"chunk_{i}",
            "source_title": "Test Source",
            "page": i,
            "topic": "Taharah",
            "text": text,
            "score": 0.8,
        }
        for i, text in enumerate(texts)
    ]


def test_rerank_returns_list(reranker):
    results = _make_results(["Wudu requires washing the face.", "Zakah is due on gold."])
    output = rerank("What is wudu?", results, reranker)
    assert isinstance(output, list)


def test_rerank_adds_rerank_score(reranker):
    results = _make_results(["Wudu requires washing the face.", "Zakah is due on gold."])
    output = rerank("What is wudu?", results, reranker)
    for r in output:
        assert "rerank_score" in r
        assert isinstance(r["rerank_score"], float)


def test_rerank_respects_top_k(reranker):
    results = _make_results([
        "Wudu requires washing the face.",
        "Zakah is due on gold.",
        "Fasting in Ramadan is obligatory.",
        "Salah is performed five times daily.",
    ])
    output = rerank("What is wudu?", results, reranker, top_k=2)
    assert len(output) == 2


def test_rerank_orders_by_relevance(reranker):
    """The most relevant chunk should rank first."""
    results = _make_results([
        "Zakah is the obligatory charity on wealth.",
        "Wudu is the ritual purification before prayer involving washing face, arms, and feet.",
    ])
    output = rerank("What are the steps of wudu?", results, reranker)
    assert "wudu" in output[0]["text"].lower() or "purif" in output[0]["text"].lower()


def test_rerank_handles_empty_results(reranker):
    output = rerank("What is wudu?", [], reranker)
    assert output == []


def test_rerank_handles_single_result(reranker):
    results = _make_results(["Wudu is required before prayer."])
    output = rerank("What is wudu?", results, reranker)
    assert len(output) == 1
    assert "rerank_score" in output[0]
