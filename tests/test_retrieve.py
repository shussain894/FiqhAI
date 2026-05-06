"""
test_retrieve.py

Tests for the retrieval module.
Requires the ChromaDB index to be built (run src/retrieval/index.py first).
"""

import pytest

from src.retrieval.retrieve import (
    DEFAULT_TOP_K,
    format_for_prompt,
    load_retriever,
    retrieve,
)


@pytest.fixture(scope="module")
def retriever():
    """
    Loads the model and collection once for all tests in this module.
    scope="module" means it's only loaded once, not per test.
    """
    try:
        model, collection = load_retriever()
        return model, collection
    except Exception as e:
        pytest.skip(f"Could not load retriever (index may not be built): {e}")


# ---------------------------------------------------------------------------
# load_retriever tests
# ---------------------------------------------------------------------------

def test_load_retriever_returns_model_and_collection(retriever):
    """load_retriever should return a model and a collection."""
    model, collection = retriever
    assert model is not None
    assert collection is not None


def test_collection_is_not_empty(retriever):
    """The indexed collection should contain chunks."""
    _, collection = retriever
    assert collection.count() > 0


# ---------------------------------------------------------------------------
# retrieve tests
# ---------------------------------------------------------------------------

def test_retrieve_returns_list(retriever):
    """retrieve should return a list."""
    model, collection = retriever
    results = retrieve("What is wudu?", model, collection)
    assert isinstance(results, list)


def test_retrieve_returns_correct_number_of_results(retriever):
    """retrieve should return exactly top_k results (or fewer if index is small)."""
    model, collection = retriever
    results = retrieve("What is wudu?", model, collection, top_k=3)
    assert len(results) <= 3
    assert len(results) > 0


def test_retrieve_default_top_k(retriever):
    """Default top_k should return DEFAULT_TOP_K results."""
    model, collection = retriever
    results = retrieve("conditions of prayer", model, collection)
    assert len(results) <= DEFAULT_TOP_K


def test_retrieve_result_has_required_keys(retriever):
    """Each result should have all expected fields."""
    model, collection = retriever
    results = retrieve("What breaks the fast?", model, collection, top_k=1)

    required_keys = {"chunk_id", "source_title", "file_name", "page", "topic", "text", "score"}
    for key in required_keys:
        assert key in results[0], f"Missing key: {key}"


def test_retrieve_scores_are_between_0_and_1(retriever):
    """Similarity scores should be in [0, 1]."""
    model, collection = retriever
    results = retrieve("prayer times", model, collection, top_k=5)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"


def test_retrieve_results_sorted_by_score(retriever):
    """Results should be sorted highest score first."""
    model, collection = retriever
    results = retrieve("zakah on gold and silver", model, collection, top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_with_topic_filter(retriever):
    """Topic filter should restrict results to the specified topic."""
    model, collection = retriever
    results = retrieve("purification", model, collection, top_k=5, topic_filter="Taharah")
    for r in results:
        assert r["topic"] == "Taharah", f"Expected Taharah, got {r['topic']}"


def test_retrieve_relevant_results_for_wudu(retriever):
    """A query about wudu should return results from Taharah-related chunks."""
    model, collection = retriever
    results = retrieve("What are the conditions for wudu?", model, collection, top_k=5)
    topics = [r["topic"] for r in results]
    # At least one result should be Taharah-related
    assert "Taharah" in topics, f"Expected Taharah in results, got: {topics}"


# ---------------------------------------------------------------------------
# format_for_prompt tests
# ---------------------------------------------------------------------------

def test_format_for_prompt_returns_string(retriever):
    """format_for_prompt should return a non-empty string."""
    model, collection = retriever
    results = retrieve("prayer", model, collection, top_k=3)
    formatted = format_for_prompt(results)
    assert isinstance(formatted, str)
    assert len(formatted) > 0


def test_format_for_prompt_includes_source_titles(retriever):
    """Formatted output should include the source title for each chunk."""
    model, collection = retriever
    results = retrieve("prayer", model, collection, top_k=2)
    formatted = format_for_prompt(results)
    for r in results:
        assert r["source_title"] in formatted


def test_format_for_prompt_empty_results():
    """format_for_prompt with empty list should return a fallback message."""
    output = format_for_prompt([])
    assert "No relevant source passages" in output
