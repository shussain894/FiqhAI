"""
test_rewrite.py

Tests for the query rewriting module.
Ollama-dependent tests are skipped if Ollama is not running.
"""

import pytest
from src.retrieval.rewrite import rewrite_query


def ollama_available() -> bool:
    try:
        import ollama
        from src.retrieval.rewrite import GEMMA_MODEL
        models = ollama.list()
        available = [m["name"] for m in models.get("models", [])]
        return GEMMA_MODEL in available
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama is not running — skipping rewrite tests"
)


@requires_ollama
def test_rewrite_returns_string():
    result = rewrite_query("What are the conditions for wudu?")
    assert isinstance(result, str)
    assert len(result) > 0


@requires_ollama
def test_rewrite_expands_query():
    """Rewritten query should be longer than the original."""
    original = "what breaks the fast"
    result = rewrite_query(original)
    assert len(result) >= len(original)


@requires_ollama
def test_rewrite_contains_fiqh_terms():
    """Rewritten query should include relevant Islamic terminology."""
    result = rewrite_query("what breaks the fast").lower()
    fiqh_terms = ["sawm", "fast", "ramadan", "iftar", "invalidat", "nullif"]
    assert any(term in result for term in fiqh_terms)


@requires_ollama
def test_rewrite_wudu_contains_taharah_terms():
    result = rewrite_query("how do I do wudu").lower()
    terms = ["wudu", "ablution", "purif", "taharah", "wash", "ritual", "minor", "hanafi", "fiqh"]
    assert any(term in result for term in terms)


def test_rewrite_falls_back_on_exception(monkeypatch):
    """If Ollama raises an exception, the original query is returned."""
    import src.retrieval.rewrite as rw

    def broken_chat(*args, **kwargs):
        raise ConnectionError("Ollama not available")

    monkeypatch.setattr(rw.ollama, "chat", broken_chat)
    result = rewrite_query("What is zakah?")
    assert result == "What is zakah?"
