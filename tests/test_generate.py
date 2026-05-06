"""
test_generate.py

Tests for the generation module.

Ollama-dependent tests are skipped automatically if Ollama is not running.
"""

import pytest

from src.generation.generate import (
    HIGH_RISK_RESPONSE,
    OLLAMA_UNAVAILABLE_RESPONSE,
    build_user_prompt,
    generate_answer,
    is_high_risk,
    load_system_prompt,
    run_rag_query,
)


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

def test_load_system_prompt_returns_string():
    """System prompt file should load without error."""
    prompt = load_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_system_prompt_contains_key_instructions():
    """System prompt must contain the core behavioural instructions."""
    prompt = load_system_prompt()
    assert "Hanafi" in prompt
    assert "mufti" in prompt
    assert "fatwa" in prompt
    assert "scholar" in prompt


# ---------------------------------------------------------------------------
# High-risk detection tests
# ---------------------------------------------------------------------------

def test_is_high_risk_divorce():
    assert is_high_risk("I want to know about divorce in Islam") is True


def test_is_high_risk_talaq():
    assert is_high_risk("What is talaq and how does it work?") is True


def test_is_high_risk_inheritance():
    assert is_high_risk("How is inheritance divided according to Hanafi fiqh?") is True


def test_is_high_risk_takfir():
    assert is_high_risk("Is it permissible to declare someone a kafir?") is True


def test_is_high_risk_normal_query():
    """Standard fiqh questions should not be flagged."""
    assert is_high_risk("What are the conditions for wudu?") is False


def test_is_high_risk_prayer_query():
    assert is_high_risk("How many rakahs is Fajr prayer?") is False


def test_is_high_risk_case_insensitive():
    """Detection should be case insensitive."""
    assert is_high_risk("DIVORCE proceedings") is True


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------

def test_build_user_prompt_contains_query():
    """The user prompt must include the original question."""
    prompt = build_user_prompt("What is wudu?", "Some context here.")
    assert "What is wudu?" in prompt


def test_build_user_prompt_contains_context():
    """The user prompt must include the retrieved context."""
    prompt = build_user_prompt("What is wudu?", "Some context here.")
    assert "Some context here." in prompt


def test_build_user_prompt_contains_grounding_instruction():
    """The user prompt must instruct the model to use only retrieved passages."""
    prompt = build_user_prompt("What is wudu?", "Some context here.")
    assert "retrieved passages" in prompt.lower()


# ---------------------------------------------------------------------------
# Generation tests (requires Ollama to be running)
# ---------------------------------------------------------------------------

def ollama_available() -> bool:
    """Returns True if Ollama is reachable and the configured model is present."""
    try:
        import ollama
        from src.generation.generate import GEMMA_MODEL
        models = ollama.list()
        available = [m["name"] for m in models.get("models", [])]
        return GEMMA_MODEL in available
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama is not running — skipping generation tests"
)


@requires_ollama
def test_generate_answer_returns_string():
    """generate_answer should return a non-empty string."""
    context = "[Source 1] Al Hidayah, page 47\nWudu is required before prayer."
    answer = generate_answer("What is wudu?", context)
    assert isinstance(answer, str)
    assert len(answer) > 0


@requires_ollama
def test_generate_answer_mentions_hanafi():
    """The answer should reference Hanafi content."""
    context = "[Source 1] Al Hidayah, page 47\nThe Hanafi madhab requires wudu before prayer."
    answer = generate_answer("What is wudu?", context)
    # Should contain some Islamic terminology
    assert any(word in answer.lower() for word in ["hanafi", "wudu", "prayer", "purif"])


# ---------------------------------------------------------------------------
# Full RAG pipeline tests (requires Ollama + ChromaDB index)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever():
    try:
        from src.retrieval.retrieve import load_retriever
        return load_retriever()
    except Exception as e:
        pytest.skip(f"Could not load retriever: {e}")


@requires_ollama
def test_run_rag_query_returns_dict(retriever):
    """run_rag_query should return a dict with required keys."""
    model, collection = retriever
    result = run_rag_query("What is wudu?", model, collection)

    assert isinstance(result, dict)
    assert "query" in result
    assert "answer" in result
    assert "sources" in result
    assert "high_risk" in result


@requires_ollama
def test_run_rag_query_high_risk_skips_retrieval(retriever):
    """High-risk queries should return the referral response without calling the LLM."""
    model, collection = retriever
    result = run_rag_query("Tell me about divorce and talaq.", model, collection)

    assert result["high_risk"] is True
    assert result["sources"] == []
    assert "scholar" in result["answer"].lower()


@requires_ollama
def test_run_rag_query_normal_has_sources(retriever):
    """Normal queries should return retrieved source chunks."""
    model, collection = retriever
    result = run_rag_query("What breaks the fast?", model, collection)

    assert result["high_risk"] is False
    assert len(result["sources"]) > 0
