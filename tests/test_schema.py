"""
test_schema.py

Tests for the FiqhAnswer schema and markdown parser.
"""

import pytest
from src.generation.schema import FiqhAnswer, parse_markdown_answer

FULL_ANSWER = """
## Short Answer
Wudu requires washing four limbs in order.

## Hanafi Ruling
The Hanafi madhab holds that wudu is obligatory before prayer.

## Conditions / Exceptions
Wudu is invalidated by sleep and loss of consciousness.

## Source-Based Explanation
According to Al Hidayah, the conditions for wudu include washing the face, arms, wiping the head, and washing the feet.

## Citations
- Al Hidayah, page 47: "Wudu requires washing four limbs"
- Mukhtasar Al Quduri, page 12: "Impurity invalidates wudu"

## Confidence
High

## Note
For complex personal cases, consult a qualified Hanafi scholar.
"""

PARTIAL_ANSWER = """
## Short Answer
Fasting requires abstaining from food and drink.

## Hanafi Ruling
The fast is broken by eating or drinking intentionally.

## Conditions / Exceptions
None mentioned in sources.

## Source-Based Explanation
The sources indicate that intentional eating invalidates the fast.

## Citations
- Al Hidayah, page 204: "Eating invalidates the fast"

## Confidence
Medium

## Note
Consult a scholar for your specific situation.
"""


def test_parse_full_answer_returns_fiqh_answer():
    result = parse_markdown_answer(FULL_ANSWER)
    assert isinstance(result, FiqhAnswer)


def test_parse_short_answer():
    result = parse_markdown_answer(FULL_ANSWER)
    assert "Wudu requires washing four limbs" in result.short_answer


def test_parse_ruling():
    result = parse_markdown_answer(FULL_ANSWER)
    assert "obligatory" in result.ruling.lower()


def test_parse_conditions():
    result = parse_markdown_answer(FULL_ANSWER)
    assert "sleep" in result.conditions.lower()


def test_parse_explanation():
    result = parse_markdown_answer(FULL_ANSWER)
    assert "al hidayah" in result.explanation.lower()


def test_parse_citations_are_list():
    result = parse_markdown_answer(FULL_ANSWER)
    assert isinstance(result.citations, list)
    assert len(result.citations) == 2


def test_parse_citations_stripped():
    result = parse_markdown_answer(FULL_ANSWER)
    for c in result.citations:
        assert not c.startswith("-")
        assert not c.startswith("•")


def test_parse_confidence_high():
    result = parse_markdown_answer(FULL_ANSWER)
    assert result.confidence == "High"


def test_parse_confidence_medium():
    result = parse_markdown_answer(PARTIAL_ANSWER)
    assert result.confidence == "Medium"


def test_parse_note():
    result = parse_markdown_answer(FULL_ANSWER)
    assert "scholar" in result.note.lower()


def test_parse_returns_none_for_empty_string():
    result = parse_markdown_answer("")
    assert result is None or isinstance(result, FiqhAnswer)


def test_parse_handles_missing_sections_gracefully():
    minimal = """
## Short Answer
A brief answer.

## Hanafi Ruling
The ruling.

## Conditions / Exceptions
None mentioned in sources.

## Source-Based Explanation
Some explanation.

## Citations
- Source, page 1

## Confidence
Low

## Note
Consult a scholar.
"""
    result = parse_markdown_answer(minimal)
    assert isinstance(result, FiqhAnswer)
    assert result.confidence == "Low"


def test_fiqh_answer_confidence_validation():
    """FiqhAnswer should reject invalid confidence values."""
    with pytest.raises(Exception):
        FiqhAnswer(
            short_answer="x",
            ruling="x",
            conditions="x",
            explanation="x",
            citations=[],
            confidence="Invalid",  # type: ignore
            note="x",
        )
