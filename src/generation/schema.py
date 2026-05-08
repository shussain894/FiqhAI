"""
schema.py

Pydantic schema for a structured Hanafi fiqh answer, and a parser
that extracts the fields from the model's markdown-formatted response.

Parsing is done on the Python side rather than asking the model to
produce JSON — small models are unreliable JSON emitters.
"""

import re
from typing import Literal

from pydantic import BaseModel, ValidationError


class FiqhAnswer(BaseModel):
    short_answer: str
    ruling: str
    conditions: str
    explanation: str
    citations: list[str]
    confidence: Literal["High", "Medium", "Low"]
    note: str


# Maps lowercase heading text from the model's output to FiqhAnswer field names
_SECTION_MAP: dict[str, str] = {
    "short answer":             "short_answer",
    "hanafi ruling":            "ruling",
    "conditions / exceptions":  "conditions",
    "conditions/exceptions":    "conditions",
    "conditions":               "conditions",
    "source-based explanation": "explanation",
    "source based explanation": "explanation",
    "citations":                "citations",
    "confidence":               "confidence",
    "note":                     "note",
}

_HEADING = re.compile(r"^## (.+)$", re.MULTILINE)


def parse_markdown_answer(text: str) -> FiqhAnswer | None:
    """
    Parses the model's markdown-formatted answer into a FiqhAnswer.

    Returns None if the text cannot be parsed into a valid schema
    (e.g. too many required fields are missing).
    """
    # Split on ## headings — parts: [preamble, heading, content, heading, content, ...]
    parts = _HEADING.split(text)

    sections: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        heading = parts[i].strip().lower()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        field = _SECTION_MAP.get(heading)
        if field:
            sections[field] = content

    # Confidence: extract High / Medium / Low from whatever the model wrote
    confidence: Literal["High", "Medium", "Low"] = "Low"
    for level in ("High", "Medium", "Low"):
        if level.lower() in sections.get("confidence", "").lower():
            confidence = level  # type: ignore[assignment]
            break

    # Citations: one entry per non-empty bullet line
    citations = [
        line.lstrip("-•* ").strip()
        for line in sections.get("citations", "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    try:
        return FiqhAnswer(
            short_answer=sections.get("short_answer", ""),
            ruling=sections.get("ruling", ""),
            conditions=sections.get("conditions", ""),
            explanation=sections.get("explanation", ""),
            citations=citations,
            confidence=confidence,
            note=sections.get("note", ""),
        )
    except ValidationError:
        return None
