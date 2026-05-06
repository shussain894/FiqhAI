"""
clean.py

Cleans raw extracted text from PDFs before chunking.

Handles common PDF extraction noise:
- Broken hyphenation across lines (e.g. "pu-\nrification" → "purification")
- Standalone page numbers
- Excessive whitespace and blank lines
- Repeated header/footer lines
"""

import re


def fix_hyphenation(text: str) -> str:
    """
    Re-joins words that were split across lines by a hyphen.
    e.g. "purifi-\ncation" → "purification"
    """
    return re.sub(r"-\n(\w)", r"\1", text)


def remove_standalone_page_numbers(text: str) -> str:
    """
    Removes lines that contain only a number (page numbers).
    e.g. a line that is just "42" or "  42  "
    """
    lines = text.split("\n")
    cleaned = [line for line in lines if not re.match(r"^\s*\d+\s*$", line)]
    return "\n".join(cleaned)


def remove_repeated_lines(text: str, min_repeats: int = 3) -> str:
    """
    Removes lines that appear 3 or more times (likely headers/footers).
    e.g. "Chapter 1: Taharah" appearing on every page.
    """
    lines = text.split("\n")

    # Count occurrences of each stripped line
    from collections import Counter
    counts = Counter(line.strip() for line in lines if line.strip())

    # Build set of lines to remove
    repeated = {line for line, count in counts.items() if count >= min_repeats}

    cleaned = [line for line in lines if line.strip() not in repeated]
    return "\n".join(cleaned)


def normalise_whitespace(text: str) -> str:
    """
    Collapses multiple blank lines into one and strips leading/trailing space.
    """
    # Replace 3+ consecutive newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def clean_text(text: str) -> str:
    """
    Runs all cleaning steps in order on a piece of text.
    Returns cleaned text ready for chunking.
    """
    text = fix_hyphenation(text)
    text = remove_standalone_page_numbers(text)
    text = remove_repeated_lines(text)
    text = normalise_whitespace(text)
    return text
