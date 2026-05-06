"""
test_chunk.py

Tests for the cleaning and chunking pipeline.
"""

from pathlib import Path

from src.ingest.clean import (
    clean_text,
    fix_hyphenation,
    normalise_whitespace,
    remove_repeated_lines,
    remove_standalone_page_numbers,
)
from src.ingest.chunk import (
    MIN_CHUNK_WORDS,
    MAX_WORDS,
    OVERLAP_WORDS,
    build_chunk_record,
    chunk_document,
    classify_topic,
    run_chunking,
    split_into_chunks,
)

CHUNKS_FILE = Path("data/chunks/hanafi_chunks.jsonl")
EXTRACTED_DIR = Path("data/processed/extracted_text")


# ---------------------------------------------------------------------------
# Cleaning tests
# ---------------------------------------------------------------------------

def test_fix_hyphenation():
    """Hyphenated line breaks should be re-joined."""
    assert fix_hyphenation("purifi-\ncation") == "purification"
    assert fix_hyphenation("no hyphen here") == "no hyphen here"


def test_remove_standalone_page_numbers():
    """Lines that are only digits should be removed."""
    text = "Some text\n42\nMore text\n  7  \nEnd"
    result = remove_standalone_page_numbers(text)
    assert "42" not in result
    assert "7" not in result
    assert "Some text" in result
    assert "More text" in result


def test_remove_repeated_lines():
    """Lines appearing 3+ times should be removed."""
    repeated = "Chapter Header"
    text = "\n".join([repeated, "unique line", repeated, "another", repeated])
    result = remove_repeated_lines(text, min_repeats=3)
    assert repeated not in result
    assert "unique line" in result


def test_normalise_whitespace():
    """Multiple blank lines should be collapsed to one."""
    text = "Line one\n\n\n\nLine two"
    result = normalise_whitespace(text)
    assert "\n\n\n" not in result
    assert "Line one" in result
    assert "Line two" in result


def test_clean_text_runs_without_error():
    """clean_text should handle typical PDF noise without crashing."""
    noisy = "purifi-\ncation\n\n42\n\nSome ruling here.\n\n\n\nAnother line."
    result = clean_text(noisy)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Topic classification tests
# ---------------------------------------------------------------------------

def test_classify_topic_taharah():
    assert classify_topic("The conditions of wudu and ablution are discussed here.") == "Taharah"


def test_classify_topic_salah():
    assert classify_topic("The obligatory acts of salah include ruku and sujud.") == "Salah"


def test_classify_topic_sawm():
    assert classify_topic("Fasting in Ramadan is obligatory and breaking the fast is called iftar.") == "Sawm"


def test_classify_topic_zakah():
    assert classify_topic("The nisab for zakah on gold is discussed.") == "Zakah"


def test_classify_topic_other():
    assert classify_topic("This text has no Islamic keywords at all.") == "Other"


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------

def _make_pages(num_pages: int = 5, words_per_page: int = 200) -> list[dict]:
    """Helper: creates fake pages with enough words to produce multiple chunks."""
    word = "word"
    return [
        {"page": i + 1, "text": (word + " ") * words_per_page}
        for i in range(num_pages)
    ]


def test_split_into_chunks_returns_list():
    pages = _make_pages()
    chunks = split_into_chunks(pages)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunks_respect_max_words():
    """No chunk should exceed MAX_WORDS + OVERLAP_WORDS words."""
    pages = _make_pages(num_pages=10, words_per_page=300)
    chunks = split_into_chunks(pages)
    for chunk in chunks:
        word_count = len(chunk["text"].split())
        # Chunks can be slightly over MAX_WORDS due to overlap carry-in
        assert word_count <= MAX_WORDS + OVERLAP_WORDS


def test_chunks_meet_minimum_size():
    """All chunks should meet the minimum word threshold."""
    pages = _make_pages(num_pages=10, words_per_page=300)
    chunks = split_into_chunks(pages)
    for chunk in chunks:
        assert len(chunk["text"].split()) >= MIN_CHUNK_WORDS


def test_chunk_has_page_number():
    """Each raw chunk must have a page number."""
    pages = _make_pages()
    chunks = split_into_chunks(pages)
    for chunk in chunks:
        assert "page" in chunk
        assert isinstance(chunk["page"], int)


def test_build_chunk_record_keys():
    """build_chunk_record should return all required metadata fields."""
    raw = {"text": "some fiqh text about wudu and prayer", "page": 5}
    record = build_chunk_record(raw, "Al Hidayah", "Al Hidayah.pdf", 1)

    required_keys = ["chunk_id", "source_title", "file_name", "page",
                     "topic", "text", "word_count", "token_count"]
    for key in required_keys:
        assert key in record, f"Missing key: {key}"


def test_chunk_id_format():
    """Chunk IDs should follow the expected slug format."""
    raw = {"text": "some text", "page": 10}
    record = build_chunk_record(raw, "Al Hidayah", "Al Hidayah.pdf", 2)
    assert record["chunk_id"].startswith("al_hidayah_")
    assert "_c0002" in record["chunk_id"]


def test_chunk_document_produces_records():
    """chunk_document should return non-empty list for a doc with enough text."""
    doc = {
        "source_title": "Test Book",
        "file_name": "test.pdf",
        "total_pages": 5,
        "pages": _make_pages(num_pages=5, words_per_page=300)
    }
    records = chunk_document(doc)
    assert len(records) > 0


# ---------------------------------------------------------------------------
# End-to-end chunking test
# ---------------------------------------------------------------------------

def test_run_chunking_creates_jsonl():
    """
    run_chunking should produce a non-empty JSONL file
    if extracted JSON files are present.
    """
    json_files = list(EXTRACTED_DIR.glob("*.json"))
    if not json_files:
        import pytest
        pytest.skip("No extracted JSON files found — run extraction first")

    chunks = run_chunking(extracted_dir=EXTRACTED_DIR, output_file=CHUNKS_FILE)

    assert CHUNKS_FILE.exists()
    assert len(chunks) > 0

    # Verify JSONL is valid — read back and check first record
    import json
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        first_line = f.readline()
    record = json.loads(first_line)

    assert "chunk_id" in record
    assert "text" in record
    assert "source_title" in record
