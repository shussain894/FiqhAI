"""
test_index.py

Tests for the ChromaDB indexing module.

Note: these tests use a temporary ChromaDB directory so they never
touch the real index at data/chroma/.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.retrieval.index import (
    EMBEDDING_MODEL,
    build_index,
    get_chroma_collection,
    load_chunks,
)

CHUNKS_FILE = Path("data/chunks/hanafi_chunks.jsonl")


def test_chunks_file_exists():
    """The JSONL chunks file must exist before indexing."""
    assert CHUNKS_FILE.exists(), "Run chunking step first: src/ingest/chunk.py"


def test_load_chunks_returns_list():
    """load_chunks should return a non-empty list of dicts."""
    if not CHUNKS_FILE.exists():
        pytest.skip("Chunks file not found")

    chunks = load_chunks(CHUNKS_FILE)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_loaded_chunks_have_required_keys():
    """Each chunk must have the fields the indexer depends on."""
    if not CHUNKS_FILE.exists():
        pytest.skip("Chunks file not found")

    chunks = load_chunks(CHUNKS_FILE)
    required = {"chunk_id", "text", "source_title", "file_name", "page", "topic", "word_count"}

    for chunk in chunks[:10]:  # check first 10
        for key in required:
            assert key in chunk, f"Missing key '{key}' in chunk {chunk.get('chunk_id')}"


def test_get_chroma_collection_creates_collection():
    """get_chroma_collection should create and return a collection."""
    with tempfile.TemporaryDirectory() as tmp:
        collection = get_chroma_collection(tmp, "test_collection")
        assert collection is not None
        assert collection.name == "test_collection"


def test_build_index_upserts_chunks():
    """
    build_index should upsert all chunks into ChromaDB.
    Uses a small subset written to a temp file so the test is fast.
    """
    if not CHUNKS_FILE.exists():
        pytest.skip("Chunks file not found")

    # Take a small sample to keep the test fast
    chunks = load_chunks(CHUNKS_FILE)[:20]

    with tempfile.TemporaryDirectory() as tmp_chroma:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp_chunks:
            for chunk in chunks:
                tmp_chunks.write(json.dumps(chunk) + "\n")
            tmp_chunks_path = Path(tmp_chunks.name)

        count = build_index(
            chunks_file=tmp_chunks_path,
            persist_dir=tmp_chroma,
            collection_name="test_hanafi",
            embedding_model_name=EMBEDDING_MODEL
        )

        assert count == len(chunks)

        # Verify the collection actually has the right number of entries
        collection = get_chroma_collection(tmp_chroma, "test_hanafi")
        assert collection.count() == len(chunks)


def test_build_index_is_idempotent():
    """
    Running build_index twice on the same data should not duplicate chunks
    (upsert behaviour).
    """
    if not CHUNKS_FILE.exists():
        pytest.skip("Chunks file not found")

    chunks = load_chunks(CHUNKS_FILE)[:10]

    with tempfile.TemporaryDirectory() as tmp_chroma:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp_chunks:
            for chunk in chunks:
                tmp_chunks.write(json.dumps(chunk) + "\n")
            tmp_chunks_path = Path(tmp_chunks.name)

        # Index twice
        build_index(tmp_chunks_path, tmp_chroma, "test_idempotent", EMBEDDING_MODEL)
        build_index(tmp_chunks_path, tmp_chroma, "test_idempotent", EMBEDDING_MODEL)

        collection = get_chroma_collection(tmp_chroma, "test_idempotent")
        # Should still be 10, not 20
        assert collection.count() == len(chunks)
