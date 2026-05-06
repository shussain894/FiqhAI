"""
index.py

Reads chunks from hanafi_chunks.jsonl, embeds them using sentence-transformers,
and stores them in a persistent ChromaDB collection.

Run this once to build the index. Re-run whenever chunks change.

ChromaDB stores:
- document text
- embeddings (computed locally via sentence-transformers)
- metadata: chunk_id, source_title, page, topic, word_count
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

CHUNKS_FILE = Path(os.getenv("CHUNKS_FILE", "data/chunks/hanafi_chunks.jsonl"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hanafi_fiqh")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Number of chunks to embed and upsert in one batch
BATCH_SIZE = 64


def load_chunks(chunks_file: Path) -> list[dict]:
    """Reads all chunk records from the JSONL file."""
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def get_chroma_collection(persist_dir: str, collection_name: str):
    """
    Returns a persistent ChromaDB collection.
    Creates it if it doesn't exist yet.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text
    )
    return collection


def build_index(
    chunks_file: Path = CHUNKS_FILE,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model_name: str = EMBEDDING_MODEL
):
    """
    Loads chunks, embeds them, and upserts into ChromaDB.

    Uses upsert so re-running is safe — existing chunks are updated,
    new chunks are added, nothing is duplicated.
    """
    # Load chunks
    print(f"Loading chunks from {chunks_file} ...")
    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks.\n")

    # Load embedding model (downloads once, then cached locally)
    print(f"Loading embedding model: {embedding_model_name} ...")
    model = SentenceTransformer(embedding_model_name)
    print("Model ready.\n")

    # Connect to ChromaDB
    collection = get_chroma_collection(persist_dir, collection_name)
    print(f"ChromaDB collection: '{collection_name}' at {persist_dir}\n")

    # Process in batches
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    total_upserted = 0

    for batch_start in tqdm(range(0, len(chunks), BATCH_SIZE), total=total_batches, desc="Indexing"):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]

        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]

        # Metadata stored alongside each chunk in ChromaDB
        metadatas = [
            {
                "source_title": c["source_title"],
                "file_name": c["file_name"],
                "page": c["page"],
                "topic": c["topic"],
                "word_count": c["word_count"],
            }
            for c in batch
        ]

        # Compute embeddings for this batch
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Upsert into ChromaDB (safe to re-run)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        total_upserted += len(batch)

    print(f"\nIndexing complete. {total_upserted} chunks in collection '{collection_name}'.")
    return total_upserted


if __name__ == "__main__":
    build_index()
