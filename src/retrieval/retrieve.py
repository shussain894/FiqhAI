"""
retrieve.py

Takes a user query, embeds it, and returns the top-k most relevant
chunks from the ChromaDB collection.

Supports optional topic filtering so retrieval can be scoped to
e.g. only Taharah or Salah chunks.

Returns a list of result dicts, each containing:
{
    "chunk_id":     "al_hidayah_p010_c002",
    "source_title": "Al Hidayah",
    "page":         10,
    "topic":        "Taharah",
    "text":         "...",
    "score":        0.87   # cosine similarity (higher = more relevant)
}
"""

import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hanafi_fiqh")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

DEFAULT_TOP_K = 5


def load_retriever():
    """
    Loads the embedding model and ChromaDB collection.
    Call once and reuse the returned objects to avoid reloading on every query.

    Returns:
        model: SentenceTransformer embedding model
        collection: ChromaDB collection
    """
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    return model, collection


def retrieve(
    query: str,
    model: SentenceTransformer,
    collection,
    top_k: int = DEFAULT_TOP_K,
    topic_filter: str | None = None
) -> list[dict]:
    """
    Retrieves the top-k chunks most relevant to the query.

    Args:
        query:        The user's question as a plain string.
        model:        Loaded SentenceTransformer model.
        collection:   ChromaDB collection to search.
        top_k:        Number of results to return.
        topic_filter: Optional topic to restrict search to
                      (e.g. "Taharah", "Salah"). None = search all topics.

    Returns:
        List of result dicts sorted by relevance (most relevant first).
    """
    # Embed the query using the same model used for indexing
    query_embedding = model.encode(query).tolist()

    # Build optional metadata filter
    where = {"topic": topic_filter} if topic_filter else None

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    # Unpack ChromaDB response and format results
    formatted = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to a 0–1 similarity score
        similarity = 1 - (distance / 2)

        formatted.append({
            "chunk_id":     chunk_id,
            "source_title": meta.get("source_title", ""),
            "file_name":    meta.get("file_name", ""),
            "page":         meta.get("page", 0),
            "topic":        meta.get("topic", ""),
            "text":         text,
            "score":        round(similarity, 4)
        })

    # Sort by score descending (most relevant first)
    formatted.sort(key=lambda x: x["score"], reverse=True)

    return formatted


def format_for_prompt(results: list[dict]) -> str:
    """
    Formats retrieved chunks into a single string ready to be injected
    into the LLM prompt as context.

    Each chunk is labelled with its source and page number.
    """
    if not results:
        return "No relevant source passages were found."

    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(
            f"[Source {i}] {r['source_title']}, page {r['page']}\n"
            f"{r['text']}"
        )

    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    # Quick manual test — run as: python -m src.retrieval.retrieve
    print("Loading retriever...")
    model, collection = load_retriever()

    query = "What are the conditions for wudu to be valid?"
    print(f"\nQuery: {query}\n")

    results = retrieve(query, model, collection, top_k=3)

    for r in results:
        print(f"[{r['score']:.3f}] {r['source_title']} p.{r['page']} ({r['topic']})")
        print(f"  {r['text'][:150]}...\n")
