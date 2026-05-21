"""
detect_topic.py

Detects the likely fiqh topic of a user query using keyword matching,
so retrieval can be scoped to the relevant ChromaDB subset.

Reuses the same keyword lists used to classify chunks at index time,
ensuring consistency between indexing and retrieval.
"""

from src.ingest.chunk import TOPIC_KEYWORDS


def detect_topic(query: str) -> str | None:
    """
    Returns the most likely topic for a query, or None if unclear.

    Uses the same keyword matching as chunk classification so the topic
    label is consistent with what's stored in ChromaDB metadata.
    """
    query_lower = query.lower()
    scores: dict[str, int] = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[topic] = score

    if not scores:
        return None

    best_topic = max(scores, key=lambda t: scores[t])
    best_score = scores[best_topic]

    # Only filter if the signal is reasonably clear — a single keyword hit
    # could be coincidental (e.g. "water" in a non-Taharah question)
    if best_score < 2:
        return None

    return best_topic
