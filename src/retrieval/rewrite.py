"""
rewrite.py

Rewrites a user query into a keyword-rich form before sending it to ChromaDB.

A short, colloquial question like "what breaks the fast?" retrieves poorly
because it shares few tokens with the dense fiqh text in the index.
The rewriter expands it with Arabic terms, Hanafi legal vocabulary, and
related concepts so the embedding search finds more relevant chunks.

The original query is preserved for generation — only retrieval uses the
rewritten form.
"""

import os

import ollama
from dotenv import load_dotenv

load_dotenv()

GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma2:2b")

_REWRITE_PROMPT = """\
You are a Hanafi fiqh search assistant. Rewrite the question below as a \
keyword-rich search query for a classical Hanafi fiqh text database.

Rules:
- Include relevant Arabic terms (e.g. wudu, ghusl, salah, sawm, zakah, \
tayammum, najis, hadath, fard, wajib, sunnah, makruh, haram)
- Include synonyms and related Hanafi legal concepts
- Return only the rewritten query on a single line
- Do not explain, do not add commentary, do not repeat the original question

Question: {query}
Rewritten query:"""


def rewrite_query(query: str, model_name: str = GEMMA_MODEL) -> str:
    """
    Expands a user query into a retrieval-optimised form.

    Returns the rewritten query, or the original query unchanged if
    rewriting fails or produces an unusable result.
    """
    prompt = _REWRITE_PROMPT.format(query=query)

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        rewritten = response["message"]["content"].strip()

        # Sanity check — reject if suspiciously short or extremely long
        if 5 < len(rewritten) < 600:
            return rewritten

    except Exception:
        pass

    return query
