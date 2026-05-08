"""
generate.py

Takes a user query and retrieved source chunks, formats a prompt,
and calls Gemma via Ollama to generate a grounded Hanafi fiqh answer.

The model is instructed to:
- Answer only from the provided passages
- Follow the Hanafi madhab
- Cite sources
- Admit uncertainty
- Refuse or redirect high-risk topics
"""

import os
from pathlib import Path

import ollama
from dotenv import load_dotenv

from src.generation.schema import parse_markdown_answer

load_dotenv()

GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma:4b")
SYSTEM_PROMPT_FILE = Path("prompts/system_prompt.txt")

# Keywords that indicate a high-risk topic requiring scholar referral
HIGH_RISK_KEYWORDS = [
    "divorce", "talaq", "talāq", "khul", "marriage dispute",
    "inheritance", "mirath", "will", "estate",
    "oath", "vow", "kaffarah",
    "medical necessity", "medication", "surgery",
    "financial contract", "mortgage", "riba", "interest",
    "abuse", "harm", "coercion", "forced",
    "criminal", "theft", "hadd", "punishment",
    "takfir", "kafir", "apostate", "apostasy",
    "sectarian", "shia", "sunni conflict",
]

# Response shown when a query is flagged as high-risk
HIGH_RISK_RESPONSE = """I cannot give a definitive ruling on this matter.

This topic falls into a category that requires personal scholarly guidance. \
The Hanafi sources in the current corpus may discuss related principles, but \
your specific situation should be taken to a qualified Hanafi scholar or local \
mufti who can ask the necessary follow-up questions.

**Confidence:** Low
**Note:** Please consult a qualified Hanafi scholar or local mufti."""

# Response shown when Ollama is not reachable
OLLAMA_UNAVAILABLE_RESPONSE = """The local model (Gemma via Ollama) is not currently running.

To start it, run in your terminal:
    ollama serve

Then ensure the model is available:
    ollama pull gemma:4b"""


def load_system_prompt() -> str:
    """Reads the system prompt from the prompts/ directory."""
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def is_high_risk(query: str) -> bool:
    """
    Returns True if the query contains keywords associated with
    high-risk topics that require scholar referral.
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in HIGH_RISK_KEYWORDS)


def build_user_prompt(query: str, context: str) -> str:
    """
    Builds the user-facing prompt by combining the question
    with the retrieved source passages.
    """
    return f"""Question:
{query}

Retrieved Hanafi source passages:
{context}

Answer using only the retrieved passages above.
If the passages do not sufficiently answer the question, clearly state:
"I do not have enough approved Hanafi source material in the current corpus to answer this confidently."
"""


def generate_answer(query: str, context: str, model_name: str = GEMMA_MODEL) -> str:
    """
    Sends the prompt to Gemma via Ollama and returns the response text.

    Returns an error message string if Ollama is not reachable.
    """
    system_prompt = load_system_prompt()
    user_prompt = build_user_prompt(query, context)

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
        )
        return response["message"]["content"]

    except Exception as e:
        error = str(e).lower()
        if "connection" in error or "refused" in error:
            return OLLAMA_UNAVAILABLE_RESPONSE
        raise


def run_rag_query(
    query: str,
    model,          # SentenceTransformer model
    collection,     # ChromaDB collection
    top_k: int = 5,
    topic_filter: str | None = None,
    llm_model: str = GEMMA_MODEL
) -> dict:
    """
    Full RAG pipeline:
      1. Safety check the query
      2. Retrieve top-k relevant chunks
      3. Format context
      4. Generate answer with Gemma

    Returns a dict with the answer, retrieved sources, and query metadata.
    """
    # Import here to avoid circular dependency
    from src.retrieval.retrieve import retrieve, format_for_prompt

    # Step 1 — safety check
    if is_high_risk(query):
        return {
            "query": query,
            "answer": HIGH_RISK_RESPONSE,
            "sources": [],
            "high_risk": True,
        }

    # Step 2 — retrieve relevant chunks
    results = retrieve(query, model, collection, top_k=top_k, topic_filter=topic_filter)

    # Step 3 — format context for prompt
    context = format_for_prompt(results)

    # Step 4 — generate answer
    answer = generate_answer(query, context, model_name=llm_model)

    return {
        "query": query,
        "answer": answer,
        "structured_answer": parse_markdown_answer(answer),
        "sources": results,
        "high_risk": False,
    }
