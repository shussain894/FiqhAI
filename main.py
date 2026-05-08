"""
main.py

Entry point for the Hanafi Fiqh Assistant.

Two modes:

  1. Setup mode  — builds the full pipeline from scratch:
                   PDF extraction → chunking → ChromaDB indexing
     Run with:   python main.py --setup

  2. Query mode  — interactive question/answer loop using the built index
     Run with:   python main.py

  Single query:  python main.py --query "What are the conditions for wudu?"

On a new machine, always run --setup first to rebuild the index from your PDFs.
"""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Setup pipeline
# ---------------------------------------------------------------------------

def run_setup():
    """
    Runs the full data pipeline:
      1. Extract text from all PDFs
      2. Chunk extracted text into JSONL
      3. Embed chunks and index into ChromaDB
    """
    print("\n=== Hanafi Fiqh Assistant — Setup ===\n")

    # Step 1 — PDF extraction
    print("Step 1/3 — Extracting text from PDFs...")
    from src.ingest.extract import run_extraction
    results = run_extraction()

    successful = [r for r in results if r["pages_extracted"] > 0]
    skipped = [r for r in results if r["pages_extracted"] == 0]

    print(f"  Extracted {len(successful)} document(s).")
    if skipped:
        print(f"  Skipped {len(skipped)} document(s) — no extractable text (scanned PDFs).")
    print()

    if not successful:
        print("ERROR: No text could be extracted. Check your PDFs in data/raw/pdf/")
        sys.exit(1)

    # Step 2 — Chunking
    print("Step 2/3 — Chunking extracted text...")
    from src.ingest.chunk import run_chunking
    chunks = run_chunking()
    print(f"  Created {len(chunks)} chunks.\n")

    if not chunks:
        print("ERROR: No chunks were created. Check the extraction output.")
        sys.exit(1)

    # Step 3 — Indexing
    print("Step 3/3 — Building ChromaDB index...")
    from src.retrieval.index import build_index
    total = build_index()
    print(f"  Indexed {total} chunks.\n")

    print("=== Setup complete. Run 'python main.py' to start asking questions. ===\n")


# ---------------------------------------------------------------------------
# Query mode
# ---------------------------------------------------------------------------

def load_pipeline():
    """Loads the retriever (embedding model + ChromaDB collection)."""
    from src.retrieval.retrieve import load_retriever

    chroma_dir = Path("data/chroma")
    if not chroma_dir.exists():
        print("ERROR: ChromaDB index not found.")
        print("Run 'python main.py --setup' first to build the index.\n")
        sys.exit(1)

    print("Loading retriever...")
    model, collection = load_retriever()

    if collection.count() == 0:
        print("ERROR: ChromaDB collection is empty.")
        print("Run 'python main.py --setup' first.\n")
        sys.exit(1)

    print(f"Ready. Index contains {collection.count()} chunks.\n")
    return model, collection


def print_answer(result: dict):
    """Prints a query result in a readable format."""
    print("\n" + "=" * 70)

    if result["high_risk"]:
        print("TOPIC: High-risk — scholar referral required")
        print("=" * 70)
        print(result["answer"])
    else:
        print(f"QUERY: {result['query']}")
        print("=" * 70)

        sa = result.get("structured_answer")
        if sa:
            print(f"\n## Short Answer\n{sa.short_answer}")
            print(f"\n## Hanafi Ruling\n{sa.ruling}")
            print(f"\n## Conditions / Exceptions\n{sa.conditions}")
            print(f"\n## Source-Based Explanation\n{sa.explanation}")
            print("\n## Citations")
            for c in sa.citations:
                print(f"  - {c}")
            print(f"\n## Confidence\n{sa.confidence}")
            print(f"\n## Note\n{sa.note}")
        else:
            # Fallback to raw text if parsing failed
            print(result["answer"])

        print("\n--- Sources retrieved ---")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {s['source_title']} p.{s['page']} | {s['topic']} | score={s['score']}")

    print("=" * 70 + "\n")


def run_single_query(query: str):
    """Runs a single query and prints the result."""
    from src.generation.generate import run_rag_query

    model, collection = load_pipeline()
    print(f"Querying: {query}\n")
    result = run_rag_query(query, model, collection)
    print_answer(result)


def run_interactive():
    """
    Starts an interactive question/answer loop.
    Type 'quit' or 'exit' or press Ctrl+C to stop.
    """
    from src.generation.generate import run_rag_query

    print("\n=== Hanafi Fiqh Assistant ===")
    print("Ask questions about Hanafi fiqh. Type 'quit' to exit.\n")
    print("This assistant answers from retrieved Hanafi source texts only.")
    print("It is not a mufti and does not issue binding fatwas.\n")

    model, collection = load_pipeline()

    while True:
        try:
            query = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        print("\nSearching sources and generating answer...\n")
        result = run_rag_query(query, model, collection)
        print_answer(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hanafi Fiqh Assistant — RAG-based educational tool"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Build the full pipeline: extract PDFs → chunk → index into ChromaDB"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a single question and exit"
    )

    args = parser.parse_args()

    if args.setup:
        run_setup()
    elif args.query:
        run_single_query(args.query)
    else:
        run_interactive()
