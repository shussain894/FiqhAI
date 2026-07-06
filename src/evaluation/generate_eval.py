"""
generate_eval.py

Generates candidate evaluation questions from the chunk corpus.

For each target topic, samples a spread of chunks and asks Gemma to write
one focused question per chunk. Output is saved to data/eval/candidate_questions.jsonl
for human review — edit, trim, or reject before using as a ground-truth eval set.

Run with:
    python -m src.evaluation.generate_eval
"""

import json
import os
import random
from pathlib import Path

import ollama
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma2:2b")
CHUNKS_FILE = Path(os.getenv("CHUNKS_FILE", "data/chunks/hanafi_chunks.jsonl"))
OUTPUT_FILE = Path("data/eval/candidate_questions.jsonl")

# How many candidate questions to generate per topic
QUESTIONS_PER_TOPIC = 25

TARGET_TOPICS = ["Taharah", "Salah", "Sawm", "Zakah"]

# Minimum chunk word count — very short chunks rarely yield good questions
MIN_WORDS = 80

_PROMPT = """\
You are building an evaluation dataset for a Hanafi fiqh assistant.

Read the passage below and write ONE clear, specific question that:
- Can be answered from this passage alone
- Asks about a concrete Hanafi ruling, condition, or principle
- Would be asked by someone genuinely learning Hanafi fiqh
- Is specific, not vague (e.g. "What invalidates wudu?" not "Tell me about wudu")

Passage from {source_title}, page {page} (topic: {topic}):
{text}

Write only the question. No explanation, no numbering, no preamble."""


def _is_english_source(chunk: dict) -> bool:
    """Exclude turath.io Arabic-source chunks — Gemma2:2b cannot generate
    reliable English questions from Arabic passages."""
    return not chunk.get("file_name", "").startswith("turath_")


def load_chunks_by_topic(chunks_file: Path) -> dict[str, list[dict]]:
    by_topic: dict[str, list[dict]] = {t: [] for t in TARGET_TOPICS}
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            topic = chunk.get("topic", "")
            if topic in by_topic and chunk.get("word_count", 0) >= MIN_WORDS and _is_english_source(chunk):
                by_topic[topic].append(chunk)
    return by_topic


def sample_chunks(chunks: list[dict], n: int) -> list[dict]:
    """
    Samples n chunks with a spread across different source books and pages
    rather than a purely random sample.
    """
    if len(chunks) <= n:
        return chunks

    # Group by source title, then sample evenly across books
    by_source: dict[str, list[dict]] = {}
    for c in chunks:
        by_source.setdefault(c["source_title"], []).append(c)

    sampled: list[dict] = []
    sources = list(by_source.keys())
    per_source = max(1, n // len(sources))

    for source_chunks in by_source.values():
        sampled.extend(random.sample(source_chunks, min(per_source, len(source_chunks))))

    # Top up to n if needed
    remaining = [c for c in chunks if c not in sampled]
    if len(sampled) < n and remaining:
        sampled.extend(random.sample(remaining, min(n - len(sampled), len(remaining))))

    return sampled[:n]


def generate_question(chunk: dict, model_name: str = GEMMA_MODEL) -> str | None:
    prompt = _PROMPT.format(
        source_title=chunk["source_title"],
        page=chunk["page"],
        topic=chunk["topic"],
        text=chunk["text"][:800],  # cap length for the 2B model
    )
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        question = response["message"]["content"].strip()
        # Reject if it looks like the model returned an explanation instead of a question
        if len(question) < 10 or len(question) > 300 or "\n\n" in question:
            return None
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"
        return question
    except Exception:
        return None


def run(seed: int = 42):
    random.seed(seed)

    print("Loading chunks...")
    by_topic = load_chunks_by_topic(CHUNKS_FILE)

    for topic in TARGET_TOPICS:
        print(f"  {topic}: {len(by_topic[topic])} eligible chunks")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_questions = []
    q_id = 1

    for topic in TARGET_TOPICS:
        print(f"\nGenerating questions for {topic}...")
        sample = sample_chunks(by_topic[topic], QUESTIONS_PER_TOPIC)

        topic_questions = []
        for chunk in tqdm(sample, desc=f"  {topic}"):
            question = generate_question(chunk)
            if question is None:
                continue

            record = {
                "id": f"{topic.lower()}_{q_id:03d}",
                "topic": topic,
                "question": question,
                "source_hint": f"{chunk['source_title']} p.{chunk['page']}",
                "chunk_id": chunk["chunk_id"],
                "approved": True,   # set to false to exclude from eval
            }
            topic_questions.append(record)
            all_questions.append(record)
            q_id += 1

        print(f"  Generated {len(topic_questions)} questions for {topic}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in all_questions:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(all_questions)} candidate questions saved to {OUTPUT_FILE}")
    print("Review the file and set 'approved': false for any questions to exclude.")


if __name__ == "__main__":
    run()
