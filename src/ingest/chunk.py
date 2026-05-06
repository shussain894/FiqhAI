"""
chunk.py

Splits cleaned extracted text into overlapping chunks with metadata.

Reads JSON files from data/processed/extracted_text/.
Writes chunks to data/chunks/hanafi_chunks.jsonl.

Each chunk record:
{
    "chunk_id":     "al_hidayah_p010_c002",
    "source_title": "Al Hidayah",
    "file_name":    "Al Hidayah.pdf",
    "page":         10,
    "topic":        "Taharah",
    "text":         "...",
    "word_count":   387,
    "token_count":  503   # approximation: words * 1.3
}
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.ingest.clean import clean_text

load_dotenv()

EXTRACTED_TEXT_DIR = Path(os.getenv("EXTRACTED_TEXT_DIR", "data/processed/extracted_text"))
CHUNKS_FILE = Path(os.getenv("CHUNKS_FILE", "data/chunks/hanafi_chunks.jsonl"))

# Chunking settings
MAX_WORDS = 400       # target chunk size
OVERLAP_WORDS = 80    # words carried over to next chunk
MIN_CHUNK_WORDS = 50  # discard chunks smaller than this

# Keywords used to classify chunks by topic
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Taharah": [
        "wudu", "ablution", "ghusl", "purif", "tahara", "impure",
        "najis", "tayammum", "ritual bath", "water", "clean", "unclean",
        "hadath", "janabah", "menstruat", "istinja"
    ],
    "Salah": [
        "prayer", "salah", "salat", "rakah", "rakat", "qibla", "imam",
        "congregation", "prostrat", "bowing", "fajr", "dhuhr", "asr",
        "maghrib", "isha", "adhan", "iqama", "witr", "sujud", "ruku"
    ],
    "Sawm": [
        "fast", "sawm", "fasting", "ramadan", "iftar", "suhur",
        "suhoor", "break the fast", "obligatory fast"
    ],
    "Zakah": [
        "zakat", "zakah", "nisab", "charity", "alms", "poor due",
        "wealth", "tithe", "eligible recipients", "sadaqah"
    ],
    "Hajj": [
        "hajj", "umrah", "mecca", "makkah", "ihram", "pilgrimage",
        "tawaf", "sa'i", "arafah", "kaaba"
    ],
    "Muamalat": [
        "trade", "sale", "contract", "transaction", "business",
        "debt", "loan", "riba", "interest", "partnership"
    ],
    "Usul": [
        "principle", "dalil", "qiyas", "ijma", "fard", "wajib",
        "sunnah", "makruh", "mubah", "haram", "halal", "obligatory",
        "permissible", "prohibited", "evidence", "ruling", "ijtihad",
        "istihsan", "urf", "madhab"
    ],
}


def classify_topic(text: str) -> str:
    """
    Assigns a topic label to a chunk based on keyword matching.
    Returns the topic with the most keyword hits, or "Other".
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score

    if not scores:
        return "Other"

    # Return the topic with the highest keyword count
    return max(scores, key=lambda t: scores[t])


def split_into_chunks(pages: list[dict]) -> list[dict]:
    """
    Splits a document's pages into overlapping word-based chunks.

    Iterates word by word across all pages, emitting a chunk every
    MAX_WORDS words. The last OVERLAP_WORDS words of each chunk
    are carried into the next chunk for context continuity.

    Returns a list of dicts with text and starting page number.
    """
    chunks = []
    buffer: list[str] = []       # words accumulated so far
    buffer_start_page: int = 1   # page where current buffer started

    for page_data in pages:
        page_num = page_data["page"]
        cleaned = clean_text(page_data["text"])
        words = cleaned.split()

        for word in words:
            # Record the page where this chunk starts
            if not buffer:
                buffer_start_page = page_num

            buffer.append(word)

            if len(buffer) >= MAX_WORDS:
                chunks.append({
                    "text": " ".join(buffer),
                    "page": buffer_start_page
                })
                # Keep the last OVERLAP_WORDS for context in next chunk
                buffer = buffer[-OVERLAP_WORDS:]
                buffer_start_page = page_num

    # Emit any remaining words as a final chunk
    if len(buffer) >= MIN_CHUNK_WORDS:
        chunks.append({
            "text": " ".join(buffer),
            "page": buffer_start_page
        })

    return chunks


def build_chunk_record(
    raw_chunk: dict,
    source_title: str,
    file_name: str,
    chunk_index: int
) -> dict:
    """
    Wraps a raw chunk (text + page) into a full metadata record.
    """
    # Build a URL-safe slug for the chunk ID
    source_slug = source_title.lower().replace(" ", "_")
    chunk_id = f"{source_slug}_p{raw_chunk['page']:03d}_c{chunk_index:04d}"

    word_count = len(raw_chunk["text"].split())
    token_count = int(word_count * 1.3)  # rough approximation

    return {
        "chunk_id": chunk_id,
        "source_title": source_title,
        "file_name": file_name,
        "page": raw_chunk["page"],
        "topic": classify_topic(raw_chunk["text"]),
        "text": raw_chunk["text"],
        "word_count": word_count,
        "token_count": token_count
    }


def chunk_document(doc: dict) -> list[dict]:
    """
    Takes a single extracted document dict and returns a list of chunk records.
    """
    raw_chunks = split_into_chunks(doc["pages"])

    records = []
    for i, raw_chunk in enumerate(raw_chunks, start=1):
        record = build_chunk_record(
            raw_chunk=raw_chunk,
            source_title=doc["source_title"],
            file_name=doc["file_name"],
            chunk_index=i
        )
        records.append(record)

    return records


def run_chunking(
    extracted_dir: Path = EXTRACTED_TEXT_DIR,
    output_file: Path = CHUNKS_FILE
):
    """
    Reads all extracted JSON files and writes chunks to a single JSONL file.
    Each line in the JSONL is one chunk record.
    """
    json_files = list(extracted_dir.glob("*.json"))

    if not json_files:
        print(f"No extracted JSON files found in {extracted_dir}")
        return []

    print(f"Found {len(json_files)} extracted file(s) to chunk.\n")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    with open(output_file, "w", encoding="utf-8") as out:
        for json_path in json_files:
            print(f"Chunking: {json_path.name} ...")

            with open(json_path, "r", encoding="utf-8") as f:
                doc = json.load(f)

            chunks = chunk_document(doc)

            for chunk in chunks:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            all_chunks.extend(chunks)
            print(f"  {len(chunks)} chunks → {output_file}\n")

    print(f"Chunking complete. Total chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    run_chunking()
