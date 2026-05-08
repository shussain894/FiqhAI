# FiqhAI — Hanafi Fiqh Assistant

A local-first, retrieval-augmented AI assistant for Hanafi Islamic jurisprudence.

Answers questions using retrieved passages from curated English Hanafi source texts. All answers are grounded in the source corpus — the model does not answer from memory.

> This tool is educational only. It is not a mufti and does not issue binding fatwas. For complex personal matters, consult a qualified Hanafi scholar.

---

## What it does

- Rewrites user queries into retrieval-optimised form before searching
- Retrieves candidate passages from Hanafi fiqh texts using semantic search
- Reranks candidates with a cross-encoder to filter out irrelevant results
- Generates structured answers grounded only in the top-ranked sources
- Cites the source book and page number for every answer
- States clearly when the corpus is insufficient to answer
- Routes high-risk topics (divorce, inheritance, etc.) to scholar referral

---

## How the retrieval pipeline works

A naive RAG system embeds the user's question and finds the closest chunks by cosine similarity. This works reasonably well but has two failure modes:

1. **Vocabulary mismatch** — a short colloquial question shares few tokens with dense classical fiqh text
2. **Cosine similarity is imprecise** — it finds chunks that are *topically nearby* but not necessarily *relevant to the question*

FiqhAI addresses both with a three-stage retrieval pipeline:

### Stage 1 — Query rewriting

Before hitting the vector database, the user's question is rewritten by the LLM into a keyword-rich retrieval query that includes Arabic terms, Hanafi legal vocabulary, and related concepts.

```
User:      "what breaks the fast?"
Retrieval: "fasting violations ramadan sawm najis ghusl tayammum haram fard wudu"
```

The original question is preserved for generation — only retrieval uses the expanded form.

### Stage 2 — Bi-encoder retrieval

The rewritten query is embedded and matched against 2,400+ indexed chunks using cosine similarity (ChromaDB + all-MiniLM-L6-v2). This retrieves the top 10 candidates quickly.

### Stage 3 — Cross-encoder reranking

A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the original question and each candidate chunk *together*, scoring them for true relevance. The top 5 are kept for generation.

**Why this matters:** cosine similarity can let irrelevant chunks through when query expansion adds noise. In testing, a query about wudu conditions retrieved a chunk about *dhihar* (a form of divorce) ranked first by embedding similarity alone. After reranking, all 5 sources were directly relevant to wudu and purification.

```
Without reranking:
  [1] Usul Al Shashi p.21  | Usul      | score=0.77  ← dhihar (irrelevant)
  [2] Al Hidayah p.76      | Salah     | score=0.75
  [3] Usul Al Shashi p.68  | Muamalat  | score=0.75
  [4] Al Hidayah p.58      | Taharah   | score=0.75
  [5] Al Hidayah p.461     | Usul      | score=0.74

With reranking:
  [1] Al Hidayah p.76      | Salah     | score=0.77  rerank=3.10
  [2] Usul Al Shashi p.138 | Salah     | score=0.72  rerank=2.86
  [3] Ascent To Felicity p.40 | Taharah | score=0.72 rerank=2.59
  [4] Al Hidayah p.58      | Taharah   | score=0.77  rerank=2.51
  [5] Al Hidayah p.58      | Usul      | score=0.74  rerank=0.23
```

---

## Example output

```
QUERY: What are the conditions for wudu to be valid?
RETRIEVAL QUERY: wudu validity conditions Hanafi fiqh

## Short Answer
Wudu must be performed before prayer and is a prerequisite for recitation
of the Quran.

## Hanafi Ruling
According to the Hanafi madhab, wudu is necessary to perform prayer and
recitation. It should be performed before offering a prayer or reciting
the Quran.

## Conditions / Exceptions
Performing the adhan does not require wudu as it is considered dhikr
(remembrance), though it is recommended.

## Source-Based Explanation
The passages clearly state that wudu should be performed before offering
prayer. The intention for prayer is important in this context.

## Citations
- Al Hidayah, page 76
- Usul Al Shashi, page 138
- Ascent To Felicity, page 40

## Confidence
High

## Note
For complex personal cases, consult a qualified Hanafi scholar or local mufti.

--- Sources retrieved ---
  [1] Al Hidayah p.76      | Salah    | score=0.7738 | rerank=3.0956
  [2] Usul Al Shashi p.138 | Salah    | score=0.7177 | rerank=2.8649
  [3] Ascent To Felicity p.40 | Taharah | score=0.7217 | rerank=2.5939
```

---

## Requirements

- Python 3.12
- [Ollama](https://ollama.com) installed and running
- Gemma 2 2B pulled in Ollama (see below)

---

## Setup (first time or new machine)

### 1. Clone the repo

```bash
git clone https://github.com/shussain894/FiqhAI.git
cd FiqhAI
```

### 2. Create a virtual environment and install dependencies

Use Python 3.12 explicitly — other versions may have dependency issues.

```bash
python3.12 -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Add your Hanafi source PDFs

Place your PDF files in:

```
data/raw/pdf/
```

The PDFs must be digital (text-based), not scanned images. Scanned PDFs will be skipped with a warning.

### 4. Create a `.env` file

Copy the example below and save it as `.env` in the project root:

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
GEMMA_MODEL=gemma2:2b

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# ChromaDB
CHROMA_PERSIST_DIR=data/chroma
CHROMA_COLLECTION_NAME=hanafi_fiqh

# Data paths
RAW_PDF_DIR=data/raw/pdf
EXTRACTED_TEXT_DIR=data/processed/extracted_text
CLEANED_TEXT_DIR=data/processed/cleaned_text
CHUNKS_FILE=data/chunks/hanafi_chunks.jsonl
```

### 5. Pull the model in Ollama

Make sure Ollama is running, then pull the model:

```bash
ollama pull gemma2:2b
```

To see what models you have available:

```bash
ollama list
```

### 6. Build the index

This runs the full pipeline: PDF extraction → chunking → ChromaDB indexing.

```bash
python main.py --setup
```

This only needs to be run once, or again whenever you add new PDFs.

---

## Usage

### Interactive mode

```bash
python main.py
```

Type your question and press Enter. Type `quit` to exit.

### Single question

```bash
python main.py --query "What are the conditions for wudu?"
```

### Rebuild the index

```bash
python main.py --setup
```

---

## Project structure

```
FiqhAI/
  main.py                          # entry point
  requirements.txt
  .env                             # your local config (not in git)
  data/
    raw/pdf/                       # source PDFs (not in git)
    processed/extracted_text/      # extracted text (generated)
    chunks/                        # chunked JSONL (generated)
    chroma/                        # vector index (generated)
  src/
    ingest/
      extract.py                   # PDF → extracted text
      clean.py                     # text cleaning
      chunk.py                     # text → chunks with metadata
    retrieval/
      index.py                     # chunks → ChromaDB
      retrieve.py                  # query → top-k chunks (bi-encoder)
      rewrite.py                   # query → retrieval-optimised query
      rerank.py                    # top-k candidates → reranked top-k (cross-encoder)
    generation/
      generate.py                  # full RAG pipeline
      schema.py                    # Pydantic answer schema + markdown parser
  prompts/
    system_prompt.txt              # system instructions for Gemma
  tests/                           # pytest test suite (85 tests)
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

Note: generation and rewrite tests require Ollama to be running with the configured model available.

---

## Tech stack

| Component | Tool |
|---|---|
| Local LLM | Gemma 2 2B via Ollama (`gemma2:2b`) |
| Embeddings (bi-encoder) | all-MiniLM-L6-v2 (sentence-transformers) |
| Reranker (cross-encoder) | ms-marco-MiniLM-L-6-v2 (sentence-transformers) |
| Vector DB | ChromaDB |
| PDF extraction | PyMuPDF |
| Schema validation | Pydantic |
| Testing | pytest |

---

## Roadmap

| Phase | Status |
|---|---|
| Phase 1 — PDF ingestion, chunking, ChromaDB indexing, basic RAG | Complete |
| Phase 2 — Query rewriting, reranking, structured outputs, eval dataset | In progress |
| Phase 3 — Synthetic training data, LoRA fine-tuning, RAG vs fine-tuned comparison | Planned |

---

## Important disclaimers

- This assistant is **not a mufti** and does not issue binding fatwas
- Answers are based solely on the source texts provided — it cannot answer beyond its corpus
- High-risk topics (divorce, inheritance, oaths, medical necessity, etc.) are routed to scholar referral
- Always verify important rulings with a qualified Hanafi scholar
