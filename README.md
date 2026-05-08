# FiqhAI — Hanafi Fiqh Assistant

A local-first, retrieval-augmented AI assistant for Hanafi Islamic jurisprudence.

Answers questions using retrieved passages from curated English Hanafi source texts. All answers are grounded in the source corpus — the model does not answer from memory.

> This tool is educational only. It is not a mufti and does not issue binding fatwas. For complex personal matters, consult a qualified Hanafi scholar.

---

## What it does

- Retrieves relevant passages from Hanafi fiqh texts using semantic search
- Generates structured answers grounded only in retrieved sources
- Cites the source book and page number for every answer
- States clearly when the corpus is insufficient to answer
- Routes high-risk topics (divorce, inheritance, etc.) to scholar referral

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Gemma model pulled in Ollama (see below)

---

## Setup (first time or new machine)

### 1. Clone the repo

```bash
git clone https://github.com/shussain894/FiqhAI.git
cd FiqhAI
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
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

Update `GEMMA_MODEL` to match whatever model you have pulled in Ollama.

### 5. Pull a model in Ollama

Make sure Ollama is running, then pull your model:

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

Start a question/answer session:

```bash
python main.py
```

Type your question and press Enter. Type `quit` to exit.

### Single question

Ask one question and exit:

```bash
python main.py --query "What are the conditions for wudu?"
```

### Rebuild the index

If you add new PDFs or want to rebuild from scratch:

```bash
python main.py --setup
```

---

## Example output

```
QUERY: What things break the fast in Ramadan?

## Short Answer
Fasting involves abstaining during the day from eating, drinking,
and sexual intercourse, coupled with the intention of fasting.

## Hanafi Ruling
Fasting requires abstaining from eating, drinking, and engaging
in sexual intercourse throughout the day.

## Conditions / Exceptions
Menstruation is noted as an obstacle by nature and by law.

## Source-Based Explanation
According to Al Hidayah: "Fasting is abstaining during the day
from eating, drinking and sexual intercourse, along with the
intention of fasting."

## Citations
- Al Hidayah, page 196: "Fasting is abstaining during the day..."
- Al Hidayah, page 296: "Menstruation is an obstacle by nature..."

## Confidence
High

## Note
For complex personal cases, consult a qualified Hanafi scholar.
```

---

## Project structure

```
FiqhAI/
  main.py                        # entry point
  requirements.txt
  .env                           # your local config (not in git)
  data/
    raw/pdf/                     # your source PDFs (not in git)
    processed/extracted_text/    # extracted text (generated)
    chunks/                      # chunked JSONL (generated)
    chroma/                      # vector index (generated)
  src/
    ingest/
      extract.py                 # PDF → extracted text
      clean.py                   # text cleaning
      chunk.py                   # text → chunks with metadata
    retrieval/
      index.py                   # chunks → ChromaDB
      retrieve.py                # query → top-k chunks
    generation/
      generate.py                # chunks + query → Gemma answer
  prompts/
    system_prompt.txt            # system instructions for Gemma
  tests/                         # pytest test suite
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

Note: generation tests require Ollama to be running with the configured model available.

---

## Tech stack

| Component | Tool |
|---|---|
| Local LLM | Gemma 2 2B via Ollama (`gemma2:2b`) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector DB | ChromaDB |
| PDF extraction | PyMuPDF |
| Validation | Pydantic |
| Testing | pytest |

---

## Important disclaimers

- This assistant is **not a mufti** and does not issue binding fatwas
- Answers are based solely on the source texts provided — it cannot answer beyond its corpus
- High-risk topics (divorce, inheritance, oaths, medical necessity, etc.) are routed to scholar referral
- Always verify important rulings with a qualified Hanafi scholar
