"""
extract.py

Reads PDFs from data/raw/pdf/ and extracts text page by page using PyMuPDF.
Saves one JSON file per PDF into data/processed/extracted_text/.

Output format per file:
{
    "source_title": "Al Hidayah",
    "file_name": "Al Hidayah.pdf",
    "total_pages": 120,
    "pages": [
        { "page": 1, "text": "..." },
        ...
    ]
}
"""

import json
import os
from pathlib import Path

import pymupdf as fitz
from dotenv import load_dotenv

load_dotenv()

# Directories from .env (with fallbacks)
RAW_PDF_DIR = Path(os.getenv("RAW_PDF_DIR", "data/raw/pdf"))
EXTRACTED_TEXT_DIR = Path(os.getenv("EXTRACTED_TEXT_DIR", "data/processed/extracted_text"))


def extract_text_from_pdf(pdf_path: Path) -> dict:
    """
    Opens a single PDF and extracts text from each page.

    Returns a dict with source title, file name, total pages,
    and a list of { page, text } entries.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Skip completely empty pages
        if text.strip():
            pages.append({
                "page": page_num + 1,  # 1-indexed for human readability
                "text": text
            })

    total_pages = len(doc)
    doc.close()

    return {
        "source_title": pdf_path.stem,  # filename without extension
        "file_name": pdf_path.name,
        "total_pages": total_pages,
        "pages": pages
    }


def save_extracted(data: dict, output_dir: Path) -> Path:
    """
    Saves the extracted document dict as a JSON file.
    Returns the path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the source title as the output filename
    safe_name = data["source_title"].replace(" ", "_")
    output_path = output_dir / f"{safe_name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def run_extraction(pdf_dir: Path = RAW_PDF_DIR, output_dir: Path = EXTRACTED_TEXT_DIR):
    """
    Finds all PDFs in pdf_dir and extracts text from each one.
    Saves results to output_dir.
    """
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return []

    print(f"Found {len(pdf_files)} PDF(s) to process.\n")
    results = []

    for pdf_path in pdf_files:
        print(f"Extracting: {pdf_path.name} ...")

        data = extract_text_from_pdf(pdf_path)
        pages_extracted = len(data["pages"])

        # Warn if no text was found — likely a scanned/image-based PDF
        if pages_extracted == 0:
            print(f"  WARNING: No text extracted from {pdf_path.name}.")
            print(f"  This PDF may be scanned or image-based. OCR required.\n")
            results.append({
                "source_title": data["source_title"],
                "pages_extracted": 0,
                "output_path": None,
                "warning": "no_text_extracted"
            })
            continue

        output_path = save_extracted(data, output_dir)
        print(f"  Saved {pages_extracted} pages → {output_path}\n")

        results.append({
            "source_title": data["source_title"],
            "pages_extracted": pages_extracted,
            "output_path": str(output_path)
        })

    print("Extraction complete.")
    return results


if __name__ == "__main__":
    run_extraction()
