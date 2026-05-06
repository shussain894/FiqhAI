"""
test_extract.py

Tests for the PDF extraction module.
Uses a real PDF from data/raw/pdf/ to verify extraction works end-to-end.
"""

import json
from pathlib import Path

import pytest

from src.ingest.extract import extract_text_from_pdf, save_extracted, run_extraction

RAW_PDF_DIR = Path("data/raw/pdf")
EXTRACTED_TEXT_DIR = Path("data/processed/extracted_text")


def get_first_pdf() -> Path:
    """Returns the first PDF found in the raw directory, or None."""
    pdfs = list(RAW_PDF_DIR.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def test_pdf_dir_exists():
    """The raw PDF directory should exist."""
    assert RAW_PDF_DIR.exists(), f"Missing directory: {RAW_PDF_DIR}"


def test_pdfs_present():
    """There should be at least one PDF to process."""
    pdfs = list(RAW_PDF_DIR.glob("*.pdf"))
    assert len(pdfs) > 0, "No PDFs found in data/raw/pdf/"


def test_extract_returns_expected_keys():
    """Extracted data should have the required top-level keys."""
    pdf = get_first_pdf()
    if pdf is None:
        pytest.skip("No PDF available for testing")

    data = extract_text_from_pdf(pdf)

    assert "source_title" in data
    assert "file_name" in data
    assert "total_pages" in data
    assert "pages" in data


def test_extract_has_pages():
    """Extracted data should contain at least one non-empty page."""
    pdf = get_first_pdf()
    if pdf is None:
        pytest.skip("No PDF available for testing")

    data = extract_text_from_pdf(pdf)

    assert len(data["pages"]) > 0, "No pages extracted from PDF"


def test_page_has_text_and_number():
    """Each page entry should have a page number and non-empty text."""
    pdf = get_first_pdf()
    if pdf is None:
        pytest.skip("No PDF available for testing")

    data = extract_text_from_pdf(pdf)
    first_page = data["pages"][0]

    assert "page" in first_page
    assert "text" in first_page
    assert isinstance(first_page["page"], int)
    assert len(first_page["text"].strip()) > 0


def test_save_extracted_creates_file(tmp_path):
    """save_extracted should write a valid JSON file."""
    pdf = get_first_pdf()
    if pdf is None:
        pytest.skip("No PDF available for testing")

    data = extract_text_from_pdf(pdf)
    output_path = save_extracted(data, tmp_path)

    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["source_title"] == data["source_title"]
    assert len(loaded["pages"]) == len(data["pages"])


def test_run_extraction_processes_all_pdfs():
    """
    run_extraction should return one result per PDF.
    PDFs with no extractable text (scanned/image-based) are warned about
    but do not cause a failure — they get pages_extracted = 0.
    At least one PDF must yield text for the test to pass.
    """
    pdfs = list(RAW_PDF_DIR.glob("*.pdf"))
    results = run_extraction(pdf_dir=RAW_PDF_DIR, output_dir=EXTRACTED_TEXT_DIR)

    # One result entry per PDF
    assert len(results) == len(pdfs)

    for result in results:
        assert "source_title" in result
        assert "pages_extracted" in result

    # At least one PDF must have extractable text
    successful = [r for r in results if r["pages_extracted"] > 0]
    assert len(successful) > 0, "No PDFs yielded any extractable text"

    # Log which PDFs had no text (scanned)
    failed = [r for r in results if r["pages_extracted"] == 0]
    for r in failed:
        print(f"  [WARNING] {r['source_title']} — no text extracted (likely scanned)")
