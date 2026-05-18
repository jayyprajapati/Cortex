"""Unit tests for file upload validation."""
import pytest
from fastapi import HTTPException
from cortex.middleware.upload import validate_upload, MAX_UPLOAD_BYTES


def test_valid_pdf():
    pdf_bytes = b"%PDF-1.4 content here"
    ext = validate_upload("document.pdf", pdf_bytes)
    assert ext == ".pdf"


def test_valid_docx():
    docx_bytes = b"PK\x03\x04" + b"\x00" * 100
    ext = validate_upload("resume.docx", docx_bytes)
    assert ext == ".docx"


def test_valid_txt():
    ext = validate_upload("notes.txt", b"plain text content")
    assert ext == ".txt"


def test_valid_md():
    ext = validate_upload("readme.md", b"# Markdown content")
    assert ext == ".md"


def test_invalid_extension():
    with pytest.raises(HTTPException) as exc_info:
        validate_upload("malware.exe", b"content")
    assert exc_info.value.status_code == 415


def test_invalid_extension_js():
    with pytest.raises(HTTPException) as exc_info:
        validate_upload("script.js", b"console.log('hi')")
    assert exc_info.value.status_code == 415


def test_pdf_magic_mismatch():
    # .pdf extension but content is not a PDF
    with pytest.raises(HTTPException) as exc_info:
        validate_upload("fake.pdf", b"this is not a pdf")
    assert exc_info.value.status_code == 415


def test_docx_magic_mismatch():
    # .docx extension but content is not a ZIP/PK file
    with pytest.raises(HTTPException) as exc_info:
        validate_upload("fake.docx", b"not a zip file")
    assert exc_info.value.status_code == 415


def test_missing_filename():
    with pytest.raises(HTTPException) as exc_info:
        validate_upload("", b"content")
    assert exc_info.value.status_code == 400


def test_case_insensitive_extension():
    pdf_bytes = b"%PDF-1.4 content"
    ext = validate_upload("DOC.PDF", pdf_bytes)
    assert ext == ".pdf"


def test_case_insensitive_extension_docx():
    docx_bytes = b"PK\x03\x04" + b"\x00" * 50
    ext = validate_upload("RESUME.DOCX", docx_bytes)
    assert ext == ".docx"


def test_max_upload_bytes_is_positive():
    assert MAX_UPLOAD_BYTES > 0


def test_pdf_with_exact_magic_prefix():
    # Minimal valid magic: exactly the 5-byte PDF header
    pdf_bytes = b"%PDF-" + b"remainder"
    ext = validate_upload("file.pdf", pdf_bytes)
    assert ext == ".pdf"
