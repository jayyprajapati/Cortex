# This module provides a unified interface to load documents of various types (PDF, DOCX, Markdown) and route them to the appropriate parser based on their complexity.
from .router import detect_file_type
from .inspector import inspect_pdf, is_complex
from .parsers import (
    parse_pdf_simple,
    parse_pdf_complex,
    parse_docx,
    parse_markdown
)

def load_document(path):

    file_type = detect_file_type(path)

    if file_type == "pdf":

        meta = inspect_pdf(path)
        print("Document metadata:", meta)
        if is_complex(meta):
            print("Using Unstructured parser")
            return parse_pdf_complex(path)
        
        print("Using PyMuPDF parser")
        return parse_pdf_simple(path)

    if file_type == "docx":
        print("Using DOCX parser")
        return parse_docx(path)

    if file_type == "markdown":
        print("Using Markdown parser")
        return parse_markdown(path)