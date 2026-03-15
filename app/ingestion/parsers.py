import fitz
# This module provides functions to parse simple documents and extract text content using PyMuPDF.
import fitz


def parse_pdf_simple(path):
    doc = fitz.open(path)
    elements = []

    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")

        for block in blocks:
            text = block[4].strip()
            if not text:
                continue

            # normalize wrapped lines inside the same block
            text = " ".join(line.strip() for line in text.splitlines() if line.strip())

            elements.append({
                "text": text,
                "page": i + 1
            })

    return elements

# For more complex documents, we can use the unstructured library to extract text and metadata from PDFs.
from unstructured.partition.pdf import partition_pdf

def parse_pdf_complex(path):

    elements = partition_pdf(path)

    docs = []

    for el in elements:
        if el.text:
            docs.append({
                "text": el.text,
                "type": str(type(el)),
                "page": el.metadata.page_number
            })

    return docs

# For Word documents, we can use the python-docx library to extract text from paragraphs.
from docx import Document

def parse_docx(path):

    doc = Document(path)

    paragraphs = []

    for p in doc.paragraphs:
        text = p.text.strip()

        if text:
            paragraphs.append({
                "text": text
            })

    return paragraphs

# For Markdown files, we can convert the Markdown to HTML and then use BeautifulSoup to extract text from paragraphs and list items.
import markdown
from bs4 import BeautifulSoup

def parse_markdown(path):

    with open(path, "r") as f:
        text = f.read()

    html = markdown.markdown(text)

    soup = BeautifulSoup(html, "html.parser")

    paragraphs = []

    for p in soup.find_all(["p", "li"]):
        paragraphs.append({
            "text": p.get_text()
        })

    return paragraphs