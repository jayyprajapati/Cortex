import fitz  # PyMuPDF


# -------------------------------
# OPTIONAL: unstructured (for complex PDFs)
# -------------------------------
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False


# -------------------------------
# OPTIONAL: DOCX
# -------------------------------
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# -------------------------------
# OPTIONAL: Markdown
# -------------------------------
try:
    import markdown
    from bs4 import BeautifulSoup
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


# -------------------------------
# SIMPLE PDF PARSER (PyMuPDF)
# -------------------------------
def parse_pdf_simple(path):
    doc = fitz.open(path)
    elements = []

    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")

        for block in blocks:
            text = block[4].strip()
            if not text:
                continue

            # normalize wrapped lines
            text = " ".join(line.strip() for line in text.splitlines() if line.strip())

            elements.append({
                "text": text,
                "page": i + 1
            })

    return elements


# -------------------------------
# COMPLEX PDF PARSER (Unstructured - OPTIONAL)
# -------------------------------
def parse_pdf_complex(path):

    if not UNSTRUCTURED_AVAILABLE:
        print("⚠️ Unstructured not available, falling back to PyMuPDF")
        return parse_pdf_simple(path)

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


# -------------------------------
# DOCX PARSER (OPTIONAL)
# -------------------------------
def parse_docx(path):

    if not DOCX_AVAILABLE:
        raise ImportError("python-docx is not installed")

    doc = Document(path)

    paragraphs = []

    for p in doc.paragraphs:
        text = p.text.strip()

        if text:
            paragraphs.append({
                "text": text
            })

    return paragraphs


# -------------------------------
# MARKDOWN PARSER (OPTIONAL)
# -------------------------------
def parse_markdown(path):

    if not MARKDOWN_AVAILABLE:
        raise ImportError("markdown or beautifulsoup4 not installed")

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