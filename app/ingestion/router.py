import os

## used to route files to the correct ingestor based on file type
def detect_file_type(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return "pdf"

    if ext == ".docx":
        return "docx"

    if ext == ".md":
        return "markdown"

    raise ValueError("Unsupported file type")