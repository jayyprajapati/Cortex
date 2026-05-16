"""File upload validation utilities."""
import os

from fastapi import HTTPException

_ALLOWED_EXTENSIONS = frozenset([".pdf", ".docx", ".txt", ".md"])

# Magic bytes: maps extension -> (expected_magic_bytes, offset)
_MAGIC = {
    ".pdf": (b"%PDF-", 0),
    ".docx": (b"PK\x03\x04", 0),
}

MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))


async def read_upload_with_size_limit(file_obj) -> bytes:
    """Read an uploaded file while enforcing MAX_UPLOAD_BYTES.

    Reads in 64 KB chunks to avoid loading the entire file into memory before
    checking the size.  Raises HTTPException(413) if the file exceeds the limit.
    """
    chunks = []
    total = 0
    chunk_size = 65536  # 64 KB
    while True:
        chunk = await file_obj.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Upload exceeds maximum allowed size of "
                    f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def validate_upload(filename: str, file_bytes: bytes) -> str:
    """Validate file extension and magic bytes.  Returns the lowercase extension.

    Raises:
        HTTPException(400): filename is missing.
        HTTPException(415): extension not in the allow-list, or magic bytes do
                            not match the declared extension.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            ),
        )

    if ext in _MAGIC:
        magic_bytes, offset = _MAGIC[ext]
        if file_bytes[offset : offset + len(magic_bytes)] != magic_bytes:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"File content does not match extension '{ext}'. "
                    f"Upload may be corrupt or misnamed."
                ),
            )

    return ext
