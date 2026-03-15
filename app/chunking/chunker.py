# This module contains the logic to create text chunks from the extracted elements of a document. 
# The create_chunks function takes a list of text elements, the document ID, and a maximum token limit for each chunk. 
# It iterates through the elements, concatenating text until the token count exceeds the specified limit. 
# When the limit is reached, it creates a new Chunk object with the accumulated text, the document ID, the page number, and a unique chunk ID. 
# This process continues until all elements have been processed, resulting in a list of Chunk objects that can be embedded and stored in a vector database for efficient retrieval during question-answering tasks. 
# The use of token counting ensures that the chunks are appropriately sized for input into language models that have token limits, optimizing the performance of the system.
from .models import Chunk
from .tokenizer import token_count


def is_heading(text: str) -> bool:
    text = text.strip()
    if not text:
        return False

    words = text.split()

    if len(words) > 12:
        return False

    if text.endswith((".", "!", "?", ":", ";")):
        return False

    capitalized = sum(1 for w in words if w[:1].isupper())
    return capitalized >= max(1, len(words) // 2)


def should_skip_front_matter(text: str) -> bool:
    text = text.strip()
    if not text:
        return True

    lower = text.lower()

    junk_patterns = [
        "sample document content",
        "title:",
    ]

    if any(lower.startswith(p) for p in junk_patterns):
        return True

    if len(text.split()) <= 5:
        return True

    return False


def group_into_sections(elements):
    sections = []
    current_section = None

    for el in elements:
        text = el["text"].strip()
        page = el.get("page")

        if not text:
            continue

        lower = text.lower()

        # skip obvious front matter
        if lower == "sample document content" or lower.startswith("title:"):
            continue

        if is_heading(text):
            if current_section:
                sections.append(current_section)

            current_section = {
                "heading": text,
                "page": page,
                "paragraphs": []
            }
            continue

        if current_section is None:
            current_section = {
                "heading": None,
                "page": page,
                "paragraphs": []
            }

        current_section["paragraphs"].append(text)

    if current_section:
        sections.append(current_section)

    return sections


def create_chunks(elements, doc_id, max_tokens=150, min_chunk_tokens=40):
    sections = group_into_sections(elements)

    chunks = []
    chunk_id = 0

    for section in sections:
        section_chunks = []

        current_parts = []
        current_tokens = 0

        heading = section["heading"]
        page = section["page"]

        if heading:
            heading_tokens = token_count(heading)
            current_parts.append(heading)
            current_tokens += heading_tokens

        for para in section["paragraphs"]:
            para = para.strip()
            if not para:
                continue

            para_tokens = token_count(para)

            if current_tokens > 0 and current_tokens + para_tokens > max_tokens:
                chunk_text = "\n\n".join(current_parts).strip()

                if chunk_text:
                    section_chunks.append({
                        "text": chunk_text,
                        "tokens": current_tokens
                    })

                current_parts = []
                current_tokens = 0

                if heading:
                    current_parts.append(heading)
                    current_tokens += token_count(heading)

            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            chunk_text = "\n\n".join(current_parts).strip()
            if chunk_text:
                section_chunks.append({
                    "text": chunk_text,
                    "tokens": current_tokens
                })

        if len(section_chunks) > 1 and section_chunks[-1]["tokens"] < min_chunk_tokens:
            section_chunks[-2]["text"] += "\n\n" + section_chunks[-1]["text"]
            section_chunks[-2]["tokens"] += section_chunks[-1]["tokens"]
            section_chunks.pop()

        for sc in section_chunks:
            chunks.append(
                Chunk(
                    text=sc["text"],
                    doc_id=doc_id,
                    page=page,
                    chunk_id=chunk_id,
                    section=heading,
                )
            )
            chunk_id += 1

    return chunks