# This module contains the logic to create text chunks from the extracted elements of a document. 
# The create_chunks function takes a list of text elements, the document ID, and a maximum token limit for each chunk. 
# It iterates through the elements, concatenating text until the token count exceeds the specified limit. 
# When the limit is reached, it creates a new Chunk object with the accumulated text, the document ID, the page number, and a unique chunk ID. 
# This process continues until all elements have been processed, resulting in a list of Chunk objects that can be embedded and stored in a vector database for efficient retrieval during question-answering tasks. 
# The use of token counting ensures that the chunks are appropriately sized for input into language models that have token limits, optimizing the performance of the system.
from .tokenizer import token_count
from .models import Chunk


def create_chunks(elements, doc_id, max_tokens=500):

    chunks = []
    current_text = ""
    chunk_id = 0
    current_page = None

    for el in elements:

        text = el["text"].strip()

        if not text:
            continue

        page = el.get("page", None)

        if token_count(current_text + text) < max_tokens:

            current_text += "\n" + text
            current_page = page

        else:

            chunks.append(
                Chunk(
                    text=current_text.strip(),
                    doc_id=doc_id,
                    page=current_page,
                    chunk_id=chunk_id
                )
            )

            chunk_id += 1
            current_text = text
            current_page = page

    if current_text:

        chunks.append(
            Chunk(
                text=current_text.strip(),
                doc_id=doc_id,
                page=current_page,
                chunk_id=chunk_id
            )
        )

    return chunks