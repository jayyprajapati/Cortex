from dataclasses import dataclass

# This module defines the data structure for a text chunk extracted from a document. 
# Each chunk contains metadata about the text content, the document ID it belongs to, the page number, and a unique chunk ID. 
# This structure is used to organize and manage the chunks of text that will be embedded and stored in a vector database for retrieval during question-answering tasks.
@dataclass
class Chunk:
    text: str
    doc_id: str
    page: int
    chunk_id: int