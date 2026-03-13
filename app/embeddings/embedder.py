# This module is responsible for generating vector embeddings from text chunks using a pre-trained sentence transformer model. 
# The embed_chunks function takes a list of Chunk objects, extracts the text from each chunk, and uses the SentenceTransformer model to encode the texts into vector representations. 
# These vectors can then be stored in a vector database for efficient retrieval during question-answering tasks. 
# The use of a pre-trained model like "BAAI/bge-small-en" allows for high-quality embeddings that capture the semantic meaning of the text, improving the performance of the retrieval system when matching user queries to relevant chunks of information. 
# This module is a crucial part of the pipeline that transforms raw text data into a format that can be effectively used for similarity search and retrieval in a RAG (Retrieval-Augmented Generation) system.
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en")

def embed_chunks(chunks):

    texts = [c.text for c in chunks]

    vectors = model.encode(texts)

    return vectors