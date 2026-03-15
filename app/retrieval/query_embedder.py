from sentence_transformers import SentenceTransformer
# This module defines the function to embed a user query into a vector representation using a pre-trained sentence transformer model.
# The embed_query function takes a query string as input, prepends the word "query: " to the text to provide context for the embedding, 
# and then encodes it using the model to generate a vector representation. This embedding can then be used to perform similarity searches against the embedded chunks in the vector database, 
# allowing the system to retrieve relevant information based on the user's query. The use of a pre-trained model ensures that the embeddings capture semantic meaning, 
# which is crucial for effective retrieval in question-answering tasks.
model = SentenceTransformer("BAAI/bge-small-en")

def embed_query(query):

    query_text = "query: " + query

    return model.encode(query_text)