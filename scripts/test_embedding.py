from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en")

vector = model.encode("What is machine learning?")

print(len(vector))