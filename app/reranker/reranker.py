from sentence_transformers import CrossEncoder

model = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return model


def rerank(query, chunks):
    model = get_model()
    pairs = [(query, c["text"]) for c in chunks]

    scores = model.predict(pairs)

    for c, score in zip(chunks, scores):
        c["rerank_score"] = float(score)

    # sort by rerank score descending
    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    return chunks