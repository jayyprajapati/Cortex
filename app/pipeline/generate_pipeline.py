from app.pipeline.retrieve_pipeline import retrieve
from app.llm.factory import get_llm


def build_prompt(query, chunks):
    context = "\n\n".join([c["text"] for c in chunks])

    return f"""
You are a helpful assistant.

Answer the question using ONLY the context below.
Cite the source section and page in your answer.

Context:
{context}

Question:
{query}

Answer:
"""


def generate_answer(query, llm_config):
    # Step 1: retrieve relevant chunks
    chunks = retrieve(query)

    # Step 2: build prompt
    prompt = build_prompt(query, chunks)

    # Step 3: get LLM
    llm = get_llm(llm_config)

    # Step 4: generate answer
    answer = llm.generate(prompt)

    return {
        "answer": answer,
        "sources": chunks
    }