from app.pipeline.retrieve_pipeline import retrieve
from app.llm.factory import get_llm


def build_prompt(query, chunks):
    context_blocks = []

    for i, c in enumerate(chunks):
        context_blocks.append(
            f"[Source {i+1} | Section: {c.get('section')} | Page: {c.get('page')}]\n{c['text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
        You are a precise and reliable assistant.

        You MUST follow these rules:
        - Answer ONLY using the provided context.
        - DO NOT use any outside knowledge.
        - If the answer is not clearly present, dont use your own knowledge to fill in the gaps.
        - Keep the answer concise but complete.
        - ALWAYS include source references inline using [Source X].

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