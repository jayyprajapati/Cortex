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


def generate_answer(query, llm_config, user_id, doc_id=None):

    chunks = retrieve(query, user_id=user_id, doc_id=doc_id)

    prompt = build_prompt(query, chunks)

    llm = get_llm(llm_config)

    answer = llm.generate(prompt)

    return {
        "answer": answer,
        "sources": chunks
    }