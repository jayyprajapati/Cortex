def build_prompt(query, context):
    return f"""
You are a precise and reliable assistant.

You MUST follow these rules:
- Answer ONLY using the provided context.
- DO NOT use any outside knowledge.
- If the answer is not clearly present, do not use your own knowledge to fill in the gaps.
- Keep the answer concise but complete.
- ALWAYS include source references inline using [Source X].

Context:
{context}

Question:
{query}

Answer:
""".strip()