from __future__ import annotations

import logging
from typing import Dict, List

from app.context import ExecutionContext
from app.chunking.tokenizer import token_count

logger = logging.getLogger(__name__)


def expand_neighbors(
    ctx: ExecutionContext,
    chunks: List[Dict],
    budget_tokens: int = 400,
) -> List[Dict]:
    """
    For each chunk that has prev_chunk_id or next_chunk_id, fetch the neighbor chunks
    from the vector store and prepend/append their text up to budget_tokens total.
    Returns the expanded chunks (text may be augmented; original chunk dict mutated in place).
    """
    if not chunks or budget_tokens <= 0:
        return chunks

    vs = ctx.components.vector_store if ctx.components else None
    if vs is None:
        return chunks

    # Collect all neighbor chunk_ids we need to fetch
    neighbor_ids_needed = set()
    for chunk in chunks:
        prev_id = chunk.get("prev_chunk_id")
        next_id = chunk.get("next_chunk_id")
        current_id = chunk.get("chunk_id")
        doc_id = chunk.get("doc_id")
        if doc_id:
            if prev_id is not None:
                neighbor_ids_needed.add((doc_id, prev_id))
            if next_id is not None:
                neighbor_ids_needed.add((doc_id, next_id))

    if not neighbor_ids_needed:
        return chunks

    # Fetch neighbors from vector store by scrolling doc chunks
    neighbor_map: Dict[tuple, str] = {}  # (doc_id, chunk_id) -> text
    seen_docs: set = set()
    for (doc_id, _) in neighbor_ids_needed:
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        try:
            doc_chunks = vs.get_doc_chunks(
                collection=ctx.collection,
                user_id=ctx.user_id,
                doc_id=doc_id,
                limit=500,
            )
            for dc in doc_chunks:
                dc_chunk_id = dc.get("chunk_id")
                if dc_chunk_id is not None:
                    neighbor_map[(doc_id, dc_chunk_id)] = str(dc.get("text") or "").strip()
        except Exception as exc:
            logger.debug("Neighbor fetch failed for doc %s: %s", doc_id, exc)

    # Augment chunks
    tokens_used = 0
    for chunk in chunks:
        if tokens_used >= budget_tokens:
            break
        doc_id = chunk.get("doc_id")
        if not doc_id:
            continue

        prev_text = neighbor_map.get((doc_id, chunk.get("prev_chunk_id")), "")
        next_text = neighbor_map.get((doc_id, chunk.get("next_chunk_id")), "")

        augmented = chunk.get("text", "")
        if prev_text:
            cost = token_count(prev_text)
            if tokens_used + cost <= budget_tokens:
                augmented = prev_text + "\n\n" + augmented
                tokens_used += cost
        if next_text:
            cost = token_count(next_text)
            if tokens_used + cost <= budget_tokens:
                augmented = augmented + "\n\n" + next_text
                tokens_used += cost

        chunk["text"] = augmented

    return chunks
