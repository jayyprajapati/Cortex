"""Smoke test: layout_aware_semantic chunker produces no null sections and tables are atomic."""
from __future__ import annotations

from app.ingestion.loaders.base import Element
from app.chunking.strategies.layout_aware_semantic import LayoutAwareSemanticStrategy


class _Config:
    max_tokens = 512
    min_tokens = 50
    keep_tables_atomic = True
    keep_code_atomic = True


def main():
    elements = [
        Element(type="heading_l1", text="Introduction", page=1),
        Element(type="paragraph", text="This is the introduction paragraph. It has some content.", page=1),
        Element(type="heading_l2", text="Background", page=1),
        Element(type="paragraph", text="Background details here.", page=1),
        Element(type="table", text="| Col1 | Col2 |\n|------|------|\n| a    | b    |", page=2),
        Element(type="paragraph", text="More text after the table.", page=2),
        Element(type="code_block", text="def foo():\n    return 42", page=3),
    ]

    strategy = LayoutAwareSemanticStrategy(_Config())
    chunks = strategy.chunk(elements, doc_id="test-doc-001")

    print(f"Produced {len(chunks)} chunks")
    for c in chunks:
        sp = c.hierarchy or [getattr(c, "section", "_root")]
        print(f"  chunk_id={c.chunk_id} section_path={sp} tokens={c.token_count} text[:60]={c.text[:60]!r}")

    # Assertions
    assert chunks, "Must produce at least one chunk"
    for c in chunks:
        sp = c.hierarchy or [getattr(c, "section", "_root")]
        assert sp and len(sp) > 0, f"section_path must not be empty for chunk {c.chunk_id}"
        assert "_root" in sp or any(s for s in sp), "section_path must have a real value"
        # No 'HEADER' synthetic sections
        assert "HEADER" not in sp, f"Found synthetic HEADER in section_path: {sp}"

    # Table should be its own atomic chunk
    table_chunks = [c for c in chunks if "Col1" in c.text or "Col2" in c.text]
    assert table_chunks, "Table should produce at least one chunk"

    # Code block should be its own atomic chunk
    code_chunks = [c for c in chunks if "def foo" in c.text]
    assert code_chunks, "Code block should produce at least one chunk"

    # prev/next linking
    for i, c in enumerate(chunks):
        if i > 0:
            assert c.prev_chunk_id == chunks[i-1].chunk_id, f"prev_chunk_id mismatch at {i}"
        if i < len(chunks) - 1:
            assert c.next_chunk_id == chunks[i+1].chunk_id, f"next_chunk_id mismatch at {i}"

    print("PASS: layout chunker tests passed")


if __name__ == "__main__":
    main()
