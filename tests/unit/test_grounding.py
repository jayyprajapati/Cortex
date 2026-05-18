"""Unit tests for grounding verification."""
from app.pipeline.generate_pipeline import _check_grounding


def test_grounding_off():
    result = _check_grounding("Anything", [], "off")
    assert result["grounded"] is True


def test_grounding_off_with_chunks():
    # mode=off always passes, even if answer contains unverifiable tokens
    chunks = [{"text": "unrelated content"}]
    result = _check_grounding("Microsoft had 50000 employees in 2023", chunks, "off")
    assert result["grounded"] is True
    assert result["unverified"] == []


def test_grounding_strict_pass():
    chunks = [{"text": "Apple revenue was $5 billion in 2023"}]
    result = _check_grounding("Apple revenue was $5 billion in 2023", chunks, "strict")
    assert result["grounded"] is True


def test_grounding_strict_fail():
    chunks = [{"text": "revenue was high"}]
    result = _check_grounding("Microsoft had 50000 employees in 2023", chunks, "strict")
    assert result["grounded"] is False
    assert len(result["unverified"]) > 0


def test_grounding_strict_unverified_contains_proper_noun():
    chunks = [{"text": "the company had many employees"}]
    result = _check_grounding("Google had many employees", chunks, "strict")
    assert result["grounded"] is False
    assert "Google" in result["unverified"]


def test_grounding_truthful_tolerates_minor():
    # Truthful mode: tolerate < 15% unverified
    chunks = [{"text": "revenue 5M growth 20% profit 1M customers 500 employees 200"}]
    # All tokens verifiable
    result = _check_grounding("Revenue 5M growth 20% profit 1M", chunks, "truthful")
    assert result["grounded"] is True


def test_grounding_truthful_fails_high_unverified():
    # One chunk with no matching proper nouns — many unverified tokens
    chunks = [{"text": "some generic content without any entity"}]
    answer = "Apple Google Microsoft Amazon Facebook are all unverified here."
    result = _check_grounding(answer, chunks, "truthful")
    assert result["grounded"] is False


def test_grounding_no_chunks_off():
    # When chunks is empty and mode != off, still returns grounded=True per implementation
    result = _check_grounding("Some answer", [], "strict")
    assert result["grounded"] is True


def test_grounding_empty_answer():
    chunks = [{"text": "Apple revenue was $5 billion"}]
    result = _check_grounding("", chunks, "strict")
    assert result["grounded"] is False


def test_grounding_deduplicates_unverified():
    chunks = [{"text": "unrelated text"}]
    # Two separate sentences referencing the same proper noun.
    # The deduplication logic must not repeat the same token in unverified.
    result = _check_grounding("Apple had losses. Apple had gains.", chunks, "strict")
    unverified = result["unverified"]
    # "Apple" should appear at most once in the deduplicated list.
    assert unverified.count("Apple") <= 1


def test_grounding_result_keys():
    chunks = [{"text": "hello world"}]
    result = _check_grounding("Hello", chunks, "strict")
    assert "grounded" in result
    assert "unverified" in result
