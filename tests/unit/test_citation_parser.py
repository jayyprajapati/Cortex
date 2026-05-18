"""Unit tests for citation parsing and validation."""
import pytest
from app.pipeline.generate_pipeline import extract_citations, _validate_citations


def test_extract_citations_basic():
    answer = "The revenue was $5M [1]. Growth was 20% [2]."
    sources = [{"text": "revenue $5M", "section": "S1"}, {"text": "growth 20%", "section": "S2"}]
    citations = extract_citations(answer, sources)
    assert len(citations) == 2
    assert citations[0]["index"] == 1
    assert citations[1]["index"] == 2


def test_extract_citations_out_of_range():
    answer = "See [5] for details."
    sources = [{"text": "foo", "section": "S1"}]
    citations = extract_citations(answer, sources)
    assert citations == []  # index 5 is out of range for 1 source


def test_extract_citations_empty():
    assert extract_citations("", []) == []
    assert extract_citations("no citations here", [{"text": "x", "section": "S1"}]) == []


def test_extract_citations_multi_index():
    answer = "See [1, 2] for details."
    sources = [{"text": "a", "section": "A"}, {"text": "b", "section": "B"}]
    citations = extract_citations(answer, sources)
    assert len(citations) == 2
    assert {c["index"] for c in citations} == {1, 2}


def test_extract_citations_deduplicates():
    answer = "Revenue [1]. Profit [1]. Growth [2]."
    sources = [{"text": "a", "section": "A"}, {"text": "b", "section": "B"}]
    citations = extract_citations(answer, sources)
    # Index 1 appears twice but should only be in output once
    assert len(citations) == 2
    indices = [c["index"] for c in citations]
    assert indices.count(1) == 1


def test_extract_citations_maps_source_fields():
    answer = "Revenue was $5M [1]."
    sources = [{"text": "revenue $5M", "section": "Finance", "page": 3}]
    citations = extract_citations(answer, sources)
    assert len(citations) == 1
    assert citations[0]["section"] == "Finance"
    assert citations[0]["page"] == 3
    assert citations[0]["text"] == "revenue $5M"


def test_validate_citations_full_coverage():
    answer = "Revenue was $5M [1]. Growth was 20% [2]. Profit was $1M [1]."
    result = _validate_citations(answer, num_sources=2, threshold=0.7)
    assert result["valid"] is True
    assert result["coverage"] >= 0.7


def test_validate_citations_out_of_range():
    answer = "See [5] for details."
    result = _validate_citations(answer, num_sources=2)
    assert result["valid"] is False
    assert 5 in result["out_of_range"]


def test_validate_citations_below_threshold():
    answer = "Revenue was $5M. Growth was 20%. Profit was $1M [1]."
    result = _validate_citations(answer, num_sources=1, threshold=0.9)
    # Only 1 of 3 sentences has a citation — coverage < 0.9
    assert result["valid"] is False


def test_validate_citations_empty_answer():
    result = _validate_citations("", num_sources=2)
    # No sentences → coverage defaults to 1.0, trivially valid
    assert result["valid"] is True
    assert result["coverage"] == 1.0
    assert result["out_of_range"] == []


def test_validate_citations_coverage_value():
    answer = "First sentence [1]. Second sentence [2]. Third sentence."
    result = _validate_citations(answer, num_sources=2, threshold=0.5)
    # 2 of 3 sentences cited → coverage ~ 0.667
    assert abs(result["coverage"] - round(2 / 3, 3)) < 0.001
