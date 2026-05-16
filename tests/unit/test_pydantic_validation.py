"""Unit tests for registry model validation."""
import pytest
from pydantic import ValidationError
from app.registry.models import (
    RetrievalConfig, GenerationConfig, ChunkingConfig, RerankingConfig
)


# ---------------------------------------------------------------------------
# RetrievalConfig
# ---------------------------------------------------------------------------

def test_retrieval_top_k_bounds():
    with pytest.raises(ValidationError):
        RetrievalConfig(top_k=0)
    with pytest.raises(ValidationError):
        RetrievalConfig(top_k=101)
    cfg = RetrievalConfig(top_k=10)
    assert cfg.top_k == 10


def test_retrieval_rrf_k_default():
    cfg = RetrievalConfig()
    assert cfg.rrf_k == 60


def test_retrieval_alpha_bounds():
    with pytest.raises(ValidationError):
        RetrievalConfig(alpha=-0.1)
    with pytest.raises(ValidationError):
        RetrievalConfig(alpha=1.1)
    cfg = RetrievalConfig(alpha=0.5)
    assert cfg.alpha == 0.5


def test_retrieval_defaults():
    cfg = RetrievalConfig()
    assert cfg.top_k == 10
    assert cfg.fusion == "rrf"
    assert cfg.query_rewrite is True


# ---------------------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------------------

def test_generation_temperature_bounds():
    with pytest.raises(ValidationError):
        GenerationConfig(temperature=-0.1)
    with pytest.raises(ValidationError):
        GenerationConfig(temperature=2.1)
    cfg = GenerationConfig(temperature=0.5)
    assert cfg.temperature == 0.5


def test_generation_temperature_boundary_values():
    # Exactly 0.0 and 2.0 must be valid
    cfg_min = GenerationConfig(temperature=0.0)
    assert cfg_min.temperature == 0.0
    cfg_max = GenerationConfig(temperature=2.0)
    assert cfg_max.temperature == 2.0


def test_generation_grounding_threshold_default():
    cfg = GenerationConfig()
    assert cfg.grounding_unverified_threshold == 0.15


def test_generation_grounding_mode_default():
    cfg = GenerationConfig()
    assert cfg.grounding_mode == "off"


def test_generation_max_retries_bounds():
    with pytest.raises(ValidationError):
        GenerationConfig(max_retries=-1)
    with pytest.raises(ValidationError):
        GenerationConfig(max_retries=6)
    cfg = GenerationConfig(max_retries=3)
    assert cfg.max_retries == 3


def test_generation_response_type_values():
    for rt in ("markdown", "json", "text"):
        cfg = GenerationConfig(response_type=rt)
        assert cfg.response_type == rt
    with pytest.raises(ValidationError):
        GenerationConfig(response_type="xml")


# ---------------------------------------------------------------------------
# ChunkingConfig
# ---------------------------------------------------------------------------

def test_chunking_min_less_than_max():
    with pytest.raises(ValidationError):
        ChunkingConfig(min_tokens=200, max_tokens=100)


def test_chunking_min_equal_to_max_invalid():
    with pytest.raises(ValidationError):
        ChunkingConfig(min_tokens=200, max_tokens=200)


def test_chunking_valid_config():
    cfg = ChunkingConfig(min_tokens=64, max_tokens=512)
    assert cfg.min_tokens == 64
    assert cfg.max_tokens == 512


def test_chunking_max_tokens_bounds():
    with pytest.raises(ValidationError):
        ChunkingConfig(max_tokens=63)
    with pytest.raises(ValidationError):
        ChunkingConfig(max_tokens=4097)


def test_chunking_min_tokens_bounds():
    with pytest.raises(ValidationError):
        ChunkingConfig(min_tokens=9, max_tokens=512)
    with pytest.raises(ValidationError):
        ChunkingConfig(min_tokens=513, max_tokens=1024)


# ---------------------------------------------------------------------------
# RerankingConfig
# ---------------------------------------------------------------------------

def test_reranking_top_k_lte_candidate_cap():
    with pytest.raises(ValidationError):
        RerankingConfig(enabled=True, top_k=50, candidate_cap=20)


def test_reranking_top_k_lte_candidate_cap_disabled():
    # When disabled, the cross-validator should not fire
    cfg = RerankingConfig(enabled=False, top_k=50, candidate_cap=20)
    assert cfg.top_k == 50
    assert cfg.candidate_cap == 20


def test_reranking_valid():
    cfg = RerankingConfig(enabled=True, top_k=5, candidate_cap=20)
    assert cfg.top_k == 5
    assert cfg.candidate_cap == 20


def test_reranking_top_k_equals_cap_valid():
    cfg = RerankingConfig(enabled=True, top_k=20, candidate_cap=20)
    assert cfg.top_k == 20


def test_reranking_diversity_bounds():
    with pytest.raises(ValidationError):
        RerankingConfig(diversity=-0.1)
    with pytest.raises(ValidationError):
        RerankingConfig(diversity=1.1)
    cfg = RerankingConfig(diversity=0.5)
    assert cfg.diversity == 0.5
