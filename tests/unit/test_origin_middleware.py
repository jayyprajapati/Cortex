"""Unit tests for OriginReferer middleware matching logic.

Note: _origin_matches(origin, allowed) takes the origin string and a list of
allowed patterns. Tests pass a single-element list to verify specific patterns.
"""
from cortex.middleware.origin import _origin_matches


def test_exact_match():
    assert _origin_matches("http://localhost:5173", ["http://localhost:5173"]) is True


def test_exact_match_no_port():
    assert _origin_matches("https://example.com", ["https://example.com"]) is True


def test_wildcard_single_subdomain():
    assert _origin_matches(
        "https://app.jayprajapati.dev", ["https://*.jayprajapati.dev"]
    ) is True


def test_wildcard_blocks_multi_subdomain():
    # evil.app.jayprajapati.dev has two labels before the suffix — blocked
    assert _origin_matches(
        "https://evil.app.jayprajapati.dev", ["https://*.jayprajapati.dev"]
    ) is False


def test_wildcard_blocks_domain_confusion():
    assert _origin_matches(
        "https://evil.com.jayprajapati.dev", ["https://*.jayprajapati.dev"]
    ) is False


def test_scheme_mismatch():
    assert _origin_matches(
        "http://app.jayprajapati.dev", ["https://*.jayprajapati.dev"]
    ) is False


def test_no_match():
    assert _origin_matches("https://evil.com", ["https://*.jayprajapati.dev"]) is False


def test_empty_origin():
    assert _origin_matches("", ["https://*.jayprajapati.dev"]) is False


def test_matches_any_in_list():
    allowed = ["http://localhost:5173", "https://*.jayprajapati.dev"]
    assert _origin_matches("http://localhost:5173", allowed) is True
    assert _origin_matches("https://app.jayprajapati.dev", allowed) is True


def test_first_pattern_miss_second_hit():
    allowed = ["https://other.com", "https://*.jayprajapati.dev"]
    assert _origin_matches("https://app.jayprajapati.dev", allowed) is True


def test_trailing_slash_stripped():
    # Origin with trailing slash should still match
    assert _origin_matches(
        "http://localhost:5173/", ["http://localhost:5173"]
    ) is True


def test_wildcard_requires_subdomain():
    # The bare domain itself (no subdomain) should not match *.domain.com
    assert _origin_matches(
        "https://jayprajapati.dev", ["https://*.jayprajapati.dev"]
    ) is False
