"""Unit tests for BlockDocsInProduction middleware."""
from cortex.middleware.docs_block import _is_docs_path, _is_dev_host


# ---------------------------------------------------------------------------
# _is_docs_path
# ---------------------------------------------------------------------------

def test_docs_paths_blocked():
    assert _is_docs_path("/docs") is True
    assert _is_docs_path("/docs/oauth2-redirect") is True
    assert _is_docs_path("/redoc") is True
    assert _is_docs_path("/openapi.json") is True
    assert _is_docs_path("/openapi") is True


def test_docs_subpath_blocked():
    assert _is_docs_path("/docs/anything") is True
    assert _is_docs_path("/redoc/sub") is True


def test_non_docs_paths_allowed():
    assert _is_docs_path("/health") is False
    assert _is_docs_path("/chat") is False
    assert _is_docs_path("/api/docs") is False  # doesn't start with /docs
    assert _is_docs_path("/") is False
    assert _is_docs_path("/query") is False


def test_docs_path_not_confused_with_prefix():
    # /documentation is NOT a docs path (it's not /docs, /docs/, etc.)
    assert _is_docs_path("/documentation") is False


# ---------------------------------------------------------------------------
# _is_dev_host
# ---------------------------------------------------------------------------

def test_dev_hosts_allowed():
    assert _is_dev_host("localhost") is True
    assert _is_dev_host("127.0.0.1") is True
    assert _is_dev_host("0.0.0.0") is True
    assert _is_dev_host("localhost:8000") is True


def test_dev_host_with_various_ports():
    assert _is_dev_host("localhost:3000") is True
    assert _is_dev_host("127.0.0.1:5000") is True


def test_prod_host_blocked():
    assert _is_dev_host("cortex.jp.dev") is False
    assert _is_dev_host("cortex.jp.dev:8000") is False  # port bypass fix
    assert _is_dev_host("example.com") is False


def test_empty_host_blocked():
    assert _is_dev_host("") is False


def test_case_insensitive_host():
    # _is_dev_host lowercases before comparison
    assert _is_dev_host("LOCALHOST") is True
    assert _is_dev_host("Localhost:8000") is True
