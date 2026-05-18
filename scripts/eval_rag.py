"""
RAG quality evaluation harness.

Usage:
    python3 -m scripts.eval_rag --suite doclens
    python3 -m scripts.eval_rag --suite resumelab
    python3 -m scripts.eval_rag --suite all

Exit code 1 if any required check fails.

Requirements before running:
    DocLens suite:
        eval/golden/doclens/resume.pdf      — a real resume PDF to index
        eval/golden/doclens/questions.yaml  — golden question set (already committed)
        A running Cortex server at CORTEX_URL (default http://localhost:8000)

    ResumeLab suite:
        eval/golden/resumelab/jd.txt        — job description text (committed)
        eval/golden/resumelab/expected.yaml — assertion definitions (committed)
        eval/golden/resumelab/profile.json  — CanonicalProfile JSON
        eval/golden/doclens/resume.pdf      — same PDF (used to create a test profile if
                                              profile.json is missing)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib.util

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests is required. pip install requests", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORTEX_URL = os.environ.get("CORTEX_URL", "http://localhost:8000").rstrip("/")
EVAL_APP   = "doclens"
EVAL_USER  = "eval_harness"
EVAL_COLLECTION = "eval_doclens"
RESULTS_DIR = Path("eval/results")
GOLDEN_DIR  = Path("eval/golden")

# LLM for eval queries — pull from env or fall back to env default (server-side).
EVAL_LLM: Optional[Dict[str, Any]] = None
if os.environ.get("EVAL_LLM_PROVIDER"):
    EVAL_LLM = {
        "provider": os.environ["EVAL_LLM_PROVIDER"],
        "model": os.environ.get("EVAL_LLM_MODEL", ""),
        "api_key": os.environ.get("EVAL_LLM_API_KEY", ""),
    }

STRICT_REFUSAL_SENTINEL = "I couldn't find that in the document."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(path: str, **kwargs) -> requests.Response:
    return requests.post(f"{CORTEX_URL}{path}", **kwargs)

def _get(path: str, **kwargs) -> requests.Response:
    return requests.get(f"{CORTEX_URL}{path}", **kwargs)


def _check_server() -> bool:
    try:
        r = _get("/healthz", timeout=5)
        return r.status_code == 200
    except Exception:
        try:
            r = _get("/", timeout=5)
            return r.status_code < 500
        except Exception:
            return False


def _ingest_pdf(pdf_path: Path, collection: str, doc_id_hint: str = "eval_resume") -> str:
    """Upload a PDF to the eval collection and return the doc_id."""
    print(f"  Ingesting {pdf_path.name} into collection '{collection}' …")
    with open(pdf_path, "rb") as fh:
        r = _post(
            "/ingest",
            files={"file": (pdf_path.name, fh, "application/pdf")},
            data={
                "app_name": EVAL_APP,
                "user_id": EVAL_USER,
                "collection": collection,
            },
            timeout=120,
        )
    if not r.ok:
        raise RuntimeError(f"Ingest failed ({r.status_code}): {r.text[:300]}")
    doc_id = r.json().get("doc_id") or r.json().get("document_id", doc_id_hint)
    print(f"  Ingested → doc_id={doc_id}")
    return doc_id


def _chat(query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "app_name": EVAL_APP,
        "user_id": EVAL_USER,
        "query": query,
        "task": "chat",
    }
    if thread_id:
        body["thread_id"] = thread_id
    if EVAL_LLM:
        body["llm"] = EVAL_LLM

    r = _post("/chat", json=body, timeout=120)
    if not r.ok:
        raise RuntimeError(f"Chat failed ({r.status_code}): {r.text[:300]}")
    return r.json()


def _query(query: str) -> Dict[str, Any]:
    """Single-shot /query (no chat thread)."""
    body: Dict[str, Any] = {
        "app_name": EVAL_APP,
        "user_id": EVAL_USER,
        "query": query,
        "task": "chat",
    }
    if EVAL_LLM:
        body["llm"] = EVAL_LLM
    r = _post("/query", json=body, timeout=120)
    if not r.ok:
        raise RuntimeError(f"Query failed ({r.status_code}): {r.text[:300]}")
    return r.json()


def _answer_text(result: Dict[str, Any]) -> str:
    answer = result.get("answer", "")
    if isinstance(answer, dict):
        answer = answer.get("answer", json.dumps(answer))
    return str(answer).strip()


def _is_refusal(text: str) -> bool:
    return STRICT_REFUSAL_SENTINEL.lower() in text.lower()


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

class CaseResult:
    def __init__(self, case_id: str, suite: str):
        self.id = case_id
        self.suite = suite
        self.checks: List[Dict[str, Any]] = []
        self.error: Optional[str] = None
        self.answer: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(c["passed"] for c in self.checks)

    def add_check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"name": name, "passed": passed, "detail": detail})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "suite": self.suite,
            "passed": self.passed,
            "error": self.error,
            "answer_snippet": (self.answer or "")[:200],
            "checks": self.checks,
        }


def _score_doclens_case(case: Dict[str, Any]) -> CaseResult:
    cid = case.get("id", "unknown")
    result = CaseResult(cid, "doclens")

    try:
        is_multi = "multi_turn" in case
        if is_multi:
            turns: List[str] = case["multi_turn"]
            thread_id: Optional[str] = None
            resp: Dict[str, Any] = {}
            for turn in turns:
                resp = _chat(turn, thread_id=thread_id)
                thread_id = resp.get("thread_id") or thread_id
            answer = _answer_text(resp)
        else:
            resp = _query(case["query"])
            answer = _answer_text(resp)

        result.answer = answer

        # grounded check
        if case.get("grounded_required"):
            grounded = resp.get("grounded", False)
            result.add_check(
                "grounded",
                bool(grounded),
                f"grounded={grounded}"
            )

        # refusal check
        if case.get("expected_refusal"):
            is_ref = _is_refusal(answer)
            result.add_check(
                "expected_refusal",
                is_ref,
                f"answer={answer[:120]!r}"
            )
        elif "expected_refusal" in case and not case["expected_refusal"]:
            # explicitly NOT a refusal
            is_ref = _is_refusal(answer)
            result.add_check(
                "not_a_refusal",
                not is_ref,
                f"answer={answer[:120]!r}"
            )

        # substring checks
        for expected in case.get("expected_answer_contains", []):
            found = expected.lower() in answer.lower()
            result.add_check(
                f"contains:{expected!r}",
                found,
                f"answer snippet: {answer[:120]!r}"
            )

    except Exception as exc:
        result.error = str(exc)

    return result


# ---------------------------------------------------------------------------
# ResumeLab scorer
# ---------------------------------------------------------------------------

def _flatten_field(data: Any, field: str):
    """Navigate dotted paths and return value, or None if missing."""
    parts = field.split(".")
    cur = data
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur


def _score_resumelab_case(case: Dict[str, Any], profile: Dict, jd: str) -> CaseResult:
    cid = case.get("id", "unknown")
    result = CaseResult(cid, "resumelab")

    mode = case.get("mode", "canonical_only")
    tweak = case.get("tweak", "")

    try:
        if mode == "match_only":
            body: Dict[str, Any] = {
                "profile": profile,
                "job_description": jd,
            }
            r = _post("/analyze/match", json=body, timeout=120)
            if not r.ok:
                raise RuntimeError(f"match failed ({r.status_code}): {r.text[:200]}")
            data = r.json()
        else:
            body = {
                "profile": profile,
                "job_description": jd,
                "generation_mode": mode,
            }
            if tweak:
                body["user_tweak_prompt"] = tweak
            r = _post("/generate/document", json=body, timeout=180)
            if not r.ok:
                raise RuntimeError(f"generate failed ({r.status_code}): {r.text[:200]}")
            data = r.json()

        result.answer = json.dumps(data)[:500]

        for assertion in case.get("assertions", []):
            atype = assertion["type"]
            desc  = assertion.get("description", atype)

            if atype == "field_present":
                field = assertion["field"]
                val = _flatten_field(data, field)
                result.add_check(f"field_present:{field}", val is not None, desc)

            elif atype == "field_type":
                field = assertion["field"]
                expected = assertion["expected_type"]
                val = _flatten_field(data, field)
                type_map = {"list": list, "dict": dict, "str": str, "int": int, "float": float}
                ok = isinstance(val, type_map.get(expected, object))
                result.add_check(f"field_type:{field}", ok, f"{field}={type(val).__name__} expected {expected}")

            elif atype == "min_length":
                field = assertion["field"]
                min_len = assertion["min"]
                val = _flatten_field(data, field)
                length = len(val) if isinstance(val, (list, str)) else 0
                ok = length >= min_len
                result.add_check(f"min_length:{field}>={min_len}", ok, f"length={length}")

            elif atype == "numeric_range":
                field = assertion["field"]
                val = _flatten_field(data, field)
                ok = (
                    isinstance(val, (int, float))
                    and assertion.get("min", float("-inf")) <= val <= assertion.get("max", float("inf"))
                )
                result.add_check(f"range:{field}", ok, f"value={val}")

            elif atype == "regex_in_field":
                field = assertion["field"]
                subfield = assertion.get("subfield")
                pattern = assertion["pattern"]
                val = _flatten_field(data, field)
                # Search in all strings within the value
                haystack = []
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and subfield:
                            sub = item.get(subfield, [])
                            if isinstance(sub, list):
                                haystack.extend(str(s) for s in sub)
                            else:
                                haystack.append(str(sub))
                        elif isinstance(item, str):
                            haystack.append(item)
                else:
                    haystack.append(str(val or ""))

                found = any(re.search(pattern, s) for s in haystack)
                result.add_check(f"regex:{pattern}", found, desc)

            elif atype == "grounded":
                grounded = data.get("grounded", True)
                result.add_check("grounded", bool(grounded), desc)

    except Exception as exc:
        result.error = str(exc)

    return result


# ---------------------------------------------------------------------------
# Suite runners
# ---------------------------------------------------------------------------

def run_doclens_suite() -> List[CaseResult]:
    print("\n=== DocLens suite ===")
    questions_path = GOLDEN_DIR / "doclens" / "questions.yaml"
    pdf_path       = GOLDEN_DIR / "doclens" / "resume.pdf"

    if not questions_path.exists():
        print(f"SKIP: {questions_path} not found")
        return []

    if not pdf_path.exists():
        print(
            f"SKIP: {pdf_path} not found.\n"
            "Place a real resume PDF at eval/golden/doclens/resume.pdf to run this suite."
        )
        return []

    cases = yaml.safe_load(questions_path.read_text())

    # Ingest the PDF into an isolated eval collection
    try:
        _ingest_pdf(pdf_path, EVAL_COLLECTION)
    except Exception as e:
        print(f"ERROR during ingest: {e}")
        return []

    results: List[CaseResult] = []
    for case in cases:
        cid = case.get("id", "?")
        print(f"  [{cid}] …", end=" ", flush=True)
        r = _score_doclens_case(case)
        status = "PASS" if r.passed else ("ERROR" if r.error else "FAIL")
        print(status)
        if not r.passed:
            if r.error:
                print(f"         error: {r.error}")
            for c in r.checks:
                if not c["passed"]:
                    print(f"         FAIL {c['name']}: {c['detail']}")
        results.append(r)

    return results


def run_resumelab_suite() -> List[CaseResult]:
    print("\n=== ResumeLab suite ===")
    expected_path = GOLDEN_DIR / "resumelab" / "expected.yaml"
    jd_path       = GOLDEN_DIR / "resumelab" / "jd.txt"
    profile_path  = GOLDEN_DIR / "resumelab" / "profile.json"

    if not expected_path.exists():
        print(f"SKIP: {expected_path} not found")
        return []
    if not jd_path.exists():
        print(f"SKIP: {jd_path} not found")
        return []
    if not profile_path.exists():
        print(
            f"SKIP: {profile_path} not found.\n"
            "Generate a profile.json from a real resume and save it at "
            "eval/golden/resumelab/profile.json to run this suite.\n"
            "Example: POST /extract with your resume, then POST /profile/merge."
        )
        return []

    cases   = yaml.safe_load(expected_path.read_text())
    jd_text = jd_path.read_text().strip()
    profile = json.loads(profile_path.read_text())

    results: List[CaseResult] = []
    for case in cases:
        cid = case.get("id", "?")
        print(f"  [{cid}] …", end=" ", flush=True)
        r = _score_resumelab_case(case, profile, jd_text)
        status = "PASS" if r.passed else ("ERROR" if r.error else "FAIL")
        print(status)
        if not r.passed:
            if r.error:
                print(f"         error: {r.error}")
            for c in r.checks:
                if not c["passed"]:
                    print(f"         FAIL {c['name']}: {c['detail']}")
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Summary + output
# ---------------------------------------------------------------------------

def _print_summary(results: List[CaseResult]) -> bool:
    total  = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print(f"\n{'='*50}")
    print(f"RESULTS  {passed}/{total} passed  |  {failed} failed")

    # per-suite breakdown
    suites: Dict[str, List[CaseResult]] = {}
    for r in results:
        suites.setdefault(r.suite, []).append(r)
    for suite, suite_results in suites.items():
        sp = sum(1 for r in suite_results if r.passed)
        print(f"  {suite}: {sp}/{len(suite_results)}")

    print('='*50)
    return failed == 0


def _save_results(results: List[CaseResult]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": ts,
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "results": [r.to_dict() for r in results],
            },
            indent=2,
        )
    )
    print(f"\nResults saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Cortex RAG quality eval")
    parser.add_argument(
        "--suite",
        choices=["doclens", "resumelab", "all"],
        default="all",
        help="Which suite to run (default: all)",
    )
    args = parser.parse_args()

    if not _check_server():
        print(f"ERROR: Cortex server not reachable at {CORTEX_URL}", file=sys.stderr)
        print("Start with: uvicorn cortex.api.main:app --reload", file=sys.stderr)
        return 1

    all_results: List[CaseResult] = []

    if args.suite in ("doclens", "all"):
        all_results.extend(run_doclens_suite())

    if args.suite in ("resumelab", "all"):
        all_results.extend(run_resumelab_suite())

    if not all_results:
        print("\nNo cases ran. Check that fixture files exist and the server is up.")
        return 0

    _save_results(all_results)
    ok = _print_summary(all_results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
