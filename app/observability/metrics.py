"""Prometheus metrics for Cortex pipeline."""
from prometheus_client import Counter, Histogram

# Request counters and latency histograms per route
request_count = Counter("cortex_requests_total", "Total requests", ["route", "method", "status"])
request_latency = Histogram("cortex_request_duration_seconds", "Request latency", ["route"])

# Pipeline-specific metrics
retrieval_latency = Histogram("cortex_retrieval_duration_seconds", "Retrieval latency", ["app"])
rerank_latency = Histogram("cortex_rerank_duration_seconds", "Rerank latency", ["app"])
generation_latency = Histogram("cortex_generation_duration_seconds", "Generation latency", ["app"])
token_usage = Counter("cortex_tokens_total", "Token usage", ["app", "type"])
grounding_results = Counter("cortex_grounding_total", "Grounding check results", ["app", "result"])  # result: pass/fail
citation_coverage = Histogram("cortex_citation_coverage", "Citation coverage fraction", ["app"])
clarification_count = Counter("cortex_clarifications_total", "Clarification requests", ["app"])
