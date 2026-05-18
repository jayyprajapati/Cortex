"""Smoke test: clarification policy evaluation."""
from __future__ import annotations

from app.conversation.clarification import should_clarify, build_clarification_context


def main():
    ambiguous_result = {
        "is_ambiguous": True,
        "clarifying_question": "Which document are you referring to?",
        "entities": [],
        "query_type": "navigational",
        "rewrites": [],
    }
    clear_result = {
        "is_ambiguous": False,
        "clarifying_question": None,
        "entities": ["Python"],
        "query_type": "factual",
        "rewrites": [],
    }

    # "never" policy never clarifies
    assert not should_clarify(ambiguous_result, "never")
    assert not should_clarify(clear_result, "never")

    # "balanced" policy clarifies only when ambiguous
    assert should_clarify(ambiguous_result, "balanced")
    assert not should_clarify(clear_result, "balanced")

    # clarification_reply=True skips clarification even in aggressive mode
    assert not should_clarify(ambiguous_result, "aggressive", clarification_reply=True)

    # "aggressive" clarifies ambiguous + navigational with no entities
    nav_no_entities = {"is_ambiguous": False, "query_type": "navigational", "entities": [], "clarifying_question": None, "rewrites": []}
    assert should_clarify(nav_no_entities, "aggressive")

    # Context building
    ctx = build_clarification_context(ambiguous_result, "tell me about the second one")
    assert ctx["original_query"] == "tell me about the second one"
    assert ctx["clarifying_question"] == "Which document are you referring to?"

    print("PASS: clarification policy tests passed")


if __name__ == "__main__":
    main()
