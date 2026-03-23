from threading import Lock


user_usage = {}
_usage_lock = Lock()


def _normalize_user_id(user_id):
    normalized = str(user_id).strip() if user_id is not None else ""
    if not normalized:
        raise ValueError("user_id is required")
    return normalized


def get_usage(user_id):
    normalized_user_id = _normalize_user_id(user_id)

    with _usage_lock:
        usage = user_usage.setdefault(normalized_user_id, {"docs": 0, "queries": 0})
        return dict(usage)


def increment_docs(user_id):
    normalized_user_id = _normalize_user_id(user_id)

    with _usage_lock:
        usage = user_usage.setdefault(normalized_user_id, {"docs": 0, "queries": 0})
        usage["docs"] += 1
        return dict(usage)


def increment_queries(user_id):
    normalized_user_id = _normalize_user_id(user_id)

    with _usage_lock:
        usage = user_usage.setdefault(normalized_user_id, {"docs": 0, "queries": 0})
        usage["queries"] += 1
        return dict(usage)
