from __future__ import annotations

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    if not text:
        return 0
    return len(_enc.encode(str(text)))
