"""Tool schemas, dispatcher, and deterministic citation verifier.

Design notes
------------
- Two explicit LLM tools: `save_memory` and `search_memory`. The names
  are verbs so the model reliably picks one based on the user's
  intent (statement vs. question).
- `verify_citations` is NOT exposed to the LLM. Verification is a
  post-processing step that runs unconditionally on every final
  assistant message, so the LLM cannot skip or bypass it.
"""

from __future__ import annotations

import json
import re

from memory_store import MemoryStore

UNKNOWN_REPLY = "I don't know — you haven't told me that yet."
NEUTRAL_REPLY = "I can't verify that from your stored memories."

_ADMISSION_RE = re.compile(
    r"\b(don't know|do not know|haven't told me|have not told me|not sure)\b"
    r"|不知道|还没告诉|没有告诉",
    re.IGNORECASE,
)

_UNCITED_CLAIM_RE = re.compile(
    r"\b(you (told|mentioned|said|shared|noted) me\b"
    r"|based on what you\b"
    r"|from your (memory|memories)\b"
    r"|according to what you\b)"
    r"|你(告诉|说过|提到|分享)(过|我)?"
    r"|根据(你的)?记忆",
    re.IGNORECASE,
)


SAVE_MEMORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "save_memory",
        "description": (
            "Store a fact the user told about themselves (preference, goal, "
            "habit, personal history). Call this IMMEDIATELY when the user "
            "states such a fact. Do not call it for questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "The full statement, rewritten in first person, "
                        "preserving all specific details."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "2-5 lowercase topical keywords to help future "
                        "retrieval (e.g. ['running','weather'])."
                    ),
                },
            },
            "required": ["content", "tags"],
        },
    },
}


SEARCH_MEMORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": (
            "Retrieve relevant stored memories. MUST be called before "
            "answering any question that depends on user preferences, "
            "history, goals, or habits. Returns an empty list if nothing "
            "relevant is stored — in that case you MUST admit you don't "
            "know instead of guessing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query describing what to recall.",
                },
                "top_k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    },
}


TOOL_SCHEMAS = [SAVE_MEMORY_SCHEMA, SEARCH_MEMORY_SCHEMA]


def _missing_field(fields: list[str], arguments: dict) -> str | None:
    for f in fields:
        if f not in arguments:
            return f
    return None


def dispatch(tool_name: str, arguments: dict, store: MemoryStore) -> dict:
    """Route an LLM tool_call to the memory store.

    Returns a JSON-serializable dict. Errors are returned as
    ``{"error": "..."}`` instead of raised, so the LLM can see and
    recover from bad tool calls.
    """
    if not isinstance(arguments, dict):
        return {"error": "arguments must be an object"}

    if tool_name == "save_memory":
        missing = _missing_field(["content", "tags"], arguments)
        if missing:
            return {"error": f"missing field: {missing}"}
        content = arguments.get("content", "")
        tags = arguments.get("tags", [])
        if not isinstance(tags, list):
            return {"error": "tags must be an array of strings"}
        try:
            new_id = store.save(content, tags)
        except ValueError as e:
            return {"error": str(e)}
        return {"id": new_id, "status": "saved"}

    if tool_name == "search_memory":
        missing = _missing_field(["query"], arguments)
        if missing:
            return {"error": f"missing field: {missing}"}
        query = arguments.get("query", "")
        top_k = int(arguments.get("top_k", 3) or 3)
        top_k = max(1, min(top_k, 10))
        results = store.search(query, top_k=top_k)
        return {"results": results}

    return {"error": f"unknown tool: {tool_name}"}


_CITATION_RE = re.compile(r"\[memory:(\d+)\]")


def _strip_invalid_citations(
    reply: str,
    store: MemoryStore,
    allowed_memory_ids: set[int] | None,
    log: list[dict],
) -> str:
    def _replace(match: re.Match) -> str:
        mid = int(match.group(1))
        if allowed_memory_ids is not None and mid not in allowed_memory_ids:
            log.append({"id": mid, "status": "fail_not_in_search"})
            return ""
        if store.exists(mid):
            log.append({"id": mid, "status": "ok"})
            return match.group(0)
        log.append({"id": mid, "status": "fail"})
        return ""

    cleaned = _CITATION_RE.sub(_replace, reply)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _has_valid_citations(text: str, store: MemoryStore) -> bool:
    for match in _CITATION_RE.finditer(text):
        if store.exists(int(match.group(1))):
            return True
    return False


def verify_citations(
    reply: str, store: MemoryStore
) -> tuple[str, list[dict]]:
    """Strip any `[memory:<id>]` whose id does not exist in the store.

    Returns (cleaned_reply, log) where ``log`` is a list of
    ``{"id": int, "status": "ok"|"fail"}`` entries, one per citation
    encountered. The verifier is deterministic and does not consult
    the LLM — that is the whole point.
    """
    if not reply:
        return reply or "", []
    log: list[dict] = []
    cleaned = _strip_invalid_citations(reply, store, None, log)
    return cleaned, log


def verify_reply(
    reply: str,
    store: MemoryStore,
    *,
    allowed_memory_ids: set[int] | None = None,
) -> tuple[str, list[dict]]:
    """Deterministic post-processing for citations and reply policy.

    ``allowed_memory_ids`` is set when ``search_memory`` ran this turn:
    - ``None`` — no search yet; only check id existence in the store
    - ``set()`` — search returned empty; enforce admission or override
    - non-empty set — citations must come from that search result
    """
    if not reply:
        return reply or "", []

    log: list[dict] = []
    cleaned = _strip_invalid_citations(reply, store, allowed_memory_ids, log)

    if allowed_memory_ids is not None and len(allowed_memory_ids) == 0:
        if not _ADMISSION_RE.search(cleaned):
            log.append({"status": "empty_search_override"})
            return UNKNOWN_REPLY, log
        return cleaned, log

    if not _has_valid_citations(cleaned, store) and _UNCITED_CLAIM_RE.search(
        cleaned
    ):
        log.append({"status": "uncited_claim"})
        return NEUTRAL_REPLY, log

    return cleaned, log


__all__ = [
    "TOOL_SCHEMAS",
    "SAVE_MEMORY_SCHEMA",
    "SEARCH_MEMORY_SCHEMA",
    "UNKNOWN_REPLY",
    "NEUTRAL_REPLY",
    "dispatch",
    "verify_citations",
    "verify_reply",
]
