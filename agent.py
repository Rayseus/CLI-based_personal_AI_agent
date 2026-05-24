"""CLI entry point for the Personal Memory Agent (Gemini-backed).

Run with:
    python agent.py

Required env:
    GEMINI_API_KEY  — Google AI Studio API key
Optional env (with defaults):
    GEMINI_MODEL    — default: gemini-2.5-flash
    MEMORY_DB_PATH  — default: memory.db
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from memory_store import MemoryStore
from tools import TOOL_SCHEMAS, dispatch, verify_reply


SYSTEM_PROMPT = """You are a Personal Memory Agent running on the user's machine.

Hard rules (violating any of these is a bug):

1. If the user STATES a personal fact, preference, goal, habit, or
   history, you MUST call the `save_memory` tool before replying.
   Extract 2-5 lowercase `tags` so future retrieval is accurate.
2. If the user ASKS a question that could depend on their own
   preferences, history, goals, or habits, you MUST call
   `search_memory` first and ground your answer in what it returns.
3. When you reference a retrieved memory in your reply, you MUST cite
   it with the token `[memory:<id>]`, using only ids that appeared in
   the most recent `search_memory` result. Do not invent ids.
4. If `search_memory` returns an empty `results` list, you MUST reply
   with an honest admission such as "I don't know — you haven't told
   me that yet." and you MUST NOT guess or invent facts. Do NOT
   include any `[memory:...]` citations in such a reply.
5. Never present a stored memory as external fact; always frame it as
   something the user told you (e.g. "You mentioned you...").

Keep replies concise (1-3 sentences) and practical.
"""

TOOL_LOOP_LIMIT = 6


_DIAG_ENABLED = False


def _configure_diagnostics(enabled: bool) -> None:
    global _DIAG_ENABLED
    _DIAG_ENABLED = enabled


def _log_diagnostics(msg: str) -> None:
    if _DIAG_ENABLED:
        print(msg, file=sys.stderr, flush=True)


def load_config() -> tuple[str, str, str]:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY is not set. Copy .env.example to .env "
            "and fill it in, or `export GEMINI_API_KEY=...`.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    db_path = os.environ.get("MEMORY_DB_PATH", "memory.db").strip() or "memory.db"
    return api_key, model, db_path


# ---------------------------------------------------------------------------
# JSON-schema → Gemini Schema conversion
# ---------------------------------------------------------------------------

def _to_gemini_schema(jschema: dict) -> Any:
    """Convert our LLM-agnostic tool JSON schema into Gemini's types.Schema."""
    from google.genai import types

    type_map = {
        "object": types.Type.OBJECT,
        "array": types.Type.ARRAY,
        "string": types.Type.STRING,
        "integer": types.Type.INTEGER,
        "number": types.Type.NUMBER,
        "boolean": types.Type.BOOLEAN,
    }
    t = type_map.get(jschema.get("type", "string"), types.Type.STRING)
    kwargs: dict[str, Any] = {"type": t}
    if "description" in jschema:
        kwargs["description"] = jschema["description"]
    if jschema.get("type") == "object":
        props = {}
        for k, v in (jschema.get("properties") or {}).items():
            props[k] = _to_gemini_schema(v)
        kwargs["properties"] = props
        if jschema.get("required"):
            kwargs["required"] = list(jschema["required"])
    elif jschema.get("type") == "array":
        items = jschema.get("items") or {"type": "string"}
        kwargs["items"] = _to_gemini_schema(items)
    return types.Schema(**kwargs)


def _build_gemini_tools() -> list:
    from google.genai import types

    declarations = []
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        declarations.append(
            types.FunctionDeclaration(
                name=fn["name"],
                description=fn["description"],
                parameters=_to_gemini_schema(fn["parameters"]),
            )
        )
    return [types.Tool(function_declarations=declarations)]


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def _collect_text(parts) -> str:
    out = []
    for p in parts or []:
        txt = getattr(p, "text", None)
        if txt:
            out.append(txt)
    return "".join(out)


def _collect_function_calls(parts) -> list:
    calls = []
    for p in parts or []:
        fc = getattr(p, "function_call", None)
        if fc and getattr(fc, "name", None):
            calls.append(fc)
    return calls


def chat(
    user_msg: str,
    history: list,
    store: MemoryStore,
    client: Any,
    model: str,
) -> str:
    """One user turn → one assistant reply, handling tool loops.

    `history` is a list of ``google.genai.types.Content`` and is
    mutated in place so multi-turn context persists.
    """
    from google.genai import types

    history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_msg)])
    )

    tools = _build_gemini_tools()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=tools,
        temperature=0.2,
    )

    allowed_memory_ids: set[int] | None = None

    for _ in range(TOOL_LOOP_LIMIT):
        resp = client.models.generate_content(
            model=model,
            contents=history,
            config=config,
        )
        if not resp.candidates:
            raise RuntimeError("gemini returned no candidates")
        content = resp.candidates[0].content
        history.append(content)

        fn_calls = _collect_function_calls(content.parts)
        if not fn_calls:
            text = _collect_text(content.parts)
            cleaned, vlog = verify_reply(
                text, store, allowed_memory_ids=allowed_memory_ids
            )
            for entry in vlog:
                if "id" in entry:
                    _log_diagnostics(
                        f"[VERIFY] id={entry['id']} status={entry['status']}"
                    )
                else:
                    _log_diagnostics(f"[VERIFY] status={entry['status']}")
            if cleaned != text:
                history[-1] = types.Content(
                    role="model", parts=[types.Part.from_text(text=cleaned)]
                )
            return cleaned

        response_parts = []
        for fc in fn_calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}
            result = dispatch(name, args, store)
            if name == "search_memory" and "results" in result:
                allowed_memory_ids = {
                    int(r["id"]) for r in result.get("results", [])
                }
            _log_diagnostics(
                f"[TOOL] {name} args={json.dumps(args, ensure_ascii=False)} "
                f"result={json.dumps(result, ensure_ascii=False)}"
            )
            response_parts.append(
                types.Part.from_function_response(name=name, response=result)
            )
        history.append(types.Content(role="user", parts=response_parts))

    raise RuntimeError("tool loop exceeded")


class _Tee:
    """Minimal tee stream; used only when AGENT_DEBUG is enabled."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def main() -> int:
    api_key, model, db_path = load_config()
    debug = os.environ.get("AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    log_enabled = os.environ.get("AGENT_LOG_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    _configure_diagnostics(debug or log_enabled)

    from google import genai

    client = genai.Client(api_key=api_key)
    store = MemoryStore(db_path)

    history: list = []

    log_file = None
    original_stderr = sys.stderr
    if log_enabled:
        log_path = os.environ.get("AGENT_LOG_PATH", "agent.log").strip() or "agent.log"
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        if debug:
            sys.stderr = _Tee(original_stderr, log_file)
        else:
            sys.stderr = log_file
        log_file.write(
            f"\n=== session start: model={model}, db={db_path} ===\n"
        )

    ready_msg = f"Personal Memory Agent ready (model={model}, db={db_path})."
    if debug or log_enabled:
        log_path = os.environ.get("AGENT_LOG_PATH", "agent.log").strip() or "agent.log"
        ready_msg += (
            f"\nTool / verify diagnostics -> {log_path}"
            f"{' (also shown here)' if debug else ''}."
        )
    print(f"{ready_msg}\nType 'exit' to quit.")
    try:
        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            try:
                reply = chat(user, history, store, client, model)
            except Exception as e:
                msg = f"[ERROR] {type(e).__name__}: {e}"
                print(msg, file=sys.stderr, flush=True)
                if log_file is not None:
                    log_file.write(msg + "\n")
                    log_file.flush()
                continue
            print(f"agent> {reply}")
    finally:
        store.close()
        if log_file is not None:
            sys.stderr = original_stderr
            log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
