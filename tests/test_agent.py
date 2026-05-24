"""Unit tests for agent.py covering Step 3.1 - 3.3.

The Gemini SDK is not called for real; we mock
``client.models.generate_content`` with lightweight fake objects that
have the minimal attribute surface (``candidates[0].content.parts``,
each part exposing ``.text`` and/or ``.function_call``).
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agent  # noqa: E402
from memory_store import MemoryStore  # noqa: E402


# ---------------------------------------------------------------------------
# Fake Gemini client
# ---------------------------------------------------------------------------


class FakePart:
    def __init__(self, *, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class FakeFunctionCall:
    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args


class FakeContent:
    def __init__(self, parts, role: str = "model"):
        self.parts = parts
        self.role = role


class FakeCandidate:
    def __init__(self, content):
        self.content = content


class FakeResponse:
    def __init__(self, content):
        self.candidates = [FakeCandidate(content)]


class FakeModels:
    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("mock ran out of scripted responses")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, scripted_responses):
        self.models = FakeModels(scripted_responses)


def _text_reply(text: str) -> FakeResponse:
    return FakeResponse(FakeContent(parts=[FakePart(text=text)]))


def _tool_call_reply(name: str, args: dict) -> FakeResponse:
    return FakeResponse(
        FakeContent(parts=[FakePart(function_call=FakeFunctionCall(name, args))])
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _TempStoreMixin:
    def _new_store(self) -> MemoryStore:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.addCleanup(lambda p=tmp.name: os.path.exists(p) and os.remove(p))
        store = MemoryStore(tmp.name)
        self.addCleanup(store.close)
        return store


# ---------------------------------------------------------------------------
# Step 3.1 — load_config
# ---------------------------------------------------------------------------


class TestLoadConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            k: os.environ.get(k)
            for k in ("GEMINI_API_KEY", "GEMINI_MODEL", "MEMORY_DB_PATH")
        }

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_TC_3_1_1_missing_key_exits(self) -> None:
        # Set to empty — ``load_dotenv`` will not override existing env
        # vars by default, so even if a real .env is on disk the empty
        # value we set here prevails.
        os.environ["GEMINI_API_KEY"] = ""

        buf = io.StringIO()
        with redirect_stderr(buf):
            with self.assertRaises(SystemExit) as ctx:
                agent.load_config()
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("GEMINI_API_KEY", buf.getvalue())

    def test_TC_3_1_2_defaults(self) -> None:
        os.environ["GEMINI_API_KEY"] = "fake-key"
        os.environ.pop("GEMINI_MODEL", None)
        os.environ.pop("MEMORY_DB_PATH", None)
        api_key, model, db_path = agent.load_config()
        self.assertEqual(api_key, "fake-key")
        self.assertEqual(model, "gemini-2.5-flash")
        self.assertEqual(db_path, "memory.db")


# ---------------------------------------------------------------------------
# Step 3.2 — SYSTEM_PROMPT
# ---------------------------------------------------------------------------


class TestSystemPrompt(unittest.TestCase):
    def test_TC_3_2_1_contains_hard_rules(self) -> None:
        prompt = agent.SYSTEM_PROMPT
        for keyword in [
            "save_memory",
            "search_memory",
            "[memory:",
            "don't know",
        ]:
            self.assertIn(keyword, prompt, f"missing keyword: {keyword}")


# ---------------------------------------------------------------------------
# Step 3.3 — chat() tool-use loop
# ---------------------------------------------------------------------------


class TestChatLoop(unittest.TestCase, _TempStoreMixin):
    def setUp(self) -> None:
        agent._configure_diagnostics(True)

    def tearDown(self) -> None:
        agent._configure_diagnostics(False)

    def test_TC_3_3_1_save_memory_call_writes_db(self) -> None:
        store = self._new_store()
        client = FakeClient(
            [
                _tool_call_reply(
                    "save_memory",
                    {
                        "content": "I'm training for a marathon",
                        "tags": ["running", "goals"],
                    },
                ),
                _text_reply("Got it, I noted that."),
            ]
        )
        history: list = []

        buf = io.StringIO()
        with redirect_stderr(buf):
            reply = agent.chat(
                "I'm training for a marathon and I hate running in the rain.",
                history,
                store,
                client,
                "mock-model",
            )

        self.assertEqual(reply, "Got it, I noted that.")
        self.assertEqual(store.count(), 1)
        self.assertIn("[TOOL] save_memory", buf.getvalue())

    def test_TC_3_3_2_search_memory_then_cited_reply(self) -> None:
        store = self._new_store()
        mid = store.save("I love congee for recovery", ["food", "recovery"])

        client = FakeClient(
            [
                _tool_call_reply(
                    "search_memory", {"query": "recovery meal", "top_k": 3}
                ),
                _text_reply(f"Try congee — you told me you love it [memory:{mid}]."),
            ]
        )
        history: list = []
        buf = io.StringIO()
        with redirect_stderr(buf):
            reply = agent.chat(
                "What's good after a long run?",
                history,
                store,
                client,
                "mock-model",
            )

        self.assertIn(f"[memory:{mid}]", reply)
        logs = buf.getvalue()
        self.assertIn("[TOOL] search_memory", logs)
        self.assertIn(f"[VERIFY] id={mid} status=ok", logs)

    def test_unknown_citation_stripped_and_logged(self) -> None:
        store = self._new_store()
        client = FakeClient(
            [
                _tool_call_reply("search_memory", {"query": "color"}),
                _text_reply("Your favorite is blue [memory:99]."),
            ]
        )
        history: list = []
        buf = io.StringIO()
        with redirect_stderr(buf):
            reply = agent.chat(
                "What's my favorite color?",
                history,
                store,
                client,
                "mock-model",
            )

        self.assertNotIn("[memory:99]", reply)
        self.assertEqual(
            reply, "I don't know — you haven't told me that yet."
        )
        self.assertIn("[VERIFY] id=99 status=fail", buf.getvalue())
        self.assertIn("[VERIFY] status=empty_search_override", buf.getvalue())

    def test_TC_3_3_3_loop_limit(self) -> None:
        store = self._new_store()
        client = FakeClient(
            [
                _tool_call_reply(
                    "save_memory", {"content": f"fact {i}", "tags": ["x"]}
                )
                for i in range(agent.TOOL_LOOP_LIMIT + 2)
            ]
        )
        history: list = []
        buf = io.StringIO()
        with redirect_stderr(buf):
            with self.assertRaises(RuntimeError) as ctx:
                agent.chat("go", history, store, client, "mock-model")
        self.assertIn("tool loop exceeded", str(ctx.exception))


# ---------------------------------------------------------------------------
# Schema conversion helpers
# ---------------------------------------------------------------------------


class TestSchemaConversion(unittest.TestCase):
    def test_builds_two_function_declarations(self) -> None:
        tools = agent._build_gemini_tools()
        self.assertEqual(len(tools), 1)
        decls = tools[0].function_declarations
        names = {d.name for d in decls}
        self.assertEqual(names, {"save_memory", "search_memory"})

    def test_nested_array_item_type(self) -> None:
        tools = agent._build_gemini_tools()
        save = next(
            d for d in tools[0].function_declarations if d.name == "save_memory"
        )
        tags = save.parameters.properties["tags"]
        self.assertEqual(tags.type.value, "ARRAY")
        self.assertEqual(tags.items.type.value, "STRING")


if __name__ == "__main__":
    unittest.main(verbosity=2)
