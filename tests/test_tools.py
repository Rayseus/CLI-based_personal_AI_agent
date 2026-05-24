"""Unit tests for tools.py covering Step 2.1 - 2.3."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from memory_store import MemoryStore  # noqa: E402
from tools import (  # noqa: E402
    NEUTRAL_REPLY,
    TOOL_SCHEMAS,
    UNKNOWN_REPLY,
    dispatch,
    verify_citations,
    verify_reply,
)


class _TempStoreMixin:
    def _new_store(self) -> MemoryStore:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.addCleanup(lambda p=tmp.name: os.path.exists(p) and os.remove(p))
        store = MemoryStore(tmp.name)
        self.addCleanup(store.close)
        return store


class TestToolSchemas(unittest.TestCase):
    def test_TC_2_1_1_two_tools(self) -> None:
        self.assertEqual(len(TOOL_SCHEMAS), 2)

    def test_TC_2_1_2_schema_shape(self) -> None:
        names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        self.assertEqual(names, {"save_memory", "search_memory"})
        for s in TOOL_SCHEMAS:
            self.assertEqual(s["type"], "function")
            fn = s["function"]
            self.assertIn("description", fn)
            self.assertEqual(fn["parameters"]["type"], "object")
            self.assertIn("properties", fn["parameters"])

        save = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "save_memory")
        self.assertEqual(
            set(save["function"]["parameters"]["required"]), {"content", "tags"}
        )

        search = next(
            s for s in TOOL_SCHEMAS if s["function"]["name"] == "search_memory"
        )
        self.assertEqual(
            set(search["function"]["parameters"]["required"]), {"query"}
        )

    def test_schemas_json_serializable(self) -> None:
        json.dumps(TOOL_SCHEMAS)


class TestDispatcher(unittest.TestCase, _TempStoreMixin):
    def test_TC_2_2_1_save_success(self) -> None:
        store = self._new_store()
        r = dispatch(
            "save_memory", {"content": "I love tea", "tags": ["drink"]}, store
        )
        self.assertEqual(r, {"id": 1, "status": "saved"})
        self.assertEqual(store.count(), 1)

    def test_TC_2_2_2_search_empty(self) -> None:
        store = self._new_store()
        r = dispatch("search_memory", {"query": "anything"}, store)
        self.assertEqual(r, {"results": []})

    def test_TC_2_2_3_unknown_tool(self) -> None:
        store = self._new_store()
        r = dispatch("foo", {}, store)
        self.assertIn("error", r)
        self.assertIn("unknown tool", r["error"])

    def test_TC_2_2_4_missing_field(self) -> None:
        store = self._new_store()
        r = dispatch("save_memory", {"tags": []}, store)
        self.assertIn("error", r)
        self.assertIn("content", r["error"])

    def test_save_with_bad_tags(self) -> None:
        store = self._new_store()
        r = dispatch(
            "save_memory", {"content": "x", "tags": "not-a-list"}, store
        )
        self.assertIn("error", r)

    def test_search_returns_hits(self) -> None:
        store = self._new_store()
        dispatch(
            "save_memory",
            {"content": "I love congee for recovery", "tags": ["food", "recovery"]},
            store,
        )
        r = dispatch("search_memory", {"query": "recovery meal"}, store)
        self.assertEqual(len(r["results"]), 1)
        self.assertEqual(r["results"][0]["id"], 1)


class TestVerifyCitations(unittest.TestCase, _TempStoreMixin):
    def test_TC_2_3_1_valid_citation_preserved(self) -> None:
        store = self._new_store()
        mid = store.save("congee fact", ["food"])
        reply = f"Try congee [memory:{mid}]."
        cleaned, log = verify_citations(reply, store)
        self.assertEqual(cleaned, reply)
        self.assertEqual(log, [{"id": mid, "status": "ok"}])

    def test_TC_2_3_2_fake_citation_stripped(self) -> None:
        store = self._new_store()
        reply = "Try X [memory:99]."
        cleaned, log = verify_citations(reply, store)
        self.assertNotIn("[memory:99]", cleaned)
        self.assertEqual(log, [{"id": 99, "status": "fail"}])

    def test_TC_2_3_3_no_citations(self) -> None:
        store = self._new_store()
        cleaned, log = verify_citations("Hello there", store)
        self.assertEqual(cleaned, "Hello there")
        self.assertEqual(log, [])

    def test_TC_2_3_4_mixed(self) -> None:
        store = self._new_store()
        mid = store.save("fact one", ["x"])
        reply = f"A [memory:{mid}] B [memory:99] C"
        cleaned, log = verify_citations(reply, store)
        self.assertIn(f"[memory:{mid}]", cleaned)
        self.assertNotIn("[memory:99]", cleaned)
        statuses = {entry["id"]: entry["status"] for entry in log}
        self.assertEqual(statuses, {mid: "ok", 99: "fail"})

    def test_empty_input(self) -> None:
        store = self._new_store()
        cleaned, log = verify_citations("", store)
        self.assertEqual(cleaned, "")
        self.assertEqual(log, [])


class TestVerifyReply(unittest.TestCase, _TempStoreMixin):
    def test_empty_search_forces_unknown(self) -> None:
        store = self._new_store()
        cleaned, log = verify_reply(
            "Your favorite color is blue.",
            store,
            allowed_memory_ids=set(),
        )
        self.assertEqual(cleaned, UNKNOWN_REPLY)
        self.assertEqual(log, [{"status": "empty_search_override"}])

    def test_empty_search_preserves_admission(self) -> None:
        store = self._new_store()
        reply = "I don't know — you haven't told me that yet."
        cleaned, log = verify_reply(reply, store, allowed_memory_ids=set())
        self.assertEqual(cleaned, reply)
        self.assertEqual(log, [])

    def test_citation_not_in_search_stripped(self) -> None:
        store = self._new_store()
        mid = store.save("congee fact", ["food"])
        reply = f"Try congee [memory:{mid}]."
        cleaned, log = verify_reply(reply, store, allowed_memory_ids=set())
        self.assertEqual(cleaned, UNKNOWN_REPLY)
        self.assertEqual(
            log,
            [
                {"id": mid, "status": "fail_not_in_search"},
                {"status": "empty_search_override"},
            ],
        )

    def test_citation_in_allowed_set_kept(self) -> None:
        store = self._new_store()
        mid = store.save("congee fact", ["food"])
        reply = f"Try congee [memory:{mid}]."
        cleaned, log = verify_reply(reply, store, allowed_memory_ids={mid})
        self.assertEqual(cleaned, reply)
        self.assertEqual(log, [{"id": mid, "status": "ok"}])

    def test_uncited_claim_becomes_neutral(self) -> None:
        store = self._new_store()
        reply = "You told me you love congee for recovery."
        cleaned, log = verify_reply(reply, store)
        self.assertEqual(cleaned, NEUTRAL_REPLY)
        self.assertEqual(log, [{"status": "uncited_claim"}])

    def test_save_ack_not_flagged(self) -> None:
        store = self._new_store()
        reply = "Got it, I noted that."
        cleaned, log = verify_reply(reply, store)
        self.assertEqual(cleaned, reply)
        self.assertEqual(log, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
