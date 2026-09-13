"""Database regressions using temporary files and no model or MCP runtime.

Run directly with ``python tests/test_store_regressions.py``. Snapshot tests
commit through a second connection between two reads, deterministically
reproducing the same isolation boundary as separate MCP client processes.
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from editlens_mcp.chains import ChainStore  # noqa: E402


class StoreRegressions(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="editlens-store-regressions-")
        self.addCleanup(scratch.cleanup)
        self.path = Path(scratch.name) / "chains.db"
        self.reader = ChainStore(self.path)
        self.addCleanup(self.reader.close)
        self.writer = ChainStore(self.path)
        self.addCleanup(self.writer.close)

    @staticmethod
    def submit(store, chain_id, segment="main", score=0.4, **kwargs):
        return store.add_step(
            chain_id, segment, "An unchanged draft.", score, 1, "test", 3,
            [0.25] * 4, **kwargs,
        )

    def _commit_before_query(self, query_fragment, action, read):
        """Commit once just before a later SELECT, after its earlier read."""
        self.assertEqual(self.reader.journal_mode, "wal")
        fired = []
        errors = []

        def trace(sql):
            if not fired and query_fragment in " ".join(sql.upper().split()):
                fired.append(True)
                try:
                    action()
                except BaseException as exc:
                    # sqlite3 does not propagate trace-callback exceptions.
                    errors.append(exc)

        self.reader._conn.set_trace_callback(trace)
        try:
            result = read()
        finally:
            self.reader._conn.set_trace_callback(None)
        self.assertEqual(fired, [True], "the concurrent write was not exercised")
        self.assertEqual(errors, [], "the concurrent write did not commit")
        return result

    def test_deleted_segment_rejects_a_submit_based_on_stale_lookup(self):
        cid = self.reader.create("draft", segments=["main", "appendix"])["chain_id"]
        self.submit(self.reader, cid, "appendix")
        self.assertIn("appendix", self.reader.segments_of(cid))
        self.writer.delete_segment(cid, "appendix")

        with self.assertRaisesRegex(KeyError, "no longer declared"):
            self.submit(self.reader, cid, "appendix", register_segment=False)

        self.assertEqual(self.reader.segments_of(cid), ["main"])
        self.assertEqual(self.reader.history(cid, "appendix"), [])
        self.assertFalse(self.reader._conn.in_transaction)
        # Deliberately creating a new segment remains supported.
        self.assertEqual(
            self.submit(self.reader, cid, "appendix", register_segment=True), 1,
        )
        self.assertIn("appendix", self.reader.segments_of(cid))

    def test_segment_registration_rolls_back_with_failed_draft(self):
        cid = self.reader.create("draft")["chain_id"]
        with self.assertRaises(sqlite3.IntegrityError):
            self.reader.add_step(
                cid, "new", None, 0.4, 1, "test", 3, [0.25] * 4,
                register_segment=True,
            )
        self.assertEqual(self.reader.segments_of(cid), ["main"])
        self.assertEqual(self.reader.history(cid, "new"), [])
        self.assertFalse(self.reader._conn.in_transaction)

    def test_deleted_chain_preserves_existing_error_contract(self):
        cid = self.reader.create("draft")["chain_id"]
        self.writer.delete(cid)
        with self.assertRaises(sqlite3.IntegrityError):
            self.submit(self.reader, cid)
        with self.assertRaisesRegex(KeyError, "no such chain"):
            self.submit(self.reader, cid, register_segment=True)

    def test_segment_stats_use_one_snapshot(self):
        cid = self.reader.create("draft")["chain_id"]
        self.submit(self.reader, cid, score=0.4)
        before = self._commit_before_query(
            "ORDER BY STEP_NO DESC LIMIT 1",
            lambda: self.submit(self.writer, cid, score=0.1),
            lambda: self.reader.segment_stats(cid, "main"),
        )
        self.assertEqual(before["steps"], 1)
        self.assertEqual(before["best_step"], 1)
        self.assertEqual(before["latest_step"], 1)
        self.assertEqual(before["best_score"], 0.4)
        self.assertEqual(before["latest_score"], 0.4)
        after = self.reader.segment_stats(cid, "main")
        self.assertEqual(after["steps"], 2)
        self.assertEqual(after["best_step"], 2)
        self.assertEqual(after["latest_score"], 0.1)

    def test_chain_list_counts_and_completion_use_one_snapshot(self):
        cid = self.reader.create("draft", target_score=0.25)["chain_id"]
        self.submit(self.reader, cid, score=0.4)
        before = self._commit_before_query(
            "SELECT CHAIN_ID, SEGMENT, MIN(SCORE)",
            lambda: self.submit(self.writer, cid, score=0.1),
            self.reader.list_chains,
        )[0]
        self.assertEqual(before["steps"], 1)
        self.assertEqual(before["best_score"], 0.4)
        self.assertEqual(before["segments_at_target"], 0)
        after = self.reader.list_chains()[0]
        self.assertEqual(after["steps"], 2)
        self.assertEqual(after["best_score"], 0.1)
        self.assertEqual(after["segments_at_target"], 1)

    def test_failed_snapshot_does_not_break_later_calls(self):
        cid = self.reader.create("draft")["chain_id"]
        self.submit(self.reader, cid)
        with patch.object(
            self.reader, "latest_step", side_effect=sqlite3.OperationalError("read failed"),
        ):
            with self.assertRaisesRegex(sqlite3.OperationalError, "read failed"):
                self.reader.segment_stats(cid, "main")
        self.assertFalse(self.reader._conn.in_transaction)
        self.assertEqual(self.submit(self.reader, cid, score=0.1), 2)
        self.assertEqual(self.reader.segment_stats(cid, "main")["best_step"], 2)

    def test_default_store_reads_configuration_at_construction(self):
        relocated = self.path.parent / "relocated.db"
        with patch.dict(os.environ, {"EDITLENS_DB": str(relocated)}):
            store = ChainStore()
            try:
                self.assertEqual(store.path, relocated.resolve())
                store.create("relocated")
            finally:
                store.close()
        self.assertTrue(relocated.exists())

    def test_unresolvable_default_path_does_not_kill_module_import(self):
        env = dict(os.environ)
        env["EDITLENS_DB"] = "~editlens_nonexistent_user_51df2687/chains.db"
        # Windows otherwise guesses another user's home from USERPROFILE.
        for key in ("USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME"):
            env.pop(key, None)
        probe = r'''
import os
from pathlib import Path
import sys

try:
    Path(os.environ["EDITLENS_DB"]).expanduser()
except RuntimeError:
    pass
else:
    raise AssertionError("fixture must be an unresolvable user path")

from editlens_mcp.chains import ChainStore
try:
    ChainStore()
except RuntimeError:
    pass
else:
    raise AssertionError("bad default was silently replaced by a different database")

# An explicit path remains usable despite the malformed configured default.
store = ChainStore(sys.argv[1])
try:
    assert store.create("explicit")["name"] == "explicit"
finally:
    store.close()
assert "torch" not in sys.modules
print("import survived; path error reported at construction")
'''
        child_path = self.path.parent / "explicit.db"
        result = subprocess.run(
            [sys.executable, "-c", probe, str(child_path)], env=env, cwd=ROOT,
            text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("import survived", result.stdout)
        self.assertTrue(child_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
