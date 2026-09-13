"""Behavioral regressions for stopping, score provenance, and gated access."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["EDITLENS_DB"] = str(Path(tempfile.mkdtemp()) / "workflow.db")

from editlens_mcp import server
from editlens_mcp.detector import EditLensDetector, Verdict


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.detector = server.detector
        self.score = 0.8
        self.words = 100
        self.calls = 0

        def detect(*args, **kwargs):
            self.calls += 1
            return Verdict(self.score, 2, "Heavily AI-edited", [0, 0, 1, 0],
                           word_count=self.words), []

        self.mock = patch.object(server.detector, "detect", side_effect=detect)
        self.mock.start()
        self.cid = server.chain_create(name="contract")["chain_id"]

    def tearDown(self):
        self.mock.stop()
        server.detector = self.detector

    def submit(self, text, **kwargs):
        return server.chain_submit(chain_id=self.cid, text=text,
                                   span_feedback=False, **kwargs)

    def test_unchanged_draft_stops_and_round_trips_verbatim(self):
        text = "  Draft\r\n\twith   original formatting.\n\n"
        self.submit(text)
        repeated = self.submit(text)
        self.assertTrue(repeated["stop_recommended"])
        self.assertEqual(repeated["revision_state"], "unchanged_draft")
        saved = server.chain_get_text(chain_id=self.cid, step="latest")
        self.assertEqual(saved["text"], text)

    def test_plateau_stops_but_does_not_prevent_saving(self):
        for i in range(5):
            result = self.submit(f"Draft revision {i}")
        self.assertEqual(result["revision_state"], "review_recommended")
        self.assertTrue(result["revision_progress"]["plateau"])
        self.assertTrue(result["stop_recommended"])
        status = server.chain_status(chain_id=self.cid)
        self.assertTrue(status["stop_recommended"])
        next_result = self.submit("A further revision after review")
        self.assertTrue(next_result["ok"])
        self.assertEqual(next_result["step"], 6)

    def test_review_budget_and_short_length_take_precedence(self):
        for i in range(8):
            self.score = 0.99 - i * 0.02
            result = self.submit(f"Meaningfully different draft {i}")
        self.assertFalse(result["revision_progress"]["plateau"])
        self.assertTrue(result["revision_progress"]["revision_budget_reached"])
        self.assertTrue(result["stop_recommended"])
        self.words, self.score = 12, 0.01
        short = self.submit("Short result")
        self.assertTrue(short["target_met"])
        self.assertEqual(short["revision_state"], "insufficient_length")
        self.assertFalse(short["calibrated"])

    def test_profile_mismatch_blocks_new_scores_before_inference(self):
        self.submit("Existing saved draft")
        calls = self.calls
        with patch.object(server.detector, "scoring_identity", return_value={"preprocessing": "different"}):
            result = self.submit("New candidate")
            self.assertFalse(result["ok"])
            self.assertIn("scoring profile", result["error"])
            self.assertFalse(server.chain_assemble(chain_id=self.cid)["ok"])
            self.assertEqual(self.calls, calls)
            self.assertEqual(server.chain_get_text(chain_id=self.cid)["text"], "Existing saved draft")
            self.assertEqual(server.chain_history(chain_id=self.cid)["total_steps"], 1)

    def test_legacy_chain_is_readable_and_explicitly_resumable(self):
        self.cid = server.store.create(name="pre-upgrade", target_score=0.25, segments=["main"])["chain_id"]
        rejected = self.submit("Reference cannot join a legacy ranking")
        self.assertFalse(rejected["ok"])
        self.assertEqual(self.calls, 0)
        profile = EditLensDetector(preprocessing="legacy").scoring_identity()
        with patch.object(server.detector, "scoring_identity", return_value=profile):
            self.assertTrue(self.submit("Explicit legacy scoring")["ok"])

    def test_ranking_uses_full_precision(self):
        verdicts = [Verdict(0.500049, 1, "Lightly AI-edited"),
                    Verdict(0.500041, 1, "Lightly AI-edited")]
        with patch.object(server.detector, "detect_many", return_value=verdicts):
            result = server.detect_batch(texts=["first", "second"])
        self.assertEqual(result["results"][0]["score"], result["results"][1]["score"])
        self.assertEqual(result["best_index"], 1)


class InstallTests(unittest.TestCase):
    def test_gated_check_requests_file_metadata(self):
        import install
        with patch("huggingface_hub.get_hf_file_metadata", side_effect=RuntimeError("gated")) as head:
            error = install.check_hf()
        self.assertIn("gated", error)
        self.assertEqual(head.call_count, 1)
        self.assertIn("/resolve/main/config.json", head.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
