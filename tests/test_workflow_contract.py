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
            # ...and names what differs, so an emoji upgrade is distinguishable
            # from a different checkpoint or preprocessing mode.
            self.assertIn("preprocessing: chain 'reference', current 'different'", result["error"])
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

    def test_finished_section_does_not_stop_an_unfinished_document(self):
        self.cid = server.chain_create(name="sections", segments=["intro", "body", "end"])["chain_id"]
        self.score = 0.05  # below the 0.25 target
        intro = self.submit("Intro draft", segment="intro")
        self.assertEqual(intro["revision_state"], "target_met")
        # chain_status says keep drafting for this state; the submit must agree.
        self.assertFalse(intro["stop_recommended"])
        self.assertIn("['body', 'end']", intro["next_action"])
        self.assertFalse(server.chain_status(chain_id=self.cid)["stop_recommended"])
        self.submit("Body draft", segment="body")
        last = self.submit("Ending draft", segment="end")
        self.assertTrue(last["stop_recommended"])
        self.assertNotIn("remaining section", last["next_action"])

    def test_multi_segment_advice_names_only_real_unstarted_sections_and_regressions(self):
        self.cid = server.chain_create(name="two", segments=["a", "b"])["chain_id"]
        self.submit("A first", segment="a")
        self.submit("B first", segment="b")
        self.score = 0.95  # worse than 0.8 by more than the regression delta
        worse = self.submit("A second", segment="a")
        self.assertEqual(worse["revision_state"], "assemble_first")
        self.assertTrue(worse["next_action"].startswith("Call chain_assemble"), worse["next_action"])
        self.assertIn("step 1, which assembly will use", worse["next_action"])
        self.assertIn(f"chain_get_text(chain_id={self.cid!r}, segment='a', step=1)", worse["next_action"])

    def test_suggested_calls_carry_the_required_chain_id(self):
        self.submit("First draft")
        text = server.chain_get_text(chain_id=self.cid)
        self.assertIn(f"chain_submit(chain_id={self.cid!r}", text["next_action"])
        created = server.chain_create(name="fresh")
        self.assertIn(f"chain_id={created['chain_id']!r}", created["next_action"])

    def test_duplicate_detection_compares_with_the_branch_point(self):
        self.submit("Text one")
        self.submit("Text two")
        same_as_parent = self.submit("Text one", branch_from=1)
        self.assertEqual(same_as_parent["revision_state"], "unchanged_draft")
        self.assertEqual(same_as_parent["delta_vs_parent"], 0.0)
        revised = self.submit("Text two", branch_from=1)
        self.assertNotEqual(revised["revision_state"], "unchanged_draft")
        self.assertEqual(server.chain_get_text(chain_id=self.cid, step=4)["parent_step"], 1)

    def test_blank_names_and_nonpositive_steps_are_rejected(self):
        self.assertEqual(server.chain_create(name="  ")["error_type"], "ValueError")
        self.assertEqual(server.chain_create(name="x", segments=["a", ""])["error_type"], "ValueError")
        self.assertEqual(self.submit("Draft", segment=" ")["error_type"], "ValueError")
        self.submit("Draft")
        self.assertEqual(server.chain_get_text(chain_id=self.cid, step=0)["error_type"], "ValueError")
        # A blank segment must never widen into deleting the whole chain.
        self.assertFalse(server.chain_delete(chain_id=self.cid, segment="")["ok"])
        self.assertTrue(server.chain_status(chain_id=self.cid)["ok"])

    def test_short_assembled_document_above_target_keeps_the_caution(self):
        self.words, self.score = 12, 0.9
        self.submit("Short draft")
        result = server.chain_assemble(chain_id=self.cid)
        self.assertTrue(result["next_action"].startswith("CAUTION:"), result["next_action"])

    def test_chain_list_segment_counts_ignore_unlisted_chains(self):
        self.submit("Listed draft")
        listed = server.chain_list(limit=1)["chains"]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["segments_total"], 1)

    def test_review_advice_mentions_spans_only_when_returned(self):
        self.score = 0.8
        result = self.submit("Draft without span feedback")
        self.assertEqual(result["revision_state"], "review_draft")
        self.assertNotIn("worst_spans", result)
        self.assertNotIn("above_target", result["next_action"])

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

    def test_torch_wheel_follows_the_hardware(self):
        import install
        with patch.dict(os.environ, {"EDITLENS_TORCH_INDEX": ""}), \
                patch("install.sys.platform", "linux"):
            with patch("install.has_nvidia_gpu", return_value=False):
                self.assertEqual(install.torch_install_cmd()[-1], install.TORCH_INDEX["cpu"])
            with patch("install.has_nvidia_gpu", return_value=True):
                self.assertEqual(install.torch_install_cmd()[-1], install.TORCH_INDEX["cuda"])
        with patch.dict(os.environ, {"EDITLENS_TORCH_INDEX": "https://mirror.example/whl"}), \
                patch("install.sys.platform", "win32"):
            self.assertEqual(install.torch_install_cmd()[-1], "https://mirror.example/whl")
        with patch("install.sys.platform", "darwin"):
            self.assertNotIn("--index-url", install.torch_install_cmd())

    def test_hf_command_prefers_the_interpreters_own_scripts_dir(self):
        import install
        with tempfile.TemporaryDirectory() as scratch:
            exe = Path(scratch) / ("python.exe" if os.name == "nt" else "python")
            hf = Path(scratch) / ("hf.exe" if os.name == "nt" else "hf")
            hf.write_text("")
            with patch("install.sys.executable", str(exe)), \
                    patch("install.shutil.which", return_value=None):
                self.assertEqual(install.hf_command(), str(hf))


class SpanAndTextTests(unittest.TestCase):
    def spans(self, text, **kwargs):
        from editlens_mcp.preprocessing import reference_text

        def many(texts, normalise=True):
            # Every unit sent to the model must have model input.
            self.assertTrue(all(reference_text(t) for t in texts), texts)
            return [Verdict(0.3, 1, "Lightly AI-edited", word_count=12) for _ in texts]

        overall = (Verdict(0.3, 1, "Lightly AI-edited", word_count=40), [])
        with patch.object(server.detector, "detect_many", side_effect=many), \
                patch.object(server.detector, "detect", return_value=overall):
            return server.detect_spans(text=text, **kwargs)

    def test_reasoning_block_is_outside_the_units(self):
        text = ("<think>Private reasoning the reference pipeline discards.</think>\n\n"
                "The finished answer starts here. It has a second sentence.\n\n"
                "A second paragraph closes the answer.")
        for granularity in ("paragraph", "sentence"):
            with self.subTest(granularity=granularity):
                result = self.spans(text, granularity=granularity, min_words=1)
                self.assertTrue(result["ok"], result)
                self.assertTrue(all("reasoning" not in u["text"] for u in result["worst_units"]),
                                result["worst_units"])

    def test_leading_boilerplate_line_is_not_ranked(self):
        text = ("Sure! Here is the essay you asked for:\n\n"
                "Rivers shape the towns built beside them. Floods taught planners humility.")
        result = self.spans(text, granularity="paragraph")
        self.assertTrue(all(not u["text"].startswith("Sure") for u in result["worst_units"]))

    def test_sentences_split_before_non_ascii_capitals_and_inverted_marks(self):
        from editlens_mcp.detector import _sentence_spans
        for text in ("First sentence. Émile went to Paris.", "Primera frase. ¿Qué pasa?",
                     "Erster Satz. Über die Brücke.", "Первое. Второе предложение."):
            with self.subTest(text=text):
                self.assertEqual(len(_sentence_spans(text)), 2)

    def test_ellipsis_splits_only_before_a_new_sentence(self):
        from editlens_mcp.detector import _sentence_spans
        self.assertEqual(len(_sentence_spans("Wait…what happened here?")), 1)
        self.assertEqual(len(_sentence_spans("It is un…believable.")), 1)
        self.assertEqual(len(_sentence_spans("He paused… Then he left.")), 2)
        self.assertEqual(len(_sentence_spans("待って…何が起きた？")), 2)

    def test_negative_span_maps_to_nothing(self):
        from editlens_mcp.detector import map_span
        self.assertEqual(map_span([(0, 5), (5, 10), (10, 15)], -2, 2, 15), (0, 0))


class StoreTests(unittest.TestCase):
    def test_snapshots_nest_and_history_reads_share_one(self):
        store = server.store
        cid = store.create("nested")["chain_id"]
        store.add_step(cid, "main", "draft", 0.5, 2, "x", 80, [0.25] * 4)
        with store.snapshot():
            # Used to fail: "cannot start a transaction within a transaction".
            stats = store.segment_stats(cid, "main")
            self.assertEqual(stats["steps"], len(store.history(cid, "main")))
        self.assertTrue(server.chain_history(chain_id=cid)["ok"])

    def test_delete_segment_reports_the_list_it_committed(self):
        cid = server.chain_create(name="del", segments=["a", "b"])["chain_id"]
        result = server.chain_delete(chain_id=cid, segment="b")
        self.assertEqual(result["remaining_segments"], ["a"])


class ConfigTests(unittest.TestCase):
    def test_whitespace_db_path_means_the_default(self):
        from editlens_mcp import chains
        with patch.dict(os.environ, {"EDITLENS_DB": "   "}):
            self.assertEqual(chains.default_db_path().name, "chains.db")

    def test_empty_model_settings_mean_unset_and_explicit_empty_is_rejected(self):
        from editlens_mcp import detector
        with patch.dict(os.environ, {"EDITLENS_CHECKPOINT": "  "}):
            self.assertEqual(detector._env_text("EDITLENS_CHECKPOINT", "default/id"), "default/id")
        with self.assertRaisesRegex(ValueError, "EDITLENS_CHECKPOINT"):
            EditLensDetector(checkpoint="")
        with self.assertRaisesRegex(ValueError, "EDITLENS_PREPROCESS"):
            EditLensDetector(preprocessing="Reference")


if __name__ == "__main__":
    unittest.main()
