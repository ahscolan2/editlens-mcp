"""Independent reference decoding and preprocessing checks on the real model.

Fixtures are test prose, not labelled human/AI examples. This checks inference
parity, not accuracy on an authorship benchmark.
"""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from editlens_mcp.detector import EditLensDetector, clean_text, warmup_imports
from editlens_mcp.preprocessing import reference_text_with_map


class PreprocessingTests(unittest.TestCase):
    def test_reference_fixtures_and_coordinate_ranges(self):
        fixtures = [
            ("Hello WORLD.\n\n Next LINE!", "hello world. next line!"),
            ("Sure, here it is.\n\nThe actual content.", "the actual content."),
            ("Title of research\rSecond line", "title of research second line"),
            ("\U0001f642 Hi!", ":slightly_smiling_face: hi!"),
            # Preserve this upstream quirk: demojization happens before header
            # detection, so the emoji shortcode prevents a 'Sure' prefix match.
            ("\U0001f642 Sure!\nActual CONTENT.", ":slightly_smiling_face: sure! actual content."),
            ("\u039f\u03a3 \u0130STANBUL", "\u03bf\u03c2 i\u0307stanbul"),
            ("<think>x</think> Keep THIS </think>tail", "keep this"),
            ("Sure!\n\n \nHere is MY text.\nDone.", "here is my text. done."),
            ("  \r\n\t ", ""),
        ]
        for source, expected in fixtures:
            with self.subTest(source=ascii(source)):
                result, ranges = reference_text_with_map(source)
                self.assertEqual(result, expected)
                self.assertEqual(len(result), len(ranges))
                self.assertTrue(all(0 <= a < b <= len(source) for a, b in ranges))
                self.assertEqual(ranges, sorted(ranges))

    def test_argmax_disagrees_with_rounded_expectation(self):
        detector = EditLensDetector(preprocessing="legacy")
        detector.windows = lambda text: [(0, len(text), text)]
        # Expected index = 1.50, rounded index = 2. Most probable bucket = 0.
        probs = [0.45, 0.05, 0.05, 0.45]
        detector._score_batch = lambda texts: [(0.5, probs) for _ in texts]
        verdict, windows = detector.detect("A complete test sentence.")
        self.assertEqual(verdict.score, 0.5)
        self.assertEqual(verdict.bucket, 0)
        self.assertEqual(verdict.label, "Human-written")
        self.assertEqual(windows[0]["bucket"], 0)

    def test_removed_header_does_not_satisfy_training_length(self):
        detector = EditLensDetector()
        detector.windows = lambda text: [(0, len(text), text)]
        detector._score_batch = lambda texts: [(0.1, [0.7, 0.3, 0, 0]) for _ in texts]
        text = "Sure " + "introductory " * 100 + "\n\nOnly three words."
        verdict, _ = detector.detect(text)
        self.assertGreater(verdict.word_count, 75)
        self.assertEqual(verdict.model_word_count, 3)
        self.assertEqual(verdict.assessment_word_count, 3)

    def test_training_word_count_uses_reference_regex(self):
        detector = EditLensDetector()
        detector.windows = lambda text: [(0, len(text), text)]
        detector._score_batch = lambda texts: [(0.1, [0.7, 0.3, 0, 0]) for _ in texts]
        verdict, _ = detector.detect("don't can't won't it's that's " * 8)
        # Upstream counts with \b\w+\b before preprocessing. Model-input
        # count uses that regex too, but the server's conservative length
        # assessment must not inflate contractions into extra readable words.
        self.assertEqual(verdict.word_count, 40)
        self.assertEqual(verdict.model_word_count, 80)
        self.assertEqual(verdict.assessment_word_count, 40)

    def test_nonfinite_logits_fail_explicitly(self):
        import torch
        detector = EditLensDetector(preprocessing="legacy")
        detector.ensure_loaded = lambda: None
        detector.torch = torch
        class Inputs(dict):
            def to(self, _device):
                return self
        detector.tokenizer = lambda *_a, **_k: Inputs(input_ids=torch.tensor([[1, 2]]))
        detector.model = lambda **_k: SimpleNamespace(logits=torch.tensor([[float("nan"), 0, 0, 0]]))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            detector._score_batch(["test"])


class RealReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        error = warmup_imports()
        if error:
            raise RuntimeError(error)
        cls.detector = EditLensDetector(preprocessing="reference")
        cls.detector.ensure_loaded()

    @classmethod
    def tearDownClass(cls):
        cls.detector.unload()

    def test_single_and_batch_match_independent_raw_inference(self):
        pairs = [
            ("A Baker kept a NOTEBOOK.\nThe Oven was warmer at the back.",
             "a baker kept a notebook. the oven was warmer at the back."),
            ("Sure, here is the answer.\n\nA Farmer checked the FENCE. \U0001f642",
             "a farmer checked the fence. :slightly_smiling_face:"),
            ("TITLE of a note\rThe next LINE remains part of it.",
             "title of a note the next line remains part of it."),
        ]
        detector, torch = self.detector, self.detector.torch
        batch = detector.detect_many([source for source, _ in pairs])
        for i, (source, expected_model_input) in enumerate(pairs):
            # Explicit reference inputs and direct raw forward pass, without
            # calling our preprocessing/windowing/score aggregation functions.
            inputs = detector.tokenizer(expected_model_input, return_tensors="pt").to(detector.device)
            with torch.no_grad():
                logits = detector.model(**inputs).logits
            probs = torch.softmax(logits.float(), dim=-1)[0].cpu()
            expected = sum(j * float(p) for j, p in enumerate(probs)) / (len(probs) - 1)
            verdict, _ = detector.detect(source)
            self.assertLess(abs(verdict.score - expected), 2e-5)
            self.assertEqual(verdict.bucket, int(probs.argmax()))
            self.assertLess(abs(batch[i].score - expected), 2e-4)
            self.assertEqual(batch[i].bucket, verdict.bucket)
            print(f"reference case {i}: raw={expected:.8f}, single={verdict.score:.8f}, batch={batch[i].score:.8f}")

    def test_long_emoji_text_fits_and_offsets_cover_scored_content(self):
        text = "Sure!\n\n" + ("An UPPERCASE detail \U0001f642 sits beside another fact.\n" * 160)
        source, model_input, ranges = self.detector._prepare(text, True)
        windows = self.detector.windows(model_input)
        self.assertGreater(len(windows), 1)
        for _, _, chunk in windows:
            self.assertLessEqual(len(self.detector.tokenizer(chunk)["input_ids"]), 512)
        verdict, details = self.detector.detect(text)
        self.assertEqual(verdict.truncated_windows, len(windows))
        self.assertEqual(details[0]["start"], clean_text(text).index("An UPPERCASE"))
        self.assertEqual(details[-1]["owned_end"], len(source))
        for left, right in zip(details, details[1:]):
            self.assertEqual(left["owned_end"], right["owned_start"])
        self.assertAlmostEqual(sum(verdict.probs), 1, places=5)


if __name__ == "__main__":
    unittest.main()
