"""Gap-closing verification: precision, long-text windowing, multi-segment
chains, restart persistence, and edge cases."""

import asyncio, os, sys, tempfile, threading, time
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ["EDITLENS_DB"] = str(TMP / "v.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from editlens_mcp.detector import (
    EditLensDetector,
    _sentence_spans,
    clean_text,
    count_words,
    pick_device,
    split_units,
    split_units_adaptive,
)

import torch  # noqa: E402

# Never hardcode "cuda": on a Mac or a CPU-only box that aborts the whole suite,
# and run_tests.py is the documented way to verify an install.
DEVICE = os.environ.get("EDITLENS_DEVICE") or pick_device(torch)
HAS_GPU = DEVICE in {"cuda", "mps"}

TEXTS = [
    "In today's rapidly evolving landscape, stakeholders must leverage synergies to "
    "drive transformative outcomes across the organizational ecosystem. Furthermore, "
    "it is important to note that robust frameworks enable scalable growth.",
    "I burnt the rice again. Third time this month. My flatmate has started calling it "
    "a tradition, which is generous of her considering she has to eat it.",
    "The mitochondrion is an organelle found in most eukaryotic cells. It generates "
    "adenosine triphosphate through oxidative phosphorylation, a process occurring "
    "across the inner mitochondrial membrane.",
    "hey are we still on for thursday? i can bring the thing if you want, lmk",
]


def test_text_pipeline_beyond_english():
    """The splitter and counter on text the model was not calibrated on.

    All four defects here shipped silently: an unspaced Chinese clause counted
    as one \\b-word (3-4x undercount, tripping every word-count threshold); a
    Chinese document had ZERO sentence boundaries because _SENT_END demanded
    trailing whitespace that CJK never writes; NFD input (macOS, PDF paste)
    doubled the word count of accented-language text; and a hyphen-bulleted
    list collapsed to one unit, silently disabling all span feedback on the
    draft shapes agents write most.
    """
    import unicodedata

    # CJK counting: ~2 chars per word-equivalent, nowhere near the clause count.
    zh = ("人工智能正在改变我们的写作方式。它可以生成流畅的文本，但有时缺乏个性。"
          "检测工具因此变得重要。")
    n = count_words(zh)
    assert 15 <= n <= 35, f"Chinese char-count heuristic off: {n} for {len(zh)} chars"

    # CJK sentence boundaries: 。 needs no trailing whitespace.
    assert len(_sentence_spans(zh)) == 3, _sentence_spans(zh)

    # NFC/NFD must agree, or the same visible string passes thresholds in one
    # normalisation form and fails them in the other.
    vi = "Trí tuệ nhân tạo đang thay đổi cách chúng ta viết văn bản hằng ngày"
    nfc, nfd = (count_words(unicodedata.normalize(f, vi)) for f in ("NFC", "NFD"))
    assert nfc == nfd, f"NFC {nfc} != NFD {nfd}"

    # Bullet lists: line-boundary fallback instead of one giant unit.
    bullets = "\n".join(
        f"- Step {i} configures the widget and validates the gadget output carefully."
        for i in range(20)
    )
    units = split_units(bullets, "sentence", 1)
    assert len(units) == 20, f"line fallback failed: {len(units)} units"
    for (a0, a1), (b0, b1) in zip(units, units[1:]):
        assert a1 == b0, "line-fallback units must tile"
    merged, used = split_units_adaptive(bullets, "sentence", 25)
    assert len(merged) >= 2, "merged bullet units must still be plural"

    # Abbreviations no longer end sentences: one boundary here, not four.
    ab = "Dr. Chen and Prof. Lima met J. Smith at 3 p.m. to review Vol. 2. It went well."
    spans = _sentence_spans(ab)
    assert len(spans) == 2, [ab[a:b] for a, b in spans]

    # ...but the digit rule must stay a LIST-MARKER rule. Its first version
    # suppressed every "<number>." boundary, so ordinary prose ending in a year
    # or a price lost its sentence breaks -- the fix mangled English to protect
    # list formatting.
    prose_nums = "The study ran until 2024. It found nothing. Costs hit $5. Nobody minded."
    assert len(_sentence_spans(prose_nums)) == 4, _sentence_spans(prose_nums)
    lst = ("1. Configure the widget carefully today.\n"
           "2. Validate the gadget output now.\n3. Ship it.")
    assert len(split_units(lst, "sentence", 1)) == 3, "list markers must not split"

    # English prose is untouched by all of the above.
    prose = "One sentence here. Another follows it. A third one closes the set."
    assert len(_sentence_spans(prose)) == 3
    assert count_words("plain english words here") == 4
    print(f"  CJK counts {n} words for {len(zh)} chars and splits into 3 sentences; "
          f"NFC==NFD; 20-bullet list yields 20 tiled units; abbreviations survive")


def test_dtype_defaults():
    """The default must be float32 -- the precision the checkpoint ships in."""
    print("=== dtype resolution ===")
    d = EditLensDetector(device=DEVICE)
    assert d.info()["dtype"] == "float32", d.info()["dtype"]
    if HAS_GPU:
        assert EditLensDetector(device=DEVICE, dtype="float16").info()["dtype"] == "float16"
    # float16 is GPU-only; on CPU the request must be downgraded, not honoured.
    assert EditLensDetector(device="cpu", dtype="float16").info()["dtype"] == "float32"
    # An unrecognised value falls back rather than crashing.
    assert EditLensDetector(device=DEVICE, dtype="bfloat16").info()["dtype"] == "float32"
    print("  default=float32, fp16 GPU-only, unknown values fall back")


def test_precision():
    """Does the float16 option agree with the float32 default?"""
    print("\n=== precision: float16 (opt-in) vs float32 (default) ===")
    if not HAS_GPU:
        print(f"  skipped: float16 needs a GPU (device={DEVICE})")
        return 0.0
    d16 = EditLensDetector(device=DEVICE, dtype="float16")
    d32 = EditLensDetector(device=DEVICE, dtype="float32")
    # Guards the mutation "always return float16", which made both detectors
    # identical and the measured difference a meaningless 0.
    assert d16.info()["dtype"] == "float16" and d32.info()["dtype"] == "float32"
    worst = 0.0
    for i, t in enumerate(TEXTS):
        s16 = d16.detect(t)[0].score
        s32 = d32.detect(t)[0].score
        diff = abs(s16 - s32)
        worst = max(worst, diff)
        print(f"  text{i}: fp16={s16:.6f}  fp32={s32:.6f}  diff={diff:.6f}")
        assert diff < 0.01, f"fp16 disagrees with fp32 by {diff} on text{i}"
    assert worst > 0.0, "fp16 and fp32 identical to the bit -- are both really loading?"
    print(f"  --> max disagreement: {worst:.6f} (non-zero, under 0.01)")
    return worst


def test_long_text(det):
    """Documents past the 512-token limit must window automatically."""
    print("\n=== long text / automatic windowing ===")
    unit = TEXTS[0] + " " + TEXTS[2] + " "
    long_doc = unit * 12
    words = count_words(long_doc)
    v, windows = det.detect(long_doc)
    print(f"  {words} words -> {v.truncated_windows} windows, score={v.score:.4f}")
    assert v.truncated_windows > 1, "should have split"
    assert len(windows) == v.truncated_windows
    # windows must tile the document without gaps at the seams
    for a, b in zip(windows, windows[1:]):
        assert b["start"] <= a["end"], f"gap between windows: {a['end']} -> {b['start']}"
    covered = windows[-1]["end"]
    print(f"  coverage: char 0..{covered} of {len(long_doc)}  seams ok")
    assert covered >= len(long_doc.strip()) - 5, "tail of document not scored"

    # Owned ranges must PARTITION the document: overlaps split at the midpoint so
    # seam text is weighted once. Without this, restoring the old double-counting
    # passes every other assertion here.
    assert windows[0]["owned_start"] == 0
    for a, b in zip(windows, windows[1:]):
        assert a["owned_end"] == b["owned_start"], (
            f"owned ranges do not tile: {a['owned_end']} -> {b['owned_start']}")
    weighted = sum(w["words"] for w in windows)
    assert abs(weighted - v.word_count) <= max(3, 0.03 * v.word_count), (
        f"window weights sum to {weighted} but document has {v.word_count} words "
        "-- overlaps are being counted twice")
    print(f"  owned ranges partition cleanly: weights {weighted} == doc {v.word_count}")

    huge = unit * 60
    v2, w2 = det.detect(huge)
    print(f"  {count_words(huge)} words -> {v2.truncated_windows} windows, score={v2.score:.4f}")
    assert 0.0 <= v2.score <= 1.0
    assert abs(sum(w["words"] for w in w2) - v2.word_count) <= max(3, 0.03 * v2.word_count)
    return True


def test_window_weighting_is_exact(det):
    """The document score IS the word-weighted mean over the owned ranges.

    The assertions in test_long_text only inspect the ranges the detector
    reports; none of them looks at what the combined score was actually built
    from. Restoring a plain unweighted mean therefore passed every one of them.

    Feed the combiner known per-window scores so the arithmetic is checkable to
    the bit, rather than hoping two real windows disagree enough for a tolerance
    to notice. The weights are recomputed here from `owned_start`/`owned_end`, so
    weighting by each window's FULL span (which double-counts the overlaps) is
    caught too.
    """
    print("\n=== window combination is word-weighted, exactly ===")
    doc = (TEXTS[0] + " " + TEXTS[2] + " ") * 12
    source = clean_text(doc)

    def fake_scores(texts):
        n = max(len(texts), 2)
        return [(i / (n - 1), [1.0 - i / (n - 1), i / (n - 1), 0.0, 0.0])
                for i in range(len(texts))]

    real = det._score_batch
    det._score_batch = fake_scores
    try:
        v, details = det.detect(doc)
    finally:
        det._score_batch = real

    assert len(details) > 1, f"need a multi-window document, got {len(details)}"
    n = max(len(details), 2)
    weights, scores = [], []
    for i, d in enumerate(details):
        w = float(count_words(source[d["owned_start"]:d["owned_end"]])) or 1.0
        weights.append(w)
        scores.append(i / (n - 1))
        assert d["words"] == int(w), (
            f"window {i} was weighted by {d['words']} words but owns {int(w)}")

    expected = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    plain = sum(scores) / len(scores)
    assert abs(expected - plain) > 1e-6, (
        "these windows are equal-weight, so the test cannot tell a weighted mean "
        "from an unweighted one -- pick a document where they differ")
    assert abs(v.score - expected) < 1e-9, (
        f"score {v.score!r} is not the word-weighted mean {expected!r} "
        f"(an unweighted mean would give {plain!r})")
    # The probability vector is combined the same way, and stays a distribution.
    assert abs(v.probs[1] - expected) < 1e-9, (v.probs, expected)
    assert abs(sum(v.probs) - 1.0) < 1e-9, v.probs
    # The label follows the most probable aggregate bucket.
    assert v.bucket == max(range(len(v.probs)), key=v.probs.__getitem__), (v.bucket, v.probs)
    assert v.label == det.bucket_names[v.bucket], (v.label, v.bucket)
    print(f"  {len(details)} windows, weights {[int(w) for w in weights]}")
    print(f"  score={v.score:.9f} == weighted {expected:.9f} "
          f"(unweighted would be {plain:.9f})")


def test_batch_agrees_with_single(det):
    """detect_batch must return the same number detect does for the same text.

    They are separate code paths -- detect_many has a one-shot batched fast path
    for short inputs, detect always windows -- and detect_batch exists so drafts
    can be compared against each other and against earlier detect calls. If the
    two paths normalise differently, or derive bucket and label differently,
    those comparisons quietly stop meaning anything.
    """
    print("\n=== batched path agrees with the single path ===")
    messy = [
        "  In today's  rapidly evolving landscape,\r\n stakeholders must leverage "
        "synergies to drive transformative outcomes.   ",
        "I burnt the rice again.\r\n\r\n\r\n\r\nThird time this month.   We had toast "
        "instead, which was fine.",
    ]
    batch = det.detect_many(messy)
    assert len(batch) == len(messy)
    for v, t in zip(batch, messy):
        single, _ = det.detect(t)
        assert abs(v.score - single.score) < 1e-6, (t[:30], v.score, single.score)
        assert v.bucket == single.bucket and v.label == single.label
        assert v.word_count == single.word_count, (v.word_count, single.word_count)
        assert v.label == det.bucket_names[v.bucket]
        assert 0.0 <= v.score <= 1.0
    print(f"  {len(messy)} messy inputs: batched and single scores identical")


def test_edges(det):
    print("\n=== edge cases ===")
    cases = {
        "3 words": "hello there friend",
        "unicode/emoji": "Café naïve — résumé 😀 \u200b test — ok?",
        "newlines only": "a\n\n\n\n\nb",
        "no sentence end": "this text just goes on and never terminates properly",
        "repeated char": "aaaaaaaaaa " * 30,
    }
    for name, txt in cases.items():
        v, _ = det.detect(txt)
        assert 0.0 <= v.score <= 1.0, name
        print(f"  {name:<18} score={v.score:.4f}  ok")
    for bad in ["", "   ", "\n\n"]:
        try:
            det.detect(bad)
            raise AssertionError(f"empty input {bad!r} should raise")
        except ValueError:
            pass
    print("  empty inputs rejected cleanly")


async def test_multisegment_and_restart():
    print("\n=== multi-segment chain + restart persistence ===")
    from fastmcp import Client
    from editlens_mcp import server

    async with Client(server.mcp) as c:
        cid = (await c.call_tool("chain_create", {
            "name": "report", "target_score": 0.5,
            "segments": ["intro", "body", "conclusion"]})).data["chain_id"]
        for seg, txt in zip(["intro", "body", "conclusion"], TEXTS):
            r = (await c.call_tool("chain_submit", {
                "chain_id": cid, "text": txt, "segment": seg,
                "span_feedback": False})).data
            print(f"  {seg:<11} score={r['score']:.4f} met={r['target_met']}")
        st = (await c.call_tool("chain_status", {"chain_id": cid})).data
        print(f"  status: {st['total_steps']} steps, pending={st['pending']}")
        asm = (await c.call_tool("chain_assemble", {"chain_id": cid})).data
        parts = [s["score"] for s in asm["per_segment"]]
        print(f"  assembled doc={asm['document_score']:.4f}  parts={parts}")
        assert len(asm["per_segment"]) == 3 and not asm["missing_segments"]

    # simulate a restart: brand-new store object against the same file
    from editlens_mcp.chains import ChainStore
    fresh = ChainStore(os.environ["EDITLENS_DB"])
    rows = fresh.list_chains()
    seg = fresh.segment_stats(cid, "body")
    text = fresh.best_step(cid, "intro")["text"]
    fresh.close()
    print(f"  after restart: {len(rows)} chain(s), body={seg['steps']} step(s), "
          f"intro text recovered={text[:24]!r}")
    assert rows and seg["steps"] == 1
    # Not just truthy: the recovered draft must BE the draft that was submitted.
    assert text.split() == TEXTS[0].split(), "recovered text does not match submission"
    return asm


def test_determinism(det):
    print("\n=== determinism ===")
    a = [det.detect(TEXTS[0])[0].score for _ in range(3)]
    print(f"  same input x3 -> {a}")
    assert len(set(a)) == 1, "scores must be reproducible"


def test_load_records_where_the_model_actually_landed():
    """A real load that falls back must be REMEMBERED, not re-predicted.

    The model unloads after a few minutes idle, so unloaded is where
    detector_info spends most of its life. Recomputing the device from the
    request there reports one already proven unusable. The earlier test faked
    `_last_device`; this one drives the real `_load`.
    """
    print("\n=== load records the real device ===")
    det = EditLensDetector(device="cuda:99", idle_unload_seconds=0)
    det.ensure_loaded()  # placement fails, falls back to CPU
    assert det.loaded
    assert det.device == "cpu" and det.dtype == "float32", (det.device, det.dtype)
    assert det._last_device == "cpu" and det._last_dtype == "float32", (
        f"_load did not record the landing device: "
        f"{det._last_device}/{det._last_dtype}")

    assert det.unload() is True
    info = det.info()
    assert info["device"] == "cpu", (
        f"unloaded info re-predicted the request: {info['device']}")
    assert info["dtype"] == "float32", info["dtype"]
    assert info["accelerator"] == "cpu", info["accelerator"]
    assert info["device_fallback"] and "cuda:99 -> cpu" in info["device_fallback"], (
        info["device_fallback"])
    assert info["vram_mb"] is None and info["loaded"] is False
    print(f"  real load fell back to {info['device']}/{info['dtype']}; "
          f"still reported after unload")


def test_unload_waits_for_an_in_flight_request():
    """A manual unload during traffic must not null the model mid-request.

    Without `_infer_lock`, `detector_unload` (or a client calling it while
    another is scoring) surfaces as `'NoneType' object is not callable` from
    windows()/detect_many(), which reach self.tokenizer directly.
    """
    print("\n=== unload vs. an in-flight request ===")
    det = EditLensDetector(device=DEVICE, idle_unload_seconds=0)
    det.detect(TEXTS[0])
    assert det.loaded

    entered, release = threading.Event(), threading.Event()
    real_score = det._score_batch

    def slow(texts):
        entered.set()
        assert release.wait(30), "test deadlock"
        return real_score(texts)

    det._score_batch = slow
    scored: list = []
    worker = threading.Thread(target=lambda: scored.append(det.detect(TEXTS[0])[0].score))
    worker.start()
    assert entered.wait(30), "request never reached the model"

    unloaded: list = []
    unloader = threading.Thread(target=lambda: unloaded.append(det.unload()))
    unloader.start()
    time.sleep(0.3)
    assert not unloaded, "unload() completed while a request was in flight"
    assert det.loaded and det.model is not None and det.tokenizer is not None, (
        "the model was dropped underneath a running request")

    release.set()
    worker.join(60)
    unloader.join(60)
    assert scored and isinstance(scored[0], float), scored
    assert unloaded == [True], unloaded
    assert not det.loaded
    print("  unload waited for the in-flight request, then freed the model")


if __name__ == "__main__":
    test_text_pipeline_beyond_english()
    test_dtype_defaults()
    worst = test_precision()
    det = EditLensDetector(device=DEVICE)
    det.ensure_loaded()
    test_long_text(det)
    test_window_weighting_is_exact(det)
    test_batch_agrees_with_single(det)
    test_edges(det)
    test_determinism(det)
    test_unload_waits_for_an_in_flight_request()
    test_load_records_where_the_model_actually_landed()
    asyncio.run(test_multisegment_and_restart())
    print(f"\nALL CHECKS PASSED (max fp16/fp32 disagreement {worst:.6f})")
