"""Gap-closing verification: precision, long-text windowing, multi-segment
chains, restart persistence, and edge cases."""

import asyncio, os, sys, tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ["EDITLENS_DB"] = str(TMP / "v.db")
sys.path.insert(0, r"C:\Users\Aryan\MCP-EditLens")

from editlens_mcp.detector import EditLensDetector, count_words

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


def test_precision():
    """Does the float16 option agree with the float32 default?"""
    print("=== precision: float16 (opt-in) vs float32 (default) ===")
    d16 = EditLensDetector(device="cuda", dtype="float16")
    d32 = EditLensDetector(device="cuda", dtype="float32")
    worst = 0.0
    for i, t in enumerate(TEXTS):
        s16 = d16.detect(t)[0].score
        s32 = d32.detect(t)[0].score
        diff = abs(s16 - s32)
        worst = max(worst, diff)
        print(f"  text{i}: fp16={s16:.6f}  fp32={s32:.6f}  diff={diff:.6f}")
    print(f"  --> max disagreement: {worst:.6f}")
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

    huge = unit * 60
    v2, w2 = det.detect(huge)
    print(f"  {count_words(huge)} words -> {v2.truncated_windows} windows, score={v2.score:.4f}")
    assert 0.0 <= v2.score <= 1.0
    return True


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
    assert rows and seg["steps"] == 1 and text
    return asm


def test_determinism(det):
    print("\n=== determinism ===")
    a = [det.detect(TEXTS[0])[0].score for _ in range(3)]
    print(f"  same input x3 -> {a}")
    assert len(set(a)) == 1, "scores must be reproducible"


if __name__ == "__main__":
    worst = test_precision()
    det = EditLensDetector(device="cuda")
    det.ensure_loaded()
    test_long_text(det)
    test_edges(det)
    test_determinism(det)
    asyncio.run(test_multisegment_and_restart())
    print(f"\nALL CHECKS PASSED (max fp16/fp32 disagreement {worst:.6f})")
