"""Offline smoke test: exercises chunking, chain storage and tool wiring with a
stubbed model, then optionally runs the real detector if it can be loaded.

    python smoke_test.py          # stub only
    python smoke_test.py --real   # also load the real checkpoint
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from editlens_mcp.chains import ChainStore  # noqa: E402
from editlens_mcp.detector import clean_text, count_words, split_units  # noqa: E402

SAMPLE = (
    "In today's rapidly evolving landscape, it is important to note that the "
    "implementation of robust solutions remains paramount. Moreover, stakeholders "
    "must carefully consider the multifaceted implications.\n\n"
    "I went to the shop yesterday. Rain the whole way. Forgot the list, obviously, "
    "so I came back with crisps and no milk."
)


def test_text_utils() -> None:
    cleaned = clean_text("a  b\r\n\r\n\r\n\r\nc   \n")
    assert cleaned == "a b\n\nc", repr(cleaned)
    assert count_words("one two three's") == 3
    units = split_units(SAMPLE, "sentence", min_words=10)
    assert len(units) >= 2, units
    for a, b in units:
        assert SAMPLE[a:b].strip()
    paras = split_units(SAMPLE, "paragraph")
    assert len(paras) == 2, paras
    print(f"  text utils ok  ({len(units)} sentence units, {len(paras)} paragraphs)")


def test_chain_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ChainStore(Path(tmp) / "t.db")
        ch = store.create("essay", target_score=0.2, segments=["intro", "body"])
        cid = ch["chain_id"]

        for i, score in enumerate([0.91, 0.62, 0.41, 0.18], start=1):
            n = store.add_step(cid, "intro", f"draft {i}", score, 2, "x", 50, [0.25] * 4, f"pass {i}")
            assert n == i
        store.add_step(cid, "body", "body draft", 0.55, 2, "x", 80, [0.25] * 4, None)

        best = store.best_step(cid, "intro")
        assert best["step_no"] == 4 and best["score"] == 0.18
        assert store.latest_step(cid, "intro")["step_no"] == 4
        assert len(store.history(cid, "intro")) == 4
        assert store.history(cid, "intro")[0]["step"] == 1  # chronological

        stats = store.segment_stats(cid, "intro")
        assert stats["steps"] == 4 and stats["best_score"] == 0.18

        store.add_segment(cid, "conclusion")
        assert "conclusion" in store.list_chains()[0]["segments"]
        assert store.list_chains()[0]["steps"] == 5

        assert store.delete(cid) == 5
        store.close()
        print("  chain store ok  (4 revisions, 3 segments, cascade delete)")


def test_tool_registration() -> None:
    import asyncio

    from editlens_mcp import server

    tools = {t.name for t in asyncio.run(server.mcp.list_tools())}
    expected = {
        "detector_info", "detect", "detect_batch", "detect_spans",
        "chain_create", "chain_submit", "chain_status", "chain_history",
        "chain_get_text", "chain_assemble", "chain_list", "chain_delete",
    }
    missing = expected - tools
    assert not missing, f"missing tools: {missing}"
    # FastMCP 3 leaves the decorated callable usable as a plain function.
    info = server.detector_info()
    assert info["checkpoint"] and "db_path" in info
    print(f"  server ok  ({len(tools)} tools registered, model lazy-loads)")


def test_real_model() -> None:
    from editlens_mcp.detector import EditLensDetector

    det = EditLensDetector()
    det.ensure_loaded()
    print(f"  loaded on {det.device} ({det.dtype}), {det.n_buckets} buckets")
    ai, human = SAMPLE.split("\n\n")
    for name, txt in (("ai-ish", ai), ("human-ish", human)):
        v, _ = det.detect(txt)
        print(f"    {name:<10} score={v.score:.3f}  {v.label}")
    batch = det.detect_many([ai, human])
    assert len(batch) == 2
    print("  real model ok")


if __name__ == "__main__":
    print("smoke test")
    test_text_utils()
    test_chain_store()
    test_tool_registration()
    if "--real" in sys.argv:
        test_real_model()
    else:
        print("  (skipping real model; pass --real to load the checkpoint)")
    print("PASS")
