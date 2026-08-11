"""Every MCP tool, over an in-memory client, WITH assertions.

This file previously only printed results, so it passed no matter what the
server returned. Mutation testing caught that: breaking chain_assemble's
completeness gate did not fail a single suite. Assert on values, not on the
absence of exceptions.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["EDITLENS_DB"] = str(Path(tempfile.mkdtemp()) / "tools.db")

from fastmcp import Client  # noqa: E402

from editlens_mcp import server  # noqa: E402

AI = ("In today's rapidly evolving landscape, stakeholders must leverage synergies to "
      "drive transformative outcomes across the organizational ecosystem. Moreover, it "
      "is important to note that robust frameworks enable scalable growth.")
HUMAN = ("I burnt the rice again. Third time this month. My flatmate calls it a tradition, "
         "which is generous of her considering she has to eat it. We had toast instead.")

EXPECTED_TOOLS = {
    "detector_info", "detector_unload", "detect", "detect_batch", "detect_spans",
    "chain_create", "chain_submit", "chain_status", "chain_history",
    "chain_get_text", "chain_assemble", "chain_list", "chain_delete",
}


async def main() -> None:
    async with Client(server.mcp) as c:

        async def call(name, args=None):
            return (await c.call_tool(name, args or {})).data

        # ---------------------------------------------------------- registration
        names = {t.name for t in await c.list_tools()}
        assert names == EXPECTED_TOOLS, f"tool mismatch: {names ^ EXPECTED_TOOLS}"
        print(f"  {len(names)} tools registered, names exact")

        # -------------------------------------------------------------- detector
        info = await call("detector_info")
        assert info["ok"] is True
        assert info["checkpoint"] and info["db_path"]
        assert info["dtype"] in {"float32", "float16"}
        assert info["accelerator"] in {"cuda", "mps", "cpu"}
        # Guards the mutation "always return float16": the default must be float32.
        assert info["dtype"] == "float32", f"default dtype should be float32, got {info['dtype']}"
        print(f"  detector_info: {info['device']}/{info['dtype']} on {info['platform']}")

        # ---------------------------------------------------------------- detect
        ai = await call("detect", {"text": AI})
        hu = await call("detect", {"text": HUMAN})
        for r in (ai, hu):
            assert r["ok"] and 0.0 <= r["score"] <= 1.0
            assert abs(sum(r["probs"]) - 1.0) < 0.01, r["probs"]
            assert r["word_count"] > 0 and r["windows"] >= 1
        # Direction matters: a detector that cannot separate these is useless.
        assert ai["score"] > hu["score"] + 0.2, (ai["score"], hu["score"])
        print(f"  detect: ai={ai['score']:.3f} > human={hu['score']:.3f}")

        long_doc = (AI + " " + HUMAN + " ") * 12
        win = await call("detect", {"text": long_doc, "include_windows": True})
        assert win["ok"] and win["windows"] > 1, win["windows"]
        detail = win["window_detail"]
        assert len(detail) == win["windows"]
        # owned ranges must partition the document exactly once (guards the
        # mutation that restores double-counted overlaps)
        assert detail[0]["owned_start"] == 0
        for a, b in zip(detail, detail[1:]):
            assert a["owned_end"] == b["owned_start"], (a["owned_end"], b["owned_start"])
        total = sum(w["words"] for w in detail)
        assert abs(total - win["word_count"]) <= max(3, 0.03 * win["word_count"]), (
            f"window weights {total} != document words {win['word_count']}")
        print(f"  detect(include_windows): {win['windows']} windows partition cleanly")

        # ----------------------------------------------------------- detect_batch
        batch = await call("detect_batch", {"texts": [AI, HUMAN, AI]})
        assert batch["ok"] and len(batch["results"]) == 3
        assert [r["index"] for r in batch["results"]] == [0, 1, 2]
        assert batch["best_index"] == 1, batch["best_index"]
        assert batch["best_score"] == min(r["score"] for r in batch["results"])
        assert abs(batch["results"][0]["score"] - batch["results"][2]["score"]) < 1e-6
        print(f"  detect_batch: best_index={batch['best_index']}, duplicates agree")

        # ----------------------------------------------------------- detect_spans
        spans = await call("detect_spans", {"text": AI + " " + HUMAN, "top": 10})
        assert spans["ok"] and spans["unit_count"] >= 2
        assert len(spans["worst_units"]) <= 10
        scores = [u["score"] for u in spans["worst_units"]]
        assert scores == sorted(scores, reverse=True), "units must be worst-first"
        assert spans["min_words_used"] is not None
        assert isinstance(spans["unreliable_units"], int)
        for u in spans["worst_units"]:
            assert isinstance(u["reliable"], bool)
            assert u["start"] < u["end"]
        print(f"  detect_spans: {spans['unit_count']} units, sorted, "
              f"{spans['unreliable_units']} unreliable")

        para = await call("detect_spans",
                          {"text": AI + "\n\n" + HUMAN, "granularity": "paragraph"})
        assert para["ok"] and para["unit_count"] == 2, para["unit_count"]
        # Paragraphs are authored boundaries; no threshold was applied.
        assert para["min_words_used"] is None and para["granularity_relaxed"] is False
        print(f"  detect_spans(paragraph): {para['unit_count']} units, min_words_used=None")

        # ----------------------------------------------------------------- chains
        ch = await call("chain_create",
                        {"name": "t", "target_score": 0.35, "segments": ["a", "b", "b"]})
        assert ch["ok"] and ch["segments"] == ["a", "b"], ch["segments"]  # de-duplicated
        cid = ch["chain_id"]
        print(f"  chain_create: duplicate segment de-duplicated -> {ch['segments']}")

        s1 = await call("chain_submit", {"chain_id": cid, "text": AI, "segment": "a",
                                         "note": "first pass"})
        assert s1["ok"] and s1["step"] == 1
        assert s1["target_met"] is False and s1["is_new_best"] is True
        assert s1["delta_vs_previous"] is None and s1["delta_vs_best"] is None
        assert s1["worst_spans"], "AI text should produce span feedback"

        s2 = await call("chain_submit", {"chain_id": cid, "text": HUMAN, "segment": "a",
                                         "note": "rewrote it"})
        assert s2["ok"] and s2["step"] == 2
        assert s2["delta_vs_previous"] < 0, "score should have improved"
        assert s2["is_new_best"] is True and s2["best_score"] == s2["score"]
        print(f"  chain_submit: {s1['score']:.3f} -> {s2['score']:.3f} "
              f"(delta {s2['delta_vs_previous']:+.3f})")

        skipped = await call("chain_submit", {"chain_id": cid, "text": HUMAN,
                                              "segment": "a", "span_feedback": False})
        assert "worst_spans" not in skipped
        assert "too short" not in skipped["next_action"].lower(), skipped["next_action"]
        print("  chain_submit(span_feedback=False): does not claim 'too short'")

        # assemble on a PARTIAL chain -- segment "b" has no drafts
        part = await call("chain_assemble", {"chain_id": cid})
        assert part["ok"] and part["missing_segments"] == ["b"]
        assert part["complete"] is False
        assert part["target_met"] is False, "incomplete document must not be target_met"
        assert "warning" in part
        print(f"  chain_assemble(partial): complete=False target_met=False "
              f"missing={part['missing_segments']}")

        await call("chain_submit", {"chain_id": cid, "text": HUMAN, "segment": "b"})
        full = await call("chain_assemble", {"chain_id": cid, "include_text": False})
        assert full["ok"] and full["complete"] is True and not full["missing_segments"]
        assert "text" not in full, "include_text=False must omit the document"
        assert [p["segment"] for p in full["per_segment"]] == ["a", "b"]
        assert full["target_met"] == (full["complete"] and full["score_met"])
        print(f"  chain_assemble(full): score={full['document_score']:.3f} complete=True")

        # ---------------------------------------------------------- status/history
        st = await call("chain_status", {"chain_id": cid})
        assert st["ok"] and st["total_steps"] == 4
        assert [s["segment"] for s in st["segments"]] == ["a", "b"]
        assert st["segments"][0]["steps"] == 3

        hist = await call("chain_history", {"chain_id": cid, "segment": "a"})
        assert hist["ok"] and hist["returned"] == 3
        assert [h["step"] for h in hist["trajectory"]] == [1, 2, 3], "must be chronological"
        assert hist["trajectory"][0]["note"] == "first pass"
        print(f"  chain_history: {hist['returned']} steps, chronological")

        bad_seg = await call("chain_history", {"chain_id": cid, "segment": "nope"})
        assert bad_seg["ok"] is False and "no such segment" in bad_seg["error"]
        print("  chain_history(bad segment): rejected, not silently empty")

        # ------------------------------------------------------------- get_text
        best = await call("chain_get_text", {"chain_id": cid, "segment": "a", "step": "best"})
        assert best["ok"] and best["text"].strip()
        assert best["score"] == min(h["score"] for h in hist["trajectory"])
        latest = await call("chain_get_text", {"chain_id": cid, "segment": "a",
                                               "step": "latest"})
        assert latest["ok"] and latest["step"] == 3
        exact = await call("chain_get_text", {"chain_id": cid, "segment": "a", "step": 1})
        assert exact["ok"] and exact["step"] == 1
        # The stored draft must actually be the draft, not an empty string.
        assert exact["text"].split() == AI.split(), "stored text does not match submission"
        print(f"  chain_get_text: best=step{best['step']} latest=step3 exact=step1, "
              f"text round-trips")

        # ----------------------------------------------------------- list/delete
        lst = await call("chain_list")
        assert lst["ok"] and any(x["chain_id"] == cid for x in lst["chains"])
        row = next(x for x in lst["chains"] if x["chain_id"] == cid)
        assert row["steps"] == 4 and row["segments"] == ["a", "b"]

        # ------------------------------------------------------------ error paths
        errors = {
            ("detect", frozenset({"text": ""}.items())): "empty text",
            ("detect_batch", frozenset({"texts": ()}.items())): "empty",
            ("chain_status", frozenset({"chain_id": "nope"}.items())): "no such chain",
            ("chain_get_text", frozenset({"chain_id": cid, "segment": "a",
                                          "step": 999}.items())): "no such step",
        }
        for (name, frozen), fragment in errors.items():
            args = dict(frozen)
            if name == "detect_batch":
                args["texts"] = []
            r = await call(name, args)
            assert r["ok"] is False, f"{name} should have failed: {r}"
            assert fragment in r["error"].lower(), (name, r["error"])
            assert "error_type" in r
        print(f"  error paths: {len(errors)} return ok=False dicts with error_type")

        # ---------------------------------------------------------------- unload
        u = await call("detector_unload")
        assert u["ok"] and isinstance(u["was_loaded"], bool)
        again = await call("detect", {"text": HUMAN})
        assert again["ok"], "must reload transparently after unload"
        print("  detector_unload: released and reloaded transparently")

        d = await call("chain_delete", {"chain_id": cid})
        assert d["ok"] and d["deleted_steps"] == 4
        gone = await call("chain_status", {"chain_id": cid})
        assert gone["ok"] is False
        print(f"  chain_delete: removed {d['deleted_steps']} steps")

    print("TOOL TESTS PASSED")


if __name__ == "__main__":
    print("tool tests (in-memory MCP client)")
    asyncio.run(main())
