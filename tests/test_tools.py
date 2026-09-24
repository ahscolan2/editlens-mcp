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
from editlens_mcp.detector import Verdict  # noqa: E402

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
        # Not merely truthy: EDITLENS_DB is how a client keeps its chains out of
        # the shared default database, and silently ignoring it would send every
        # test in this file at %LOCALAPPDATA%.
        assert Path(info["db_path"]) == Path(os.environ["EDITLENS_DB"]), (
            info["db_path"], os.environ["EDITLENS_DB"])
        assert Path(info["db_path"]).exists(), "store did not create its database"
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

        # ------------------------------------------- score / bucket / label agree
        # The README promises a score in [0,1] "along with the discrete bucket
        # label and the full probability distribution". Range alone is a weak
        # promise: it holds for any number between 0 and 1. These three are the
        # SAME quantity in three forms, and they must stay derivable from one
        # another -- the score is the expected bucket index under the softmax,
        # normalised to [0,1]; the bucket is the largest probability on the index
        # scale; the label is that bucket's name. Without this, mutations that
        # returned the argmax instead of the expectation, scaled the score, or
        # shifted the label by one bucket all passed every suite.
        loaded = await call("detector_info")
        names, nb = loaded["bucket_names"], loaded["n_buckets"]
        assert len(names) == nb, (names, nb)
        for tag, r in (("ai", ai), ("human", hu)):
            expected = sum(i * p for i, p in enumerate(r["probs"])) / (nb - 1)
            assert abs(r["score"] - expected) < 0.002, (
                f"{tag}: score {r['score']} is not the expectation over probs "
                f"{r['probs']} ({expected:.4f})")
            assert r["bucket"] == max(range(nb), key=lambda i: r["probs"][i]), (
                f"{tag}: bucket {r['bucket']} does not match score {r['score']}")
            assert 0 <= r["bucket"] < nb, r["bucket"]
            assert r["label"] == names[r["bucket"]], (
                f"{tag}: label {r['label']!r} is not bucket {r['bucket']} "
                f"({names[r['bucket']]!r})")
        print(f"  score/bucket/label consistent: ai -> bucket {ai['bucket']} "
              f"'{ai['label']}', human -> bucket {hu['bucket']} '{hu['label']}'")

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

        # The flag is a flag: the same document without it must stay compact.
        # chain_submit is called after every draft, so leaking per-window detail
        # into the default response is the difference between a chain that runs
        # for hundreds of steps and one that fills the context window.
        plain_long = await call("detect", {"text": long_doc})
        assert plain_long["ok"] and plain_long["windows"] == win["windows"]
        assert "window_detail" not in plain_long, "include_windows=False must omit detail"
        assert plain_long["score"] == win["score"], "the flag must not change the score"
        print("  detect(default): same score, no window_detail")

        # ----------------------------------------------------------- detect_batch
        batch = await call("detect_batch", {"texts": [AI, HUMAN, AI]})
        assert batch["ok"] and len(batch["results"]) == 3
        assert [r["index"] for r in batch["results"]] == [0, 1, 2]
        assert batch["best_index"] == 1, batch["best_index"]
        assert batch["best_score"] == min(r["score"] for r in batch["results"])
        assert abs(batch["results"][0]["score"] - batch["results"][2]["score"]) < 1e-6
        assert abs(batch["mean_score"]
                   - sum(r["score"] for r in batch["results"]) / 3) < 0.001
        for r in batch["results"]:
            assert r["label"] == names[r["bucket"]], (r["label"], r["bucket"])
            assert abs(r["score"]
                       - sum(i * p for i, p in enumerate(r["probs"])) / (nb - 1)) < 0.002
        print(f"  detect_batch: best_index={batch['best_index']}, duplicates agree, "
              f"labels match buckets")

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
            # `reliable` is a claim about this unit, not decoration: the docstring
            # says units under 75 words are indicative rather than precise. Only
            # asserting isinstance(bool) let "always reliable" pass.
            assert u["reliable"] == (u["assessment_word_count"] >= 75), (
                f"unit {u['unit']}: {u['words']} words but reliable={u['reliable']}")
            # The offsets must slice the caller's own text back out.
            assert (AI + " " + HUMAN)[u["start"]:u["end"]].split() == u["text"].split()
        print(f"  detect_spans: {spans['unit_count']} units, sorted, "
              f"{spans['unreliable_units']} unreliable, reliable flag matches word count")

        # unreliable_units counts EVERY unit, not just the ones reported back.
        all_units = await call("detect_spans",
                               {"text": AI + " " + HUMAN, "top": 50, "min_words": 1})
        assert all_units["unit_count"] == len(all_units["worst_units"])
        assert all_units["unreliable_units"] == sum(
            1 for u in all_units["worst_units"] if u["assessment_word_count"] < 75), (
            all_units["unreliable_units"],
            [u["words"] for u in all_units["worst_units"]])

        # Adaptive relaxation: two short sentences cannot reach a 25-word merge
        # threshold, so the splitter must back off rather than hand back one unit
        # covering everything -- and must report the threshold it settled on.
        tiny = "The cat sat down. The dog stood up."
        relaxed = await call("detect_spans", {"text": tiny, "min_words": 25})
        assert relaxed["ok"], relaxed
        assert relaxed["unit_count"] >= 2, (
            f"min_words=25 swallowed a short text whole: {relaxed['unit_count']} unit(s)")
        assert relaxed["granularity_relaxed"] is True
        assert relaxed["min_words_used"] < 25, relaxed["min_words_used"]
        print(f"  detect_spans(min_words=25 on a short text): relaxed to "
              f"{relaxed['min_words_used']}, {relaxed['unit_count']} units")

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
        # A fresh chain must say what to do first -- the instructions promise a
        # next_action, and "which segment do I draft?" is the immediate question.
        assert "segment='a'" in ch["next_action"], ch
        print(f"  chain_create: duplicate segment de-duplicated -> {ch['segments']}; "
              f"next_action names the first segment")

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

        # The completeness gate, without leaning on what the model returns.
        # target_score=1.0 makes score_met true for ANY score, so target_met can
        # only be false because a segment is missing. The previous check happened
        # to pass for the other reason too, which made it a weak guard.
        loose = await call("chain_create",
                           {"name": "loose", "target_score": 1.0, "segments": ["p", "q"]})
        lid = loose["chain_id"]
        await call("chain_submit", {"chain_id": lid, "text": HUMAN, "segment": "p",
                                    "span_feedback": False})
        la = await call("chain_assemble", {"chain_id": lid, "include_text": False})
        assert la["score_met"] is True, la["document_score"]
        assert la["complete"] is False and la["missing_segments"] == ["q"]
        assert la["target_met"] is False, (
            "a document missing a whole section reported target_met with "
            f"score_met={la['score_met']}")
        print("  chain_assemble: target_met stays False on an incomplete document "
              "even when the score passes")

        # A submit that FAILS must not leave its segment registered. Registering
        # first means a typo'd segment name (or an OOM) permanently marks the
        # chain incomplete with no way to remove the segment again.
        segs_before = [s["segment"] for s in
                       (await call("chain_status", {"chain_id": lid}))["segments"]]
        ghost = await call("chain_submit", {"chain_id": lid, "text": "   ",
                                            "segment": "typoo"})
        assert ghost["ok"] is False, ghost
        segs_after = [s["segment"] for s in
                      (await call("chain_status", {"chain_id": lid}))["segments"]]
        assert segs_after == segs_before, (
            f"a failed submit registered segment(s): {set(segs_after) - set(segs_before)}")
        asm_after = await call("chain_assemble", {"chain_id": lid, "include_text": False})
        assert "typoo" not in asm_after["missing_segments"], asm_after["missing_segments"]
        print(f"  failed chain_submit left segments untouched: {segs_after}")

        # Span feedback is the point of chain_submit's compact response: the
        # WORST few units, worst first, capped. Only asserting it is non-empty
        # left the ordering and the cap untested.
        many = " ".join(
            f"Sentence number {i} exists here and says something fairly plain about "
            f"the weather outside in an ordinary way." for i in range(12))
        spanned = await call("chain_submit", {"chain_id": lid, "text": many,
                                              "segment": "q"})
        assert spanned["ok"], spanned
        ws = spanned["worst_spans"]
        assert 0 < len(ws) <= 5, f"worst_spans should be capped at 5, got {len(ws)}"
        ss = [u["score"] for u in ws]
        assert ss == sorted(ss, reverse=True), f"worst_spans not worst-first: {ss}"
        for u in ws:
            assert many[u["start"]:u["end"]].split() == u["text"].split(), u
        print(f"  chain_submit worst_spans: {len(ws)} of many, worst-first, "
              f"offsets slice the submitted text")

        # --------------------------------------------- exact scoring arithmetic
        # target_score is documented as "stop when the score is at or below this",
        # and the deltas are the only thing a caller sees between drafts. A real
        # model never lands on a round number, so substitute known scores and
        # check the arithmetic and the boundary exactly, instead of asserting
        # loose inequalities that a sign flip or a `<` for `<=` slips through.
        def stub(score):
            v = Verdict(score=score, bucket=int(round(score * (nb - 1))),
                        label=names[int(round(score * (nb - 1)))],
                        probs=[0.25] * nb, word_count=9, char_count=40)
            return lambda *_a, **_k: (v, [])

        edge = await call("chain_create", {"name": "edge", "target_score": 0.25})
        eid = edge["chain_id"]

        async def submit_with(score, **extra):
            server.detector.__dict__["detect"] = stub(score)
            try:
                return await call("chain_submit", dict(
                    {"chain_id": eid, "text": HUMAN, "span_feedback": False}, **extra))
            finally:
                server.detector.__dict__.pop("detect", None)

        on_target = await submit_with(0.25)
        assert on_target["ok"] and on_target["score"] == 0.25, on_target
        assert on_target["target_met"] is True, (
            "a score exactly ON the target must count as met -- the field is "
            "documented as 'at or below'")
        assert on_target["is_new_best"] is True
        assert on_target["delta_vs_previous"] is None and on_target["delta_vs_best"] is None
        assert on_target["best_score"] == 0.25
        assert on_target["revision_state"] == "insufficient_length", on_target
        assert on_target["stop_recommended"] and not on_target["length_sufficient"], on_target

        worse = await submit_with(0.4, note="regressed")
        assert worse["step"] == 2 and worse["score"] == 0.4
        assert worse["target_met"] is False
        assert worse["is_new_best"] is False, "0.4 is not better than 0.25"
        assert worse["best_score"] == 0.25, worse["best_score"]
        assert worse["delta_vs_previous"] == 0.15, worse["delta_vs_previous"]
        assert worse["delta_vs_best"] == 0.15, worse["delta_vs_best"]

        better = await submit_with(0.1)
        assert better["is_new_best"] is True and better["best_score"] == 0.1
        assert better["delta_vs_previous"] == -0.3, better["delta_vs_previous"]
        assert better["delta_vs_best"] == -0.15, better["delta_vs_best"]

        server.detector.__dict__["detect"] = stub(0.25)
        try:
            asm_edge = await call("chain_assemble", {"chain_id": eid,
                                                     "include_text": False})
        finally:
            server.detector.__dict__.pop("detect", None)
        assert asm_edge["complete"] is True
        assert asm_edge["score_met"] is True and asm_edge["target_met"] is True, (
            "chain_assemble must treat a score exactly on the target as met")
        assert [p["score"] for p in asm_edge["per_segment"]] == [0.1]
        print("  scoring arithmetic: target boundary inclusive, deltas exact "
              "(+0.150 / -0.300 vs previous, -0.150 vs best)")

        await call("chain_submit", {"chain_id": cid, "text": HUMAN, "segment": "b"})
        full = await call("chain_assemble", {"chain_id": cid, "include_text": False})
        assert full["ok"] and full["complete"] is True and not full["missing_segments"]
        assert "text" not in full, "include_text=False must omit the document"
        assert [p["segment"] for p in full["per_segment"]] == ["a", "b"]
        assert full["target_met"] == (full["complete"] and full["score_met"])
        print(f"  chain_assemble(full): score={full['document_score']:.3f} complete=True")

        # ------------------------------------------- assemble vs edge whitespace
        # Drafts are stored verbatim, so a segment now arrives at chain_assemble
        # carrying its own leading indentation and trailing blank lines. Those
        # must not leak into the scored document: clean_text collapses an indent
        # run to one space instead of dropping it, and on the real checkpoint that
        # single character moved a 314-word document by +0.15 -- enough to flip a
        # target. The separator decides what sits between sections.
        wid = (await call("chain_create", {"name": "ws", "segments": ["p", "q"]}))["chain_id"]
        indented = "    for attempt in range(times):\n        return fn()\n\n\n"
        trailing = HUMAN + "   \n\n\n"
        await call("chain_submit", {"chain_id": wid, "text": indented, "segment": "p",
                                    "span_feedback": False})
        await call("chain_submit", {"chain_id": wid, "text": trailing, "segment": "q",
                                    "span_feedback": False})
        ws = await call("chain_assemble", {"chain_id": wid})
        assert ws["ok"], ws
        assert ws["text"] == "\n\n".join([indented.strip(), trailing.strip()]), (
            f"assembled document carries segment edge whitespace: {ws['text']!r}")
        # ...while the drafts themselves stay byte-for-byte what was submitted.
        for seg, sent in (("p", indented), ("q", trailing)):
            got = await call("chain_get_text", {"chain_id": wid, "segment": seg})
            assert got["text"] == sent, f"segment {seg} not stored verbatim: {got['text']!r}"
        await call("chain_delete", {"chain_id": wid})
        print("  chain_assemble: segment edge whitespace stripped from the document, "
              "drafts still stored verbatim")

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
        # best_score is a cross-segment MIN and reads like a completion signal;
        # these two are the honest ones, counted against the DECLARED list so an
        # unstarted segment counts as not-at-target instead of dropping out.
        assert row["segments_total"] == 2, row
        assert 0 <= row["segments_at_target"] <= 2, row

        # -------------------------------------------- get_text guidance/validation
        # A typo'd segment must be named as such, with the valid names -- not
        # reported as a missing step, which reads as "your draft is gone" at the
        # exact moment the regression-recovery flow sent the caller here.
        typo = await call("chain_get_text", {"chain_id": cid, "segment": "aa"})
        assert typo["ok"] is False and "no such segment 'aa'" in typo["error"], typo
        assert "'a'" in typo["error"], f"valid names missing from: {typo['error']}"
        # Recovery is a two-step procedure; the response that hands back the
        # draft must name step two, segment included.
        assert f"branch_from={best['step']}" in best["next_action"], best
        assert "segment='a'" in best["next_action"], best

        # ------------------------------------------------------ segment delete
        # The escape hatch for a typo'd segment name, which otherwise leaves
        # chain_assemble reporting the chain incomplete forever.
        wid = (await call("chain_create", {"name": "typo", "segments": ["real", "raal"]}))["chain_id"]
        await call("chain_submit", {"chain_id": wid, "text": HUMAN, "segment": "raal",
                                    "span_feedback": False})
        gone = await call("chain_delete", {"chain_id": wid, "segment": "raal"})
        assert gone["ok"] and gone["deleted_steps"] == 1, gone
        assert gone["remaining_segments"] == ["real"], gone
        st_after = await call("chain_status", {"chain_id": wid})
        assert [s["segment"] for s in st_after["segments"]] == ["real"], st_after
        bad_seg_del = await call("chain_delete", {"chain_id": wid, "segment": "ghost"})
        assert bad_seg_del["ok"] is False and "no such segment" in bad_seg_del["error"]
        last = await call("chain_delete", {"chain_id": wid, "segment": "real"})
        assert last["ok"] is False and "only segment" in last["error"], last
        await call("chain_delete", {"chain_id": wid})
        print("  chain_delete(segment=...): removes drafts and the declared entry "
              "together; refuses ghosts and the last segment")

        # ------------------------------------------------------- info db fields
        inf2 = await call("detector_info")
        assert Path(inf2["db_path"]).is_absolute(), inf2["db_path"]
        assert inf2["db_path_configured"] == os.environ["EDITLENS_DB"], inf2
        assert "db_error" not in inf2, "healthy store must not report db_error"

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

        # ------------------------------------------------- tools never raise
        # Those four exercise each tool's OWN try/except. The _guard decorator
        # exists for everything else -- a CUDA OOM arrives as a bare RuntimeError
        # and a broken Windows torch install as an OSError, neither of which the
        # inner handlers catch. Nothing here provoked such an exception, so
        # deleting _guard outright passed every suite. Provoke one per tool.
        class Simulated(RuntimeError):
            pass

        def boom(*_a, **_k):
            raise Simulated("simulated CUDA out of memory")

        probes = [
            ("detector_info", server.detector, "info", {}),
            ("detector_unload", server.detector, "unload", {}),
            ("detect", server.detector, "detect", {"text": HUMAN}),
            ("detect_batch", server.detector, "detect_many", {"texts": [HUMAN]}),
            ("detect_spans", server.detector, "detect_many", {"text": AI + " " + HUMAN}),
            ("chain_create", server.store, "create", {"name": "x"}),
            ("chain_submit", server.store, "get", {"chain_id": cid, "text": HUMAN}),
            ("chain_status", server.store, "get", {"chain_id": cid}),
            ("chain_history", server.store, "segments_of", {"chain_id": cid}),
            ("chain_get_text", server.store, "get", {"chain_id": cid}),
            ("chain_assemble", server.store, "get", {"chain_id": cid}),
            ("chain_list", server.store, "list_chains", {}),
            ("chain_delete", server.store, "delete", {"chain_id": cid}),
        ]
        for tool, owner, attr, args in probes:
            had = attr in owner.__dict__
            original = owner.__dict__.get(attr)
            setattr(owner, attr, boom)
            try:
                # Must not raise out of the client: a raised tool becomes an
                # opaque ToolError and the diagnosable message is lost.
                r = await call(tool, args)
            finally:
                if had:
                    setattr(owner, attr, original)
                else:
                    owner.__dict__.pop(attr, None)
            assert isinstance(r, dict), f"{tool} returned {type(r).__name__}, not a dict"
            assert r.get("ok") is False, f"{tool} hid an unexpected failure: {r}"
            assert "simulated" in str(r.get("error", "")).lower(), (tool, r)
            assert r.get("error_type") == "Simulated", (tool, r)
            if tool == "detector_info":
                # The diagnostic tool must still say where the data lives when
                # the detector (in shared mode: the worker) is unreachable.
                assert r.get("db_path") and "score_note" in r, r
        # And the server is still healthy afterwards.
        assert (await call("chain_status", {"chain_id": cid}))["ok"] is True
        print(f"  {len(probes)} tools turned an unexpected RuntimeError into an "
              "ok=False dict instead of raising")

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
