"""Guidance, not plumbing: does a model driving these tools get told the right
thing to do next?

Every case here was hit while actually using the server for its stated purpose
(a write -> score -> revise loop), and each one previously produced advice that
was wrong, absent, or self-contradictory. The detector is stubbed with exact
scores throughout, because the point is the guidance derived from a score, not
the score -- and a real model never lands on the numbers that make these cases
reproducible.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["EDITLENS_DB"] = str(Path(tempfile.mkdtemp()) / "usability.db")

from fastmcp import Client  # noqa: E402

from editlens_mcp import server  # noqa: E402
from editlens_mcp.detector import Verdict  # noqa: E402

TEXT = ("I burnt the rice again. Third time this month. My flatmate calls it a "
        "tradition, which is generous of her considering she has to eat it. We had "
        "toast instead, and nobody complained about that either.")


# Fixed bucket names rather than the detector's own: stubbing both entry points
# means the model never loads, and this suite must not be the one that pulls
# weights onto the GPU just to name a label.
BUCKETS = ["Human-written", "Lightly AI-edited", "Heavily AI-edited", "Fully AI-generated"]


def _verdict(score: float, words: int = 100) -> Verdict:
    nb = len(BUCKETS)
    idx = int(round(score * (nb - 1)))
    return Verdict(score=score, bucket=idx, label=BUCKETS[idx],
                   probs=[1.0 / nb] * nb, word_count=words, char_count=160)


class _Stub:
    """Force the document score, and optionally the per-span scores.

    `words` sets the word count of the DOCUMENT verdict only; span verdicts keep
    the default. Length drives the short-text guidance, so a case about that
    guidance has to be able to set it without also changing every span.
    """

    def __init__(self, doc: float, spans: list[float] | None = None, words: int = 100):
        self.doc, self.spans, self.words = doc, spans, words

    def __enter__(self):
        server.detector.__dict__["detect"] = (
            lambda *_a, **_k: (_verdict(self.doc, self.words), [])
        )
        if self.spans is not None:
            it = list(self.spans)
            server.detector.__dict__["detect_many"] = (
                lambda texts, **_k: [_verdict(it[i % len(it)]) for i in range(len(texts))]
            )
        return self

    def __exit__(self, *_exc):
        server.detector.__dict__.pop("detect", None)
        server.detector.__dict__.pop("detect_many", None)


# Twenty-four sentences of ~17 words. At the default 25-word merge threshold
# they pair up into 12 units -- more than any default span window can show --
# and at min_words=1 they stay as 24, so the same fixture exercises both the
# truncation cases and the granularity knob.
TWELVE_UNITS = " ".join(
    f"Sentence number {i} carries on for a little while and clears about fifteen "
    f"words entirely on its own here." for i in range(24)
)


async def main() -> None:
    async with Client(server.mcp) as c:

        revision = 0

        async def call(name, args=None):
            nonlocal revision
            # Forced different scores represent different drafts. Keep them
            # distinct so unchanged-draft stopping is tested separately below.
            args = dict(args or {})
            if name == "chain_submit":
                revision += 1
                args["text"] = f"v{revision:04d} " + args["text"]
            return (await c.call_tool(name, args)).data

        cid = (await call("chain_create", {"name": "spans", "target_score": 0.25}))["chain_id"]
        with _Stub(0.9, spans=[0.9, 0.1]):
            r = await call("chain_submit", {"chain_id": cid, "text": TEXT})
        assert r["ok"], r
        flags = {s["score"]: s["above_target"] for s in r["worst_spans"]}
        assert flags == {0.9: True, 0.1: False}, flags
        assert r["spans_above_target"] == sum(
            1 for s in r["worst_spans"] if s["score"] > 0.25), r
        assert 0 < r["spans_above_target"] < len(r["worst_spans"]), (
            "this fixture must produce a mix, or it proves nothing", r["worst_spans"])
        assert "above_target" in r["next_action"], r["next_action"]
        print(f"  spans carry above_target; {r['spans_above_target']} of "
              f"{len(r['worst_spans'])} above the numerical target")

        with _Stub(0.2992, spans=[0.2011, 0.0646]):
            dead = await call("chain_submit", {"chain_id": cid, "text": TEXT})
        assert dead["target_met"] is False and dead["spans_above_target"] == 0, dead
        na = dead["next_action"].lower()
        assert "bottomed out" in na, dead["next_action"]
        assert "do not rewrite the spans" in na, dead["next_action"]
        # It must also explain WHY the parts look fine while the whole does not.
        assert "not comparable across lengths" in dead["analysis_note"], dead
        print("  all-spans-under-target: told to stop span-hunting, not to loop")

        rid = (await call("chain_create", {"name": "regress", "target_score": 0.25}))["chain_id"]
        with _Stub(0.04, spans=[0.04, 0.04]):
            good = await call("chain_submit", {"chain_id": rid, "text": TEXT})
        assert good["target_met"] is True and good["best_step"] == good["step"]
        with _Stub(0.99, spans=[0.99, 0.99]):
            bad = await call("chain_submit", {"chain_id": rid, "text": TEXT})
        assert bad["is_new_best"] is False and bad["best_step"] == good["step"], bad
        # best_step is the field that makes the advice actionable: branch_from
        # takes a step number, and best_score alone does not supply one. The
        # segment must be named in BOTH calls: they default to 'main' and step
        # numbers are per-segment, so the unqualified instruction reads -- and
        # writes -- another segment in any multi-segment chain.
        na = bad["next_action"]
        assert f"chain_get_text(segment='main', step={good['step']})" in na, na
        assert f"segment='main', branch_from={good['step']}" in na, na
        assert "do not keep editing this draft" in na.lower(), na
        print(f"  regression 0.04 -> 0.99: names step {good['step']}, "
              f"chain_get_text and branch_from")

        # The recovery route it advertises has to actually work.
        recovered = await call("chain_get_text", {"chain_id": rid, "step": "best"})
        assert recovered["step"] == good["step"], recovered
        with _Stub(0.03, spans=[0.03, 0.03]):
            forked = await call("chain_submit", {"chain_id": rid, "text": TEXT,
                                                 "branch_from": good["step"]})
        assert forked["parent_step"] == good["step"], forked
        print(f"  advertised recovery works: step {forked['step']} "
              f"branched from {forked['parent_step']}")

        # Noise must NOT trigger it -- a draft a hair worse than the best is
        # not a reason to abandon the line of work.
        with _Stub(0.05, spans=[0.05, 0.05]):
            noise = await call("chain_submit", {"chain_id": rid, "text": TEXT})
        assert noise["is_new_best"] is False
        assert "branch_from" not in noise["next_action"], noise["next_action"]
        assert "target met" in noise["next_action"].lower(), noise["next_action"]
        print("  a 0.02 wobble under target does not trigger recovery advice")

        mid = (await call("chain_create", {
            "name": "multi", "target_score": 0.25,
            "segments": ["intro", "body", "end"]}))["chain_id"]

        empty = await call("chain_status", {"chain_id": mid})
        assert empty["unstarted"] == ["intro", "body", "end"], empty
        assert empty["above_target"] == [], empty
        assert "intro" in empty["next_action"] and "chain_submit" in empty["next_action"]
        print(f"  chain_status(empty): next_action -> {empty['next_action'][:52]}...")

        with _Stub(0.05, spans=[0.05, 0.05]):
            await call("chain_submit", {"chain_id": mid, "text": TEXT, "segment": "intro"})
        with _Stub(0.73, spans=[0.73, 0.73]):
            await call("chain_submit", {"chain_id": mid, "text": TEXT, "segment": "body"})

        part = await call("chain_status", {"chain_id": mid})
        assert part["unstarted"] == ["end"], part
        assert part["above_target"] == ["body"], part
        assert part["pending"] == ["body", "end"], "pending stays as it was, additive only"
        assert "end" in part["next_action"], part["next_action"]
        print(f"  chain_status: unstarted={part['unstarted']} "
              f"above_target={part['above_target']} (pending merged both)")

        with _Stub(0.05, spans=[0.05, 0.05]):
            await call("chain_submit", {"chain_id": mid, "text": TEXT, "segment": "end"})

        # Now nothing is unstarted and one segment is above target. The advice
        # must be to ASSEMBLE, not to revise: see case 5 for why.
        drafted = await call("chain_status", {"chain_id": mid})
        assert drafted["unstarted"] == [] and drafted["above_target"] == ["body"]
        assert "chain_assemble" in drafted["next_action"], drafted["next_action"]
        assert drafted["next_action"].index("chain_assemble") < \
            drafted["next_action"].index("chain_submit"), (
            "assembling must be offered before revising", drafted["next_action"])
        print("  chain_status with a failing section: assemble first, then revise")

        with _Stub(0.1077):
            asm = await call("chain_assemble", {"chain_id": mid, "include_text": False})
        assert asm["target_met"] is True and asm["complete"] is True, asm
        assert asm["segments_above_target"] == ["body"], asm
        na = asm["next_action"]
        assert "ignore that" in na.lower() and "chain_status" in na, na
        assert "body" in na, na
        print(f"  chain_assemble(0.1077) with section 0.73: reconciles with "
              f"chain_status, segments_above_target={asm['segments_above_target']}")

        # The reverse: document above target. Section scores are a poor guide,
        # so point at detect_spans on the assembled text.
        with _Stub(0.8):
            hi = await call("chain_assemble", {"chain_id": mid, "include_text": False})
        assert hi["target_met"] is False
        assert "detect_spans" in hi["next_action"], hi["next_action"]
        print("  chain_assemble above target: points at detect_spans on the document")

        # An incomplete document must ask for the missing sections first,
        # whatever the score of the part that exists.
        pid = (await call("chain_create", {
            "name": "partial", "target_score": 0.25, "segments": ["a", "b"]}))["chain_id"]
        with _Stub(0.05, spans=[0.05, 0.05]):
            await call("chain_submit", {"chain_id": pid, "text": TEXT, "segment": "a"})
        with _Stub(0.05):
            gap = await call("chain_assemble", {"chain_id": pid, "include_text": False})
        assert gap["complete"] is False and gap["missing_segments"] == ["b"]
        assert "b" in gap["next_action"] and "chain_submit" in gap["next_action"]
        assert "incomplete" in gap["next_action"].lower(), gap["next_action"]
        print("  chain_assemble(incomplete): asks for the missing section, not for a rewrite")

        sid = (await call("chain_create", {"name": "stale", "target_score": 0.25}))["chain_id"]
        with _Stub(0.05, spans=[0.05, 0.05]):
            await call("chain_submit", {"chain_id": sid, "text": TEXT})
        with _Stub(0.95, spans=[0.95, 0.95]):
            await call("chain_submit", {"chain_id": sid, "text": TEXT})
        st = await call("chain_status", {"chain_id": sid})
        seg = st["segments"][0]
        assert seg["target_met"] is True and seg["latest_is_best"] is False, seg
        assert st["segments_with_better_earlier_draft"] == ["main"], st
        assert "chain_get_text" in st["next_action"], st["next_action"]
        print("  chain_status flags a segment whose latest draft is not the one "
              "assembly will use")

        with _Stub(0.5, spans=[0.5, 0.99, 0.66]):
            b = await call("detect_batch", {"texts": ["one", "two", "three"]})
        assert b["ok"] and "comparison_note" in b, b
        assert "rank" in b["comparison_note"].lower(), b["comparison_note"]
        assert "detect" in b["comparison_note"], b["comparison_note"]
        print("  detect_batch carries comparison_note: rank, then re-score after splicing")

        instr = server.mcp.instructions
        assert "next_action" in instr, instr
        assert "branch_from" in instr and "chain_get_text" in instr, instr
        assert "not comparable across lengths" in instr, instr
        print("  server instructions name next_action, the scale caveat, and recovery")

        tid = (await call("chain_create", {"name": "trunc", "target_score": 0.25}))["chain_id"]
        with _Stub(0.9, spans=[0.9]):
            t = await call("chain_submit", {"chain_id": tid, "text": TWELVE_UNITS})
        assert t["span_unit_count"] == 12, t["span_unit_count"]
        assert t["spans_above_target_total"] == 12, t
        assert t["spans_above_target"] == len(t["worst_spans"]) == 5, t
        assert t["spans_truncated"] is True, t
        na = t["next_action"]
        # The exact false claim. It must not survive anywhere in the response.
        assert "already at or below target" not in na, na
        # Showing more fragments must never turn a score into a mandate to
        # rewrite the whole document.
        assert "Preserve sound content" in na, na
        assert "rewrite the whole passage" not in na, na
        print("  all spans high: preserve sound content; display limit never mandates a rewrite")

        # A window that happens to hide only SOME failing units is the same bug.
        with _Stub(0.9, spans=[0.9, 0.1]):
            half = await call("chain_submit", {"chain_id": tid, "text": TWELVE_UNITS})
        assert half["spans_above_target_total"] == 6 and half["spans_above_target"] == 5, half
        assert half["spans_truncated"] is True, half
        assert "already at or below target" not in half["next_action"], half["next_action"]

        # ...and when nothing is hidden, the count of units to leave alone must
        # come from the whole draft, not from the reported slice.
        with _Stub(0.9, spans=[0.9, 0.1]):
            full = await call("chain_submit", {"chain_id": tid, "text": TWELVE_UNITS,
                                               "span_top": 12})
        assert full["spans_truncated"] is False, full
        assert full["spans_above_target"] == full["spans_above_target_total"] == 6, full
        assert full["revision_state"] == half["revision_state"] == "review_draft", full
        print("  changing the display window preserves the same review advice")

        with _Stub(0.9, spans=[0.9]):
            allbad = await call("chain_submit", {"chain_id": tid, "text": TWELVE_UNITS,
                                                 "span_top": 12})
        assert allbad["spans_above_target"] == allbad["span_unit_count"] == 12, allbad
        assert allbad["spans_truncated"] is False, allbad
        na = allbad["next_action"]
        assert "Preserve sound content" in na and "nothing here to preserve" not in na, na
        assert "already at or below target" not in na and "Leave the other" not in na, na
        print("  all 12 units high: still told to preserve sound content")

        with _Stub(0.9, spans=[0.9]):
            coarse = await call("chain_submit", {"chain_id": tid, "text": TWELVE_UNITS})
            fine = await call("chain_submit", {"chain_id": tid, "text": TWELVE_UNITS,
                                               "span_min_words": 1, "span_top": 50})
        assert coarse["span_min_words_used"] == 25, coarse
        assert fine["span_min_words_used"] == 1, fine
        assert fine["span_unit_count"] > coarse["span_unit_count"], (
            "span_min_words did not reach the splitter",
            fine["span_unit_count"], coarse["span_unit_count"])
        assert len(fine["worst_spans"]) > len(coarse["worst_spans"]), (
            "span_top did not widen the reported window",
            len(fine["worst_spans"]), len(coarse["worst_spans"]))
        assert max(s["words"] for s in fine["worst_spans"]) <= \
            max(s["words"] for s in coarse["worst_spans"]), (
            "finer spans must not be larger -- that is the whole point")
        print(f"  span_min_words 25 -> {coarse['span_unit_count']} units, "
              f"1 -> {fine['span_unit_count']} units; span_top widened "
              f"{len(coarse['worst_spans'])} -> {len(fine['worst_spans'])}")

        wid = (await call("chain_create", {"name": "wee", "target_score": 0.25}))["chain_id"]
        with _Stub(0.46, spans=[0.9, 0.1], words=26):
            tiny = await call("chain_submit", {"chain_id": wid, "text": TWELVE_UNITS})
        assert tiny["reliable"] is False and not tiny["length_sufficient"], tiny
        assert "reliability_note" in tiny and "26 words" in tiny["reliability_note"], tiny
        na = tiny["next_action"]
        assert na.startswith("CAUTION:"), na
        assert "chain_assemble" in na, na
        # The caution leads; the ordinary advice still follows it.
        assert "Do not revise to chase this score" in na, na
        print(f"  26-word draft over target: next_action leads with the caution, "
              f"not with 'rewrite it'")

        # Below the per-unit floor the wording is stronger still.
        with _Stub(0.46, spans=[0.9, 0.1], words=12):
            tinier = await call("chain_submit", {"chain_id": wid, "text": TWELVE_UNITS})
        assert tinier["reliable"] is False, tinier
        assert "not a measurement" in tinier["reliability_note"], tinier["reliability_note"]

        # A long draft must NOT be nagged, or the caution stops carrying weight.
        with _Stub(0.46, spans=[0.9, 0.1], words=600):
            big = await call("chain_submit", {"chain_id": wid, "text": TWELVE_UNITS})
        assert big["reliable"] is True and "reliability_note" not in big, big
        assert not big["next_action"].startswith("CAUTION:"), big["next_action"]

        # A low score on short text also needs the length caveat.
        with _Stub(0.05, spans=[0.05], words=26):
            ok_small = await call("chain_submit", {"chain_id": wid, "text": TWELVE_UNITS})
        assert ok_small["next_action"].startswith("CAUTION:"), ok_small["next_action"]
        print("  short drafts receive the length caveat on either side of target")

        # `detect` reported a bare 4-decimal score for any length at all.
        with _Stub(0.46, words=26):
            d = await call("detect", {"text": TWELVE_UNITS})
        assert d["reliable"] is False and "26 words" in d["reliability_note"], d
        with _Stub(0.46, words=600):
            d2 = await call("detect", {"text": TWELVE_UNITS})
        assert d2["reliable"] is True and "reliability_note" not in d2, d2
        print("  detect carries the same length caveat detect_spans always had")

        lid = (await call("chain_create", {"name": "long", "target_score": 0.01}))["chain_id"]
        with _Stub(0.30, spans=[0.30]):
            await call("chain_submit", {"chain_id": lid, "text": TEXT, "note": "the good one"})
        with _Stub(0.80, spans=[0.80]):
            for i in range(44):
                await call("chain_submit", {"chain_id": lid, "text": TEXT, "note": f"pass {i}"})

        window = await call("chain_history", {"chain_id": lid, "limit": 10})
        assert window["total_steps"] == 45 and window["returned"] == 10, window
        assert window["truncated"] is True, window
        assert window["best_step"] == 1 and window["best_score"] == 0.3, window
        assert all(r["step"] > 1 for r in window["trajectory"]), "step 1 is outside this window"
        # The best score in the window is worse than the segment's actual best:
        # exactly the trap, so the response has to name the way out.
        assert min(r["score"] for r in window["trajectory"]) > window["best_score"], window
        assert "note" in window and "branch_from=1" in window["note"], window
        assert "chain_get_text" in window["note"], window["note"]
        print(f"  chain_history(limit=10) of 45: truncated=True, best_step=1 named "
              f"even though the window starts at {window['trajectory'][0]['step']}")

        whole = await call("chain_history", {"chain_id": lid, "limit": 200})
        assert whole["returned"] == whole["total_steps"] == 45, whole
        assert whole["truncated"] is False and "note" not in whole, whole
        print("  a window that contains the best step carries no such note")

        # An untouched segment must not claim a best step it does not have.
        eid = (await call("chain_create", {
            "name": "empty", "target_score": 0.25, "segments": ["solo"]}))["chain_id"]
        blank = await call("chain_history", {"chain_id": eid, "segment": "solo"})
        assert blank["total_steps"] == 0 and blank["truncated"] is False, blank
        assert blank["best_step"] is None and "note" not in blank, blank
        print("  an empty segment reports total_steps=0 and no best step")

        solo = (await call("chain_create", {"name": "solo", "target_score": 0.25}))["chain_id"]
        with _Stub(0.9, spans=[0.9, 0.9], words=80):
            await call("chain_submit", {"chain_id": solo, "text": TEXT})
        s1 = await call("chain_status", {"chain_id": solo})
        assert "assembling adds nothing" in s1["next_action"], s1["next_action"]
        assert "chain_assemble first" not in s1["next_action"], s1["next_action"]

        multi = (await call("chain_create",
                            {"name": "multi", "segments": ["intro", "body"]}))["chain_id"]
        with _Stub(0.9, spans=[0.9, 0.9], words=80):
            m = await call("chain_submit", {"chain_id": multi, "text": TEXT,
                                            "segment": "intro"})
        assert "chain_assemble" in m["next_action"], m["next_action"]
        s2 = await call("chain_status", {"chain_id": multi})
        assert "chain_submit with segment='body'" in s2["next_action"], s2["next_action"]
        print("  single-segment status skips the useless assemble; multi-segment "
              "submit names chain_assemble")

        shorty = (await call("chain_create", {"name": "short", "target_score": 0.25}))["chain_id"]
        with _Stub(0.11, spans=[0.11], words=12):
            sh = await call("chain_submit", {"chain_id": shorty, "text": "tiny draft"})
        assert sh["target_met"] is True and sh["reliable"] is False
        na = sh["next_action"]
        assert na.startswith("CAUTION:"), na
        assert sh["stop_recommended"] and sh["revision_state"] == "insufficient_length", sh
        print("  target met on 12 words: next_action withholds the bare 'Stop'")

        # chain_assemble carries the same short-text caveat on its "Done."
        with _Stub(0.11, spans=[0.11], words=12):
            asm_short = await call("chain_assemble", {"chain_id": shorty,
                                                      "include_text": False})
        assert asm_short["reliable"] is False and "reliability_note" in asm_short
        assert "provisional" in asm_short["next_action"], asm_short["next_action"]
        # ...and when it cannot hand back the text, it says how to get it.
        with _Stub(0.9, spans=[0.9], words=80):
            asm_fail = await call("chain_assemble", {"chain_id": shorty,
                                                     "include_text": False})
        assert "include_text=true" in asm_fail["next_action"], asm_fail["next_action"]
        print("  assemble: short-text caveat on Done; include_text=False never "
              "points at text the response does not carry")

    print("USABILITY TESTS PASSED")


asyncio.run(main())
