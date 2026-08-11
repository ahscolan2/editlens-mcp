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


def _verdict(score: float) -> Verdict:
    nb = len(BUCKETS)
    idx = int(round(score * (nb - 1)))
    return Verdict(score=score, bucket=idx, label=BUCKETS[idx],
                   probs=[1.0 / nb] * nb, word_count=30, char_count=160)


class _Stub:
    """Force the document score, and optionally the per-span scores."""

    def __init__(self, doc: float, spans: list[float] | None = None):
        self.doc, self.spans = doc, spans

    def __enter__(self):
        server.detector.__dict__["detect"] = lambda *_a, **_k: (_verdict(self.doc), [])
        if self.spans is not None:
            it = list(self.spans)
            server.detector.__dict__["detect_many"] = (
                lambda texts, **_k: [_verdict(it[i % len(it)]) for i in range(len(texts))]
            )
        return self

    def __exit__(self, *_exc):
        server.detector.__dict__.pop("detect", None)
        server.detector.__dict__.pop("detect_many", None)


async def main() -> None:
    async with Client(server.mcp) as c:

        async def call(name, args=None):
            return (await c.call_tool(name, args or {})).data

        # ------------------------------------------------------------------
        # 1. Spans that are already at or below target must not be handed back
        #    as work. Observed live: three spans came back at 0.20 / 0.08 /
        #    0.06 against a target of 0.25, under "Rewrite the worst spans".
        #    Two of them were labelled Human-written.
        # ------------------------------------------------------------------
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
              f"{len(r['worst_spans'])} flagged for rewrite")

        # ------------------------------------------------------------------
        # 2. The dead end: document above target, but NO span above it. Telling
        #    the caller to rewrite the worst spans here is an instruction to
        #    rewrite text this same response calls acceptable, and it never
        #    terminates. Observed live at document 0.2992 with spans 0.2011 /
        #    0.0769 / 0.0646 against target 0.25.
        # ------------------------------------------------------------------
        with _Stub(0.2992, spans=[0.2011, 0.0646]):
            dead = await call("chain_submit", {"chain_id": cid, "text": TEXT})
        assert dead["target_met"] is False and dead["spans_above_target"] == 0, dead
        na = dead["next_action"].lower()
        assert "bottomed out" in na, dead["next_action"]
        assert "do not rewrite the spans" in na, dead["next_action"]
        # It must also explain WHY the parts look fine while the whole does not.
        assert "not comparable across lengths" in na, dead["next_action"]
        print("  all-spans-under-target: told to stop span-hunting, not to loop")

        # ------------------------------------------------------------------
        # 3. A regression must point at the recovery route. History is
        #    append-only and branch_from exists, but neither it nor
        #    chain_get_text appeared anywhere in the response, so the only
        #    advice after a 0.035 -> 0.999 step was "keep editing this draft".
        # ------------------------------------------------------------------
        rid = (await call("chain_create", {"name": "regress", "target_score": 0.25}))["chain_id"]
        with _Stub(0.04, spans=[0.04, 0.04]):
            good = await call("chain_submit", {"chain_id": rid, "text": TEXT})
        assert good["target_met"] is True and good["best_step"] == good["step"]
        with _Stub(0.99, spans=[0.99, 0.99]):
            bad = await call("chain_submit", {"chain_id": rid, "text": TEXT})
        assert bad["is_new_best"] is False and bad["best_step"] == good["step"], bad
        # best_step is the field that makes the advice actionable: branch_from
        # takes a step number, and best_score alone does not supply one.
        na = bad["next_action"]
        assert f"chain_get_text(step={good['step']})" in na, na
        assert f"branch_from={good['step']}" in na, na
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

        # ------------------------------------------------------------------
        # 4. chain_status told nobody what to do. `pending` also conflated a
        #    segment with no drafts at all with one that has drafts scoring too
        #    high -- opposite situations under one name.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 5. chain_assemble and chain_status can disagree, and both are right.
        #    Observed live: sections at 0.733 and 0.5054 assembled into a 0.1077
        #    document. chain_status said pending; assemble said target_met. A
        #    caller given both with no reconciliation revises finished work.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 6. A segment whose latest draft is worse than an earlier one still
        #    counts as target_met and still assembles from the earlier draft.
        #    That is correct and completely invisible.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 7. detect_batch ranks; it does not judge. Observed live: candidates at
        #    0.50 / 0.99 / 0.66, and the winner spliced in took its document to
        #    0.06. best_score compared against a target rejects all three.
        # ------------------------------------------------------------------
        with _Stub(0.5, spans=[0.5, 0.99, 0.66]):
            b = await call("detect_batch", {"texts": ["one", "two", "three"]})
        assert b["ok"] and "comparison_note" in b, b
        assert "rank" in b["comparison_note"].lower(), b["comparison_note"]
        assert "detect" in b["comparison_note"], b["comparison_note"]
        print("  detect_batch carries comparison_note: rank, then re-score after splicing")

        # ------------------------------------------------------------------
        # 8. The server instructions are the only guidance a model gets before
        #    it reads any tool. Both counter-intuitive facts have to be there.
        # ------------------------------------------------------------------
        instr = server.mcp.instructions
        assert "next_action" in instr, instr
        assert "branch_from" in instr and "chain_get_text" in instr, instr
        assert "not comparable across lengths" in instr, instr
        print("  server instructions name next_action, the scale caveat, and recovery")

    print("USABILITY TESTS PASSED")


asyncio.run(main())
