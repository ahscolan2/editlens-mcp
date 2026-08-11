"""Two design limitations, and the guards against them.

1. History was linear: a revision that made things worse could not be forked
   from an earlier draft in a way the chain recorded. `branch_from` fixes that.
2. Span offsets go stale the moment the text is edited. `source_fingerprint`
   lets a caller detect that instead of splicing at coordinates that moved.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["EDITLENS_DB"] = str(Path(tempfile.mkdtemp()) / "branch.db")

from fastmcp import Client  # noqa: E402

from editlens_mcp import server  # noqa: E402
from editlens_mcp.detector import clean_text, text_fingerprint  # noqa: E402

GOOD = ("I burnt the rice again. Third time this month. My flatmate calls it a tradition, "
        "which is generous of her considering she has to eat it. We had toast instead.")
BAD = ("In today's rapidly evolving landscape, stakeholders must leverage synergies to "
       "drive transformative outcomes across the organizational ecosystem.")
MESSY = "Alpha beta gamma delta.\r\n\r\nEpsilon zeta eta theta.  Iota kappa lambda mu.\r\n"


async def main() -> None:
    async with Client(server.mcp) as c:
        async def call(n, a=None):
            return (await c.call_tool(n, a or {})).data

        # ------------------------------------------------------------ branching
        cid = (await call("chain_create", {"name": "b", "target_score": 0.1}))["chain_id"]

        s1 = await call("chain_submit", {"chain_id": cid, "text": GOOD,
                                         "span_feedback": False})
        assert s1["parent_step"] is None, s1["parent_step"]

        s2 = await call("chain_submit", {"chain_id": cid, "text": BAD,
                                         "span_feedback": False, "note": "went wrong"})
        assert s2["step"] == 2 and s2["parent_step"] == 1
        assert s2["score"] > s1["score"], "step 2 should be worse for this test"

        # Fork from step 1 rather than continuing from the bad step 2.
        s3 = await call("chain_submit", {"chain_id": cid, "text": GOOD + " Truly.",
                                         "span_feedback": False, "branch_from": 1})
        assert s3["step"] == 3 and s3["parent_step"] == 1, s3["parent_step"]
        print(f"  branch: step1(parent=None) -> step2(parent=1) -> step3(parent=1)")

        hist = await call("chain_history", {"chain_id": cid})
        parents = [(h["step"], h["parent_step"]) for h in hist["trajectory"]]
        assert parents == [(1, None), (2, 1), (3, 1)], parents
        print(f"  chain_history exposes the tree: {parents}")

        bad_branch = await call("chain_submit", {"chain_id": cid, "text": GOOD,
                                                 "branch_from": 99})
        assert bad_branch["ok"] is False and "no such step" in bad_branch["error"]
        # A rejected branch must not have consumed a step number.
        assert (await call("chain_status", {"chain_id": cid}))["total_steps"] == 3
        print("  invalid branch_from rejected without consuming a step")

        # The earlier draft is still intact and still wins.
        best = await call("chain_get_text", {"chain_id": cid, "step": "best"})
        assert best["step"] in (1, 3), best["step"]

        # "best" means the LOWEST score, whatever the model happened to return.
        # `best["step"] in (1, 3)` only encodes what these particular texts score;
        # this encodes the feature. Reversing best_step's ORDER BY is caught here
        # without depending on the detector agreeing with the test author.
        scores = {h["step"]: h["score"] for h in hist["trajectory"]}
        lowest = min(scores.values())
        assert best["score"] == lowest, (best["score"], scores)
        assert scores[best["step"]] == lowest, (best["step"], scores)
        assert best["step"] == min(s for s, v in scores.items() if v == lowest), (
            "ties must resolve to the earliest step")
        # "latest" is a different question and must not collapse into "best".
        latest = await call("chain_get_text", {"chain_id": cid, "step": "latest"})
        assert latest["step"] == max(scores), (latest["step"], sorted(scores))
        print(f"  best draft survives a bad revision: step {best['step']} "
              f"(score {best['score']} = min of {sorted(scores.values())}), "
              f"latest is step {latest['step']}")

        # --------------------------------------------------- offset staleness
        spans = await call("detect_spans", {"text": MESSY, "min_words": 1})
        fp = spans["source_fingerprint"]
        assert fp == text_fingerprint(MESSY)
        # A help string, not a guarantee -- the behaviour it describes is asserted
        # further down, where the fingerprint is shown to actually track the text.
        assert "stale" in spans["offsets_note"].lower()

        # Offsets must slice the ORIGINAL text, CRLF included.
        for u in spans["worst_units"]:
            assert MESSY[u["start"]:u["end"]].split() == u["text"].split(), (
                f"[{u['start']}:{u['end']}]={MESSY[u['start']:u['end']]!r} "
                f"vs {u['text']!r}")
        print(f"  detect_spans: fingerprint={fp}, {len(spans['worst_units'])} spans align")

        # After an edit the fingerprint changes, so stale offsets are detectable.
        # "sha256 of two different strings differs" is arithmetic, not a product
        # guarantee. What has to hold is that the fingerprint tracks the exact
        # string the offsets were computed against -- so first show the edit
        # really did move them, then show the fingerprint saw it coming.
        edited = MESSY.replace("Alpha beta gamma delta.", "Alpha delta.")
        moved = [u for u in spans["worst_units"]
                 if edited[u["start"]:u["end"]].split() != u["text"].split()]
        assert moved, ("this edit shifted no offsets, so it cannot demonstrate "
                       "staleness -- choose an edit that does")
        again = await call("detect_spans", {"text": edited, "min_words": 1})
        assert again["source_fingerprint"] != fp, "edit must change the fingerprint"
        assert again["source_fingerprint"] == text_fingerprint(edited)
        # Re-calling is the documented fix, so it must actually fix them.
        for u in again["worst_units"]:
            assert edited[u["start"]:u["end"]].split() == u["text"].split(), u
        print(f"  after edit: {len(moved)} offsets moved, fingerprint {fp} -> "
              f"{again['source_fingerprint']} (detectable), re-call realigns them")

        # The fingerprint must cover the CALLER'S text, not the normalised copy.
        # These two differ only in line endings, so the model sees exactly the
        # same string -- but their offsets are different, and a caller holding
        # one set must not be told the other set is current.
        crlf = "Alpha beta gamma delta.\r\n\r\nEpsilon zeta eta theta. Iota kappa mu."
        lf = crlf.replace("\r\n", "\n")
        assert clean_text(crlf) == clean_text(lf), "these must normalise identically"
        a = await call("detect_spans", {"text": crlf, "min_words": 1})
        b = await call("detect_spans", {"text": lf, "min_words": 1})
        offs_a = [(u["start"], u["end"]) for u in a["worst_units"]]
        offs_b = [(u["start"], u["end"]) for u in b["worst_units"]]
        assert offs_a != offs_b, ("these inputs were supposed to produce different "
                                  f"offsets: {offs_a}")
        assert a["source_fingerprint"] != b["source_fingerprint"], (
            "two texts that normalise identically but carry different offsets share "
            "a fingerprint -- it is being taken over the cleaned copy, so a caller "
            "cannot tell whose coordinates it is holding")
        # Stable for an unchanged input, and the same across tools.
        repeat = await call("detect_spans", {"text": crlf, "min_words": 1})
        assert repeat["source_fingerprint"] == a["source_fingerprint"]
        det_fp = await call("detect", {"text": crlf, "include_windows": True})
        if "source_fingerprint" in det_fp:
            assert det_fp["source_fingerprint"] == a["source_fingerprint"]
        print(f"  fingerprint distinguishes CRLF/LF twins that normalise the same: "
              f"{a['source_fingerprint']} vs {b['source_fingerprint']}")

        sub = await call("chain_submit", {"chain_id": cid, "text": MESSY})
        assert sub["source_fingerprint"] == text_fingerprint(MESSY)
        for u in sub.get("worst_spans", []):
            assert MESSY[u["start"]:u["end"]].split() == u["text"].split()
        print("  chain_submit carries the fingerprint and aligned spans")

        # ------------------------------------------- detect window offsets too
        long_doc = (GOOD + " " + BAD + " ") * 14
        w = await call("detect", {"text": long_doc, "include_windows": True})
        assert w["windows"] > 1
        det = w["window_detail"]
        assert det[0]["owned_start"] == 0
        # Trailing whitespace is stripped before scoring, so it belongs to no
        # window; every character of actual content must still be covered.
        assert det[-1]["owned_end"] >= len(long_doc.rstrip()), (
            det[-1]["owned_end"], len(long_doc.rstrip()))
        for a, b in zip(det, det[1:]):
            assert a["owned_end"] == b["owned_start"]
        assert w["source_fingerprint"] == text_fingerprint(long_doc)
        print(f"  detect windows tile 0..{det[-1]['owned_end']} of {len(long_doc)} "
              f"in ORIGINAL coordinates (trailing space stripped)")

    print("BRANCHING + STALENESS TESTS PASSED")


if __name__ == "__main__":
    print("branching and offset-staleness tests")
    asyncio.run(main())
