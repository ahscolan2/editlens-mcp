"""Span offsets must index the caller's ORIGINAL text.

They used to index the internally-normalised copy, which the caller never sees,
so on any text with CRLF endings or double spaces they silently pointed at the
wrong characters -- and a client splicing a rewrite at them would corrupt its
own document.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from editlens_mcp.detector import clean_text, clean_text_with_map, map_span  # noqa: E402

PY = sys.executable
SCRIPT = str(Path(__file__).resolve().parent.parent / "run_server.py")

# CRLF endings, double spaces after full stops, leading blank lines, trailing
# spaces, a non-breaking space -- i.e. an ordinary Windows document.
MESSY = (
    "\r\n\r\n   The quarterly proposal was circulated on Monday.  "
    "Several members raised concerns about the timeline.   \r\n"
    "\r\n\r\n\r\nMoreover,\u00a0it is important to note that robust frameworks "
    "enable scalable growth.  I forgot to send the invoice though.   Sorry about that.\r\n\r\n"
)


def test_map_agrees_with_clean_text():
    cleaned, imap = clean_text_with_map(MESSY)
    assert cleaned == clean_text(MESSY)
    assert len(imap) == len(cleaned)
    print(f"  raw={len(MESSY)} cleaned={len(cleaned)} map entries={len(imap)}")


def test_map_span_round_trip():
    """Each mapped span must start at the same word it does in the cleaned text."""
    cleaned, imap = clean_text_with_map(MESSY)
    for a in range(0, len(cleaned) - 10, 7):
        b = min(a + 25, len(cleaned))
        o0, o1 = map_span(imap, a, b, len(MESSY))
        assert 0 <= o0 <= o1 <= len(MESSY)
        # First non-space character must match through the mapping.
        assert MESSY[o0] == cleaned[a] or cleaned[a] == " ", (
            f"span {a}:{b} -> {o0}:{o1}: {MESSY[o0]!r} vs {cleaned[a]!r}"
        )
    print("  all mapped spans land on the matching original character")


async def _spans_from_server():
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    env = dict(os.environ)
    env["EDITLENS_DB"] = str(Path(tempfile.mkdtemp()) / "off.db")
    async with Client(StdioTransport(command=PY, args=[SCRIPT], env=env)) as c:
        spans = (await c.call_tool(
            "detect_spans", {"text": MESSY, "min_words": 1, "top": 50})).data
        cid = (await c.call_tool("chain_create", {"name": "o"})).data["chain_id"]
        sub = (await c.call_tool(
            "chain_submit", {"chain_id": cid, "text": MESSY})).data
        return spans, sub


def test_server_offsets_index_original():
    spans, sub = asyncio.run(_spans_from_server())
    assert spans["ok"], spans

    bad = []
    for u in spans["worst_units"]:
        raw_slice = MESSY[u["start"]:u["end"]]
        # The slice from the ORIGINAL text must contain the same words, in order,
        # as the normalised unit text the tool reported.
        if raw_slice.split() != u["text"].split():
            bad.append((u["start"], u["end"], u["text"][:40], raw_slice[:40]))
    print(f"  detect_spans: {len(spans['worst_units'])} units, {len(bad)} misaligned")
    for b in bad[:3]:
        print(f"    [{b[0]}:{b[1]}] tool={b[2]!r} raw={b[3]!r}")
    assert not bad, f"{len(bad)} units have offsets that do not index the caller's text"

    wbad = []
    for u in sub.get("worst_spans", []):
        if MESSY[u["start"]:u["end"]].split() != u["text"].split():
            wbad.append(u)
    print(f"  chain_submit: {len(sub.get('worst_spans', []))} spans, {len(wbad)} misaligned")
    assert not wbad, f"{len(wbad)} worst_spans misaligned"


def test_clean_text_offsets_unchanged_when_text_is_already_clean():
    tidy = "One sentence here. Another sentence follows. A third one closes it."
    cleaned, imap = clean_text_with_map(tidy)
    assert cleaned == tidy
    assert imap == list(range(len(tidy))), "identity map expected for already-clean text"
    print("  already-clean text maps to itself")


if __name__ == "__main__":
    print("offset tests")
    test_map_agrees_with_clean_text()
    test_map_span_round_trip()
    test_clean_text_offsets_unchanged_when_text_is_already_clean()
    test_server_offsets_index_original()
    print("OFFSET TESTS PASSED")
