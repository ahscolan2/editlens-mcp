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

from editlens_mcp.detector import (  # noqa: E402
    clean_text,
    clean_text_with_map,
    map_span,
    split_units,
)

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
        # The mapped slice must carry the same words as the cleaned slice.
        # Exact character equality cannot hold: cleaning collapses CRLF to LF
        # and space runs to one, so the original slice legitimately holds more.
        assert MESSY[o0:o1].split() == cleaned[a:b].split(), (
            f"span {a}:{b} -> {o0}:{o1}: {MESSY[o0:o1]!r} vs {cleaned[a:b]!r}"
        )
    print("  all mapped spans carry the same words as the cleaned slice")


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


def test_spans_tile_the_original_exactly():
    """Character-exact coverage, not word-level.

    Comparing with .split() masks a missing newline, so the CRLF truncation bug
    (span ending at the \\r instead of after the \\n) survived every earlier
    assertion. Units tile the cleaned text contiguously, so their mapped spans
    must tile the original contiguously too -- any truncation opens a gap.
    """
    cleaned, ranges = clean_text_with_map(MESSY)
    units = split_units(cleaned, "sentence", 1)
    mapped = [map_span(ranges, a, b, len(MESSY)) for a, b in units]

    for (a0, a1), (b0, b1) in zip(mapped, mapped[1:]):
        assert a1 == b0, f"gap/overlap in original coordinates: {a1} -> {b0}"

    # Every character between the first and last span must be accounted for.
    covered = MESSY[mapped[0][0]:mapped[-1][1]]
    assert covered == "".join(MESSY[s:e] for s, e in mapped), "spans do not tile"

    # And the untouched head/tail is whitespace only (what strip() removed).
    assert not MESSY[:mapped[0][0]].strip()
    assert not MESSY[mapped[-1][1]:].strip()
    print(f"  {len(mapped)} spans tile chars {mapped[0][0]}..{mapped[-1][1]} exactly")


def test_ranges_are_contiguous():
    """Consecutive cleaned chars must map to consecutive source ranges.

    Every source character has to belong to some cleaned character's range,
    including whitespace that cleaning dropped. A gap means those characters are
    unaccounted for, and a span ending at one silently loses them.
    """
    cases = [
        "x  \na",                       # trailing spaces before a newline
        "a\t\t\nb",                     # trailing tabs
        "one   \r\n   two",             # both sides of a CRLF
        "p  \nq",             # non-breaking spaces
        "line   \n\n\n\n   next",       # dropped run plus collapsed blank lines
        "solo",                         # nothing to drop
        "a \n b \n c",
    ]
    for text in cases:
        cleaned, ranges = clean_text_with_map(text)
        if not cleaned:
            continue
        for i, ((_, e), (s, _)) in enumerate(zip(ranges, ranges[1:])):
            assert e == s, (
                f"{text!r}: gap between range {i} and {i + 1}: ends {e}, next starts {s} "
                "-- source characters unmapped")
        assert ranges[0][0] >= 0 and ranges[-1][1] <= len(text)
    print(f"  {len(cases)} inputs: source ranges contiguous, nothing unmapped")


def test_crlf_span_keeps_both_characters():
    """A span whose cleaned form ends in \\n must include the full \\r\\n."""
    text = "First line with CRLF.\r\nSecond line with CRLF.\r\nThird line here."
    cleaned, ranges = clean_text_with_map(text)
    end = cleaned.index("\n") + 1
    o0, o1 = map_span(ranges, 0, end, len(text))
    sliced = text[o0:o1]
    assert cleaned[:end].endswith("\n")
    assert sliced.endswith("\r\n"), (
        f"span truncated mid-CRLF: {sliced!r} -- splicing here orphans a newline")
    # Splicing must reconstruct the document without loss.
    spliced = text[:o0] + "REPLACED\r\n" + text[o1:]
    assert spliced == "REPLACED\r\nSecond line with CRLF.\r\nThird line here.", spliced
    print(f"  CRLF span = {sliced!r}; splice reconstructs cleanly")


def test_clean_text_offsets_unchanged_when_text_is_already_clean():
    tidy = "One sentence here. Another sentence follows. A third one closes it."
    cleaned, imap = clean_text_with_map(tidy)
    assert cleaned == tidy
    assert imap == [(i, i + 1) for i in range(len(tidy))], "identity ranges expected"
    print("  already-clean text maps to itself")


if __name__ == "__main__":
    print("offset tests")
    test_map_agrees_with_clean_text()
    test_map_span_round_trip()
    test_spans_tile_the_original_exactly()
    test_ranges_are_contiguous()
    test_crlf_span_keeps_both_characters()
    test_clean_text_offsets_unchanged_when_text_is_already_clean()
    test_server_offsets_index_original()
    print("OFFSET TESTS PASSED")
