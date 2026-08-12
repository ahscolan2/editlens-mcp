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
    split_units_adaptive,
    text_fingerprint,
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
        # span_min_words=1 because MESSY's sentences are short: at the default of
        # 25 they merge into a single unit, _worst_spans returns nothing, and
        # every per-span assertion below passes vacuously.
        sub = (await c.call_tool(
            "chain_submit", {"chain_id": cid, "text": MESSY, "span_min_words": 1})).data
        # The string the caller can actually get back -- the one it has to splice
        # the offsets into.
        got = (await c.call_tool(
            "chain_get_text", {"chain_id": cid, "step": "latest"})).data
        return spans, sub, got


_SERVER_RESULT = None


def _server_result():
    """One subprocess for both server tests: each launch reloads the model."""
    global _SERVER_RESULT
    if _SERVER_RESULT is None:
        _SERVER_RESULT = asyncio.run(_spans_from_server())
    return _SERVER_RESULT


def test_server_offsets_index_original():
    spans, sub, _ = _server_result()
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


def test_stored_draft_is_the_text_the_caller_sent():
    """chain_get_text must hand back the submitted document unchanged.

    The store used to hold clean_text(text), which collapses every run of spaces
    to one -- flattening markdown nesting, code indentation and table alignment,
    and collapsing every indent level to the SAME single space, so the nesting
    could not be reconstructed from what came back.

    It also made the offsets and source_fingerprint in the very same
    chain_submit response describe a string no tool would return: they are
    computed against the raw text, so a client that retrieved its draft and
    spliced a rewrite at them cut across sentence boundaries. Nothing caught it
    because no test ever sliced chain_get_text's output at those offsets --
    they were only ever checked against the fixture the test itself submitted.
    """
    _, sub, got = _server_result()
    assert got["ok"], got
    assert got["text"] == MESSY, (
        f"stored draft is not what was submitted: {len(got['text'])} chars back "
        f"vs {len(MESSY)} sent"
    )

    spans = sub.get("worst_spans", [])
    assert spans, "no spans returned -- the checks below would pass vacuously"
    for u in spans:
        assert got["text"][u["start"]:u["end"]].split() == u["text"].split(), (
            f"span [{u['start']}:{u['end']}] does not index the retrieved draft: "
            f"{got['text'][u['start']:u['end']]!r} vs {u['text']!r}"
        )

    # The fingerprint has to be reproducible from that same retrieved string, or
    # a client cannot tell stale offsets from current ones.
    assert text_fingerprint(got["text"]) == sub["source_fingerprint"], (
        f"{text_fingerprint(got['text'])} != {sub['source_fingerprint']}"
    )
    print(f"  stored draft round-trips exactly ({len(MESSY)} chars); "
          f"{len(spans)} spans index it; fingerprint matches")


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


def test_paragraph_units_are_the_paragraphs_exactly():
    """Paragraph spans must be the authored blocks -- no separator, nothing clipped.

    Only the unit COUNT was ever asserted, so a paragraph span that swallowed the
    blank line before it, or lost its final character, passed every suite. Both
    corrupt a splice: the first re-inserts the separator, the second eats a full
    stop.
    """
    cases = [
        # (text, expected unit texts)
        ("First para line one.\nStill first para.\n\nSecond para here.\n\nThird ends.",
         ["First para line one.\nStill first para.", "Second para here.", "Third ends."]),
        ("alpha\n\n\n\nbeta", ["alpha", "beta"]),
        # Blank blocks must be dropped, not handed to the model as empty units
        # (detect_many raises "item N is empty" on those).
        ("a\n\nb\n\n", ["a", "b"]),
        ("\n\n\na\n\nb", ["a", "b"]),
        # Separators made of whitespace-only lines.
        ("one\n   \ntwo", ["one", "two"]),
        ("single paragraph only", ["single paragraph only"]),
    ]
    for text, expected in cases:
        units = split_units(text, "paragraph")
        got = [text[a:b] for a, b in units]
        assert got == expected, f"{text!r} -> {got!r}, expected {expected!r}"
        for a, b in units:
            assert text[a:b] == text[a:b].strip(), (
                f"unit {(a, b)} of {text!r} carries separator whitespace: {text[a:b]!r}")
        for (_, a1), (b0, _) in zip(units, units[1:]):
            assert a1 <= b0, f"paragraph units overlap in {text!r}: {units}"
    print(f"  {len(cases)} paragraph inputs split on authored boundaries exactly")


def test_paragraph_offsets_index_the_original():
    """Paragraph granularity gets the same offset guarantee as sentences."""
    raw = ("\r\n\r\n  First para line one.\r\nStill  first para.   \r\n"
           "\r\n\r\n\r\nSecond  para here.  \r\n\r\nThird para ends.\r\n\r\n")
    cleaned, ranges = clean_text_with_map(raw)
    units = split_units(cleaned, "paragraph")
    assert len(units) == 3, [cleaned[a:b] for a, b in units]
    mapped = [map_span(ranges, a, b, len(raw)) for a, b in units]
    for (a, b), (o0, o1) in zip(units, mapped):
        assert raw[o0:o1].split() == cleaned[a:b].split(), (
            f"paragraph {(a, b)} -> {(o0, o1)}: {raw[o0:o1]!r} vs {cleaned[a:b]!r}")
        # A mapped paragraph must not start or end inside the separator.
        assert raw[o0:o1].strip() == raw[o0:o1].strip("\r\n "), raw[o0:o1]
    for (_, x1), (y0, _) in zip(mapped, mapped[1:]):
        assert x1 <= y0, f"mapped paragraphs overlap: {mapped}"
    assert mapped[-1][1] <= len(raw)
    print(f"  3 paragraphs map to {mapped} in the original")


def test_paragraph_granularity_applies_no_threshold():
    """split_units_adaptive must report 0 for paragraphs, because min_words was
    never applied -- that 0 is what makes the tool answer `min_words_used: null`
    instead of naming a threshold it never enforced."""
    text = "One. Two. Three.\n\nFour. Five. Six."
    units, used = split_units_adaptive(text, granularity="paragraph")
    assert [text[a:b] for a, b in units] == ["One. Two. Three.", "Four. Five. Six."]
    assert used == 0, f"paragraphs reported a threshold of {used}"
    # Even with a threshold that would swallow both, paragraphs stay paragraphs.
    units2, used2 = split_units_adaptive(text, granularity="paragraph", min_words=200)
    assert units2 == units and used2 == 0, (units2, used2)
    # A single paragraph must not be relaxed into sentences either.
    solo, solo_used = split_units_adaptive("One. Two. Three.", granularity="paragraph")
    assert len(solo) == 1 and solo_used == 0, (solo, solo_used)
    print("  paragraph granularity reports no threshold and never sentence-splits")


def test_map_span_refuses_impossible_spans():
    """An empty, inverted or out-of-range span must map to nothing, and no span
    may end past the text the caller actually holds."""
    cleaned, ranges = clean_text_with_map("alpha  beta\r\ngamma delta")
    n = len(cleaned)
    original = len("alpha  beta\r\ngamma delta")

    assert map_span(ranges, 5, 5, original) == (0, 0), "empty span must map to nothing"
    assert map_span(ranges, 7, 2, original) == (0, 0), "inverted span must map to nothing"
    assert map_span(ranges, n, n + 4, original) == (0, 0), "span past the map"
    assert map_span([], 0, 3, original) == (0, 0)

    # The clamp exists for the caller who hands us a shorter string than the one
    # the map was built from; offsets past its end are unsliceable.
    o0, o1 = map_span(ranges, 0, n, 4)
    assert o1 <= 4, f"offset {o1} points past the caller's {4}-char text"
    assert o0 <= o1

    # And the ordinary case is untouched.
    o0, o1 = map_span(ranges, 0, n, original)
    assert (o0, o1) == (0, original), (o0, o1)
    print("  map_span rejects empty/inverted/out-of-range spans and clamps the end")


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
    test_paragraph_units_are_the_paragraphs_exactly()
    test_paragraph_offsets_index_the_original()
    test_paragraph_granularity_applies_no_threshold()
    test_map_span_refuses_impossible_spans()
    test_clean_text_offsets_unchanged_when_text_is_already_clean()
    test_server_offsets_index_original()
    test_stored_draft_is_the_text_the_caller_sent()
    print("OFFSET TESTS PASSED")
