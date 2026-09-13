"""Model preprocessing matching pangramlabs/EditLens scripts/preprocess.py.

This is deliberately separate from the whitespace-only display/segmentation
normalization in detector.py. Every output character retains a half-open range
in the input, including emoji names and Unicode lowercase expansions.
"""

from __future__ import annotations

import re

REFERENCE_VERSION = "reference-v1"
LEGACY_VERSION = "legacy-whitespace-v1"
BOILERPLATE_STARTS = (
    "Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly",
)


def reference_text_with_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Apply the official preprocessing, preserving original character ranges.

    Preserve the reference's exact ordering and case-sensitive header check.
    Its think-tag operation takes split('</think>')[1], including the unusual
    behavior of discarding text after a *second* closing tag.
    """
    import emoji  # noqa: PLC0415 - legacy mode does not need this dependency

    # analyze(join_emoji=False) exposes the same RGI matches used by demojize,
    # and keeps ZWJs between non-RGI emoji. Finding non-emoji characters from
    # the previous endpoint also accounts for discarded variation selectors.
    chars: list[str] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for token in emoji.analyze(text, non_emoji=True, join_emoji=False):
        if isinstance(token.value, str):
            start = text.index(token.chars, cursor)
            end = start + len(token.chars)
        else:
            start, end = token.value.start, token.value.end
        replacement = emoji.demojize(token.chars)
        chars.append(replacement)
        ranges.extend([(start, end)] * len(replacement))
        cursor = end
    current = "".join(chars)

    def strip() -> None:
        nonlocal current, ranges
        left = len(current) - len(current.lstrip())
        right = len(current.rstrip())
        current, ranges = current[left:right], ranges[left:right]

    tag = "</think>"
    first = current.find(tag)
    if first >= 0:
        begin = first + len(tag)
        second = current.find(tag, begin)
        end = len(current) if second < 0 else second
        current, ranges = current[begin:end], ranges[begin:end]
        strip()

    paragraphs = [m for m in re.finditer(r"[^\n]+", current) if m.group().strip()]
    if paragraphs:
        first_paragraph = re.sub(r"^[^a-zA-Z0-9]*", "", paragraphs[0].group())
        first_paragraph = emoji.replace_emoji(first_paragraph, "")
        if len(paragraphs) > 1 and first_paragraph.startswith(BOILERPLATE_STARTS):
            kept = paragraphs[1:]
            chars, new_ranges = [], []
            for i, paragraph in enumerate(kept):
                if i:
                    chars.append("\n")
                    # The inserted newline represents the original separator,
                    # including any blank lines skipped by the official code.
                    new_ranges.append((ranges[kept[i - 1].end()][0],
                                       ranges[paragraph.start() - 1][1]))
                chars.append(paragraph.group())
                new_ranges.extend(ranges[paragraph.start():paragraph.end()])
            current, ranges = "".join(chars), new_ranges

    # Lowercase the WHOLE string: per-character lower() mishandles Greek final
    # sigma. Only use individual lowercase lengths to map expansions (e.g. İ).
    lowered_ranges = [span for char, span in zip(current, ranges)
                      for _ in char.lower()]
    current, ranges = current.lower(), lowered_ranges

    chars, collapsed_ranges = [], []
    cursor = 0
    for match in re.finditer(r"\s+", current):
        chars.append(current[cursor:match.start()])
        collapsed_ranges.extend(ranges[cursor:match.start()])
        chars.append(" ")
        collapsed_ranges.append((ranges[match.start()][0], ranges[match.end() - 1][1]))
        cursor = match.end()
    chars.append(current[cursor:])
    collapsed_ranges.extend(ranges[cursor:])
    current, ranges = "".join(chars), collapsed_ranges
    strip()
    return current, ranges


def reference_text(text: str) -> str:
    """Return model text without changing the original document."""
    return reference_text_with_map(text)[0]
