"""EditLens MCP server.

Exposes a local AI-text detector to an MCP client, plus a durable chain store so
a model can run long write -> score -> revise loops without carrying the history
in its context window.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

from . import __version__
from .chains import ChainStore, default_db_path
from .guidance import SCORE_NOTE, SPAN_NOTE, TARGET_NOTE, length_assessment, revision_progress, submission_advice
from .preprocessing import reference_text, reference_text_with_map
from .detector import (
    DetectorUnavailable,
    EditLensDetector,
    clean_text_with_map,
    map_span,
    split_units_adaptive,
    text_fingerprint,
    warmup_imports,
)

# MCP tool annotations. Clients use these to decide what needs confirmation:
# every tool here is local (openWorldHint=False), scoring and reads never
# change state, and only chain_delete destroys saved drafts.
_READ_ONLY = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}
_WRITES = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False,
           "openWorldHint": False}

mcp = FastMCP(
    name="editlens",
    version=__version__,
    instructions=(
        "Local document-level estimate of AI editing magnitude. Scores are not probabilities "
        "of AI authorship or writing-quality scores. detect_spans scores fragments independently; "
        "it does not explain the document score. Scores are not comparable across lengths. "
        "Use next_action for workflow context, but preserve meaning, facts, and voice over a "
        "lower number. Do not rewrite short text solely to move its score. A chain stores drafts "
        "durably; best means lowest-scoring, not best writing. Use chain_get_text and branch_from "
        "to inspect earlier drafts. Stop score-driven loops when stop_recommended is true; "
        "you may still save further revisions. A configured target is not a calibrated boundary. "
        "Score complete documents and assemble multi-section drafts before considering revisions. "
        "Independent MCP sessions share one local inference worker by default. Requests queue; "
        "prefer detect_batch to many separate calls. The first scoring call loads the checkpoint."
    ),
)


def _env_number(name: str, default, cast):
    """Read a numeric setting, falling back rather than dying at import.

    This module runs at import time, so an unusable value here kills the process
    before the server can say anything and the MCP client just sees a launch that
    exited. Empty is the common case -- client configs routinely emit
    `"EDITLENS_BATCH_SIZE": ""` for an unset field -- and it is the exact failure
    already fixed for EDITLENS_DB. EDITLENS_DEVICE/EDITLENS_DTYPE tolerate it via
    `or None`; these two did not.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = cast(raw)
        if not math.isfinite(value):
            raise ValueError
        return value
    except (TypeError, ValueError):
        print(
            f"[editlens] warning: ignoring {name}={raw!r} (not a number); using {default}",
            file=sys.stderr,
        )
        return default


DETECTOR_CONFIG = dict(
    device=os.environ.get("EDITLENS_DEVICE") or None,
    batch_size=_env_number("EDITLENS_BATCH_SIZE", 8, int),
    dtype=os.environ.get("EDITLENS_DTYPE") or None,
    idle_unload_seconds=_env_number("EDITLENS_IDLE_UNLOAD", 300.0, float),
    # Case/whitespace-insensitive, like the other settings: "Legacy" or a
    # trailing space in a JSON client config is not a different profile.
    preprocessing=(os.environ.get("EDITLENS_PREPROCESS") or "").strip().lower() or "reference",
)
# Direct Python use remains in-process. main() replaces this configuration-only
# object with the shared adapter before accepting any MCP request.
detector = EditLensDetector(**DETECTOR_CONFIG)

# default_db_path() re-reads EDITLENS_DB and treats an empty value as unset.
# Passing os.environ.get(...) here instead handed it "" and crashed at import.
#
# The construction is guarded because this line is the only I/O at import and
# was the only statement in the file outside _guard's protection: an unwritable
# directory, a path component that is a file, or a chains.db that arrived
# half-copied ("file is not a database") killed the process before FastMCP
# registered a single tool. Chain tools now fail per-call, scoring stays usable,
# and detector_info reports the database error.
class _BrokenStore:
    """Stands in for ChainStore when the database could not be opened.

    Every attribute access raises the captured error, so each tool that needs
    the store returns an ok=False dict (via _guard) naming the actual problem
    instead of the server dying at import.
    """

    def __init__(self, error: str, path) -> None:
        self.error = error
        self.path = path

    def __getattr__(self, name: str):
        raise RuntimeError(self.error)


def _open_store():
    path = None
    try:
        path = default_db_path()
        return ChainStore(path), None
    except Exception as exc:  # noqa: BLE001 - import-time boundary
        msg = (
            f"chain store failed to open: {type(exc).__name__}: {exc} "
            f"(EDITLENS_DB={os.environ.get('EDITLENS_DB')!r} resolved to '{path}'). "
            f"Only the chain tools need this "
            f"database; fix the path and restart the server."
        )
        print(f"[editlens] warning: {msg}", file=sys.stderr)
        return _BrokenStore(msg, path), msg


store, STORE_ERROR = _open_store()

MAX_SPAN_REPORT = 5
# Merge threshold for span feedback, matching detect_spans' own default.
SPAN_MIN_WORDS = 25
# Compatibility flag: this is only a length check, not calibrated confidence.
RELIABLE_WORDS = 75
# Workflow heuristic for pointing out a higher score; not a noise estimate.
REGRESSION_DELTA = 0.05

SCALE_NOTE = SPAN_NOTE


def _short_text_note(words: int) -> str | None:
    if words >= 75:
        return None
    return (
        f"Only {words} words: below the reference training floor of 75 words. "
        "This is an uncalibrated estimate, not a measurement of authorship. "
        "Do not rewrite prose to move it; review the full document when available."
    )


def _scoring_profile() -> dict:
    return detector.scoring_identity()


def _check_chain_profile(chain) -> None:
    saved = json.loads(chain["meta"] or "{}").get("scoring_profile")
    current = _scoring_profile()
    if saved is None and current["preprocessing"] == "legacy":
        return  # pre-profile chains used the whitespace-only scoring path
    if saved != current:
        # Name what differs. A routine `pip install -U emoji` changes only
        # emoji_version, and without this the caller could not tell that from
        # a different checkpoint or preprocessing mode.
        if saved is None:
            diff = "this chain predates scoring profiles (legacy whitespace-only scoring)"
        else:
            keys = sorted(set(saved) | set(current))
            diff = "; ".join(
                f"{k}: chain {saved.get(k)!r}, current {current.get(k)!r}"
                for k in keys if saved.get(k) != current.get(k)
            )
        raise ValueError(
            f"This chain was scored with a different scoring profile ({diff}). "
            "Saved drafts remain available through chain_get_text/history. "
            "Create a new chain and resubmit a chosen draft to use current scoring, "
            "or use EDITLENS_PREPROCESS=legacy for an older whitespace-only chain. "
            "Scores from different pipelines must not be ranked together."
        )


def _require_name(value: str, what: str) -> None:
    """Reject blank chain and segment names.

    A blank segment became a section no listing can show legibly, and advice
    read "Draft segment '' and call chain_submit(segment='')". Names are kept
    exactly as given otherwise; they are identifiers, not display text.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must not be empty or whitespace-only")


def _fail(exc: Exception) -> dict:
    # str(KeyError) wraps the message in quotes; unwrap it for readability.
    msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
    return {"ok": False, "error": str(msg), "error_type": type(exc).__name__}


def _guard(fn):
    """Every tool must return a dict, never raise.

    Catching only the expected exception types is not enough: a CUDA OOM arrives
    as a plain RuntimeError and a broken Windows torch install as an OSError, and
    either escaping turns a diagnosable message into an opaque client-side
    ToolError. Anything unexpected still reaches the caller -- as data.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberate tool-boundary catch
            return _fail(exc)

    return wrapper


def _scoreable(
    text: str, source: str, imap: list[tuple[int, int]], units: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Keep only units inside the region the document score actually uses.

    Reference preprocessing discards a reasoning block up to the first
    </think> and a leading boilerplate line ("Sure, here is..."), as the
    upstream pipeline does. Units there were still scored and ranked as
    "worst spans" although the document score ignores that text -- and a unit
    lying wholly inside a reasoning block had no model input at all, raising
    "text is empty after reference preprocessing" and failing the whole
    detect_spans call, or silently removing all of chain_submit's span
    feedback, on exactly the drafts reasoning models emit.
    """
    if detector.preprocessing != "reference":
        return [(a, b) for a, b in units if source[a:b].strip()]
    _, ranges = reference_text_with_map(text)
    if not ranges:
        return []
    lo, hi = ranges[0][0], ranges[-1][1]
    kept = []
    for a, b in units:
        o0, o1 = map_span(imap, a, b, len(text))
        if o1 <= lo or o0 >= hi:
            continue  # wholly within text the document score discards
        # Clip a unit straddling the boundary ("...</think>\n\nFirst sentence.")
        # so its text and offsets show only what is scored.
        while a < b and imap[a][0] < lo:
            a += 1
        while b > a and imap[b - 1][1] > hi:
            b -= 1
        while a < b and source[a].isspace():
            a += 1
        if a < b and reference_text(source[a:b]):  # the unit alone must have input
            kept.append((a, b))
    return kept


def _worst_spans(
    text: str,
    granularity: str = "sentence",
    top: int = MAX_SPAN_REPORT,
    target: float | None = None,
    min_words: int = SPAN_MIN_WORDS,
) -> tuple[list[dict], dict]:
    """Score sentence-groups and return the highest-scoring ones, plus totals.

    `start`/`end` index the caller's ORIGINAL text, not the normalised copy the
    model sees -- offsets you cannot splice against are worse than no offsets.

    `above_target` is only a numerical comparison. An isolated fragment score
    does not identify a writing defect or establish sentence provenance.

    The second return value describes the WHOLE text, not the `top` slice of it.
    Reporting only the slice is how a caller ends up being told that 31 units it
    never saw are "already at or below target": on a 1548-word document all 36
    units scored above a 0.25 target, and the response named 5.
    """
    source, imap = clean_text_with_map(text)
    units, used = split_units_adaptive(
        source, granularity=granularity, min_words=min_words
    )
    units = _scoreable(text, source, imap, units)
    meta = {"unit_count": len(units), "above_target_total": 0, "min_words_used": used}
    if len(units) < 2:
        return [], meta
    verdicts = detector.detect_many([source[a:b] for a, b in units], normalise=False)
    rows = []
    for (a, b), v in zip(units, verdicts):
        o0, o1 = map_span(imap, a, b, len(text))
        row = {
            "start": o0,
            "end": o1,
            "score": round(v.score, 4),
            "label": v.label,
            "words": v.word_count,
            "model_word_count": v.model_word_count if v.model_word_count is not None else v.word_count,
            "assessment_word_count": v.assessment_word_count,
            # The same per-unit caveat detect_spans already carries. A 9-word
            # unit scored 0.99 is not the same evidence as a 40-word one.
            "reliable": v.assessment_word_count >= RELIABLE_WORDS,
            "text": source[a:b],
        }
        if target is not None:
            row["above_target"] = v.score > target
        rows.append(row)
    rows.sort(key=lambda r: r["score"], reverse=True)
    if target is not None:
        meta["above_target_total"] = sum(1 for r in rows if r["above_target"])
    return rows[:top], meta


# --------------------------------------------------------------------- detector


@mcp.tool(annotations=_READ_ONLY)
@_guard
def detector_info() -> dict:
    """Report detector status: checkpoint, device, dtype, bucket labels, whether the
    model is loaded yet, and whether an HF token is visible. Call this first if
    anything errors."""
    # "ok" on the success path too: _guard supplies it on failure, and a client
    # branching on result["ok"] should not KeyError when nothing went wrong.
    try:
        info = {"ok": True, **detector.info()}
    except Exception as exc:  # noqa: BLE001 - diagnostics must survive a dead worker
        # In shared mode info() is an RPC that first starts the worker. When
        # that fails -- a broken torch install, a startup timeout, a busy
        # owner -- this tool used to return only the bare error, dropping the
        # database, runtime and log locations needed to diagnose it. Keep the
        # failure contract (ok/error/error_type) and add what is known locally.
        offline = getattr(detector, "offline_info", None)
        info = {**(offline() if offline else {}), **_fail(exc)}
    info.setdefault("backend", "local")
    info["score_note"] = SCORE_NOTE
    info["reliability_basis"] = "Length only (75-word training floor); not calibrated confidence."
    # Resolved, not the configured string: a relative EDITLENS_DB used to be
    # echoed back verbatim ("chains.db"), which names no location on disk.
    info["db_path"] = str(Path(store.path).resolve()) if store.path is not None else None
    info["db_path_configured"] = os.environ.get("EDITLENS_DB")
    if STORE_ERROR is not None:
        # This tool's whole job is answering "why is everything broken?" --
        # so it reports a dead chain store rather than dying of one.
        info["db_error"] = STORE_ERROR
        info["duplicate_steps_present"] = None
    else:
        info["duplicate_steps_present"] = store.duplicate_steps_present
    return info


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
@_guard
def detector_unload() -> dict:
    """Release the model from GPU memory immediately.

    Rarely needed -- the model unloads itself after a few minutes idle and
    reloads automatically on the next call. Use this only to free VRAM right now,
    e.g. before starting a game or another GPU job. The shared model is used by
    every MCP session on this machine, so this affects all of them. It never
    starts an inference worker.
    """
    freed = detector.unload()
    return {
        "ok": True,
        "was_loaded": freed,
        "message": (
            "Model unloaded, GPU memory released. It reloads automatically on the next call."
            if freed
            else "Model was not loaded; no GPU memory was in use."
        ),
    }


@mcp.tool(annotations=_READ_ONLY)
@_guard
def detect(
    text: Annotated[str, Field(description="The text to score.")],
    include_windows: Annotated[
        bool, Field(description="Also return per-window scores for long inputs.")
    ] = False,
) -> dict:
    """Estimate the extent of AI editing in one document.

    Returns expected bucket index scaled to [0,1], the most probable bucket,
    and the probability distribution. These are uncalibrated model estimates.
    Inputs longer than the model's 512-token window are split into overlapping
    windows and combined by word-count-weighted average.
    """
    try:
        source, ranges = clean_text_with_map(text)
        if not source.strip():
            return _fail(ValueError("empty text"))
        verdict, windows = detector.detect(text)
    except (DetectorUnavailable, ValueError) as exc:
        return _fail(exc)
    out: dict[str, Any] = {"ok": True, **verdict.as_dict()}
    out["target_hint"] = "lower estimates less AI editing; it does not measure writing quality"
    out["score_note"] = SCORE_NOTE
    out["scoring_profile"] = _scoring_profile()
    out.update(length_assessment(verdict.assessment_word_count))
    # detect_spans has flagged short units as unreliable since it was written;
    # `detect` reported a bare 4-decimal score for a 26-word input and said
    # nothing. That is the number a caller acts on, so it is the one that most
    # needs the caveat.
    out["reliable"] = verdict.assessment_word_count >= RELIABLE_WORDS
    note = _short_text_note(verdict.assessment_word_count)
    if note:
        out["reliability_note"] = note
    if include_windows and len(windows) > 1:
        # Window offsets, like span offsets, must index the caller's own text.
        for w in windows:
            w["start"], w["end"] = map_span(ranges, w["start"], w["end"], len(text))
            w["owned_start"], w["owned_end"] = map_span(
                ranges, w["owned_start"], w["owned_end"], len(text)
            )
        out["window_detail"] = windows
        out["source_fingerprint"] = text_fingerprint(text)
    return out


@mcp.tool(annotations=_READ_ONLY)
@_guard
def detect_batch(
    texts: Annotated[list[str], Field(description="Texts to score in a single pass.")],
) -> dict:
    """Score candidate documents together. best_index means lowest detector
    score, not best writing. Compare similar-length revisions of the same full
    document, review quality independently, and inspect each length assessment.
    """
    if not texts:
        return _fail(ValueError("texts is empty"))
    try:
        verdicts = detector.detect_many(texts)
    except (DetectorUnavailable, ValueError) as exc:
        return _fail(exc)
    results = [{"index": i, **v.as_dict(), **length_assessment(v.assessment_word_count),
                "reliable": v.assessment_word_count >= RELIABLE_WORDS,
                **({"reliability_note": _short_text_note(v.assessment_word_count)} if v.assessment_word_count < 75 else {})}
               for i, v in enumerate(verdicts)]
    best_index = min(range(len(verdicts)), key=lambda i: verdicts[i].score)
    best = results[best_index]
    return {
        "ok": True,
        "results": results,
        "best_index": best["index"],
        "score_note": SCORE_NOTE,
        "scoring_profile": _scoring_profile(),
        "best_score": best["score"],
        "mean_score": round(sum(r["score"] for r in results) / len(results), 4),
        # Observed: three candidate paragraphs scored 0.50/0.99/0.66, and the
        # best one spliced into its document took that document to 0.06. A
        # caller comparing best_score against its target rejects all three.
        "comparison_note": (
            f"This ranking is only by detector score, not writing quality. Compare similar-length "
            f"versions of the same complete document. {SCALE_NOTE}"
        ),
    }


@mcp.tool(annotations=_READ_ONLY)
@_guard
def detect_spans(
    text: Annotated[str, Field(description="The text to break apart and score.")],
    granularity: Annotated[
        Literal["sentence", "paragraph"], Field(description="Unit of analysis.")
    ] = "sentence",
    top: Annotated[int, Field(description="How many worst-scoring units to return.", ge=1, le=50)] = 10,
    min_words: Annotated[
        int, Field(description="Merge sentences until a unit reaches this many words.", ge=1, le=200)
    ] = 25,
) -> dict:
    """Score sentence groups or paragraphs independently, sorted highest first.

    These fragment scores do not explain the document score or identify text
    that must be rewritten. Units shorter than the 75-word training floor are
    flagged. start/end index your original text; text is display-normalized.
    The model applies the reported scoring_profile to each unit independently.
    """
    try:
        source, imap = clean_text_with_map(text)
        units, used = split_units_adaptive(
            source, granularity=granularity, min_words=min_words
        )
        units = _scoreable(text, source, imap, units)
        if not units:
            return _fail(ValueError("no scoreable units"))
        verdicts = detector.detect_many([source[a:b] for a, b in units], normalise=False)
        overall, _ = detector.detect(text)
    except (DetectorUnavailable, ValueError) as exc:
        return _fail(exc)

    rows = []
    for i, ((a, b), v) in enumerate(zip(units, verdicts)):
        o0, o1 = map_span(imap, a, b, len(text))
        rows.append(
            {
                "unit": i,
                "start": o0,
                "end": o1,
                "score": round(v.score, 4),
                "label": v.label,
                "words": v.word_count,
                "model_word_count": v.model_word_count if v.model_word_count is not None else v.word_count,
                "assessment_word_count": v.assessment_word_count,
                # Length check only; fragment estimates remain uncalibrated.
                "reliable": v.assessment_word_count >= RELIABLE_WORDS,
                "text": source[a:b],
            }
        )
    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)[:top]
    out = {
        "ok": True,
        "document_score": round(overall.score, 4),
        "document_label": overall.label,
        "unit_count": len(rows),
        # Offsets are valid ONLY for text with this fingerprint. Rewrite one
        # sentence and every later offset shifts; re-call rather than reusing.
        "source_fingerprint": text_fingerprint(text),
        "offsets_note": (
            "start/end index the exact text you passed. After any edit they are "
            "stale -- call detect_spans again, or match on `text` instead."
        ),
        "worst_units": ranked,
    }
    if granularity == "sentence":
        out["min_words_used"] = used
        out["granularity_relaxed"] = used < min_words
    else:
        # Paragraphs are authored boundaries -- min_words was never applied.
        out["min_words_used"] = None
        out["granularity_relaxed"] = False
    out["unreliable_units"] = sum(1 for r in rows if not r["reliable"])
    # The same document-level length flag detect/chain_submit/chain_assemble carry.
    out["reliable"] = overall.assessment_word_count >= RELIABLE_WORDS
    note = _short_text_note(overall.assessment_word_count)
    if note:
        out["reliability_note"] = note
    out["analysis_note"] = SPAN_NOTE
    out["score_note"] = SCORE_NOTE
    out["scoring_profile"] = _scoring_profile()
    out.update(length_assessment(overall.assessment_word_count))
    return out


# ------------------------------------------------------------------------ chains


@mcp.tool(annotations=_WRITES)
@_guard
def chain_create(
    name: Annotated[str, Field(description="Human-readable name for this chain.")],
    target_score: Annotated[
        float, Field(description="Stop when the score is at or below this.", ge=0.0, le=1.0)
    ] = 0.25,
    goal: Annotated[str | None, Field(description="What is being written, for your own recall.")] = None,
    segments: Annotated[
        list[str] | None,
        Field(description="Ordered section names for a multi-part document. Defaults to ['main']."),
    ] = None,
) -> dict:
    """Open a durable write/score/revise chain.

    State is stored in SQLite, so the chain survives restarts and can run for
    hundreds of steps without accumulating in your context. Use `segments` when
    composing a long document section by section.
    """
    try:
        _require_name(name, "name")
        for s in segments or []:
            _require_name(s, "segment names")
        created = store.create(name, target_score, goal, segments,
                               meta={"scoring_profile": _scoring_profile()})
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)
    first = created["segments"][0] if created.get("segments") else "main"
    return {
        "ok": True,
        **created,
        "scoring_profile": _scoring_profile(),
        "target_note": TARGET_NOTE,
        "next_action": (
            f"Draft segment {first!r} and call "
            f"chain_submit(chain_id={created['chain_id']!r}, segment={first!r}, text=...)."
        ),
    }


@mcp.tool(annotations=_WRITES)
@_guard
def chain_submit(
    chain_id: Annotated[str, Field(description="Chain to append to.")],
    text: Annotated[str, Field(description="The draft to score and store.")],
    segment: Annotated[str, Field(description="Which segment this draft belongs to.")] = "main",
    note: Annotated[str | None, Field(description="What you changed, for the trajectory log.")] = None,
    span_feedback: Annotated[
        bool, Field(description="Include the worst-scoring sentences in the response.")
    ] = True,
    span_top: Annotated[
        int,
        Field(description="How many worst-scoring spans to return. Raise it on long "
                          "drafts, where the default shows a small window of the units "
                          "that are actually above target.", ge=1, le=50),
    ] = MAX_SPAN_REPORT,
    span_min_words: Annotated[
        int,
        Field(description="Merge sentences until a span reaches this many words. Lower "
                          "it for smaller, more precisely targeted spans; raise it for "
                          "steadier per-span scores.", ge=1, le=200),
    ] = SPAN_MIN_WORDS,
    branch_from: Annotated[
        int | None,
        Field(description="Step number this draft was derived from. Use to fork an "
                          "alternative from an earlier draft instead of the latest one.",
              ge=1),
    ] = None,
) -> dict:
    """Score and save a draft verbatim, returning compact workflow feedback.

    best_step means lowest detector score; review writing quality independently.
    Span scores are exploratory comparisons, not document attribution or rewrite
    instructions. next_action and stop_recommended flag short text, regressions,
    unchanged drafts, plateaus and an eight-submission review point. These are
    advisory; every valid submission is stored. branch_from records an earlier
    draft as the parent. Never mix scoring profiles within one chain.
    """
    try:
        _require_name(segment, "segment")
        chain = store.get(chain_id)
        _check_chain_profile(chain)
        segs = json.loads(chain["segments"])
        is_new_segment = segment not in segs

        # The caller-visible previous draft: what the agent was actually
        # revising when it built this submission. Used ONLY for parent_step --
        # everything comparative (is_new_best, best_step, the deltas) is
        # recomputed inside add_step's transaction below, because these
        # pre-scoring reads are seconds stale by insert time and under
        # concurrent submits every racer would otherwise crown itself the best.
        prev = store.latest_step(chain_id, segment)
        # The draft this one revises: the branch point when given, else the
        # latest. Duplicate detection must compare against it -- comparing a
        # branched draft with the latest step missed an unchanged resubmission
        # of the branch point and flagged a real revision as "unchanged".
        parent = prev
        if branch_from is not None:
            parent = store.get_step(chain_id, segment, branch_from)
            if parent is None:
                return _fail(KeyError(f"cannot branch from step {branch_from}: no such step "
                                      f"in segment '{segment}'"))
        # Score BEFORE touching the store. Scoring first means a failed submit
        # -- empty text, a CUDA OOM -- writes nothing at all. The segment is
        # then registered inside add_step's OWN transaction (register_segment),
        # not in a separate call before it: a separate registration that
        # committed before an add_step that then failed ("database is locked")
        # left a declared segment with zero steps, which chain_assemble
        # reported as incomplete forever.
        verdict, _ = detector.detect(text)
        step_result = store.add_step(
            chain_id,
            segment,
            # The caller's ORIGINAL, not clean_text(text). Normalisation belongs
            # on the way into the model, not into the store. It collapses every
            # run of spaces to one, so a draft came back from chain_get_text with
            # markdown nesting flattened, code indentation destroyed (a fenced
            # Python block returned as an IndentationError) and table alignment
            # gone -- 14 of 27 lines altered on an ordinary document, and every
            # indent level collapsed to the SAME single space, so the nesting
            # could not be reconstructed. It also left the offsets and
            # source_fingerprint below -- both computed against `text` --
            # describing a string no tool ever returned: splicing a rewrite at
            # them into what chain_get_text handed back cut across sentence
            # boundaries, and the drift grows with the document (129 characters
            # at 3 kB). detect() normalises internally, so scores, word counts,
            # labels and assembled documents are unaffected by storing the
            # original here.
            text,
            verdict.score,
            verdict.bucket,
            verdict.label,
            verdict.assessment_word_count,
            verdict.probs,
            note,
            # Default parent is the previous draft; branch_from forks elsewhere.
            branch_from if branch_from is not None else (prev["step_no"] if prev else None),
            register_segment=is_new_segment,
            with_lineage=True,
        )
        step_no, tx_latest, tx_best = step_result
        if is_new_segment:
            segs = store.segments_of(chain_id)
    except sqlite3.IntegrityError as exc:
        # The only foreign key on `steps` is chain_id -> chains(id), so this means
        # the chain was deleted between the lookup above and the insert -- the
        # window is wide because scoring sits inside it. Submitting to an
        # already-deleted chain returns a plain "no such chain"; losing the race
        # must not instead hand the caller "FOREIGN KEY constraint failed", which
        # names neither the chain nor what to do about it.
        if "foreign key" in str(exc).lower():
            return _fail(KeyError(
                f"no such chain: {chain_id} (deleted while this draft was being "
                f"scored; the draft was not stored)"
            ))
        return _fail(exc)
    except (DetectorUnavailable, ValueError, KeyError) as exc:
        return _fail(exc)

    target = float(chain["target_score"])
    # tx_latest/tx_best were read inside the SAME transaction that inserted this
    # step, so they see every draft committed before it -- including ones by
    # concurrent submitters during the seconds this call spent scoring. Computed
    # from the pre-scoring snapshot instead, 12 racing submits each reported
    # is_new_best=true and best_step=<itself>, steering the caller onto the
    # worst draft while the regressed branch below never fired.
    prev_best = tx_best
    is_new_best = prev_best is None or verdict.score < prev_best["score"]
    best_score = min(verdict.score, prev_best["score"]) if prev_best else verdict.score
    # The step NUMBER of the best draft, not just its score. branch_from takes a
    # step number, so a response that reports best_score without best_step tells
    # the caller a better draft exists but not how to reach it -- it has to go
    # back through chain_status or chain_history to recover its own history.
    best_step = step_no if is_new_best else prev_best["step_no"]
    out: dict[str, Any] = {
        "ok": True,
        "chain_id": chain_id,
        "segment": segment,
        "step": step_no,
        "score": round(verdict.score, 4),
        "label": verdict.label,
        "words": verdict.word_count,
        "model_word_count": verdict.model_word_count if verdict.model_word_count is not None else verdict.word_count,
        "assessment_word_count": verdict.assessment_word_count,
        "target_score": target,
        "target_met": verdict.score <= target,
        "best_score": round(best_score, 4),
        "best_step": best_step,
        "is_new_best": is_new_best,
        # delta_vs_previous compares against the true previous step in the
        # segment (in-transaction read); parent_step stays the draft the CALLER
        # was revising -- under a race those are different rows, and stamping a
        # racer's draft as the parent would fabricate lineage between drafts
        # written independently.
        "delta_vs_previous": round(verdict.score - tx_latest["score"], 4) if tx_latest else None,
        "delta_vs_best": round(verdict.score - prev_best["score"], 4) if prev_best else None,
        "parent_step": branch_from if branch_from is not None else (prev["step_no"] if prev else None),
        # Against the draft actually revised; after branch_from this differs
        # from delta_vs_previous, which compares with the segment's latest step.
        "delta_vs_parent": round(verdict.score - parent["score"], 4) if parent else None,
    }
    # The score itself carries a caveat when there is barely any text to score.
    out["reliable"] = verdict.assessment_word_count >= RELIABLE_WORDS
    short_note = _short_text_note(verdict.assessment_word_count)
    if short_note:
        out["reliability_note"] = short_note

    spans: list[dict] = []
    span_meta = {"unit_count": 0, "above_target_total": 0, "min_words_used": span_min_words}
    span_status = "skipped"  # skipped | ok | too_short | failed
    if span_feedback:
        try:
            spans, span_meta = _worst_spans(
                text, target=target, top=span_top, min_words=span_min_words
            )
            span_status = "ok" if spans else "too_short"
        except Exception as exc:  # noqa: BLE001 - never lose the draft over feedback
            span_status = "failed"
            # Surface it: silently returning [] here reads as "nothing to fix",
            # which is the opposite of what a CUDA OOM or a load failure means.
            out["span_error"] = f"{type(exc).__name__}: {exc}"
        out["worst_spans"] = spans
        # Count among the spans actually RETURNED -- next_action says "the N
        # span(s) below", and that phrase has to match the list beneath it.
        out["spans_above_target"] = sum(1 for s in spans if s.get("above_target"))
        # ...and the counts for the whole draft, which is what decides how much
        # work is left. Without these the caller cannot tell a list of 5 spans
        # that IS the whole text from a list of 5 that is a window onto 36.
        out["span_unit_count"] = span_meta["unit_count"]
        out["spans_above_target_total"] = span_meta["above_target_total"]
        out["spans_truncated"] = span_meta["above_target_total"] > out["spans_above_target"]
        out["span_min_words_used"] = span_meta["min_words_used"]
        out["source_fingerprint"] = text_fingerprint(text)

    regressed = prev_best is not None and verdict.score - prev_best["score"] > REGRESSION_DELTA
    out["score_note"] = SCORE_NOTE
    out["target_note"] = TARGET_NOTE
    out["analysis_note"] = SPAN_NOTE
    out["scoring_profile"] = _scoring_profile()
    out.update(length_assessment(verdict.assessment_word_count))
    started = store.started_segments(chain_id)
    out.update(submission_advice(
        out, words=verdict.assessment_word_count, segments=segs,
        history=store.history(chain_id, segment, limit=5), regressed=regressed,
        duplicate=parent is not None and parent["text"] == text,
        unstarted=[s for s in segs if s not in started],
    ))
    return out


@mcp.tool(annotations=_READ_ONLY)
@_guard
def chain_status(
    chain_id: Annotated[str, Field(description="Chain to inspect.")],
) -> dict:
    """Per-segment summary of a chain, and what to do next.

    `pending` is every segment not yet at target, which mixes two situations
    needing opposite responses; `unstarted` and `above_target` split them.
    `target_met` and assembly both use each segment's BEST draft, so a segment
    whose latest draft is worse still counts as met -- `latest_is_best` says
    whether the draft you last submitted is the one that will be assembled.
    """
    try:
        # One snapshot: the per-segment rows, totals and the single-segment
        # progress below must describe the same database revision.
        with store.snapshot():
            chain = store.get(chain_id)
            segs = json.loads(chain["segments"])
            stats = [store.segment_stats(chain_id, s) for s in segs]
            single = None
            if len(segs) == 1 and stats[0]["steps"]:
                single = (store.latest_step(chain_id, segs[0]),
                          store.history(chain_id, segs[0], limit=5))
    except KeyError as exc:
        return _fail(exc)
    target = float(chain["target_score"])
    for s in stats:
        s["target_met"] = s["best_score"] is not None and s["best_score"] <= target
        s["latest_is_best"] = s["steps"] > 0 and s["latest_step"] == s["best_step"]
    unstarted = [s["segment"] for s in stats if s["steps"] == 0]
    above = [s["segment"] for s in stats if s["steps"] > 0 and not s["target_met"]]
    stale = [s["segment"] for s in stats if s["steps"] > 0 and not s["latest_is_best"]]
    out = {
        "ok": True,
        "chain_id": chain_id,
        "name": chain["name"],
        "goal": chain["goal"],
        "target_score": target,
        "segments": stats,
        "total_steps": sum(s["steps"] for s in stats),
        "all_targets_met": all(s["target_met"] for s in stats) if stats else False,
        "pending": [s["segment"] for s in stats if not s["target_met"]],
        "unstarted": unstarted,
        "above_target": above,
        "segments_with_better_earlier_draft": stale,
    }
    out["score_note"] = SCORE_NOTE
    out["target_note"] = TARGET_NOTE
    out["scoring_profile"] = json.loads(chain["meta"] or "{}").get("scoring_profile")
    out["stop_recommended"] = False
    if not stats:
        out["next_action"] = "This chain declares no segments."
    elif unstarted:
        out["next_action"] = (
            f"Draft the segment(s) with no steps yet: {unstarted}. Call chain_submit "
            f"with segment={unstarted[0]!r}."
        )
    elif len(stats) == 1:
        latest, history = single
        progress = revision_progress(history, latest["step_no"])
        out["revision_progress"] = progress
        if latest["words"] < RELIABLE_WORDS:
            out["next_action"] = _short_text_note(latest["words"])
            out["stop_recommended"] = True
        elif progress["plateau"] or progress["revision_budget_reached"]:
            out["next_action"] = "Stop score-driven revisions for review. Inspect saved drafts and choose by accuracy, meaning, and voice."
            out["stop_recommended"] = True
        elif above:
            out["next_action"] = "One segment: review its saved text directly; assembling adds nothing to its content. Review the writing before choosing whether to use chain_submit."
        else:
            out["next_action"] = "The segment meets the workflow target. Stop score-driven edits and review facts, meaning, and voice before using it."
            out["stop_recommended"] = True
    else:
        out["next_action"] = (
            "Call chain_assemble first to assess the complete document. "
            + SCALE_NOTE + " Review the writing before choosing whether to use chain_submit."
        )
    if stale:
        # One instruction PER segment, each naming its segment. A single
        # chain_get_text(step='best') covering a list of segments defaults to
        # segment='main' and reads the wrong one.
        recover = "; ".join(
            f"chain_get_text(chain_id={chain_id!r}, segment={s!r}, step='best')" for s in stale
        )
        out["next_action"] += (
            f" Note: in {stale} your latest draft scores worse than an earlier one; "
            f"assembly will use the earlier one. Retrieve with {recover}."
        )
    return out


@mcp.tool(annotations=_READ_ONLY)
@_guard
def chain_history(
    chain_id: Annotated[str, Field(description="Chain to read.")],
    segment: Annotated[str, Field(description="Segment whose trajectory you want.")] = "main",
    limit: Annotated[int, Field(description="Most recent N steps.", ge=1, le=200)] = 30,
) -> dict:
    """Score trajectory for one segment -- step number, score, and note only.

    Returns the most recent `limit` steps. On a long chain that is a window, not
    the history: `total_steps` and `truncated` say so, and `best_step` /
    `best_score` describe the whole segment, so the best draft stays reachable
    even when it scrolled out of the window. Do not take the lowest score in
    `trajectory` for the best draft -- compare against `best_step`.

    Text is omitted on purpose; use `chain_get_text` for a specific step.
    """
    try:
        known = store.segments_of(chain_id)
    except KeyError as exc:
        return _fail(exc)
    # A mistyped segment used to return an empty trajectory indistinguishable
    # from a real segment with no steps yet.
    if segment not in known:
        return _fail(KeyError(f"no such segment '{segment}' in this chain; have {known}"))
    with store.snapshot():
        rows = store.history(chain_id, segment, limit)
        stats = store.segment_stats(chain_id, segment)
    out = {
        "ok": True,
        "chain_id": chain_id,
        "segment": segment,
        "returned": len(rows),
        # A 65-step segment queried at the default limit returned steps 36..65
        # and called that the trajectory. The best draft was step 4. Nothing in
        # the response said either that steps were missing or where the best one
        # was, so the obvious move -- scan the trajectory, branch from its
        # lowest score -- silently picked the best of the last thirty.
        "total_steps": stats["steps"],
        "truncated": stats["steps"] > len(rows),
        "best_step": stats["best_step"],
        "best_score": stats["best_score"],
        "trajectory": rows,
    }
    shown = {r["step"] for r in rows}
    if stats["best_step"] is not None and stats["best_step"] not in shown:
        # segment= interpolated: both suggested calls default to 'main', and
        # step numbers are per-segment, so the bare form reads/writes another
        # segment in any multi-segment chain.
        out["note"] = (
            f"The best draft of this segment is step {stats['best_step']} "
            f"({stats['best_score']}), which is older than the {len(rows)} step(s) shown. "
            f"Raise `limit` to see it, chain_get_text(chain_id={chain_id!r}, "
            f"segment={segment!r}, step='best') to read it, or "
            f"chain_submit(chain_id={chain_id!r}, segment={segment!r}, "
            f"branch_from={stats['best_step']}, text=...) to continue from it."
        )
    return out


@mcp.tool(annotations=_READ_ONLY)
@_guard
def chain_get_text(
    chain_id: Annotated[str, Field(description="Chain to read from.")],
    segment: Annotated[str, Field(description="Segment to read.")] = "main",
    step: Annotated[
        int | Literal["best", "latest"],
        Field(description="Step number, or 'best' (lowest score) or 'latest'."),
    ] = "best",
) -> dict:
    """Retrieve the stored text for one step, byte for byte as it was submitted.
    Use this to resume work after a restart, or to recover a draft that scored
    better than your current one."""
    try:
        known = store.segments_of(chain_id)
    except KeyError as exc:
        return _fail(exc)
    # Same fix chain_history got: a mistyped segment used to return "no such
    # step in segment 'intro'", which reads as "the step is gone" -- a dead end
    # in the documented regression recovery -- rather than "you typed the
    # segment wrong", with no list of valid names to correct against.
    if segment not in known:
        return _fail(KeyError(f"no such segment '{segment}' in this chain; have {known}"))
    if step == "best":
        row = store.best_step(chain_id, segment)
    elif step == "latest":
        row = store.latest_step(chain_id, segment)
    elif int(step) < 1:
        return _fail(ValueError("step numbers start at 1; use 'best' or 'latest' otherwise"))
    else:
        row = store.get_step(chain_id, segment, int(step))
    if row is None:
        return _fail(KeyError(f"no such step in segment '{segment}'"))
    return {
        "ok": True,
        "chain_id": chain_id,
        "segment": segment,
        "step": row["step_no"],
        "score": round(row["score"], 4),
        "label": row["label"],
        "words": row["words"],
        "note": row["note"],
        "parent_step": row["parent_step"],
        "text": row["text"],
        # This tool is step one of the recovery procedure the instructions
        # describe; without a next_action here the second step (branch_from)
        # lived only in prose the caller may never have seen.
        "next_action": (
            f"Review this saved draft for accuracy, meaning, and voice. If you choose "
            f"to improve it, use chain_submit(chain_id={chain_id!r}, segment={segment!r}, "
            f"branch_from={row['step_no']}, text=...) to record the parent draft."
        ),
    }


@mcp.tool(annotations=_READ_ONLY)
@_guard
def chain_assemble(
    chain_id: Annotated[str, Field(description="Chain to assemble.")],
    separator: Annotated[str, Field(description="Joiner between segments.")] = "\n\n",
    include_text: Annotated[bool, Field(description="Return the assembled document.")] = True,
) -> dict:
    """Join the lowest-scoring saved draft of each segment and score the document.

    Section scores do not bound or explain the assembled score. Inspect the
    selected drafts for accuracy and meaning before using the assembled text.
    """
    try:
        # Every segment's draft from one database revision, so a concurrent
        # delete_segment cannot leave a declared list and drafts that disagree.
        with store.snapshot():
            chain = store.get(chain_id)
            _check_chain_profile(chain)
            segs = json.loads(chain["segments"])
            best_rows = [(s, store.best_step(chain_id, s)) for s in segs]
    except KeyError as exc:
        return _fail(exc)
    parts, missing, per_segment = [], [], []
    for s, row in best_rows:
        if row is None:
            missing.append(s)
            continue
        # Assembly owns the separator. chain_get_text preserves the saved draft
        # exactly; assembly strips only each part's leading/trailing whitespace.
        parts.append(row["text"].strip())
        per_segment.append(
            {"segment": s, "step": row["step_no"], "score": round(row["score"], 4),
             "words": row["words"]}
        )
    if not parts:
        return _fail(ValueError("chain has no steps yet"))

    document = separator.join(parts)
    try:
        verdict, windows = detector.detect(document)
    except (DetectorUnavailable, ValueError) as exc:
        return _fail(exc)

    target = float(chain["target_score"])
    # An incomplete document is not a finished one: scoring only the sections
    # that exist and calling that "target met" invites shipping a draft with a
    # whole section missing.
    complete = not missing
    out: dict[str, Any] = {
        "ok": True,
        "chain_id": chain_id,
        "document_score": round(verdict.score, 4),
        "document_label": verdict.label,
        "words": verdict.word_count,
        "model_word_count": verdict.model_word_count if verdict.model_word_count is not None else verdict.word_count,
        "assessment_word_count": verdict.assessment_word_count,
        "target_score": target,
        "target_met": complete and verdict.score <= target,
        "score_met": verdict.score <= target,
        "complete": complete,
        "per_segment": per_segment,
        "missing_segments": missing,
        "windows": len(windows),
    }
    out["reliable"] = verdict.assessment_word_count >= RELIABLE_WORDS
    short_note = _short_text_note(verdict.assessment_word_count)
    if short_note:
        out["reliability_note"] = short_note
    # chain_status calls these segments pending; this tool may simultaneously
    # report the document as finished. Both are true, and a caller given the two
    # answers with nothing to reconcile them either revises text that is already
    # good or ships while believing it is behind.
    above = [p["segment"] for p in per_segment if p["score"] > target]
    out["segments_above_target"] = above
    if missing:
        out["warning"] = (
            f"{len(missing)} declared segment(s) have no drafts and are absent from this "
            f"document: {missing}. The score describes only what was assembled."
        )
        out["next_action"] = (
            f"Incomplete: draft {missing} with chain_submit, then assemble again. The "
            f"score above covers only the sections that exist."
        )
    elif verdict.score <= target:
        if short_note:
            out["next_action"] = (
                f"Document meets target and every segment has a draft -- BUT "
                f"{short_note} Treat this pass as provisional."
            )
        else:
            out["next_action"] = "Document meets target and every segment has a draft. Stop score-driven edits and review facts, meaning, and voice."
        if above:
            out["next_action"] += (
                f" Segments {above} score above target ON THEIR OWN, and chain_status will "
                f"list them as pending -- ignore that. {SCALE_NOTE} The document score is "
                f"what the workflow threshold compares, and it is below that threshold."
            )
    else:
        # "the assembled text above" only exists when include_text is true;
        # with it false the instruction pointed at a field not in the response,
        # and no other tool returns the assembled document.
        where = (
            "the assembled text above"
            if include_text
            else "the assembled text (re-call chain_assemble with include_text=true to get it)"
        )
        out["next_action"] = (
            f"Document scores {round(verdict.score, 4)}, above target {target}. Review "
            f"{where} for a concrete writing problem. detect_spans can show independent "
            f"fragment estimates; it cannot identify the cause. Preserve sound content "
            f"and stop if further edits would only chase the number."
        )
        if short_note:
            # The met-target branch already carried this; above target is where
            # a caller is most tempted to rewrite short text to move the number.
            out["next_action"] = f"CAUTION: {short_note} " + out["next_action"]
    out["score_note"] = SCORE_NOTE
    out["target_note"] = TARGET_NOTE
    out["scoring_profile"] = _scoring_profile()
    out.update(length_assessment(verdict.assessment_word_count))
    out["stop_recommended"] = complete
    if len(segs) == 1:
        out["note"] = (
            "Single-segment assembly uses the lowest-scoring saved draft, with edge "
            "whitespace stripped. Review that draft directly with chain_get_text."
        )
    if include_text:
        out["text"] = document
    return out


@mcp.tool(annotations=_READ_ONLY)
@_guard
def chain_list(
    limit: Annotated[int, Field(description="Most recent N chains.", ge=1, le=200)] = 25,
) -> dict:
    """List stored chains, most recently updated first."""
    return {"ok": True, "chains": store.list_chains(limit)}


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False})
@_guard
def chain_delete(
    chain_id: Annotated[str, Field(description="Chain to delete permanently.")],
    segment: Annotated[
        str | None,
        Field(description="Delete only this segment (its drafts and its entry in the "
                          "declared list) instead of the whole chain. The escape hatch "
                          "for a typo'd segment name, which otherwise marks the chain "
                          "incomplete forever."),
    ] = None,
) -> dict:
    """Delete a chain and every step it holds -- or, with `segment`, just that
    segment. Either way this cannot be undone."""
    try:
        if segment is not None:
            # A blank segment from an empty form field must not fall through to
            # anything broader; deleting the whole chain means omitting it.
            _require_name(segment, "segment (omit it to delete the whole chain)")
            n, remaining = store.delete_segment(chain_id, segment, with_remaining=True)
            return {
                "ok": True,
                "chain_id": chain_id,
                "segment": segment,
                "deleted_steps": n,
                "remaining_segments": remaining,
            }
        n = store.delete(chain_id)
    except (KeyError, ValueError) as exc:
        return _fail(exc)
    return {"ok": True, "chain_id": chain_id, "deleted_steps": n}


def main() -> None:
    global detector
    # Blank (including whitespace-only) means unset, like every other setting.
    backend = (os.environ.get("EDITLENS_BACKEND") or "").strip().lower() or "shared"
    if backend == "shared":
        from .shared import SharedDetector
        detector = SharedDetector(**DETECTOR_CONFIG)
    elif backend == "local":
        err = warmup_imports()
        if err:
            print(f"[editlens] warning: torch/transformers import failed: {err}", file=sys.stderr)
    else:
        raise ValueError(f"Unknown EDITLENS_BACKEND={backend!r}; use shared or local")
    # `or "stdio"`, not a plain default: an EMPTY value reached fastmcp and died
    # with `ValueError: Unknown transport: ` -- which names nothing -- and empty
    # is what a client config emits for a field the user left blank. Same reason
    # EDITLENS_DEVICE/EDITLENS_DTYPE use `or None` above. A genuinely wrong value
    # still errors, but that message at least quotes what was set.
    #
    # show_banner=False: FastMCP's banner also runs its update check, a
    # blocking request to pypi.org (up to 2 s) on every server launch. This
    # server promises local operation, and the banner itself is stderr noise
    # in every client's MCP log.
    transport = (os.environ.get("EDITLENS_TRANSPORT") or "").strip() or "stdio"
    mcp.run(transport=transport, show_banner=False)


if __name__ == "__main__":
    main()
