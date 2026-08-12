"""EditLens MCP server.

Exposes a local AI-text detector to an MCP client, plus a durable chain store so
a model can run long write -> score -> revise loops without carrying the history
in its context window.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

from .chains import ChainStore, default_db_path
from .detector import (
    DetectorUnavailable,
    EditLensDetector,
    clean_text_with_map,
    count_words,
    map_span,
    split_units_adaptive,
    text_fingerprint,
    warmup_imports,
)

mcp = FastMCP(
    name="editlens",
    instructions=(
        "Local AI-text detector (pangram/editlens_roberta-large). `detect` scores one text "
        "0.0 (human-written) to 1.0 (fully AI-generated). `detect_spans` shows which sentences "
        "drive that score. For iterative work, open a chain with `chain_create`, then call "
        "`chain_submit` after every draft: it stores the text on disk and returns only the score "
        "delta plus the worst spans, so a chain can run for hundreds of steps cheaply. "
        "Use `chain_assemble` to score the concatenation of a multi-section chain. "
        "chain_create, chain_submit, chain_status, chain_get_text and chain_assemble return "
        "a `next_action` -- follow it rather than inferring a plan "
        "from the scores, because two things about those scores are counter-intuitive. "
        "First, scores are not comparable across lengths: the model scores a sentence or a "
        "section harder than the same words inside a whole document, so parts routinely score "
        "far above the document they compose, and the document score is the one that decides "
        "whether you are finished. Second, chain history is append-only and nothing is ever "
        "overwritten, so when a revision scores worse the fix is to go back -- "
        "`chain_get_text(step='best')` to read the better draft, then `chain_submit` with "
        "`branch_from` set to that step -- not to keep editing the worse one."
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
        return cast(raw)
    except (TypeError, ValueError):
        print(
            f"[editlens] warning: ignoring {name}={raw!r} (not a number); using {default}",
            file=sys.stderr,
        )
        return default


detector = EditLensDetector(
    device=os.environ.get("EDITLENS_DEVICE") or None,
    batch_size=_env_number("EDITLENS_BATCH_SIZE", 8, int),
    dtype=os.environ.get("EDITLENS_DTYPE") or None,
    idle_unload_seconds=_env_number("EDITLENS_IDLE_UNLOAD", 300.0, float),
)
# default_db_path() re-reads EDITLENS_DB and treats an empty value as unset.
# Passing os.environ.get(...) here instead handed it "" and crashed at import.
#
# The construction is guarded because this line is the only I/O at import and
# was the only statement in the file outside _guard's protection: an unwritable
# directory, a path component that is a file, or a chains.db that arrived
# half-copied ("file is not a database") killed the process before FastMCP
# registered a single tool -- including detector_info, whose docstring says to
# call it first when things break. Now the 12 chain/detect tools fail per-call
# with the real reason, and detector_info stays alive to report it.
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
    path = default_db_path()
    try:
        return ChainStore(path), None
    except Exception as exc:  # noqa: BLE001 - import-time boundary
        msg = (
            f"chain store failed to open: {type(exc).__name__}: {exc} "
            f"(EDITLENS_DB={os.environ.get('EDITLENS_DB')!r} resolved to '{path}'). "
            f"detect/detect_batch/detect_spans and the chain tools need this "
            f"database; fix the path and restart the server."
        )
        print(f"[editlens] warning: {msg}", file=sys.stderr)
        return _BrokenStore(msg, path), msg


store, STORE_ERROR = _open_store()

MAX_SPAN_REPORT = 5
# Merge threshold for span feedback, matching detect_spans' own default.
SPAN_MIN_WORDS = 25
# Below this the model's score is indicative rather than precise.
RELIABLE_WORDS = 25
# ...and below this it is still noisy enough to be worth a second opinion.
# Measured by cutting five known-human passages to fixed lengths and scoring
# each prefix: the spread across those five texts was 0.30 at 10 words, 0.63 at
# 15, 0.37 at 20, then 0.24 at 30, 0.19 at 40, 0.12 at 50 and 0.06 by 60. One of
# the five crossed a 0.25 target on nothing but length. So a score on a short
# draft is not evidence that the draft needs rewriting.
NOISY_WORDS = 60
# How much worse than the best draft counts as a regression worth recovering
# from, rather than run-to-run noise.
REGRESSION_DELTA = 0.05

# The detector is a document-level classifier, and it scores short text harder
# than the same prose inside a longer document. Both directions were observed
# driving this server for real: a 96-word paragraph scored 0.299 while every one
# of its sentence-groups scored 0.20 or below, and a four-section report
# assembled to 0.108 from sections scoring up to 0.733. So a caller cannot
# compare a fragment's score to a document's, and every place this server hands
# back both numbers has to say which one decides.
SCALE_NOTE = (
    "Scores are not comparable across lengths: the model scores short fragments "
    "harder than the same words inside a full document."
)


def _short_text_note(words: int) -> str | None:
    """The caveat that belongs on any score computed from very little text.

    Genuinely human passages cut to 15 words scored anywhere from 0.06 to 0.69
    on this model. A caller handed 0.46 for a 26-word note, with no indication
    of that spread, rewrites prose that was never the problem.
    """
    if words >= NOISY_WORDS:
        return None
    if words < RELIABLE_WORDS:
        return (
            f"Only {words} words: below ~{RELIABLE_WORDS} this score is indicative, not a "
            f"measurement -- known-human passages this short have scored anywhere from "
            f"0.02 to 0.69. Do not rewrite prose to move it. Score the text in its full "
            f"context instead."
        )
    return (
        f"Only {words} words: under ~{NOISY_WORDS} this score is noisy (human passages of "
        f"this length vary by ~0.2 on nothing but length). Prefer the score of the whole "
        f"document over this one."
    )


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

    `above_target` marks the spans actually worth rewriting. Without it the
    caller sees five spans under a heading that says "worst" and rewrites all
    five -- including ones the same response labels Human-written, which is how
    a revision loop makes a draft worse while believing it is following
    instructions.

    The second return value describes the WHOLE text, not the `top` slice of it.
    Reporting only the slice is how a caller ends up being told that 31 units it
    never saw are "already at or below target": on a 1548-word document all 36
    units scored above a 0.25 target, and the response named 5.
    """
    source, imap = clean_text_with_map(text)
    units, used = split_units_adaptive(
        source, granularity=granularity, min_words=min_words
    )
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
            # The same per-unit caveat detect_spans already carries. A 9-word
            # unit scored 0.99 is not the same evidence as a 40-word one.
            "reliable": v.word_count >= RELIABLE_WORDS,
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


@mcp.tool
@_guard
def detector_info() -> dict:
    """Report detector status: checkpoint, device, dtype, bucket labels, whether the
    model is loaded yet, and whether an HF token is visible. Call this first if
    anything errors."""
    # "ok" on the success path too: _guard supplies it on failure, and a client
    # branching on result["ok"] should not KeyError when nothing went wrong.
    info = {"ok": True, **detector.info()}
    # Resolved, not the configured string: a relative EDITLENS_DB used to be
    # echoed back verbatim ("chains.db"), which names no location on disk.
    info["db_path"] = str(Path(store.path).resolve())
    info["db_path_configured"] = os.environ.get("EDITLENS_DB")
    if STORE_ERROR is not None:
        # This tool's whole job is answering "why is everything broken?" --
        # so it reports a dead chain store rather than dying of one.
        info["db_error"] = STORE_ERROR
        info["duplicate_steps_present"] = None
    else:
        info["duplicate_steps_present"] = store.duplicate_steps_present
    return info


@mcp.tool
@_guard
def detector_unload() -> dict:
    """Release the model from GPU memory immediately.

    Rarely needed -- the model unloads itself after a few minutes idle and
    reloads automatically on the next call. Use this only to free VRAM right now,
    e.g. before starting a game or another GPU job.
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


@mcp.tool
@_guard
def detect(
    text: Annotated[str, Field(description="The text to score.")],
    include_windows: Annotated[
        bool, Field(description="Also return per-window scores for long inputs.")
    ] = False,
) -> dict:
    """Score one text for AI authorship.

    Returns `score` in [0,1] -- 0.0 = human-written, 1.0 = fully AI-generated --
    along with the discrete bucket label and the full probability distribution.
    Inputs longer than the model's 512-token window are split into overlapping
    windows and combined by word-count-weighted average.
    """
    try:
        source, ranges = clean_text_with_map(text)
        if not source.strip():
            return _fail(ValueError("empty text"))
        verdict, windows = detector.detect(source, normalise=False)
    except (DetectorUnavailable, ValueError) as exc:
        return _fail(exc)
    out: dict[str, Any] = {"ok": True, **verdict.as_dict()}
    out["target_hint"] = "lower is more human-like"
    # detect_spans has flagged short units as unreliable since it was written;
    # `detect` reported a bare 4-decimal score for a 26-word input and said
    # nothing. That is the number a caller acts on, so it is the one that most
    # needs the caveat.
    out["reliable"] = verdict.word_count >= RELIABLE_WORDS
    note = _short_text_note(verdict.word_count)
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


@mcp.tool
@_guard
def detect_batch(
    texts: Annotated[list[str], Field(description="Texts to score in a single pass.")],
) -> dict:
    """Score many texts at once in one batched forward pass.

    Use this to compare candidate drafts side by side -- generate N variants,
    score them together, keep the lowest.

    Treat these scores as a RANKING, not as a verdict. When the candidates are
    sentences or sections rather than whole documents, all of them will score
    higher than the document they end up in, so do not discard a whole batch for
    missing a document-level target -- pick the lowest, splice it in, and score
    the result with `detect`.
    """
    if not texts:
        return _fail(ValueError("texts is empty"))
    try:
        verdicts = detector.detect_many(texts)
    except (DetectorUnavailable, ValueError) as exc:
        return _fail(exc)
    results = [{"index": i, **v.as_dict()} for i, v in enumerate(verdicts)]
    best = min(results, key=lambda r: r["score"])
    return {
        "ok": True,
        "results": results,
        "best_index": best["index"],
        "best_score": best["score"],
        "mean_score": round(sum(r["score"] for r in results) / len(results), 4),
        # Observed: three candidate paragraphs scored 0.50/0.99/0.66, and the
        # best one spliced into its document took that document to 0.06. A
        # caller comparing best_score against its target rejects all three.
        "comparison_note": (
            f"Use these to rank the candidates against each other. {SCALE_NOTE} "
            f"Re-score with `detect` after splicing the winner in."
        ),
    }


@mcp.tool
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
    """Localise the score: split the text and score each unit separately.

    Returns units sorted worst-first, so you know which passages to rewrite
    rather than redrafting the whole thing. Short sentences are merged toward
    `min_words` because the model is unreliable on very short inputs; if that
    would leave the whole text as one unit, the threshold is relaxed
    automatically and `min_words_used` reports what it settled on. Treat units
    under ~25 words as indicative rather than precise.

    `start`/`end` are offsets into the text YOU passed in, so you can splice a
    rewrite straight back. `text` is the normalised form the model scored
    (whitespace collapsed), which may differ from that slice.
    """
    try:
        source, imap = clean_text_with_map(text)
        units, used = split_units_adaptive(
            source, granularity=granularity, min_words=min_words
        )
        if not units:
            return _fail(ValueError("no scoreable units"))
        verdicts = detector.detect_many([source[a:b] for a, b in units], normalise=False)
        overall, _ = detector.detect(source, normalise=False)
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
                # The model is a document-level classifier; below ~25 words its
                # score is indicative at best. Say so per unit rather than
                # letting a 1-word paragraph look as solid as a paragraph.
                "reliable": v.word_count >= RELIABLE_WORDS,
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
    return out


# ------------------------------------------------------------------------ chains


@mcp.tool
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
        created = store.create(name, target_score, goal, segments)
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)
    first = created["segments"][0] if created.get("segments") else "main"
    return {
        "ok": True,
        **created,
        "next_action": (
            f"Draft segment '{first}' and call chain_submit(segment='{first}')."
        ),
    }


@mcp.tool
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
                          "alternative from an earlier draft instead of the latest one."),
    ] = None,
) -> dict:
    """Submit a draft: score it, score its parts, store it, and say what to do next.

    The response is deliberately compact -- score, movement against the previous
    and best steps, target status, and the worst spans. The draft text itself is
    kept on disk, not echoed back, so you can iterate indefinitely.

    Read `next_action` first; it accounts for the cases that are easy to get
    wrong. Rewrite only spans marked `above_target` -- the list is worst-first,
    not a to-do list, and its tail is often text that is already fine. But check
    `spans_truncated`: `worst_spans` holds at most `span_top` entries, and
    `spans_above_target_total` counts every unit above target, including the ones
    not shown. History is append-only, so a draft that scores worse costs you
    nothing: `best_step` names the best draft, `chain_get_text` retrieves it and
    `branch_from` continues from it.

    On a draft of more than a few hundred words, tune `span_min_words`. It sets
    how much text each span covers, which is how much text you have to rewrite to
    act on one. Driving a 1548-word document to a 0.25 target took 4 rounds and
    rewrote 56% of it at the default 25; at `span_min_words=15` the same loop
    took 2 rounds and rewrote 15%. Smaller spans score more noisily -- `reliable`
    is false below ~25 words -- but they let you replace the sentences that
    actually score badly instead of the paragraphs containing them.

    Span scores and the document score are not on one scale -- short fragments
    score higher than the same words in a full document -- so the spans can all
    sit below target while the document stays above it.
    """
    try:
        chain = store.get(chain_id)
        segs = json.loads(chain["segments"])
        is_new_segment = segment not in segs

        # The caller-visible previous draft: what the agent was actually
        # revising when it built this submission. Used ONLY for parent_step --
        # everything comparative (is_new_best, best_step, the deltas) is
        # recomputed inside add_step's transaction below, because these
        # pre-scoring reads are seconds stale by insert time and under
        # concurrent submits every racer would otherwise crown itself the best.
        prev = store.latest_step(chain_id, segment)
        if branch_from is not None and store.get_step(chain_id, segment, branch_from) is None:
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
            verdict.word_count,
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
    }
    # The score itself carries a caveat when there is barely any text to score.
    out["reliable"] = verdict.word_count >= RELIABLE_WORDS
    short_note = _short_text_note(verdict.word_count)
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

    # A draft that lost ground against the best one is the case where "keep
    # revising" is the wrong instruction: the caller already holds a better
    # draft and every further edit compounds off the worse one. Recovering means
    # chain_get_text + branch_from, and neither appears anywhere in a response
    # unless it is said here -- observed for real, where a step went 0.035 ->
    # 0.999 and next_action still just said "rewrite the worst spans".
    regressed = (
        prev_best is not None
        and not is_new_best
        and (
            verdict.score - prev_best["score"] > REGRESSION_DELTA
            or (prev_best["score"] <= target < verdict.score)
        )
    )

    if verdict.score <= target:
        if short_note:
            # A short draft passes on the same noise that fails one: known-human
            # passages under 25 words scored 0.02..0.69 on nothing but length.
            # The old branch said a bare "Stop" here, declaring a chain finished
            # on a score the SAME response flagged unreliable. False completion
            # is the worse direction of that error.
            out["next_action"] = (
                f"Target met, BUT {short_note} Do not treat this as a pass on its "
                f"own -- score it inside the full document (chain_assemble, or "
                f"detect on the whole piece) before stopping."
            )
        else:
            out["next_action"] = (
                "Target met. Stop, or call chain_assemble if other segments remain."
            )
    elif regressed:
        # segment= is interpolated into BOTH calls. Without it they fall back
        # to segment='main', and step numbers are per-segment: in a
        # multi-segment chain, following the bare instruction verbatim silently
        # returned another segment's draft and filed the rewrite under it, all
        # with ok=True.
        out["next_action"] = (
            f"Worse than step {best_step} ({round(prev_best['score'], 4)} vs "
            f"{round(verdict.score, 4)}). Do not keep editing this draft. Call "
            f"chain_get_text(segment='{segment}', step={best_step}) to recover the better "
            f"one, then submit your next attempt with segment='{segment}', "
            f"branch_from={best_step}. Nothing is lost -- this draft stays in the "
            f"history as step {step_no}."
        )
    elif span_status == "ok" and out["spans_above_target"] == 0:
        # Every span is already at or below target and the document is not. More
        # span-hunting cannot help; the caller has to act on the whole passage.
        # Left as "rewrite the worst spans", this is an infinite loop that asks a
        # model to rewrite sentences the same response calls Human-written.
        out["next_action"] = (
            f"No span scores above the target -- span-level rewriting has bottomed out, so "
            f"do not rewrite the spans below. {SCALE_NOTE} Whatever is left is document-level: "
            f"try reordering, cutting the opening or closing sentence, varying sentence "
            f"length, or rewriting the passage from scratch. Then call chain_submit again."
        )
    elif span_status == "ok" and out["spans_above_target_total"] == out["span_unit_count"]:
        # Tested BEFORE the truncated branch, and on the whole-draft totals
        # rather than the returned slice: when every unit is failing, whether
        # the caller was told "patch these 5" or "rewrite the whole thing" used
        # to depend only on span_top -- two mutually exclusive strategies
        # selected by a display parameter.
        out["next_action"] = (
            f"Every one of the {out['span_unit_count']} units in this draft is above target, "
            f"so there is nothing here to preserve on score grounds -- rewrite the whole "
            f"passage in your own voice rather than patching spans, then call chain_submit "
            f"again. The spans below are ranked worst-first if you want somewhere to start."
        )
    elif span_status == "ok" and out["spans_truncated"]:
        # "Leave the rest alone -- they are already at or below target" was a
        # flat falsehood here. On a 1548-word draft every one of 36 units scored
        # above target and the response named 5, so that sentence described 31
        # failing units as acceptable. A caller acting on it under-fixes the
        # document on every round and is told it has finished the job each time.
        hidden = out["spans_above_target_total"] - out["spans_above_target"]
        out["next_action"] = (
            f"Rewrite the {out['spans_above_target']} span(s) below marked above_target=true "
            f"in your own voice, then call chain_submit again. These are the worst of "
            f"{out['spans_above_target_total']} units above target out of "
            f"{out['span_unit_count']} in this draft -- the other {hidden} are NOT shown and "
            f"are NOT fine. Raise span_top to see more of them per round, or lower "
            f"span_min_words for smaller spans, so each rewrite replaces less text."
        )
    elif span_status == "ok":
        keep = out["span_unit_count"] - out["spans_above_target_total"]
        out["next_action"] = (
            f"Rewrite the {out['spans_above_target']} span(s) below marked above_target=true "
            f"in your own voice, then call chain_submit again. Leave the other {keep} "
            f"unit(s) alone -- they are already at or below target."
        )
    elif span_status == "too_short":
        # One unit has nothing to rank against -- but say WHY there is one
        # unit. This branch used to claim "too short" unconditionally, and the
        # split failing on formatting (a bullet list, a table, verse) is not
        # shortness: a 950-word draft was told it was too short to analyse
        # while three numeric fields said zero units were above target, which
        # reads as "nothing to fix". With the line-boundary fallback in
        # split_units this now fires mostly on genuinely short text, but the
        # single-long-line case still exists.
        if verdict.word_count >= RELIABLE_WORDS:
            out["next_action"] = (
                "Could not divide this draft into rankable spans (no sentence "
                "boundaries or line breaks found -- the document score above is "
                "still valid). Rewrite the whole passage in your own voice, or "
                "add paragraph breaks and resubmit to get span-level feedback."
            )
        else:
            out["next_action"] = (
                "Too short to pinpoint spans. Rewrite the whole passage in your own voice, "
                "then call chain_submit again."
            )
    elif span_status == "failed":
        out["next_action"] = (
            "Span analysis failed (see span_error); the score above is still valid. "
            "Revise and call chain_submit again."
        )
    else:
        out["next_action"] = (
            "Revise and call chain_submit again. Pass span_feedback=true, or call "
            "detect_spans, to see which passages score worst."
        )
    # In a multi-segment chain, every failing branch above says "revise and
    # resubmit" -- and an agent following that literally grinds each section to
    # target in isolation, dozens of rewrites, when the assembled document
    # would already have passed: sections routinely score far above the
    # document they compose. Only chain_assemble's docstring said so, and an
    # agent revising in a submit loop never has a reason to read it. Skipped
    # when the short-text CAUTION below fires, which gives the same advice.
    if len(segs) > 1 and verdict.score > target and not regressed and not short_note:
        out["next_action"] += (
            f" This is one of {len(segs)} sections, and section scores run high in "
            f"isolation. Once every section has a draft, call chain_assemble before "
            f"grinding this one further -- the document score decides completion."
        )
    # Every branch above except "target met" tells the caller to rewrite something.
    # On a draft this short that instruction is built on a number that moves by
    # more than the target itself between human passages of the same length, so
    # the caveat has to come FIRST -- a caller that reads only next_action would
    # otherwise never see it. Observed: a genuinely human 26-word note scored
    # 0.4579 and was told to rewrite the offending span.
    if short_note and verdict.score > target:
        out["next_action"] = (
            f"CAUTION: {short_note} If this draft is a section of something longer, "
            f"finish the other sections and judge it with chain_assemble instead of "
            f"revising it against this score. Otherwise: {out['next_action']}"
        )
    return out


@mcp.tool
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
        chain = store.get(chain_id)
    except KeyError as exc:
        return _fail(exc)
    segs = json.loads(chain["segments"])
    target = float(chain["target_score"])
    stats = [store.segment_stats(chain_id, s) for s in segs]
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
    # Without this a caller reads `pending` and starts revising, when the thing
    # to do first is usually to assemble: a whole document routinely scores far
    # below its own sections, so sections listed here as failing can already be
    # good enough and the revision is wasted.
    if not stats:
        out["next_action"] = "This chain declares no segments."
    elif unstarted:
        out["next_action"] = (
            f"Draft the segment(s) with no steps yet: {unstarted}. Call chain_submit "
            f"with segment='{unstarted[0]}'."
        )
    elif above and len(stats) == 1:
        # One segment: the assembled document IS this segment's best draft, so
        # the assemble-first advice below would send the caller on a
        # guaranteed-useless round trip -- the score is arithmetically the
        # best_score already shown here -- while promising a change that cannot
        # happen.
        out["next_action"] = (
            f"One segment: its score IS the document score, so assembling adds "
            f"nothing. Revise '{stats[0]['segment']}' with chain_submit."
        )
    elif above:
        out["next_action"] = (
            f"Call chain_assemble first: {SCALE_NOTE} The assembled document often "
            f"scores below every section that failed on its own, and the document score "
            f"is what decides completion. Only if the document is still above target, "
            f"revise {above} with chain_submit."
        )
    elif len(stats) == 1:
        out["next_action"] = (
            "The segment meets target, and with one segment its score IS the "
            "document score. Done -- no assemble needed."
        )
    else:
        out["next_action"] = (
            "Every segment meets target. Call chain_assemble to score the whole "
            "document -- assembly is what decides completion, not these per-segment scores."
        )
    if stale:
        # One instruction PER segment, each naming its segment. A single
        # chain_get_text(step='best') covering a list of segments defaults to
        # segment='main' and reads the wrong one.
        recover = "; ".join(
            f"chain_get_text(segment='{s}', step='best')" for s in stale
        )
        out["next_action"] += (
            f" Note: in {stale} your latest draft scores worse than an earlier one; "
            f"assembly will use the earlier one. Retrieve with {recover}."
        )
    return out


@mcp.tool
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
            f"Raise `limit` to see it, chain_get_text(segment='{segment}', step='best') "
            f"to read it, or chain_submit(segment='{segment}', "
            f"branch_from={stats['best_step']}) to continue from it."
        )
    return out


@mcp.tool
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
        "text": row["text"],
        # This tool is step one of the recovery procedure the instructions
        # describe; without a next_action here the second step (branch_from)
        # lived only in prose the caller may never have seen.
        "next_action": (
            f"Revise this text, then chain_submit(segment='{segment}', "
            f"branch_from={row['step_no']}) so the lineage records what you "
            f"built on."
        ),
    }


@mcp.tool
@_guard
def chain_assemble(
    chain_id: Annotated[str, Field(description="Chain to assemble.")],
    separator: Annotated[str, Field(description="Joiner between segments.")] = "\n\n",
    include_text: Annotated[bool, Field(description="Return the assembled document.")] = True,
) -> dict:
    """Join the best draft of every segment in order and score the whole document.

    Always assemble before calling a multi-segment chain finished: the document
    score is the one that counts, and it is NOT bounded by the section scores in
    either direction. A document can come out above every section it is made of,
    and it can come out far below them -- sections scoring 0.73 and 0.51 have
    assembled into a 0.11 document, because the model scores short text harder
    than the same words inside a longer piece. So assemble before you decide a
    section needs more work, not after.
    """
    try:
        chain = store.get(chain_id)
    except KeyError as exc:
        return _fail(exc)
    segs = json.loads(chain["segments"])
    parts, missing, per_segment = [], [], []
    for s in segs:
        row = store.best_step(chain_id, s)
        if row is None:
            missing.append(s)
            continue
        # Strip each part's own edge whitespace: the separator decides what goes
        # between sections, not whatever blank lines a draft happened to end on.
        # This is also what keeps the document score stable now that drafts are
        # stored verbatim. clean_text collapses an indent run to ONE space rather
        # than dropping it, so a segment whose first line is indented -- a code
        # block, a nested bullet -- used to reach the model dedented and now
        # reaches it with a leading space. Measured on the real checkpoint, that
        # one character moved a 314-word document from 0.4881 to 0.6356. Stripping
        # here reproduces the pre-change model input exactly (delta +0.0000 across
        # every fixture and separator tested). Fidelity is chain_get_text's job,
        # and it still returns the draft byte for byte.
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
        "target_score": target,
        "target_met": complete and verdict.score <= target,
        "score_met": verdict.score <= target,
        "complete": complete,
        "per_segment": per_segment,
        "missing_segments": missing,
        "windows": len(windows),
    }
    # The single most consequential instruction in the server is this tool's
    # "Done." -- so it carries the same short-text caveat detect and
    # chain_submit already carry. A 40-word assembled blurb can pass (or fail)
    # the target on nothing but length noise, and this response used to say
    # "Done." with no hint of the +/-0.3 spread chain_submit would have
    # attached to the very same words.
    out["reliable"] = verdict.word_count >= RELIABLE_WORDS
    short_note = _short_text_note(verdict.word_count)
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
            out["next_action"] = "Document meets target and every segment has a draft. Done."
        if above:
            out["next_action"] += (
                f" Segments {above} score above target ON THEIR OWN, and chain_status will "
                f"list them as pending -- ignore that. {SCALE_NOTE} The document score is "
                f"what decides completion, and it passed."
            )
    else:
        worst = max(per_segment, key=lambda p: p["score"])["segment"]
        # "the assembled text above" only exists when include_text is true;
        # with it false the instruction pointed at a field not in the response,
        # and no other tool returns the assembled document.
        where = (
            "the assembled text above"
            if include_text
            else "the assembled text (re-call chain_assemble with include_text=true to get it)"
        )
        out["next_action"] = (
            f"Document scores {round(verdict.score, 4)}, above the target {target}. Call "
            f"detect_spans on {where} to find which passages drive it -- "
            f"section scores are a poor guide here. Otherwise revise segment "
            f"'{worst}' with chain_submit(segment='{worst}') and assemble again."
        )
    if len(segs) == 1:
        # For a single-segment chain this whole call is a round trip to the
        # segment's own best score; say so, so a driving agent stops scheduling
        # assemble passes that cannot tell it anything new.
        out["note"] = (
            "Single-segment chain: the assembled score is the segment's best "
            "score by construction. chain_submit's feedback is all there is."
        )
    if include_text:
        out["text"] = document
    return out


@mcp.tool
@_guard
def chain_list(
    limit: Annotated[int, Field(description="Most recent N chains.", ge=1, le=200)] = 25,
) -> dict:
    """List stored chains, most recently updated first."""
    return {"ok": True, "chains": store.list_chains(limit)}


@mcp.tool
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
            n = store.delete_segment(chain_id, segment)
            return {
                "ok": True,
                "chain_id": chain_id,
                "segment": segment,
                "deleted_steps": n,
                "remaining_segments": store.segments_of(chain_id),
            }
        n = store.delete(chain_id)
    except (KeyError, ValueError) as exc:
        return _fail(exc)
    return {"ok": True, "chain_id": chain_id, "deleted_steps": n}


def main() -> None:
    # Must happen on the main thread before any tool call -- see warmup_imports().
    err = warmup_imports()
    if err:
        print(f"[editlens] warning: torch/transformers import failed: {err}", file=sys.stderr)
    # `or "stdio"`, not a plain default: an EMPTY value reached fastmcp and died
    # with `ValueError: Unknown transport: ` -- which names nothing -- and empty
    # is what a client config emits for a field the user left blank. Same reason
    # EDITLENS_DEVICE/EDITLENS_DTYPE use `or None` above. A genuinely wrong value
    # still errors, but that message at least quotes what was set.
    mcp.run(transport=os.environ.get("EDITLENS_TRANSPORT") or "stdio")


if __name__ == "__main__":
    main()
