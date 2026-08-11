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
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

from .chains import ChainStore, default_db_path
from .detector import (
    DetectorUnavailable,
    EditLensDetector,
    clean_text,
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
        "Every chain tool returns a `next_action` -- follow it rather than inferring a plan "
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
store = ChainStore(default_db_path())

MAX_SPAN_REPORT = 5
# Below this the model's score is indicative rather than precise.
RELIABLE_WORDS = 25
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
) -> list[dict]:
    """Score sentence-groups and return the highest-scoring ones.

    `start`/`end` index the caller's ORIGINAL text, not the normalised copy the
    model sees -- offsets you cannot splice against are worse than no offsets.

    `above_target` marks the spans actually worth rewriting. Without it the
    caller sees five spans under a heading that says "worst" and rewrites all
    five -- including ones the same response labels Human-written, which is how
    a revision loop makes a draft worse while believing it is following
    instructions.
    """
    source, imap = clean_text_with_map(text)
    units, _ = split_units_adaptive(source, granularity=granularity)
    if len(units) < 2:
        return []
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
            "text": source[a:b],
        }
        if target is not None:
            row["above_target"] = v.score > target
        rows.append(row)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top]


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
    info["db_path"] = str(store.path)
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
        return {"ok": True, **store.create(name, target_score, goal, segments)}
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


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

    Read `next_action` first; it accounts for the three cases that are easy to
    get wrong. Rewrite only spans marked `above_target` -- the list is worst-first,
    not a to-do list, and the tail of it is usually text that is already fine.
    History is append-only, so a draft that scores worse costs you nothing:
    `best_step` names the best draft, `chain_get_text` retrieves it and
    `branch_from` continues from it.

    Span scores and the document score are not on one scale -- short fragments
    score higher than the same words in a full document -- so the spans can all
    sit below target while the document stays above it.
    """
    try:
        chain = store.get(chain_id)
        segs = json.loads(chain["segments"])
        is_new_segment = segment not in segs

        prev = store.latest_step(chain_id, segment)
        prev_best = store.best_step(chain_id, segment)
        if branch_from is not None and store.get_step(chain_id, segment, branch_from) is None:
            return _fail(KeyError(f"cannot branch from step {branch_from}: no such step "
                                  f"in segment '{segment}'"))
        # Score BEFORE registering a new segment. Registering first means a
        # failed submit -- empty text, a typo'd segment name, a CUDA OOM -- still
        # commits the segment, after which chain_assemble reports the chain
        # permanently incomplete with no way to remove it again.
        verdict, _ = detector.detect(text)
        if is_new_segment:
            segs = store.add_segment(chain_id, segment)
        step_no = store.add_step(
            chain_id,
            segment,
            clean_text(text),
            verdict.score,
            verdict.bucket,
            verdict.label,
            verdict.word_count,
            verdict.probs,
            note,
            # Default parent is the previous draft; branch_from forks elsewhere.
            branch_from if branch_from is not None else (prev["step_no"] if prev else None),
        )
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
        "delta_vs_previous": round(verdict.score - prev["score"], 4) if prev else None,
        "delta_vs_best": round(verdict.score - prev_best["score"], 4) if prev_best else None,
        "parent_step": branch_from if branch_from is not None else (prev["step_no"] if prev else None),
    }
    spans: list[dict] = []
    span_status = "skipped"  # skipped | ok | too_short | failed
    if span_feedback:
        try:
            spans = _worst_spans(text, target=target)
            span_status = "ok" if spans else "too_short"
        except Exception as exc:  # noqa: BLE001 - never lose the draft over feedback
            span_status = "failed"
            # Surface it: silently returning [] here reads as "nothing to fix",
            # which is the opposite of what a CUDA OOM or a load failure means.
            out["span_error"] = f"{type(exc).__name__}: {exc}"
        out["worst_spans"] = spans
        out["spans_above_target"] = sum(1 for s in spans if s.get("above_target"))
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
        out["next_action"] = "Target met. Stop, or call chain_assemble if other segments remain."
    elif regressed:
        out["next_action"] = (
            f"Worse than step {best_step} ({round(prev_best['score'], 4)} vs "
            f"{round(verdict.score, 4)}). Do not keep editing this draft. Call "
            f"chain_get_text(step={best_step}) to recover the better one, then submit your "
            f"next attempt with branch_from={best_step}. Nothing is lost -- this draft "
            f"stays in the history as step {step_no}."
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
    elif span_status == "ok":
        out["next_action"] = (
            f"Rewrite the {out['spans_above_target']} span(s) below marked above_target=true "
            f"in your own voice, then call chain_submit again. Leave the rest alone -- they "
            f"are already at or below target."
        )
    elif span_status == "too_short":
        # One sentence has nothing to rank against.
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
    elif above:
        out["next_action"] = (
            f"Call chain_assemble first: {SCALE_NOTE} The assembled document often "
            f"scores below every section that failed on its own, and the document score "
            f"is what decides completion. Only if the document is still above target, "
            f"revise {above} with chain_submit."
        )
    else:
        out["next_action"] = (
            "Every segment meets target. Call chain_assemble to score the whole "
            "document -- assembly is what decides completion, not these per-segment scores."
        )
    if stale:
        out["next_action"] += (
            f" Note: in {stale} your latest draft scores worse than an earlier one; "
            f"assembly will use the earlier one. chain_get_text(step='best') retrieves it."
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
    return {
        "ok": True,
        "chain_id": chain_id,
        "segment": segment,
        "returned": len(rows),
        "trajectory": rows,
    }


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
    """Retrieve the stored text for one step. Use this to resume work after a
    restart, or to recover a draft that scored better than your current one."""
    try:
        store.get(chain_id)
    except KeyError as exc:
        return _fail(exc)
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
        parts.append(row["text"])
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
        out["next_action"] = "Document meets target and every segment has a draft. Done."
        if above:
            out["next_action"] += (
                f" Segments {above} score above target ON THEIR OWN, and chain_status will "
                f"list them as pending -- ignore that. {SCALE_NOTE} The document score is "
                f"what decides completion, and it passed."
            )
    else:
        worst = max(per_segment, key=lambda p: p["score"])["segment"]
        out["next_action"] = (
            f"Document scores {round(verdict.score, 4)}, above the target {target}. Call "
            f"detect_spans on the assembled text above to find which passages drive it -- "
            f"section scores are a poor guide here. Otherwise revise segment "
            f"'{worst}' with chain_submit and assemble again."
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
) -> dict:
    """Delete a chain and every step it holds. This cannot be undone."""
    try:
        n = store.delete(chain_id)
    except KeyError as exc:
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
