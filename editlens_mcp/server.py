"""EditLens MCP server.

Exposes a local AI-text detector to an MCP client, plus a durable chain store so
a model can run long write -> score -> revise loops without carrying the history
in its context window.
"""

from __future__ import annotations

import json
import os
import sys
from functools import wraps
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

from .chains import DEFAULT_DB, ChainStore
from .detector import (
    DetectorUnavailable,
    EditLensDetector,
    clean_text,
    clean_text_with_map,
    count_words,
    map_span,
    split_units_adaptive,
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
        "Use `chain_assemble` to score the concatenation of a multi-section chain."
    ),
)

detector = EditLensDetector(
    device=os.environ.get("EDITLENS_DEVICE") or None,
    batch_size=int(os.environ.get("EDITLENS_BATCH_SIZE", "8")),
    dtype=os.environ.get("EDITLENS_DTYPE") or None,
    idle_unload_seconds=float(os.environ.get("EDITLENS_IDLE_UNLOAD", "300")),
)
store = ChainStore(os.environ.get("EDITLENS_DB", DEFAULT_DB))

MAX_SPAN_REPORT = 5
# Below this the model's score is indicative rather than precise.
RELIABLE_WORDS = 25


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


def _worst_spans(text: str, granularity: str = "sentence", top: int = MAX_SPAN_REPORT) -> list[dict]:
    """Score sentence-groups and return the highest-scoring ones.

    `start`/`end` index the caller's ORIGINAL text, not the normalised copy the
    model sees -- offsets you cannot splice against are worse than no offsets.
    """
    source, imap = clean_text_with_map(text)
    units, _ = split_units_adaptive(source, granularity=granularity)
    if len(units) < 2:
        return []
    verdicts = detector.detect_many([source[a:b] for a, b in units], normalise=False)
    rows = []
    for (a, b), v in zip(units, verdicts):
        o0, o1 = map_span(imap, a, b, len(text))
        rows.append(
            {
                "start": o0,
                "end": o1,
                "score": round(v.score, 4),
                "label": v.label,
                "words": v.word_count,
                "text": source[a:b],
            }
        )
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
        verdict, windows = detector.detect(text)
    except (DetectorUnavailable, ValueError) as exc:
        return _fail(exc)
    out: dict[str, Any] = {"ok": True, **verdict.as_dict()}
    out["target_hint"] = "lower is more human-like"
    if include_windows and len(windows) > 1:
        out["window_detail"] = windows
    return out


@mcp.tool
@_guard
def detect_batch(
    texts: Annotated[list[str], Field(description="Texts to score in a single pass.")],
) -> dict:
    """Score many texts at once in one batched forward pass.

    Use this to compare candidate drafts side by side -- generate N variants,
    score them together, keep the lowest.
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
) -> dict:
    """Submit a draft: score it, store it, and get back what to fix.

    The response is deliberately compact -- score, movement against the previous
    and best steps, target status, and the worst spans. The draft text itself is
    kept on disk, not echoed back, so you can iterate indefinitely.
    """
    try:
        chain = store.get(chain_id)
        segs = json.loads(chain["segments"])
        is_new_segment = segment not in segs

        prev = store.latest_step(chain_id, segment)
        prev_best = store.best_step(chain_id, segment)
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
        )
    except (DetectorUnavailable, ValueError, KeyError) as exc:
        return _fail(exc)

    target = float(chain["target_score"])
    best_score = min(verdict.score, prev_best["score"]) if prev_best else verdict.score
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
        "is_new_best": prev_best is None or verdict.score < prev_best["score"],
        "delta_vs_previous": round(verdict.score - prev["score"], 4) if prev else None,
        "delta_vs_best": round(verdict.score - prev_best["score"], 4) if prev_best else None,
    }
    spans: list[dict] = []
    span_status = "skipped"  # skipped | ok | too_short | failed
    if span_feedback:
        try:
            spans = _worst_spans(text)
            span_status = "ok" if spans else "too_short"
        except Exception as exc:  # noqa: BLE001 - never lose the draft over feedback
            span_status = "failed"
            # Surface it: silently returning [] here reads as "nothing to fix",
            # which is the opposite of what a CUDA OOM or a load failure means.
            out["span_error"] = f"{type(exc).__name__}: {exc}"
        out["worst_spans"] = spans

    if verdict.score <= target:
        out["next_action"] = "Target met. Stop, or call chain_assemble if other segments remain."
    elif span_status == "ok":
        out["next_action"] = (
            "Rewrite the worst spans below in your own voice, then call chain_submit again."
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
    """Per-segment summary of a chain: step counts, best and latest scores, and
    which segments still miss the target."""
    try:
        chain = store.get(chain_id)
    except KeyError as exc:
        return _fail(exc)
    segs = json.loads(chain["segments"])
    target = float(chain["target_score"])
    stats = [store.segment_stats(chain_id, s) for s in segs]
    for s in stats:
        s["target_met"] = s["best_score"] is not None and s["best_score"] <= target
    return {
        "ok": True,
        "chain_id": chain_id,
        "name": chain["name"],
        "goal": chain["goal"],
        "target_score": target,
        "segments": stats,
        "total_steps": sum(s["steps"] for s in stats),
        "all_targets_met": all(s["target_met"] for s in stats) if stats else False,
        "pending": [s["segment"] for s in stats if not s["target_met"]],
    }


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

    A document can score higher than any of its parts, so always assemble before
    calling a multi-segment chain finished.
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
    if missing:
        out["warning"] = (
            f"{len(missing)} declared segment(s) have no drafts and are absent from this "
            f"document: {missing}. The score describes only what was assembled."
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
    mcp.run(transport=os.environ.get("EDITLENS_TRANSPORT", "stdio"))


if __name__ == "__main__":
    main()
