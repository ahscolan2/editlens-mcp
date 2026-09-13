"""Workflow advice, kept separate from statistical model predictions.

These stopping rules are editing workflow heuristics, not calibrated detector
thresholds. They never prevent a caller from saving another draft.
"""

SCORE_NOTE = (
    "An estimate of editing magnitude from this checkpoint, not a probability "
    "of AI authorship, a percentage of AI-written words, or a writing-quality score. "
    "The label is the most probable model bucket."
)
SPAN_NOTE = (
    "Spans are scored independently outside their document context. They do not "
    "explain the document score or identify sentences that must be rewritten. "
    "Scores are not comparable across lengths; either the fragment or the full "
    "document can score higher. Re-score the whole document after an edit."
)
TARGET_NOTE = (
    "target_score is a user-selected workflow threshold, not a calibrated "
    "human/AI boundary. Meeting it does not establish authorship or writing quality."
)


def length_assessment(words: int) -> dict:
    # The official roberta.yaml and paper Table 14 use a 75-word training floor.
    # This is a length check, never a claim that longer texts are calibrated.
    return {
        "length_sufficient": words >= 75,
        "assessment": "below_training_length" if words < 75 else "document_estimate",
        "calibrated": False,
    }


def revision_progress(history: list[dict], step: int) -> dict:
    recent = [row for row in history if row["step"] <= step][-5:]
    improvement = (recent[0]["score"] - min(row["score"] for row in recent[1:])) if len(recent) >= 5 else None
    plateau = improvement is not None and improvement < 0.01
    return {
        "steps_reviewed": len(recent),
        "plateau": plateau,
        "revision_budget_reached": step >= 8,
        "review_after_steps": 8,
        "plateau_min_improvement": 0.01,
        "note": "Workflow heuristics; not statistical confidence bounds or mandatory limits.",
    }


def submission_advice(out: dict, *, words: int, segments: list[str], history: list[dict],
                      regressed: bool, duplicate: bool = False) -> dict:
    progress = revision_progress(history, out["step"])
    segment, best = out["segment"], out["best_step"]
    # repr gives quoted tool parameters even for segment names containing quotes.
    recover = f"chain_get_text(segment={segment!r}, step={best})"
    stop = True
    if words < 75:
        state = "insufficient_length"
        action = (
            f"CAUTION: Only {words} words; below the reference training floor of 75. "
            "Do not revise to chase this score. For a section, finish the document "
            "and call chain_assemble; for a standalone short text, review its meaning "
            "and clarity directly."
        )
    elif out["target_met"]:
        state = "target_met"
        action = (
            "Target met. Stop score-driven revisions and review facts, meaning, and "
            "voice before using the draft. Call chain_assemble for a multi-section document."
        )
    elif duplicate:
        state = "unchanged_draft"
        action = "This draft is unchanged. Stop rescoring it; the repeated result adds no evidence."
    elif progress["plateau"] or progress["revision_budget_reached"]:
        state = "review_recommended"
        action = (
            "Stop score-driven revisions for review: "
            + ("the last five submissions show less than 0.01 improvement. " if progress["plateau"]
               else "the segment has reached the eight-submission review point. ")
            + f"Use {recover} to inspect the lowest-scoring draft, then choose by factual "
              "accuracy, meaning, and voice. Further submissions remain available."
        )
    elif len(segments) > 1:
        state = "assemble_first"
        stop = False
        action = (
            "Finish any unstarted sections, then call chain_assemble before revising "
            "this section further. Section scores do not predict the assembled score."
        )
    elif regressed:
        state = "regression"
        action = (
            f"This score is higher than step {best}. Do not keep editing this draft "
            f"solely to reduce the score. Inspect {recover}; if the earlier draft is "
            f"also better writing, continue with segment={segment!r}, branch_from={best}."
        )
    elif "span_error" in out:
        state = "feedback_failed"
        action = "Span analysis failed (see span_error). Your draft was saved. Review it before retrying analysis."
    elif out.get("worst_spans") and out.get("spans_above_target") == 0:
        state = "no_span_signal"
        action = (
            "Span-level score chasing has bottomed out: do not rewrite the spans "
            "solely to reach the document target. Review the document's clarity and "
            "meaning; there is no supported rewrite instruction in these scores."
        )
    else:
        state = "review_draft"
        stop = False
        action = (
            "Review the draft for a specific problem with meaning, clarity, or voice. "
            "above_target marks an isolated score comparison, not a defect. Preserve "
            "sound content even if every span scores high. If a revision improves the "
            "writing, call chain_submit and compare the full draft; otherwise stop."
        )
    return {"next_action": action, "revision_state": state, "stop_recommended": stop,
            "revision_progress": progress}
