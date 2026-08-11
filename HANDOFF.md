# HANDOFF — read this first

You are picking up EditLens MCP mid-flight. This file is the whole context you need.
Nothing here is speculative: every claim is either verified, or explicitly labelled unverified.

**Repo:** `C:\Users\Aryan\MCP-EditLens` on Windows, `git@github.com:ahscolan2/editlens-mcp`
(private). Clean at `af7bc8d`, all 11 suites green **on Windows**.

**What it is:** an MCP server wrapping a local, gated HuggingFace AI-text detector
(`pangram/editlens_roberta-large`; 0.0 = human-written, 1.0 = fully AI-generated), plus a
SQLite chain store for long write → score → revise loops. 13 tools. See `README.md`.

**The owner (Aryan) is not a Python developer.** Explain in plain language, lead with the
answer, skip the writeup. He is moving this to a Mac.

---

## Current priority: an external audit found real bugs

An agent on the owner's Mac audited commit `af7bc8d` from the public repo, ran 10 of 11
suites against a stand-in checkpoint (`EDITLENS_CHECKPOINT=distilroberta-base`, no gated
access needed — **this is a genuinely useful trick, reuse it**), and filed findings. The
owner has the full report. Below is the triaged list.

### Confirmed by me before handoff

| # | Bug | Where |
| --- | --- | --- |
| **A1** | `cwd="C:\\Windows"` is hardcoded, so **the suite cannot pass on macOS or Linux**. This is the suite the README calls the most important one. `cwd=tempfile.mkdtemp()` reportedly fixes it. | `tests/test_client.py:21` |
| **A2** | Dependencies unpinned with no upper bound. `transformers>=4.44` resolves to 5.15 today. | `requirements.txt` |
| **A3** | README says paths like `MCP-EditLens/` but the repo clones as `editlens-mcp` (3 places). | `README.md` |

### Reported and credible, NOT yet verified by me — verify before fixing

Ordered by how much I believe them and how much they matter.

1. **Stored drafts are whitespace-normalised.** `server.py` stores `clean_text(text)`, which
   collapses every run of spaces — flattening markdown nesting, code indentation and table
   alignment. `chain_get_text` hands that mangled text back. *Scoring on normalised text is
   correct; storing it is the bug.* Fix: store the original, normalise only into the model.
   **This is the one with a real design decision in it** — changing what `chain_get_text`
   returns affects `chain_assemble` and every offset. Think before you act.
2. **Offsets/`source_fingerprint` describe a string no tool returns.** They are computed
   against the raw text but the *cleaned* text is stored, so they slice the wrong text out of
   what `chain_get_text` gives you. Compounds with (1); fix them together.
3. **`next_action` logic bugs, all in the default configuration:**
   - A 900-word bullet list gets "too short to pinpoint spans" with
     `spans_above_target_total: 0`, which reads as "nothing to fix". Root cause is
     `_worst_spans` bailing on `len(units) < 2`, which means "no `[.!?]` found", not "short".
   - `span_top` alone flips the instruction between "patch these 5" and "rewrite the whole
     thing" — the truncated branch is tested before the all-units-above-target branch.
   - Single-segment chains (the default) are told to `chain_assemble` first, where the
     assembled score is arithmetically identical.
4. **`EDITLENS_DB` is never `expanduser()`d** — `~/foo.db` creates a literal `~` directory and
   loses chains on restart, silently.
5. **A bad `EDITLENS_DB` kills all 13 tools at import**, including `detector_info`, whose own
   docstring says to call it first when things break. `store = ChainStore(...)` at module
   scope is unguarded — the one thing in that file that isn't.
6. **Concurrent submits fabricate lineage.** Step *numbers* are correct (verified), but
   `parent_step`, `is_new_best`, `best_step` and both deltas are read before the slow scoring
   window. 12 concurrent submits → all 12 reported `parent_step: None`. Fix: re-read inside the
   same `BEGIN IMMEDIATE` that inserts.
7. **CJK/Japanese/Thai silently unsupported.** `count_words` is `\b[\w'’-]+\b`, so an unspaced
   Chinese document counts as 1 word — disabling `reliable`, span feedback and window
   weighting. Nothing says the tool is English-only. Documenting it is a legitimate fix.
8. Smaller: `chain_list.best_score` is `MIN` across *all* segments (a 0.90 intro + 0.05 body
   lists as 0.05); `segment_stats.words` pairs the latest step's word count with the best
   step's score; `chain_get_text` doesn't validate the segment name (`chain_history` was
   fixed for this, the recovery path wasn't); README claims testing "can never" write your
   real database but `run_tests.py` only sets `EDITLENS_DB` *if unset*, contradicting itself
   two sections earlier.

### Audit claims I would NOT act on without re-verifying

The Mac agent used `distilroberta-base`, not the real checkpoint. Anything about *scores* or
*bucket labels* is suspect. It also self-corrected twice mid-audit (it wrongly predicted the
22-token window margin was too thin — worst case measured 494/512, so that margin is fine).

---

## My own open items

- **`mut2` agent died on a session limit before doing any work.** It was pointed at areas
  three prior mutation rounds never reached: `_sentence_spans` regexes and the blank-line
  rule, `split_units`' merge pass, `windows()` step/overlap arithmetic, the PEFT-adapter
  fallback and `_setup_hint`'s gated-repo detection, `add_step`'s retry ceiling,
  `list_chains`/`segment_stats` aggregation, `chain_status`'s `all_targets_met`. Still unrun.
- **`span_min_words` default is 25; measurement favoured 15** (a 1548-word document went from
  55.8% rewritten to 14.7%). Left at 25 on n=2 evidence — deliberately, not by oversight.
  More data would settle it.
- **No segment lifecycle API.** No remove/rename/reorder, so a typo'd segment name marks a
  chain permanently incomplete and `chain_delete` is the only escape.
- **macOS/Metal has never run on real Apple Silicon.** Implemented, with a CPU fallback that
  is tested by simulation only.
- **No `pyproject.toml`, no CI, no `requires-python`** (code needs ≥3.10). `setup.py` at the
  repo root shadows the setuptools convention.

---

## How this project has been worked, and why

Seven rounds of review agents. Every round found something real. The two highest-yield
techniques, by a wide margin:

1. **Mutation testing.** Deliberately break the code in a *copy*, run the suite, see if it
   fails. This caught that `test_tools.py` once had **zero assertions** — it printed results
   and always exited 0, so several fixes were sitting completely unguarded. 60+ mutations have
   been run; harness at `%TEMP%\claude\C--\e0bc8a8c-*\scratchpad\mutate.py` with specs in
   `muts/`. Two mutations survive *on purpose* because they are provably equivalent, not
   uncaught; both are documented in the README.
2. **Actually using the server for its job.** Five rounds of correctness agents left the code
   hard to crash but never noticed that `next_action` told a model to rewrite spans the same
   response labelled `Human-written` — an infinite loop that passed every suite. If you only
   hunt crashes you will not find this class.

**Working method that worked:** two agents at a time, never more (five at once saturated the
owner's 8 GB GPU and made his machine unusable — he asked for batches of 1–2). Each gets its
own `git worktree` on its own branch with write access; I review the diff, run the full suite
on the merged result, then push. Agents are told not to commit or touch git.

---

## Landmines — do not undo these

Each was a real bug that cost real time. The code comments say so at each site.

- **torch must be imported on the main thread at startup** (`warmup_imports`). MCP runs sync
  tools in worker threads, and importing torch from a worker hangs *indefinitely* on Windows —
  2 s vs never. The client times out and the server looks dead. In-process tests do **not**
  catch this; only the stdio-subprocess suite does.
- **Requests are serialised** via `_infer_lock`. HuggingFace's Rust tokenizer raises
  `Already borrowed` under concurrent use.
- **Lock order is always `_infer_lock` before `_lock`.** Fixing an unload race the other way
  round deadlocks.
- **`PRAGMA journal_mode=WAL` does not honour `busy_timeout` under a RESERVED lock** (which is
  what `BEGIN IMMEDIATE` takes). The manual retry loop in `_set_wal` is load-bearing.
- **`COMMIT` belongs inside the `try`**; outside it, a failed commit wedges every later write
  permanently.
- **The `parent_step` migration must tolerate losing the ALTER race** — check-then-act killed
  the server at import for whichever client lost.
- **float32 is the intentional default.** float16 differs by ≤0.001 and saves nothing on a
  single paragraph (14 ms either way).
- **EditLens genuinely false-positives on short informal text.** A casual human paragraph can
  score 0.99. That is the model, not a bug. How the server *guides* an agent in that
  situation is in scope; the score is not.

---

## Suggested first moves

1. Fix A1 (one line) and confirm the suite passes on macOS — that unblocks everything else.
2. Verify (1) and (2) above yourself, then fix them together. They are one problem.
3. The three `next_action` bugs (3) — cheap, and they directly break the loop the tool exists
   to run.
4. `expanduser` (4) and the import guard (5) — both small, both silent-data-loss class.
5. Then the unrun mutation areas, if you want another round of agents.

Run `python run_tests.py` after everything. Eleven suites; it forces `EDITLENS_DB` to a temp
path (but see finding 8 — that floor is weaker than the README claims).
