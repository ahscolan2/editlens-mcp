# EditLens MCP

A local AI-text detector as an MCP server, plus a durable chain store so a model can run
long write → score → revise loops without carrying every draft in its context.

Backed by [`pangram/editlens_roberta-large`](https://huggingface.co/pangram/editlens_roberta-large):
a RoBERTa-large 4-bucket classifier (Human-written / Lightly AI-edited / Heavily AI-edited /
Fully AI-generated). The continuous score is the expected bucket index under the softmax,
normalised to `[0, 1]` — the same formula the official demo Space uses.
**0.0 = human-written, 1.0 = fully AI-generated.**

Everything runs on your machine. No text leaves it.

Runs on Windows, macOS (Apple Silicon via Metal), and Linux.

---

## Setup

### 1. Install PyTorch

**macOS** — the default wheels include Metal support:

```bash
pip install torch
```

**Windows / Linux with an NVIDIA GPU:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**No GPU:** plain `pip install torch` works everywhere; it will run on CPU.

### 2. Install the rest

```bash
pip install -r requirements.txt
```

### 3. Get access to the model

The checkpoint is **gated**. Accept the licence at
<https://huggingface.co/pangram/editlens_roberta-large>, then log in so the token is cached:

```bash
hf auth login
```

Paste a **Read** token from <https://huggingface.co/settings/tokens>. Alternatively set
`HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) in the environment.

### 4. Verify

First run downloads ~1.4 GB of weights into the Hugging Face cache, then reuses it forever.

```bash
python run_tests.py
```

### Shortcut

`setup.py` does steps 1, 2 and 4 for you, checks gated-model access, and prints the exact
MCP client config for your machine — including the right interpreter path:

```bash
python setup.py
```

### Note for macOS

Developed and verified on Windows with CUDA. The macOS path (Metal/MPS) is implemented and
its fallback logic is tested, but has not been run on real Apple Silicon. If Metal
misbehaves, `EDITLENS_DEVICE=cpu` always works — this model is small enough that CPU is
usable. Please open an issue with `detector_info` output if you hit anything.

CPU fallback on Apple Silicon is not the penalty it sounds like: PyTorch's macOS wheels link
Apple's Accelerate framework, which routes matrix multiplies through the AMX units in the CPU
cores. `detector_info.cpu_backend` reports which BLAS is actually in use — expect
`Accelerate` on a Mac. What CPU mode does skip is the GPU and the Neural Engine.

Reaching the Neural Engine would mean converting the model to Core ML, and squeezing more out
of the GPU would mean an MLX port. Both are second inference paths to maintain and re-validate
against these scores, so neither is here. For a 355M-parameter model, MPS is already fast
enough that the added complexity would not pay for itself.

---

## Register with an MCP client

The server speaks stdio. Point your client at `run_server.py` with an absolute path —
`run_server.py` exists precisely because clients disagree about honouring a `cwd` setting.

**Claude Code:**

```bash
claude mcp add editlens --scope user -- python /absolute/path/to/MCP-EditLens/run_server.py
```

**Any client that uses a JSON config** (Claude Desktop, Antigravity, Cursor, …):

```json
{
  "mcpServers": {
    "editlens": {
      "command": "python",
      "args": ["/absolute/path/to/MCP-EditLens/run_server.py"]
    }
  }
}
```

On Windows use double backslashes in JSON (`"C:\\Users\\you\\MCP-EditLens\\run_server.py"`)
and, if `python` is not on PATH, give the full interpreter path as `command`.

Config file locations:

| Client | Path |
| --- | --- |
| Claude Code | `~/.claude.json` |
| Claude Desktop (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop (Windows) | `%APPDATA%\Claude\claude_desktop_config.json` |
| Antigravity app | `~/.gemini/antigravity/mcp_config.json` |
| `agy` CLI | `~/.gemini/config/mcp_config.json` |

The Antigravity app and the `agy` CLI read **separate** files — register in both if you
want it in both.

---

## Tools

Thirteen tools. Every one returns a JSON object; failures come back as
`{"ok": false, "error": "...", "error_type": "..."}` rather than raising.

### Scoring

| Tool | Parameters | What it does |
| --- | --- | --- |
| `detect` | `text`, `include_windows=false` | Score one text. Returns `score`, `bucket`, `label`, `probs`, `word_count`, `char_count`, `windows` (a *count*), `target_hint`, `reliable`, and `reliability_note` under 60 words. |
| `detect_batch` | `texts` | Score N texts in one forward pass. Returns `results`, `best_index`, `best_score`, `mean_score`. |
| `detect_spans` | `text`, `granularity="sentence"` (`"sentence"`\|`"paragraph"`), `top=10` (1–50), `min_words=25` (1–200) | Split and score each unit, worst-first. |
| `detector_info` | — | Checkpoint, base model, `platform`, `accelerator`, `device`, `dtype`, load state, VRAM, idle counters, token visibility, plus `db_path` and `duplicate_steps_present`. |
| `detector_unload` | — | Release GPU memory now. Returns `was_loaded`. |

`detect(include_windows=true)` adds `window_detail` — but only when the text actually
needed more than one window; a short text returns none. Each entry carries `start`/`end`
(the window's own char span) and `owned_start`/`owned_end` (the sub-range that window was
weighted by, after overlaps are split at their midpoint), all four in your original
coordinates, plus `words`, `score`, `label` and a short `preview`.

`detector_info.device_fallback` is `null` unless the chosen accelerator failed its load-time
probe, in which case it says what happened. `cpu_backend` reports `{blas, threads}`.
`duplicate_steps_present` is `true` only for a legacy database written before the UNIQUE
step index existed and that still holds duplicate step numbers; the store leaves that data
alone rather than deduplicating it, and says so here. `warmup_error` is non-null if the
startup torch/transformers import failed.

**Span offsets index the text you passed in.** `detect_spans`, `chain_submit`, and
`detect(include_windows=true)` return `start`/`end` in your original coordinates, so you can
splice a rewrite straight back. The `text` field on each unit is the *normalised* form the
model scored (whitespace collapsed), which may differ from that exact slice.

**Offsets go stale the moment you edit.** Rewrite one sentence and every later offset shifts.
Any response that carries offsets also carries a `source_fingerprint` — a digest of the exact
string the offsets were computed from. `detect_spans` always returns one; `chain_submit`
returns one whenever `span_feedback=true`; `detect` returns one only alongside
`window_detail`. Check it before reusing offsets you cached; if it does not match your current
text, they have moved. `detect_spans` also returns `offsets_note`, which says the same thing
in the response itself. For repeated rewrite cycles, matching on each unit's `text` is more
robust than splicing by index.

`detect_spans` returns `document_score`, `document_label`, `unit_count`, `worst_units` (the
`top` highest-scoring units), `source_fingerprint`, `offsets_note`, `min_words_used`,
`granularity_relaxed` and `unreliable_units`.

It merges short sentences toward `min_words` because the model is unreliable on very short
inputs. If that would leave the whole text as one unit, the threshold is relaxed
automatically — it retries at 15, then 8, then 1 (skipping any that is not smaller than the
`min_words` you asked for), stopping at the first that yields two or more units. If none of
them divides the text, the original `min_words` is reported and nothing was relaxed.
`min_words_used` reports what it settled on and `granularity_relaxed` says
whether it backed off. Each unit carries `reliable`, which is a fixed **≥25 words** and does
*not* track `min_words`; `unreliable_units` counts the rest. With
`granularity="paragraph"`, paragraphs are returned as authored, `min_words_used` is `null`
and `granularity_relaxed` is always `false`, because no threshold was applied.

### Chains

A chain is a write → score → revise loop whose state lives in SQLite, not in the model's
context. That is what makes long chains practical: step 200 costs the same context as step 2.
A chain holds ordered **segments** (sections of a document); each segment accumulates
numbered **steps** (revisions).

| Tool | Parameters | What it does |
| --- | --- | --- |
| `chain_create` | `name`, `target_score=0.25` (0.0–1.0), `goal=null`, `segments=null` | Open a chain. Duplicate segment names are de-duplicated. |
| `chain_submit` | `chain_id`, `text`, `segment="main"`, `note=null`, `span_feedback=true`, `span_top=5` (1–50), `span_min_words=25` (1–200), `branch_from=null` | Score and store a draft; returns movement and worst spans, **not** the text. |
| `chain_status` | `chain_id` | Per-segment best/latest scores and what is still pending. |
| `chain_history` | `chain_id`, `segment="main"`, `limit=30` (1–200) | Score trajectory. Numbers and notes only. `limit` keeps the most recent N steps; they come back oldest-first, with `returned` counting them. `total_steps`, `truncated`, `best_step` and `best_score` describe the whole segment, so a window never hides the best draft. An unknown `segment` is an error, not an empty trajectory. |
| `chain_get_text` | `chain_id`, `segment="main"`, `step="best"` | Retrieve a stored draft — `"best"`, `"latest"`, or a step number. |
| `chain_assemble` | `chain_id`, `separator="\n\n"`, `include_text=true` | Join the best of every segment and score the whole document. |
| `chain_list` | `limit=25` (1–200) | List chains, most recently updated first. |
| `chain_delete` | `chain_id` | Delete a chain and all its steps. Irreversible. |

`chain_submit` returns `step`, `parent_step`, `score`, `label`, `words`, `target_score`,
`target_met`, `best_score`, `best_step`, `is_new_best`, `delta_vs_previous`, `delta_vs_best`,
`reliable`, `next_action`, and — when `span_feedback=true` — `worst_spans`,
`spans_above_target`, `span_unit_count`, `spans_above_target_total`, `spans_truncated`,
`span_min_words_used` and `source_fingerprint`. `delta_vs_previous` and `delta_vs_best` are
`null` on the first step of a segment. If span analysis fails, `span_error` carries the
reason rather than silently reporting no spans to fix, and `next_action` says so. Submitting
to a segment the chain does not yet declare adds it, but only after the draft scores: a
failed submit never registers the segment.

**Rewrite only the spans marked `above_target`.** `worst_spans` is a ranking, not a to-do
list — its tail is routinely text the same response labels `Human-written`, and rewriting
that is how a loop makes a draft worse while believing it is following orders. When
`spans_above_target` is 0 and the document is still above target, span-level work has
bottomed out and `next_action` says so instead of sending you round again.

**`worst_spans` is a window, not the list.** It holds at most `span_top` entries.
`spans_above_target` counts the ones shown; `spans_above_target_total` counts every unit
above target in the draft, and `spans_truncated` says whether those differ. On a 1548-word
document all 36 units scored above a 0.25 target and five were reported — so treat the
unshown remainder as unexamined, not as passing.

**On long drafts, tune `span_min_words`.** A span is the quantum of rewriting: acting on one
means replacing all of it. Driving that same 1548-word document to target rewrote 56% of it
over 4 rounds at the default 25, and 15% over 2 rounds at `span_min_words=15`; on a 584-word
document it was 35% versus 18%. Smaller spans score more noisily — each span carries
`reliable`, false below 25 words — but they let you replace the sentences that score badly
instead of the paragraphs around them. `span_min_words_used` reports what the splitter
settled on, which can be lower than you asked for if the draft would not otherwise divide.

**Short drafts get a caveat instead of a rewrite order.** `detect` and `chain_submit` return
`reliable` (≥25 words) and, under 60 words, a `reliability_note`; below 60 and still above
target, `next_action` leads with it. Five known-human passages cut to a fixed length scored
across a 0.63 range at 15 words, 0.24 at 30 and 0.06 by 60, and one crossed a 0.25 target on
length alone — so a short draft scoring high is not evidence that it needs rewriting.

`chain_status` returns `pending` (everything not at target) and splits it into `unstarted`
and `above_target`, which need opposite responses. `latest_is_best` per segment — and
`segments_with_better_earlier_draft` at the top — tell you whether the draft you last
submitted is the one assembly will actually use.

`chain_assemble` returns `document_score`, `document_label`, `words`, `windows`,
`per_segment`, `missing_segments`, `complete`, `score_met`, `target_met`,
`segments_above_target`, a `warning` when segments are missing, and `text` unless
`include_text=false`. **`target_met` requires `complete`** — a document missing a declared
section is not finished, however well the parts that exist happen to score; `score_met` is
the score test alone.

Every chain tool returns `next_action`. It is the field to read first: it accounts for the
regression case, the bottomed-out case, and the status/assemble disagreement, none of which
are obvious from the numbers.

---

## How a long chain runs

Single-segment revision loop:

```
chain_create("essay", target_score=0.2)          → ch_a1b2c3
chain_submit(ch, draft_1)                        → 0.87   worst_spans: [...]
chain_submit(ch, draft_2, note="cut the hedging")→ 0.61   Δ -0.26
chain_submit(ch, draft_3)                        → 0.34   Δ -0.27
chain_submit(ch, draft_4)                        → 0.18   target_met ✓
chain_get_text(ch, step="best")                  → the winning draft
```

Multi-segment (horizontal) composition:

```
chain_create("report", segments=["intro","method","results","discussion"])
chain_submit(ch, text, segment="intro")     ← each section revises independently
chain_submit(ch, text, segment="method")    ← chain_status shows what's pending
...
chain_assemble(ch)                          ← best-of-each, scored as one document
```

Assemble before you decide a section needs more work, not after. The document score is
not bounded by the section scores in either direction, and the gap is large: sections
scoring 0.73 and 0.51 have assembled into a 0.11 document, because the model scores short
text harder than the same words inside a longer piece. So `chain_status` can list a
section as pending while `chain_assemble` calls the document finished — both are right,
and `chain_assemble`'s `segments_above_target` plus its `next_action` say which one
governs. It goes the other way too: a whole document can land above every part of it,
which is why assembling is what ends a chain.

**When a revision makes things worse**, history is append-only — nothing is lost. Pull the
earlier draft back and fork from it, and the chain records the branch:

```
chain_submit(ch, draft_5)                          → 0.71   worse than step 2
chain_get_text(ch, step=2)                         → the draft that worked
chain_submit(ch, new_attempt, branch_from=2)       → step 6, parent_step=2
chain_history(ch)                                  → [(1,None), (2,1), … (6,2)]
```

`chain_history` reports each step's `parent_step`, so several attempts from the same
ancestor are distinguishable from a straight line of revisions. `chain_get_text(step="best")`
and `chain_assemble` always use the lowest-scoring draft regardless of branch, so a bad
detour never costs you the good one.

**What "hundreds of steps" actually costs.** Measured over 65 real submits on one segment:
the `chain_submit` response stayed flat at ~1.8 KB (first 1830 bytes, last 1835, largest
1836) and each call took ~40 ms, because the response never carries history — only this
step, its deltas, and the spans. `chain_status` stayed under 1 KB at 65 steps. Nothing
accumulates in your context.

`chain_history` is the one response that grows with the chain, so it is capped: `limit`
keeps the most recent N steps and defaults to 30. On a long chain that window can exclude
the best draft entirely — at 65 steps the default returns steps 36–65, and the best was step
4. So do **not** take the lowest score in `trajectory` for the best draft. `total_steps` and
`truncated` say whether you are looking at a window, `best_step`/`best_score` describe the
whole segment either way, and when the best step falls outside the window a `note` names it
along with `chain_get_text` and `branch_from`.

To widen rather than deepen: generate several candidates per step and use `detect_batch` to
keep the best before submitting. That converges in fewer chain steps than revising a single
line of drafts.

---

## Notes

- **Device selection** is automatic: CUDA → Apple Silicon (Metal/MPS) → CPU. Override with
  `EDITLENS_DEVICE`. On load the chosen device runs one tiny forward pass to prove it works;
  if that fails the server falls back to CPU rather than erroring on every later call, and
  `detector_info.device_fallback` says what happened. On macOS,
  `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before torch is imported so the few ops without a
  Metal kernel run on CPU instead of killing the process.
- **Precision** defaults to `float32`, the precision the checkpoint is published in.
  Measured on an RTX 2080, `float16` moves scores by at most 0.001 and saves nothing on a
  single paragraph (14 ms either way); it is ~2.5× faster on long documents and ~1.6× on
  batches. Set `EDITLENS_DTYPE=float16` if you run big batches. float16 requires a GPU.
- **GPU memory is released when idle.** Nothing loads until the first `detect`
  (`detector_info.vram_mb` is `null` until then). After 5 minutes without a call the model
  unloads and returns ~1.4 GB; the next call reloads in ~2 s. The watchdog checks on a tick of
  `idle/4`, clamped to 5–30 s, so the unload can land up to half a minute past the deadline,
  and it never interrupts a request in flight. `auto_unloads` counts how many times it has
  fired. Tune with `EDITLENS_IDLE_UNLOAD` (`0` or negative disables), or call `detector_unload`.
- **Startup imports torch on the main thread**, costing ~2 s. This is deliberate: MCP servers
  run tool functions in worker threads, and importing torch from a worker thread hangs
  indefinitely on Windows — the client times out and the server looks dead. Do not make these
  imports lazy again.
- **Requests are serialised.** HuggingFace's Rust tokenizer raises `Already borrowed` if two
  threads encode at once, so one request runs at a time. Inference is 15–90 ms, and two
  clients can safely share one server process and one database.
- **Long inputs are windowed.** Anything over 490 content tokens is split into overlapping
  windows (the margin below 512 exists because windows are character slices that get
  re-tokenised, and can come back slightly longer). Windows are combined by word count over
  the region each one *owns* — overlaps are split at their midpoint so no text is counted
  twice.
- **Short text is noisy.** This is a document-level model. Treat anything under ~25 words as
  indicative rather than precise; that is what the `reliable` flag marks.
- **Scores are not ground truth.** EditLens estimates degree of AI editing and has false
  positives, particularly on formal or technical prose. A low score means "this checkpoint
  finds it human-like", not "undetectable" — other detectors disagree with it routinely.

---

## Configuration

| Env var | Default | Empty value |
| --- | --- | --- |
| `EDITLENS_CHECKPOINT` | `pangram/editlens_roberta-large` | **used as-is** — an empty value is an empty repo id and the load fails |
| `EDITLENS_BASE_MODEL` | `FacebookAI/roberta-large` (tokenizer fallback) | **used as-is**, same caveat |
| `EDITLENS_DEVICE` | auto: `cuda` → `mps` → `cpu` | treated as unset |
| `EDITLENS_DTYPE` | `float32`. Set `float16` for speed on long/batched work (GPU only). | treated as unset |
| `EDITLENS_BATCH_SIZE` | `8` | treated as unset |
| `EDITLENS_IDLE_UNLOAD` | `300` seconds idle before GPU memory is released (`0` or negative = never) | treated as unset |
| `EDITLENS_TRANSPORT` | `stdio` | treated as unset |
| `EDITLENS_DB` | see below | treated as unset |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | falls back to the `hf auth login` cache | treated as unset (falls through to the other, then the cache) |

**Empty is not a configuration error.** MCP client configs routinely emit `"EDITLENS_DTYPE": ""`
for a field the user left blank, and a server that dies at import over that just looks dead to
the client. So every variable above except the two model ids treats `""` exactly as if it were
unset. Whitespace-only is a narrower story: `EDITLENS_BATCH_SIZE` and `EDITLENS_IDLE_UNLOAD`
also ignore `"   "`, but the others do not — `EDITLENS_DEVICE="   "` is a device name, and
`EDITLENS_DB="   "` is a path.

`EDITLENS_BATCH_SIZE` and `EDITLENS_IDLE_UNLOAD` also survive garbage: a non-numeric value
prints one warning to stderr and falls back to the default rather than raising at import. A
batch size below 1 is clamped up to 1; a negative idle-unload disables the watchdog.
An unrecognised `EDITLENS_DTYPE` warns once and uses `float32`. A wrong-but-non-empty
`EDITLENS_TRANSPORT` still errors, but the message quotes what was set.

Default database location (used only when `EDITLENS_DB` is unset or empty):

| Platform | Path |
| --- | --- |
| Windows | `%LOCALAPPDATA%\editlens-mcp\chains.db` (or `~\AppData\Local\...` if that is unset) |
| macOS | `~/Library/Application Support/editlens-mcp/chains.db` |
| Linux | `$XDG_DATA_HOME/editlens-mcp/chains.db` (or `~/.local/share/...`) |

The store opens in WAL mode where the filesystem allows it, and falls back to whatever
journal mode it can get rather than refusing to start — some network filesystems reject WAL.
`run_tests.py` points `EDITLENS_DB` at a fresh temp path before any suite runs — unless you
set one yourself, which it leaves alone — so no suite can fall through to the default and run
schema migrations against the operator's real database.

---

## Testing

```bash
python run_tests.py
```

Eleven suites, in the order `run_tests.py` runs them:

| Suite | Covers |
| --- | --- |
| `smoke_test.py` | plumbing and tool wiring against a stubbed model, plus one real scoring pass |
| `tests/test_tools.py` | all 13 tools over an in-memory client, including error paths |
| `tests/test_usability.py` | the guidance a model actually follows: `next_action` in the bottomed-out, regression, status-vs-assemble, truncated-span, short-draft and long-history cases |
| `tests/test_concurrency.py` | SQLite concurrency and cross-process step allocation |
| `tests/test_offsets.py` | span offsets index the caller's original text |
| `tests/test_branching.py` | `branch_from`/`parent_step` and offset staleness |
| `tests/test_robustness.py` | failure modes: simultaneous cold starts, failed commits, disk-full reporting, accelerator fallback |
| `tests/test_entrypoints.py` | startup configuration, the helper scripts, and test-database isolation |
| `tests/test_detector.py` | precision, windowing, multi-segment chains, restart persistence, edge cases, determinism |
| `tests/test_gpu_memory.py` | idle unload, manual unload, threading under load |
| `tests/test_client.py` | the real path — the server as a stdio subprocess, which is what MCP clients actually do |

`run_tests.py` forces `EDITLENS_DB` to a temp path before any suite runs, so testing can
never migrate or write your real chain database.

Run this after changing anything. The subprocess suite in particular catches failures the
in-process ones cannot, because tool functions run on a worker thread there.

The suite is mutation-tested: deliberate bugs are introduced one at a time, in a copy of the
tree, and the suite has to fail. The first round covered the scoring core — dropping the score
normalisation, skipping the last window, restoring double-counted overlaps, inverting
`best_step`, storing empty drafts, ignoring missing segments, shifting the offset map, dropping
the UNIQUE index, forcing float16. Later rounds pushed into the startup and recovery paths:
the migration retry loop and its "duplicate column" tolerance, the `duplicate_steps_present`
legacy flag, `chain_history`'s `limit`, paragraph splitting and its offsets,
`split_units_adaptive`'s "no threshold applied" report, `map_span`'s empty/inverted/overrun
guards, the last window's ownership of the document tail, `_load` recording where the model
actually landed, a manual unload racing a request, a non-positive `EDITLENS_IDLE_UNLOAD`, and
the `EDITLENS_DB` floor in `run_tests.py`. Every survivor was closed by strengthening an
assertion, never by weakening one.

Two mutations survive on purpose, because they are equivalent rather than uncaught: the
watchdog's `_inflight == 0` check (unreachable while `_infer_lock` is held for whole requests)
and the explicit `DELETE FROM steps` in `chain_delete` (the `ON DELETE CASCADE` already does
it; break both and the suite fails).

On a machine without a GPU (or with `EDITLENS_DEVICE=cpu`), the GPU-memory suite skips itself
and the float16 comparison is skipped; everything else runs.

---

## Licence

The code in this repository is MIT (see `LICENSE`).

The **model is not**. `pangram/editlens_roberta-large` is CC-BY-NC-SA-4.0 — non-commercial
use only, share-alike. No weights are distributed here; you download them yourself under that
licence after accepting it on Hugging Face. Check its terms before using this for anything
commercial.
