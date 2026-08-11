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
| `detect` | `text`, `include_windows=false` | Score one text. Returns `score`, `bucket`, `label`, `probs`, `word_count`, `windows`. |
| `detect_batch` | `texts` | Score N texts in one forward pass. Returns `results`, `best_index`, `best_score`, `mean_score`. |
| `detect_spans` | `text`, `granularity="sentence"`, `top=10`, `min_words=25` | Split and score each unit, worst-first. |
| `detector_info` | — | Checkpoint, platform, accelerator, device, dtype, load state, VRAM, token visibility. |
| `detector_unload` | — | Release GPU memory now. |

**Span offsets index the text you passed in.** `detect_spans`, `chain_submit`, and
`detect(include_windows=true)` return `start`/`end` in your original coordinates, so you can
splice a rewrite straight back. The `text` field on each unit is the *normalised* form the
model scored (whitespace collapsed), which may differ from that exact slice.

**Offsets go stale the moment you edit.** Rewrite one sentence and every later offset shifts.
Each response carries a `source_fingerprint` — a digest of the exact string the offsets were
computed from. Check it before reusing offsets you cached; if it does not match your current
text, they have moved. For repeated rewrite cycles, matching on each unit's `text` is more
robust than splicing by index.

`detect_spans` merges short sentences toward `min_words` because the model is unreliable on
very short inputs. If that would leave the whole text as one unit, the threshold is relaxed
automatically — `min_words_used` reports what it settled on and `granularity_relaxed` says
whether it backed off. Each unit carries `reliable` (≥25 words), and `unreliable_units`
counts the rest. With `granularity="paragraph"`, paragraphs are returned as authored and
`min_words_used` is `null`, because no threshold was applied.

### Chains

A chain is a write → score → revise loop whose state lives in SQLite, not in the model's
context. That is what makes long chains practical: step 200 costs the same context as step 2.
A chain holds ordered **segments** (sections of a document); each segment accumulates
numbered **steps** (revisions).

| Tool | Parameters | What it does |
| --- | --- | --- |
| `chain_create` | `name`, `target_score=0.25`, `goal=null`, `segments=null` | Open a chain. Duplicate segment names are de-duplicated. |
| `chain_submit` | `chain_id`, `text`, `segment="main"`, `note=null`, `span_feedback=true`, `branch_from=null` | Score and store a draft; returns movement and worst spans, **not** the text. |
| `chain_status` | `chain_id` | Per-segment best/latest scores and what is still pending. |
| `chain_history` | `chain_id`, `segment="main"`, `limit=30` | Score trajectory. Numbers and notes only. |
| `chain_get_text` | `chain_id`, `segment="main"`, `step="best"` | Retrieve a stored draft — `"best"`, `"latest"`, or a step number. |
| `chain_assemble` | `chain_id`, `separator="\n\n"`, `include_text=true` | Join the best of every segment and score the whole document. |
| `chain_list` | `limit=25` | List chains, most recently updated first. |
| `chain_delete` | `chain_id` | Delete a chain and all its steps. Irreversible. |

`chain_submit` returns `score`, `label`, `words`, `target_met`, `best_score`, `best_step`,
`is_new_best`, `delta_vs_previous`, `delta_vs_best`, `next_action`, `worst_spans`, and
`spans_above_target`. If span analysis fails, `span_error` carries the reason rather than
silently reporting no spans to fix.

**Rewrite only the spans marked `above_target`.** `worst_spans` is a ranking, not a
to-do list — its tail is routinely text the same response labels `Human-written`, and
rewriting that is how a loop makes a draft worse while believing it is following orders.
When `spans_above_target` is 0 and the document is still above target, span-level work has
bottomed out and `next_action` says so instead of sending you round again.

`chain_status` returns `pending` (everything not at target) and splits it into `unstarted`
and `above_target`, which need opposite responses. `latest_is_best` per segment — and
`segments_with_better_earlier_draft` at the top — tell you whether the draft you last
submitted is the one assembly will actually use.

`chain_assemble` returns `document_score`, `per_segment`, `missing_segments`, `complete`,
`score_met`, `target_met`, and `segments_above_target`. **`target_met` requires `complete`** —
a document missing a declared section is not finished, however well the parts that exist
happen to score.

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
- **GPU memory is released when idle.** Nothing loads until the first `detect` (0 MB until
  then). After 5 minutes without a call the model unloads and returns ~1.4 GB; the next call
  reloads in ~2 s. Tune with `EDITLENS_IDLE_UNLOAD` (`0` disables), or call `detector_unload`.
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

| Env var | Default |
| --- | --- |
| `EDITLENS_CHECKPOINT` | `pangram/editlens_roberta-large` |
| `EDITLENS_BASE_MODEL` | `FacebookAI/roberta-large` (tokenizer fallback) |
| `EDITLENS_DEVICE` | auto: `cuda` → `mps` → `cpu` |
| `EDITLENS_DTYPE` | `float32`. Set `float16` for speed on long/batched work (GPU only). |
| `EDITLENS_BATCH_SIZE` | `8` |
| `EDITLENS_IDLE_UNLOAD` | `300` seconds idle before GPU memory is released (`0` = never) |
| `EDITLENS_TRANSPORT` | `stdio` |
| `EDITLENS_DB` | see below |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | falls back to the `hf auth login` cache |

Default database location:

| Platform | Path |
| --- | --- |
| Windows | `%LOCALAPPDATA%\editlens-mcp\chains.db` |
| macOS | `~/Library/Application Support/editlens-mcp/chains.db` |
| Linux | `$XDG_DATA_HOME/editlens-mcp/chains.db` (or `~/.local/share/...`) |

---

## Testing

```bash
python run_tests.py
```

Ten suites: plumbing plus one real scoring pass; all 13 tools over an in-memory client
including error paths; SQLite concurrency and cross-process step allocation; span-offset
correctness; branching and offset staleness; failure modes (simultaneous cold starts, failed
commits, disk-full error reporting, accelerator fallback); entry points and startup config;
precision, windowing, edge cases and determinism; GPU idle-unload and threading under load;
and the real path — the server as a stdio subprocess, which is what MCP clients actually do.

`run_tests.py` forces `EDITLENS_DB` to a temp path before any suite runs, so testing can
never migrate or write your real chain database.

Run this after changing anything. The subprocess suite in particular catches failures the
in-process ones cannot, because tool functions run on a worker thread there.

The suite is mutation-tested: deliberate bugs are introduced one at a time and the suite must
fail. Over 30 have been tried — dropping the score normalisation, skipping the last window,
restoring double-counted overlaps, inverting `best_step`, storing empty drafts, ignoring
missing segments, truncating CRLF spans, dropping the UNIQUE index, forcing float16, removing
the WAL retry, scoring by argmax instead of expectation, deleting the `_guard` decorator —
and every one is now caught. Several were not, until the tests were strengthened to catch
them; a test that cannot fail is worse than no test.

On a machine without a GPU (or with `EDITLENS_DEVICE=cpu`), the GPU-memory suite skips itself
and the float16 comparison is skipped; everything else runs.

---

## Licence

The code in this repository is MIT (see `LICENSE`).

The **model is not**. `pangram/editlens_roberta-large` is CC-BY-NC-SA-4.0 — non-commercial
use only, share-alike. No weights are distributed here; you download them yourself under that
licence after accepting it on Hugging Face. Check its terms before using this for anything
commercial.
