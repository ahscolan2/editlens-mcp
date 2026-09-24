# EditLens MCP

Local estimates of AI editing magnitude, with a durable store for comparing and
recovering drafts. Windows, macOS and Linux; Python 3.10 or newer.

The default checkpoint is
[`pangram/editlens_roberta-large`](https://huggingface.co/pangram/editlens_roberta-large),
a four-bucket RoBERTa classifier. Independent MCP clients share one local inference
worker per matching environment and configuration. Requests queue behind one model.

## What the score means

The continuous score is the expected bucket index, divided by three:

```text
score = (0*p_human + 1*p_light + 2*p_heavy + 3*p_generated) / 3
label = the bucket with the largest probability
```

The label can differ from the bucket nearest the continuous score. Scores are
estimates of editing magnitude, not AI-authorship probabilities, percentages of
AI-written words, or measures of writing quality. A low score does not establish
authorship, factual accuracy, or whether a draft is ready to use.

The reference pipeline demojizes emoji, removes certain leading headers and think
blocks, lowercases, and collapses whitespace. This happens to a separate model
input; saved drafts retain their original text. The default checkpoint is pinned
to revision `f93e1ace74528cfb48f337ab2fe946fb71a728cb`.

The reference training configuration uses a 75-word minimum. Responses include
`length_sufficient`, `assessment` and `calibrated=false`. The retained `reliable`
field is an alias for the length check only. It is **not confidence**. The server
uses the smaller of the original readable-word count and the model-input count
as `assessment_word_count`, so removed headers and expanded emoji/contractions
cannot inflate it. This conservative check extends the training filter. Language and domain shifts remain
unvalidated; the text utilities' ability to split CJK text is not evidence of model
accuracy on it.

`detect_spans` scores fragments independently. These scores neither explain the
document score nor identify sentences that must be rewritten. Fragment scores and
full-document scores are not interchangeable. See the
[research and validation notes](docs/RESEARCH.md) for sources and limitations.

## Setup

Run from the repository directory:

```powershell
python install.py
```

Use `python3 install.py` on macOS/Linux. The installer creates or reuses `.venv`,
installs dependencies there, checks access to a gated model file, and prints an
MCP configuration with absolute paths. It does not register clients automatically.
`EDITLENS_USE_CURRENT_PYTHON=1` opts into another interpreter.

The checkpoint is gated. Accept access on its
[model page](https://huggingface.co/pangram/editlens_roberta-large), then log in with
a Hugging Face read token:

```powershell
.venv\Scripts\hf.exe auth login
```

On macOS/Linux use `.venv/bin/hf auth login`. `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN` can also supply credentials. The first scoring request
downloads approximately 1.4 GB of weights into the Hugging Face cache.

For manual installation, create a virtual environment, install the appropriate
[PyTorch build](https://pytorch.org/get-started/locally/), then install
`requirements.txt` using that environment's Python. On Windows/Linux the
installer uses the CUDA 12.6 wheel index when `nvidia-smi` reports an NVIDIA GPU
and the much smaller CPU wheel otherwise; `EDITLENS_TORCH_INDEX` overrides the
index (ROCm, another CUDA version, a mirror). macOS wheels include Metal/MPS
support.

## Register an MCP client

Point each client to the same checkout and project interpreter. Example Windows
configuration; substitute the paths printed by the installer:

```json
{
  "mcpServers": {
    "editlens": {
      "command": "C:\\path\\editlens-mcp\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\editlens-mcp\\run_server.py"]
    }
  }
}
```

On macOS/Linux the command is `/path/editlens-mcp/.venv/bin/python`. The launcher
works from any current directory. It also switches an older client configuration
that names global Python into this checkout's `.venv`, when one exists (on
Windows it stays running as a thin parent, like the venv's own `python.exe`, so
the client never sees the process it launched exit early).

Restart existing MCP sessions after upgrading. A session that started before the
checkout changed cannot start a matching worker; its scoring calls say so
("EditLens was updated on disk after this MCP session started") instead of
failing obscurely. Saved chains are unaffected.

Call `detector_info` to check `backend`, `worker_pid`, `device`, `dtype`, scoring
profile, load state, database path, `hf_token_present` (environment variables or
a stored `hf auth login` token), and the worker's `runtime_dir` and `worker_log`.
Separate clients with matching settings should report the same `worker_pid`.
Status starts the worker and imports its native runtime but does not load model
weights. If the worker cannot start, `detector_info` still returns `ok: false`
with the error plus everything known locally, including the log path.

## Model sharing and resource use

Each stdio MCP frontend remains lightweight. A worker elected with OS file locks
owns the model and processes a bounded FIFO queue. It survives the frontend that
started it. Workers bind only to `127.0.0.1`, use a per-worker authentication token
and bypass HTTP proxy environment variables. Text is sent over this local
connection; it is not sent to an inference service. Model downloads contact Hugging
Face, and your MCP client has its own data handling policies.

Sharing requires a matching interpreter, source and model configuration within the
same runtime directory. Different databases can use the same worker. Different
devices, precision, batching, preprocessing, model revisions or environments can
create separate workers intentionally. Keep settings aligned across agents.
`EDITLENS_BACKEND=local` loads a separate model inside each MCP process.

At most 32 inference requests can wait in the queue. A full queue returns an error.
Expired queued work is cancelled; an already-running native inference may finish.
Requests that may have run are never replayed. The one automatic retry is a
request refused by a worker that was already exiting, which never ran it; that
request goes once to a fresh worker. A later call can restart a dead worker. An
unresponsive worker that still holds its lifetime lock is not replaced by a
duplicate. Workers accept only loopback requests addressed to `127.0.0.1` or
`localhost` that carry their token.

By default, weights unload after 300 idle seconds and the worker process exits
after 600 idle seconds. The model watchdog checks periodically, so unloading can
lag its threshold by up to 30 seconds. `detector_unload` releases the shared model
after preceding requests finish; later scoring reloads it. This affects every
client using that worker. It never starts a worker: with none running it simply
reports that nothing was loaded. Diagnostics are in the worker's `.log` file
(path in `detector_info`), rotated to `.log.1` once it exceeds 1 MiB.

## Tools and draft workflow

All 13 tools return structured data; execution failures return
`{"ok": false, "error": "...", "error_type": "..."}`. Schema validation errors
may be reported by MCP before the tool executes. Tools carry MCP annotations:
everything is local (`openWorldHint: false`), scoring and reading tools are
read-only, and only `chain_delete` is marked destructive. Chain and segment
names must not be blank.

| Tool | Main parameters | Purpose |
| --- | --- | --- |
| `detect` | `text`, `include_windows=false` | Score a document. |
| `detect_batch` | `texts` | Score candidates and return the lowest-scoring index; compare similar-length complete drafts. |
| `detect_spans` | `text`, `granularity="sentence"`, `top=10`, `min_words=25` | Explore independent fragment estimates. Paragraph mode preserves paragraph boundaries. |
| `detector_info` | — | Diagnose model, worker, environment and database state. |
| `detector_unload` | — | Release shared model weights. |
| `chain_create` | `name`, `target_score=0.25`, `goal=null`, `segments=null` | Create a chain; default segment is `main`. |
| `chain_submit` | `chain_id`, `text`, `segment="main"`, `note=null`, `span_feedback=true`, `span_top=5`, `span_min_words=25`, `branch_from=null` | Score and save the exact draft; return changes and review guidance. |
| `chain_status` | `chain_id` | Inspect best/latest scores, unstarted segments and suggested review. |
| `chain_history` | `chain_id`, `segment="main"`, `limit=30` | Recent steps oldest-first, plus whole-segment best/total metadata. |
| `chain_get_text` | `chain_id`, `segment="main"`, `step="best"` | Retrieve exact text; accepts `best`, `latest` or a step number. |
| `chain_assemble` | `chain_id`, `separator="\n\n"`, `include_text=true` | Join the lowest-scoring draft of each segment and score it. |
| `chain_list` | `limit=25` | List saved chains and segment progress. |
| `chain_delete` | `chain_id`, `segment=null` | Permanently delete a chain, or one segment and its drafts. |

`best` means **lowest-scoring**, not best writing. Assembly strips each segment's
edge whitespace and inserts the separator; `chain_get_text` preserves saved drafts
exactly. Assembly's `target_met` also requires every declared segment to have a
draft. A target is a workflow setting, not a calibrated authorship boundary.

Review writing for concrete problems before revising. `chain_submit` returns
`revision_state`, `stop_recommended`, `revision_progress` and `next_action`. It
recommends stopping score-driven edits for short input, unchanged drafts, targets
reached, significant regressions, or stalled progress. Review heuristics are less
than 0.01 improvement over five recent submissions and a review point at eight
submissions per segment. These are workflow rules, not confidence bounds or hard
limits; further drafts can still be saved. While other declared segments have no
draft yet, a finished or short section does not recommend stopping; the advice
names the sections still to draft. Suggested calls in `next_action` include the
`chain_id`, so they can be run as written. "Unchanged draft" compares against the
draft being revised (the `branch_from` step when given), and `delta_vs_parent`
reports the change against it.

Span analysis (`detect_spans`, `chain_submit` feedback) covers only the text the
document score uses: a reasoning block up to `</think>` and a discarded leading
boilerplate line are left out rather than ranked.

For multi-section work, finish the sections and call `chain_assemble` before
considering more revisions. Inspect the selected drafts for meaning, facts and
voice. A lower score does not prove the revision helped. To recover and branch:

```text
chain_get_text(chain_id=ch, segment="main", step=2)
chain_submit(chain_id=ch, segment="main", text=revised_text, branch_from=2)
```

History is append-only until explicitly deleted. `chain_history.best_step`
describes the whole segment even when that step is outside the returned window.

## Offsets and long documents

Span and window offsets index the original supplied string using Python character
indices. These differ from JavaScript UTF-16 indices for characters outside the
BMP. Responses with offsets include a `source_fingerprint`; request fresh offsets
after editing. Display snippets retain case with whitespace cleanup; the model
applies its preprocessing separately.

Inputs exceeding 512 tokens use overlapping windows. Every actual slice is
re-tokenized to ensure it fits. Overlap ownership is partitioned before weighted
aggregation to avoid double-counting. This aggregation is a server extension,
not a published EditLens calibration method. Headers/think text discarded by
reference preprocessing are outside the scored ranges.

## Upgrading existing chains

New chains record a `scoring_profile`. Submitting or assembling under a different
profile is rejected before inference, and the error names each field that differs
(for example `emoji_version` after upgrading the `emoji` package). Saved text,
history and status stay readable. Older chains without a profile used
whitespace-only preprocessing. Either:

1. Retrieve a chosen draft, create a new chain and submit it under `reference`; or
2. Set `EDITLENS_PREPROCESS=legacy` and restart the client to resume the old chain.

Legacy mode retains the old model inputs/window size. New labels still use
corrected argmax decoding, so they can differ from old saved labels. Precision,
dependency versions and custom model contents can also move scores; keep the
environment stable for comparisons. Pin `EDITLENS_REVISION` for custom checkpoints.

## Configuration

| Environment variable | Default / meaning |
| --- | --- |
| `EDITLENS_BACKEND` | `shared`; `local` loads within each MCP process. |
| `EDITLENS_PREPROCESS` | `reference`; `legacy` for old whitespace-only chains. Case-insensitive. |
| `EDITLENS_CHECKPOINT` | `pangram/editlens_roberta-large` |
| `EDITLENS_REVISION` | Pinned SHA above for the default checkpoint; otherwise the custom repo default. |
| `EDITLENS_BASE_MODEL` | `FacebookAI/roberta-large`, tokenizer/adapter fallback. |
| `EDITLENS_DEVICE` | Auto: CUDA → MPS → CPU. A failed accelerator probe falls back to CPU and reports why. |
| `EDITLENS_DTYPE` | `float32`; optional `float16` on GPU can change scores. |
| `EDITLENS_BATCH_SIZE` | `8`; values below one clamp to one. |
| `EDITLENS_IDLE_UNLOAD` | `300` seconds; zero or negative disables model auto-unload. |
| `EDITLENS_WORKER_IDLE` | `600` seconds; zero disables worker idle exit. The first client starting a worker sets this policy. |
| `EDITLENS_REQUEST_TIMEOUT` | `300` seconds for queueing plus inference; positive and at most 3600. |
| `EDITLENS_STARTUP_TIMEOUT` | `60` seconds for discovery/startup, capped by request timeout. |
| `EDITLENS_RUNTIME_DIR` | Per-user local runtime; keep it private and on a local filesystem. |
| `EDITLENS_DB` | Platform data directory below. |
| `EDITLENS_TRANSPORT` | `stdio`; other FastMCP transports need their own deployment/access configuration. |
| `EDITLENS_USE_CURRENT_PYTHON` | `1` bypasses automatic project `.venv` selection. |
| `EDITLENS_TORCH_INDEX` | Installer only: PyTorch wheel index URL, overriding the GPU/CPU choice. |
| `EDITLENS_SUITE_TIMEOUT` | Test runner only: seconds before a hung suite is killed and failed (`1800`). |

An empty or whitespace-only value means unset for every setting, because client
configurations often emit `""` for blank fields. Malformed batch-size/model-idle
numbers warn and use defaults. Invalid worker timeout settings
(`EDITLENS_REQUEST_TIMEOUT`, `EDITLENS_STARTUP_TIMEOUT`, `EDITLENS_WORKER_IDLE`)
fail at server start with the setting's name. An unknown `EDITLENS_PREPROCESS`
value also fails at start.

| Platform | Default database |
| --- | --- |
| Windows | `%LOCALAPPDATA%\editlens-mcp\chains.db` |
| macOS | `~/Library/Application Support/editlens-mcp/chains.db` |
| Linux | `$XDG_DATA_HOME/editlens-mcp/chains.db`, or `~/.local/share/editlens-mcp/chains.db` |

Runtime defaults to `%LOCALAPPDATA%\editlens-mcp\runtime` on Windows and
`$XDG_RUNTIME_DIR/editlens-mcp` or `~/.cache/editlens-mcp` elsewhere. `EDITLENS_DB`
expands home/environment variables and resolves relative paths at startup. A
database failure leaves scoring available; chain tools report the cause and
`detector_info.db_error` describes it. SQLite uses WAL where available.

## Verification

```powershell
.venv\Scripts\python.exe run_tests.py
```

On macOS/Linux use `.venv/bin/python run_tests.py`. The runner includes 15 suites:
MCP tools/subprocesses, guidance, workflow contracts, reference inference, shared
worker lifecycle, offsets, database races, branching, startup, windowing, precision
and GPU memory release. Real scoring needs access to the checkpoint.

`run_tests.py --no-model` runs only the 7 suites that need neither the checkpoint
nor a GPU; this is what CI runs on Windows, macOS and Linux with Python 3.10 and
3.13 (`.github/workflows/ci.yml`). Name suites to run a subset, for example
`run_tests.py tools offsets`. A suite that hangs is killed after
`EDITLENS_SUITE_TIMEOUT` seconds and reported as failed.

Tests use temporary databases and worker runtimes. `EDITLENS_TEST_DB` is the
explicit database override. Failed runs retain their scratch directory. The tests
establish software behavior and reference parity, not detector accuracy. Real
hardware verification has been on Windows with an RTX 2080; native macOS/Linux
verification has not been performed in this repair.

## Licence

Repository code: MIT, see `LICENSE`. Model weights have their own
[CC-BY-NC-SA-4.0 terms](https://huggingface.co/pangram/editlens_roberta-large).
Weights are downloaded separately and are not redistributed here.
