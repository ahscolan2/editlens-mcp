# EditLens MCP

Local AI-text detector as an MCP server, plus a durable chain store so a model can
run long write → score → revise loops.

Backed by [`pangram/editlens_roberta-large`](https://huggingface.co/pangram/editlens_roberta-large):
a RoBERTa-large 4-bucket classifier (Human-written / Lightly AI-edited / Heavily
AI-edited / Fully AI-generated). The continuous score is the expected bucket index
under the softmax, normalised to `[0, 1]` — the same formula the official demo Space
uses. **0.0 = human-written, 1.0 = fully AI-generated.**

Everything runs on your machine. No text leaves it.

## Status — installed and registered

Verified working on this machine: torch 2.13.0+cu126, transformers 5.15.0, model
running on CUDA in float16, logged in to Hugging Face as `lemonsad`.

Registered in all three clients (backups written alongside each as `*.editlens-bak`):

| Config file | Client |
| --- | --- |
| `C:\Users\Aryan\.claude.json` | Claude Code |
| `C:\Users\Aryan\.gemini\antigravity\mcp_config.json` | Antigravity app |
| `C:\Users\Aryan\.gemini\config\mcp_config.json` | agy CLI |

All three use the same entry. `run_server.py` sets its own import path, so no
working-directory setting is needed — that's the part MCP clients disagree about:

```json
{
  "editlens": {
    "command": "C:\\Users\\Aryan\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
    "args": ["C:\\Users\\Aryan\\MCP-EditLens\\run_server.py"]
  }
}
```

### Rebuilding from scratch

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

```bash
pip install -r C:\Users\Aryan\MCP-EditLens\requirements.txt
```

The checkpoint is gated — accept the licence on the model page, then `hf auth login`.
Verify with `python smoke_test.py --real` (downloads ~1.4 GB the first time, then
caches to `C:\Users\Aryan\.cache\huggingface` permanently).

## Testing

```bash
python C:\Users\Aryan\MCP-EditLens\run_tests.py
```

Five suites: plumbing and one real scoring pass; all 13 tools over an in-memory client
including error paths; precision, windowing, edge cases, determinism and chains; GPU idle
unload and threading under load; and the real path — the server as a stdio subprocess,
which is what Claude Code and Antigravity actually do.

Run this after changing anything. The subprocess suite in particular catches failures the
in-process ones cannot, because tool functions run on a worker thread there.

## Tools

### Scoring

| Tool | What it does |
| --- | --- |
| `detect(text)` | Score one text. Inputs over 512 tokens are split into overlapping windows and combined by word-count-weighted average. |
| `detect_batch(texts)` | Score N texts in one forward pass. Returns `best_index` — generate several variants, keep the lowest. |
| `detect_spans(text, granularity, top)` | Split into sentences or paragraphs, score each, return worst-first. Tells you *which passages* drive the score. |
| `detector_info()` | Checkpoint, device, dtype, load state, VRAM in use, token visibility. Call this first when something errors. |
| `detector_unload()` | Release GPU memory now. Rarely needed — see idle unloading below. |

### Chains

A chain is a write → score → revise loop whose state lives in SQLite, not in the
model's context. That is what makes long chains practical: step 200 costs the same
context as step 2. A chain holds ordered **segments** (sections of a document); each
segment accumulates numbered **steps** (revisions).

| Tool | What it does |
| --- | --- |
| `chain_create(name, target_score, goal, segments)` | Open a chain. `segments` for multi-section documents. |
| `chain_submit(chain_id, text, segment, note)` | Score and store a draft. Returns score, movement vs. previous and best, target status, and worst spans — **not** the draft text. |
| `chain_status(chain_id)` | Per-segment best/latest scores and which segments still miss target. |
| `chain_history(chain_id, segment)` | Score trajectory. Numbers and notes only, no text. |
| `chain_get_text(chain_id, segment, step)` | Retrieve a stored draft — `"best"`, `"latest"`, or a step number. Survives restarts. |
| `chain_assemble(chain_id)` | Join the best draft of every segment and score the whole document. |
| `chain_list()` / `chain_delete(id)` | Manage stored chains. |

## How a long chain runs

Single-segment revision loop:

```
chain_create("essay", target_score=0.2)   → ch_a1b2c3
chain_submit(ch, draft_1)                 → 0.87  worst_spans: [...]   rewrite those
chain_submit(ch, draft_2, note="cut the hedging")  → 0.61  Δ -0.26
chain_submit(ch, draft_3)                 → 0.34  Δ -0.27
chain_submit(ch, draft_4)                 → 0.18  target_met ✓
chain_get_text(ch, step="best")           → the winning draft
```

Multi-segment (horizontal) composition:

```
chain_create("report", segments=["intro","method","results","discussion"])
chain_submit(ch, text, segment="intro")      ← each section revises independently
chain_submit(ch, text, segment="method")     ← chain_status shows what's pending
...
chain_assemble(ch)   ← joins best-of-each and scores the full document
```

Assemble at the end always. A document can score higher than any of its parts,
because the model sees consistency across sections that it cannot see in one section
alone.

Widening instead of deepening: generate several candidates per step and use
`detect_batch` to keep the best one before submitting it. That converges in fewer
chain steps than revising a single line of drafts.

## Notes

- **Device and precision**: auto-selects CUDA; runs float32 by default, the precision the
  checkpoint is published in. Measured on the RTX 2080, float16 would move scores by at most
  0.001 and saves nothing on single paragraphs:

  | | float16 | float32 | speedup |
  | --- | --- | --- | --- |
  | one paragraph | 14.2 ms | 14.1 ms | none |
  | long document | 34.5 ms | 86.7 ms | 2.5x |
  | batch of 16 | 29.2 ms | 45.7 ms | 1.6x |
  | peak VRAM | 709 MB | 1408 MB | |

  Single-text scoring is dominated by overhead, not math, so float16 only wins on long
  documents and batches — and 1.4 GB fits the 8 GB card easily. Set `EDITLENS_DTYPE=float16`
  if you later run big batches and want the speed back.
- **GPU memory is released when idle.** Nothing is loaded until the first `detect` call
  (0 MB until then). After 5 minutes with no calls the model unloads itself and hands the
  ~1.4 GB back; the next call reloads it in about 2 s. Tune with `EDITLENS_IDLE_UNLOAD`
  (seconds, `0` disables), or call `detector_unload` to free it immediately.
- **Startup imports torch on the main thread**, which takes ~2 s. This is deliberate: MCP
  servers run tool functions in worker threads, and importing torch from a worker thread
  hangs indefinitely on Windows — the client times out and the server looks dead. Do not
  make these imports lazy again.
- **First `detect` takes ~2 s** while weights load from the local cache; subsequent calls
  are milliseconds.
- **Requests are serialised.** HuggingFace's Rust tokenizer raises `Already borrowed` if
  two threads encode at once, so one request runs at a time. Inference is 15–90 ms, and
  two clients (Claude Code + Antigravity) can safely share one server.
- **Short text is noisy.** This is a document-level model. `detect_spans` merges
  sentences up to ~25 words for that reason; treat anything under ~40 words as unreliable.
- **Scores are not ground truth.** EditLens estimates degree of AI editing; it has false
  positives, particularly on formal or technical prose. A low score means "this checkpoint
  finds it human-like", not "this is undetectable" — other detectors disagree with it
  routinely.
- **It is not easily talked out of a verdict.** Measured on this machine: corporate-register
  AI prose scored 0.9995. An AI rewrite of the same content into deliberately casual,
  first-person voice — contractions, short sentences, a self-deprecating aside — still
  scored 0.9103. Genuinely offhand human writing scored 0.18. Surface-level voice changes
  move the score far less than they feel like they should, so expect chains to need real
  structural rewriting rather than a few word swaps.
- **Licence**: the checkpoint is CC-BY-NC-SA-4.0. Non-commercial use only.

## Config

| Env var | Default |
| --- | --- |
| `EDITLENS_CHECKPOINT` | `pangram/editlens_roberta-large` |
| `EDITLENS_DEVICE` | auto (`cuda` if available) |
| `EDITLENS_DTYPE` | `float32` (as published). Set `float16` for ~2.5x on long docs, at 0.001 score cost. |
| `EDITLENS_BATCH_SIZE` | `8` |
| `EDITLENS_IDLE_UNLOAD` | `300` seconds of inactivity before GPU memory is released (`0` = never) |
| `EDITLENS_DB` | `%LOCALAPPDATA%\editlens-mcp\chains.db` |
| `HF_TOKEN` | falls back to the `hf auth login` cache |
