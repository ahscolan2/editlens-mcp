# Research and validation notes

Reviewed during the September 2026 repair. The implementation follows the public
RoBERTa reference where possible; workflow extensions are identified separately.

## Sources reviewed

| Source | Relevant evidence |
| --- | --- |
| [EditLens paper, arXiv 2510.03154v1](https://arxiv.org/html/2510.03154v1) | Task definition, data generation, evaluation, limitations and appendices. |
| [Official repository at 05a588f](https://github.com/pangramlabs/EditLens/tree/05a588f15d792330ccaf91be8ee4fdb54ce26835) | Training, inference, preprocessing and configurations. |
| [RoBERTa training configuration](https://github.com/pangramlabs/EditLens/blob/05a588f15d792330ccaf91be8ee4fdb54ce26835/configs/roberta.yaml) | Four buckets; max length 512; minimum 75 words; target discretization settings. |
| [Reference preprocessing](https://github.com/pangramlabs/EditLens/blob/05a588f15d792330ccaf91be8ee4fdb54ce26835/scripts/preprocess.py) | Exact transformation order and word-count regex. |
| [Reference training script](https://github.com/pangramlabs/EditLens/blob/05a588f15d792330ccaf91be8ee4fdb54ce26835/scripts/train.py) | The minimum-length filter runs before preprocessing/tokenization. |
| [Public demo application](https://huggingface.co/spaces/multimodalart/EditLens/blob/main/app.py) | Continuous expected-index score, argmax label, illustrative dataset examples. |
| [Checkpoint](https://huggingface.co/pangram/editlens_roberta-large/tree/f93e1ace74528cfb48f337ab2fe946fb71a728cb) | Pinned weights, configuration and tokenizer; separately licensed, gated download. |
| [Mac-oriented fork](https://github.com/ahscolan2/editlens-llama-mcp/tree/cae6f78e0f3d9a2c2ae4a1d84ccf7989892b1c24) | Compared architecture, installation and chain-store fixes. |

Runtime references included the official documentation for
[Transformers classification models](https://huggingface.co/docs/transformers/model_doc/roberta),
[Hugging Face downloads and file metadata](https://huggingface.co/docs/huggingface_hub/package_reference/file_download),
[FastMCP server execution](https://gofastmcp.com/deployment/running-server),
[Python subprocesses](https://docs.python.org/3/library/subprocess.html),
[queues](https://docs.python.org/3/library/queue.html),
[Windows file locks](https://docs.python.org/3/library/msvcrt.html),
[HTTP serving](https://docs.python.org/3/library/http.server.html),
[sqlite3 transactions](https://docs.python.org/3/library/sqlite3.html), and
[SQLite isolation](https://sqlite.org/isolation.html).

## What the evidence supports

The paper's task is estimating the extent of editing in a document. Its headline
results concern a Mistral Small 24B system; they do not establish the performance
of this 355M-parameter RoBERTa checkpoint. The paper does not establish sentence
provenance detection, writing-quality evaluation, or a calibrated threshold for
this server's automatic revision workflow. These limits informed removal of
mandatory rewrite instructions.

The training targets discretize an editing-similarity measure into buckets. The
inference score averages bucket indices under the probability distribution; the
label uses the maximum-probability bucket. It is incorrect to recover the label
by rounding the averaged index, especially for multimodal distributions.

The reference preprocessing order is:

1. Convert recognized emoji to textual shortcodes.
2. If a closing think tag exists, retain the portion after the first and before
   a second closing tag, matching the upstream split behavior.
3. Remove a leading nonblank line when the reference's case-sensitive boilerplate
   prefixes match and another nonblank line exists.
4. Lowercase the whole string and collapse whitespace.

Some choices are surprising but intentional for inference parity. For example,
an emoji shortcode before `Sure` prevents the prefix match. Greek final sigma
requires whole-string lowercasing; per-character lowercasing is not equivalent.
Unicode lowercase and emoji expansions need ranges, rather than one-to-one
character offsets, to map scored text back to the saved document.

The published training script filters raw input at 75 words using `\b\w+\b`.
The demo warns below 50 words. Neither proves a confidence cutoff. This server
uses 75 as a conservative workflow floor and checks the smaller of readable input
words and remaining model-input words. That additional check prevents long removed
headers, emoji names and split contractions from overstating usable text length.
It does not change model preprocessing or scores.

## Deliberate server extensions

- Long documents use overlapping, re-tokenization-checked windows and aggregation
  by exclusive character-range ownership/word weights. The demo truncates to one
  window. Long-document aggregation is not a validated calibration method.
- Independent span scoring is exploratory. High-scoring spans do not establish
  which text caused the document result, and scores across lengths can differ in
  either direction.
- Five-submission plateaus, the eight-submission review point, a 0.05 regression
  prompt and user-selected targets are workflow heuristics. They do not prevent
  saving a draft and do not claim statistical confidence.
- Chains retain exact draft strings. New chains record scoring profiles; incompatible
  submissions require a new chain or an explicit legacy configuration.
- Both previous repositories used process-local locks. Neither implemented a
  machine-wide worker for independent stdio processes. The new worker owns the
  native runtime under an OS lifetime lock and serializes requests through a queue.

## Verification evidence and practical limits

The final complete run passed all 15 suites (exit 0, 179.92 seconds). Its real
worker suite passed 17 tests, and the MCP subprocess suite observed one worker PID
across two independent frontends while recovering the exact draft and branch.
The launcher was subsequently checked with one frontend starting through global
Python and the other through the project environment; both shared the worker.
`pip check` reported no broken requirements.

Real CUDA inference was compared with an independent direct model forward pass
using explicitly prepared reference inputs. Single and batched scores matched
within 0.0002, and argmax labels agreed. Long emoji inputs were checked for the
512-token limit and valid original-text offsets. A separate seeded comparison of
6,000 Unicode/emoji/header combinations matched the downloaded upstream
preprocessor exactly. These are implementation checks, not an accuracy benchmark.

The two examples supplied by the public demo (identified there as test-set review
rows) were also scored. This is an illustrative sanity check, not an independent
held-out evaluation:

| Demo example | Readable words | Legacy score | Reference score | Reference label |
| --- | --- | --- | --- | --- |
| Human-written review | 94 | 0.020123 | 0.019630 | Human-written |
| AI-generated review | 79 | 0.999610 | 0.999607 | Fully AI-generated |

The worker lifecycle test exercises three simultaneous client processes, one
worker PID, survival after client exit, forced worker death and restart, equal
scores, and idle shutdown. Protocol tests cover authentication, bounded queueing,
responsive health checks, timeout cancellation, no replay and proxy bypass.
The MCP subprocess suite additionally checks two independent stdio connections,
shared worker identity, exact saved-text retrieval and branch metadata.

Database tests cover concurrent processes, deleted segments during scoring,
transaction rollback, consistent read snapshots and malformed database paths.
Workflow tests distinguish low scores from short-text confidence, preserve draft
content and reject mixed scoring profiles before inference.

The repaired Windows environment is isolated in `.venv`:

| Component | Verified version |
| --- | --- |
| Python | 3.13.15 |
| PyTorch | 2.14.0+cu126 |
| Transformers | 5.17.0 |
| FastMCP | 3.4.7 |
| Hugging Face Hub | 1.31.0 |
| Emoji | 2.15.0 |
| Pydantic / core | 2.13.5 / 2.46.5 |
| Tokenizers | 0.23.2 |

The machine uses an NVIDIA RTX 2080. The original global environment had a
Pydantic/core incompatibility and CPU-only PyTorch. It was left unchanged.
Native macOS/Linux hardware and out-of-domain detector accuracy remain untested.
Reduced precision, dependency changes and unpinned custom checkpoints can change
scores; a profile is not a substitute for keeping the inference environment stable.
