"""EditLens detector: lazy-loaded local RoBERTa-large AI-text classifier.

The checkpoint is a 4-bucket sequence classifier. A single continuous score is
derived the same way the official demo Space does it: the expected bucket index
under the softmax, normalised to [0, 1].
"""

from __future__ import annotations

import gc
import hashlib
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Sequence

CHECKPOINT = os.environ.get("EDITLENS_CHECKPOINT", "pangram/editlens_roberta-large")
BASE_MODEL = os.environ.get("EDITLENS_BASE_MODEL", "FacebookAI/roberta-large")
MAX_LENGTH = 512
# Room for <s> and </s>, plus slack. Windows are chosen on the full document's
# token stream but handed to the model as CHARACTER slices, which get
# re-tokenised; a slice loses the leading-space context of its first token, so it
# can come back longer than it went in (measured: 510 -> 514) and then be
# silently truncated. The extra margin keeps re-tokenisation inside the limit.
MAX_CONTENT_TOKENS = MAX_LENGTH - 22

BUCKET_NAMES: dict[int, list[str]] = {
    2: ["Human-written", "AI-generated"],
    3: ["Human-written", "AI-edited", "Fully AI-generated"],
    4: ["Human-written", "Lightly AI-edited", "Heavily AI-edited", "Fully AI-generated"],
    5: [
        "Human-written",
        "Lightly AI-edited",
        "Moderately AI-edited",
        "Heavily AI-edited",
        "Fully AI-generated",
    ],
}


class DetectorUnavailable(RuntimeError):
    """Raised when the model cannot be loaded, with actionable setup guidance."""


_warmup_error: str | None = None
_warmed = False


def warmup_imports() -> str | None:
    """Import torch and transformers on the MAIN thread, at startup.

    MCP servers run sync tool functions in a worker thread, and importing large
    native extensions like torch from a non-main thread hangs on Windows (2 s on
    the main thread vs. indefinite in a worker). Clients time the call out and the
    server looks dead. Doing the imports here makes every later import a cache hit.

    Returns None on success, or an error string if the imports failed.
    """
    global _warmed, _warmup_error
    if _warmed:
        return _warmup_error
    # Must be set before torch is imported. A handful of ops still have no Metal
    # kernel; without this the process dies instead of quietly using the CPU for
    # that one op.
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch  # noqa: F401, PLC0415
        from transformers import (  # noqa: F401, PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        pick_device(torch)  # force the accelerator probe here too
    except Exception as exc:  # noqa: BLE001 - surfaced via detector_info
        _warmup_error = f"{type(exc).__name__}: {exc}"
    _warmed = True
    return _warmup_error


def pick_device(torch) -> str:
    """CUDA, else Apple Silicon (Metal), else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return "mps"
    return "cpu"


def accelerator_of(device: str) -> str:
    """'cuda' | 'mps' | 'cpu' for a device string like 'cuda:1'."""
    return device.split(":", 1)[0]


@dataclass
class Verdict:
    score: float
    bucket: int
    label: str
    probs: list[float] = field(default_factory=list)
    word_count: int = 0
    char_count: int = 0
    truncated_windows: int = 1

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "bucket": self.bucket,
            "label": self.label,
            "probs": [round(p, 4) for p in self.probs],
            "word_count": self.word_count,
            "char_count": self.char_count,
            "windows": self.truncated_windows,
        }


_WS_RUN = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")
_WORD = re.compile(r"\b[\w'’-]+\b", re.UNICODE)


def clean_text(text: str) -> str:
    """Conservative normalisation. Whitespace only -- never rewrites wording,
    because that would change what the detector actually sees."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def clean_text_with_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    """clean_text(), plus each cleaned char's source RANGE in the original.

    Callers get char offsets for the spans we flag. Those offsets used to index
    the cleaned string, which the caller never receives -- so on any text with
    CRLF line endings or double spaces they pointed at the wrong characters, and
    a client splicing a rewrite at them would corrupt its own document.

    `ranges[i]` is the half-open `(start, end)` slice of `text` that produced
    cleaned character `i`. A range, not a single index, because cleaning is
    many-to-one: `\\r\\n` collapses to one `\\n` and a run of five spaces to one
    space. Storing only the start truncated the span before the run's last
    character, which for CRLF left an orphaned newline behind on every splice.
    """
    # 1. newline normalisation
    c1: list[str] = []
    r1: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "\r":
            width = 2 if (i + 1 < n and text[i + 1] == "\n") else 1
            c1.append("\n")
            r1.append((i, i + width))
            i += width
            continue
        c1.append(text[i])
        r1.append((i, i + 1))
        i += 1
    s1 = "".join(c1)

    # 2. collapse runs of horizontal whitespace to one space
    c2: list[str] = []
    r2: list[tuple[int, int]] = []
    pos = 0
    for m in _WS_RUN.finditer(s1):
        c2.extend(s1[pos : m.start()])
        r2.extend(r1[pos : m.start()])
        c2.append(" ")
        r2.append((r1[m.start()][0], r1[m.end() - 1][1]))
        pos = m.end()
    c2.extend(s1[pos:])
    r2.extend(r1[pos:])

    # 3. rstrip each line
    c3: list[str] = []
    r3: list[tuple[int, int]] = []
    line_start = 0
    for k in range(len(c2) + 1):
        if k == len(c2) or c2[k] == "\n":
            keep = len("".join(c2[line_start:k]).rstrip())
            c3.extend(c2[line_start : line_start + keep])
            r3.extend(r2[line_start : line_start + keep])
            if k < len(c2):
                # The newline absorbs whatever trailing whitespace was dropped.
                drop_end = r2[k - 1][1] if k > line_start + keep else r2[k][0]
                c3.append("\n")
                r3.append((min(r2[k][0], drop_end), r2[k][1]))
            line_start = k + 1
    s3 = "".join(c3)

    # 4. collapse 3+ blank lines to one blank line
    c4: list[str] = []
    r4: list[tuple[int, int]] = []
    pos = 0
    for m in _BLANKS.finditer(s3):
        c4.extend(s3[pos : m.start()])
        r4.extend(r3[pos : m.start()])
        c4.extend("\n\n")
        # Second newline absorbs the whole collapsed run, so nothing is orphaned.
        r4.append(r3[m.start()])
        r4.append((r3[m.start() + 1][0], r3[m.end() - 1][1]))
        pos = m.end()
    c4.extend(s3[pos:])
    r4.extend(r3[pos:])
    s4 = "".join(c4)

    # 5. strip
    a, b = 0, len(s4)
    while a < b and s4[a].isspace():
        a += 1
    while b > a and s4[b - 1].isspace():
        b -= 1
    return s4[a:b], r4[a:b]


def map_span(
    ranges: list[tuple[int, int]], start: int, end: int, original_len: int
) -> tuple[int, int]:
    """Translate a [start, end) span over cleaned text into original coordinates."""
    if not ranges or start >= len(ranges) or end <= start:
        return 0, 0
    o_start = ranges[start][0]
    o_end = ranges[min(end, len(ranges)) - 1][1]
    return o_start, min(max(o_end, o_start), original_len)


def text_fingerprint(text: str) -> str:
    """Short digest of the exact string a set of offsets was computed against.

    Offsets are only valid for the text that produced them: rewrite one sentence
    and every later offset shifts. Returning a fingerprint lets a caller detect
    that its cached offsets are stale instead of splicing at coordinates that
    have silently moved.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


class EditLensDetector:
    """Thread-safe lazy wrapper around the local checkpoint."""

    def __init__(
        self,
        checkpoint: str = CHECKPOINT,
        base_model: str = BASE_MODEL,
        device: str | None = None,
        batch_size: int = 8,
        dtype: str | None = None,
        idle_unload_seconds: float = 300.0,
    ) -> None:
        self.checkpoint = checkpoint
        self.base_model = base_model
        self._requested_device = device
        self._requested_dtype = dtype
        self.batch_size = batch_size
        self.idle_unload_seconds = idle_unload_seconds

        # Re-entrant: detect() -> detect_many() -> detect() nests the guard.
        self._lock = threading.RLock()
        # HuggingFace's Rust tokenizer raises "Already borrowed" if two threads
        # encode at once, so whole requests are serialised. Inference is 15-90 ms,
        # and two MCP clients can share one server process.
        self._infer_lock = threading.RLock()
        self._loaded = False
        self._load_error: str | None = None
        self._last_used = 0.0
        self._inflight = 0
        self._watchdog: threading.Thread | None = None
        self._unloads = 0
        self._device_fallback: str | None = None

        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device = "cpu"
        self.dtype = "float32"
        self.n_buckets = 4
        self.bucket_names: list[str] = BUCKET_NAMES[4]

    # ---------------------------------------------------------------- loading

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _planned_device(self) -> tuple[str, str]:
        """What device/dtype a load would pick right now. Reported before the
        model is loaded so `detector_info` never claims CPU on a CUDA machine."""
        if self._loaded:
            return self.device, self.dtype
        try:
            import torch  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            # A broken Windows torch install raises OSError (WinError 126,
            # "error loading fbgemm.dll"), not ImportError. This is the tool
            # people call to diagnose that, so it must not raise.
            return f"unavailable ({type(exc).__name__})", "unknown"
        device = self._requested_device or pick_device(torch)
        return device, self._resolve_dtype_name(device)

    def _resolve_dtype_name(self, device: str) -> str:
        """Default float32 -- the precision the checkpoint is published in.

        float16 is ~2.5x faster on long documents but identical on single
        paragraphs (14 ms either way), so the default favours running the weights
        unconverted. Set EDITLENS_DTYPE=float16 to trade 0.001 of score accuracy
        for speed on long or batched work. float16 needs a GPU (CUDA or Metal).
        """
        requested = (self._requested_dtype or "").lower()
        if not requested:
            return "float32"
        if requested in {"float16", "fp16", "16"}:
            return "float16" if accelerator_of(device) in {"cuda", "mps"} else "float32"
        if requested in {"float32", "fp32", "32"}:
            return "float32"
        # Unrecognised value: fall back rather than crash, but say so once.
        print(
            f"[editlens] warning: unknown EDITLENS_DTYPE={requested!r}; using float32",
            file=sys.stderr,
        )
        return "float32"

    def info(self) -> dict:
        device, dtype = self._planned_device()
        return {
            "checkpoint": self.checkpoint,
            "base_model": self.base_model,
            "loaded": self._loaded,
            "load_error": self._load_error,
            "device": device,
            "dtype": dtype,
            "n_buckets": self.n_buckets,
            "bucket_names": self.bucket_names,
            "max_length": MAX_LENGTH,
            "batch_size": self.batch_size,
            "hf_token_present": bool(
                os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            ),
            "platform": sys.platform,
            "accelerator": accelerator_of(device),
            "device_fallback": self._device_fallback,
            "cpu_backend": self._cpu_backend(),
            "warmup_error": _warmup_error,
            "idle_unload_seconds": self.idle_unload_seconds,
            "idle_seconds": (
                round(time.monotonic() - self._last_used, 1) if self._last_used else None
            ),
            "vram_mb": self._vram_mb(),
            "auto_unloads": self._unloads,
        }

    @staticmethod
    def _cpu_backend() -> dict | None:
        """Which BLAS the CPU path uses, and how many threads.

        On Apple Silicon this should report Accelerate, which routes matrix
        multiplies through the AMX units -- so CPU fallback is much faster than
        the name suggests. If it reports something else, that is worth knowing
        before blaming the model for slow inference.
        """
        try:
            import torch  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return None
        cfg = ""
        try:
            cfg = torch.__config__.show()
        except Exception:  # noqa: BLE001
            pass
        blas = "unknown"
        for name in ("Accelerate", "MKL", "OpenBLAS", "BLIS", "Eigen"):
            if name.lower() in cfg.lower():
                blas = name
                break
        try:
            threads = torch.get_num_threads()
        except Exception:  # noqa: BLE001
            threads = None
        return {"blas": blas, "threads": threads}

    def _vram_mb(self) -> float | None:
        if not self._loaded or self.torch is None:
            return None
        accel = accelerator_of(self.device)
        try:
            if accel == "cuda":
                return round(self.torch.cuda.memory_allocated() / 1024**2, 1)
            if accel == "mps":
                return round(self.torch.mps.current_allocated_memory() / 1024**2, 1)
        except Exception:  # noqa: BLE001 - reporting must never break info()
            return None
        return None

    def _empty_cache(self) -> None:
        accel = accelerator_of(self.device)
        try:
            if accel == "cuda":
                self.torch.cuda.empty_cache()
            elif accel == "mps":
                self.torch.mps.empty_cache()
        except Exception:  # noqa: BLE001 - best effort
            pass

    @contextmanager
    def _active(self):
        """Serialise a request and keep the idle watchdog from unloading mid-call.

        `_infer_lock` is held for the whole request (tokeniser + forward pass);
        `_lock` is taken only briefly to update the in-flight counter, so the
        watchdog can always observe state without waiting on inference.
        """
        with self._infer_lock:
            with self._lock:
                self._inflight += 1
                self._last_used = time.monotonic()
            try:
                yield
            finally:
                with self._lock:
                    self._inflight -= 1
                    self._last_used = time.monotonic()

    # LOCK ORDER: always _infer_lock before _lock. _active() takes them in that
    # order, so unload() and the watchdog must too -- acquiring _lock first and
    # then waiting on _infer_lock deadlocks against an in-flight request.

    def _unload_locked(self) -> bool:
        """Caller must already hold _infer_lock and _lock."""
        if not self._loaded:
            return False
        self.model = None
        self.tokenizer = None
        self._loaded = False
        gc.collect()
        if self.torch is not None:
            self._empty_cache()
        return True

    def unload(self) -> bool:
        """Drop the model and release GPU memory. Reloads on the next call.

        Holds `_infer_lock` so it cannot null out the model or tokenizer while a
        request is mid-flight -- otherwise a manual unload during traffic
        surfaces as `'NoneType' object is not callable`.
        """
        with self._infer_lock:
            with self._lock:
                return self._unload_locked()

    def _start_watchdog(self) -> None:
        if self.idle_unload_seconds <= 0 or self._watchdog is not None:
            return

        def loop() -> None:
            tick = min(30.0, max(5.0, self.idle_unload_seconds / 4))
            while True:
                time.sleep(tick)
                # Don't block traffic waiting to unload; try again next tick.
                if not self._infer_lock.acquire(timeout=1.0):
                    continue
                try:
                    with self._lock:
                        idle = time.monotonic() - self._last_used
                        if (
                            self._inflight == 0
                            and idle > self.idle_unload_seconds
                            and self._unload_locked()
                        ):
                            self._unloads += 1
                finally:
                    self._infer_lock.release()

        self._watchdog = threading.Thread(
            target=loop, daemon=True, name="editlens-idle-unload"
        )
        self._watchdog.start()

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                self._load()
            except DetectorUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller as guidance
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise DetectorUnavailable(self._setup_hint(exc)) from exc
            self._loaded = True
            self._load_error = None
            self._last_used = time.monotonic()
            self._start_watchdog()

    @staticmethod
    def torch_install_command() -> str:
        """Platform-correct pip line. Mac wheels ship Metal support; the CUDA
        index URL exists only for Windows and Linux."""
        if sys.platform == "darwin":
            return "pip install torch"
        return "pip install torch --index-url https://download.pytorch.org/whl/cu126"

    def _setup_hint(self, exc: Exception) -> str:
        msg = str(exc)
        lines = [f"Could not load '{self.checkpoint}': {type(exc).__name__}: {msg}"]
        if "torch" in msg or "transformers" in msg or isinstance(exc, ModuleNotFoundError):
            lines.append(
                f"Install deps:  {self.torch_install_command()}"
                "  &&  pip install fastmcp transformers safetensors"
            )
        if (
            "401" in msg
            or "403" in msg
            or "gated" in msg.lower()
            or "restricted" in msg.lower()
            or "is not a valid model identifier" in msg
            or "private repository" in msg.lower()
        ):
            lines.append(
                f"The repo is GATED. Accept the licence at https://huggingface.co/{self.checkpoint} "
                "then set HF_TOKEN to a read token (hf auth login, or set the env var)."
            )
        return "\n".join(lines)

    def _validate_device(self, model, tokenizer, torch, device: str, dtype_name: str):
        """Run one tiny forward pass; fall back to CPU if the accelerator fails."""
        try:
            probe = tokenizer("ok", return_tensors="pt").to(device)
            with torch.no_grad():
                model(**probe)
            return model, device, dtype_name
        except Exception as exc:  # noqa: BLE001
            if accelerator_of(device) == "cpu":
                raise
            print(
                f"[editlens] {device} failed its warm-up pass "
                f"({type(exc).__name__}: {exc}); falling back to CPU.",
                file=sys.stderr,
            )
            self._device_fallback = f"{device} -> cpu: {type(exc).__name__}"
            # float16 on CPU is slow and poorly supported; go back to float32.
            return model.to(device="cpu", dtype=torch.float32).eval(), "cpu", "float32"

    def _load(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        kwargs = {"token": token} if token else {}

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, **kwargs)
        except Exception as primary:  # tokenizer files may live only on the base repo
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.base_model, **kwargs)
            except Exception:
                raise primary from None

        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, **kwargs)
        except Exception as primary:
            # Fall back to the PEFT-adapter layout if the repo ships an adapter
            # only. If that fallback cannot even start, re-raise the ORIGINAL
            # error: peft is not installed by default, so letting its
            # ModuleNotFoundError win means every real failure (gated repo, bad
            # token, no network) is reported as "No module named 'peft'" and the
            # gated-repo guidance below becomes unreachable.
            try:
                from peft import PeftConfig, PeftModel  # noqa: PLC0415

                cfg = PeftConfig.from_pretrained(self.checkpoint, **kwargs)
                n_labels = getattr(cfg, "num_labels", None) or 4
                base = AutoModelForSequenceClassification.from_pretrained(
                    cfg.base_model_name_or_path or self.base_model,
                    num_labels=n_labels,
                    **kwargs,
                )
                model = PeftModel.from_pretrained(
                    base, self.checkpoint, **kwargs
                ).merge_and_unload()
            except Exception:
                raise primary from None

        if self._requested_device:
            device = self._requested_device
        else:
            device = pick_device(torch)

        dtype_name = self._resolve_dtype_name(device)
        dtype = torch.float16 if dtype_name == "float16" else torch.float32
        model = model.to(device=device, dtype=dtype).eval()

        # Prove the accelerator can actually run this model before serving. Metal
        # in particular can accept .to("mps") and then fail on the first forward
        # pass; discovering that here costs one tiny inference, whereas
        # discovering it later turns every detect call into an error.
        model, device, dtype_name = self._validate_device(
            model, tokenizer, torch, device, dtype_name
        )

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = str(dtype).replace("torch.", "")
        self.n_buckets = int(model.config.num_labels)
        self.bucket_names = BUCKET_NAMES.get(
            self.n_buckets, [f"bucket_{i}" for i in range(self.n_buckets)]
        )

    # -------------------------------------------------------------- inference

    def _score_batch(self, texts: Sequence[str]) -> list[tuple[float, list[float]]]:
        """Raw model pass. Each text must already fit in one window."""
        self.ensure_loaded()
        torch = self.torch
        # Bind locally: the idle watchdog may null these out between batches.
        model, tokenizer = self.model, self.tokenizer
        out: list[tuple[float, list[float]]] = []
        labels = torch.arange(self.n_buckets, dtype=torch.float32)

        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            inputs = tokenizer(
                chunk,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            for row in probs:
                score = float((row @ labels).item() / (self.n_buckets - 1))
                out.append((score, [float(p) for p in row]))
        return out

    def windows(self, text: str, overlap_tokens: int = 64) -> list[tuple[int, int, str]]:
        """Split text into (start_char, end_char, chunk) windows that fit the model."""
        self.ensure_loaded()
        # verbose=False: we tokenise the whole document on purpose to find split
        # points, so the "longer than max sequence length" warning is expected noise.
        enc = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        if len(ids) <= MAX_CONTENT_TOKENS:
            return [(0, len(text), text)]

        step = max(1, MAX_CONTENT_TOKENS - overlap_tokens)
        spans: list[tuple[int, int, str]] = []
        pos = 0
        while pos < len(ids):
            end_tok = min(pos + MAX_CONTENT_TOKENS, len(ids))
            c0 = offsets[pos][0]
            c1 = offsets[end_tok - 1][1]
            spans.append((c0, c1, text[c0:c1]))
            if end_tok >= len(ids):
                break
            pos += step
        return spans

    @staticmethod
    def _owned_ranges(
        spans: list[tuple[int, int, str]], text_len: int
    ) -> list[tuple[int, int]]:
        """Partition the document among overlapping windows, splitting each
        overlap at its midpoint so the ranges tile without double-counting."""
        if len(spans) == 1:
            return [(spans[0][0], spans[0][1])]
        owned: list[tuple[int, int]] = []
        for i, (c0, c1, _) in enumerate(spans):
            start = c0 if i == 0 else max(c0, (spans[i - 1][1] + c0) // 2)
            end = c1 if i == len(spans) - 1 else min(c1, (c1 + spans[i + 1][0]) // 2)
            owned.append((start, max(start, end)))
        # Guarantee full coverage even if a window is fully contained in another.
        owned[0] = (spans[0][0], owned[0][1])
        owned[-1] = (owned[-1][0], max(owned[-1][1], spans[-1][1], text_len))
        return owned

    def detect(self, text: str, normalise: bool = True) -> tuple[Verdict, list[dict]]:
        """Score a document. Long inputs are windowed and length-weighted."""
        source = clean_text(text) if normalise else text
        if not source.strip():
            raise ValueError("empty text")

        with self._active():
            spans = self.windows(source)
            scored = self._score_batch([s[2] for s in spans])

        # Weight each window by the text it exclusively OWNS, not by its full
        # length. Windows overlap by design (the model needs context either side
        # of a split), but weighting by full length counts the overlap twice --
        # measured up to +0.03 score inflation on documents just over one window,
        # i.e. ordinary essay length. Overlaps are split at their midpoint so
        # every character is counted exactly once.
        owned = self._owned_ranges(spans, len(source))

        details: list[dict] = []
        total_w = 0.0
        acc_score = 0.0
        acc_probs = [0.0] * self.n_buckets
        for (c0, c1, chunk), (o0, o1), (score, probs) in zip(spans, owned, scored):
            w = float(count_words(source[o0:o1])) or 1.0
            total_w += w
            acc_score += score * w
            acc_probs = [a + p * w for a, p in zip(acc_probs, probs)]
            details.append(
                {
                    "start": c0,
                    "end": c1,
                    "owned_start": o0,
                    "owned_end": o1,
                    "words": int(w),
                    "score": round(score, 4),
                    "label": self.bucket_names[int(round(score * (self.n_buckets - 1)))],
                    "preview": chunk[:120].replace("\n", " "),
                }
            )

        score = acc_score / total_w
        probs = [p / total_w for p in acc_probs]
        bucket = int(round(score * (self.n_buckets - 1)))
        verdict = Verdict(
            score=score,
            bucket=bucket,
            label=self.bucket_names[bucket],
            probs=probs,
            word_count=count_words(source),
            char_count=len(source),
            truncated_windows=len(spans),
        )
        return verdict, details

    def detect_many(self, texts: Sequence[str], normalise: bool = True) -> list[Verdict]:
        """Short-text fast path: one batched forward pass for the whole list.

        Any item too long for a single window falls back to windowed scoring.
        """
        prepared = [clean_text(t) if normalise else t for t in texts]
        simple_idx: list[int] = []
        simple_txt: list[str] = []
        results: list[Verdict | None] = [None] * len(prepared)

        with self._active():
            # Inside _active(), not before it: between an outer ensure_loaded()
            # and acquiring the lock, an unload could null the tokenizer.
            self.ensure_loaded()
            for i, t in enumerate(prepared):
                if not t.strip():
                    raise ValueError(f"item {i} is empty")
                n_tok = len(
                    self.tokenizer(t, add_special_tokens=False, verbose=False)["input_ids"]
                )
                if n_tok <= MAX_CONTENT_TOKENS:
                    simple_idx.append(i)
                    simple_txt.append(t)
                else:
                    results[i], _ = self.detect(t, normalise=False)
            scored_simple = self._score_batch(simple_txt)

        for i, (score, probs) in zip(simple_idx, scored_simple):
            bucket = int(round(score * (self.n_buckets - 1)))
            results[i] = Verdict(
                score=score,
                bucket=bucket,
                label=self.bucket_names[bucket],
                probs=probs,
                word_count=count_words(prepared[i]),
                char_count=len(prepared[i]),
                truncated_windows=1,
            )
        return [r for r in results if r is not None]


# A sentence terminator, any trailing closing punctuation, then whitespace.
# `re` forbids variable-width lookbehind, so boundaries are found with finditer
# and the following character is checked manually.
_SENT_END = re.compile(r"[.!?][\"'”’)\]]*\s+")
_SENT_START = re.compile(r"[A-Z0-9\"'“(\[]")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        nxt = text[end : end + 1]
        # A blank line always ends a sentence; otherwise require a plausible start.
        if not nxt or _SENT_START.match(nxt) or "\n\n" in m.group():
            if text[start:end].strip():
                spans.append((start, end))
            start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def split_units(text: str, granularity: str = "sentence", min_words: int = 25) -> list[tuple[int, int]]:
    """Return (start, end) char spans for scoring granularity.

    Sentences shorter than `min_words` are merged forward, because a 512-token
    document model produces noise on very short inputs.
    """
    if granularity == "paragraph":
        pieces: list[tuple[int, int]] = []
        pos = 0
        for block in re.split(r"\n\s*\n", text):
            start = text.index(block, pos) if block else pos
            pieces.append((start, start + len(block)))
            pos = start + len(block)
        # Paragraphs are returned as authored -- merging them would defeat the
        # point of asking for paragraph granularity. `min_words` therefore does
        # not apply here, and split_units_adaptive reports that honestly rather
        # than claiming a threshold it never enforced.
        return [p for p in pieces if text[p[0] : p[1]].strip()]

    raw = _sentence_spans(text)
    if min_words <= 1:
        return [r for r in raw if text[r[0] : r[1]].strip()]

    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if merged and count_words(text[merged[-1][0] : merged[-1][1]]) < min_words:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) > 1 and count_words(text[merged[-1][0] : merged[-1][1]]) < min_words:
        last = merged.pop()
        merged[-1] = (merged[-1][0], last[1])
    return [m for m in merged if text[m[0] : m[1]].strip()]


def split_units_adaptive(
    text: str,
    granularity: str = "sentence",
    min_words: int = 25,
    min_units: int = 2,
) -> tuple[list[tuple[int, int]], int]:
    """Split into units, relaxing `min_words` until the text actually divides.

    A 25-word merge threshold swallows an ordinary paragraph whole, which makes
    span feedback useless exactly where it matters most. So back off to smaller
    units rather than returning one unit covering everything. Returns the spans
    and the threshold actually used, since smaller units are noisier and the
    caller should be able to say so.
    """
    if granularity != "sentence":
        # Paragraph units are authored boundaries; no threshold was applied.
        return split_units(text, granularity, min_words), 0

    units = split_units(text, granularity, min_words)
    if len(units) >= min_units:
        return units, min_words
    for relaxed in (15, 8, 1):
        if relaxed >= min_words:
            continue
        alt = split_units(text, granularity, relaxed)
        if len(alt) >= min_units:
            return alt, relaxed
    return units, min_words
