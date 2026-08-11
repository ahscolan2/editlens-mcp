"""EditLens detector: lazy-loaded local RoBERTa-large AI-text classifier.

The checkpoint is a 4-bucket sequence classifier. A single continuous score is
derived the same way the official demo Space does it: the expected bucket index
under the softmax, normalised to [0, 1].
"""

from __future__ import annotations

import gc
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Sequence

CHECKPOINT = os.environ.get("EDITLENS_CHECKPOINT", "pangram/editlens_roberta-large")
BASE_MODEL = os.environ.get("EDITLENS_BASE_MODEL", "FacebookAI/roberta-large")
MAX_LENGTH = 512
# Room for <s> and </s>.
MAX_CONTENT_TOKENS = MAX_LENGTH - 2

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
    try:
        import torch  # noqa: F401, PLC0415
        from transformers import (  # noqa: F401, PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        torch.cuda.is_available()  # force CUDA probe here too
    except Exception as exc:  # noqa: BLE001 - reported through detector_info
        _warmup_error = f"{type(exc).__name__}: {exc}"
    _warmed = True
    return _warmup_error


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
        except ImportError:
            return "unknown (torch not installed)", "unknown"
        device = self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        return device, self._resolve_dtype_name(device)

    def _resolve_dtype_name(self, device: str) -> str:
        """Default float32 -- the precision the checkpoint is published in.

        float16 is ~2.5x faster on long documents but identical on single
        paragraphs (14 ms either way), so the default favours running the weights
        unconverted. Set EDITLENS_DTYPE=float16 to trade 0.001 of score accuracy
        for speed on long or batched work. float16 is GPU-only.
        """
        requested = (self._requested_dtype or "").lower()
        if requested in {"float16", "fp16", "16"}:
            return "float16" if device.startswith("cuda") else "float32"
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
            "idle_unload_seconds": self.idle_unload_seconds,
            "idle_seconds": (
                round(time.monotonic() - self._last_used, 1) if self._last_used else None
            ),
            "vram_mb": self._vram_mb(),
            "auto_unloads": self._unloads,
        }

    def _vram_mb(self) -> float | None:
        if not self._loaded or self.torch is None or not self.device.startswith("cuda"):
            return None
        return round(self.torch.cuda.memory_allocated() / 1024**2, 1)

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

    def unload(self) -> bool:
        """Drop the model and release GPU memory. Reloads on the next call."""
        with self._lock:
            if not self._loaded:
                return False
            self.model = None
            self.tokenizer = None
            self._loaded = False
            gc.collect()
            if self.torch is not None and self.device.startswith("cuda"):
                self.torch.cuda.empty_cache()
            return True

    def _start_watchdog(self) -> None:
        if self.idle_unload_seconds <= 0 or self._watchdog is not None:
            return

        def loop() -> None:
            tick = min(30.0, max(5.0, self.idle_unload_seconds / 4))
            while True:
                time.sleep(tick)
                with self._lock:
                    idle = time.monotonic() - self._last_used
                    if (
                        self._loaded
                        and self._inflight == 0
                        and idle > self.idle_unload_seconds
                        and self.unload()
                    ):
                        self._unloads += 1

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

    def _setup_hint(self, exc: Exception) -> str:
        msg = str(exc)
        lines = [f"Could not load '{self.checkpoint}': {type(exc).__name__}: {msg}"]
        if "torch" in msg or "transformers" in msg or isinstance(exc, ModuleNotFoundError):
            lines.append(
                "Install deps:  pip install torch --index-url "
                "https://download.pytorch.org/whl/cu126  &&  pip install transformers safetensors"
            )
        if "401" in msg or "403" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
            lines.append(
                f"The repo is GATED. Accept the licence at https://huggingface.co/{self.checkpoint} "
                "then set HF_TOKEN to a read token (hf auth login, or set the env var)."
            )
        return "\n".join(lines)

    def _load(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        kwargs = {"token": token} if token else {}

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, **kwargs)
        except Exception:  # tokenizer files may live only on the base repo
            tokenizer = AutoTokenizer.from_pretrained(self.base_model, **kwargs)

        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, **kwargs)
        except Exception:
            # Fall back to the PEFT-adapter layout if the repo ships an adapter only.
            from peft import PeftConfig, PeftModel  # noqa: PLC0415

            cfg = PeftConfig.from_pretrained(self.checkpoint, **kwargs)
            n_labels = getattr(cfg, "num_labels", None) or 4
            base = AutoModelForSequenceClassification.from_pretrained(
                cfg.base_model_name_or_path or self.base_model,
                num_labels=n_labels,
                **kwargs,
            )
            model = PeftModel.from_pretrained(base, self.checkpoint, **kwargs).merge_and_unload()

        if self._requested_device:
            device = self._requested_device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_name = self._resolve_dtype_name(device)
        dtype = torch.float16 if dtype_name == "float16" else torch.float32
        model = model.to(device=device, dtype=dtype).eval()

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

    def detect(self, text: str, normalise: bool = True) -> tuple[Verdict, list[dict]]:
        """Score a document. Long inputs are windowed and length-weighted."""
        source = clean_text(text) if normalise else text
        if not source.strip():
            raise ValueError("empty text")

        with self._active():
            spans = self.windows(source)
            scored = self._score_batch([s[2] for s in spans])

        details: list[dict] = []
        total_w = 0.0
        acc_score = 0.0
        acc_probs = [0.0] * self.n_buckets
        for (c0, c1, chunk), (score, probs) in zip(spans, scored):
            w = float(count_words(chunk)) or 1.0
            total_w += w
            acc_score += score * w
            acc_probs = [a + p * w for a, p in zip(acc_probs, probs)]
            details.append(
                {
                    "start": c0,
                    "end": c1,
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

        self.ensure_loaded()
        with self._active():
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
    units = split_units(text, granularity, min_words)
    if len(units) >= min_units or granularity != "sentence":
        return units, min_words
    for relaxed in (15, 8, 1):
        if relaxed >= min_words:
            continue
        alt = split_units(text, granularity, relaxed)
        if len(alt) >= min_units:
            return alt, relaxed
    return units, min_words
