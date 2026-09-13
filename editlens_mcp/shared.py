"""Local shared inference, independent of MCP sessions and chain databases.

Stdio clients use SharedDetector without importing torch. A per-user worker owns
the native runtime and consumes a bounded FIFO. OS locks elect one worker for a
given configuration; an unresponsive owner is never replaced while it holds its
lock. Requests are authenticated JSON over loopback, never pickle or a proxy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import errno
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from .detector import DetectorUnavailable, EditLensDetector, Verdict, warmup_imports

PROTOCOL_VERSION = 1
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_QUEUE = 32
_ROOT = Path(__file__).resolve().parent.parent


def _seconds(name: str, default: float, *, allow_zero: bool = False) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw) if raw else default
        if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise ValueError
        return value
    except ValueError:
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number") from None


def runtime_dir() -> Path:
    override = os.environ.get("EDITLENS_RUNTIME_DIR", "").strip()
    if override:
        path = Path(os.path.expandvars(override)).expanduser().resolve()
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        path = base / "editlens-mcp" / "runtime"
    else:
        base = Path(os.environ.get("XDG_RUNTIME_DIR") or Path.home() / ".cache")
        path = base / "editlens-mcp"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        if path.stat().st_uid != os.getuid():
            raise PermissionError("EditLens runtime directory belongs to another user")
        path.chmod(0o700)
    return path.resolve()


class _FileLock:
    """An OS-owned lock; its file is stable and is never deleted."""

    def __init__(self, path: Path):
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        os.set_inheritable(self.fd, False)
        self.held = False

    def acquire(self, timeout: float = 0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            try:
                os.lseek(self.fd, 0, os.SEEK_SET)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.held = True
                return True
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if time.monotonic() >= deadline:
                    return False
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    def close(self):
        if self.fd is None:
            return
        try:
            if self.held:
                os.lseek(self.fd, 0, os.SEEK_SET)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None
            self.held = False


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: Any):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(_json_bytes(value))
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict:
    with path.open("rb") as source:
        raw = source.read(65537)
    if len(raw) > 65536:
        raise ValueError("worker metadata is too large")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("worker metadata must be an object")
    return value


def _canonical_config(config: dict) -> tuple[dict, EditLensDetector]:
    local = EditLensDetector(**config)
    canonical = {
        "checkpoint": local.checkpoint,
        "revision": local.revision,
        "base_model": local.base_model,
        "device": local._requested_device,
        "dtype": local._requested_dtype,
        "batch_size": local.batch_size,
        "idle_unload_seconds": local.idle_unload_seconds,
        "preprocessing": local.preprocessing,
    }
    _json_bytes(canonical)  # Reject NaN/Inf before creating files or processes.
    return canonical, local


def _python_executable() -> str:
    # POSIX venv interpreters are commonly symlinks. Following that symlink
    # launches the global interpreter and loses the venv's installed packages.
    return os.path.normcase(os.path.abspath(sys.executable))


def _worker_key(config: dict) -> str:
    digest = hashlib.sha256()
    digest.update(_json_bytes({
        "protocol": PROTOCOL_VERSION,
        "python": _python_executable(),
        "config": config,
    }))
    for name in ("__init__.py", "shared.py", "detector.py", "preprocessing.py"):
        source = Path(__file__).resolve().parent / name
        digest.update(name.encode())
        digest.update(source.read_bytes())
    digest.update((_ROOT / "run_worker.py").read_bytes())
    return digest.hexdigest()


def _http(record: dict, method: str, path: str, payload=None, timeout: float = 2) -> tuple[int, dict]:
    # http.client never reads HTTP_PROXY/HTTPS_PROXY, so document text stays local.
    conn = http.client.HTTPConnection("127.0.0.1", record["port"], timeout=timeout)
    body = None if payload is None else _json_bytes(payload)
    if body is not None and len(body) > MAX_BODY_BYTES:
        raise ValueError(f"EditLens request exceeds {MAX_BODY_BYTES // 1024**2} MiB")
    try:
        conn.request(method, path, body=body, headers={
            "Authorization": "Bearer " + record["token"],
            "Content-Type": "application/json",
            "Connection": "close",
        })
        response = conn.getresponse()
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise DetectorUnavailable("EditLens worker response exceeds the size limit")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("worker response must be an object")
        return response.status, result
    finally:
        conn.close()


class SharedDetector:
    """Detector-compatible client; construction and provenance stay lightweight."""

    def __init__(self, **config):
        self.config, self._local = _canonical_config(config)
        self.key = _worker_key(self.config)
        self.timeout = _seconds("EDITLENS_REQUEST_TIMEOUT", 300)
        if self.timeout > 3600:
            raise ValueError("EDITLENS_REQUEST_TIMEOUT must not exceed 3600 seconds")
        self.startup_timeout = _seconds("EDITLENS_STARTUP_TIMEOUT", 60)
        self.runtime = runtime_dir()

    @property
    def preprocessing(self):
        return self._local.preprocessing

    @property
    def preprocessing_version(self):
        return self._local.preprocessing_version

    @property
    def batch_size(self):
        return self._local.batch_size

    @property
    def idle_unload_seconds(self):
        return self._local.idle_unload_seconds

    def scoring_identity(self):
        return self._local.scoring_identity()

    def _available(self, timeout: float | None = None):
        try:
            record = _read_json(self.runtime / f"{self.key}.json")
            if (
                record.get("key") != self.key
                or record.get("protocol") != PROTOCOL_VERSION
                or type(record.get("port")) is not int
                or not 1 <= record["port"] <= 65535
                or type(record.get("pid")) is not int
                or record["pid"] <= 0
                or not isinstance(record.get("token"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", record["token"])
            ):
                return None
            health_timeout = min(1, self.timeout, timeout if timeout is not None else 1)
            if health_timeout <= 0:
                return None
            status, health = _http(record, "GET", "/health", timeout=health_timeout)
            if (status == 200 and health.get("ready") and health.get("accepting")
                    and health.get("key") == self.key and health.get("pid") == record["pid"]
                    and health.get("protocol") == PROTOCOL_VERSION):
                return record
        except (OSError, ValueError, http.client.HTTPException, DetectorUnavailable):
            pass
        return None

    def _spawn(self):
        _write_json(self.runtime / f"{self.key}.config.json", self.config)
        command = [_python_executable(), str(_ROOT / "run_worker.py"),
                   "--runtime", str(self.runtime), "--key", self.key]
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        # A file cannot fill and block like an unread PIPE. It never contains
        # RPC bodies or the authentication token.
        log_path = self.runtime / f"{self.key}.log"
        with log_path.open("ab") as log:
            child = subprocess.Popen(command, cwd=str(_ROOT), stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=log,
                                     close_fds=True, **options)
        # Reap the child if it exits while this client stays alive. The daemon
        # waiter has no ownership role and never terminates the child.
        threading.Thread(target=child.wait, daemon=True, name="editlens-worker-reaper").start()
        return child

    def _owner_is_free(self):
        owner = _FileLock(self.runtime / f"{self.key}.owner.lock")
        try:
            return owner.acquire()
        finally:
            owner.close()

    def _worker(self):
        deadline = time.monotonic() + min(self.timeout, self.startup_timeout)
        available = self._available(deadline - time.monotonic())
        if available:
            return available
        startup = _FileLock(self.runtime / f"{self.key}.startup.lock")
        try:
            if not startup.acquire(max(0, deadline - time.monotonic())):
                raise DetectorUnavailable("Timed out waiting for EditLens worker startup; another client is starting it")
            available = self._available(deadline - time.monotonic())
            if available:
                return available
            free = self._owner_is_free()
            if time.monotonic() >= deadline:
                raise DetectorUnavailable(
                    "Timed out waiting for EditLens worker startup; no new worker was started"
                )
            child = self._spawn() if free else None
            while time.monotonic() < deadline:
                available = self._available(deadline - time.monotonic())
                if available:
                    return available
                # An owner that was exiting when discovery began may now be
                # gone. The OS lock, not a failed health check or a PID guess,
                # proves it is safe to start its replacement. This happens
                # before any RPC has been sent, so it cannot replay inference.
                if (child is None and self._owner_is_free()
                        and time.monotonic() < deadline):
                    child = self._spawn()
                if child is not None and child.poll() not in (None, 0):
                    raise DetectorUnavailable(
                        f"EditLens worker exited during startup (code {child.returncode}); "
                        f"see {self.runtime / (self.key + '.log')}"
                    )
                # Exit code zero may be a child that lost the lifetime-lock
                # race to a worker spawned by a client that died mid-startup.
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            raise DetectorUnavailable(
                "EditLens worker did not become ready in time. Its owner may be busy or "
                "unresponsive; no duplicate model was started. "
                f"See {self.runtime / (self.key + '.log')}"
            )
        finally:
            startup.close()

    def _call(self, operation: str, **arguments):
        # Serialize/validate before starting a worker for an invalid request.
        payload = {"operation": operation, "arguments": arguments, "timeout": self.timeout}
        if len(_json_bytes(payload)) > MAX_BODY_BYTES:
            raise ValueError("EditLens request exceeds 8 MiB")
        record = self._worker()
        try:
            status, response = _http(record, "POST", "/rpc", payload, timeout=self.timeout + 1)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise DetectorUnavailable(
                f"EditLens worker request failed ({type(exc).__name__}); the request was not replayed. "
                "A running inference may still finish; a later call can reconnect."
            ) from exc
        if status != 200 or response.get("ok") is not True:
            message = str(response.get("error", f"worker returned HTTP {status}"))
            if response.get("error_type") == "ValueError":
                raise ValueError(message)
            if response.get("error_type") == "TypeError":
                raise TypeError(message)
            raise DetectorUnavailable(message)
        return response["result"]

    def detect(self, text: str, normalise: bool = True):
        result = self._call("detect", text=text, normalise=normalise)
        return Verdict(**result["verdict"]), result["windows"]

    def detect_many(self, texts, normalise: bool = True):
        return [Verdict(**item) for item in self._call("detect_many", texts=list(texts), normalise=normalise)]

    def info(self):
        return self._call("info")

    def health(self):
        record = self._worker()
        status, health = _http(record, "GET", "/health", timeout=min(2, self.timeout))
        if status != 200:
            raise DetectorUnavailable("EditLens worker health check failed")
        return health

    def unload(self):
        return self._call("unload")


@dataclass
class _Job:
    operation: str
    arguments: dict
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False
    result: dict | None = None


class _InferenceService:
    """One consumer owns the detector; the lock also makes idle exit atomic."""

    def __init__(self, key: str, *, max_queue: int = MAX_QUEUE, idle_seconds: float = 600):
        self.key = key
        self.jobs: queue.Queue[_Job] = queue.Queue(maxsize=max_queue)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.ready = False
        self.accepting = True
        self.pending = self.active = self.completed = self.cancelled = 0
        self.last_activity = time.monotonic()
        self.idle_seconds = idle_seconds
        self.detector = None
        self.consumer = None

    def start(self, detector):
        self.detector = detector
        with self.lock:
            self.ready = True
            self.last_activity = time.monotonic()
        self.consumer = threading.Thread(target=self._consume, daemon=True, name="editlens-inference")
        self.consumer.start()

    def snapshot(self):
        with self.lock:
            return {
                "protocol": PROTOCOL_VERSION, "key": self.key, "pid": os.getpid(),
                "backend": "shared", "worker_pid": os.getpid(),
                "ready": self.ready, "accepting": self.accepting,
                "pending": self.pending, "active": self.active,
                "completed": self.completed, "cancelled": self.cancelled,
                "queue_limit": self.jobs.maxsize,
                "worker_idle_seconds": self.idle_seconds,
            }

    def enqueue(self, operation: str, arguments: dict, timeout: float) -> _Job:
        job = _Job(operation, arguments, time.monotonic() + timeout)
        with self.lock:
            if not self.ready or not self.accepting:
                raise DetectorUnavailable("EditLens worker is starting or stopping; retry a later call")
            self.jobs.put_nowait(job)
            self.pending += 1
            self.last_activity = time.monotonic()
        return job

    def cancel(self, job: _Job):
        with self.lock:
            job.cancelled = True

    def expire_if_idle(self):
        with self.lock:
            if (self.ready and self.accepting and self.idle_seconds > 0
                    and self.pending == 0 and self.active == 0
                    and time.monotonic() - self.last_activity >= self.idle_seconds):
                self.accepting = False
                self.stopping.set()
                return True
        return False

    def stop(self):
        with self.lock:
            self.accepting = False
            self.stopping.set()

    def _execute(self, job: _Job):
        if job.operation == "detect":
            verdict, windows = self.detector.detect(**job.arguments)
            # Verdict.as_dict() rounds and renames windows; transport must keep
            # the full precision needed for chain comparisons.
            return {"verdict": asdict(verdict), "windows": windows}
        if job.operation == "detect_many":
            return [asdict(verdict) for verdict in self.detector.detect_many(**job.arguments)]
        if job.operation == "info":
            return {**self.detector.info(), **self.snapshot()}
        if job.operation == "unload":
            return self.detector.unload()
        raise ValueError("unknown worker operation")

    def _consume(self):
        while not self.stopping.is_set():
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            with self.lock:
                self.pending -= 1
                skip = job.cancelled or time.monotonic() >= job.deadline
                if skip:
                    self.cancelled += 1
                else:
                    self.active = 1
            try:
                if skip:
                    job.result = {"ok": False, "error": "Request expired before inference", "error_type": "TimeoutError"}
                else:
                    try:
                        job.result = {"ok": True, "result": self._execute(job)}
                    except Exception as exc:
                        job.result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
            finally:
                with self.lock:
                    if not skip:
                        self.active = 0
                        self.completed += 1
                    self.last_activity = time.monotonic()
                job.done.set()
                self.jobs.task_done()


def _validate_request(payload: dict):
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    operation = payload.get("operation")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    if operation in {"info", "unload"}:
        if arguments:
            raise ValueError("this operation takes no arguments")
    elif operation in {"detect", "detect_many"}:
        field_name = "text" if operation == "detect" else "texts"
        if set(arguments) - {field_name, "normalise"} or field_name not in arguments:
            raise ValueError("invalid inference arguments")
        value = arguments[field_name]
        if operation == "detect" and not isinstance(value, str):
            raise ValueError("text must be a string")
        if operation == "detect_many" and (not isinstance(value, list) or any(not isinstance(t, str) for t in value)):
            raise ValueError("texts must be a list of strings")
        if type(arguments.get("normalise", True)) is not bool:
            raise ValueError("normalise must be a boolean")
    else:
        raise ValueError("unknown worker operation")
    timeout = payload.get("timeout", 300)
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError("request timeout must be between 0 and 3600 seconds")
    return operation, arguments, float(timeout)


class _RequestHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *_args):
        pass  # Never log document content, credentials, or HTTP headers.

    def _reply(self, status, data):
        body = _json_bytes(data)
        if len(body) > MAX_RESPONSE_BYTES:
            status = 413
            body = _json_bytes({"ok": False, "error": "Worker response exceeds size limit"})
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (OSError, http.client.HTTPException):
            pass  # A disconnected client does not terminate the worker.
        self.close_connection = True

    def _authorized(self):
        supplied = self.headers.get("Authorization", "").encode("utf-8")
        if (self.headers.get("Origin") is not None
                or not hmac.compare_digest(supplied, self.server.authorization)):
            self._reply(401, {"ok": False, "error": "Unauthorized"})
            return False
        return True

    def do_GET(self):
        if not self._authorized():
            return
        if self.path != "/health":
            self._reply(404, {"ok": False, "error": "Unknown endpoint"})
            return
        self._reply(200, self.server.service.snapshot())

    def do_POST(self):
        if not self._authorized():
            return
        if self.path != "/rpc":
            self._reply(404, {"ok": False, "error": "Unknown endpoint"})
            return
        try:
            if self.headers.get("Transfer-Encoding") is not None:
                raise ValueError("chunked requests are not supported")
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                raise ValueError("Content-Type must be application/json")
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0 or length > MAX_BODY_BYTES:
                self._reply(413, {"ok": False, "error": "Request must fit within 8 MiB"})
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete request body")
            operation, arguments, timeout = _validate_request(json.loads(raw))
            job = self.server.service.enqueue(operation, arguments, timeout)
        except queue.Full:
            self._reply(503, {"ok": False, "error": "EditLens inference queue is full; try later", "error_type": "QueueFull"})
            return
        except DetectorUnavailable as exc:
            self._reply(503, {"ok": False, "error": str(exc), "error_type": type(exc).__name__})
            return
        except (ValueError, TypeError, OSError) as exc:
            self._reply(400, {"ok": False, "error": str(exc), "error_type": "ValueError"})
            return
        if not job.done.wait(timeout):
            self.server.service.cancel(job)
            self._reply(504, {"ok": False, "error": "EditLens request timed out. Queued work was cancelled; running inference may finish. The request was not replayed.", "error_type": "TimeoutError"})
            return
        self._reply(200, job.result)


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, service: _InferenceService, token: str):
        self.service = service
        self.authorization = ("Bearer " + token).encode("ascii")
        self._slots = threading.BoundedSemaphore(MAX_QUEUE + 8)
        super().__init__(("127.0.0.1", 0), _RequestHandler)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def worker_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="EditLens shared local inference worker")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-f0-9]{64}", args.key):
        parser.error("invalid worker key")
    runtime = args.runtime.resolve()
    owner = _FileLock(runtime / f"{args.key}.owner.lock")
    server = service = None
    token = secrets.token_urlsafe(32)
    registry = runtime / f"{args.key}.json"
    try:
        # Losing this race is harmless: only the winner can import torch/load.
        if not owner.acquire():
            return 0
        config, detector = _canonical_config(_read_json(runtime / f"{args.key}.config.json"))
        if _worker_key(config) != args.key:
            raise RuntimeError("EditLens source changed during worker startup; restart the client")
        service = _InferenceService(args.key, idle_seconds=_seconds("EDITLENS_WORKER_IDLE", 600, allow_zero=True))
        server = _HTTPServer(service, token)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1},
                         daemon=True, name="editlens-http").start()
        _write_json(registry, {"key": args.key, "protocol": PROTOCOL_VERSION,
                               "port": server.server_port, "pid": os.getpid(), "token": token})
        # Native imports exclusively on this process's main thread, before the
        # consumer starts; health stays responsive while warmup is in progress.
        warmup_imports()
        service.start(detector)
        while not service.stopping.wait(0.25):
            service.expire_if_idle()
        return 0
    finally:
        if service is not None:
            service.stop()
        if server is not None:
            server.shutdown()
            server.server_close()
        if owner.held:
            try:
                if _read_json(registry).get("token") == token:
                    registry.unlink()
            except (OSError, ValueError):
                pass
        # The native runtime may still own memory while Python shuts down.
        # Keep the raw descriptor locked through that phase; process exit (or
        # a crash) releases it. Unlocking here lets a new model overlap with
        # the old process's finalization and releases Windows files too early.
        if not owner.held:
            owner.close()
