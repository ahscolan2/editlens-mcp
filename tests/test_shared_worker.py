"""Shared inference transport/lifecycle tests.

Default: deterministic stdlib-only service tests, no model load.
``python tests/test_shared_worker.py --real`` also launches simultaneous real
clients, verifies one model worker, crashes/restarts it, and observes idle exit.
All workers and files in this suite belong to temporary runtime directories.
"""

from __future__ import annotations

import http.client
import json
import math
import os
from pathlib import Path
import queue
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from editlens_mcp.detector import DetectorUnavailable, Verdict  # noqa: E402
from editlens_mcp.shared import (  # noqa: E402
    MAX_BODY_BYTES, MAX_LOG_BYTES, PROTOCOL_VERSION, SharedDetector, WorkerStopping,
    _FileLock, _HTTPServer, _InferenceService, _http, _read_json, _rotate_log,
)

REAL = "--real" in sys.argv
if REAL:
    sys.argv.remove("--real")


def wait_for(check, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true before its deadline")


class FakeDetector:
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def detect(self, text, normalise=True):
        self.calls.append(text)
        if text == "blocked":
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError("test did not release blocked inference")
        return Verdict(0.123456789, 0, "test", [0.8, 0.1, 0.05, 0.05], 1, len(text)), []

    def detect_many(self, texts, normalise=True):
        return [self.detect(text, normalise)[0] for text in texts]

    def info(self):
        return {"loaded": True, "checkpoint": "test"}

    def unload(self):
        return True


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeDetector()
        self.service = _InferenceService("test-key", max_queue=2, idle_seconds=10)
        self.service.start(self.fake)
        self.server = _HTTPServer(self.service, secrets.token_urlsafe(32))
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.record = {"port": self.server.server_port,
                       "token": self.server.authorization.decode().split(" ", 1)[1]}

    def tearDown(self):
        self.fake.release.set()
        wait_for(lambda: self.service.snapshot()["active"] == 0 and self.service.snapshot()["pending"] == 0)
        self.service.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.service.consumer.join(2)

    def rpc(self, operation, arguments=None, timeout=2):
        return _http(self.record, "POST", "/rpc", {
            "operation": operation, "arguments": arguments or {}, "timeout": timeout,
        }, timeout=timeout + 1)

    def test_fifo_and_queue_bound_with_responsive_health(self):
        first = self.service.enqueue("detect", {"text": "blocked"}, 3)
        self.assertTrue(self.fake.started.wait(2))
        second = self.service.enqueue("detect", {"text": "second"}, 3)
        third = self.service.enqueue("detect", {"text": "third"}, 3)
        with self.assertRaises(queue.Full):
            self.service.enqueue("detect", {"text": "overflow"}, 3)
        start = time.monotonic()
        status, health = _http(self.record, "GET", "/health", timeout=1)
        self.assertEqual(status, 200)
        self.assertLess(time.monotonic() - start, 1)
        self.assertEqual((health["active"], health["pending"]), (1, 2))
        self.assertNotIn("token", health)
        status, result = self.rpc("info")
        self.assertEqual(status, 503)
        self.assertEqual(result["error_type"], "QueueFull")
        self.fake.release.set()
        for job in (first, second, third):
            self.assertTrue(job.done.wait(2))
            self.assertTrue(job.result["ok"])
        self.assertEqual(self.fake.calls, ["blocked", "second", "third"])

    def test_queued_timeout_is_cancelled_without_inference(self):
        first = self.service.enqueue("detect", {"text": "blocked"}, 3)
        self.assertTrue(self.fake.started.wait(2))
        status, result = self.rpc("detect", {"text": "expired"}, timeout=0.05)
        self.assertEqual(status, 504)
        self.assertIn("not replayed", result["error"])
        self.fake.release.set()
        self.assertTrue(first.done.wait(2))
        wait_for(lambda: self.service.snapshot()["pending"] == 0)
        self.assertEqual(self.fake.calls, ["blocked"])
        self.assertEqual(self.service.snapshot()["cancelled"], 1)

    def test_running_timeout_finishes_once_and_worker_recovers(self):
        status, result = self.rpc("detect", {"text": "blocked"}, timeout=0.05)
        self.assertEqual(status, 504)
        self.assertEqual(self.service.snapshot()["active"], 1)
        self.fake.release.set()
        wait_for(lambda: self.service.snapshot()["active"] == 0)
        status, result = self.rpc("detect", {"text": "next"})
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertEqual(self.fake.calls, ["blocked", "next"])

    def test_authentication_and_operation_allowlist(self):
        wrong = {**self.record, "token": "not-the-secret"}
        self.assertEqual(_http(wrong, "GET", "/health")[0], 401)
        self.assertEqual(_http(wrong, "POST", "/rpc", {"operation": "info"})[0], 401)
        status, result = self.rpc("__getattribute__", {"name": "model"})
        self.assertEqual(status, 400)
        self.assertEqual(self.fake.calls, [])

    def test_foreign_host_header_is_rejected_even_with_the_token(self):
        # A DNS-rebinding page reaches 127.0.0.1 under its own hostname.
        for host, expected in (("attacker.example:1234", 401), ("localhost", 200)):
            with self.subTest(host=host):
                conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
                try:
                    conn.putrequest("GET", "/health", skip_host=True)
                    conn.putheader("Host", host)
                    conn.putheader("Authorization", "Bearer " + self.record["token"])
                    conn.endheaders()
                    response = conn.getresponse()
                    response.read()
                    self.assertEqual(response.status, expected)
                finally:
                    conn.close()

    def test_body_limit_is_enforced_before_reading(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            conn.request("POST", "/rpc", headers={
                "Authorization": "Bearer " + self.record["token"],
                "Content-Type": "application/json",
                "Content-Length": str(MAX_BODY_BYTES + 1),
            })
            response = conn.getresponse()
            self.assertEqual(response.status, 413)
            response.read()
        finally:
            conn.close()
        self.assertEqual(self.fake.calls, [])

    def test_proxy_keeps_full_precision_and_ignores_http_proxy(self):
        with tempfile.TemporaryDirectory(prefix="editlens-proxy-") as scratch:
            with patch.dict(os.environ, {"EDITLENS_RUNTIME_DIR": scratch,
                                         "HTTP_PROXY": "http://127.0.0.1:1",
                                         "HTTPS_PROXY": "http://127.0.0.1:1"}):
                proxy = SharedDetector()
                with patch.object(proxy, "_worker", return_value=self.record):
                    verdict, windows = proxy.detect("first")
                    self.assertEqual(verdict.score, 0.123456789)
                    self.assertNotEqual(verdict.score, round(verdict.score, 4))
                    self.assertEqual(windows, [])
                    self.assertEqual(len(proxy.detect_many(["a", "b"])), 2)
                    info = proxy.info()
                    self.assertEqual(info["backend"], "shared")
                    self.assertEqual(Path(info["worker_log"]).parent, Path(scratch).resolve())
                # Unload talks only to an already-running worker.
                with patch.object(proxy, "_available", return_value=self.record):
                    self.assertTrue(proxy.unload())

    def test_unload_and_offline_info_never_start_a_worker(self):
        with tempfile.TemporaryDirectory(prefix="editlens-unload-") as scratch:
            with patch.dict(os.environ, {"EDITLENS_RUNTIME_DIR": scratch}):
                proxy = SharedDetector()
                with patch.object(proxy, "_spawn") as spawn, \
                        patch.object(proxy, "_worker") as worker:
                    self.assertFalse(proxy.unload())
                    offline = proxy.offline_info()
                spawn.assert_not_called()
                worker.assert_not_called()
                self.assertIsNone(offline["worker_pid"])
                self.assertFalse(offline["loaded"])
                self.assertEqual(offline["checkpoint"], proxy.scoring_identity()["checkpoint"])
                self.assertTrue(offline["worker_log"].endswith(".log"))

    def test_request_refused_by_an_exiting_worker_is_retried_once(self):
        with tempfile.TemporaryDirectory(prefix="editlens-stopping-") as scratch:
            with patch.dict(os.environ, {"EDITLENS_RUNTIME_DIR": scratch}):
                proxy = SharedDetector()
                refused = (503, {"ok": False, "error": "stopping", "error_type": "WorkerStopping"})
                served = (200, {"ok": True, "result": {"loaded": False}})
                with patch.object(proxy, "_worker", return_value=self.record) as worker, \
                        patch("editlens_mcp.shared._http", side_effect=[refused, served]) as request:
                    self.assertFalse(proxy.info()["loaded"])
                self.assertEqual(worker.call_count, 2)
                self.assertEqual(request.call_count, 2)
                # ...but only once: a second refusal is reported, not looped on.
                with patch.object(proxy, "_worker", return_value=self.record), \
                        patch("editlens_mcp.shared._http", side_effect=[refused, refused, served]):
                    with self.assertRaisesRegex(DetectorUnavailable, "stopping"):
                        proxy.info()

    def test_dead_consumer_is_not_advertised_as_healthy(self):
        class Fatal(BaseException):
            pass

        class Dying(FakeDetector):
            def info(self):
                raise Fatal("inference thread died")

        service = _InferenceService("dying", idle_seconds=10)
        service.start(Dying())
        job = service.enqueue("info", {}, 2)
        self.assertTrue(job.done.wait(2))
        service.consumer.join(2)
        self.assertFalse(service.consumer.is_alive())
        self.assertFalse(service.snapshot()["ready"])
        with self.assertRaises(WorkerStopping):
            service.enqueue("info", {}, 1)

    def test_queued_jobs_are_answered_when_the_worker_stops(self):
        service = _InferenceService("draining", idle_seconds=10)
        service.ready = True  # no consumer: jobs stay queued
        jobs = [service.enqueue("info", {}, 30) for _ in range(2)]
        service.stop()
        self.assertEqual(service.fail_queued("stopped"), 2)
        for job in jobs:
            self.assertTrue(job.done.is_set())
            self.assertEqual(job.result["error_type"], "WorkerStopping")
        self.assertEqual(service.snapshot()["pending"], 0)

    def test_transport_failure_never_replays_rpc(self):
        with tempfile.TemporaryDirectory(prefix="editlens-no-replay-") as scratch:
            with patch.dict(os.environ, {"EDITLENS_RUNTIME_DIR": scratch}):
                proxy = SharedDetector()
                with patch.object(proxy, "_worker", return_value=self.record):
                    with patch("editlens_mcp.shared._http", side_effect=TimeoutError) as request:
                        with self.assertRaisesRegex(DetectorUnavailable, "not replayed"):
                            proxy.detect("do once")
                        self.assertEqual(request.call_count, 1)

    def test_idle_exit_excludes_active_and_queued_requests(self):
        first = self.service.enqueue("detect", {"text": "blocked"}, 3)
        self.assertTrue(self.fake.started.wait(2))
        self.service.last_activity -= 100
        self.assertFalse(self.service.expire_if_idle())
        second = self.service.enqueue("detect", {"text": "queued"}, 3)
        self.service.last_activity -= 100
        self.assertFalse(self.service.expire_if_idle())
        self.fake.release.set()
        self.assertTrue(first.done.wait(2))
        self.assertTrue(second.done.wait(2))
        self.service.last_activity -= 100
        self.assertTrue(self.service.expire_if_idle())
        with self.assertRaises(DetectorUnavailable):
            self.service.enqueue("info", {}, 1)


class CoordinationTests(unittest.TestCase):
    def test_owner_exiting_during_discovery_can_be_replaced(self):
        with tempfile.TemporaryDirectory(prefix="editlens-owner-exit-") as scratch:
            with patch.dict(os.environ, {"EDITLENS_RUNTIME_DIR": scratch,
                                         "EDITLENS_STARTUP_TIMEOUT": "1"}):
                proxy = SharedDetector()
                owner = _FileLock(proxy.runtime / f"{proxy.key}.owner.lock")
                ready = {"pid": 123, "port": 456, "token": "test"}
                probes = []
                try:
                    self.assertTrue(owner.acquire())
                    with patch.object(proxy, "_spawn") as spawn:
                        spawn.return_value.poll.return_value = None

                        def available(_timeout):
                            probes.append(True)
                            if len(probes) == 3:
                                owner.close()  # Old worker completes idle exit.
                            return ready if spawn.called else None

                        with patch.object(proxy, "_available", side_effect=available):
                            self.assertEqual(proxy._worker(), ready)
                        spawn.assert_called_once()
                finally:
                    owner.close()

    def test_startup_deadline_prevents_a_late_spawn(self):
        with tempfile.TemporaryDirectory(prefix="editlens-startup-deadline-") as scratch:
            with patch.dict(os.environ, {"EDITLENS_RUNTIME_DIR": scratch,
                                         "EDITLENS_STARTUP_TIMEOUT": "1"}):
                proxy = SharedDetector()
                now = [100.0]

                def unavailable(_timeout):
                    now[0] += 1
                    return None

                with patch("editlens_mcp.shared.time.monotonic", side_effect=lambda: now[0]):
                    with patch.object(proxy, "_available", side_effect=unavailable):
                        with patch.object(proxy, "_spawn") as spawn:
                            with self.assertRaisesRegex(DetectorUnavailable, "no new worker"):
                                proxy.info()
                            spawn.assert_not_called()

    def test_child_that_lost_the_owner_race_can_be_replaced(self):
        # A child exiting 0 lost the lifetime lock to another worker. If that
        # worker dies too, the free lock must allow a replacement rather than
        # waiting out the whole startup deadline.
        with tempfile.TemporaryDirectory(prefix="editlens-exit-zero-") as scratch:
            with patch.dict(os.environ, {"EDITLENS_RUNTIME_DIR": scratch,
                                         "EDITLENS_STARTUP_TIMEOUT": "5"}):
                proxy = SharedDetector()
                ready = {"port": 1, "pid": 1, "token": "x" * 32}
                exited = unittest.mock.Mock(returncode=0)
                exited.poll.return_value = 0
                with patch.object(proxy, "_spawn", return_value=exited) as spawn, \
                        patch.object(proxy, "_owner_is_free", return_value=True), \
                        patch.object(proxy, "_available",
                                     side_effect=lambda *_: ready if spawn.call_count >= 2 else None):
                    started = time.monotonic()
                    self.assertEqual(proxy._worker(), ready)
                self.assertEqual(spawn.call_count, 2)
                self.assertLess(time.monotonic() - started, 4)

    def test_source_updated_after_session_start_is_reported_not_spawned(self):
        with tempfile.TemporaryDirectory(prefix="editlens-updated-") as scratch:
            with patch.dict(os.environ, {"EDITLENS_RUNTIME_DIR": scratch}):
                proxy = SharedDetector()
                with patch("editlens_mcp.shared._worker_key", return_value="0" * 64):
                    with patch.object(proxy, "_spawn") as spawn:
                        with self.assertRaisesRegex(DetectorUnavailable, "Restart the MCP client"):
                            proxy.detect("text")
                        spawn.assert_not_called()

    def test_symlinked_venvs_keep_their_interpreter_and_separate_workers(self):
        with tempfile.TemporaryDirectory(prefix="editlens-venv-symlink-") as scratch:
            first_path = str(Path(scratch) / "first-venv" / "bin" / "python")
            second_path = str(Path(scratch) / "second-venv" / "bin" / "python")
            base_path = str(Path(scratch) / "global-python")
            realpath = os.path.realpath

            def follow_symlink(path, *args, **kwargs):
                # Reproduce POSIX venv symlinks on Windows too, without asking
                # for symlink privileges or installing an extra interpreter.
                if os.fspath(path) in {first_path, second_path}:
                    return base_path
                return realpath(path, *args, **kwargs)

            with patch.dict(os.environ, {"EDITLENS_RUNTIME_DIR": scratch}):
                with patch("os.path.realpath", side_effect=follow_symlink):
                    with patch("sys.executable", first_path):
                        first = SharedDetector()
                        with patch("editlens_mcp.shared.subprocess.Popen") as popen:
                            popen.return_value.wait.return_value = 0
                            first._spawn()
                            self.assertEqual(popen.call_args.args[0][0], os.path.normcase(first_path))
                    with patch("sys.executable", second_path):
                        second = SharedDetector()
                    self.assertNotEqual(first.key, second.key)

    def test_request_timeout_rejects_values_outside_wire_protocol_bound(self):
        for value in ("3601", "inf", "nan", "-1", "0"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"EDITLENS_REQUEST_TIMEOUT": value}):
                    with self.assertRaisesRegex(ValueError, "EDITLENS_REQUEST_TIMEOUT"):
                        SharedDetector()

    def test_invalid_worker_idle_fails_at_client_construction(self):
        # Before: accepted here, then every call reported only that the
        # worker "exited during startup".
        for value in ("-5", "nan", "soon"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"EDITLENS_WORKER_IDLE": value}):
                    with self.assertRaisesRegex(ValueError, "EDITLENS_WORKER_IDLE"):
                        SharedDetector()

    def test_worker_log_rotation_keeps_one_previous_generation(self):
        with tempfile.TemporaryDirectory(prefix="editlens-log-") as scratch:
            log = Path(scratch) / "k.log"
            log.write_bytes(b"small")
            _rotate_log(log)
            self.assertEqual(log.read_bytes(), b"small")
            log.write_bytes(b"x" * (MAX_LOG_BYTES + 1))
            _rotate_log(log)
            self.assertFalse(log.exists())
            self.assertEqual((Path(scratch) / "k.log.1").stat().st_size, MAX_LOG_BYTES + 1)
            _rotate_log(log)  # missing file: no error

    def test_file_lock_has_one_owner_and_reuses_same_file(self):
        with tempfile.TemporaryDirectory(prefix="editlens-lock-") as scratch:
            path = Path(scratch) / "owner.lock"
            first, second = _FileLock(path), _FileLock(path)
            try:
                self.assertTrue(first.acquire())
                self.assertFalse(second.acquire())
                first.close()
                self.assertTrue(second.acquire())
            finally:
                first.close()
                second.close()
            self.assertTrue(path.exists())

    def test_worker_identity_ignores_database_but_respects_model_config(self):
        with tempfile.TemporaryDirectory(prefix="editlens-identity-") as scratch:
            with patch.dict(os.environ, {"EDITLENS_RUNTIME_DIR": scratch, "EDITLENS_DB": "one.db"}):
                first = SharedDetector()
                with patch.dict(os.environ, {"EDITLENS_DB": "two.db"}):
                    same = SharedDetector()
                    different = SharedDetector(preprocessing="legacy")
                    different_revision = SharedDetector(revision="another-checkpoint-revision")
            self.assertEqual(first.key, same.key)
            self.assertNotEqual(first.key, different.key)
            self.assertNotEqual(first.key, different_revision.key)
            self.assertEqual(first.config["revision"], first.scoring_identity()["checkpoint_revision"])
            self.assertEqual(first.scoring_identity()["preprocessing"], first.preprocessing)

    def test_busy_owner_without_registry_cannot_spawn_duplicate(self):
        with tempfile.TemporaryDirectory(prefix="editlens-busy-owner-") as scratch:
            with patch.dict(os.environ, {"EDITLENS_RUNTIME_DIR": scratch,
                                         "EDITLENS_STARTUP_TIMEOUT": "0.1"}):
                proxy = SharedDetector()
                owner = _FileLock(proxy.runtime / f"{proxy.key}.owner.lock")
                try:
                    self.assertTrue(owner.acquire())
                    with patch.object(proxy, "_spawn") as spawn:
                        with self.assertRaisesRegex(DetectorUnavailable, "no duplicate"):
                            proxy.info()
                        spawn.assert_not_called()
                finally:
                    owner.close()

    def test_proxy_import_and_provenance_never_import_torch(self):
        with tempfile.TemporaryDirectory(prefix="editlens-no-torch-") as scratch:
            env = dict(os.environ, EDITLENS_RUNTIME_DIR=scratch)
            code = ("import sys; from editlens_mcp.shared import SharedDetector; "
                    "d=SharedDetector(); d.scoring_identity(); "
                    "assert 'torch' not in sys.modules; print('no torch')")
            result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("no torch", result.stdout)


@unittest.skipUnless(REAL, "pass --real to load the real model in a temporary shared worker")
class RealWorkerTests(unittest.TestCase):
    def test_concurrent_clients_share_model_survive_exit_restart_and_idle(self):
        with tempfile.TemporaryDirectory(prefix="editlens-real-worker-") as scratch:
            env = dict(os.environ, EDITLENS_RUNTIME_DIR=scratch, EDITLENS_WORKER_IDLE="5",
                       EDITLENS_REQUEST_TIMEOUT="300", EDITLENS_STARTUP_TIMEOUT="90")
            config = {"device": os.environ.get("EDITLENS_TEST_DEVICE") or None}
            env["EDITLENS_TEST_CONFIG"] = json.dumps(config)
            text = "I burnt the rice again. Third time this month. I turned off the stove and ate the carrots straight from the pan while my brother laughed at the smoke."
            env["EDITLENS_TEST_TEXT"] = text
            code = r'''
import json, os, sys
from editlens_mcp.shared import SharedDetector
d = SharedDetector(**json.loads(os.environ["EDITLENS_TEST_CONFIG"]))
info = d.info()
v, _ = d.detect(os.environ["EDITLENS_TEST_TEXT"])
assert "torch" not in sys.modules
print(json.dumps({"pid": info["worker_pid"], "score": v.score, "probs": v.probs}))
'''
            clients = []
            proxy = None

            def owner_free():
                lock = _FileLock(proxy.runtime / f"{proxy.key}.owner.lock")
                try:
                    return lock.acquire()
                finally:
                    lock.close()

            def kill_owned_worker():
                registry = proxy.runtime / f"{proxy.key}.json"
                if not registry.exists():
                    return
                record = _read_json(registry)
                try:
                    status, health = _http(record, "GET", "/health")
                except OSError:
                    return
                if status == 200 and health["pid"] == record["pid"] and health["key"] == proxy.key:
                    os.kill(record["pid"], signal.SIGTERM)
                    wait_for(owner_free, timeout=15)

            def logs_released():
                # Windows' venv launcher can retain inherited stderr briefly
                # after the actual worker exits. The model's lifetime lock is
                # already free; wait for those separate launcher handles too.
                for log in Path(scratch).glob("*.log"):
                    try:
                        log.unlink()
                    except PermissionError:
                        return False
                return True

            with patch.dict(os.environ, env):
                proxy = SharedDetector(**config)
                try:
                    for _ in range(3):
                        clients.append(subprocess.Popen([sys.executable, "-c", code], cwd=ROOT,
                                                        env=env, text=True, stdout=subprocess.PIPE,
                                                        stderr=subprocess.PIPE))
                    results = []
                    for child in clients:
                        out, err = child.communicate(timeout=300)
                        self.assertEqual(child.returncode, 0, out + err)
                        results.append(json.loads(out.strip()))
                    pids = {result["pid"] for result in results}
                    self.assertEqual(len(pids), 1, f"clients loaded separate workers: {pids}")
                    for result in results:
                        self.assertTrue(math.isfinite(result["score"]))
                        self.assertTrue(0 <= result["score"] <= 1)
                        self.assertAlmostEqual(sum(result["probs"]), 1, places=5)
                        self.assertAlmostEqual(result["score"], results[0]["score"], places=6)
                    # All three clients exited; their worker is still available.
                    info = proxy.info()
                    self.assertEqual(info["worker_pid"], results[0]["pid"])
                    self.assertTrue(info["loaded"])
                    self.assertGreaterEqual(info["completed"], 6)
                    old_pid = info["worker_pid"]
                    kill_owned_worker()
                    wait_for(logs_released, timeout=10)
                    fresh = proxy.info()
                    self.assertNotEqual(fresh["worker_pid"], old_pid)
                    self.assertFalse(fresh["loaded"])
                    verdict, _ = proxy.detect(text)
                    self.assertAlmostEqual(verdict.score, results[0]["score"], places=6)
                    self.assertNotIn("torch", sys.modules)
                    # Health polling must not keep an otherwise idle model alive.
                    registry = proxy.runtime / f"{proxy.key}.json"
                    wait_for(lambda: not registry.exists(), timeout=20)
                    wait_for(owner_free, timeout=5)
                    print(f"  3 real clients: one worker {old_pid}; restart {fresh['worker_pid']}; "
                          f"score={verdict.score:.8f}; client-exit survival and idle exit verified")
                finally:
                    for child in clients:
                        if child.poll() is None:
                            child.kill()
                            child.communicate(timeout=15)
                    kill_owned_worker()
                    wait_for(logs_released, timeout=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
