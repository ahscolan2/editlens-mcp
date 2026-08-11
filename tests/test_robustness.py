"""Failure-mode regressions: startup races, transaction wedging, device fallback.

Each of these was a real defect. They are cheap to reintroduce and expensive to
notice, because they only bite on a second client, a full disk, or a Mac.
"""

import multiprocessing as mp
import os
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from editlens_mcp.chains import ChainStore, default_db_path  # noqa: E402

TMP = Path(tempfile.mkdtemp())


def _open_and_write(path: str, barrier, results) -> None:
    try:
        barrier.wait(timeout=30)
        s = ChainStore(path)
        s.create("x")
        s.close()
        results.append("ok")
    except Exception as exc:  # noqa: BLE001
        results.append(f"{type(exc).__name__}: {exc}")


def test_simultaneous_open_threads():
    """Many clients opening one brand-new database must not race on WAL setup."""
    path = str(TMP / "race.db")
    barrier = threading.Barrier(12)
    results: list[str] = []
    threads = [threading.Thread(target=_open_and_write, args=(path, barrier, results))
               for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    failures = [r for r in results if r != "ok"]
    print(f"  12 simultaneous opens: {len(results) - len(failures)} ok, {len(failures)} failed")
    for f in failures[:3]:
        print(f"    {f}")
    assert not failures, failures[:3]


def test_simultaneous_open_processes():
    """Same, across real processes -- the actual two-client deployment."""
    path = str(TMP / "race_proc.db")
    with mp.Manager() as mgr:
        results = mgr.list()
        barrier = mgr.Barrier(6)
        procs = [mp.Process(target=_open_and_write, args=(path, barrier, results))
                 for _ in range(6)]
        [p.start() for p in procs]
        [p.join(timeout=90) for p in procs]
        got = list(results)
    failures = [r for r in got if r != "ok"]
    print(f"  6 simultaneous processes: {len(got) - len(failures)} ok, {len(failures)} failed")
    for f in failures[:3]:
        print(f"    {f}")
    assert got, "no results collected"
    assert not failures, failures[:3]


def test_journal_mode_reported():
    s = ChainStore(TMP / "mode.db")
    assert s.journal_mode in {"wal", "delete", "truncate", "persist", "memory", "off"}, \
        s.journal_mode
    print(f"  journal_mode recorded: {s.journal_mode}")
    s.close()


class FailingConn:
    """Delegates to a real connection but raises for chosen statements.

    sqlite3.Connection.execute is read-only, so the failure is injected by
    substituting the whole connection rather than patching a method.
    """

    def __init__(self, real, should_fail):
        self._real = real
        self._should_fail = should_fail

    def execute(self, sql, *args):
        err = self._should_fail(sql)
        if err:
            raise sqlite3.OperationalError(err)
        return self._real.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_failed_commit_does_not_wedge_the_store():
    """A failed COMMIT must not leave the transaction open forever."""
    s = ChainStore(TMP / "wedge.db")
    cid = s.create("c")["chain_id"]
    real = s._conn

    s._conn = FailingConn(real, lambda sql: "database is locked" if sql == "COMMIT" else None)
    try:
        s.create("will fail")
        raise AssertionError("expected the injected COMMIT failure")
    except sqlite3.OperationalError:
        pass
    s._conn = real

    assert not real.in_transaction, "transaction left open after failed COMMIT"
    # The store must still be usable afterwards.
    s.add_step(cid, "main", "draft", 0.5, 2, "x", 5, [0.25] * 4)
    assert s.latest_step(cid, "main")["step_no"] == 1
    print("  failed COMMIT rolled back; store still usable")
    s.close()


def test_rollback_failure_preserves_real_error():
    """If ROLLBACK also fails, the ORIGINAL error must reach the caller."""
    s = ChainStore(TMP / "shadow.db")
    real = s._conn

    def which(sql):
        if sql == "ROLLBACK":
            return "cannot rollback - no transaction is active"
        if sql.startswith("INSERT"):
            return "database or disk is full"
        return None

    s._conn = FailingConn(real, which)
    try:
        s.create("x")
        raise AssertionError("expected the injected INSERT failure")
    except sqlite3.OperationalError as exc:
        assert "disk is full" in str(exc), f"real cause was shadowed: {exc}"
        print(f"  real cause preserved: {exc}")
    finally:
        s._conn = real
        try:
            real.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        s.close()


def test_integrity_retry_scoped_to_unique():
    """A FOREIGN KEY violation must fail fast, not retry five times."""
    import time
    s = ChainStore(TMP / "fk.db")
    t0 = time.monotonic()
    try:
        s.add_step("ch_does_not_exist", "main", "d", 0.5, 2, "x", 3, [0.25] * 4)
        raise AssertionError("expected a FOREIGN KEY failure")
    except sqlite3.IntegrityError as exc:
        elapsed = time.monotonic() - t0
        assert "foreign key" in str(exc).lower(), exc
        assert elapsed < 0.05, f"retried a non-retryable error for {elapsed:.3f}s"
        print(f"  FOREIGN KEY failed fast in {elapsed * 1000:.1f}ms (no retry storm)")
    s.close()


def test_empty_env_db_path():
    """EDITLENS_DB set-but-empty must fall back, not crash."""
    old = os.environ.get("EDITLENS_DB")
    os.environ["EDITLENS_DB"] = ""
    try:
        p = default_db_path()
        assert p.name == "chains.db" and str(p) != "", p
        print(f"  EDITLENS_DB='' -> {p}")
    finally:
        if old is None:
            os.environ.pop("EDITLENS_DB", None)
        else:
            os.environ["EDITLENS_DB"] = old


def test_owned_ranges_always_partition():
    """Even with contained or degenerate windows, ranges must tile."""
    from editlens_mcp.detector import EditLensDetector
    cases = [
        [(4, 4, ""), (37, 67, ""), (38, 54, ""), (40, 75, "")],
        [(0, 50, ""), (10, 20, ""), (45, 90, "")],
        [(0, 10, ""), (0, 10, ""), (0, 10, "")],
        [(0, 100, ""), (99, 100, "")],
    ]
    for spans in cases:
        text_len = max(s[1] for s in spans)
        owned = EditLensDetector._owned_ranges(spans, text_len)
        assert len(owned) == len(spans)
        for a, b in zip(owned, owned[1:]):
            assert a[1] == b[0], f"gap/overlap in {owned} from {spans}"
        for a, b in owned:
            assert a <= b, f"inverted range {(a, b)} in {owned}"
        assert owned[-1][1] >= text_len, (owned, text_len)
    print(f"  {len(cases)} pathological window sets all partition cleanly")


def _legacy_db(path: Path) -> str:
    """A database as written before the parent_step column existed."""
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chains (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, goal TEXT,
            target_score REAL NOT NULL, segments TEXT NOT NULL,
            created_at REAL NOT NULL, updated_at REAL NOT NULL, meta TEXT);
        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_id TEXT NOT NULL REFERENCES chains(id) ON DELETE CASCADE,
            segment TEXT NOT NULL, step_no INTEGER NOT NULL, text TEXT NOT NULL,
            score REAL NOT NULL, bucket INTEGER NOT NULL, label TEXT NOT NULL,
            words INTEGER NOT NULL, probs TEXT NOT NULL, note TEXT,
            created_at REAL NOT NULL);
    """)
    conn.execute("INSERT INTO chains VALUES ('ch_old','old',NULL,0.25,'[\"main\"]',0,0,'{}')")
    conn.execute("INSERT INTO steps (chain_id, segment, step_no, text, score, bucket,"
                 " label, words, probs, note, created_at)"
                 " VALUES ('ch_old','main',1,'legacy draft',0.5,2,'x',2,'[]',NULL,0)")
    conn.close()
    return str(path)


def test_concurrent_migration_of_a_legacy_db():
    """Two clients opening a pre-branching database must not race on ALTER TABLE.

    Checking table_info then running ALTER is check-then-act: both processes see
    the column missing, both ALTER, and the loser dies with 'duplicate column
    name' at import -- the client sees a dead process, not an error.
    """
    path = _legacy_db(TMP / "legacy.db")
    barrier = threading.Barrier(16)
    results: list[str] = []

    def opener():
        try:
            barrier.wait(timeout=30)
            s = ChainStore(path)
            assert s.latest_step("ch_old", "main")["step_no"] == 1
            s.close()
            results.append("ok")
        except Exception as exc:  # noqa: BLE001
            results.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=opener) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    failures = [r for r in results if r != "ok"]
    print(f"  16 clients migrating a legacy DB: {len(results) - len(failures)} ok, "
          f"{len(failures)} failed")
    for f in failures[:3]:
        print(f"    {f}")
    assert not failures, failures[:3]

    # The migration must actually have happened, and legacy rows survive.
    s = ChainStore(path)
    cols = {r["name"] for r in s._conn.execute("PRAGMA table_info(steps)")}
    assert "parent_step" in cols
    assert s.latest_step("ch_old", "main")["text"] == "legacy draft"
    s.add_step("ch_old", "main", "new draft", 0.3, 1, "y", 2, [0.25] * 4, parent_step=1)
    assert s.history("ch_old", "main")[-1]["parent_step"] == 1
    s.close()
    print("  legacy rows preserved; parent_step usable after migration")


def test_failed_init_closes_the_connection():
    """A ChainStore whose setup fails must not leak the file handle."""
    path = TMP / "leak.db"
    real_connect = sqlite3.connect

    class Boom(Exception):
        pass

    store = None
    try:
        orig_migrate = ChainStore._migrate
        ChainStore._migrate = lambda self, t: (_ for _ in ()).throw(Boom("setup failed"))
        try:
            store = ChainStore(path)
            raise AssertionError("expected the injected setup failure")
        except Boom:
            pass
    finally:
        ChainStore._migrate = orig_migrate

    # If the connection leaked, Windows refuses to delete the file.
    try:
        path.unlink()
        print("  failed init released the file handle")
    except PermissionError as exc:
        raise AssertionError(f"connection leaked on failed init: {exc}") from exc


def test_info_reports_reality_after_unload():
    """detector_info must not re-predict a device already proven unusable."""
    from editlens_mcp.detector import EditLensDetector
    import torch

    class BadModel:
        def to(self, **kw):
            if kw.get("device") != "cpu":
                raise RuntimeError("CUDA error: invalid device ordinal")
            return self
        def eval(self):
            return self
        def __call__(self, **kw):
            return None

    det = EditLensDetector(device="cuda:99", dtype="float16")
    det.torch = torch
    _, device, dtype = det._place_model(BadModel(), None, torch, "cuda:99", "float16")
    det.device, det.dtype = device, dtype
    det._last_device, det._last_dtype = device, dtype

    # Unloaded is the normal state after 5 minutes idle.
    info = det.info()
    assert info["device"] == "cpu" and info["dtype"] == "float32", (
        f"unloaded info re-predicted the request: {info['device']}/{info['dtype']}")
    print(f"  after fallback, unloaded info reports {info['device']}/{info['dtype']}")


def test_device_fallback_cleared_on_recovery():
    """A transient failure must not leave a permanent fallback notice."""
    from editlens_mcp.detector import EditLensDetector
    import torch

    class _Batch(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        def __call__(self, *a, **k):
            return _Batch()

    class Flaky:
        def __init__(self):
            self.calls = 0
        def to(self, **kw):
            self.calls += 1
            if self.calls == 1 and kw.get("device") != "cpu":
                raise RuntimeError("CUDA out of memory")
            return self
        def eval(self):
            return self
        def __call__(self, **kw):
            return None

    det = EditLensDetector(device="cuda")
    det.torch = torch
    m = Flaky()
    tok = FakeTokenizer()
    _, device, _ = det._place_model(m, tok, torch, "cuda", "float32")
    assert device == "cpu" and det._device_fallback, "first attempt should have fallen back"
    # Reload after an idle unload: the GPU is free again this time.
    _, device2, _ = det._place_model(m, tok, torch, "cuda", "float32")
    assert device2 == "cuda", device2
    assert det._device_fallback is None, (
        f"stale fallback survived a successful reload: {det._device_fallback}")
    print("  fallback notice cleared once the device works again")


def test_device_fallback_covers_placement_failure():
    """A torch build without the requested accelerator must fall back, not die."""
    from editlens_mcp.detector import EditLensDetector
    import torch

    class BadModel:
        def to(self, **kw):
            if kw.get("device") != "cpu":
                raise RuntimeError("PyTorch is not linked with support for mps devices")
            return self
        def eval(self):
            return self
        def __call__(self, **kw):
            return None

    det = EditLensDetector(device="mps")
    det.torch = torch
    model, device, dtype = det._place_model(BadModel(), None, torch, "mps", "float16")
    assert device == "cpu" and dtype == "float32", (device, dtype)
    assert det._device_fallback and "mps -> cpu" in det._device_fallback
    print(f"  .to('mps') failure -> {device}/{dtype}, reported: "
          f"{det._device_fallback.split(':')[0]}")


if __name__ == "__main__":
    print("robustness tests")
    test_journal_mode_reported()
    test_simultaneous_open_threads()
    test_simultaneous_open_processes()
    test_failed_commit_does_not_wedge_the_store()
    test_rollback_failure_preserves_real_error()
    test_integrity_retry_scoped_to_unique()
    test_empty_env_db_path()
    test_owned_ranges_always_partition()
    test_device_fallback_covers_placement_failure()
    test_concurrent_migration_of_a_legacy_db()
    test_failed_init_closes_the_connection()
    test_info_reports_reality_after_unload()
    test_device_fallback_cleared_on_recovery()
    print("ROBUSTNESS TESTS PASSED")
