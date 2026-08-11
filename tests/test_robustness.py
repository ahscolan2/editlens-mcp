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
import time
from contextlib import contextmanager
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
    """WAL, specifically -- not merely 'some valid journal mode'.

    Accepting any of the six legal modes here meant a mutation that deleted the
    WAL switch entirely still passed: `delete` is a valid mode. WAL is the whole
    reason the switch exists (concurrent readers alongside one writer), so assert
    it, and assert the connection is really in it rather than trusting the
    attribute the store set on itself.
    """
    path = TMP / "mode.db"
    s = ChainStore(path)
    assert s.journal_mode == "wal", s.journal_mode
    live = s._conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    assert live == "wal", f"store reported {s.journal_mode!r} but connection is in {live!r}"
    # A -wal sidecar appears only once the database really journals into one.
    s.create("x")
    assert Path(str(path) + "-wal").exists(), "no -wal sidecar: WAL is not actually on"
    print("  journal_mode is WAL on the live connection, -wal sidecar present")
    s.close()


def _hold_write_lock(path, hold_seconds, ready, released):
    """Take a RESERVED lock on `path` for `hold_seconds`, then release it.

    BEGIN IMMEDIATE is exactly what ChainStore._write() does, so this is a real
    second client mid-write rather than an artificial lock.
    """
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None,
                           timeout=30)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS lockbait (x)")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO lockbait VALUES (1)")
        ready.set()
        time.sleep(hold_seconds)
        conn.execute("COMMIT")
    finally:
        released.set()
        conn.close()


def test_wal_switch_retries_through_a_held_write_lock():
    """Opening a store while another client holds a write lock must still work.

    SQLite does NOT run the busy handler for a journal-mode change when another
    connection holds a RESERVED lock: `PRAGMA journal_mode=WAL` fails instantly
    with "database is locked", busy_timeout notwithstanding (verified: under
    BEGIN IMMEDIATE it raises in ~0ms even with busy_timeout=30000). That is why
    _set_wal retries by hand.

    The old test only had twelve threads open a database at the same time and
    hoped the race fired; it never did, so deleting the retry loop passed. This
    forces the contention instead of hoping for it.
    """
    path = TMP / "wal_contended.db"
    # A pre-existing NON-WAL database: journal_mode=WAL is a no-op once the file
    # is already in WAL, so the switch must be a real one for this to bite.
    seed = sqlite3.connect(str(path), isolation_level=None)
    seed.execute("PRAGMA journal_mode=DELETE")
    seed.execute("CREATE TABLE IF NOT EXISTS lockbait (x)")
    seed.close()

    hold = 1.0
    ready, released = threading.Event(), threading.Event()
    blocker = threading.Thread(
        target=_hold_write_lock, args=(path, hold, ready, released), daemon=True)
    blocker.start()
    assert ready.wait(10), "could not acquire the blocking write lock"

    t0 = time.monotonic()
    store = ChainStore(path, timeout=10.0)   # must not raise "database is locked"
    elapsed = time.monotonic() - t0

    assert store.journal_mode == "wal", (
        f"opened under contention but never reached WAL: {store.journal_mode!r}")
    assert elapsed >= hold * 0.5, (
        f"returned in {elapsed:.2f}s -- the lock was not actually held, so this "
        "test proves nothing")
    assert elapsed < 10.0, f"took {elapsed:.2f}s, longer than the store timeout"
    # And it is a working store, not just a constructed one.
    cid = store.create("after contention")["chain_id"]
    store.add_step(cid, "main", "d", 0.5, 2, "x", 1, [0.25] * 4)
    assert store.latest_step(cid, "main")["step_no"] == 1
    store.close()
    blocker.join(timeout=10)
    print(f"  WAL switch retried through a {hold:.1f}s write lock, succeeded in "
          f"{elapsed:.2f}s")


class RecordingConn:
    """Delegates to a real connection, recording the SQL it is asked to run."""

    def __init__(self, real, log):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_log", log)

    def execute(self, sql, *args):
        self._log.append(sql)
        return self._real.execute(sql, *args)

    def executescript(self, sql):
        self._log.append("<executescript>")
        return self._real.executescript(sql)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_real"), name, value)


def test_busy_timeout_precedes_every_other_statement():
    """busy_timeout must be armed BEFORE the journal-mode switch, not after.

    It governs every statement issued after it, so setting it late leaves the
    startup statements -- the ones two clients actually collide on -- running
    with whatever default happens to be in force.
    """
    log: list[str] = []
    real_connect = sqlite3.connect

    def recording_connect(*a, **kw):
        return RecordingConn(real_connect(*a, **kw), log)

    import editlens_mcp.chains as chains_mod
    chains_mod.sqlite3.connect = recording_connect
    try:
        s = ChainStore(TMP / "order.db", timeout=12.0)
    finally:
        chains_mod.sqlite3.connect = real_connect

    lowered = [q.lower() for q in log]
    busy = next((i for i, q in enumerate(lowered) if "busy_timeout=" in q), None)
    journal = next((i for i, q in enumerate(lowered) if "journal_mode=" in q), None)
    assert busy is not None, f"busy_timeout was never set: {log}"
    assert journal is not None, f"journal_mode was never set: {log}"
    assert busy < journal, (
        f"busy_timeout set at statement {busy}, after journal_mode at {journal}: {log}")
    # And it is actually in force on the connection, at the configured value.
    live = s._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert live == 12000, f"busy_timeout is {live}, expected 12000"
    print(f"  busy_timeout armed at statement {busy}, journal_mode at {journal}; "
          f"live value {live}ms")
    s.close()


def test_writes_wait_out_a_concurrent_writer():
    """A write that collides with another client's transaction must block, not die.

    This is what busy_timeout buys. Without it (on either the connect() call or
    the PRAGMA) an ordinary chain_submit racing a second client fails outright
    with "database is locked".
    """
    path = TMP / "write_contended.db"
    store = ChainStore(path, timeout=15.0)
    cid = store.create("busy")["chain_id"]

    hold = 1.0
    ready, released = threading.Event(), threading.Event()
    blocker = threading.Thread(
        target=_hold_write_lock, args=(path, hold, ready, released), daemon=True)
    blocker.start()
    assert ready.wait(10), "could not acquire the blocking write lock"

    t0 = time.monotonic()
    step = store.add_step(cid, "main", "written under contention", 0.5, 2, "x", 3,
                          [0.25] * 4)
    elapsed = time.monotonic() - t0

    assert step == 1, step
    assert elapsed >= hold * 0.5, (
        f"add_step returned in {elapsed:.2f}s -- the lock was not really held")
    assert store.latest_step(cid, "main")["text"] == "written under contention"
    store.close()
    blocker.join(timeout=10)
    print(f"  add_step waited out a {hold:.1f}s foreign write lock ({elapsed:.2f}s) "
          "instead of failing")


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
        last_span_end = max(s[1] for s in spans)
        # The document is longer than the last window's char span whenever the
        # tokeniser's final offset stops short of the end of the string. That
        # tail still has to be weighted by SOMETHING, or those words vanish from
        # the score entirely.
        for text_len in (last_span_end, last_span_end + 13):
            owned = EditLensDetector._owned_ranges(spans, text_len)
            assert len(owned) == len(spans)
            for a, b in zip(owned, owned[1:]):
                assert a[1] == b[0], f"gap/overlap in {owned} from {spans}"
            for a, b in owned:
                assert a <= b, f"inverted range {(a, b)} in {owned}"
            assert owned[0][0] == spans[0][0], (owned, spans)
            assert owned[-1][1] >= text_len, (
                f"last window owns up to {owned[-1][1]} but the text is "
                f"{text_len} long: the tail is unweighted")
            covered = sum(b - a for a, b in owned)
            assert covered == owned[-1][1] - owned[0][0], (owned, covered)
    print(f"  {len(cases)} pathological window sets partition cleanly, tail included")


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


class _ScriptedConn:
    """A sqlite3 connection wrapper that scripts what happens around ALTER TABLE.

    The 16-thread test above only reaches the migration race by luck: whether a
    thread loses the ALTER and sees "duplicate column name" depends on the GIL,
    and in practice it usually does not. Deleting the tolerance therefore passed
    every suite. These hooks make each branch of _migrate() happen on purpose.
    """

    def __init__(self, conn, on_table_info=None, alter_failures=()):
        object.__setattr__(self, "_c", conn)
        object.__setattr__(self, "_on_table_info", on_table_info)
        object.__setattr__(self, "_alter_failures", list(alter_failures))
        object.__setattr__(self, "_alters", 0)

    def execute(self, sql, *args):
        if self._on_table_info is not None and "table_info(steps)" in sql:
            rows = self._c.execute(sql, *args).fetchall()
            hook = self._on_table_info
            object.__setattr__(self, "_on_table_info", None)  # once only
            hook()  # another client migrates between our read and our ALTER
            return rows
        if "ADD COLUMN parent_step" in sql:
            object.__setattr__(self, "_alters", self._alters + 1)
            if self._alter_failures:
                raise sqlite3.OperationalError(self._alter_failures.pop(0))
        return self._c.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._c, name)

    def __setattr__(self, name, value):
        setattr(self._c, name, value)


@contextmanager
def _scripted_sqlite(**kw):
    real = sqlite3.connect

    def fake(*a, **k):
        return _ScriptedConn(real(*a, **k), **kw)

    sqlite3.connect = fake
    try:
        yield real
    finally:
        sqlite3.connect = real


def test_migration_tolerates_a_lost_alter_race():
    """Losing the ALTER race is success: the column exists either way.

    Without the "duplicate column" tolerance the loser dies at module import and
    the MCP client sees a process that exited, not an error.
    """
    path = Path(_legacy_db(TMP / "dupcol.db"))
    real_connect = sqlite3.connect  # the hook must not re-enter the wrapper

    def someone_else_migrates():
        other = real_connect(str(path), isolation_level=None)
        try:
            other.execute("ALTER TABLE steps ADD COLUMN parent_step INTEGER")
        finally:
            other.close()

    with _scripted_sqlite(on_table_info=someone_else_migrates):
        store = ChainStore(path)  # must not raise "duplicate column name"

    cols = [r["name"] for r in store._conn.execute("PRAGMA table_info(steps)")]
    assert cols.count("parent_step") == 1, cols
    assert store._conn._alters == 1, f"the losing ALTER never ran: {store._conn._alters}"
    assert store.latest_step("ch_old", "main")["text"] == "legacy draft"
    store.add_step("ch_old", "main", "after the race", 0.3, 1, "y", 3, [0.25] * 4,
                   parent_step=1)
    assert store.history("ch_old", "main")[-1]["parent_step"] == 1
    store.close()
    print("  lost ALTER race tolerated; legacy rows intact, parent_step usable")


def test_migration_retries_a_locked_alter_and_still_gives_up():
    """The retry loop: wait out a lock, re-raise anything else, honour the deadline."""
    # 1. Two locked ALTERs in a row, then success.
    path = Path(_legacy_db(TMP / "lockedalter.db"))
    with _scripted_sqlite(alter_failures=["database is locked", "database table is busy"]):
        store = ChainStore(path)
    assert store._conn._alters == 3, (
        f"migration did not retry the locked ALTER: {store._conn._alters} attempt(s)")
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(steps)")}
    assert "parent_step" in cols, cols
    store.close()

    # 2. An error that is NOT a lock must propagate, not be retried or swallowed.
    path2 = Path(_legacy_db(TMP / "badalter.db"))
    with _scripted_sqlite(alter_failures=["no such table: steps"]):
        try:
            ChainStore(path2)
            raise AssertionError("a non-lock migration failure must not be swallowed")
        except sqlite3.OperationalError as exc:
            assert "no such table" in str(exc), exc

    # 3. A lock that never clears must end at the deadline, not retry forever and
    #    not give up instantly.
    path3 = Path(_legacy_db(TMP / "stucklock.db"))
    t0 = time.monotonic()
    with _scripted_sqlite(alter_failures=["database is locked"] * 100000):
        try:
            ChainStore(path3, timeout=0.2)
            raise AssertionError("expected the migration to give up eventually")
        except sqlite3.OperationalError as exc:
            assert "locked" in str(exc), exc
    elapsed = time.monotonic() - t0
    # Floor is max(2.0, timeout); giving up sooner means it never retried.
    assert 1.5 <= elapsed < 30.0, f"gave up after {elapsed:.2f}s"
    print(f"  locked ALTER retried to success; non-lock error propagated; "
          f"stuck lock gave up after {elapsed:.1f}s")


def test_legacy_duplicate_steps_are_flagged_not_deleted():
    """A pre-UNIQUE database keeps its duplicates; the flag is how callers learn."""
    path = Path(_legacy_db(TMP / "dupsteps.db"))
    seed = sqlite3.connect(str(path), isolation_level=None)
    seed.execute(
        "INSERT INTO steps (chain_id, segment, step_no, text, score, bucket, label,"
        " words, probs, note, created_at)"
        " VALUES ('ch_old','main',1,'duplicate draft',0.4,1,'y',2,'[]',NULL,0)")
    seed.close()

    store = ChainStore(path)
    assert store.duplicate_steps_present is True, (
        "a legacy database with duplicate step numbers must report the flag")
    rows = store._conn.execute(
        "SELECT text FROM steps WHERE chain_id='ch_old' AND segment='main'"
        " AND step_no=1").fetchall()
    assert {r["text"] for r in rows} == {"legacy draft", "duplicate draft"}, rows
    assert len(rows) == 2, f"duplicate rows were silently destroyed: {len(rows)} left"
    # The index is absent, so it must not be reported as enforced.
    idx = [r["name"] for r in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'")]
    assert "idx_steps_unique" not in idx, idx
    store.close()

    clean = ChainStore(TMP / "clean_flag.db")
    assert clean.duplicate_steps_present is False, (
        "a fresh database must not claim duplicates")
    clean_idx = [r["name"] for r in clean._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'")]
    assert "idx_steps_unique" in clean_idx, clean_idx
    clean.close()
    print("  legacy duplicates flagged and preserved; fresh DB clean and indexed")


def test_non_positive_idle_unload_disables_the_watchdog():
    """`EDITLENS_IDLE_UNLOAD=0` is documented as "never". A negative value is
    the same instruction typed differently -- and must not instead start a
    watchdog whose deadline is already in the past, which unloads the model
    between every pair of calls."""
    from editlens_mcp.detector import EditLensDetector

    for seconds in (0, 0.0, -1, -300.0):
        det = EditLensDetector(device="cpu", idle_unload_seconds=seconds)
        det._start_watchdog()
        assert det._watchdog is None, f"idle_unload={seconds!r} started a watchdog"

    live = EditLensDetector(device="cpu", idle_unload_seconds=5)
    live._start_watchdog()
    assert live._watchdog is not None and live._watchdog.daemon, (
        "a positive idle timeout must start a daemon watchdog")
    first = live._watchdog
    live._start_watchdog()
    assert live._watchdog is first, "watchdog started twice"
    print("  idle_unload <= 0 starts no watchdog; a positive one starts exactly one")


def test_history_limit_returns_the_most_recent_steps():
    """`limit` must trim the OLDEST steps, and the result stays chronological."""
    store = ChainStore(TMP / "hist.db")
    cid = store.create("h")["chain_id"]
    for i in range(1, 8):
        store.add_step(cid, "main", f"draft {i}", 0.9 - i * 0.05, 1, "x", 10, [0.25] * 4,
                       note=f"n{i}")
    assert len(store.history(cid, "main")) == 7

    trimmed = store.history(cid, "main", limit=3)
    steps = [r["step"] for r in trimmed]
    assert steps == [5, 6, 7], f"limit=3 should keep the last three, got {steps}"
    assert [r["note"] for r in trimmed] == ["n5", "n6", "n7"]
    assert len(store.history(cid, "main", limit=1)) == 1
    assert store.history(cid, "main", limit=1)[0]["step"] == 7
    # A limit larger than the trajectory is not an error.
    assert len(store.history(cid, "main", limit=200)) == 7
    store.close()
    print("  history(limit=N) keeps the N most recent steps, oldest-first")


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
    test_wal_switch_retries_through_a_held_write_lock()
    test_busy_timeout_precedes_every_other_statement()
    test_writes_wait_out_a_concurrent_writer()
    test_simultaneous_open_threads()
    test_simultaneous_open_processes()
    test_failed_commit_does_not_wedge_the_store()
    test_rollback_failure_preserves_real_error()
    test_integrity_retry_scoped_to_unique()
    test_empty_env_db_path()
    test_owned_ranges_always_partition()
    test_device_fallback_covers_placement_failure()
    test_concurrent_migration_of_a_legacy_db()
    test_non_positive_idle_unload_disables_the_watchdog()
    test_migration_tolerates_a_lost_alter_race()
    test_migration_retries_a_locked_alter_and_still_gives_up()
    test_legacy_duplicate_steps_are_flagged_not_deleted()
    test_history_limit_returns_the_most_recent_steps()
    test_failed_init_closes_the_connection()
    test_info_reports_reality_after_unload()
    test_device_fallback_cleared_on_recovery()
    print("ROBUSTNESS TESTS PASSED")
