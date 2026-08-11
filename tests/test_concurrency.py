"""Concurrency regression tests.

MCP servers dispatch sync tool functions on a thread pool, and two clients can
share one database file. The original store used a single SQLite connection with
no locking, which silently duplicated step numbers and dropped drafts. These
tests are the guard against that returning.
"""

import os
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from editlens_mcp.chains import ChainStore  # noqa: E402

TMP = Path(tempfile.mkdtemp())


def test_parallel_add_step():
    """N threads submitting to one segment must produce N distinct step numbers."""
    store = ChainStore(TMP / "par.db")
    cid = store.create("x", segments=["main"])["chain_id"]
    n, errors, got = 40, [], []
    barrier = threading.Barrier(8)

    def worker(k):
        try:
            barrier.wait()
            for j in range(n // 8):
                got.append(store.add_step(
                    cid, "main", f"draft {k}-{j}", 0.5, 2, "x", 10, [0.25] * 4))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    dupes = len(got) - len(set(got))
    print(f"  {len(got)} inserts, {len(set(got))} unique step numbers, "
          f"{dupes} duplicates, {len(errors)} errors")
    assert not errors, errors[:3]
    assert dupes == 0, f"{dupes} duplicate step numbers"
    assert sorted(got) == list(range(1, len(got) + 1)), "step numbers not contiguous"

    rows = store._read("SELECT COUNT(*) AS n FROM steps WHERE chain_id = ?", (cid,))
    assert rows[0]["n"] == len(got), "rows lost"
    store.close()


def test_parallel_add_segment():
    """Concurrent add_segment must not lose names to a read-modify-write race."""
    store = ChainStore(TMP / "seg.db")
    cid = store.create("x", segments=["main"])["chain_id"]
    names = [f"seg{i}" for i in range(24)]
    errors = []
    barrier = threading.Barrier(len(names))

    def worker(nm):
        try:
            barrier.wait()
            store.add_segment(cid, nm)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(nm,)) for nm in names]
    [t.start() for t in threads]
    [t.join() for t in threads]

    final = store.segments_of(cid)
    missing = set(names) - set(final)
    print(f"  {len(names)} concurrent add_segment -> {len(final)} in list, {len(missing)} lost")
    assert not errors, errors[:3]
    assert not missing, f"lost segments: {sorted(missing)}"
    store.close()


def test_two_processes_share_db():
    """Two ChainStore objects on one file (i.e. two client processes) stay consistent."""
    path = TMP / "shared.db"
    a, b = ChainStore(path), ChainStore(path)
    cid = a.create("x", segments=["main"])["chain_id"]
    errors, got = [], []
    barrier = threading.Barrier(6)

    def worker(store, k):
        try:
            barrier.wait()
            for j in range(5):
                got.append(store.add_step(
                    cid, "main", f"{k}-{j}", 0.4, 1, "x", 5, [0.25] * 4))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(a if k % 2 else b, k)) for k in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    dupes = len(got) - len(set(got))
    print(f"  2 stores x 3 threads: {len(got)} inserts, {dupes} duplicates, {len(errors)} errors")
    assert not errors, errors[:3]
    assert dupes == 0, f"{dupes} duplicate step numbers across processes"
    a.close(); b.close()


def test_unique_index_enforced():
    """The DB itself must reject a duplicate (chain, segment, step_no)."""
    store = ChainStore(TMP / "uniq.db")
    cid = store.create("x", segments=["main"])["chain_id"]
    store.add_step(cid, "main", "one", 0.5, 2, "x", 3, [0.25] * 4)
    try:
        store._conn.execute(
            "INSERT INTO steps (chain_id, segment, step_no, text, score, bucket, label,"
            " words, probs, note, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cid, "main", 1, "collision", 0.9, 3, "y", 3, "[]", None, 0.0),
        )
        raise AssertionError("duplicate step_no was accepted")
    except sqlite3.IntegrityError:
        print("  duplicate (chain, segment, step_no) rejected by UNIQUE index")
    store.close()


def test_delete_isolated_from_insert():
    """delete() must not commit another operation's half-finished work."""
    store = ChainStore(TMP / "del.db")
    keep = store.create("keep", segments=["main"])["chain_id"]
    drop = store.create("drop", segments=["main"])["chain_id"]
    for i in range(5):
        store.add_step(keep, "main", f"k{i}", 0.3, 1, "x", 4, [0.25] * 4)
        store.add_step(drop, "main", f"d{i}", 0.3, 1, "x", 4, [0.25] * 4)
    assert store.delete(drop) == 5
    remaining = store._read("SELECT COUNT(*) AS n FROM steps WHERE chain_id = ?", (keep,))
    assert remaining[0]["n"] == 5, "unrelated chain lost rows"
    print("  delete removed only its own 5 rows")
    store.close()


if __name__ == "__main__":
    print("concurrency tests")
    test_parallel_add_step()
    test_parallel_add_segment()
    test_two_processes_share_db()
    test_unique_index_enforced()
    test_delete_isolated_from_insert()
    print("CONCURRENCY TESTS PASSED")
