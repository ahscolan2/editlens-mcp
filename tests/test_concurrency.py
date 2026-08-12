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


def test_lineage_read_inside_the_insert_transaction():
    """with_lineage returns the state the insert actually landed on.

    chain_submit used to read latest/best BEFORE scoring -- a window of seconds
    -- and compute is_new_best/best_step from that stale snapshot. 12 racing
    submits each saw an empty segment, so each reported is_new_best=true and
    best_step=<itself>, steering the caller onto the worst draft. The reads now
    live inside add_step's BEGIN IMMEDIATE, where they see every draft
    committed before this one.
    """
    store = ChainStore(TMP / "lineage.db")
    cid = store.create("x", segments=["main"])["chain_id"]
    results, errors = [], []
    barrier = threading.Barrier(8)

    def worker(k):
        try:
            barrier.wait()
            step_no, prev_latest, prev_best = store.add_step(
                cid, "main", f"racer {k}", 0.5 + k * 0.01, 2, "x", 10,
                [0.25] * 4, with_lineage=True)
            results.append((step_no, prev_latest, prev_best))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors[:3]
    assert len(results) == 8

    # Exactly ONE racer may see an empty segment (the one whose transaction ran
    # first). Every other one must see at least the first insert -- the stale
    # pre-scoring read used to hand all 8 a None.
    empty = [r for r in results if r[1] is None]
    assert len(empty) == 1, f"{len(empty)} racers saw an empty segment; only the first may"
    # And each racer's view must be consistent with its own step number: the
    # latest it saw is the step numbered immediately before its own.
    for step_no, prev_latest, prev_best in results:
        if prev_latest is not None:
            assert prev_latest["step_no"] == step_no - 1, (
                step_no, prev_latest["step_no"])
            assert prev_best is not None
    print("  8 racing add_steps: 1 first insert saw None, 7 saw their true "
          "predecessor (stale snapshot would have handed all 8 a None)")

    # register_segment: the declared list and the insert commit together.
    step_no, prev_latest, prev_best = store.add_step(
        cid, "newseg", "first draft", 0.4, 2, "x", 10, [0.25] * 4,
        register_segment=True, with_lineage=True)
    assert step_no == 1 and prev_latest is None
    assert "newseg" in store.segments_of(cid)
    print("  register_segment commits the declared list with the insert")
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
    # The count is a report, not the deletion: assert the rows are actually gone,
    # or a delete that removes nothing still returns 5 and passes.
    gone = store._read("SELECT COUNT(*) AS n FROM steps WHERE chain_id = ?", (drop,))
    assert gone[0]["n"] == 0, f"{gone[0]['n']} orphaned steps survived the delete"
    assert store._read("SELECT 1 FROM chains WHERE id = ?", (drop,)) == []
    try:
        store.get(drop)
        raise AssertionError("deleted chain is still readable")
    except KeyError:
        pass
    # Deleting it again is an error, not a silent success.
    try:
        store.delete(drop)
        raise AssertionError("deleting a missing chain should raise")
    except KeyError:
        pass
    print("  delete removed only its own 5 rows, and removed them")
    store.close()


if __name__ == "__main__":
    print("concurrency tests")
    test_parallel_add_step()
    test_lineage_read_inside_the_insert_transaction()
    test_parallel_add_segment()
    test_two_processes_share_db()
    test_unique_index_enforced()
    test_delete_isolated_from_insert()
    print("CONCURRENCY TESTS PASSED")
