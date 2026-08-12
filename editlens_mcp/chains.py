"""SQLite-backed chain store.

A *chain* is a long-running write -> score -> revise loop. State lives on disk,
not in the model's context, so a chain can run for hundreds of steps and across
process restarts. Each chain holds one or more ordered *segments* (sections of a
document); each segment accumulates numbered *steps* (revisions).

Concurrency: MCP servers dispatch sync tool functions on a thread pool, and two
clients (e.g. Claude Code and Antigravity) can share one database file. Every
method therefore runs under an instance lock, writes go through BEGIN IMMEDIATE
transactions, and (chain_id, segment, step_no) is enforced UNIQUE so a racing
second process cannot duplicate a step number.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

def default_db_path() -> Path:
    """Per-platform application-data location for the chain database."""
    override = os.environ.get("EDITLENS_DB")
    if override:
        # MCP client configs are JSON, not a shell: nothing else ever expands
        # `~` or `$HOME`. Taken verbatim, "~/foo.db" creates a directory
        # literally named `~` under whatever cwd the launcher happened to give
        # us, and the chains silently accumulate wherever that was -- a new
        # location per launcher. Expand, then resolve, so the path means one
        # place on disk regardless of who started the process.
        # expandvars first: "$HOME/foo" and "%LOCALAPPDATA%\\foo" both work.
        return Path(os.path.expandvars(override)).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return (base.expanduser() / "editlens-mcp" / "chains.db").resolve()


DEFAULT_DB = default_db_path()

SCHEMA = """
CREATE TABLE IF NOT EXISTS chains (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    goal         TEXT,
    target_score REAL NOT NULL,
    segments     TEXT NOT NULL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    meta         TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chain_id   TEXT NOT NULL REFERENCES chains(id) ON DELETE CASCADE,
    segment    TEXT NOT NULL,
    step_no    INTEGER NOT NULL,
    text       TEXT NOT NULL,
    score      REAL NOT NULL,
    bucket     INTEGER NOT NULL,
    label      TEXT NOT NULL,
    words      INTEGER NOT NULL,
    probs      TEXT NOT NULL,
    note       TEXT,
    created_at REAL NOT NULL,
    parent_step INTEGER
);
CREATE INDEX IF NOT EXISTS idx_steps_chain ON steps(chain_id, segment, step_no);
CREATE INDEX IF NOT EXISTS idx_steps_score ON steps(chain_id, segment, score);
"""

# Applied separately: fails if a pre-existing database already holds duplicates.
UNIQUE_STEP_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_steps_unique "
    "ON steps(chain_id, segment, step_no)"
)


class ChainStore:
    def __init__(self, path: Path | str = DEFAULT_DB, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Re-entrant so a method may call another without self-deadlocking.
        self._lock = threading.RLock()
        # isolation_level=None -> autocommit; transactions are opened explicitly
        # so a read-then-write sequence cannot be split by another writer.
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, timeout=timeout, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self.duplicate_steps_present = False
        # Close the connection if setup fails part-way. Leaking it holds an OS
        # handle on the database plus its -wal/-shm sidecars, so the file cannot
        # even be deleted afterwards.
        try:
            # busy_timeout FIRST: it governs every later statement. Setting it
            # after journal_mode leaves the riskiest statement unprotected.
            self._conn.execute("PRAGMA busy_timeout=%d" % int(timeout * 1000))
            self.journal_mode = self._set_wal(timeout)
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._migrate(timeout)
            try:
                self._conn.execute(UNIQUE_STEP_INDEX)
            except sqlite3.IntegrityError:
                # Legacy database written before the constraint existed. Leave
                # the data alone -- callers are told via this flag rather than
                # silently losing rows to a dedupe we never asked permission for.
                self.duplicate_steps_present = True
        except BaseException:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            raise

    def _migrate(self, timeout: float) -> None:
        """Add columns missing from databases written by older versions.

        Checking `table_info` and then running ALTER is a check-then-act race:
        two processes opening a pre-branching database both see the column
        missing, both ALTER, and the loser dies with "duplicate column name" --
        at module import, so the client just sees a dead process. Treat losing
        the race as success, which it is: the column exists either way.
        """
        deadline = time.monotonic() + max(2.0, timeout)
        delay = 0.02
        while True:
            try:
                cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(steps)")}
                if "parent_step" in cols:
                    return
                self._conn.execute("ALTER TABLE steps ADD COLUMN parent_step INTEGER")
                return
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "duplicate column" in msg:
                    return  # another process migrated it first
                if "locked" not in msg and "busy" not in msg:
                    raise
                if time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.25)

    def _set_wal(self, timeout: float) -> str:
        """Switch to WAL, retrying by hand.

        SQLite does NOT run the busy handler for a journal-mode change, so
        `busy_timeout` does not cover this statement. Two clients starting
        together therefore raced and one died with "database is locked" -- at
        module import, before the server could report anything, so the client
        just saw a dead process. Retry, and accept a non-WAL mode rather than
        refusing to start (some network filesystems reject WAL outright).
        """
        deadline = time.monotonic() + max(2.0, timeout)
        delay = 0.02
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                row = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()
                return (row[0] if row else "unknown").lower()
            except sqlite3.OperationalError as exc:
                last = exc
                time.sleep(delay)
                delay = min(delay * 2, 0.25)
        try:
            row = self._conn.execute("PRAGMA journal_mode").fetchone()
            return (row[0] if row else "unknown").lower()
        except sqlite3.Error:
            return f"unknown ({type(last).__name__})" if last else "unknown"

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _write(self):
        """Serialise writers in-process and across processes."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                # COMMIT must be inside the try. Outside it, a failing COMMIT
                # left the transaction open forever and every later write died
                # with "cannot start a transaction within a transaction".
                self._conn.execute("COMMIT")
            except BaseException:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    # SQLite auto-rolls back on disk-full and I/O errors, so
                    # ROLLBACK then fails and would SHADOW the real cause --
                    # reporting "cannot rollback" instead of "disk is full".
                    pass
                raise

    def _read(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _read_one(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, args).fetchone()

    # ------------------------------------------------------------------ chains

    def create(
        self,
        name: str,
        target_score: float = 0.25,
        goal: str | None = None,
        segments: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict:
        chain_id = f"ch_{uuid.uuid4().hex[:12]}"
        now = time.time()
        # De-duplicate while preserving declared order: a repeated segment would
        # otherwise be assembled into the document twice.
        segs = list(dict.fromkeys(segments or ["main"]))
        if not segs:
            segs = ["main"]
        with self._write() as conn:
            conn.execute(
                "INSERT INTO chains (id, name, goal, target_score, segments, created_at,"
                " updated_at, meta) VALUES (?,?,?,?,?,?,?,?)",
                (chain_id, name, goal, float(target_score), json.dumps(segs), now, now,
                 json.dumps(meta or {})),
            )
        return {
            "chain_id": chain_id,
            "name": name,
            "goal": goal,
            "target_score": target_score,
            "segments": segs,
        }

    def get(self, chain_id: str) -> sqlite3.Row:
        row = self._read_one("SELECT * FROM chains WHERE id = ?", (chain_id,))
        if row is None:
            raise KeyError(f"no such chain: {chain_id}")
        return row

    def segments_of(self, chain_id: str) -> list[str]:
        return json.loads(self.get(chain_id)["segments"])

    def list_chains(self, limit: int = 50) -> list[dict]:
        rows = self._read(
            """
            SELECT c.id, c.name, c.goal, c.target_score, c.segments, c.updated_at,
                   COUNT(s.id) AS steps, MIN(s.score) AS best
            FROM chains c LEFT JOIN steps s ON s.chain_id = c.id
            GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?
            """,
            (limit,),
        )
        # Per-segment bests in one query, merged in Python. `best_score` alone
        # misled: MIN across ALL segments, so a chain with a 0.90 intro and a
        # 0.05 body listed as 0.05 and read as finished. The honest signals are
        # how many declared segments have a draft at target -- counted against
        # the DECLARED list, so an unstarted segment counts as not-at-target
        # rather than silently dropping out of the arithmetic.
        seg_rows = self._read(
            "SELECT chain_id, segment, MIN(score) AS seg_best FROM steps"
            " GROUP BY chain_id, segment",
            (),
        )
        seg_best: dict[str, dict[str, float]] = {}
        for r in seg_rows:
            seg_best.setdefault(r["chain_id"], {})[r["segment"]] = r["seg_best"]
        out = []
        for r in rows:
            declared = json.loads(r["segments"])
            bests = seg_best.get(r["id"], {})
            target = float(r["target_score"])
            at_target = sum(
                1 for s in declared if s in bests and bests[s] <= target
            )
            out.append({
                "chain_id": r["id"],
                "name": r["name"],
                "goal": r["goal"],
                "target_score": r["target_score"],
                "segments": declared,
                "steps": r["steps"],
                # Lowest step score anywhere in the chain -- NOT the document
                # score and NOT "every segment passed"; see the two fields below.
                "best_score": round(r["best"], 4) if r["best"] is not None else None,
                "segments_at_target": at_target,
                "segments_total": len(declared),
                "updated_at": r["updated_at"],
            })
        return out

    def delete(self, chain_id: str) -> int:
        with self._write() as conn:
            if conn.execute("SELECT 1 FROM chains WHERE id = ?", (chain_id,)).fetchone() is None:
                raise KeyError(f"no such chain: {chain_id}")
            n = conn.execute("DELETE FROM steps WHERE chain_id = ?", (chain_id,)).rowcount
            conn.execute("DELETE FROM chains WHERE id = ?", (chain_id,))
        return n

    def delete_segment(self, chain_id: str, segment: str) -> int:
        """Remove one segment: its steps and its entry in the declared list.

        The escape hatch for a typo'd segment name. Without it, one mistyped
        `segment=` on chain_submit registers a phantom segment that makes
        chain_assemble report the chain incomplete forever, and the only way
        out was deleting the whole chain. Steps and the declared list must
        change in ONE transaction -- removing the entry first would orphan the
        steps out of status/assembly while they still exist, and removing the
        steps first would leave the phantom on a crash between the two.
        """
        with self._write() as conn:
            row = conn.execute(
                "SELECT segments FROM chains WHERE id = ?", (chain_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no such chain: {chain_id}")
            segs = json.loads(row["segments"])
            if segment not in segs:
                raise KeyError(
                    f"no such segment '{segment}' in this chain; have {segs}"
                )
            if len(segs) == 1:
                raise ValueError(
                    "cannot delete the only segment; use chain_delete on the "
                    "whole chain instead"
                )
            segs.remove(segment)
            n = conn.execute(
                "DELETE FROM steps WHERE chain_id = ? AND segment = ?",
                (chain_id, segment),
            ).rowcount
            conn.execute(
                "UPDATE chains SET segments = ?, updated_at = ? WHERE id = ?",
                (json.dumps(segs), time.time(), chain_id),
            )
        return n

    def add_segment(self, chain_id: str, segment: str) -> list[str]:
        """Read-modify-write the segment list inside one transaction.

        Doing this outside a transaction loses segments: two callers read the
        same list, each appends its own name, and the second write erases the
        first. The orphaned drafts then never appear in status or assembly.
        """
        with self._write() as conn:
            row = conn.execute("SELECT segments FROM chains WHERE id = ?", (chain_id,)).fetchone()
            if row is None:
                raise KeyError(f"no such chain: {chain_id}")
            segs = json.loads(row["segments"])
            if segment not in segs:
                segs.append(segment)
                conn.execute(
                    "UPDATE chains SET segments = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(segs), time.time(), chain_id),
                )
        return segs

    # ------------------------------------------------------------------- steps

    def next_step_no(self, chain_id: str, segment: str) -> int:
        row = self._read_one(
            "SELECT COALESCE(MAX(step_no), 0) AS n FROM steps WHERE chain_id = ? AND segment = ?",
            (chain_id, segment),
        )
        return int(row["n"]) + 1

    def add_step(
        self,
        chain_id: str,
        segment: str,
        text: str,
        score: float,
        bucket: int,
        label: str,
        words: int,
        probs: list[float],
        note: str | None = None,
        parent_step: int | None = None,
        register_segment: bool = False,
        with_lineage: bool = False,
    ) -> int | tuple[int, sqlite3.Row | None, sqlite3.Row | None]:
        """Allocate the next step number and insert, atomically.

        `parent_step` records which draft this one was derived from. History is
        append-only, so a revision that makes things worse is never destructive
        -- but without a parent link there is no way to say "this is a second
        attempt from step 2" rather than "this follows step 5".

        The MAX(step_no) read and the INSERT must share one transaction, or two
        concurrent submits both read N and both write N+1 -- after which
        chain_get_text(step=N+1) returns an arbitrary one of the two drafts.

        `register_segment` adds the segment to the chain's declared list in the
        SAME transaction as the insert. Registering in a separate call first
        meant a failed insert (a "database is locked" timeout, say) left a
        declared segment with zero steps -- a phantom that chain_assemble
        reports as incomplete forever.

        `with_lineage=True` returns (step_no, prev_latest, prev_best), both read
        inside this transaction just before the insert. The caller's own
        pre-scoring reads are seconds stale by insert time: under concurrent
        submits every racer computed is_new_best/best_step from the same empty
        snapshot and each declared ITSELF the best, steering the caller onto a
        worse draft. parent_step deliberately does NOT switch to the
        in-transaction latest -- the parent is the draft the caller was actually
        revising, which is the one it read before scoring; stamping a racer's
        draft as the parent would fabricate lineage between drafts written
        independently.
        """
        now = time.time()
        attempts = 0
        while True:
            attempts += 1
            try:
                with self._write() as conn:
                    if register_segment:
                        row = conn.execute(
                            "SELECT segments FROM chains WHERE id = ?", (chain_id,)
                        ).fetchone()
                        if row is None:
                            raise KeyError(f"no such chain: {chain_id}")
                        segs = json.loads(row["segments"])
                        if segment not in segs:
                            segs.append(segment)
                            conn.execute(
                                "UPDATE chains SET segments = ? WHERE id = ?",
                                (json.dumps(segs), chain_id),
                            )
                    prev_latest = prev_best = None
                    if with_lineage:
                        prev_latest = conn.execute(
                            "SELECT * FROM steps WHERE chain_id = ? AND segment = ?"
                            " ORDER BY step_no DESC LIMIT 1",
                            (chain_id, segment),
                        ).fetchone()
                        prev_best = conn.execute(
                            "SELECT * FROM steps WHERE chain_id = ? AND segment = ?"
                            " ORDER BY score ASC, step_no ASC LIMIT 1",
                            (chain_id, segment),
                        ).fetchone()
                    row = conn.execute(
                        "SELECT COALESCE(MAX(step_no), 0) AS n FROM steps"
                        " WHERE chain_id = ? AND segment = ?",
                        (chain_id, segment),
                    ).fetchone()
                    step_no = int(row["n"]) + 1
                    conn.execute(
                        "INSERT INTO steps (chain_id, segment, step_no, text, score, bucket,"
                        " label, words, probs, note, created_at, parent_step)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (chain_id, segment, step_no, text, float(score), int(bucket), label,
                         int(words), json.dumps(probs), note, now,
                         int(parent_step) if parent_step is not None else None),
                    )
                    conn.execute(
                        "UPDATE chains SET updated_at = ? WHERE id = ?", (now, chain_id)
                    )
                if with_lineage:
                    return step_no, prev_latest, prev_best
                return step_no
            except sqlite3.IntegrityError as exc:
                # Only a UNIQUE collision is worth retrying: another process
                # claimed this step number between our read and write, so
                # re-read rather than overwrite their draft. A FOREIGN KEY or
                # NOT NULL violation (e.g. the chain was deleted underneath us)
                # will never succeed, and retrying just burns 100 ms first.
                if "unique" not in str(exc).lower():
                    raise
                if attempts >= 5:
                    raise
                time.sleep(0.01 * attempts)

    def best_step(self, chain_id: str, segment: str) -> sqlite3.Row | None:
        return self._read_one(
            "SELECT * FROM steps WHERE chain_id = ? AND segment = ?"
            " ORDER BY score ASC, step_no ASC LIMIT 1",
            (chain_id, segment),
        )

    def latest_step(self, chain_id: str, segment: str) -> sqlite3.Row | None:
        return self._read_one(
            "SELECT * FROM steps WHERE chain_id = ? AND segment = ? ORDER BY step_no DESC LIMIT 1",
            (chain_id, segment),
        )

    def get_step(self, chain_id: str, segment: str, step_no: int) -> sqlite3.Row | None:
        return self._read_one(
            "SELECT * FROM steps WHERE chain_id = ? AND segment = ? AND step_no = ?",
            (chain_id, segment, step_no),
        )

    def history(self, chain_id: str, segment: str, limit: int = 30) -> list[dict]:
        rows = self._read(
            "SELECT step_no, score, label, words, note, created_at, parent_step FROM steps"
            " WHERE chain_id = ? AND segment = ? ORDER BY step_no DESC LIMIT ?",
            (chain_id, segment, limit),
        )
        return [
            {
                "step": r["step_no"],
                "score": round(r["score"], 4),
                "label": r["label"],
                "words": r["words"],
                "note": r["note"],
                "parent_step": r["parent_step"],
            }
            for r in reversed(rows)
        ]

    def segment_stats(self, chain_id: str, segment: str) -> dict:
        with self._lock:
            best = self.best_step(chain_id, segment)
            latest = self.latest_step(chain_id, segment)
            row = self._read_one(
                "SELECT COUNT(*) AS n FROM steps WHERE chain_id = ? AND segment = ?",
                (chain_id, segment),
            )
        return {
            "segment": segment,
            "steps": int(row["n"]),
            "best_step": best["step_no"] if best else None,
            "best_score": round(best["score"], 4) if best else None,
            "latest_step": latest["step_no"] if latest else None,
            "latest_score": round(latest["score"], 4) if latest else None,
            # `words` is the BEST draft's count -- the draft assembly will use --
            # so it agrees with chain_assemble's per_segment.words. It used to be
            # the latest draft's count sitting beside the best draft's score,
            # which paired a number with a score describing a different text.
            "words": best["words"] if best else 0,
            "best_words": best["words"] if best else 0,
            "latest_words": latest["words"] if latest else 0,
        }
