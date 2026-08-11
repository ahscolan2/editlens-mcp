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
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_DB = Path(
    os.environ.get(
        "EDITLENS_DB",
        Path(os.environ.get("LOCALAPPDATA", Path.home())) / "editlens-mcp" / "chains.db",
    )
)

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
    created_at REAL NOT NULL
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
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=%d" % int(timeout * 1000))
        self._conn.executescript(SCHEMA)
        self.duplicate_steps_present = False
        try:
            self._conn.execute(UNIQUE_STEP_INDEX)
        except sqlite3.IntegrityError:
            # Legacy database written before the constraint existed. Leave the
            # data alone -- callers are told via this flag rather than silently
            # losing rows to a dedupe we never asked permission for.
            self.duplicate_steps_present = True

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
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

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
        return [
            {
                "chain_id": r["id"],
                "name": r["name"],
                "goal": r["goal"],
                "target_score": r["target_score"],
                "segments": json.loads(r["segments"]),
                "steps": r["steps"],
                "best_score": round(r["best"], 4) if r["best"] is not None else None,
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def delete(self, chain_id: str) -> int:
        with self._write() as conn:
            if conn.execute("SELECT 1 FROM chains WHERE id = ?", (chain_id,)).fetchone() is None:
                raise KeyError(f"no such chain: {chain_id}")
            n = conn.execute("DELETE FROM steps WHERE chain_id = ?", (chain_id,)).rowcount
            conn.execute("DELETE FROM chains WHERE id = ?", (chain_id,))
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
    ) -> int:
        """Allocate the next step number and insert, atomically.

        The MAX(step_no) read and the INSERT must share one transaction, or two
        concurrent submits both read N and both write N+1 -- after which
        chain_get_text(step=N+1) returns an arbitrary one of the two drafts.
        """
        now = time.time()
        attempts = 0
        while True:
            attempts += 1
            try:
                with self._write() as conn:
                    row = conn.execute(
                        "SELECT COALESCE(MAX(step_no), 0) AS n FROM steps"
                        " WHERE chain_id = ? AND segment = ?",
                        (chain_id, segment),
                    ).fetchone()
                    step_no = int(row["n"]) + 1
                    conn.execute(
                        "INSERT INTO steps (chain_id, segment, step_no, text, score, bucket,"
                        " label, words, probs, note, created_at)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (chain_id, segment, step_no, text, float(score), int(bucket), label,
                         int(words), json.dumps(probs), note, now),
                    )
                    conn.execute(
                        "UPDATE chains SET updated_at = ? WHERE id = ?", (now, chain_id)
                    )
                return step_no
            except sqlite3.IntegrityError:
                # Another process claimed this step number between our read and
                # write. Re-read and retry rather than overwrite their draft.
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
            "SELECT step_no, score, label, words, note, created_at FROM steps"
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
            "words": latest["words"] if latest else 0,
        }
