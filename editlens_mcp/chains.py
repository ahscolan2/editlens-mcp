"""SQLite-backed chain store.

A *chain* is a long-running write -> score -> revise loop. State lives on disk,
not in the model's context, so a chain can run for hundreds of steps and across
process restarts. Each chain holds one or more ordered *segments* (sections of a
document); each segment accumulates numbered *steps* (revisions).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
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


class ChainStore:
    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

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
        segs = segments or ["main"]
        self._conn.execute(
            "INSERT INTO chains (id, name, goal, target_score, segments, created_at, updated_at, meta)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (chain_id, name, goal, float(target_score), json.dumps(segs), now, now,
             json.dumps(meta or {})),
        )
        self._conn.commit()
        return {
            "chain_id": chain_id,
            "name": name,
            "goal": goal,
            "target_score": target_score,
            "segments": segs,
        }

    def get(self, chain_id: str) -> sqlite3.Row:
        row = self._conn.execute("SELECT * FROM chains WHERE id = ?", (chain_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such chain: {chain_id}")
        return row

    def list_chains(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT c.id, c.name, c.goal, c.target_score, c.segments, c.updated_at,
                   COUNT(s.id) AS steps, MIN(s.score) AS best
            FROM chains c LEFT JOIN steps s ON s.chain_id = c.id
            GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
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
        self.get(chain_id)
        n = self._conn.execute("DELETE FROM steps WHERE chain_id = ?", (chain_id,)).rowcount
        self._conn.execute("DELETE FROM chains WHERE id = ?", (chain_id,))
        self._conn.commit()
        return n

    def add_segment(self, chain_id: str, segment: str) -> list[str]:
        row = self.get(chain_id)
        segs = json.loads(row["segments"])
        if segment not in segs:
            segs.append(segment)
            self._conn.execute(
                "UPDATE chains SET segments = ?, updated_at = ? WHERE id = ?",
                (json.dumps(segs), time.time(), chain_id),
            )
            self._conn.commit()
        return segs

    # ------------------------------------------------------------------- steps

    def next_step_no(self, chain_id: str, segment: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(step_no), 0) AS n FROM steps WHERE chain_id = ? AND segment = ?",
            (chain_id, segment),
        ).fetchone()
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
        step_no = self.next_step_no(chain_id, segment)
        now = time.time()
        self._conn.execute(
            "INSERT INTO steps (chain_id, segment, step_no, text, score, bucket, label, words,"
            " probs, note, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (chain_id, segment, step_no, text, float(score), int(bucket), label, int(words),
             json.dumps(probs), note, now),
        )
        self._conn.execute("UPDATE chains SET updated_at = ? WHERE id = ?", (now, chain_id))
        self._conn.commit()
        return step_no

    def best_step(self, chain_id: str, segment: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM steps WHERE chain_id = ? AND segment = ?"
            " ORDER BY score ASC, step_no ASC LIMIT 1",
            (chain_id, segment),
        ).fetchone()

    def latest_step(self, chain_id: str, segment: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM steps WHERE chain_id = ? AND segment = ? ORDER BY step_no DESC LIMIT 1",
            (chain_id, segment),
        ).fetchone()

    def get_step(self, chain_id: str, segment: str, step_no: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM steps WHERE chain_id = ? AND segment = ? AND step_no = ?",
            (chain_id, segment, step_no),
        ).fetchone()

    def history(self, chain_id: str, segment: str, limit: int = 30) -> list[dict]:
        rows = self._conn.execute(
            "SELECT step_no, score, label, words, note, created_at FROM steps"
            " WHERE chain_id = ? AND segment = ? ORDER BY step_no DESC LIMIT ?",
            (chain_id, segment, limit),
        ).fetchall()
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
        best = self.best_step(chain_id, segment)
        latest = self.latest_step(chain_id, segment)
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM steps WHERE chain_id = ? AND segment = ?",
            (chain_id, segment),
        ).fetchone()
        return {
            "segment": segment,
            "steps": int(row["n"]),
            "best_step": best["step_no"] if best else None,
            "best_score": round(best["score"], 4) if best else None,
            "latest_step": latest["step_no"] if latest else None,
            "latest_score": round(latest["score"], 4) if latest else None,
            "words": latest["words"] if latest else 0,
        }
