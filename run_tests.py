"""Run every test suite. Usage:  python run_tests.py

Suites, fastest first:
  smoke_test.py           plumbing + one real scoring pass
  tests/test_tools.py     all 13 tools over an in-memory MCP client, error paths
  tests/test_entrypoints.py  startup config, helper scripts, DB isolation
  tests/test_detector.py  precision, windowing, edge cases, determinism, chains
  tests/test_gpu_memory.py  idle unload, manual unload, threading under load
  tests/test_client.py    the real path: server as a stdio subprocess

Nothing here may touch the operator's real chain database; EDITLENS_DB is forced
to a temp path below before any suite runs.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Every suite inherits this. Suites that want their own still set EDITLENS_DB
# before importing the server; this is the floor, so that no suite -- present or
# future -- can fall through to default_db_path() and run schema migrations
# against the operator's live chain database. Empty counts as unset, because
# default_db_path() treats it that way.
if not (os.environ.get("EDITLENS_DB") or "").strip():
    os.environ["EDITLENS_DB"] = str(Path(tempfile.mkdtemp()) / "run_tests.db")

SUITES = [
    ("smoke", [sys.executable, "-u", str(ROOT / "smoke_test.py"), "--real"]),
    ("tools", [sys.executable, "-u", str(ROOT / "tests" / "test_tools.py")]),
    ("concurrency", [sys.executable, "-u", str(ROOT / "tests" / "test_concurrency.py")]),
    ("offsets", [sys.executable, "-u", str(ROOT / "tests" / "test_offsets.py")]),
    ("branching", [sys.executable, "-u", str(ROOT / "tests" / "test_branching.py")]),
    ("robustness", [sys.executable, "-u", str(ROOT / "tests" / "test_robustness.py")]),
    ("entrypoints", [sys.executable, "-u", str(ROOT / "tests" / "test_entrypoints.py")]),
    ("detector", [sys.executable, "-u", str(ROOT / "tests" / "test_detector.py")]),
    ("gpu_memory", [sys.executable, "-u", str(ROOT / "tests" / "test_gpu_memory.py")]),
    ("client", [sys.executable, "-u", str(ROOT / "tests" / "test_client.py")]),
]

if __name__ == "__main__":
    failed = []
    for name, cmd in SUITES:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            failed.append(name)
    print(f"\n{'=' * 70}")
    print(f"FAILED: {', '.join(failed)}" if failed else "ALL SUITES PASSED")
    sys.exit(1 if failed else 0)
