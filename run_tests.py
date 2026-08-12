"""Run every test suite. Usage:  python run_tests.py

Eleven suites, in the order they run (the SUITES list below is the authority;
the README's testing table describes each one):
  smoke, tools, usability, concurrency, offsets, branching, robustness,
  entrypoints, detector, gpu_memory, client

Nothing here may touch the operator's real chain database: EDITLENS_DB is
forced to a temp path below before any suite runs -- unconditionally, even if
one is already set in the environment. An exported EDITLENS_DB is exactly the
setup where a test run would otherwise migrate the real database, which is the
one thing this file promises can never happen. To aim the tests at a location
of your choosing anyway, set EDITLENS_TEST_DB.

All suite temp files live under one directory, removed at the end when every
suite passed and kept (with its path printed) when one failed, so the evidence
survives.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# One parent directory for everything the run creates. The suites call
# tempfile.mkdtemp(), which honours TMPDIR/TEMP/TMP, so pointing those here
# corrals every scratch database and worktree a suite makes -- ten runs used to
# leave ~250 orphaned directories in the system temp.
_RUN_TMP = tempfile.mkdtemp(prefix="editlens-tests-")

# The floor, so that no suite -- present or future -- can fall through to
# default_db_path() and run schema migrations against the operator's live
# chain database. Unconditional: an exported EDITLENS_DB used to be honoured,
# which contradicted the guarantee two paragraphs up and made "verify the
# install" the one command that could touch the real data.
_test_db = (os.environ.get("EDITLENS_TEST_DB") or "").strip()
os.environ["EDITLENS_DB"] = _test_db or str(Path(_RUN_TMP) / "run_tests.db")
for _var in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_var] = _RUN_TMP

SUITES = [
    ("smoke", [sys.executable, "-u", str(ROOT / "smoke_test.py"), "--real"]),
    ("tools", [sys.executable, "-u", str(ROOT / "tests" / "test_tools.py")]),
    ("usability", [sys.executable, "-u", str(ROOT / "tests" / "test_usability.py")]),
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
    try:
        for name, cmd in SUITES:
            print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
            if subprocess.run(cmd, cwd=ROOT).returncode != 0:
                failed.append(name)
    finally:
        if failed:
            print(f"\nKept test scratch dir for inspection: {_RUN_TMP}")
        else:
            shutil.rmtree(_RUN_TMP, ignore_errors=True)
    print(f"\n{'=' * 70}")
    print(f"FAILED: {', '.join(failed)}" if failed else "ALL SUITES PASSED")
    sys.exit(1 if failed else 0)
