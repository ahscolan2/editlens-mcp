"""Run every test suite. Usage:  python run_tests.py

Suites, fastest first:
  smoke_test.py           plumbing + one real scoring pass
  tests/test_tools.py     all 13 tools over an in-memory MCP client, error paths
  tests/test_detector.py  precision, windowing, edge cases, determinism, chains
  tests/test_gpu_memory.py  idle unload, manual unload, threading under load
  tests/test_client.py    the real path: server as a stdio subprocess
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITES = [
    ("smoke", [sys.executable, "-u", str(ROOT / "smoke_test.py"), "--real"]),
    ("tools", [sys.executable, "-u", str(ROOT / "tests" / "test_tools.py")]),
    ("concurrency", [sys.executable, "-u", str(ROOT / "tests" / "test_concurrency.py")]),
    ("offsets", [sys.executable, "-u", str(ROOT / "tests" / "test_offsets.py")]),
    ("branching", [sys.executable, "-u", str(ROOT / "tests" / "test_branching.py")]),
    ("robustness", [sys.executable, "-u", str(ROOT / "tests" / "test_robustness.py")]),
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
