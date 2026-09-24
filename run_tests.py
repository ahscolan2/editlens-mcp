"""Run every test suite. Usage:

    python run_tests.py              # all suites (needs the gated checkpoint)
    python run_tests.py --no-model   # only suites that need no checkpoint/GPU
    python run_tests.py tools offsets   # just the named suites

The SUITES list below is the authority. `--no-model` is what CI runs: it needs
neither Hugging Face access nor a GPU. Each suite is killed and counted as
failed after EDITLENS_SUITE_TIMEOUT seconds (default 1800), so one hung
subprocess cannot stall the whole run.

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
#
# On macOS it lives under /tmp. The per-user temp dir there is already about
# 50 characters (/var/folders/xx/.../T/), and the robustness suite's
# multiprocessing.Manager binds a Unix socket inside TMPDIR: on Python 3.10
# and 3.11 that overflowed the 104-byte AF_UNIX limit and the suite died with
# "AF_UNIX path too long". (3.12+ falls back to a short directory itself.)
_RUN_TMP = tempfile.mkdtemp(
    prefix="editlens-tests-", dir="/tmp" if sys.platform == "darwin" else None
)

# The floor, so that no suite -- present or future -- can fall through to
# default_db_path() and run schema migrations against the operator's live
# chain database. Unconditional: an exported EDITLENS_DB used to be honoured,
# which contradicted the guarantee two paragraphs up and made "verify the
# install" the one command that could touch the real data.
_test_db = (os.environ.get("EDITLENS_TEST_DB") or "").strip()
os.environ["EDITLENS_DB"] = _test_db or str(Path(_RUN_TMP) / "run_tests.db")
# Test clients must neither reuse nor unload an operator's inference worker.
os.environ["EDITLENS_RUNTIME_DIR"] = str(Path(_RUN_TMP) / "runtime")
os.environ["PYTHONIOENCODING"] = "utf-8"
for _var in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_var] = _RUN_TMP


def _suite(script: str, *args: str) -> list[str]:
    return [sys.executable, "-u", str(ROOT / script), *args]


# (name, command, needs_model). needs_model: loads the gated checkpoint.
SUITES = [
    ("smoke", _suite("smoke_test.py", "--real"), True),
    ("tools", _suite("tests/test_tools.py"), True),
    ("usability", _suite("tests/test_usability.py"), False),
    ("workflow_contract", _suite("tests/test_workflow_contract.py"), False),
    ("store_regressions", _suite("tests/test_store_regressions.py"), False),
    ("reference_parity", _suite("tests/test_reference_parity.py"), True),
    ("shared_worker", _suite("tests/test_shared_worker.py", "--real"), True),
    ("concurrency", _suite("tests/test_concurrency.py"), False),
    ("offsets", _suite("tests/test_offsets.py"), True),
    ("branching", _suite("tests/test_branching.py"), True),
    ("robustness", _suite("tests/test_robustness.py"), False),
    ("entrypoints", _suite("tests/test_entrypoints.py"), True),
    ("detector", _suite("tests/test_detector.py"), True),
    ("gpu_memory", _suite("tests/test_gpu_memory.py"), True),
    ("client", _suite("tests/test_client.py"), True),
]
# Model-free variants of the two suites whose --real half needs the checkpoint.
NO_MODEL_VARIANTS = {
    "smoke": _suite("smoke_test.py"),
    "shared_worker": _suite("tests/test_shared_worker.py"),
}


def select(argv: list[str]) -> list[tuple[str, list[str]]]:
    no_model = "--no-model" in argv
    names = [a for a in argv if not a.startswith("-")]
    known = {name for name, _, _ in SUITES}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"unknown suite(s): {', '.join(unknown)}; have {', '.join(sorted(known))}")
    chosen = []
    for name, cmd, needs_model in SUITES:
        if names and name not in names:
            continue
        if no_model:
            if name in NO_MODEL_VARIANTS:
                cmd = NO_MODEL_VARIANTS[name]
            elif needs_model:
                continue
        chosen.append((name, cmd))
    return chosen


if __name__ == "__main__":
    suite_timeout = float(os.environ.get("EDITLENS_SUITE_TIMEOUT") or 1800)
    try:
        chosen = select(sys.argv[1:])
    except SystemExit:
        shutil.rmtree(_RUN_TMP, ignore_errors=True)
        raise
    failed = []
    try:
        for name, cmd in chosen:
            print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
            try:
                code = subprocess.run(cmd, cwd=ROOT, timeout=suite_timeout).returncode
            except subprocess.TimeoutExpired:
                print(f"\n{name}: no result after {suite_timeout:.0f} s; killed", flush=True)
                code = None
            if code != 0:
                failed.append(name)
    finally:
        if failed:
            print(f"\nKept test scratch dir for inspection: {_RUN_TMP}")
        else:
            shutil.rmtree(_RUN_TMP, ignore_errors=True)
    print(f"\n{'=' * 70}")
    print(f"FAILED: {', '.join(failed)}" if failed else f"ALL {len(chosen)} SUITES PASSED")
    sys.exit(1 if failed else 0)
