"""Startup configuration and the helper scripts.

These paths run before the server can report anything, or instead of it:

* a bad value in an env var kills the process at import, and the MCP client sees
  only a launch that exited;
* the smoke test imports the server, which opens a ChainStore the moment it is
  imported -- at the operator's real database unless something says otherwise;
* setup.py is the tool people run when their install is broken, so it must
  survive the broken install rather than tracebacking on it.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Set before importing the server: it opens a ChainStore at import time.
os.environ["EDITLENS_DB"] = str(Path(tempfile.mkdtemp()) / "entry.db")

PY = sys.executable


def _run(code: str, env_extra: dict, timeout: float = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["EDITLENS_DB"] = str(Path(tempfile.mkdtemp()) / "case.db")
    env.update(env_extra)
    return subprocess.run([PY, "-c", code], env=env, capture_output=True,
                          text=True, cwd=str(ROOT), timeout=timeout)


IMPORT_PROBE = (
    "import sys; sys.path.insert(0, r'%s')\n"
    "from editlens_mcp import server\n"
    "print('BATCH', server.detector.batch_size)\n"
    "print('IDLE', server.detector.idle_unload_seconds)\n"
) % ROOT


def test_unusable_env_numbers_do_not_kill_the_import():
    """`EDITLENS_BATCH_SIZE=''` used to raise ValueError at module import.

    Client configs routinely emit an empty string for a field the user left
    blank. EDITLENS_DEVICE and EDITLENS_DTYPE already tolerate it (`or None`),
    and EDITLENS_DB was fixed for exactly this; these two were not, so the server
    died before `main()` and the client saw a dead process with no diagnostic.
    """
    cases = [
        ({"EDITLENS_BATCH_SIZE": ""}, 8, 300.0),
        ({"EDITLENS_IDLE_UNLOAD": ""}, 8, 300.0),
        ({"EDITLENS_BATCH_SIZE": "   "}, 8, 300.0),
        ({"EDITLENS_BATCH_SIZE": "abc"}, 8, 300.0),
        ({"EDITLENS_IDLE_UNLOAD": "five minutes"}, 8, 300.0),
        ({"EDITLENS_BATCH_SIZE": "4", "EDITLENS_IDLE_UNLOAD": "12"}, 4, 12.0),
    ]
    for extra, want_batch, want_idle in cases:
        p = _run(IMPORT_PROBE, extra)
        assert p.returncode == 0, (
            f"{extra} killed the server at import:\n{p.stderr[-600:]}")
        out = dict(line.split(" ", 1) for line in p.stdout.strip().splitlines()
                   if line.startswith(("BATCH ", "IDLE ")))
        assert int(out["BATCH"]) == want_batch, (extra, out)
        assert float(out["IDLE"]) == want_idle, (extra, out)
    print(f"  {len(cases)} env configurations import cleanly and parse as expected")


def test_empty_transport_falls_back_to_stdio():
    """`EDITLENS_TRANSPORT=''` died with `ValueError: Unknown transport: `.

    Same empty-string-from-a-client-config case as the numeric settings, but on
    the startup line itself, and the message names nothing at all. A genuinely
    wrong value should still fail -- it quotes what was set.
    """
    boot = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from editlens_mcp import server\n"
        "server.mcp.run = lambda **kw: print('TRANSPORT', repr(kw.get('transport')))\n"
        "server.main()\n"
    ) % ROOT

    empty = _run(boot, {"EDITLENS_TRANSPORT": ""})
    assert empty.returncode == 0, (
        f"empty EDITLENS_TRANSPORT killed startup:\n{empty.stderr[-500:]}")
    assert "TRANSPORT 'stdio'" in empty.stdout, empty.stdout[-300:]

    unset = _run(boot, {})
    assert "TRANSPORT 'stdio'" in unset.stdout, unset.stdout[-300:]

    explicit = _run(boot, {"EDITLENS_TRANSPORT": "http"})
    assert "TRANSPORT 'http'" in explicit.stdout, explicit.stdout[-300:]
    print("  empty EDITLENS_TRANSPORT falls back to stdio; an explicit one is honoured")


def test_batch_size_below_one_is_clamped():
    """batch_size 0/-1 is not slow, it is broken on every single call.

    `_score_batch` steps `range(0, n, batch_size)`: 0 raises "range() arg 3 must
    not be zero" and a negative value scores nothing and then divides by zero.
    The server started fine and then failed every request with a message naming
    nothing the operator had set.
    """
    from editlens_mcp.detector import EditLensDetector

    for given in (0, -4, -1):
        assert EditLensDetector(batch_size=given).batch_size == 1, given
    assert EditLensDetector(batch_size=8).batch_size == 8

    p = _run(IMPORT_PROBE, {"EDITLENS_BATCH_SIZE": "0"})
    assert p.returncode == 0, p.stderr[-400:]
    batch = int(p.stdout.split("BATCH ", 1)[1].split()[0])
    assert batch >= 1, f"EDITLENS_BATCH_SIZE=0 left batch_size={batch}"

    # End to end: a real scoring call must succeed under the clamped value.
    # Pinned to CPU -- the batching loop is device-independent, and this keeps the
    # check off a GPU that may be busy with another job.
    scored = _run(
        IMPORT_PROBE + (
            "r = server.detect(text='I burnt the rice again. Third time this month.')\n"
            "print('DETECT', r.get('ok'), r.get('error'))\n"
        ),
        {"EDITLENS_BATCH_SIZE": "0", "EDITLENS_DEVICE": "cpu"},
        timeout=900,
    )
    assert "DETECT True" in scored.stdout, (
        f"detect failed with EDITLENS_BATCH_SIZE=0:\n{scored.stdout[-400:]}\n"
        f"{scored.stderr[-400:]}")
    print("  batch_size <= 0 clamped to 1; a real detect succeeds under it")


def test_smoke_test_never_touches_the_default_database():
    """`python run_tests.py` must not write the operator's live chain store.

    smoke_test.py imports editlens_mcp.server, and that import alone opens a
    ChainStore -- creating the file, converting it to WAL and running ALTER
    TABLE / CREATE UNIQUE INDEX against whatever is already there.
    """
    fake_home = Path(tempfile.mkdtemp())
    env = dict(os.environ)
    for var in ("EDITLENS_DB",):
        env.pop(var, None)
    # default_db_path() reads these, per platform.
    env["LOCALAPPDATA"] = str(fake_home)
    env["XDG_DATA_HOME"] = str(fake_home)
    env["USERPROFILE"] = str(fake_home)
    env["HOME"] = str(fake_home)

    p = subprocess.run([PY, "-u", str(ROOT / "smoke_test.py")], env=env,
                       capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    assert p.returncode == 0, p.stdout[-600:] + p.stderr[-600:]

    stray = sorted(str(q.relative_to(fake_home)) for q in fake_home.rglob("chains.db*"))
    assert not stray, (
        f"smoke_test.py wrote the default database location: {stray}")
    print("  smoke_test.py leaves the default DB location untouched")


def test_setup_survives_a_torch_that_raises_oserror():
    """setup.py repairs broken installs, so it must not die on one.

    A torch that is installed but unloadable raises OSError on Windows
    ("[WinError 126] ... error loading fbgemm.dll") -- detector.py catches
    broadly for precisely this. setup.py caught only ImportError, so the case it
    exists to fix escaped as a raw traceback and no reinstall was attempted.
    """
    fake = Path(tempfile.mkdtemp())
    (fake / "torch").mkdir()
    (fake / "torch" / "__init__.py").write_text(
        'raise OSError("[WinError 126] The specified module could not be found. '
        'Error loading \\"fbgemm.dll\\" or one of its dependencies.")\n',
        encoding="utf-8",
    )

    harness = (
        "import sys\n"
        f"sys.path.insert(0, r'{fake}')\n"
        f"sys.path.insert(0, r'{ROOT}')\n"
        "import setup\n"
        "calls = []\n"
        "setup.run = lambda cmd: (calls.append(cmd), 0)[1]\n"
        "setup.check_hf = lambda: None\n"
        "rc = setup.main()\n"
        "print('RC', rc)\n"
        "print('INSTALLED_TORCH', any('torch' in c for c in calls))\n"
    )
    p = subprocess.run([PY, "-c", harness], capture_output=True, text=True,
                       cwd=str(ROOT), timeout=600)
    assert "Traceback" not in p.stderr, (
        f"setup.py crashed on an unloadable torch:\n{p.stderr[-800:]}")
    assert "RC " in p.stdout, p.stdout[-400:] + p.stderr[-400:]
    assert "INSTALLED_TORCH True" in p.stdout, (
        f"setup.py did not attempt to reinstall torch:\n{p.stdout[-400:]}")
    print("  unloadable torch triggers the reinstall path instead of a traceback")


def test_submit_losing_a_race_with_delete_reports_the_chain(monkey_text="Some draft text here. It has two sentences."):
    """chain_delete between the lookup and the insert must not leak SQL.

    chain_submit reads the chain, then SCORES the draft (slow), then inserts.
    A delete inside that window hit the chain_id foreign key and the caller got
    "FOREIGN KEY constraint failed" -- naming neither the chain nor the cause --
    while submitting to an already-deleted chain returns a plain "no such chain".
    """
    from editlens_mcp import server

    cid = server.chain_create(name="doomed")["chain_id"]
    server.chain_submit(chain_id=cid, text=monkey_text, span_feedback=False)

    real = server.detector

    class DeletesMidScore:
        """Stands in for chain_delete landing while the draft is being scored."""

        def __init__(self, wrapped, chain_id):
            self._wrapped = wrapped
            self._chain_id = chain_id

        def detect(self, text, normalise=True):
            server.store.delete(self._chain_id)
            return self._wrapped.detect(text, normalise=normalise)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    server.detector = DeletesMidScore(real, cid)
    try:
        r = server.chain_submit(chain_id=cid, text=monkey_text, span_feedback=False)
    finally:
        server.detector = real

    assert r["ok"] is False, r
    assert "foreign key" not in r["error"].lower(), (
        f"raw SQLite error leaked to the caller: {r['error']}")
    assert "no such chain" in r["error"].lower(), r["error"]
    assert cid in r["error"], r["error"]
    assert r["error_type"] == "KeyError", r["error_type"]

    # And the untimed version of the same situation still reads the same way.
    plain = server.chain_submit(chain_id=cid, text=monkey_text, span_feedback=False)
    assert plain["ok"] is False and "no such chain" in plain["error"].lower(), plain
    assert plain["error_type"] == r["error_type"]
    print(f"  delete-during-submit -> {r['error_type']}: {r['error'][:60]}...")


if __name__ == "__main__":
    print("entrypoint / startup tests")
    test_unusable_env_numbers_do_not_kill_the_import()
    test_empty_transport_falls_back_to_stdio()
    test_batch_size_below_one_is_clamped()
    test_smoke_test_never_touches_the_default_database()
    test_setup_survives_a_torch_that_raises_oserror()
    test_submit_losing_a_race_with_delete_reports_the_chain()
    print("ENTRYPOINT TESTS PASSED")
