"""Entry point for MCP clients.

Launch with an absolute path and no working-directory assumption:

    python /path/to/editlens-mcp/run_server.py

This exists because MCP clients differ in whether they honour a `cwd` setting,
and `python -m editlens_mcp.server` silently fails without one.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _venv_is_installed(venv: Path) -> bool:
    """Whether install.py finished in this environment.

    A `.venv` that exists but holds no FastMCP (an interrupted install, or one
    some other tool created) cannot run the server. Switching into it would
    turn a working launch into "No module named 'fastmcp'".
    """
    if os.name == "nt":
        return (venv / "Lib" / "site-packages" / "fastmcp").is_dir()
    return any((venv / "lib").glob("python*/site-packages/fastmcp"))


# Reuse the isolated project environment when a client is still configured with
# a global Python. Escape hatch for operators intentionally using another venv.
project_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
_switch = (__name__ == "__main__" and project_python.exists()
           and (os.environ.get("EDITLENS_USE_CURRENT_PYTHON") or "").strip() != "1"
           and os.path.normcase(str(Path(sys.executable).absolute())) != os.path.normcase(str(project_python.absolute())))
if _switch and not _venv_is_installed(ROOT / ".venv"):
    print("[editlens] ignoring .venv: FastMCP is not installed in it. Run install.py "
          "to finish setup; continuing with the current Python.", file=sys.stderr)
    _switch = False
if _switch:
    argv = [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    if os.name == "nt":
        # Windows has no exec: os.execv starts a NEW process and exits this
        # one immediately with code 0. The MCP client then sees the process
        # it launched exit while the real server runs on the inherited pipes,
        # and a client that watches that exit reports a dead server. Stay as a
        # thin waiting parent instead -- what the venv's own python.exe
        # launcher already does on Windows -- and pass the exit code through.
        import subprocess

        try:
            code = subprocess.call(argv)
        except KeyboardInterrupt:
            code = 130
        raise SystemExit(code)
    os.execv(argv[0], argv)

sys.path.insert(0, str(ROOT))

from editlens_mcp.server import main  # noqa: E402

if __name__ == "__main__":
    main()
