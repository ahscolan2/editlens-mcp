"""Cross-platform install helper.

    python install.py

Installs the right PyTorch for this machine, then the remaining dependencies,
then checks Hugging Face access to the gated checkpoint and prints the exact
MCP client config for this install. Safe to re-run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT = "pangram/editlens_roberta-large"
TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu126",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


def quote(cmd: list[str]) -> str:
    """A command line the user can paste, even when a path contains spaces."""
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    import shlex  # noqa: PLC0415
    return shlex.join(cmd)


def run(cmd: list[str]) -> int:
    print(f"\n$ {quote(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def has_nvidia_gpu() -> bool:
    """True when an NVIDIA driver is installed and reports a GPU.

    nvidia-smi ships with the driver on Windows and Linux. Newer Windows
    drivers put it in System32, which is on PATH; older ones did not.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None and os.name == "nt":
        legacy = Path(os.environ.get("ProgramW6432") or r"C:\Program Files") / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        smi = str(legacy) if legacy.exists() else None
    if smi is None:
        return False
    try:
        probe = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and "GPU" in probe.stdout


def torch_install_cmd() -> list[str]:
    base = [sys.executable, "-m", "pip", "install", "torch"]
    if sys.platform == "darwin":
        return base  # macOS wheels ship Metal support
    # Escape hatch for other builds (ROCm, a different CUDA version, a mirror).
    override = (os.environ.get("EDITLENS_TORCH_INDEX") or "").strip()
    if override:
        return base + ["--index-url", override]
    # The CUDA wheel is a ~2.5 GB download that only helps with an NVIDIA GPU;
    # without one the CPU wheel runs the same model from a fraction of that.
    return base + ["--index-url", TORCH_INDEX["cuda" if has_nvidia_gpu() else "cpu"]]


def hf_command() -> str:
    """The `hf` CLI installed next to this interpreter, else whatever is on PATH.

    The venv's Scripts/bin directory is not on PATH unless the venv is
    activated, so shutil.which alone reported a missing command that was
    installed and working.
    """
    local = Path(sys.executable).parent / ("hf.exe" if os.name == "nt" else "hf")
    if local.exists():
        return str(local)
    return shutil.which("hf") or ""


def check_torch() -> str | None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return f"torch not importable: {type(exc).__name__}: {exc}"
    if torch.cuda.is_available():
        where = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        mps = getattr(torch.backends, "mps", None)
        where = "Apple Silicon (Metal)" if mps and mps.is_available() else "CPU"
    print(f"  torch {torch.__version__} -> {where}")
    return None


def check_hf() -> str | None:
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
    except Exception as exc:  # noqa: BLE001
        return f"huggingface_hub not importable: {exc}"
    try:
        # Public model metadata is visible even without gated-file access.
        # A HEAD request to an actual file checks the download permission.
        get_hf_file_metadata(hf_hub_url(CHECKPOINT, "config.json"))
    except Exception as exc:  # noqa: BLE001
        login = quote([hf_command() or "hf", "auth", "login"])
        return (
            f"cannot reach {CHECKPOINT}: {type(exc).__name__}: {exc}\n"
            f"  The checkpoint is GATED. Accept the licence at\n"
            f"    https://huggingface.co/{CHECKPOINT}\n"
            f"  then run:  {login}   (paste a Read token)"
        )
    print(f"  {CHECKPOINT}: access OK")
    return None


def client_config() -> str:
    return json.dumps(
        {
            "mcpServers": {
                "editlens": {
                    "command": sys.executable,
                    "args": [str(ROOT / "run_server.py")],
                }
            }
        },
        indent=2,
    )


def main() -> int:
    # Before spending bandwidth: the server needs 3.10+ (runtime `int | Literal`
    # unions in tool signatures). Without this check, an old python downloads
    # ~2.5 GB of torch successfully and THEN dies on fastmcp with an error that
    # names fastmcp, not Python -- undiagnosable for a non-Python developer.
    # macOS system python3 is still 3.9 on many installs, so this is the first
    # thing a Mac user would have hit.
    if sys.version_info < (3, 10):
        print(
            f"EditLens MCP needs Python 3.10 or newer; this is "
            f"{sys.version.split()[0]} ({sys.executable}).\n"
            f"Install a newer Python (e.g. from python.org or `brew install "
            f"python`) and re-run this script with it."
        )
        return 1
    project_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if (os.environ.get("EDITLENS_USE_CURRENT_PYTHON") != "1"
            and os.path.normcase(str(Path(sys.executable).absolute()))
            != os.path.normcase(str(project_python.absolute()))):
        if not project_python.exists():
            print(f"Creating isolated environment: {ROOT / '.venv'}", flush=True)
            venv.create(ROOT / ".venv", with_pip=True)
        return run([str(project_python), str(ROOT / "install.py")])
    print(f"EditLens MCP setup — {sys.platform}, Python {sys.version.split()[0]}")
    print(f"Project: {ROOT}")

    try:
        import torch  # noqa: F401
        print("\ntorch already installed; skipping install")
    except Exception as exc:  # noqa: BLE001
        # NOT `except ImportError`. The failure this script exists to repair --
        # a torch that is installed but unloadable -- raises OSError on Windows
        # ("[WinError 126] ... error loading fbgemm.dll"), which escaped as a raw
        # traceback instead of triggering the reinstall. detector.py already
        # catches broadly for exactly this reason.
        print(f"\ntorch not usable ({type(exc).__name__}: {exc}); (re)installing")
        if run(torch_install_cmd()) != 0:
            print("\nPyTorch install failed. See https://pytorch.org/get-started/locally/")
            return 1

    if run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")]) != 0:
        return 1

    print("\nChecking install:")
    problems = [p for p in (check_torch(), check_hf()) if p]
    if problems:
        print("\nNot ready yet:")
        for p in problems:
            print(f"  - {p}")
        if not hf_command():
            print("  - the `hf` command is missing; reinstall huggingface_hub")
        return 1

    print("\nReady. Verify with:")
    print(f"  {quote([sys.executable, str(ROOT / 'run_tests.py')])}")
    print("\nMCP client config for this install:\n")
    print(client_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
