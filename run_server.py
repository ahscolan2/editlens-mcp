"""Entry point for MCP clients.

Launch with an absolute path and no working-directory assumption:

    python C:\\Users\\Aryan\\MCP-EditLens\\run_server.py

This exists because MCP clients differ in whether they honour a `cwd` setting,
and `python -m editlens_mcp.server` silently fails without one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from editlens_mcp.server import main  # noqa: E402

if __name__ == "__main__":
    main()
