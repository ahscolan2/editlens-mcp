"""Private entry point for the shared worker, launched by SharedDetector."""

from editlens_mcp.shared import worker_main


if __name__ == "__main__":
    raise SystemExit(worker_main())
