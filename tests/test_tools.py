import asyncio, os, sys, tempfile
from pathlib import Path

tmp = tempfile.mkdtemp()
os.environ["EDITLENS_DB"] = str(Path(tmp) / "rt.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import Client
from editlens_mcp import server


async def main():
    async with Client(server.mcp) as c:
        tools = await c.list_tools()
        print(f"tools: {len(tools)}")

        r = await c.call_tool("chain_create", {
            "name": "rt", "target_score": 0.2, "segments": ["intro", "body"]})
        cid = r.data["chain_id"]
        print("chain_create ->", r.data)

        r = await c.call_tool("chain_status", {"chain_id": cid})
        print("chain_status ->", r.data["pending"], r.data["total_steps"])

        # union-typed arg: this is what would break at the schema layer
        r = await c.call_tool("chain_get_text", {"chain_id": cid, "step": "best"})
        print("chain_get_text(best, empty) ->", r.data)
        r = await c.call_tool("chain_get_text", {"chain_id": cid, "step": 3})
        print("chain_get_text(int, empty) ->", r.data)

        r = await c.call_tool("chain_list", {})
        print("chain_list ->", len(r.data["chains"]))

        # error paths must return ok:false, not raise
        for name, args in [
            ("detect", {"text": ""}),
            ("chain_status", {"chain_id": "nope"}),
            ("chain_assemble", {"chain_id": cid}),
            ("detect_batch", {"texts": []}),
        ]:
            r = await c.call_tool(name, args)
            print(f"{name} err-path -> ok={r.data.get('ok')} {str(r.data.get('error'))[:70]}")

        r = await c.call_tool("chain_delete", {"chain_id": cid})
        print("chain_delete ->", r.data)
    print("ROUNDTRIP OK")


asyncio.run(main())
