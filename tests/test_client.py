"""Launch the server exactly as the MCP clients do -- a real stdio subprocess --
and drive it. This is the path Antigravity uses.
"""

import asyncio, json, os, sys, tempfile
from pathlib import Path

PY = r"C:\Users\Aryan\AppData\Local\Programs\Python\Python313\python.exe"
SCRIPT = r"C:\Users\Aryan\MCP-EditLens\run_server.py"

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def main():
    env = dict(os.environ)
    env["EDITLENS_DB"] = str(Path(tempfile.mkdtemp()) / "sub.db")
    # Launch from an unrelated cwd: clients rarely set one.
    env.pop("PYTHONPATH", None)

    transport = StdioTransport(command=PY, args=[SCRIPT], env=env, cwd="C:\\Windows")
    async with Client(transport) as c:
        tools = sorted(t.name for t in await c.list_tools())
        print(f"connected. {len(tools)} tools: {', '.join(tools)}")

        info = (await c.call_tool("detector_info", {})).data
        print(f"info -> device={info['device']} dtype={info['dtype']} "
              f"loaded={info['loaded']} idle_unload={info['idle_unload_seconds']}s")

        r = (await c.call_tool("detect", {"text":
            "In today's rapidly evolving landscape, stakeholders must leverage "
            "synergies to drive transformative outcomes across the ecosystem."})).data
        print(f"detect -> ok={r['ok']} score={r['score']} label={r['label']}")
        assert r["ok"], r

        r = (await c.call_tool("detect", {"text":
            "I burnt the rice again. Third time this month. My flatmate has started "
            "calling it a tradition, which is generous considering she has to eat it."})).data
        print(f"detect(human) -> score={r['score']} label={r['label']}")

        r = (await c.call_tool("detect_spans", {"text":
            "The quarterly results exceeded expectations. Moreover, it is important "
            "to note that robust frameworks enable scalable growth across verticals. "
            "I forgot to send the invoice though. Sorry about that."})).data
        print(f"detect_spans -> doc={r['document_score']} units={r['unit_count']} "
              f"worst={r['worst_units'][0]['score']}")

        cid = (await c.call_tool("chain_create", {
            "name": "sub", "target_score": 0.3, "segments": ["a", "b"]})).data["chain_id"]
        s1 = (await c.call_tool("chain_submit", {
            "chain_id": cid, "text": "Moreover, leveraging synergies remains paramount "
            "for stakeholders navigating the evolving landscape.", "segment": "a"})).data
        print(f"chain_submit -> step={s1['step']} score={s1['score']} "
              f"spans={len(s1.get('worst_spans', []))} next={s1['next_action'][:38]}...")
        s2 = (await c.call_tool("chain_submit", {
            "chain_id": cid, "text": "We kept the same suppliers. Costs went up anyway, "
            "mostly freight. Nobody saw that coming.", "segment": "b"})).data
        print(f"chain_submit -> step={s2['step']} score={s2['score']} met={s2['target_met']}")

        asm = (await c.call_tool("chain_assemble", {"chain_id": cid})).data
        print(f"chain_assemble -> doc={asm['document_score']} segments={len(asm['per_segment'])}")

        st = (await c.call_tool("chain_status", {"chain_id": cid})).data
        print(f"chain_status -> steps={st['total_steps']} pending={st['pending']}")

        u = (await c.call_tool("detector_unload", {})).data
        print(f"detector_unload -> {u['message']}")

        r = (await c.call_tool("detect", {"text": "reloads fine after an unload, hopefully"})).data
        print(f"detect after unload -> ok={r['ok']} score={r['score']}")
        assert r["ok"]

        # concurrent calls over one connection: the "Already borrowed" repro
        rs = await asyncio.gather(*[
            c.call_tool("detect", {"text": f"Concurrent request number {i} of many."})
            for i in range(10)
        ])
        oks = [x.data["ok"] for x in rs]
        print(f"10 concurrent detects -> all_ok={all(oks)}")
        assert all(oks), [x.data for x in rs if not x.data["ok"]]

    print("\nSUBPROCESS TEST PASSED")


asyncio.run(main())
