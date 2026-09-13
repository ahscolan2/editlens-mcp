"""Launch the server exactly as the MCP clients do -- a real stdio subprocess --
and drive it. This is the path Antigravity uses.
"""

import asyncio, json, os, sys, tempfile, time
from pathlib import Path

PY = sys.executable
SCRIPT = str(Path(__file__).resolve().parent.parent / "run_server.py")

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def main():
    workdir = tempfile.mkdtemp()
    env = dict(os.environ)
    env["EDITLENS_DB"] = str(Path(workdir) / "sub.db")
    runtime = Path(workdir) / "runtime"
    env["EDITLENS_RUNTIME_DIR"] = str(runtime)
    env["EDITLENS_WORKER_IDLE"] = "3"
    env["EDITLENS_BACKEND"] = "shared"
    # Launch from an unrelated cwd: clients rarely set one. It must not be the repo
    # root, or the server would import editlens_mcp from cwd and hide a packaging bug.
    # A temp dir is unrelated on every platform; a hardcoded one is not.
    env.pop("PYTHONPATH", None)

    transport = StdioTransport(command=PY, args=[SCRIPT], env=env, cwd=workdir)
    async with Client(transport) as c:
        tools = sorted(t.name for t in await c.list_tools())
        print(f"connected. {len(tools)} tools: {', '.join(tools)}")

        info = (await c.call_tool("detector_info", {})).data
        assert info["backend"] == "shared", info
        worker_pid = info["worker_pid"]
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

        # Independent stdio processes must use the same worker and database.
        # An older client configuration may still name global Python. The
        # launcher must select this checkout's .venv before importing FastMCP.
        global_python = getattr(sys, "_base_executable", PY)
        other_transport = StdioTransport(command=global_python, args=[SCRIPT], env=env, cwd=workdir)
        async with Client(other_transport) as other:
            other_info = (await other.call_tool("detector_info", {})).data
            assert other_info["worker_pid"] == worker_pid, (info, other_info)
            statuses = await asyncio.gather(
                c.call_tool("detect", {"text": "The two clients share inference."}),
                other.call_tool("detect", {"text": "The two clients share inference."}),
            )
            assert all(result.data["ok"] for result in statuses), statuses
            assert statuses[0].data["score"] == statuses[1].data["score"], statuses
            original = "  Saved from client two.\r\n\tExact whitespace stays.\n"
            saved = (await other.call_tool("chain_submit", {
                "chain_id": cid, "segment": "a", "text": original,
                "span_feedback": False, "branch_from": 1,
            })).data
            recovered = (await c.call_tool("chain_get_text", {
                "chain_id": cid, "segment": "a", "step": saved["step"],
            })).data
            assert saved["ok"] and saved["parent_step"] == 1, saved
            assert recovered["text"] == original, recovered
            print(f"2 independent MCP processes -> worker PID {worker_pid}; equal scores, shared draft and branch recovered exactly")

    # The test worker must leave before the runner removes its private runtime.
    deadline = time.monotonic() + 20
    while any(runtime.glob("*.json")) and time.monotonic() < deadline:
        registries = [p for p in runtime.glob("*.json") if not p.name.endswith(".config.json")]
        if not registries:
            break
        await asyncio.sleep(0.1)
    assert not [p for p in runtime.glob("*.json") if not p.name.endswith(".config.json")], "test worker did not exit when idle"
    # Windows' Python venv launcher may hold the worker log briefly after the
    # registry disappears. Only these test-owned logs are removed.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        for log in runtime.glob("*.log"):
            try:
                log.unlink()
            except PermissionError:
                pass
        if not list(runtime.glob("*.log")):
            break
        await asyncio.sleep(0.1)
    assert not list(runtime.glob("*.log")), "test worker retained log handles after exit"

    print("\nSUBPROCESS TEST PASSED")


asyncio.run(main())
