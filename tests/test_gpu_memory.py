"""Verify GPU memory is actually released on idle and on demand."""

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from editlens_mcp.detector import EditLensDetector  # noqa: E402
import torch  # noqa: E402

TXT = "In today's rapidly evolving landscape, stakeholders leverage synergies to drive outcomes."


def mb():
    return torch.cuda.memory_allocated() / 1024**2


print("=== manual unload ===")
d = EditLensDetector(device="cuda", idle_unload_seconds=0)  # watchdog off
print(f"  before first use: loaded={d.loaded}  vram={mb():.0f}MB")
s1 = d.detect(TXT)[0].score
after = mb()
print(f"  after detect:     loaded={d.loaded}  vram={after:.0f}MB  score={s1:.4f}")
assert after > 500, "model should occupy VRAM once loaded"

assert d.unload() is True
freed = mb()
print(f"  after unload:     loaded={d.loaded}  vram={freed:.0f}MB")
assert freed < 50, f"VRAM not released: {freed:.0f}MB"
assert d.unload() is False, "second unload should report nothing to do"

s2 = d.detect(TXT)[0].score
print(f"  auto-reload:      vram={mb():.0f}MB  score={s2:.4f}  same={s1 == s2}")
assert s1 == s2, "score changed across unload/reload"
d.unload()

print("\n=== idle watchdog ===")
w = EditLensDetector(device="cuda", idle_unload_seconds=3)
w.detect(TXT)
print(f"  loaded, vram={mb():.0f}MB, idle timeout 3s")
deadline = time.time() + 25
while time.time() < deadline and w.loaded:
    time.sleep(1)
time.sleep(0.5)  # let the watchdog finish its bookkeeping before reading counters
print(f"  after {'unload' if not w.loaded else 'TIMEOUT'}: loaded={w.loaded} "
      f"vram={mb():.0f}MB auto_unloads={w.info()['auto_unloads']}")
assert not w.loaded, "watchdog never fired"
assert mb() < 50, "watchdog did not free VRAM"
assert w.info()["auto_unloads"] == 1, w.info()["auto_unloads"]

print("\n=== watchdog must not unload mid-request ===")
r = EditLensDetector(device="cuda", idle_unload_seconds=1)
r.detect(TXT)
errors = []
long_doc = TXT * 200


def hammer():
    try:
        for _ in range(12):
            r.detect(long_doc)
            r.detect_many([TXT + f" {i}" for i in range(6)])
    except Exception as e:
        errors.append(e)


threads = [threading.Thread(target=hammer) for _ in range(3)]
[t.start() for t in threads]
[t.join() for t in threads]
print(f"  3 threads x 12 rounds under a 1s idle timeout -> errors={errors}")
assert not errors, errors
r.unload()
print(f"  final vram={mb():.0f}MB")

print("\nUNLOAD TESTS PASSED")
