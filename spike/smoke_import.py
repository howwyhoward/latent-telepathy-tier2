"""Smallest possible Isaac Sim health check for this machine.

Launches Kit headless, creates an empty stage, steps physics a few times,
and reports the active GPU. If this passes, the driver-550-vs-validated-580
question is answered for the headless render stack. Run:

    source setup/env.sh && python spike/smoke_import.py
"""

import time

t0 = time.time()
from isaacsim import SimulationApp

# First launch compiles RTX shaders; it can sit silent for several minutes.
app = SimulationApp({"headless": True})
print(f"[smoke] SimulationApp up in {time.time() - t0:.1f}s")

import carb
import omni.usd
from pxr import UsdGeom

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Cube.Define(stage, "/World/cube")
print("[smoke] stage created, prims:", [p.GetPath() for p in stage.Traverse()])

for i in range(10):
    app.update()
print("[smoke] 10 app updates OK")

gpu = carb.settings.get_settings().get("/renderer/activeGpu")
print(f"[smoke] active GPU index (within CUDA_VISIBLE_DEVICES): {gpu}")

app.close()
print(f"[smoke] PASS — total {time.time() - t0:.1f}s")
