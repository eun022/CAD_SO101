"""Render the grip demo to mp4. MUJOCO_GL=osmesa python render_grip_video.py grip_demo.mp4"""
import json
import os
import sys

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import demo_grip as G
from cube_force import CubeForces

OUT = sys.argv[1] if len(sys.argv) > 1 else "grip_demo.mp4"
LEVEL, CLOSE = "medium", 2.5
FPS = 25
COLORS = {"soft": (0.48, 0.66, 0.91), "medium": (0.16, 0.47, 0.84), "hard": (0.08, 0.25, 0.48)}
INK, MUTED = "#1f2328", "#6e7781"

P = json.load(open(os.path.join(G.CUBES, "cube_params.json")))
model = G.build(LEVEL)
data = mujoco.MjData(model)
cubes = CubeForces(model, os.path.join(G.CUBES, "cube_params.json"))
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{LEVEL}")
renderer = mujoco.Renderer(model, 600, 760)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.026]
cam.distance, cam.elevation, cam.azimuth = 0.17, -10, 100
steps_per_frame = int(round(1.0 / FPS / model.opt.timestep))

fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
ax3d = fig.add_axes([0.02, 0.17, 0.56, 0.68])
ax3d.axis("off")
img = ax3d.imshow(np.zeros((600, 760, 3), dtype=np.uint8))
fig.text(0.02, 0.94, "Grip, hold and lift a free cube", fontsize=17, color=INK, weight="bold")
fig.text(0.02, 0.905, "Left jaw fixed, right jaw moves. Each side of the cube is a pad that follows the FEM grip curve",
         fontsize=11, color=MUTED)
readout = fig.text(0.30, 0.09, "", ha="center", fontsize=13, color=INK, family="DejaVu Sans Mono")

ax = fig.add_axes([0.66, 0.17, 0.31, 0.66])
c = np.array(P["grip_total"]["closure_mm"])
F1 = np.array(P["grip_total"]["F1_N_per_MPa"])
for lvl in ("soft", "medium", "hard"):
    E = P["levels"][lvl]["E_MPa"]
    ax.plot(c, E * F1, color=COLORS[lvl], lw=2.2 if lvl == LEVEL else 1.2, alpha=0.9 if lvl == LEVEL else 0.35)
    ax.text(2.97, E * F1[-1] + 0.04, lvl, ha="right", va="bottom", fontsize=10, color=COLORS[lvl])
dot = ax.plot([], [], "o", ms=11, color=COLORS[LEVEL], mec="white", mew=1.5, zorder=3)[0]
ax.set_xlim(0, 3)
ax.set_ylim(0, 2.1)
ax.set_xlabel("Jaw closure after first contact (mm)", fontsize=11, color=INK)
ax.set_ylabel("Grip force on each jaw (N)", fontsize=11, color=INK)
ax.tick_params(colors=MUTED, labelsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.grid(color="#d0d7de", lw=0.6)
ax.set_axisbelow(True)
fig.text(0.66, 0.905, "Force follows the FEM of the real jaws", fontsize=13, color=INK, weight="bold")
fig.text(0.66, 0.87, "lines: E × FEM grip curve    dot: simulation now", fontsize=10, color=MUTED)
clock = fig.text(0.97, 0.03, "", ha="right", fontsize=10, color=MUTED)

writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=None,
                            ffmpeg_params=["-preset", "veryfast", "-crf", "20"], pixelformat="yuv420p")
n_frames = int(G.PERIOD * FPS)
for k in range(n_frames):
    for _ in range(steps_per_frame):
        data.ctrl[:] = G.control(data.time, CLOSE)
        cubes.apply(data)
        mujoco.mj_step(model, data)
    renderer.update_scene(data, cam)
    img.set_data(renderer.render())
    closure, F = cubes.grip(data)[LEVEL]
    readout.set_text(f"closure {closure:4.2f} mm   grip {F:4.2f} N   height {max(data.xpos[bid][2], 0) * 1e3:4.1f} mm")
    dot.set_data([closure], [F])
    clock.set_text(f"t = {data.time:4.2f} s")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    writer.append_data(frame)
    if k in (int(2.0 * FPS), int(4.5 * FPS)):
        imageio.imwrite(f"_frame_{k}.png", frame)
writer.close()
print("wrote", OUT, n_frames, "frames")
