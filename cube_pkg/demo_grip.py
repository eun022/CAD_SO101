"""Two flat jaws close on a cube, lift it and put it back.

python demo_grip.py medium
python demo_grip.py medium --no-viewer
python demo_grip.py hard --close 2.5
"""
import argparse
import os
import time

import mujoco
import numpy as np

from cube_force import CubeForces

HERE = os.path.dirname(os.path.abspath(__file__))
CUBES = os.path.join(HERE, "cubes")
SIZE = 0.030
GAP = 0.003          # start gap between jaw and cube [m]
JAW_T = 0.004
KP_JAW = 5000.0      # N/m
LIFT = 0.020
T_CLOSE, T_HOLD, T_LIFT = 1.5, 1.0, 1.5


def scene_xml(level, floor_friction=0.6, timestep=0.001):
    h = SIZE / 2
    xj = h + GAP + JAW_T / 2
    jaw = f'type="box" size="{JAW_T/2} 0.010 0.012" friction="1 0.005 0.0001" rgba="0.82 0.82 0.82 1"'
    return f"""<mujoco model="grip_demo">
  <option timestep="{timestep}" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
  <visual><global offwidth="1920" offheight="1080"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.92 0.92 0.92" rgb2="0.85 0.85 0.85"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <include file="cubes/cube_{level}.xml"/>
  <worldbody>
    <light pos="0 -0.3 0.5" dir="0 0.5 -1"/>
    <geom name="floor" type="plane" size="0.3 0.3 0.01" material="grid" friction="{floor_friction} 0.005 0.0001"/>
    <body name="hand" pos="0 0 0.016" gravcomp="1">
      <joint name="hand_z" type="slide" axis="0 0 1" damping="20"/>
      <geom type="box" size="{xj + 0.004} 0.010 0.003" pos="0 0 0.030" mass="0.2" rgba="0.6 0.6 0.6 1"
            contype="0" conaffinity="0"/>
      <geom name="jaw_fixed" pos="{-xj} 0 0" mass="0.02" {jaw}/>
      <body name="jaw_moving_body" pos="{xj} 0 0" gravcomp="1">
        <joint name="jaw_x" type="slide" axis="-1 0 0" damping="20"/>
        <geom name="jaw_moving" mass="0.05" {jaw}/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="lift" joint="hand_z" kp="5000" ctrlrange="0 0.05"/>
    <position name="close" joint="jaw_x" kp="{KP_JAW:g}" ctrlrange="0 0.02"/>
  </actuator>
</mujoco>"""


def build(level, floor_friction=0.6, timestep=0.001):
    path = os.path.join(HERE, "_grip_scene.xml")
    with open(path, "w") as fh:
        fh.write(scene_xml(level, floor_friction, timestep))
    return mujoco.MjModel.from_xml_path(path)


def ramp(t, t0, dur):
    s = np.clip((t - t0) / dur, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * s)


def control(t, close_mm, lift=True):
    """Lift and jaw targets [m] at time t."""
    t1 = T_CLOSE
    t2 = t1 + T_HOLD
    t3 = t2 + T_LIFT
    t4 = t3 + T_HOLD
    t5 = t4 + T_LIFT
    t6 = t5 + T_HOLD
    jaw_full = 2 * GAP + close_mm * 1e-3
    jaw = jaw_full * (ramp(t, 0, T_CLOSE) - ramp(t, t6, T_CLOSE))
    z = LIFT * (ramp(t, t2, T_LIFT) - ramp(t, t4, T_LIFT)) if lift else 0.0
    return z, jaw


PERIOD = 2 * T_CLOSE + 3 * T_HOLD + 2 * T_LIFT + 0.5


def jaw_contact_forces(model, data, level):
    """Normal contact force [N] between each jaw and its pad."""
    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    pairs = {"xneg": {gid("jaw_fixed"), gid(f"cube_{level}_geom_xneg")},
             "xpos": {gid("jaw_moving"), gid(f"cube_{level}_geom_xpos")}}
    out = {"xneg": 0.0, "xpos": 0.0}
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        for face, pair in pairs.items():
            if {c.geom1, c.geom2} == pair:
                mujoco.mj_contactForce(model, data, i, f6)
                out[face] += f6[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("level", nargs="?", default="medium", choices=["soft", "medium", "hard"])
    ap.add_argument("--close", type=float, default=2.0, help="jaw target past first contact [mm]")
    ap.add_argument("--no-viewer", action="store_true")
    a = ap.parse_args()

    model = build(a.level)
    data = mujoco.MjData(model)
    cubes = CubeForces(model, os.path.join(CUBES, "cube_params.json"))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{a.level}")
    last = -1.0

    def step():
        nonlocal last
        data.ctrl[:] = control(data.time % PERIOD, a.close)
        cubes.apply(data)
        mujoco.mj_step(model, data)
        if data.time - last >= 0.25:
            last = data.time
            c, F = cubes.grip(data)[a.level]
            f = cubes.faces(data)[a.level]
            print(f"t={data.time:5.2f}s  closure {c:4.2f} mm  grip {F:4.2f} N  "
                  f"(fixed side {f['xneg'][0]:4.2f} mm, moving side {f['xpos'][0]:4.2f} mm)  "
                  f"cube height {data.xpos[bid][2]*1e3:5.1f} mm")

    if a.no_viewer:
        while data.time < PERIOD:
            step()
        return
    from mujoco import viewer
    with viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [0, 0, 0.025]
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 0.16, -8, 90
        while v.is_running():
            t0 = time.time()
            for _ in range(10):
                step()
            v.sync()
            time.sleep(max(0.0, 0.01 - (time.time() - t0)))


if __name__ == "__main__":
    main()
