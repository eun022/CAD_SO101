"""Checks for the cubes. python validate_faces.py"""
import json
import os

import mujoco
import numpy as np

import demo_grip as G
from cube_force import CubeForces

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "cubes", "cube_params.json")))
LEVELS = ("soft", "medium", "hard")
R = 0.010


def fem_press(level, d_mm):
    c = P["curves"]["press"]
    return P["levels"][level]["E_MPa"] * float(np.interp(d_mm, c["depth_mm"], c["F1_N_per_MPa"]))


def fem_grip(level, closure_mm):
    g = P["grip_total"]
    return P["levels"][level]["E_MPa"] * float(np.interp(closure_mm, g["closure_mm"], g["F1_N_per_MPa"]))


def press_scene(level, theta_deg, mass, cube_dir):
    """Probe pressing the top face along a line tilted by theta."""
    t = np.radians(theta_deg)
    a = np.array([np.sin(t), 0.0, -np.cos(t)])
    start = np.array([0.0, 0.0, G.SIZE + R]) - 0.003 * a
    return f"""<mujoco>
  <option timestep="0.001" cone="elliptic" impratio="10"/>
  <include file="{cube_dir}/cube_{level}.xml"/>
  <worldbody>
    <geom name="floor" type="plane" size="0.3 0.3 0.01"/>
    <body name="probe" pos="{start[0]} 0 {start[2]}" gravcomp="1">
      <joint name="probe_s" type="slide" axis="{a[0]} 0 {a[2]}" damping="5"/>
      <geom name="probe" type="sphere" size="{R}" mass="{mass}" friction="1 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator><position joint="probe_s" kp="3000" ctrlrange="0 0.02"/></actuator>
</mujoco>"""


def run_press(level, theta_deg=0.0, mass=0.05, cube_dir="cubes", travel_mm=5.5):
    path = os.path.join(HERE, "_press_scene.xml")
    open(path, "w").write(press_scene(level, theta_deg, mass, cube_dir))
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    cubes = CubeForces(m, os.path.join(HERE, cube_dir, "cube_params.json"))
    while d.time < 2.5:
        d.ctrl[0] = travel_mm * 1e-3 * G.ramp(d.time, 0, 1.5)
        cubes.apply(d)
        mujoco.mj_step(m, d)
    gp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe")
    gt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"cube_{level}_geom_top")
    fn = ft = 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if {c.geom1, c.geom2} == {gp, gt}:
            mujoco.mj_contactForce(m, d, i, f6)
            fn += f6[0]
            ft += np.hypot(f6[1], f6[2])
    depth, f_pad = cubes.force(d)[level]
    return depth, f_pad, fn, ft


def run_grip(level, close_mm, floor_friction=0.6, timestep=0.001, pad_solref=None):
    m = G.build(level, floor_friction, timestep)
    if pad_solref is not None:
        for g in range(m.ngeom):
            if "_geom_" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""):
                m.geom_solref[g, 0] = pad_solref
    d = mujoco.MjData(m)
    cubes = CubeForces(m, os.path.join(G.CUBES, "cube_params.json"))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"cube_{level}")
    t_table = G.T_CLOSE + G.T_HOLD - 0.01
    t_air0 = G.T_CLOSE + G.T_HOLD + G.T_LIFT + 0.05
    t_air1 = t_air0 + G.T_HOLD - 0.1
    out = {}
    while d.time < t_air1 + 0.01:
        d.ctrl[:] = G.control(d.time, close_mm)
        cubes.apply(d)
        mujoco.mj_step(m, d)
        for key, t in (("table", t_table), ("air0", t_air0), ("air1", t_air1)):
            if key not in out and d.time >= t:
                out[key] = {"faces": cubes.faces(d)[level], "grip": cubes.grip(d)[level],
                            "contact": G.jaw_contact_forces(m, d, level), "z_mm": d.xpos[bid][2] * 1e3}
    return out


def main():
    print("1. vertical press on the top pad (free cube on the table), probe mass 0.02 / 0.1 / 0.5 kg")
    for lvl in LEVELS:
        row = []
        for mass in (0.02, 0.1, 0.5):
            depth, f_pad, fn, _ = run_press(lvl, 0.0, mass)
            row.append(f"{depth:.2f} mm: pad {f_pad:.3f}, contact {fn:.3f}, FEM {fem_press(lvl, depth):.3f} N")
        print(f"  {lvl:6s} " + " | ".join(row))

    print("2. oblique press on the top pad (cube welded), probe travels 5.5 mm along a line tilted by theta")
    print("   theta  normal depth  contact normal  FEM at that depth  tangential (friction)")
    for th in (0, 15, 30, 45):
        depth, f_pad, fn, ft = run_press("medium", th, 0.05, "cubes_fixed")
        print(f"   {th:5d}  {depth:8.2f} mm  {fn:10.3f} N  {fem_press('medium', depth):12.3f} N  {ft:10.3f} N")

    print("3. grip: jaw servo target 1 / 2 / 3 mm past first contact, cube held in the air")
    print("   level   closure  pad -x   pad +x   contact -x  contact +x  FEM(closure)  slip in 0.9 s")
    for lvl in LEVELS:
        for close in (1.0, 2.0, 3.0):
            r = run_grip(lvl, close)
            a = r["air1"]
            c = a["grip"][0]
            print(f"   {lvl:6s} {c:6.2f} mm {a['faces']['xneg'][1]:7.3f}  {a['faces']['xpos'][1]:7.3f}  "
                  f"{a['contact']['xneg']:9.3f}  {a['contact']['xpos']:9.3f}  {fem_grip(lvl, c):10.3f} N  "
                  f"{r['air0']['z_mm'] - a['z_mm']:8.3f} mm   (lifted to {a['z_mm']:.1f} mm)")

    print("4. grip on the table before lifting, medium, target 2 mm: effect of table friction")
    for mu in (0.0, 0.6):
        t = run_grip("medium", 2.0, floor_friction=mu)["table"]
        print(f"   table friction {mu}: pad -x {t['faces']['xneg'][1]:.3f} N, pad +x {t['faces']['xpos'][1]:.3f} N, "
              f"FEM {fem_grip('medium', t['grip'][0]):.3f} N")

    print("5. timestep 2 ms, cube held in the air: pad solref 0.004 (default) and 0.01")
    for lvl, close in (("soft", 1.0), ("medium", 2.0), ("hard", 2.0), ("hard", 3.0)):
        row = []
        for sr in (0.004, 0.01):
            a = run_grip(lvl, close, timestep=0.002, pad_solref=sr)["air1"]
            row.append(f"solref {sr}: closure {a['grip'][0]:.2f} mm, pads {a['faces']['xneg'][1]:.3f} / "
                       f"{a['faces']['xpos'][1]:.3f} N, FEM {fem_grip(lvl, a['grip'][0]):.3f} N, height {a['z_mm']:.1f} mm")
        print(f"   {lvl:6s} target {close:.0f} mm  " + "  |  ".join(row))

if __name__ == "__main__":
    main()
