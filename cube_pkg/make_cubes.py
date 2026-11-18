"""Write the soft / medium / hard cube XML files and cube_params.json into cubes/.

python make_cubes.py
python make_cubes.py --pos 0.2 0 0 --mass 0.015
python make_cubes.py --fixed-jaw pos
python make_cubes.py --fixed
python make_cubes.py --solref 0.01
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ("soft", "medium", "hard")
COLORS = {"soft": "0.48 0.66 0.91 1", "medium": "0.16 0.47 0.84 1", "hard": "0.08 0.25 0.48 1"}
PAD_MASS = 0.003
FLOOR_GAP = 0.5  # mm, keeps the side pads off the table


def press_curve(size_mm):
    tab = json.load(open(os.path.join(HERE, "fem_unit_curve.json")))
    D = np.array(tab["depth_mm"])
    sizes = np.array(sorted(float(k) for k in tab["F1_by_size_mm"]))
    curves = np.array([tab["F1_by_size_mm"][f"{s:g}"] for s in sizes])
    s = float(np.clip(size_mm, sizes[0], sizes[-1]))
    return D, np.array([np.interp(s, sizes, curves[:, j]) for j in range(len(D))])


def grip_curves():
    """Split the two-jaw FEM result into one curve per jaw."""
    g = json.load(open(os.path.join(HERE, "fem_grip_curve.json")))
    c = np.array([0.0] + g["closure_mm"])
    F1 = np.array([0.0] + g["F1_N_per_MPa"])
    d_fixed = np.array([0.0] + g["fixed_jaw_share_mm"])
    return (d_fixed, F1), (c - d_fixed, F1), (c, F1)


def cube_xml(name, L, pad, pos, k_lin, rgba, max_depth, mass, free, solref):
    m = 1e-3
    t, h = pad, L / 2
    base_mass = mass - 3 * PAD_MASS
    side_h = L - FLOOR_GAP

    def pad_body(face, centre, size, axis):
        return f"""
      <body name="{name}_pad_{face}" pos="{centre[0]*m:.5f} {centre[1]*m:.5f} {centre[2]*m:.5f}">
        <joint name="{name}_{face}" type="slide" axis="{axis}" limited="true"
               range="{-max_depth*m:.5f} 0" stiffness="{k_lin[face]:.3f}" springref="0" damping="2"/>
        <geom name="{name}_geom_{face}" type="box" size="{size[0]*m:.5f} {size[1]*m:.5f} {size[2]*m:.5f}"
              mass="{PAD_MASS}" rgba="{rgba}" friction="1 0.005 0.0001" solref="{solref:g} 1" solmix="100"/>
      </body>"""

    pads = (pad_body("top", (0, 0, L - t / 2), (h - t, h, t / 2), "0 0 1")
            + pad_body("xneg", (-(h - t / 2), 0, FLOOR_GAP + side_h / 2), (t / 2, h, side_h / 2), "-1 0 0")
            + pad_body("xpos", (h - t / 2, 0, FLOOR_GAP + side_h / 2), (t / 2, h, side_h / 2), "1 0 0"))
    parts = [name] + [f"{name}_pad_{f}" for f in ("top", "xneg", "xpos")]
    excl = "".join(f'\n    <exclude body1="{a}" body2="{b}"/>'
                   for i, a in enumerate(parts) for b in parts[i + 1:])
    joint = "" if not free else f'\n      <freejoint name="{name}_free"/>'
    return f"""<mujoco model="{name}">
  <worldbody>
    <body name="{name}" pos="{pos[0]:g} {pos[1]:g} {pos[2]:g}">{joint}
      <geom name="{name}_base" type="box" size="{(h - t)*m:.5f} {h*m:.5f} {(L - t)/2*m:.5f}"
            pos="0 0 {(L - t)/2*m:.5f}" mass="{base_mass:.4f}" rgba="{rgba}" friction="0.6 0.005 0.0001"/>{pads}
    </body>
  </worldbody>
  <contact>{excl}
  </contact>
</mujoco>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=30.0, help="cube edge [mm], grip curve is for 30 mm")
    ap.add_argument("--pad", type=float, default=4.0, help="pad thickness [mm]")
    ap.add_argument("--targets", type=float, nargs=3, default=[1.0, 2.0, 3.5], help="top press force [N] at the reference depth")
    ap.add_argument("--ref-depth", type=float, default=2.0, help="[mm]")
    ap.add_argument("--max-depth", type=float, default=3.0, help="pad joint limit [mm]")
    ap.add_argument("--mass", type=float, default=0.015, help="cube mass [kg]")
    ap.add_argument("--pos", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="bottom centre of the cube [m]")
    ap.add_argument("--fixed-jaw", choices=["neg", "pos"], default="neg", help="side of the fixed jaw")
    ap.add_argument("--fixed", action="store_true", help="no free joint")
    ap.add_argument("--solref", type=float, default=0.004, help="pad contact solref [s], 0.01 for a 2 ms timestep")
    ap.add_argument("--out", default=os.path.join(HERE, "cubes"))
    a = ap.parse_args()
    if abs(a.size - 30.0) > 1e-6:
        print("grip curve is for a 30 mm cube")

    os.makedirs(a.out, exist_ok=True)
    D, F1 = press_curve(a.size)
    fixed, moving, total = grip_curves()
    side = {"xneg": "grip_fixed", "xpos": "grip_moving"} if a.fixed_jaw == "neg" else \
           {"xneg": "grip_moving", "xpos": "grip_fixed"}
    curves = {"press": {"depth_mm": D.tolist(), "F1_N_per_MPa": F1.tolist()},
              "grip_fixed": {"depth_mm": fixed[0].round(4).tolist(), "F1_N_per_MPa": fixed[1].tolist()},
              "grip_moving": {"depth_mm": moving[0].round(4).tolist(), "F1_N_per_MPa": moving[1].tolist()}}
    F1_ref = float(np.interp(a.ref_depth, D, F1))
    params = {"size_mm": a.size, "pad_mm": a.pad, "max_depth_mm": a.max_depth, "mass_kg": a.mass,
              "free": not a.fixed, "fixed_jaw_face": "xneg" if a.fixed_jaw == "neg" else "xpos",
              "curves": curves,
              "grip_total": {"closure_mm": total[0].tolist(), "F1_N_per_MPa": total[1].tolist()},
              "levels": {}}
    for lvl, F_target in zip(LEVELS, a.targets):
        E = F_target / F1_ref
        k_lin = {"top": F_target / (a.ref_depth * 1e-3)}
        for face, cname in side.items():
            d, f = curves[cname]["depth_mm"], curves[cname]["F1_N_per_MPa"]
            k_lin[face] = E * f[-1] / (d[-1] * 1e-3)
        name = f"cube_{lvl}"
        with open(os.path.join(a.out, f"{name}.xml"), "w") as fh:
            fh.write(cube_xml(name, a.size, a.pad, a.pos, k_lin, COLORS[lvl], a.max_depth, a.mass, not a.fixed, a.solref))
        params["levels"][lvl] = {
            "body": name, "E_MPa": round(E, 5), "joint": f"{name}_top",
            "pads": {"top": {"joint": f"{name}_top", "curve": "press"},
                     **{face: {"joint": f"{name}_{face}", "curve": c} for face, c in side.items()}},
            "press_F_at_1_2_3mm_N": [round(E * float(np.interp(d, D, F1)), 3) for d in (1, 2, 3)],
            "grip_F_at_1_2_3mm_closure_N": [round(E * float(np.interp(c, *total)), 3) for c in (1, 2, 3)]}
        print(f"{lvl:6s}  E = {E:.4f} MPa   press {params['levels'][lvl]['press_F_at_1_2_3mm_N']} N"
              f"   grip {params['levels'][lvl]['grip_F_at_1_2_3mm_closure_N']} N")
    json.dump(params, open(os.path.join(a.out, "cube_params.json"), "w"), indent=1)
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
