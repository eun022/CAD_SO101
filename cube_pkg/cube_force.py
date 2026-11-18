"""Push each cube pad back with the FEM force. Call apply(data) before every mj_step."""
import json

import mujoco
import numpy as np


class CubeForces:
    def __init__(self, model, params_path):
        p = json.load(open(params_path))
        self.curves = {k: (np.array(c["depth_mm"]), np.array(c["F1_N_per_MPa"])) for k, c in p["curves"].items()}
        self.cubes = {}
        self._weight = {}
        self._g = model.opt.gravity.copy()
        for lvl, c in p["levels"].items():
            pads = {}
            for face, pad in c["pads"].items():
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pad["joint"])
                if jid < 0:
                    continue
                model.jnt_stiffness[jid] = 0.0
                bid = model.jnt_bodyid[jid]
                pads[face] = (model.jnt_qposadr[jid], model.jnt_dofadr[jid], bid,
                              model.jnt_axis[jid].copy(), pad["curve"])
                self._weight[model.jnt_dofadr[jid]] = (bid, model.jnt_axis[jid].copy(), model.body_mass[bid])
            if pads:
                self.cubes[lvl] = (c["E_MPa"], pads)
        if not self.cubes:
            raise ValueError("no cube joints from cube_params.json found in the model")

    def _force(self, depth_mm, E, curve):
        D, F1 = self.curves[curve]
        d = max(depth_mm, 0.0)
        if d <= D[-1]:
            return E * float(np.interp(d, D, F1))
        slope = (F1[-1] - F1[-2]) / (D[-1] - D[-2])
        return E * float(F1[-1] + slope * (d - D[-1]))

    def apply(self, data):
        for E, pads in self.cubes.values():
            for qadr, dadr, bid, axis, curve in pads.values():
                # cancel the pad's own weight along the slide
                own = self._weight[dadr][2] * float(self._g @ (data.xmat[bid].reshape(3, 3) @ axis))
                data.qfrc_applied[dadr] = self._force(-data.qpos[qadr] * 1e3, E, curve) - own

    def faces(self, data):
        out = {}
        for lvl, (E, pads) in self.cubes.items():
            out[lvl] = {}
            for face, (qadr, _, _, _, curve) in pads.items():
                d = float(-data.qpos[qadr] * 1e3)
                out[lvl][face] = (d, self._force(d, E, curve))
        return out

    def force(self, data):
        return {lvl: f["top"] for lvl, f in self.faces(data).items() if "top" in f}

    def force_world(self, data):
        out = {}
        for lvl, (E, pads) in self.cubes.items():
            out[lvl] = {}
            for face, (qadr, _, bid, axis, curve) in pads.items():
                n = data.xmat[bid].reshape(3, 3) @ axis
                out[lvl][face] = self._force(-data.qpos[qadr] * 1e3, E, curve) * n
        return out

    def grip(self, data):
        out = {}
        for lvl, f in self.faces(data).items():
            if "xneg" in f and "xpos" in f:
                out[lvl] = (float(max(f["xneg"][0], 0.0) + max(f["xpos"][0], 0.0)), 0.5 * (f["xneg"][1] + f["xpos"][1]))
        return out
