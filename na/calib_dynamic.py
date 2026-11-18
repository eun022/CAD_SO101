"""
서보 load <-> 시뮬레이션 액추에이터 토크 관계를 '실측 궤적 재생'으로 맞춤 (cad2026, 2026-11-17)
사용법: python na/calib_dynamic.py

na/calib_static.py(정지 구간 + 중력 토크)는 정지 시 서보 부하가 마찰/히스테리시스에 좌우돼 부호가 거꾸로 나옴
(재생 결과 load 상관 -0.85~-0.33). 여기서는 train 폴더 실측 CSV의 goal_pos를 MuJoCo에서 재생하며
시뮬레이션 액추에이터 토크 tau를 기록하고, 실측 load ≈ k * tau + b 를 관절별로 최소제곱 (움직이는 구간 포함)
결과는 config/na_telemetry.json의 load_fit을 덮어씀 (이전 값은 load_fit_static으로 보관)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import glob
import json
import os

import mujoco
import numpy as np

from na.telemetry import DATA_DT, JOINTS

HERE = ROOT
NA = os.path.join(HERE, "external", "Dynamics-Modeling", "NeuralActuator")
SCENE = os.path.join(HERE, "models", "so101_scene_1khz.xml")
CFG = os.path.join(HERE, "config", "na_telemetry.json")


def replay(m, path, payload):
    R = np.genfromtxt(path, delimiter=",", names=True)
    goal = np.c_[tuple(R[f"goal_pos{i}"] for i in range(1, 7))]
    d = mujoco.MjData(m)
    qadr = [m.joint(j).qposadr[0] for j in JOINTS]
    aid = [m.actuator(j).id for j in JOINTS]
    d.qpos[qadr] = np.c_[tuple(R[f"pos{i}"] for i in range(1, 7))][0]
    sid, bid = m.site("gripperframe").id, m.body("gripper").id
    F = np.array([0, 0, -payload * 9.81])
    dadr = [m.joint(j).dofadr[0] for j in JOINTS]
    taus, vels, t = [], [], 0.0
    for i in range(len(R)):
        d.ctrl[aid] = goal[i]
        acc, n = np.zeros(6), 0
        while d.time < t + DATA_DT - 1e-9:
            if payload > 0:
                d.xfrc_applied[bid, :3] = F
                d.xfrc_applied[bid, 3:] = np.cross(d.site_xpos[sid] - d.xipos[bid], F)
            mujoco.mj_step(m, d)
            acc += d.actuator_force[aid]; n += 1
        t += DATA_DT
        taus.append(acc / n)  # 샘플 구간 평균 토크 (순간값은 ±2.94 한계에 자주 붙음)
        vels.append(d.qvel[dadr].copy())
    return (np.array(taus), np.array(vels),
            np.c_[tuple(R[f"load{i}"] for i in range(1, 7))].astype(float))


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    T, V, L = [], [], []
    files = sorted(glob.glob(os.path.join(NA, "data", "so101", "*", "*", "train", "*.csv")))
    for f in files[::3]:  # 1/3만 사용 (시간 절약)
        p = os.path.basename(os.path.dirname(os.path.dirname(f)))
        tau, vel, load = replay(m, f, 0.0 if p == "empty" else float(p.rstrip("g")) / 1000)
        T.append(tau); V.append(vel); L.append(load)
    T, V, L = np.concatenate(T), np.concatenate(V), np.concatenate(L)
    print(f"{len(files[::3])}개 파일, {len(T)}샘플")
    cfg = json.load(open(CFG, encoding="utf-8"))
    cfg.setdefault("load_fit_static", cfg["load_fit"])
    fit = {}
    for j, name in enumerate(JOINTS):
        x, v, y = T[:, j], V[:, j], L[:, j]
        r1 = np.corrcoef(x, y)[0, 1]
        A = np.c_[x, v, np.ones_like(x)]  # load ≈ k*tau + c*vel + b (Feetech load는 PWM 듀티에 가까움)
        (k, c, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = np.corrcoef(A @ np.array([k, c, b]), y)[0, 1]
        print(f"  {name:13s}: load ≈ {k:7.1f}*tau {c:+7.1f}*vel {b:+7.1f}  상관 토크만 {r1:+.2f} / 토크+속도 {r2:+.2f}  "
              f"(평균 tau {x.min():+.2f}~{x.max():+.2f} N·m)")
        fit[name] = {"k_counts_per_Nm": float(k), "c_counts_per_radps": float(c), "offset": float(b),
                     "corr_tau_only": float(r1), "corr": float(r2)}
    cfg["load_fit"] = fit
    cfg["load_fit_note"] = "실측 train CSV 재생(1/3)으로 맞춤: 실측 load ≈ k * 시뮬레이션 actuator_force + offset (2026-11-17)"
    json.dump(cfg, open(CFG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("저장: config/na_telemetry.json (load_fit 갱신, 이전 값은 load_fit_static)")


if __name__ == "__main__":
    main()
