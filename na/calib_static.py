"""
NeuralActuator 입력용 시뮬레이션 텔레메트리 보정 (cad2026 환경, mujoco만 사용, 2026-11-17)
사용법: python na/calib_static.py

1) 관절 각도 기준 확인: 같은 q에서 우리 Menagerie SO-101과 NeuralActuator 저장소의 so101_new_calib.xml의
   그리퍼 위치가 같은지 (다르면 각도 오프셋/부호 변환이 필요)
2) 서보 부하(load, STS3215 부호 있는 카운트) <-> 토크 관계: 실측 CSV(빈손, 거의 정지한 구간)에서
   load_j ≈ k_j * (중력 토크_j, 우리 모델 qfrc_bias) 로 관절별 k_j(부호 포함)를 최소제곱으로 구함
   (STS3215 load 레지스터는 최대 토크 대비 0.1% 단위로 알려져 있음 -> 2.94N·m 기준이면 약 340 카운트/N·m 예상)
3) 전압/온도 평균 (시뮬레이션에서는 고정값으로 넣음)
결과: config/na_telemetry.json
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

HERE = ROOT
NA = os.path.join(HERE, "external", "Dynamics-Modeling", "NeuralActuator")
OURS = os.path.join(HERE, "models", "mujoco_menagerie", "robotstudio_so101", "so101.xml")
THEIRS = os.path.join(NA, "robot_so101", "so101_new_calib.xml")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    mo, mt = mujoco.MjModel.from_xml_path(OURS), mujoco.MjModel.from_xml_path(THEIRS)
    do, dt = mujoco.MjData(mo), mujoco.MjData(mt)
    qo = [mo.joint(j).qposadr[0] for j in JOINTS]
    qt = [mt.joint(j).qposadr[0] for j in JOINTS]
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(200):
        q = rng.uniform(-1.2, 1.2, 6)
        do.qpos[qo] = q; dt.qpos[qt] = q
        mujoco.mj_kinematics(mo, do); mujoco.mj_kinematics(mt, dt)
        errs.append(np.linalg.norm(do.body("gripper").xpos - dt.body("gripper").xpos))
    print(f"1) 같은 q에서 그리퍼 바디 위치 차이: 최대 {max(errs)*1e3:.3f} mm (0이면 각도 기준 동일)")

    # cad2026에는 pandas가 없음 (환경에 패키지를 늘리지 않으려고 numpy로 읽음)
    # 페이로드(빈손~500g)를 그리퍼 끝(gripperframe)에 단 질점으로 보고 중력 토크에 더함 -> 토크 범위를 넓혀 맞춤
    arrs, masses = [], []
    for f in glob.glob(os.path.join(NA, "data", "so101", "*", "*", "*", "*.csv")):
        p = os.path.basename(os.path.dirname(os.path.dirname(f)))
        a = np.genfromtxt(f, delimiter=",", names=True)
        arrs.append(a)
        masses.append(np.full(len(a), 0.0 if p == "empty" else float(p.rstrip("g")) / 1000))
    D, mass = np.concatenate(arrs), np.concatenate(masses)

    def cols(name):
        return np.c_[tuple(D[f"{name}{i}"] for i in range(1, 7))].astype(float)

    pos, vel, load = cols("pos"), cols("vel"), cols("load")
    still = np.abs(vel).max(1) < 30  # 엔코더 스텝/s, 거의 정지
    print(f"2) 데이터 {len(D)}행 (페이로드 0~500g) 중 거의 정지 {still.sum()}행")
    sid = mo.site("gripperframe").id
    dofs = [mo.joint(j).dofadr[0] for j in JOINTS]
    Jp = np.zeros((3, mo.nv))
    tau = np.zeros((still.sum(), 6))
    for k, (q, mp) in enumerate(zip(pos[still], mass[still])):
        do.qpos[:] = 0; do.qpos[qo] = q; do.qvel[:] = 0
        mujoco.mj_forward(mo, do)
        mujoco.mj_jacSite(mo, do, Jp, None, sid)
        # 페이로드 무게 (0,0,-m g)를 버티는 데 필요한 관절 토크 = -J^T F = J^T (0,0,m g)
        tau[k] = do.qfrc_bias[dofs] + (Jp.T @ np.array([0, 0, mp * 9.81]))[dofs]
    L = load[still]
    out = {}
    for j in range(5):
        x, y = tau[:, j], L[:, j]
        if np.std(x) < 1e-3:
            print(f"   {JOINTS[j]:13s}: 중력 토크 변화가 거의 없어 맞출 수 없음 (std {np.std(x):.4f} N·m)")
            out[JOINTS[j]] = None
            continue
        A = np.c_[x, np.ones_like(x)]
        (k, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        r = np.corrcoef(x, y)[0, 1]
        print(f"   {JOINTS[j]:13s}: load ≈ {k:8.1f} * tau {b:+7.1f}  (상관 {r:+.2f}, tau 범위 {x.min():+.2f}~{x.max():+.2f} N·m, "
              f"load 범위 {y.min():+.0f}~{y.max():+.0f})")
        out[JOINTS[j]] = {"k_counts_per_Nm": float(k), "offset": float(b), "corr": float(r)}
    volts = cols("volts").mean(0)
    temps = cols("temp").mean(0)
    print(f"3) 평균 전압(0.1V) {np.round(volts, 1)}, 평균 온도(°C) {np.round(temps, 1)}")
    os.makedirs(os.path.join(HERE, "config"), exist_ok=True)
    json.dump({"kinematics_max_diff_mm": float(max(errs) * 1e3), "load_fit": out,
               "volts_mean": volts.round(2).tolist(), "temp_mean": temps.round(2).tolist(),
               "vel_counts_per_rad": 4096.0 / (2 * np.pi), "data_dt": 0.01605,
               "note": "load_fit: 빈손 정지 구간에서 load ≈ k * (우리 Menagerie 모델 중력 토크) + offset"},
              open(os.path.join(HERE, "config", "na_telemetry.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("저장: config/na_telemetry.json")


if __name__ == "__main__":
    main()
