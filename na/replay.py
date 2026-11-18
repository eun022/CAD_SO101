"""
실측 NeuralActuator SO-101 CSV의 목표 관절 궤적을 우리 MuJoCo에서 재생 -> 시뮬레이션 텔레메트리 CSV 생성 (cad2026, 2026-11-17)
사용법: python na/replay.py <실측 csv> <출력 csv> [페이로드 kg]

- goal_pos를 0.01605 s마다 위치 액추에이터 목표로 (사이에는 유지), 시뮬레이션 1kHz
- 페이로드: 그리퍼 끝(gripperframe)에 (0, 0, -m g) 외력으로 흉내 (그리퍼로 잡은 물체 무게)
- 출력 CSV는 실측과 같은 열 이름 -> neural_actuator 환경에서 infer_actuator.py --force_only로 바로 추론 가능
  (force_x/y/z는 실측 CSV 값을 그대로 복사해 MAE 비교용으로 둠)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import os
import sys

import mujoco
import numpy as np

from na.telemetry import DATA_DT, SimTelemetry

HERE = ROOT
SCENE = os.path.join(HERE, "models", "so101_scene_1khz.xml")


def main():
    src, dst = sys.argv[1], sys.argv[2]
    payload = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    R = np.genfromtxt(src, delimiter=",", names=True)
    goal = np.c_[tuple(R[f"goal_pos{i}"] for i in range(1, 7))]
    pos0 = np.c_[tuple(R[f"pos{i}"] for i in range(1, 7))][0]
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    tel = SimTelemetry(m)
    d.qpos[tel.qadr] = pos0
    d.ctrl[tel.aid] = goal[0]
    mujoco.mj_forward(m, d)
    sid, bid = m.site("gripperframe").id, m.body("gripper").id
    F = np.array([0, 0, -payload * 9.81])
    rows, t = [], 0.0
    for i in range(len(R)):
        d.ctrl[tel.aid] = goal[i]
        while d.time < t + DATA_DT - 1e-9:
            d.xfrc_applied[:] = 0
            if payload > 0:  # 그리퍼 바디에 외력 + 그 힘이 gripperframe에 걸린 만큼의 모멘트
                d.xfrc_applied[bid, :3] = F
                d.xfrc_applied[bid, 3:] = np.cross(d.site_xpos[sid] - d.xipos[bid], F)
            mujoco.mj_step(m, d)
            tel.accumulate(d)
        t += DATA_DT
        f = tel.features(d)
        rows.append([t] + list(f[6:12]) + list(f[0:6]) + [0] * 6 + list(f[18:24]) + list(f[12:18])
                    + list(f[30:36]) + list(f[24:30]) + [R["force_x"][i], R["force_y"][i], R["force_z"][i]])
    head = (["timestamp"] + [f"pos{i}" for i in range(1, 7)] + [f"goal_pos{i}" for i in range(1, 7)]
            + [f"current{i}" for i in range(1, 7)] + [f"vel{i}" for i in range(1, 7)] + [f"load{i}" for i in range(1, 7)]
            + [f"temp{i}" for i in range(1, 7)] + [f"volts{i}" for i in range(1, 7)] + ["force_x", "force_y", "force_z"])
    np.savetxt(dst, np.array(rows), delimiter=",", header=",".join(head), comments="", fmt="%.6g")
    # 실측과 시뮬레이션 텔레메트리 비교 (관절별)
    real_load = np.c_[tuple(R[f"load{i}"] for i in range(1, 7))]
    real_pos = np.c_[tuple(R[f"pos{i}"] for i in range(1, 7))]
    sim = np.array(rows)
    print(f"{os.path.basename(src)} 페이로드 {payload*1000:.0f}g, {len(R)}샘플")
    print("  위치 오차 실측-시뮬 (deg, 관절별 평균):", np.round(np.degrees(np.abs(real_pos - sim[:, 1:7]).mean(0)), 2))
    print("  load 실측 평균:", np.round(real_load.mean(0)), " 시뮬 평균:", np.round(sim[:, 25:31].mean(0)))
    print("  load 상관(실측 vs 시뮬):", np.round([np.corrcoef(real_load[:, j], sim[:, 25 + j])[0, 1] if real_load[:, j].std() > 0 else np.nan
                                            for j in range(6)], 2))


if __name__ == "__main__":
    main()
