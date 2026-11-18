"""
(2a) 위->아래 누르기 시나리오 (MuJoCo, 1kHz, 뷰어로 보기)
사용법: python teleop/press_demo.py

- 장면: models/so101_press_scene.xml (SO-101 + 고정 블록, 윗면 z=0.05m)
- 그리퍼 끝(gripperframe)을 블록 위에서 아래로: 접근 -> 누르기(목표를 윗면보다 PRESS_DEPTH 아래로) -> 유지 -> 올라가기
- 관절 목표는 간단한 IK로 계산 (위치 3개 + 손목 각도 합 1개, 그리퍼가 아래를 향하게), 위치 액추에이터로 추종
- 매 스텝(1ms) 그리퍼-블록 접촉력(월드 좌표, 블록이 받는 힘)을 기록 -> data/press_<시각>.csv
- 뷰어: 접촉점과 접촉력 화살표 표시. 시나리오를 반복 재생, 창을 닫으면 종료
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

HERE = ROOT
SCENE = os.path.join(HERE, "models", "so101_press_scene.xml")
DATA_DIR = os.path.join(HERE, "data")

BLOCK_TOP_Z = 0.05
XY = np.array([0.25, 0.0])
Z_START = 0.10  # 시작 높이
# 누를 때 끝 목표 높이. 턱 모서리가 먼저 닿아서 끝 높이 약 0.0593m에서 접촉이 시작되고,
# 위치 액추에이터 강성이 커서 목표를 접촉 높이보다 조금만 낮춰도 힘이 크게 오름 (0.1mm당 약 2.8N, 0.040이면 토크 한계로 약 35N)
# 헤드리스 탐색 결과 0.05913 -> 유지 구간 평균 |F| 약 4.0N (Fz -3.7, Fx +1.4), 접촉 시작 순간 최대 약 9N
PRESS_Z = 0.05913
PITCH_SUM = 1.45  # shoulder_lift + elbow_flex + wrist_flex (그리퍼가 아래를 향하는 값, IK 탐색으로 정함)


def phases(press_z=None):
    """구간: (끝 시각 s, 목표 z) - 구간 안에서 선형 보간"""
    pz = PRESS_Z if press_z is None else press_z
    return [(1.0, Z_START), (3.0, pz), (4.0, pz), (6.0, Z_START), (7.0, Z_START)]


PHASES = phases()


def target_z(t, ph=None):
    t0, z0 = 0.0, Z_START
    for t1, z1 in ph or PHASES:
        if t <= t1:
            return z0 + (z1 - z0) * (t - t0) / (t1 - t0)
        t0, z0 = t1, z1
    return Z_START


class IK:
    """위치 3개 + 손목 각도 합 1개를 맞추는 감쇠 최소제곱 IK (별도 MjData로 계산, 접촉 영향 없음)"""

    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.J = np.zeros((3, m.nv))
        self.d.qpos[:6] = [0, 0, 0.5, 1.0, 0, 0]  # 로봇 관절 6개

    def solve(self, target, iters=20):
        m, d = self.m, self.d
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = np.r_[target - d.site_xpos[self.sid], PITCH_SUM - d.qpos[1:4].sum()]
            mujoco.mj_jacSite(m, d, self.J, None, self.sid)
            A = np.vstack([self.J[:, :4], [0, 1, 1, 1]])
            d.qpos[:4] += 0.5 * (A.T @ np.linalg.solve(A @ A.T + 1e-4 * np.eye(4), err))
            d.qpos[:] = np.clip(d.qpos, m.jnt_range[:, 0], m.jnt_range[:, 1])
        return d.qpos.copy()


def block_force(m, d, block_gid, buf):
    """블록과 닿은 모든 접촉의 힘 합 (월드 좌표, 블록이 받는 힘)"""
    total = np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        if block_gid not in (c.geom1, c.geom2):
            continue
        mujoco.mj_contactForce(m, d, i, buf)
        f = c.frame.reshape(3, 3).T @ buf[:3]  # 접촉 좌표 -> 월드. 법선은 geom1 -> geom2, f는 geom2가 받는 힘
        total += f if c.geom2 == block_gid else -f
    return total


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    block_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "block")
    ik, buf = IK(m), np.zeros(6)
    cycle_t = PHASES[-1][0]

    d.qpos[:] = ik.solve(np.r_[XY, Z_START], iters=200)
    d.ctrl[:] = d.qpos[:6]
    mujoco.mj_forward(m, d)

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"press_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    rows, recorded = [], False

    with mujoco.viewer.launch_passive(m, d) as v:
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        wall0, sim0, n = time.perf_counter(), d.time, 0
        while v.is_running():
            t = (d.time - sim0) % cycle_t
            if n % 10 == 0:  # IK는 100Hz로 갱신
                d.ctrl[:] = ik.solve(np.r_[XY, target_z(t)])
            mujoco.mj_step(m, d)
            n += 1
            if not recorded:
                f = block_force(m, d, block_gid, buf)
                tip = d.site_xpos[ik.sid]
                rows.append((d.time - sim0, target_z(t), *tip, *f))
                if d.time - sim0 >= cycle_t:
                    recorded = True
                    with open(path, "w", newline="") as fp:
                        w = csv.writer(fp)
                        w.writerow(["t", "target_z", "tip_x", "tip_y", "tip_z", "Fx", "Fy", "Fz"])
                        w.writerows([[f"{x:.6f}" for x in r] for r in rows])
                    fz = np.array([r[7] for r in rows])
                    on = np.flatnonzero(np.abs(fz) > 0.05)
                    print(f"1회차 기록 저장: {path}")
                    print(f"  최대 |Fz| = {np.abs(fz).max():.2f} N, 최대 누름 시 끝 높이 = {min(r[4] for r in rows):.4f} m")
                    if on.size:
                        print(f"  접촉 시작 t={rows[on[0]][0]:.3f}s, 해제 t={rows[on[-1]][0]:.3f}s")
            # 실시간 속도로 재생, 화면은 약 60Hz로 갱신
            if n % 16 == 0:
                v.sync()
                lag = (d.time - sim0) - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)


if __name__ == "__main__":
    main()
