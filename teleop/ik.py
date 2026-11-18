"""
SO-101 그리퍼 끝 도달 범위 계산 (시뮬레이션만)
사용법: python teleop/ik.py [down]
- 기본: 위치 전용 IK (PosIK, 끝 위치 3개만 맞추고 그리퍼 각도는 자유, 아래를 향하는 자세를 약하게 선호) - (4) 텔레오퍼레이션용
- down: 그리퍼가 아래를 향하게 고정 (so101_press_demo.IK, 손목 각도 합 고정) - 범위가 매우 좁음
- 격자 점마다 IK 오차 < 1mm이고 관절 한계에서 0.02rad 이상 떨어져 있으면 도달 가능
- 블록 위치(x=0.25, y=0) 주변으로 축 정렬 작업 상자 후보를 찾아 출력
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import itertools
import sys

import mujoco
import numpy as np

from teleop import press_demo as p

XS = np.arange(0.10, 0.42, 0.02)
YS = np.arange(-0.24, 0.25, 0.02)
ZS = np.arange(0.04, 0.26, 0.02)


class PosIK:
    """위치 전용 IK: 관절 4개(pan, lift, elbow, wrist_flex)로 끝 위치 3개를 맞춤 (1차 목표).
    남는 자유도 1개(null-space)로만 2차 목표를 따름: 손목 각도 합을 PITCH_SUM(아래를 향함) 쪽으로,
    관절은 범위 중앙 쪽으로 약하게 끌어당겨 한계에 붙지 않게 함"""

    K_PITCH, K_MID = 0.3, 0.1

    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.J = np.zeros((3, m.nv))
        self.d.qpos[:6] = [0, 0, 0.5, 1.0, 0, 0]  # 로봇 관절 6개 (장면에 자유 물체가 있으면 qpos가 더 김)
        self.pitch_target = p.PITCH_SUM  # 2차 목표: lift+elbow+wrist_flex (텔레오퍼레이션에서 스타일러스 숙임으로 바꿈)

    def solve(self, target, iters=20):
        m, d = self.m, self.d
        lo, hi = m.jnt_range[:4, 0], m.jnt_range[:4, 1]
        mid = (lo + hi) / 2
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            q = d.qpos[:4]
            err = target - d.site_xpos[self.sid]
            mujoco.mj_jacSite(m, d, self.J, None, self.sid)
            J = self.J[:, :4]
            q2 = self.K_PITCH * (self.pitch_target - q[1:].sum()) * np.array([0, 1, 1, 1]) + self.K_MID * (mid - q)
            # 한계에 닿은 관절이 바깥쪽으로 가려 하면 그 관절을 빼고 다시 풂 (안 그러면 한계에 붙어 수렴하지 못함)
            free = np.ones(4, bool)
            for _ in range(4):
                Jf = J[:, free]
                Jp = Jf.T @ np.linalg.inv(Jf @ Jf.T + 1e-5 * np.eye(3))
                dq = np.zeros(4)
                dq[free] = Jp @ err + (np.eye(free.sum()) - Jp @ Jf) @ q2[free]
                stuck = free & (((q >= hi - 1e-6) & (dq > 0)) | ((q <= lo + 1e-6) & (dq < 0)))
                if not stuck.any():
                    break
                free &= ~stuck
            d.qpos[:4] = np.clip(q + 0.5 * dq, lo, hi)
        return d.qpos.copy()

    def solve_pose(self, target, iters=20, w_pitch=0.02, max_step=0.15, damping=1e-4, margin=0.0):
        """끝 위치 3개 + 손목 각도 합(pitch_target) 1개를 관절 4개로 동시에 맞춤 (2026-11-17, 텔레오퍼레이션용).
        solve()와 달리 각도가 보조 목표가 아니라 위치와 같은 1차 목표 -> 도달 가능하면 둘 다 정확히 맞음.
        도달 불가(관절 한계)면 가중 최소제곱으로 타협: w_pitch [m/rad] = 각도 1rad 오차를 위치 몇 m 오차로 칠지"""
        m, d = self.m, self.d
        lo, hi = m.jnt_range[:4, 0] + margin, m.jnt_range[:4, 1] - margin  # margin: 관절 한계 여유 (명령을 자를 범위와 맞춤)
        row = np.array([[0.0, 1, 1, 1]]) * w_pitch
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            q = d.qpos[:4]
            err = np.r_[target - d.site_xpos[self.sid], w_pitch * (self.pitch_target - q[1:].sum())]
            mujoco.mj_jacSite(m, d, self.J, None, self.sid)
            J = np.vstack([self.J[:, :4], row])
            free = np.ones(4, bool)
            for _ in range(4):  # 한계에 붙은 관절이 바깥으로 가려 하면 빼고 다시 풂
                Jf = J[:, free]
                dq = np.zeros(4)
                dq[free] = np.linalg.solve(Jf.T @ Jf + damping * np.eye(free.sum()), Jf.T @ err)
                stuck = free & (((q >= hi - 1e-6) & (dq > 0)) | ((q <= lo + 1e-6) & (dq < 0)))
                if not stuck.any():
                    break
                free &= ~stuck
            s = np.abs(dq).max()
            if s > max_step:  # 한 번에 크게 뛰면 발산 (처음 버전에서 확인) -> 관절 이동량 제한
                dq *= max_step / s
            d.qpos[:4] = np.clip(q + dq, lo, hi)
        return d.qpos.copy()


def reachable(m, ik, target):
    q = ik.solve(target, iters=60)
    d = ik.d
    mujoco.mj_kinematics(m, d)
    err = np.linalg.norm(d.site_xpos[ik.sid] - target)
    lo, hi = m.jnt_range[:4, 0], m.jnt_range[:4, 1]
    margin = min((q[:4] - lo).min(), (hi - q[:4]).min())
    return err < 1e-3 and margin > 0.02


def reach_grid(m, ik):
    """XS x YS x ZS 격자점마다 도달 가능 여부 (bool 배열)"""
    ok = np.zeros((len(XS), len(YS), len(ZS)), bool)
    for (i, x), (j, y), (k, z) in itertools.product(enumerate(XS), enumerate(YS), enumerate(ZS)):
        ik.d.qpos[:6] = [0, 0, 0.5, 1.0, 0, 0]
        ok[i, j, k] = reachable(m, ik, np.array([x, y, z]))
    return ok


def main():
    m = mujoco.MjModel.from_xml_path(p.SCENE)
    down = len(sys.argv) > 1 and sys.argv[1] == "down"
    ik = p.IK(m) if down else PosIK(m)
    print("IK:", "그리퍼 아래 고정" if down else "위치 전용")
    ok = reach_grid(m, ik)
    print(f"도달 가능 점 {ok.sum()} / {ok.size}")
    for k, z in enumerate(ZS):
        xs = [XS[i] for i in range(len(XS)) if ok[i, len(YS) // 2, k]]
        ys = [YS[j] for j in range(len(YS)) if ok[np.argmin(abs(XS - 0.25)), j, k]]
        print(f"  z={z:.2f}: y=0에서 x {min(xs) if xs else '-'}~{max(xs) if xs else '-'} / x=0.25에서 y "
              f"{min(ys) if ys else '-'}~{max(ys) if ys else '-'}")

    # 블록(0.25, 0) 중심, z는 아래로 블록 윗면 근처까지 포함하는 가장 큰 상자 (모든 격자점 도달 가능)
    best = None
    for hx in np.arange(0.02, 0.16, 0.02):
        for hy in np.arange(0.02, 0.26, 0.02):
            for z0, z1 in [(a, b) for a in ZS for b in ZS if b > a and a <= 0.06]:
                sel = ((np.abs(XS - 0.25) <= hx + 1e-9)[:, None, None] & (np.abs(YS) <= hy + 1e-9)[None, :, None]
                       & ((ZS >= z0 - 1e-9) & (ZS <= z1 + 1e-9))[None, None, :])
                if sel.any() and ok[sel].all():
                    vol = (2 * hx) * (2 * hy) * (z1 - z0)
                    if best is None or vol > best[0]:
                        best = (vol, hx, hy, z0, z1)
    vol, hx, hy, z0, z1 = best
    print(f"\n작업 상자 후보 (전 격자점 도달 가능): x {0.25 - hx:.2f}~{0.25 + hx:.2f}, y {-hy:.2f}~{hy:.2f}, z {z0:.2f}~{z1:.2f} m"
          f"  (크기 {2 * hx * 100:.0f} x {2 * hy * 100:.0f} x {(z1 - z0) * 100:.0f} cm)")


if __name__ == "__main__":
    main()
