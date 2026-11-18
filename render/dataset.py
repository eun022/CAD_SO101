"""
(3) 렌더링 비교용 데이터 생성 (cad2026, 헤드리스, 2026-11-18)

MuJoCo에서 SO-101이 (그리퍼를 닫은 채) 블록 위 FEM 큐브 윗면을 정해진 방식으로 누름 -> 1kHz로 기록
  - hand: 손(=로봇 목표) 위치, 로봇 좌표 m. 실제 텔레오퍼레이션에서는 Minverse 위치를 배율로 옮긴 값에 해당
  - tip: 로봇 기준점(고정 턱 끝) 실제 위치
  - F: 정답 접촉력 = 팔 전체가 받는 접촉력 합 (로봇 좌표 N, teleop_force.gripper_contact_force)
  - top_mm, top_N: 큐브 윗면 패드 눌림/FEM 힘
시나리오 x 큐브 강도(soft/medium/hard) 조합마다 data/render/<level>_<scenario>.npz

사용법: python render/dataset.py            # 전부
        python render/dataset.py medium tap  # 하나만
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

from teleop import run as T
from teleop.ik import PosIK
from teleop.force import gripper_contact_force

HERE = ROOT
OUT = os.path.join(HERE, "data", "render")
LEVELS = ("soft", "medium", "hard")
XY = np.array([0.25, 0.0])  # 큐브 중심 (블록 위)
Z_TOP = 0.080  # 큐브 윗면 패드 표면 높이 (블록 윗면 0.05 + 큐브 30mm)
Z_ABOVE = Z_TOP + 0.015
SETTLE_S = 1.0
DMAX = 3.2e-3  # 최대 누름 목표 깊이 (손 기준. 로봇이 살짝 밀려 패드는 약 2.3mm, 한계 3mm 안쪽)


def cos_ramp(t, t0, t1, a, b):
    """t0~t1 사이에 a -> b 로 매끄럽게 (코사인)"""
    u = np.clip((t - t0) / (t1 - t0), 0, 1)
    return a + (b - a) * (1 - np.cos(np.pi * u)) / 2


def press(v, hold=1.0, depth=DMAX):
    """위에서 속도 v(m/s)로 내려와 depth까지 누르고 hold초 유지 후 같은 속도로 올라감"""
    dz = Z_ABOVE - (Z_TOP - depth)
    tr = dz / v * 1.5  # 코사인 램프라 평균 속도 v가 되도록 (최고 속도 약 1.6v)
    t_end = 0.3 + tr + hold + tr + 0.5

    def z(t):
        if t < 0.3 + tr:
            return cos_ramp(t, 0.3, 0.3 + tr, Z_ABOVE, Z_TOP - depth)
        return cos_ramp(t, 0.3 + tr + hold, 0.3 + 2 * tr + hold, Z_TOP - depth, Z_ABOVE)
    return z, t_end


def sine(freq, cycles=3):
    """윗면에 닿아 깊이 1~3mm 사이를 freq Hz로 오르내림 -> 마지막에 뗌"""
    t_in, t_s = 1.2, 1.2 + 0.5
    t_end = t_s + cycles / freq + 1.2

    def z(t):
        if t < t_in:
            return cos_ramp(t, 0.2, t_in, Z_ABOVE, Z_TOP - 2.0e-3)
        if t < t_s:
            return Z_TOP - 2.0e-3
        if t < t_s + cycles / freq:
            return Z_TOP - 2.0e-3 - 1.0e-3 * np.sin(2 * np.pi * freq * (t - t_s))
        return cos_ramp(t, t_s + cycles / freq, t_s + cycles / freq + 0.6, Z_TOP - 2.0e-3, Z_ABOVE)
    return z, t_end


def taps(n=3, v=0.1, depth=3.0e-3, hold=0.15, gap=0.5):
    """빠르게 톡톡 두드리기"""
    dz = Z_ABOVE - (Z_TOP - depth)
    tr = dz / v * 1.5
    per = 2 * tr + hold + gap
    t_end = 0.3 + n * per + 0.3

    def z(t):
        k = int((t - 0.3) // per) if t > 0.3 else 0
        tt = t - 0.3 - k * per
        if k >= n or t < 0.3:
            return Z_ABOVE
        if tt < tr:
            return cos_ramp(tt, 0, tr, Z_ABOVE, Z_TOP - depth)
        return cos_ramp(tt, tr + hold, 2 * tr + hold, Z_TOP - depth, Z_ABOVE)
    return z, t_end


SCENARIOS = {
    "press_slow": press(0.01),   # 1 cm/s
    "press_fast": press(0.05),   # 5 cm/s
    "sine_1hz": sine(1.0),
    "sine_3hz": sine(3.0),
    "tap": taps(),               # 10 cm/s
}


def run(level, name):
    zfun, t_end = SCENARIOS[name]
    m = T.load_scene(level)
    sid = m.site("gripperframe").id
    m.site_pos[sid] = T.FIXED_JAW_TIP  # 텔레오퍼레이션과 같은 기준점 (고정 턱 끝)
    pads = [m.joint(f"cube_{level}_{f}") for f in ("top", "xneg", "xpos")]
    for j in pads:  # 텔레오퍼레이션과 같은 패드 한계 보정
        m.jnt_solref[j.id] = T.PAD_LIMIT_SOLREF
        m.jnt_solimp[j.id, :3] = T.PAD_LIMIT_SOLIMP
    pad_adr = [(j.qposadr[0], j.dofadr[0]) for j in pads]
    cubes = T.CubeForces(m, T.CUBE_PARAMS)
    d = mujoco.MjData(m)
    ik = PosIK(m)
    ik.pitch_target = np.radians(90.0)  # 그리퍼 아래 향함
    bodies = {m.body(b).id for b in ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")}
    j_lo = m.jnt_range[:5, 0] + T.JOINT_MARGIN
    j_hi = m.jnt_range[:5, 1] - T.JOINT_MARGIN

    goal = np.r_[XY, Z_ABOVE]
    for _ in range(3):
        ik.solve_pose(goal, iters=100, w_pitch=0.02, margin=T.JOINT_MARGIN)
    d.qpos[:5] = ik.d.qpos[:5]
    d.qpos[5] = T.GRIP_CLOSED
    d.ctrl[:6] = d.qpos[:6]
    mujoco.mj_forward(m, d)
    q_ik = d.ctrl[:4].copy()
    rows = []
    n_tot = int((SETTLE_S + t_end) / m.opt.timestep)
    for n in range(n_tot):
        t = n * m.opt.timestep - SETTLE_S
        goal = np.r_[XY, zfun(max(t, 0.0))]
        if n % 10 == 0:
            ik.solve_pose(goal, w_pitch=0.02, margin=T.JOINT_MARGIN)
            q_ik = ik.d.qpos[:4].copy()
        q_new = np.clip(np.r_[q_ik, 0.0], j_lo, j_hi)
        dq = T.JOINT_RATE * m.opt.timestep
        d.ctrl[:5] += np.clip(q_new - d.ctrl[:5], -dq, dq)
        d.ctrl[5] = T.GRIP_CLOSED
        cubes.apply(d)
        for qa, da in pad_adr:
            if d.qpos[qa] > 0:
                d.qfrc_applied[da] -= T.PAD_OUT_K * d.qpos[qa] + T.PAD_OUT_B * d.qvel[da]
        mujoco.mj_step(m, d)
        if t >= 0:
            f = gripper_contact_force(m, d, bodies)
            top = cubes.faces(d)[level]["top"]
            rows.append((t, *goal, *d.site_xpos[sid], *f, top[0], top[1]))
    A = np.array(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{level}_{name}.npz")
    np.savez(path, t=A[:, 0], hand=A[:, 1:4], tip=A[:, 4:7], F=A[:, 7:10], top_mm=A[:, 10], top_N=A[:, 11])
    fm = np.linalg.norm(A[:, 7:10], axis=1)
    print(f"{level:6s} {name:10s} {A[-1,0]:5.2f}s  |F| 최대 {fm.max():5.2f}N  윗면 패드 최대 {A[:,10].max():4.2f}mm {A[:,11].max():4.2f}N  "
          f"끝-목표 z 최대 차 {np.abs(A[:,6]-A[:,3]).max()*1e3:4.1f}mm -> {os.path.relpath(path, HERE)}")
    return path


if __name__ == "__main__":
    lv = [sys.argv[1]] if len(sys.argv) > 1 else LEVELS
    sc = [sys.argv[2]] if len(sys.argv) > 2 else list(SCENARIOS)
    for L in lv:
        for s in sc:
            run(L, s)


