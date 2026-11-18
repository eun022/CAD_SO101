"""
잡기 강성 추정용 데이터 (cad2026, 헤드리스, 2026-11-18)

큐브(블록 위)를 양쪽 턱 사이에 두고 그리퍼를 일정 속도로 닫음 -> 패드 닫힘 GRIP_TO_MM까지 -> 유지 -> 엶
1kHz 기록 -> data/grasp/grip_<level>_<speed>.npz
  q_grip: 그리퍼 관절 각도 (rad, 로봇이 아는 값)
  p_jaw: 움직이는 턱 닫힘 거리 (m) = 그리퍼 관절 각도 -> 운동학 (턱의 접촉 지점이 닫히는 방향으로 움직인 거리)
  f_jaw: 움직이는 턱이 큐브에서 받는 힘 (월드 N, 3) / f_n: 닫힘 방향 성분 (N, 미는 쪽 +)
  closure_gt: 패드 눌림 합 (mm, 정답), f_side_gt: 패드 FEM 힘 한쪽 평균 (N, 정답)
그리퍼 전체 힘 합은 양쪽 턱이 상쇄돼 거의 0 -> 잡기 강성은 턱 하나의 힘으로 봐야 함
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from teleop import run as T  # noqa: E402
from teleop.ik import PosIK  # noqa: E402

OUT = os.path.join(ROOT, "data", "grasp")
LEVELS = ("soft", "medium", "hard")
SPEEDS = {"slow": 0.1, "fast": 0.4}  # 그리퍼 관절 닫는 속도 (rad/s), 턱 끝 기준 약 7 / 30 mm/s
Q_START = 0.40                     # 턱 사이 약 46mm에서 시작
GRIP_TO_MM = 2.6                   # 패드 닫힘 이만큼에서 멈춤 (한계 3mm 안쪽)
HOLD_S = 0.5
# 고정 턱 끝(FIXED_JAW_TIP) 기준 위치: 큐브(중심 x 0.25, 반폭 15mm) -x 옆면에서 3mm 바깥, 큐브 높이 중간쯤
# (처음 버전은 두 턱 사이 점으로 놓아 고정 턱이 패드와 몸통 사이에 박힌 채 시작 -> 턱 힘이 FEM의 3배로 엉터리)
GOAL = np.array([0.25 - 0.015 - 0.003, 0.0, 0.060])
ABOVE = 0.05  # 이만큼 위에서 내려옴 (턱이 큐브에 박힌 채 시작하지 않게)


def path_name(level, speed):
    return os.path.join(OUT, f"grip_{level}_{speed}.npz")


def jaw_closing(m, level):
    """움직이는 턱 위의 '패드 접촉 지점'(턱 좌표)과 닫힘 거리 함수 p(q)"""
    d = mujoco.MjData(m)
    jaw = m.body("moving_jaw_so101_v1").id
    grip = m.body("gripper").id
    jg = m.joint("gripper")

    def pt_in_gripper(q, pt_local):
        d.qpos[jg.qposadr[0]] = q
        mujoco.mj_kinematics(m, d)
        xw = d.xpos[jaw] + d.xmat[jaw].reshape(3, 3) @ pt_local
        return d.xmat[grip].reshape(3, 3).T @ (xw - d.xpos[grip])
    return pt_in_gripper


def run(level, speed):
    m = T.load_scene(level)
    m.site_pos[m.site("gripperframe").id] = T.FIXED_JAW_TIP
    pads =[m.joint(f"cube_{level}_{f}") for f in ("top", "xneg", "xpos")]
    for j in pads:
        m.jnt_solref[j.id] = T.PAD_LIMIT_SOLREF
        m.jnt_solimp[j.id, :3] = T.PAD_LIMIT_SOLIMP
    pad_adr = [(j.qposadr[0], j.dofadr[0]) for j in pads]
    cubes = T.CubeForces(m, T.CUBE_PARAMS)
    d = mujoco.MjData(m)
    ik = PosIK(m)
    ik.pitch_target = np.radians(90)
    start = GOAL + [0, 0, ABOVE]
    for _ in range(3):
        ik.solve_pose(start, iters=100, w_pitch=0.02, margin=T.JOINT_MARGIN)
    d.qpos[:5] = ik.d.qpos[:5]
    d.qpos[5] = Q_START
    d.ctrl[:6] = d.qpos[:6]
    mujoco.mj_forward(m, d)
    j_lo = m.jnt_range[:5, 0] + T.JOINT_MARGIN
    j_hi = m.jnt_range[:5, 1] - T.JOINT_MARGIN
    q_ik = d.ctrl[:4].copy()
    T_DOWN = 1.5  # 내려오는 시간
    jaw = m.body("moving_jaw_so101_v1").id
    cube_bodies = {m.body(f"cube_{level}").id} | {m.body(f"cube_{level}_pad_{f}").id for f in ("top", "xneg", "xpos")}
    jq = m.joint("gripper").qposadr[0]
    rows, cpts = [], []
    q_cmd, phase, t_hold = Q_START, "settle", None
    f6 = np.zeros(6)
    for n in range(int(10.0 / m.opt.timestep)):
        t = n * m.opt.timestep
        u = min(1.0, t / T_DOWN)
        goal = start + (GOAL - start) * (1 - np.cos(np.pi * u)) / 2
        if n % 10 == 0:
            ik.solve_pose(goal, w_pitch=0.02, margin=T.JOINT_MARGIN)
            q_ik = ik.d.qpos[:4].copy()
        q_new = np.clip(np.r_[q_ik, 0.0], j_lo, j_hi)
        d.ctrl[:5] += np.clip(q_new - d.ctrl[:5], -T.JOINT_RATE * m.opt.timestep, T.JOINT_RATE * m.opt.timestep)
        closure, f_side = cubes.grip(d)[level]
        if phase == "settle" and t > T_DOWN + 0.5:
            phase = "close"
        elif phase == "close":
            q_cmd -= SPEEDS[speed] * m.opt.timestep
            if closure >= GRIP_TO_MM or q_cmd <= T.GRIP_CLOSED:
                phase, t_hold = "hold", t
        elif phase == "hold" and t - t_hold > HOLD_S:
            phase = "open"
        elif phase == "open":
            q_cmd = min(Q_START, q_cmd + SPEEDS[speed] * m.opt.timestep)
            if q_cmd >= Q_START and closure < 0.01:
                phase, t_end = "done", t + 0.3
        if phase == "done" and t > t_end:
            break
        d.ctrl[5] = q_cmd
        cubes.apply(d)
        for qa, da in pad_adr:
            if d.qpos[qa] > 0:
                d.qfrc_applied[da] -= T.PAD_OUT_K * d.qpos[qa] + T.PAD_OUT_B * d.qvel[da]
        mujoco.mj_step(m, d)
        fj = np.zeros(3)
        for i in range(d.ncon):  # 움직이는 턱이 큐브에서 받는 힘
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if {b1, b2} & {jaw} and {b1, b2} & cube_bodies:
                mujoco.mj_contactForce(m, d, i, f6)
                fw = c.frame.reshape(3, 3).T @ f6[:3]
                fj += fw if b2 == jaw else -fw
                R = d.xmat[jaw].reshape(3, 3)
                cpts.append(R.T @ (c.pos - d.xpos[jaw]))
        rows.append((t, d.qpos[jq], *fj, closure, f_side))
    A = np.array(rows)
    # 턱의 접촉 지점 (턱 좌표, 정답 접촉 위치 평균 = 한 번 하는 기하 보정에 해당)
    pt = np.mean(cpts, axis=0) if cpts else np.zeros(3)
    if level == "medium" and speed == "slow":  # 1회 기하 보정값 저장 (그리퍼 제어가 관절각 -> 턱 닫힘 거리 변환에 사용)
        import json
        json.dump({"jaw_contact_point_local_m": pt.tolist(), "from": os.path.relpath(path_name(level, speed), ROOT),
                   "note": "움직이는 턱(moving_jaw_so101_v1) 좌표에서 큐브 패드와 닿는 점 평균"},
                  open(os.path.join(ROOT, "config", "grip_jaw.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    ptg = jaw_closing(m, level)
    P = np.array([ptg(q, pt) for q in A[:, 1]])
    axis = P[0] - P[np.argmin(A[:, 1])]  # 닫히는 방향 (그리퍼 좌표)
    axis_n = axis / np.linalg.norm(axis)
    p_jaw = -(P - P[0]) @ axis_n  # 닫힌 거리 (m, +)
    p_jaw = -p_jaw if p_jaw.max() < 0 else p_jaw
    # 힘의 닫힘 방향 성분: 턱이 받는 힘은 닫는 방향의 반대 -> 정답 힘 최댓값 방향으로 부호/방향 정함
    fj = A[:, 2:5]
    u = fj[np.argmax(np.linalg.norm(fj, axis=1))]
    u = u / np.linalg.norm(u)
    f_n = fj @ u
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"grip_{level}_{speed}.npz")
    np.savez(path, t=A[:, 0], q_grip=A[:, 1], p_jaw=p_jaw, f_jaw=fj, f_dir=u, f_n=f_n, closure_gt=A[:, 5],
             f_side_gt=A[:, 6])
    print(f"grip {level:6s} {speed:4s} {A[-1,0]:4.1f}s 패드 닫힘 최대 {A[:,5].max():4.2f}mm 한쪽 FEM {A[:,6].max():4.2f}N "
          f"턱 힘 최대 {f_n.max():4.2f}N 턱 이동 {p_jaw.max()*1e3:5.1f}mm -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    for L in LEVELS:
        for s in SPEEDS:
            run(L, s)
