"""
그리퍼 제어 비교 (cad2026, 헤드리스, 2026-11-18)

시나리오: 큐브 옆에 그리퍼를 내려 놓음 -> 잡기 -> 0.5초 안정 -> 1초에 걸쳐 10cm 들어 올림 -> 1초 유지
방법
  full    : 끝까지 닫기 (그리퍼 관절 -0.17, 서보 최대 힘) - 기존 방식
  ref2mm  : 지금 텔레오퍼레이션 방식 (정답 패드 닫힘 2mm에서 멈춤, 정답을 쓰므로 실물에선 불가 - 참고용)
  stiff   : 강성 기반 (grasp/grip_control.py), 턱 힘에 60Hz + 노이즈 sigma + 33ms 지연
지표: 최대 패드 변형(정답, 두 패드 합 / 가장 깊은 패드), 유지 중 한쪽 힘, 들 때 미끄러짐(큐브가 턱 대비 내려간 거리),
      들기 성공(큐브가 8cm 넘게 올라감), 강성 기반은 추정 k와 목표 힘
결과: data/grasp/grip_compare.csv + 표
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from grasp import grip_dataset as G  # noqa: E402
from grasp.grip_control import GripController, JawGeometry  # noqa: E402
from teleop import run as T  # noqa: E402
from teleop.ik import PosIK  # noqa: E402

OUT = os.path.join(ROOT, "data", "grasp")
SIGMAS = (0.3, 0.5, 0.7)
SEEDS = 5
LIFT = 0.10
# 잡기 위치: 고정 턱 끝을 큐브 옆면 1.5mm 밖에 (사람이 잡을 때처럼 거의 붙여 놓음).
# 강성 추정 데이터(grip_dataset)는 3mm 밖이라 큐브를 먼저 밀어 옮기는 구간이 있음
GOAL = np.array([0.25 - 0.015 - 0.0015, 0.0, 0.060])


def trial(level, method, sigma=0.0, seed=0):
    rng = np.random.default_rng(seed * 101 + 5)
    m = T.load_scene(level)
    m.site_pos[m.site("gripperframe").id] = T.FIXED_JAW_TIP
    pads = [m.joint(f"cube_{level}_{f}") for f in ("top", "xneg", "xpos")]
    for j in pads:
        m.jnt_solref[j.id] = T.PAD_LIMIT_SOLREF
        m.jnt_solimp[j.id, :3] = T.PAD_LIMIT_SOLIMP
    pad_adr = [(j.qposadr[0], j.dofadr[0]) for j in pads]
    cubes = T.CubeForces(m, T.CUBE_PARAMS)
    d = mujoco.MjData(m)
    ik = PosIK(m)
    ik.pitch_target = np.radians(90)
    start = GOAL + [0, 0, G.ABOVE]
    for _ in range(3):
        ik.solve_pose(start, iters=100, w_pitch=0.02, margin=T.JOINT_MARGIN)
    d.qpos[:5] = ik.d.qpos[:5]
    d.qpos[5] = G.Q_START
    d.ctrl[:6] = d.qpos[:6]
    mujoco.mj_forward(m, d)
    j_lo, j_hi = m.jnt_range[:5, 0] + T.JOINT_MARGIN, m.jnt_range[:5, 1] - T.JOINT_MARGIN
    q_ik = d.ctrl[:4].copy()
    jaw_b = m.body("moving_jaw_so101_v1").id
    cube_b = m.body(f"cube_{level}").id
    cube_bodies = {cube_b} | {m.body(f"cube_{level}_pad_{f}").id for f in ("top", "xneg", "xpos")}
    jq = m.joint("gripper").qposadr[0]
    sid = m.site("gripperframe").id
    jaw = JawGeometry(m)
    gc = GripController(jaw, sigma=max(sigma, 0.1))
    f6 = np.zeros(6)
    u_dir = None
    t_grip = 2.0                       # 내려온 뒤 잡기 시작
    q_cmd = G.Q_START
    t_held = None                      # 잡기 완료(안정) 시각
    next_s, s_phase = 0.0, rng.uniform(0, 1 / 60)
    queue = []                         # (도착 시각, 측정 시각, 값)
    max_cl = max_pad = 0.0
    f_hold, rel0, slip, t_end = [], None, 0.0, None
    for n in range(int(14.0 / m.opt.timestep)):
        t = n * m.opt.timestep
        # 팔: 내려오기 -> (잡힌 뒤) 들어 올리기
        u = min(1.0, t / 1.5)
        goal = start + (GOAL - start) * (1 - np.cos(np.pi * u)) / 2
        if t_held is not None and t > t_held + 0.5:
            v = min(1.0, (t - t_held - 0.5) / 1.0)
            goal = goal + [0, 0, LIFT * (1 - np.cos(np.pi * v)) / 2]
            if v >= 1.0 and t_end is None:
                t_end = t + 1.0
        if n % 10 == 0:
            ik.solve_pose(goal, w_pitch=0.02, margin=T.JOINT_MARGIN)
            q_ik = ik.d.qpos[:4].copy()
        d.ctrl[:5] += np.clip(np.clip(np.r_[q_ik, 0.0], j_lo, j_hi) - d.ctrl[:5], -T.JOINT_RATE * 1e-3, T.JOINT_RATE * 1e-3)
        # 그리퍼
        closure, f_side = cubes.grip(d)[level]
        if t >= t_grip:
            if method == "full":
                q_cmd = max(T.GRIP_CLOSED, q_cmd - gc.RATE * 1e-3)
                if q_cmd <= T.GRIP_CLOSED and t_held is None:
                    t_held = t + 0.3
            elif method == "ref2mm":
                if t_held is None:
                    q_cmd -= 1e-3 * T.GRIP_RATE * np.clip((2.0 - closure) / 0.5, -1, 1)
                    if closure >= 1.95:
                        t_held = t + 0.3
                else:
                    q_cmd -= 1e-3 * T.GRIP_RATE * np.clip((2.0 - closure) / 0.5, -1, 1)
            else:
                if gc.state == "idle" and t_held is None and q_cmd == G.Q_START:
                    gc.start(t, d.qpos[jq])
                qc = gc.step(t, d.qpos[jq])
                if qc is not None:
                    q_cmd = qc
                if gc.state == "hold" and t_held is None:
                    t_held = t + 1.0  # 유지 중 힘 보정 시간
                elif gc.state != "hold" and t_held is not None and t < t_held + 0.5:
                    t_held = None  # 들기 전에 판단이 취소되면 다시 기다림
        d.ctrl[5] = q_cmd
        cubes.apply(d)
        for qa, da in pad_adr:
            if d.qpos[qa] > 0:
                d.qfrc_applied[da] -= T.PAD_OUT_K * d.qpos[qa] + T.PAD_OUT_B * d.qvel[da]
        mujoco.mj_step(m, d)
        # 움직이는 턱 힘 -> 60Hz + 노이즈 + 지연 -> 제어기
        if method == "stiff":
            fj = np.zeros(3)
            for i in range(d.ncon):
                c = d.contact[i]
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if jaw_b in (b1, b2) and (b1 in cube_bodies or b2 in cube_bodies):
                    mujoco.mj_contactForce(m, d, i, f6)
                    fw = c.frame.reshape(3, 3).T @ f6[:3]
                    fj += fw if b2 == jaw_b else -fw
            if u_dir is None:  # 닫히는 방향(턱이 받는 힘 방향) = 그리퍼 좌표 고정 축 -> 월드
                u_dir = np.array([1.0, 0, 0])
            Rg = d.xmat[m.body("gripper").id].reshape(3, 3)
            if t >= s_phase + next_s:
                axis = Rg @ AX_G
                queue.append((t + 0.033, t, float(fj @ axis) + rng.normal(0, sigma)))
                next_s += 1 / 60
            while queue and queue[0][0] <= t + 1e-9:
                _, tc, fv = queue.pop(0)
                gc.push_force(tc, fv)
        faces = cubes.faces(d)[level]
        max_cl = max(max_cl, closure)
        max_pad = max(max_pad, max(faces["xneg"][0], faces["xpos"][0]))
        if t_held is not None and t_held <= t < t_held + 0.5:
            f_hold.append(f_side)
        rel = d.xpos[cube_b][2] - d.site_xpos[sid][2]
        if t_held is not None and rel0 is None and t >= t_held + 0.4:
            rel0 = rel
        if rel0 is not None:
            slip = max(slip, rel0 - rel)
        if t_end is not None and t > t_end:
            break
    lift = d.xpos[cube_b][2] - 0.05
    return dict(level=level, method=method, sigma=sigma, seed=seed, held=t_held is not None,
                max_closure_mm=max_cl, max_pad_mm=max_pad, f_hold=float(np.mean(f_hold)) if f_hold else np.nan,
                slip_mm=slip * 1e3, lifted=lift > 0.08, lift_mm=lift * 1e3,
                k=gc.k if method == "stiff" else np.nan, f_target=gc.f_target if method == "stiff" else np.nan)


def _axis():
    """닫힘 방향(움직이는 턱이 받는 힘의 방향)을 그리퍼 좌표로: 잡기 데이터의 정답 힘 방향 평균"""
    m = T.load_scene("medium")
    d = mujoco.MjData(m)
    Z = np.load(os.path.join(OUT, "grip_medium_slow.npz"))
    # 데이터에는 월드 방향만 있음 -> 같은 자세로 그리퍼 회전을 구해 그리퍼 좌표로
    m.site_pos[m.site("gripperframe").id] = T.FIXED_JAW_TIP
    ik = PosIK(m)
    ik.pitch_target = np.radians(90)
    for _ in range(3):
        ik.solve_pose(G.GOAL, iters=100, w_pitch=0.02, margin=T.JOINT_MARGIN)
    d.qpos[:5] = ik.d.qpos[:5]
    mujoco.mj_kinematics(m, d)
    Rg = d.xmat[m.body("gripper").id].reshape(3, 3)
    return Rg.T @ Z["f_dir"]


AX_G = _axis()


def job(a):
    return trial(*a)


def main():
    jobs = [(lv, "full", 0.0, 0) for lv in G.LEVELS] + [(lv, "ref2mm", 0.0, 0) for lv in G.LEVELS] + \
        [(lv, "stiff", s, sd) for lv in G.LEVELS for s in SIGMAS for sd in range(SEEDS)]
    with ProcessPoolExecutor(max_workers=min(15, os.cpu_count() or 4)) as ex:
        rows = list(ex.map(job, jobs))
    with open(os.path.join(OUT, "grip_compare.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'큐브':6s} {'방법':16s} | {'패드 합 최대':>9s} {'가장 깊은 패드':>11s} {'유지 힘(한쪽)':>11s} | "
          f"{'미끄러짐':>7s} {'들기 성공':>7s} | {'추정 k':>8s} {'목표 힘':>7s}")
    for lv in G.LEVELS:
        for meth, s in [("full", 0.0), ("ref2mm", 0.0)] + [("stiff", x) for x in SIGMAS]:
            R = [r for r in rows if r["level"] == lv and r["method"] == meth and r["sigma"] == s]
            name = {"full": "끝까지 닫기", "ref2mm": "2mm(정답 사용)", "stiff": f"강성 기반 {s}N"}[meth]
            avg = lambda k: np.nanmean([r[k] for r in R])  # noqa: E731
            print(f"{lv:6s} {name:16s} | {avg('max_closure_mm'):8.2f}mm {avg('max_pad_mm'):9.2f}mm {avg('f_hold'):10.2f}N | "
                  f"{avg('slip_mm'):6.1f}mm {sum(r['lifted'] for r in R):3d}/{len(R)} | "
                  f"{avg('k'):7.0f} {avg('f_target'):6.2f}N")


if __name__ == "__main__":
    main()


