"""
강성 기반 그리퍼 제어 (2026-11-18, 강성 인지 파지 2단계)

    jaw = JawGeometry(m)                 # 그리퍼 관절각 q <-> 움직이는 턱 닫힘 거리 p (m)
    gc = GripController(jaw, sigma)      # sigma: 힘 노이즈 추정 (N)
    gc.start(t, q_now)                   # 버튼 b를 누른 순간
    q_cmd = gc.step(t, q_meas)           # 1kHz, 그리퍼 관절 명령 반환
    gc.push_force(t_cap, f)              # 저속 힘 샘플 (움직이는 턱 힘, 닫히는 쪽 +)
    gc.state, gc.k, gc.k_std, gc.p0, gc.f_target, gc.f_now   # 화면 표시용

동작
  closing: 일정 속도로 닫으며, 닫기 시작 이후 모든 샘플에 힌지 맞춤 f = k*max(0, p-p0) (grasp.stiffness.hinge_fit)
    - 확신 조건: k > 0, k/k_std > CONF, 접촉 뒤 MIN_DEPTH 넘게 들어감
    - 확신되면 목표: 목표 힘 F = clip(k * D_NOM, F_MIN, F_MAX)  (단단하면 크게, 부드러우면 작게)
                     목표 깊이 = min(F / (k + k_std), D_MAX)  (불확실하면 강성을 크게 잡아 덜 조임)
                     목표 위치 p_t = p0 + 목표 깊이 -> 이미 넘었으면 그 자리(되돌아가지 않음), 아니면 p_t까지 더 닫고 hold
    - 안전: 확신 전이라도 (추정 접촉점 + D_MAX)를 넘으면 멈춤
  hold: p_t에 해당하는 관절각 유지
깊이/힘은 '실효'(턱 위치로 잰) 값. 실물: f = 그리퍼 서보 전류 기반 힘, p = 서보 각도 (CLAUDE.md 4번 메모)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from grasp.stiffness import hinge_fit  # noqa: E402

JAW_CFG = os.path.join(ROOT, "config", "grip_jaw.json")


class JawGeometry:
    """그리퍼 관절각 q -> 움직이는 턱 접촉점의 닫힘 거리 p (q가 작아질수록 p 증가, q_ref에서 0)"""

    def __init__(self, m, q_ref=0.40, q_min=-0.175):
        pt = np.array(json.load(open(JAW_CFG, encoding="utf-8"))["jaw_contact_point_local_m"])
        d = mujoco.MjData(m)
        jaw, grip = m.body("moving_jaw_so101_v1").id, m.body("gripper").id
        qa = m.joint("gripper").qposadr[0]
        qs = np.linspace(q_min, q_ref, 400)
        P = []
        for q in qs:
            d.qpos[qa] = q
            mujoco.mj_kinematics(m, d)
            xw = d.xpos[jaw] + d.xmat[jaw].reshape(3, 3) @ pt
            P.append(d.xmat[grip].reshape(3, 3).T @ (xw - d.xpos[grip]))
        P = np.array(P)
        ax = P[0] - P[-1]
        ax /= np.linalg.norm(ax)
        self.qs = qs
        self.ps = (P - P[-1]) @ ax  # q_ref에서 0, 닫을수록 +
        self.q_ref = q_ref
        self.axis_g = ax  # 그리퍼 좌표에서 턱이 닫히는 방향 (턱이 큐브에서 받는 힘은 이 반대)
        self.jaw_id, self.grip_id = jaw, grip

    def jaw_force(self, m, d, other_bodies):
        """움직이는 턱이 other_bodies에서 받는 접촉력의 '미는' 성분 (N, +) = 정답 (시뮬레이션)"""
        f6, fj = np.zeros(6), np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if self.jaw_id in (b1, b2) and (b1 in other_bodies or b2 in other_bodies):
                mujoco.mj_contactForce(m, d, i, f6)
                fw = c.frame.reshape(3, 3).T @ f6[:3]
                fj += fw if b2 == self.jaw_id else -fw
        return float(-(d.xmat[self.grip_id].reshape(3, 3).T @ fj) @ self.axis_g)

    def p(self, q):
        return float(np.interp(q, self.qs, self.ps))

    def q(self, p):
        return float(np.interp(p, self.ps[::-1], self.qs[::-1]))


class GripController:
    RATE = 0.10        # rad/s, 닫는 속도 (턱 접촉점 약 7mm/s)
    FAST_RATE = 0.8    # rad/s, 턱 사이가 46mm(q 0.40)보다 넓을 때 (텔레오퍼레이션은 q 1.0 = 91mm에서 시작)
    CONF = 3.0         # k / k_std
    MIN_DEPTH = 0.8e-3
    MIN_IN = 8         # 접촉점 뒤 샘플 수 (60Hz면 0.13초)
    D_NOM = 1.5e-3     # m, 목표 힘 = k * D_NOM
    D_MAX = 2.5e-3     # m, 최대 찌그러짐 (실효 깊이, 패드 약 1.8mm = 한계 3mm 안쪽)
    F_MIN, F_MAX = 0.25, 1.2   # N
    HOLD_LOW, HOLD_STEP = 0.8, 0.05e-3  # 유지 중 0.5초 평균 힘이 목표의 80% 미만이면 0.1초마다 0.05mm 더 조임
    REDO = 0.4         # 유지 중 0.5초 평균 힘이 목표의 40% 미만이면 판단 취소, 다시 닫기

    def __init__(self, jaw, sigma=0.5, q_closed=-0.17):
        self.jaw, self.sigma, self.q_closed = jaw, sigma, q_closed
        self.reset()

    def reset(self):
        self.state = "idle"
        self.samples = []      # (p at t_cap, f)
        self.ph = []           # (t, p)
        self.k = self.k_std = self.p0 = None
        self.p_target = self.f_target = None
        self.f_now = 0.0
        self.q_cmd = None
        self.n_hold = 0
        self.redo = 0  # 판단 취소 횟수 (표시/기록용)

    def start(self, t, q_now):
        self.reset()
        self.state, self.q_cmd = "closing", q_now

    def release(self):
        self.reset()

    STATE_CODE = {"idle": 0, "closing": 1, "approach": 2, "hold": 3}

    def _p_at(self, tc):
        for tt, pp in reversed(self.ph):
            if tt <= tc + 1e-9:
                return pp
        return self.ph[0][1] if self.ph else 0.0

    def push_force(self, t_cap, f):
        if self.state == "idle":
            return
        self.samples.append((self._p_at(t_cap), float(f)))
        tail = [s[1] for s in self.samples[-6:]]
        self.f_now = float(np.mean(tail))
        if self.state == "hold" and self.p_target is not None and self.p0 is not None:
            # 멈춘 뒤 힘이 풀리면(큐브가 자리를 잡으며 목표의 약 절반으로 떨어짐, 2026-11-18 시험) 조금씩 더 조임
            self.n_hold += 1
            f30 = float(np.mean([s[1] for s in self.samples[-30:]]))
            if self.n_hold >= 30 and f30 < self.REDO * self.f_target:
                # 유지 0.5초 평균이 목표의 40%도 안 됨 = 노이즈 때문에 너무 일찍 판단한 것 -> 다시 닫으며 추정 계속
                # (2026-11-18 시험: 특정 노이즈 패턴에서 패드 0.3~0.6mm일 때 판단 -> 힘 0.01N으로 들기 실패)
                self.state, self.n_hold, self.redo = "closing", 0, self.redo + 1
                return
            if self.n_hold >= 30 and self.n_hold % 6 == 0 and f30 < self.HOLD_LOW * self.f_target \
                    and self.p_target < self.p0 + self.D_MAX:
                self.p_target = min(self.p_target + self.HOLD_STEP, self.p0 + self.D_MAX)
            return
        if self.state != "closing" or len(self.samples) < 8:
            return
        P = np.array([s[0] for s in self.samples])
        F = np.array([s[1] for s in self.samples])
        r = hinge_fit(P, F)
        if r is None:
            return
        self.k, self.p0, self.k_std, _ = r
        p_now = self.ph[-1][1]
        depth = p_now - self.p0
        snr = self.k / max(self.k_std, 1e-9)
        # 지금 실제로 힘이 걸려 있다는 증거 (닿기 전 노이즈로 가짜 접촉점을 잡는 것 방지):
        # 최근 12개 평균 > 3시그마, 접촉점 뒤 샘플 MIN_IN개 이상 (처음 1.5시그마/6개는 노이즈로 수시로 넘어 허공에서 멈춤)
        f12 = float(np.mean(F[-12:]))
        n_in = int((P > self.p0).sum())
        # + 평균 힘이 F_MIN 이상: 고정 턱에 닿기 전 큐브를 밀고만 있을 때(블록 마찰 약 0.1N)를 '쥐었다'로 착각하지 않게
        #   (2026-11-18 시험: 고정 턱 쪽 패드는 0mm인데 미는 힘으로 판단 -> 멈추자 큐브가 미끄러져 힘이 풀림)
        f_evid = f12 > max(3 * self.sigma / np.sqrt(12), self.F_MIN) and n_in >= self.MIN_IN
        confident = self.k > 0 and snr > self.CONF and depth > self.MIN_DEPTH and f_evid
        if confident:
            self.f_target = float(np.clip(self.k * self.D_NOM, self.F_MIN, self.F_MAX))
            d_t = min(self.f_target / (self.k + self.k_std), self.D_MAX)
            # 이미 지나쳤으면 되돌아가지 않고 그 자리 유지 (D_MAX 안에서):
            # 큐브가 먼저 살짝 밀리며 힘이 천천히 올라 p0가 실제보다 약 1mm 일찍 잡힘 -> 되돌아가면 놓침 (2026-11-18 시험)
            self.p_target = max(self.p0 + d_t, min(p_now, self.p0 + self.D_MAX))
            self.state = "approach" if p_now < self.p_target else "hold"
        elif self.k > 0 and snr > 1.5 and f_evid and depth > self.D_MAX:  # 확신 전이라도 너무 깊으면 멈춤
            self.p_target, self.state = self.p0 + self.D_MAX, "hold"
            self.f_target = float(np.clip(self.k * self.D_NOM, self.F_MIN, self.F_MAX))

    def step(self, t, q_meas):
        p = self.jaw.p(q_meas)
        self.ph.append((t, p))
        if len(self.ph) > 2000:
            del self.ph[:1000]
        if self.state == "idle":
            return None
        if self.state == "closing":  # 턱이 큐브 크기보다 넓게 열려 있는 동안(q > q_ref)은 빠르게
            rate = self.FAST_RATE if self.q_cmd > self.jaw.q_ref else self.RATE
            self.q_cmd = max(self.q_closed, self.q_cmd - rate * 0.001)
        elif self.state == "approach":
            q_t = self.jaw.q(self.p_target)
            self.q_cmd = max(q_t, self.q_cmd - self.RATE * 0.001)
            if self.q_cmd <= q_t + 1e-6:
                self.state = "hold"
        if self.state == "hold":
            self.q_cmd = self.jaw.q(self.p_target)
        return self.q_cmd
