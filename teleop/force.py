"""
텔레오퍼레이션 힘 피드백 부품 (cad2026 환경, 2026-11-17)

- gripper_contact_force(m, d, bodies): 그리퍼(고정 턱 + 움직이는 턱)가 받는 접촉력 합 (월드 좌표, N) = MuJoCo 정답
  잡을 때 양쪽 턱이 받는 힘은 서로 상쇄되고, 누를 때는 위(+z)로 밀리는 힘이 남음
- NAClient: na/server.py(neural_actuator 환경)와 UDP로 주고받음. 62Hz로 특징 전송, 응답은 비차단으로 받아 최신값 유지(ZOH)
  시작 후 BASELINE_S 동안의 평균을 기준으로 빼서 치우침 보정 (시뮬레이션 텔레메트리 추정은 접촉 없이도 약 -1.3N 치우침)
- ForceRenderer: 로봇 좌표 힘 -> Minverse 좌표, 배율, 20Hz 저역통과, 크기 상한 (F_HARD_MAX를 절대 넘지 않음)
- force_msg: set_cursor_force 메시지 (probe_position과 같이 보내지 않음)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import socket
import struct
import time

import mujoco
import numpy as np

F_HARD_MAX = 3.5  # N, Minverse 출력 상한 (2026-11-18 사용자 결정 2N -> 3.5N, 장치 최대 약 4N). 인자로도 이보다 크게 못 함
NA_ADDR = ("127.0.0.1", 47101)
NA_RESET = 0xFFFFFFFF
BASELINE_S = 2.0
NA_TIMEOUT_S = 0.5


def gripper_contact_force(m, d, bodies):
    """bodies에 속한 geom이 다른 물체에서 받는 접촉력 합 (월드 좌표)"""
    f6, total = np.zeros(6), np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        in1, in2 = b1 in bodies, b2 in bodies
        if in1 == in2:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        fw = c.frame.reshape(3, 3).T @ f6[:3]  # 접촉 좌표 -> 월드. 법선은 geom1->geom2, 값은 geom2가 받는 힘
        total += fw if in2 else -fw  # 그리퍼가 받는 힘
    return total


def contact_force_between(m, d, a_bodies, b_bodies):
    """a_bodies가 b_bodies에서 받는 접촉력 합 (월드 N). 예: 그리퍼(로봇)가 큐브에서 받는 힘 = 들고 있을 때 무게 + 관성"""
    f6, total = np.zeros(6), np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 in a_bodies and b2 in b_bodies:
            sign = -1.0  # 값은 geom2(b)가 받는 힘 -> a가 받는 힘은 반대
        elif b2 in a_bodies and b1 in b_bodies:
            sign = 1.0
        else:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        total += sign * (c.frame.reshape(3, 3).T @ f6[:3])
    return total


class NAClient:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:  # Windows: 서버가 아직 없을 때 ICMP 때문에 recv가 ConnectionResetError를 내는 것 방지
            self.sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        except (AttributeError, OSError):
            pass
        self.sock.setblocking(False)
        self.seq = 0
        self.force_raw = np.zeros(3)
        self.gate = 0.0
        self.last_reply = None
        self.t0 = None
        self.base_sum, self.base_n = np.zeros(3), 0
        self.baseline = None

    def reset(self):
        self.sock.sendto(struct.pack("<I", NA_RESET), NA_ADDR)
        self.t0 = time.perf_counter()
        self.base_sum[:], self.base_n, self.baseline = 0, 0, None

    def rezero(self):
        """지금부터 BASELINE_S 동안 다시 기준을 잡음 (버튼 a)"""
        self.t0 = time.perf_counter()
        self.base_sum[:], self.base_n, self.baseline = 0, 0, None

    def send(self, feat):
        self.seq = (self.seq + 1) % NA_RESET
        self.sock.sendto(struct.pack("<I", self.seq) + np.asarray(feat, "<f4").tobytes(), NA_ADDR)

    def poll(self):
        """받은 응답을 모두 읽고 최신값 유지. 반환: (기준을 뺀 힘 또는 None, 응답이 끊겼는지)"""
        while True:
            try:
                pkt = self.sock.recv(1024)
            except (BlockingIOError, ConnectionResetError):
                break
            v = np.frombuffer(pkt, "<f4", offset=4)
            self.force_raw, self.gate = v[:3].astype(float), float(v[3])
            self.last_reply = time.perf_counter()
            if self.baseline is None and self.t0 is not None:
                self.base_sum += self.force_raw
                self.base_n += 1
                if time.perf_counter() - self.t0 >= BASELINE_S and self.base_n:
                    self.baseline = self.base_sum / self.base_n
                    print(f"NeuralActuator 기준(치우침) {np.round(self.baseline, 2)} N -> 이후 이 값을 뺌")
        stale = self.last_reply is None or time.perf_counter() - self.last_reply > NA_TIMEOUT_S
        if stale or self.baseline is None:
            return None, stale
        return self.force_raw - self.baseline, False


class ForceRenderer:
    def __init__(self, R_haply_to_robot, scale, f_max, dt, cutoff_hz=20.0):
        self.Rt = np.asarray(R_haply_to_robot, float).T  # 로봇 -> Minverse
        self.scale = scale
        self.f_max = min(f_max, F_HARD_MAX)
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        self.alpha = dt / (rc + dt)
        self.f = np.zeros(3)

    def update(self, f_robot, enable=True):
        target = self.scale * (self.Rt @ f_robot) if enable else np.zeros(3)
        self.f += self.alpha * (target - self.f)
        nrm = np.linalg.norm(self.f)
        if nrm > self.f_max:
            self.f *= self.f_max / nrm
        return self.f.copy()


class ContactRenderer:
    """로컬 접촉 모델 (2026-11-17, --force-mode contact): 접촉력을 그대로 돌려주지 않고
    '손 목표가 표면 안쪽으로 들어간 깊이 x 스프링 + 감쇠'로 Minverse 힘을 만듦.

    배경: 단단한 블록을 꾹 누르면 위치 제어 로봇이 닿았다 떨어졌다를 20ms 안에 반복 (정답 0 <-> 40N, 최대 78N).
    이를 그대로(raw) 렌더링하면 2N 상한에 잘린 채 0~2N을 오가며 손이 튕김 (teleop_cube_20261117_133341.csv)
    - 접촉 시작: 그리퍼 접촉력 > ON_N. 그 순간 '목표 기준 표면 평면'을 고정 (점 = 그때의 목표 + 끝-목표 차이, 법선)
      -> 깊이 = 목표가 그 평면 아래로 들어간 거리. 로봇 끝이 표면에서 튕겨도(채터링) 깊이는 손 목표로만 정해져 매끄러움
      (1차 버전은 현재 끝 위치를 써서 끝의 튕김이 그대로 힘에 섞였음, 2026-11-17 비교 그래프)
      (방향 우선 IK 때문에 원래 있던 끝-목표 차이(최대 3cm)는 시작 순간 값으로 빼서 힘으로 바뀌지 않게)
    - 접촉 끝: 손 목표가 평면 위로 나오면(깊이 0) 끝. 예외로 실제 접촉이 OFF_S 넘게 없으면(옆으로 미끄러져 벗어남) 끝
      (1차 버전은 30ms만 접촉이 끊겨도 끝나 힘이 0으로 떨어졌다 올라가기를 반복)
    - 접촉 판정은 접촉력 크기를 MAG_HZ로 거른 값으로 함 (2026-11-17 수정, teleop_cube_20261117_140403.csv):
      큐브를 들고 있으면 양쪽 턱 힘이 상쇄돼 합이 0.1~0.4N인데, 순간 튀는 값(상위 1% 12N)이 0.5N 판정을 넘어
      '접촉'으로 잘못 켜지고, 손을 움직이면 깊이가 최대 98mm로 잡혀 2N이 짧게 켜졌다 꺼짐 = '뚜둑뚜둑'
      -> 시작: 걸러진 크기 > ON_N 이 ON_S 넘게 유지될 때만 / 끝: 걸러진 크기 < OFF_N 이 OFF_S 넘게 (블록 채터링 틈은 필터가 메움)
    - 법선: 접촉 중 그리퍼가 받는 접촉력 방향을 NORMAL_HZ로 걸러서 천천히 갱신
    - 힘(Minverse 좌표) = K_H * (깊이 / 위치 배율) * 법선 - B_H * 손 속도(접촉 중에만), 크기 상한 f_max
    """
    ON_N, ON_S, OFF_N, OFF_S = 1.0, 0.02, 0.5, 0.05
    MAG_HZ, NORMAL_HZ = 10.0, 5.0
    UNILATERAL = True  # 밀기만 (2026-11-18). False = 예전 방식

    def __init__(self, R_haply_to_robot, pos_scale, k_h, b_h, f_max, dt):
        self.Rt = np.asarray(R_haply_to_robot, float).T
        self.s, self.k, self.b = pos_scale, k_h, b_h
        self.f_max = min(f_max, F_HARD_MAX)
        self.dt = dt
        rc = 1.0 / (2 * np.pi * self.NORMAL_HZ)
        self.a_n = dt / (rc + dt)
        rc = 1.0 / (2 * np.pi * self.MAG_HZ)
        self.a_m = dt / (rc + dt)
        self.mag = 0.0  # 걸러진 접촉력 크기
        self.on_t = 0.0
        self.in_contact, self.off_t = False, 0.0
        self.nf = np.zeros(3)
        self.plane0 = np.zeros(3)
        self.depth = 0.0
        self.wall = 0.0  # '갈 수 없는 곳' 벽 안으로 들어간 거리 (로봇 m)
        self.f = np.zeros(3)
        self.k_obj = None  # 공유받은 물체 강성 (N/m, set_object_stiffness)

    def set_object_stiffness(self, k_n_per_m):
        """강성 인지 파지(grasp/)에서 추정한 물체 강성 공유 (2026-11-18). 지금은 저장만 하고 렌더링 스프링은 --k 그대로
        (Minverse 스프링으로 옮기는 규칙은 아직 미정: 물체 강성은 수백~수천 N/m, Minverse는 안정 범위가 따로 있음)"""
        self.k_obj = float(k_n_per_m)

    def update(self, f_contact, tip, goal, hand_vel, enable=True, hand=None, pen=None):
        """hand: 손이 원하는 로봇 목표(속도 제한 전, 작업 상자로만 자름). 없으면 goal
        pen: '로봇이 갈 수 없는 곳' 벽 (2026-11-18) = 로봇이 갈 수 있는 점 - 손이 원하는 점 (로봇 좌표 m).
             작업 상자 밖 + IK가 닿지 못하는 곳. 접촉 여부와 무관하게 같은 스프링으로 되밀어 줌"""
        hand = goal if hand is None else hand
        fn = float(np.linalg.norm(f_contact))
        self.mag += self.a_m * (fn - self.mag)
        if fn > 1e-6:  # 법선 방향은 힘이 있을 때만 갱신 (크기 가중)
            self.nf += self.a_n * (f_contact / max(fn, self.ON_N) - self.nf)
        if not self.in_contact:
            self.on_t = self.on_t + self.dt if self.mag > self.ON_N else 0.0
            if self.on_t >= self.ON_S:
                self.in_contact, self.off_t = True, 0.0
                self.plane0 = goal.copy()  # 접촉 시작 순간 목표 위치 = 평면 위의 점 (목표 기준)
        n = self.nf / np.linalg.norm(self.nf) if np.linalg.norm(self.nf) > 1e-6 else np.zeros(3)
        # 깊이는 '손'으로 잼 (2026-11-18): 예전엔 속도 제한된 로봇 목표(goal)로 재서, 손이 더 깊이 들어가도
        # 목표가 따라오지 못하면 깊이가 늘지 않음 -> 손이 그냥 통과
        self.depth = max(0.0, float((self.plane0 - hand) @ n)) if self.in_contact else 0.0
        if self.in_contact:
            self.off_t = 0.0 if self.mag > self.OFF_N else self.off_t + self.dt
            if self.off_t > self.OFF_S:
                self.in_contact, self.on_t = False, 0.0
        hv = np.asarray(hand_vel, float)
        f_h = np.zeros(3)
        # 표면은 밀기만 하고 당기지 않음 (2026-11-18, 사용자 "짚고 뗄 때 찐득하게 떼짐"):
        # 예전엔 접촉 중 감쇠 -b*손속도가 모든 방향에 걸려, 손을 떼는 순간(스프링 힘은 줄었는데 접촉 판정은 아직 켜짐)
        # 손을 표면 쪽으로 끌어당김. 이제 법선 방향 힘 = max(0, 스프링 + 감쇠), 옆 방향 감쇠 없음
        if self.in_contact and self.UNILATERAL:
            nh = self.Rt @ n
            f_h += max(0.0, self.k * (self.depth / self.s) - self.b * float(hv @ nh)) * nh if self.depth > 0 else 0.0
        elif self.in_contact:  # 예전 방식 (비교용)
            f_h += self.k * (self.depth / self.s) * (self.Rt @ n) - self.b * hv
        self.wall = 0.0
        if pen is not None:
            pen = np.asarray(pen, float)
            self.wall = float(np.linalg.norm(pen))
            if self.wall > 1e-5:
                u = self.Rt @ (pen / self.wall)  # Minverse 좌표, 갈 수 있는 쪽
                if self.UNILATERAL:  # 벽도 같은 방식 (밀기만)
                    f_h += max(0.0, self.k * (self.wall / self.s) - self.b * float(hv @ u)) * u
                else:
                    f_h += self.k * (self.wall / self.s) * u
                    if not self.in_contact:
                        f_h -= self.b * (hv @ u) * u
        if not enable or not np.any(f_h):
            self.f = np.zeros(3)
            return self.f.copy()
        nrm = np.linalg.norm(f_h)
        if nrm > self.f_max:
            f_h *= self.f_max / nrm
        self.f = f_h
        return f_h.copy()


def force_msg(device_id, f):
    f = np.asarray(f, float)
    nrm = np.linalg.norm(f)
    if nrm > F_HARD_MAX:  # 마지막 방어선
        f = f * (F_HARD_MAX / nrm)
    return json.dumps({"inverse3": [{"device_id": device_id, "commands": {
        "set_cursor_force": {"vector": {"x": float(f[0]), "y": float(f[1]), "z": float(f[2])}}}}]})
