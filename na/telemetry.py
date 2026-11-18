"""
MuJoCo SO-101 상태 -> NeuralActuator 42차원 입력(서보 텔레메트리 흉내) (cad2026 환경, 2026-11-17)

NeuralActuator SO-101 입력 (evaluate_actuator_so101.build_features):
  0-5 goal_pos(rad) / 6-11 pos(rad) / 12-17 load(STS3215 부호 있는 카운트) / 18-23 vel(엔코더 스텝/s)
  24-29 volts(0.1V) / 30-35 temp(°C) / 36-41 goal_pos - pos(rad).  샘플 주기 0.01605 s (약 62Hz), 히스토리 8프레임
관절 각도 기준: 우리 Menagerie 모델 = NeuralActuator so101_new_calib (같은 q에서 위치 차이 0, na/calib_static.py)
load: load ≈ k * (샘플 구간 평균 액추에이터 토크, N·m) + c * (관절 속도, rad/s) + b  (config/na_telemetry.json)
  - na/calib_dynamic.py: 실측 train CSV 궤적을 재생해 맞춤 (상관 0.38~0.75. 순간 토크는 ±2.94 한계에 자주 붙어 평균을 씀,
    Feetech load는 PWM 듀티에 가까워 속도 항이 큼)
  - 매 시뮬레이션 스텝 accumulate(d), 샘플 시점에 features(d)
전압/온도: 실측 평균으로 고정
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import os

import numpy as np

HERE = ROOT
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DATA_DT = 0.01605
K_DEFAULT = -210.0


class SimTelemetry:
    def __init__(self, m, cfg_path=os.path.join(HERE, "config", "na_telemetry.json")):
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        self.qadr = np.array([m.joint(j).qposadr[0] for j in JOINTS])
        self.dadr = np.array([m.joint(j).dofadr[0] for j in JOINTS])
        self.aid = np.array([m.actuator(j).id for j in JOINTS])
        fit = cfg["load_fit"]
        self.k = np.array([(fit.get(j) or {}).get("k_counts_per_Nm", K_DEFAULT) for j in JOINTS])
        self.c = np.array([(fit.get(j) or {}).get("c_counts_per_radps", 0.0) for j in JOINTS])
        self.b = np.array([(fit.get(j) or {}).get("offset", 0.0) for j in JOINTS])
        self.volts = np.array(cfg["volts_mean"])
        self.temps = np.array(cfg["temp_mean"])
        self.vscale = cfg["vel_counts_per_rad"]
        self.tau_sum, self.n = np.zeros(6), 0

    def accumulate(self, d):
        """매 시뮬레이션 스텝 호출: 샘플 구간 평균 토크용"""
        self.tau_sum += d.actuator_force[self.aid]
        self.n += 1

    def features(self, d):
        goal = d.ctrl[self.aid].copy()
        pos = d.qpos[self.qadr].copy()
        tau = self.tau_sum / self.n if self.n else d.actuator_force[self.aid]
        self.tau_sum[:], self.n = 0, 0
        load = np.round(self.k * tau + self.c * d.qvel[self.dadr] + self.b)  # 실물 레지스터는 정수
        vel = np.round(d.qvel[self.dadr] * self.vscale)
        return np.concatenate([goal, pos, load, vel, self.volts, self.temps, goal - pos]).astype(np.float32)
