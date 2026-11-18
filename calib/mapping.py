"""
Minverse 이동 범위 -> SO-101 작업 상자 매핑 설정 만들기 (시뮬레이션 계산만, 장치 명령 없음)
사용법: python calib/mapping.py
입력: config/haply_workspace.json (calib/workspace.py 결과)
출력: config/teleop_mapping.json

매핑 (절대 위치):  robot = robot_center + scale * R @ (haply - haply_center)
- R: Minverse 앞(+y) -> 로봇 앞(+x), 오른쪽(+x) -> 로봇 오른쪽(-y), 위(+z) -> 위(+z)
- scale: 세 축 같은 배율 (모양/힘 방향 보존), 1/3 ~ 3 사이
- 작업 상자(= Minverse 1~99% 범위를 옮긴 상자) 안 격자점이 모두 도달 가능하고, 누르기 블록 윗점을 포함하도록
  가능한 가장 큰 scale과 robot_center를 찾음. 마지막에 상자 안 무작위 점으로 IK를 다시 확인
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import os
import time

import mujoco
import numpy as np

from teleop import press_demo as p
from teleop.ik import XS, YS, ZS, PosIK, reach_grid, reachable

HERE = ROOT
R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)  # haply -> robot 축 회전
SCALE_MIN, SCALE_MAX = 1 / 3, 3.0
MUST_INCLUDE = np.array([0.25, 0.0, 0.06])  # 누르기 블록(윗면 0.05) 바로 위
GRID = 0.02


def main():
    hw = json.load(open(os.path.join(HERE, "config", "haply_workspace.json"), encoding="utf-8"))
    h_center = np.array(hw["center_p1_p99_m"])
    h_half = np.array(hw["range_p1_p99_m"]) / 2
    r_half_unit = np.abs(R) @ h_half  # scale=1일 때 로봇 쪽 상자 반폭

    m = mujoco.MjModel.from_xml_path(p.SCENE)
    ik = PosIK(m)
    print("로봇 도달 격자 계산 중...")
    ok = reach_grid(m, ik)

    def box_ok(c, half):
        lo, hi = c - half - GRID / 2, c + half + GRID / 2  # 반 칸 여유를 두고 보수적으로 검사
        if (lo < [XS[0], YS[0], ZS[0]]).any() or (hi > [XS[-1], YS[-1], ZS[-1]]).any():
            return False
        sel = (((XS >= lo[0]) & (XS <= hi[0]))[:, None, None] & ((YS >= lo[1]) & (YS <= hi[1]))[None, :, None]
               & ((ZS >= lo[2]) & (ZS <= hi[2]))[None, None, :])
        return bool(ok[sel].all()) and bool((np.abs(MUST_INCLUDE - c) <= half).all())

    centers = [np.array([x, y, z]) for y in (0.0, 0.02, -0.02, 0.04, -0.04)
               for x in np.arange(0.14, 0.40, 0.01) for z in np.arange(0.06, 0.24, 0.01)]
    best = None
    for s in np.arange(SCALE_MAX, SCALE_MIN - 1e-9, -0.01):
        for c in centers:
            if box_ok(c, s * r_half_unit):
                best = (s, c)
                break
        if best:
            break
    if best is None:
        print("조건을 만족하는 상자를 찾지 못함")
        return
    s, c = best
    half = s * r_half_unit

    # 상자 안 무작위 점 IK 재확인
    rng = np.random.default_rng(0)
    pts = c + (rng.random((300, 3)) * 2 - 1) * half
    good = sum(reachable(m, ik, q) for q in pts)

    cfg = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "formula": "robot = robot_center + scale * R @ (haply - haply_center)",
        "scale": round(float(s), 3),
        "R_haply_to_robot": R.astype(int).tolist(),
        "axis_map": {"haply +y (앞)": "robot +x (앞)", "haply +x (오른쪽)": "robot -y (오른쪽)", "haply +z (위)": "robot +z (위)"},
        "haply_center_m": h_center.round(4).tolist(),
        "haply_half_range_m": h_half.round(4).tolist(),
        "robot_center_m": c.round(4).tolist(),
        "robot_box_min_m": (c - half).round(4).tolist(),
        "robot_box_max_m": (c + half).round(4).tolist(),
        "must_include_m": MUST_INCLUDE.tolist(),
        "ik_check": f"{good}/{len(pts)} random points reachable",
        "source": "config/haply_workspace.json " + hw["measured_at"],
        "note": "haply_center는 측정 당시 값. 보정하면 원점이 바뀔 수 있으므로 세션 시작 시 다시 측정하거나 중심을 다시 잡을 것",
    }
    with open(os.path.join(HERE, "config", "teleop_mapping.json"), "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, ensure_ascii=False, indent=2)
    print(json.dumps(cfg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
