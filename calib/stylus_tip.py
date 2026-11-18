"""
스타일러스 볼 끝 위치 측정 (피벗 보정, 읽기 전용 - probe_position만, 힘 없음)
사용법: python calib/stylus_tip.py [초, 기본 15]

목적: Minverse가 주는 cursor_position은 스타일러스가 달린 지점이고 볼 끝이 아님.
      텔레오퍼레이션에서 "스타일러스 볼 끝 = 그리퍼 끝"으로 쓰려면 커서 -> 볼 끝 벡터(스타일러스 자체 좌표)가 필요
방법: 볼 끝을 한 점에 고정(다른 손가락으로 집거나 책상 위 한 점에 대고)한 채 스타일러스를 여러 방향으로 기울이고 돌림
      tip = cursor + B @ R(q) @ p  (B: 자세 좌표 -> Minverse 축, config/stylus_axis.json)
      -> 모든 샘플에서 tip이 같다는 조건으로 tip(3)과 p(3)를 최소제곱으로 구함
결과: config/stylus_tip.json (p_local_m, 길이, 잔차)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import os
import sys
import time

import numpy as np
from websockets.sync.client import connect

from calib.stylus_axis import CANDIDATES, qmat

URI = "ws://localhost:10001"
HERE = ROOT


def frame_B():
    ax = json.load(open(os.path.join(HERE, "config", "stylus_axis.json"), encoding="utf-8"))
    up = np.array(ax["frame_up"]); up /= np.linalg.norm(up)
    fw = np.array(ax["frame_forward"]); fw -= (fw @ up) * up; fw /= np.linalg.norm(fw)
    return np.vstack([np.cross(fw, up), fw, up]), CANDIDATES[ax["ball_axis_local"]]


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    B, ball_local = frame_B()
    with connect(URI) as ws:
        device_id = json.loads(ws.recv(timeout=5))["inverse3"][0]["device_id"]
        probe = json.dumps({"inverse3": [{"device_id": device_id, "commands": {"probe_position": {}}}]})
        print("볼 끝을 한 점에 고정하고(손가락으로 집거나 책상의 한 점에 대고), 3초 뒤부터 "
              f"{dur:.0f}초 동안 스타일러스를 앞뒤/좌우로 크게 기울이고 돌려 주세요. 볼 끝은 움직이지 않게.")
        for k in (3, 2, 1):
            print(f"  {k}...")
            time.sleep(1)
        print("기록 중...")
        H, Q, t0, last = [], [], time.perf_counter(), 0.0
        while (t := time.perf_counter() - t0) < dur:
            ws.send(probe)
            msg = json.loads(ws.recv(timeout=1))
            inv, g = msg.get("inverse3") or [], msg.get("wireless_verse_grip") or []
            if not inv or not g or t - last < 0.005:  # 약 200Hz로 솎음
                continue
            last = t
            c = inv[0]["state"]["cursor_position"]
            o = g[0]["state"]["orientation"]
            H.append([c["x"], c["y"], c["z"]])
            Q.append(B @ qmat(np.array([o["w"], o["x"], o["y"], o["z"]])))
    H, n = np.array(H), len(H)
    if n < 100:
        raise SystemExit(f"샘플이 너무 적습니다 ({n}개)")
    # cursor_i = tip - M_i p  ->  [I, -M_i] [tip; p] = cursor_i
    A = np.vstack([np.hstack([np.eye(3), -M]) for M in Q])
    x, *_ = np.linalg.lstsq(A, H.reshape(-1), rcond=None)
    tip, p = x[:3], x[3:]
    res = (A @ x - H.reshape(-1)).reshape(-1, 3)
    rms = float(np.sqrt((res ** 2).sum(1).mean()))
    spread = float(np.degrees(np.arccos(np.clip(min((M @ ball_local) @ (Q[0] @ ball_local) for M in Q), -1, 1))))
    L = float(np.linalg.norm(p))
    ang = float(np.degrees(np.arccos(np.clip(p @ ball_local / L, -1, 1))))
    print(f"\n샘플 {n}개, 스타일러스 방향 변화 최대 {spread:.0f}도 (30도 이상 권장)")
    print(f"커서 -> 볼 끝: {L*1e3:.1f} mm, 스타일러스 좌표 p = {np.round(p*1e3, 1)} mm, 볼 축(+y 등)과 {ang:.0f}도 차이")
    print(f"잔차 RMS {rms*1e3:.1f} mm (볼 끝이 얼마나 고정됐었는지, 5mm 이하면 좋음)")
    ok = spread >= 30 and rms < 0.008
    if not ok:
        print("결과가 불안정합니다 -> 더 크게 기울이고, 볼 끝을 확실히 고정해서 다시 측정 권장")
    out = {"measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "p_local_m": p.round(5).tolist(),
           "length_m": round(L, 5), "angle_to_ball_axis_deg": round(ang, 1), "rms_m": round(rms, 5),
           "orientation_spread_deg": round(spread, 1), "samples": n, "ok": ok,
           "note": "볼 끝(Minverse 좌표) = cursor + B @ R(q) @ p_local, B는 stylus_axis.json의 frame_* 로 만든 행렬"}
    with open(os.path.join(HERE, "config", "stylus_tip.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print("저장: config/stylus_tip.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+C로 중단됨")
