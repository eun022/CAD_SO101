"""
스타일러스(VerseGrip) 자세 축 확인 (읽기 전용 - 힘은 보내지 않음, probe_position만 사용)
사용법: python calib/stylus_ori.py

평소처럼 스타일러스를 잡은 자세를 기준으로, 아래 세 동작을 각각 약 30~45도 해서 어느 축 회전인지 기록
  1) 숙이기(pitch): 펜 끝을 아래로 숙임
  2) 비틀기(roll): 펜을 자기 축을 중심으로 시계 방향으로 비틂 (위에서 보는 손 기준)
  3) 돌리기(yaw): 펜 끝을 왼쪽으로 돌림
결과: config/stylus_orientation.json  (기준 자세 대비 상대 회전의 축을 스타일러스 자체 좌표 기준으로 기록)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import os
import time

import numpy as np
from websockets.sync.client import connect

URI = "ws://localhost:10001"
HERE = ROOT
N_AVG = 100
STEPS = [("pitch", "숙이기: 펜 끝을 아래로 약 30~45도 숙임"),
         ("roll", "비틀기: 펜을 자기 축을 중심으로 시계 방향으로 약 30~45도 비틂"),
         ("yaw", "돌리기: 펜 끝을 왼쪽으로 약 30~45도 돌림")]


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotvec(q):
    """단위 쿼터니언 -> 회전 벡터(축 * 각도, rad)"""
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    return np.zeros(3) if s < 1e-9 else q[1:] / s * 2 * np.arctan2(s, q[0])


def main():
    with connect(URI) as ws:
        device_id = json.loads(ws.recv(timeout=5))["inverse3"][0]["device_id"]
        probe = json.dumps({"inverse3": [{"device_id": device_id, "commands": {"probe_position": {}}}]})

        def read_q():
            ws.send(probe)
            g = json.loads(ws.recv(timeout=1)).get("wireless_verse_grip") or []
            if not g:
                raise RuntimeError("스타일러스(wireless_verse_grip) 데이터가 없습니다. 스타일러스 전원/연결을 확인하세요.")
            o = g[0]["state"]["orientation"]
            return np.array([o["w"], o["x"], o["y"], o["z"]])

        def snapshot():
            qs = [read_q() for _ in range(N_AVG)]
            ref = qs[0]
            qs = [q if np.dot(q, ref) >= 0 else -q for q in qs]  # 부호 맞춰 평균
            q = np.mean(qs, axis=0)
            return q / np.linalg.norm(q)

        result = {}
        for key, how in STEPS:
            input(f"\n[{key}] 스타일러스를 평소처럼 잡은 기준 자세로 두고 멈춘 뒤 엔터...")
            q0 = snapshot()
            input(f"  -> {how}, 그 상태로 멈춘 뒤 엔터...")
            q1 = snapshot()
            rv = rotvec(qmul(qconj(q0), q1))  # 기준 자세 대비 회전, 스타일러스 자체 좌표 기준
            ang = np.degrees(np.linalg.norm(rv))
            i = int(np.argmax(np.abs(rv)))
            axis = ("+" if rv[i] > 0 else "-") + "xyz"[i]
            print(f"  회전 {ang:5.1f}도, 축 성분(deg) x {np.degrees(rv[0]):+6.1f}, y {np.degrees(rv[1]):+6.1f}, "
                  f"z {np.degrees(rv[2]):+6.1f}  -> {axis}" + ("" if ang > 15 else "  (회전이 작음, 다시 측정 권장)"))
            result[key] = {"axis": axis, "angle_deg": round(float(ang), 1), "rotvec_deg": np.degrees(rv).round(1).tolist()}

    result["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    result["note"] = "스타일러스 자체 좌표 기준 상대 회전 축. 텔레오퍼레이션에서는 따라가기 시작 순간 자세를 기준으로 삼음"
    os.makedirs(os.path.join(HERE, "config"), exist_ok=True)
    with open(os.path.join(HERE, "config", "stylus_orientation.json"), "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(f"\n결과: 숙이기 {result['pitch']['axis']}, 비틀기 {result['roll']['axis']}, 돌리기 {result['yaw']['axis']}")
    print("저장: config/stylus_orientation.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+C로 중단됨")
    except Exception as e:
        print(f"오류: {e!r}")
    input("\n엔터를 누르면 창이 닫힙니다...")
