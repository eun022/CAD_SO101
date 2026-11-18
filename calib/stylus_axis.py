"""
스타일러스 펜 축 방향 측정 (읽기 전용 - probe_position만)
사용법: python calib/stylus_axis.py

목적: 스타일러스의 "뭉툭한 끝 -> 볼" 방향을 그리퍼 방향(손목 -> 손가락 끝)으로 쓰기 위해
  - 스타일러스 자체 좌표에서 볼 쪽을 가리키는 축(a)
  - 자세 값(orientation) 기준 좌표계에서 사용자 기준 오른쪽/앞/위 방향
을 찾음. 볼이 위 / 앞 / 오른쪽을 가리키는 세 자세를 측정
결과: config/stylus_axis.json
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
POSES = [("up", "볼이 똑바로 위(천장)를 가리키게 세움"),
         ("forward", "볼이 앞(몸에서 멀어지는 쪽)을 수평으로 가리키게 눕힘"),
         ("right", "볼이 오른쪽을 수평으로 가리키게 눕힘")]
CANDIDATES = {f"{s}{k}": sg * np.eye(3)[i] for i, k in enumerate("xyz") for s, sg in (("+", 1), ("-", -1))}


def qmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    with connect(URI) as ws:
        device_id = json.loads(ws.recv(timeout=5))["inverse3"][0]["device_id"]
        probe = json.dumps({"inverse3": [{"device_id": device_id, "commands": {"probe_position": {}}}]})

        def snapshot():
            qs = []
            for _ in range(N_AVG):
                ws.send(probe)
                g = json.loads(ws.recv(timeout=1)).get("wireless_verse_grip") or []
                if not g:
                    raise RuntimeError("스타일러스 데이터가 없습니다. 전원/연결을 확인하세요.")
                o = g[0]["state"]["orientation"]
                qs.append(np.array([o["w"], o["x"], o["y"], o["z"]]))
            qs = [q if np.dot(q, qs[0]) >= 0 else -q for q in qs]
            return np.mean(qs, axis=0)

        Rs = {}
        for key, how in POSES:
            input(f"\n[{key}] {how}. 그 자세로 멈춘 뒤 엔터...")
            Rs[key] = qmat(snapshot())
            print("  기록함")

    # 볼 방향 로컬 축 a 후보마다: 세 자세에서의 방향(자세 기준 좌표)이 서로 직교하고 오른손 좌표계(오른쪽 x 앞 = 위)인지
    scores = {}
    for name, a in CANDIDATES.items():
        u, f, r = (Rs[k] @ a for k in ("up", "forward", "right"))
        ortho = abs(u @ f) + abs(u @ r) + abs(f @ r)
        scores[name] = (ortho, float(np.cross(r, f) @ u), u, f, r)
    best = min((n for n in scores if scores[n][1] > 0), key=lambda n: scores[n][0])
    ortho, hand, u, f, r = scores[best]
    for n, (o, h, *_) in sorted(scores.items(), key=lambda kv: kv[1][0]):
        print(f"  후보 {n}: 직교 오차 {o:.3f}, 오른손 판정 {h:+.2f}")
    result = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ball_axis_local": best,
        "orthogonality_error": round(ortho, 3),
        "handedness": round(hand, 3),
        "frame_up": u.round(4).tolist(), "frame_forward": f.round(4).tolist(), "frame_right": r.round(4).tolist(),
        "note": "볼 방향 = R(q) @ ball_axis_local (자세 기준 좌표). 사용자 기준 위/앞/오른쪽은 frame_* 벡터와 내적해서 구함",
    }
    os.makedirs(os.path.join(HERE, "config"), exist_ok=True)
    with open(os.path.join(HERE, "config", "stylus_axis.json"), "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(f"\n결과: 볼 방향 축 = 스타일러스 {best}, 직교 오차 {ortho:.3f} (0에 가까울수록 좋음, 0.3 넘으면 다시 측정 권장)")
    print("저장: config/stylus_axis.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+C로 중단됨")
    except Exception as e:
        print(f"오류: {e!r}")
    input("\n엔터를 누르면 창이 닫힙니다...")
