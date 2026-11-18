"""
Minverse 이동 범위 측정 + 축 방향 확인 (읽기 전용 - 힘은 보내지 않음, probe_position만 사용)
사용법: python calib/workspace.py [휘젓는 시간 s, 기본 25] [--auto]
        --auto: 엔터 입력 없이 바로 범위만 기록 (축 방향 확인 생략, 이전에 확인한 축 표 사용)

1) 축 방향 확인: 오른쪽 / 앞(몸에서 멀어지는 쪽) / 위로 각각 약 5cm 움직여 어느 축(±x/y/z)인지 기록
2) 범위 측정: 정해진 시간 동안 스타일러스를 갈 수 있는 끝까지 천천히 휘저음 -> 축별 최소/최대, 1~99% 범위, 중심
결과: config/haply_workspace.json (요약), data/haply_workspace_<시각>.csv (원시 위치)
- 보정할 때마다 좌표 원점이 바뀔 수 있으므로 매핑에는 범위 크기와 중심만 사용
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv
import json
import os
import sys
import time

import numpy as np
from websockets.sync.client import connect

URI = "ws://localhost:10001"
HERE = ROOT
RATE_HZ = 200
N_AVG = 200
MIN_MOVE_MM = 20
STEPS = [("오른쪽", "right"), ("앞 (몸에서 멀어지는 쪽)", "forward"), ("위", "up")]


AXES_KNOWN = {"right": "+x", "forward": "+y", "up": "+z"}  # 2026-10-27 haply_axis_probe.py 결과


def main():
    auto = "--auto" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--auto"]
    sweep_s = float(args[0]) if args else 25.0
    with connect(URI) as ws:
        first = json.loads(ws.recv(timeout=5))["inverse3"][0]
        device_id = first["device_id"]
        probe = json.dumps({"inverse3": [{"device_id": device_id, "commands": {"probe_position": {}}}]})

        def read():
            ws.send(probe)
            s = json.loads(ws.recv(timeout=1))["inverse3"][0]
            return np.array([s["state"]["cursor_position"][k] for k in "xyz"]), s["status"]["calibrated"]

        def snapshot():
            return np.mean([read()[0] for _ in range(N_AVG)], axis=0)

        if not first["status"]["calibrated"]:
            print("주의: 장치가 보정되지 않은 상태입니다 (calibrated=false). 보정 후 다시 측정하는 것을 권장합니다.")

        # 1) 축 방향 확인
        axes = dict(AXES_KNOWN) if auto else {}
        if not auto:
            print("\n=== 1) 축 방향 확인 ===")
        for label, key in ([] if auto else STEPS):
            input(f"\n[{label}] 스타일러스를 가운데쯤에 두고 멈춘 뒤 엔터...")
            p0 = snapshot()
            input(f"  -> {label}으로 약 5cm 움직이고 멈춘 뒤 엔터...")
            dmm = 1000 * (snapshot() - p0)
            i = int(np.argmax(np.abs(dmm)))
            ok = abs(dmm[i]) >= MIN_MOVE_MM
            axes[key] = (("+" if dmm[i] > 0 else "-") + "xyz"[i]) if ok else None
            print(f"  변화(mm): x {dmm[0]:+.1f}, y {dmm[1]:+.1f}, z {dmm[2]:+.1f}  -> {axes[key] or '판별 불가 (2cm 미만)'}")

        # 2) 범위 측정
        print("\n=== 2) 범위 측정 ===")
        msg = (f"{sweep_s:.0f}초 동안 기록합니다.\n"
               "스타일러스를 좌우/앞뒤/위아래로 갈 수 있는 끝까지 천천히 휘저어 주세요 (무리하게 밀지 말 것)")
        print("\n" + msg) if auto else input(f"\n엔터를 누르면 {msg}...")
        pts, t0 = [], time.perf_counter()
        next_t, last_print = t0, -1
        while (t := time.perf_counter() - t0) < sweep_s:
            pos, cal = read()
            pts.append((t, *pos, cal))
            if int(t) != last_print:
                last_print = int(t)
                print(f"  남은 시간 {sweep_s - t:4.0f}s", end="\r", flush=True)
            next_t += 1.0 / RATE_HZ
            time.sleep(max(0.0, next_t - time.perf_counter()))
        print()

    P = np.array([r[1:4] for r in pts])
    lo, hi = P.min(axis=0), P.max(axis=0)
    p1, p99 = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
    summary = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calibrated": bool(all(r[4] for r in pts)),
        "axes_user_to_device": axes,
        "samples": len(P),
        "min_m": lo.round(4).tolist(), "max_m": hi.round(4).tolist(),
        "p1_m": p1.round(4).tolist(), "p99_m": p99.round(4).tolist(),
        "range_p1_p99_m": (p99 - p1).round(4).tolist(),
        "center_p1_p99_m": ((p1 + p99) / 2).round(4).tolist(),
        "note": "보정할 때마다 원점이 바뀔 수 있음 -> 매핑에는 range/center만 사용하고, 세션 시작 시 center를 다시 잡을 것",
    }
    os.makedirs(os.path.join(HERE, "config"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "config", "haply_workspace.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    raw = os.path.join(HERE, "data", f"haply_workspace_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(raw, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["t", "x", "y", "z", "calibrated"])
        w.writerows(pts)

    print("\n=== 결과 ===")
    print(f"축 방향: 오른쪽 -> {axes['right']}, 앞 -> {axes['forward']}, 위 -> {axes['up']}")
    for k, a, b, c in zip("xyz", p1, p99, p99 - p1):
        print(f"  {k}: {1000 * a:+7.1f} ~ {1000 * b:+7.1f} mm  (범위 {1000 * c:6.1f} mm, 1~99%)")
    print("저장: config/haply_workspace.json, " + os.path.relpath(raw, HERE))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+C로 중단됨")
    except TimeoutError:
        print("접속은 됐지만 데이터가 오지 않습니다. Minverse 연결/보정 상태를 확인하세요.")
    except OSError:
        print("접속 실패: Haply 허브가 켜져 있는지, 포트가 10001이 맞는지 확인하세요.")
    if "--auto" not in sys.argv:
        input("\n엔터를 누르면 창이 닫힙니다...")
