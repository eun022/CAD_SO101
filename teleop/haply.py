"""
힘 부호 확인 테스트 (수평축 x/y만, 0 -> F_PEAK 램프 -> 유지 -> 램프 -> 0)
사용법: python teleop/haply.py <x|y> <+|-> [힘 N, 기본 0.3]    예) python teleop/haply.py x +

- 1kHz 루프: perf_counter 기반 하이브리드 대기 (짧은 time.sleep + spin)
- 힘 프로파일: 0N 대기 200ms -> 100ms 램프업 -> 1s 유지 -> 100ms 램프다운 -> 0N 대기 200ms
- 안전
  - 성분별 F_MAX 클램프
  - 램프 시작 위치에서 MAX_DISP_M(2cm) 넘게 움직이면 즉시 0N으로 중단
  - Ctrl+C/예외 시에도 finally에서 0N을 반복 송신 후 종료
- z축은 중력 보상의 영향을 받으므로 x, y 부호를 확인한 뒤 따로 진행
- 출력: 틱 간격 통계, 명령값 vs 보고값 틱별 비교, 위치 변화, CSV(logs/)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv
import json
import os
import statistics
import sys
import time

from websockets.sync.client import connect

URI = "ws://localhost:10001"
RATE_HZ = 1000
PERIOD = 1.0 / RATE_HZ
SPIN_S = 0.0005  # 목표 시각 0.5ms 전까지는 sleep, 그 이후는 spin

F_DEFAULT = 0.3  # N
F_MAX = 1.0      # N, 성분별 클램프 (어떤 경우에도 이 값을 넘지 않음)
MAX_DISP_M = 0.02
PRE_S, RAMP_S, HOLD_S, POST_S = 0.2, 0.1, 1.0, 0.2
TOTAL_S = PRE_S + RAMP_S + HOLD_S + RAMP_S + POST_S
ZERO_REPEAT = 50  # 종료 시 0N 반복 송신 횟수
LOG_DIR = os.path.join(ROOT, "logs")


def profile(t, peak):
    """경과 시간 t(s) -> 힘 크기(N). 사다리꼴 램프."""
    t -= PRE_S
    if t <= 0:
        return 0.0
    if t < RAMP_S:
        return peak * t / RAMP_S
    t -= RAMP_S
    if t < HOLD_S:
        return peak
    t -= HOLD_S
    if t < RAMP_S:
        return peak * (1.0 - t / RAMP_S)
    return 0.0


def clamp(v):
    return max(-F_MAX, min(F_MAX, v))


def force_msg(device_id, f):
    f = {k: clamp(v) for k, v in f.items()}
    return json.dumps({"inverse3": [{"device_id": device_id,
                                     "commands": {"set_cursor_force": {"vector": f}}}]})


def handshake(ws, first, profile_name):
    """Hub 데모와 같은 첫 송신: 세션 프로필 + basis/preset(현재 값 그대로) + 0N.
    이게 없으면 status.in_use가 False로 남고 힘 명령이 장치에 출력되지 않음 (2026-10-27 확인).
    first: 접속 직후 첫 프레임의 inverse3[0]. 반환: 응답의 inverse3[0]"""
    cfg = first["config"]
    ws.send(json.dumps({"session": {"configure": {"profile": {"name": profile_name}}},
                        "inverse3": [{"device_id": first["device_id"],
                                      "configure": {"basis": {"permutation": cfg["basis"]["permutation"]},
                                                    "preset": {"preset": cfg["preset"]}},
                                      "commands": {"set_cursor_force": {"vector": {"x": 0.0, "y": 0.0, "z": 0.0}}}}]}))
    return json.loads(ws.recv(timeout=1))["inverse3"][0]


def wait_until(target):
    while True:
        rem = target - time.perf_counter()
        if rem <= 0:
            return
        if rem > SPIN_S:
            time.sleep(rem - SPIN_S)


def stats_us(xs):
    xs_us = [x * 1e6 for x in xs]
    q = statistics.quantiles(xs_us, n=20)
    return f"중앙값 {statistics.median(xs_us):.0f}, 95% {q[-1]:.0f}, 최대 {max(xs_us):.0f}"


def parse_args():
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in ("x", "y") or sys.argv[2] not in ("+", "-"):
        return None
    peak = float(sys.argv[3]) if len(sys.argv) == 4 else F_DEFAULT
    if not 0.0 < peak <= F_MAX:
        return None
    return sys.argv[1], (1.0 if sys.argv[2] == "+" else -1.0), peak


def main():
    args = parse_args()
    if args is None:
        print(__doc__)
        return
    axis, sign, peak = args

    rows = []  # (t, dt, 명령 xyz, 보고 xyz, 위치 xyz, mode, control_mode)
    ticks, aborted, p_ref = [], None, None
    with connect(URI) as ws:
        first = json.loads(ws.recv(timeout=5))["inverse3"][0]
        device_id = first["device_id"]
        zero = force_msg(device_id, {"x": 0.0, "y": 0.0, "z": 0.0})

        input(f"\n{axis}{sys.argv[2]} 방향으로 최대 {peak}N을 {HOLD_S}s 출력합니다.\n"
              f"스타일러스를 작업공간 가운데 높이까지 들어 올려, 손가락으로 아주 살짝만 받친 채 엔터를 누르세요.\n"
              f"{MAX_DISP_M * 100:.0f}cm 넘게 움직이면 자동으로 0N이 됩니다 (중단: Ctrl+C)...")
        handshake(ws, first, "kr.ac.hanyang.cad:force-test")

        try:
            t0 = time.perf_counter()
            next_t, last_t = t0, None
            while True:
                now = time.perf_counter()
                t = now - t0
                if t >= TOTAL_S:
                    break
                dt = 0.0 if last_t is None else now - last_t
                if last_t is not None:
                    ticks.append(dt)
                last_t = now

                f = {"x": 0.0, "y": 0.0, "z": 0.0}
                f[axis] = sign * profile(t, peak)
                ws.send(force_msg(device_id, f))
                state = json.loads(ws.recv(timeout=1))["inverse3"][0]["state"]
                rep, pos = state["current_cursor_force"], state["cursor_position"]
                rows.append((t, dt, f, rep, pos, state["mode"], state["control_mode"]))

                if t >= PRE_S:
                    if p_ref is None:
                        p_ref = pos  # 램프 시작 위치 기준
                    disp = sum((pos[k] - p_ref[k]) ** 2 for k in "xyz") ** 0.5
                    if disp > MAX_DISP_M:
                        aborted = f"이동 거리 {disp * 100:.1f}cm 초과 (t={t:.3f}s)"
                        break

                next_t += PERIOD
                if time.perf_counter() > next_t + PERIOD:  # 크게 밀리면 따라잡기 대신 재기준
                    next_t = time.perf_counter()
                wait_until(next_t)
        finally:
            # Ctrl+C, 예외, 이동 거리 초과, 정상 종료 모두 0N으로 마무리
            for _ in range(ZERO_REPEAT):
                try:
                    ws.send(zero)
                    ws.recv(timeout=0.1)
                except Exception:
                    break
            print("\n종료: 0N 송신 완료" + (f"  [자동 중단: {aborted}]" if aborted else ""))
            if rows:
                os.makedirs(LOG_DIR, exist_ok=True)
                path = os.path.join(LOG_DIR, f"force_{axis}{sys.argv[2].replace('+', 'p').replace('-', 'm')}"
                                             f"_{peak:.1f}N_{time.strftime('%Y%m%d_%H%M%S')}.csv")
                with open(path, "w", newline="") as fp:
                    w = csv.writer(fp)
                    w.writerow(["t", "dt", "cmd_x", "cmd_y", "cmd_z", "rep_x", "rep_y", "rep_z",
                                "pos_x", "pos_y", "pos_z", "mode", "control_mode"])
                    for t, dt, f, rep, pos, mode, cmode in rows:
                        w.writerow([f"{t:.6f}", f"{dt:.6f}", *(f[k] for k in "xyz"),
                                    *(rep[k] for k in "xyz"), *(pos[k] for k in "xyz"), mode, cmode])
                print(f"CSV 저장: {path}")

    if len(ticks) > 2:
        late = sum(dt > 2 * PERIOD for dt in ticks)
        print(f"\n틱 간격(us): {stats_us(ticks)}  (목표 {PERIOD * 1e6:.0f}, 2ms 초과 {late}/{len(ticks)})")

    # 명령값 vs 보고값: 같은 틱(lag 0)과 한 틱 전 명령(lag 1) 중 어느 쪽과 일치하는지
    ramp = [i for i, r in enumerate(rows) if i > 0 and PRE_S <= r[0] < PRE_S + RAMP_S]
    if ramp:
        lag0 = max(abs(rows[i][3][axis] - rows[i][2][axis]) for i in ramp)
        lag1 = max(abs(rows[i][3][axis] - rows[i - 1][2][axis]) for i in ramp)
        print(f"램프업 구간 |보고-명령| 최대: 같은 틱 {lag0:.4f}N, 한 틱 전 {lag1:.4f}N  ({len(ramp)}틱)")
        print("  -> 둘 중 하나가 ~0이면 보고값은 명령을 그대로 돌려주는 값 (실측 아님)")

    if p_ref is not None:
        after = [r[4] for r in rows if r[0] >= PRE_S]
        peak_d = {k: max((p[k] - p_ref[k] for p in after), key=abs) * 1000 for k in "xyz"}
        print("램프 시작 대비 최대 위치 변화(mm): " + ", ".join(f"{k} {peak_d[k]:+.1f}" for k in "xyz"))
        print(f"  -> {axis}{sys.argv[2]} 힘 명령에 장치가 움직인 방향: "
              f"{'+' if peak_d[axis] > 0 else '-'}{axis} 성분 {peak_d[axis]:+.1f}mm")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+C로 중단됨")
    except TimeoutError:
        print("접속은 됐지만 데이터가 오지 않습니다. Minverse 연결/보정 상태를 확인하세요.")
    except OSError:
        print("접속 실패: Haply 허브가 켜져 있는지, 포트가 10001이 맞는지 확인하세요.")
