"""
NeuralActuator 실시간 추론 서버 (neural_actuator 환경에서 실행, 2026-11-17)
사용법: C:\\Users\\ehfkw\\anaconda3\\envs\\neural_actuator\\python.exe na/server.py [체크포인트 이름, 기본 so101_weight_extended]

- 텔레오퍼레이션(cad2026 환경)과 로컬 UDP(127.0.0.1:47101)로 통신. 두 conda 환경 패키지를 섞지 않으려고 프로세스를 나눔
  (실물 SO-101에서도 같은 서버에 실제 서보 텔레메트리를 보내면 됨)
- 요청: seq(uint32) + 42개 float32 (NeuralActuator SO-101 입력 1프레임, na_sim_telemetry.SimTelemetry.features)
  응답: seq(uint32) + force(3) + gate(1) + tau(6) = float32 10개  (force: 끝 외력 추정, 로봇 베이스 좌표 N, 무게 = -z)
- 특수 요청: seq = 0xFFFFFFFF -> 히스토리 초기화 (텔레오퍼레이션 시작 시)
- infer_actuator._build_force_only의 1스텝 함수를 그대로 사용 (정규화, 8프레임 히스토리 포함), CPU 약 1ms/스텝
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import os
import socket
import struct
import sys
import time

import numpy as np

HERE = ROOT
NA = os.path.join(HERE, "external", "Dynamics-Modeling", "NeuralActuator")
ADDR = ("127.0.0.1", 47101)
RESET = 0xFFFFFFFF


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "so101_weight_extended"
    os.chdir(NA)  # 저장소 코드가 상대 경로(models/, configs/)를 씀
    sys.path.insert(0, NA)
    import jax
    import jax.numpy as jnp
    import yaml
    import evaluate_actuator_so101 as ev
    from infer_actuator import _build_force_only

    config = yaml.safe_load(open("configs/so101_weight.yaml"))
    config["use_ema_params"] = False
    model, params, H, F, norm = ev.load_model(f"checkpoints/{name}.pkl", config)
    _, step_fn = _build_force_only(model, config, norm, "so101")
    zero_hist = jnp.zeros((H, F), dtype=jnp.float32)
    carry = (zero_hist, params)
    carry, _ = step_fn(carry, jnp.zeros(F, dtype=jnp.float32))  # JIT 준비
    carry = (zero_hist, params)
    print(f"NeuralActuator 서버: {name}, 입력 {F}차원 x 히스토리 {H}, {jax.devices()[0].platform}, {ADDR[0]}:{ADDR[1]}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(ADDR)
    n, t_sum, t_rep = 0, 0.0, time.perf_counter()
    while True:
        pkt, addr = sock.recvfrom(4096)
        seq = struct.unpack_from("<I", pkt)[0]
        if seq == RESET:
            carry = (zero_hist, params)
            print("히스토리 초기화")
            continue
        feat = np.frombuffer(pkt, dtype="<f4", offset=4, count=F)
        t0 = time.perf_counter()
        carry, (tau, force, gate, _cond) = step_fn(carry, jnp.asarray(feat))
        force, gate, tau = np.asarray(force), float(gate), np.asarray(tau)
        t_sum += time.perf_counter() - t0
        n += 1
        sock.sendto(struct.pack("<I", seq) + np.r_[force, gate, tau].astype("<f4").tobytes(), addr)
        if time.perf_counter() - t_rep > 5:
            print(f"  {n}회, 평균 추론 {t_sum / n * 1e3:.2f} ms, 최근 힘 {np.round(force, 2)} N, gate {gate:.2f}")
            n, t_sum, t_rep = 0, 0.0, time.perf_counter()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("서버 종료")
