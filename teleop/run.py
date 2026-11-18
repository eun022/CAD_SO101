"""
(4)-③ Minverse로 MuJoCo SO-101 조작 (읽기 전용 - Minverse에 힘/위치 명령을 보내지 않음, probe_position만 사용)
사용법: python teleop/run.py [--scale 배율] [--level soft|medium|hard] [--close mm] [--no-log]
  --scale: 기본 config/teleop_mapping.json의 scale_user, 없으면 scale
  --level: FEM 큐브 강성 (기본 medium), --close: 잡을 때 처음 닿은 뒤 더 닫을 거리 (기본 2.0mm)

- FEM 큐브 (2026-11-17, cube_pkg): 윗면/양옆 패드가 FEM 곡선대로 힘을 냄. 매 스텝 mj_step 전에 CubeForces.apply()
  - 콘솔에 0.25초마다 윗면 힘, 잡는 힘(닫힘 mm, 한쪽당 N) 출력 (힘이 있을 때만)
  - 1kHz 기록 -> data/teleop_cube_<시각>.csv (창을 닫을 때 저장, --no-log면 저장 안 함)
- 버튼 b 잡기: 누르는 동안 천천히 닫다가 큐브 닫힘이 --close에 도달하면 그 자리 유지 (패드 한계 3mm를 넘어 FEM 곡선을 벗어나지 않게).
  큐브가 턱 사이에 없으면 끝까지 닫힘. 놓으면 바로 열림

- 매핑: config/teleop_mapping.json  (robot = robot_center + scale * R @ (haply - haply_center))
  - 로봇이 닿을 수 없는 곳은 IK가 가장 가까운 자세로 멈춤 (배율이 커서 작업 상자가 도달 범위를 넘을 수 있음)
  - (2026-11-17) 위치 = 스타일러스 **볼 끝** (config/stylus_tip.json, haply_stylus_tip_probe.py로 측정. 없으면 커서 위치)
  - (2026-11-17) **방향 우선 IK**: 숙임 각도를 먼저 맞추고, 끝 위치가 POS_TOL(3cm) 넘게 벗어나면 방향을 양보
  - 관절 명령은 한계에서 JOINT_MARGIN 안쪽, 관절 속도 JOINT_RATE 이하로 제한 (실물에도 그대로 쓰기 위해)
- 로봇은 버튼 없이 항상 따라감 (2026-10-27 사용자 요청. 이전에는 버튼 a를 누를 때만)
- 버튼 b: 누르고 있는 동안 그리퍼 닫기, 놓으면 열기
- 버튼 a: 비틀기(roll) 기준을 지금 자세로 다시 잡기 (눌렀을 때 한 번)
- 버튼 c: 상자(cube)를 처음 자리(블록 위)로 되돌리기 (눌렀을 때 한 번)
- 스타일러스 방향 = 그리퍼 방향 (2026-10-27): 볼 = 그리퍼 끝, 뭉툭한 끝 = 손목 쪽
  - config/stylus_axis.json: 볼 방향 로컬 축 + 자세 좌표계의 오른쪽/앞/위 -> 스타일러스 방향을 로봇 좌표로 옮김
  - 로봇 팔 평면 안에서 그 방향의 각도 theta (팔이 뻗은 수평 방향 = 0, 위 = +) 를 그리퍼가 따름
    : 그리퍼 축 각도 = -(shoulder_lift + elbow_flex + wrist_flex) 관계 (시뮬레이션에서 정확히 확인) -> pitch_target = -theta
  - 비틀기(roll)는 상대값: 시작 순간(또는 a) 자세 대비 펜 축 회전 -> 손목 회전 (wrist_roll)
  - 좌우 방향(yaw)은 반영 못 함 (팔 관절 5개라 그리퍼는 항상 팔 평면 안을 가리킴)
  - stylus_axis.json이 없으면 예전 방식(숙이기도 상대값, stylus_orientation.json)
- 공 마커 = 매핑된 목표점 (초록)
- 시작 순간 목표가 멀리 있어도 튀지 않게 IK 목표점 이동 속도를 MAX_TARGET_SPEED로 제한
- 서비스 응답에 inverse3가 비어 오는 프레임은 건너뜀 (2026-10-27 한 번 발생해 스크립트가 멈춘 적 있음)
- 시뮬레이션 1kHz(timestep 0.001) 실시간, IK 100Hz, 화면 약 60Hz. 창을 닫으면 종료
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

import mujoco
import mujoco.viewer
import numpy as np
from websockets.sync.client import connect

from grasp.grip_control import GripController, JawGeometry
from teleop.haply import handshake
from calib.stylus_ori import qconj, qmul, rotvec
from calib.stylus_axis import CANDIDATES, qmat
from na.telemetry import DATA_DT, SimTelemetry
from teleop.ik import XS, YS, ZS, PosIK
from teleop.force import contact_force_between
from teleop.force import F_HARD_MAX, ContactRenderer, ForceRenderer, NAClient, force_msg, gripper_contact_force

HERE = ROOT
sys.path.insert(0, os.path.join(HERE, "cube_pkg"))
from cube_force import CubeForces  # noqa: E402  (받은 패키지, 수정하지 않음)

SCENE = os.path.join(HERE, "models", "so101_teleop_scene.xml")
SCENE3 = os.path.join(HERE, "models", "so101_teleop3_scene.xml")  # 강성 3종 큐브 좌/중/우 (2026-11-18)
CUBE_PARAMS = os.path.join(HERE, "models", "cubes_teleop", "cube_params.json")
GRIP_RATE = 0.5  # rad/s, b를 누르는 동안 그리퍼 닫는 속도 (닫힘 목표 0.5mm 안쪽에서는 비례해서 줄어듦)
# 손목 각도 목표 변화 속도 제한 (2026-11-17): 시작 순간 스타일러스 각도가 한 번에 들어가면(1.45 -> 0 rad)
# 팔이 자세를 바꾸는 동안 끝이 최대 126mm 출렁여 큐브를 침 (시뮬 재현). 1.5 rad/s 제한 시 12mm
PITCH_RATE, ROLL_RATE = 1.5, 3.0  # rad/s
# 방향 우선 IK (2026-11-17 사용자 결정: 방향 우선, 단 로봇이 낼 수 없는 값은 내보내지 않음)
#  - SO-101은 위치마다 낼 수 있는 숙임 각도 범위가 좁음 (예: z 14cm 이상에서는 똑바로 아래를 못 봄)
#  - 숙임 각도를 맞추는 쪽(가중치 큼)부터 풀고, 끝 위치가 목표에서 POS_TOL 넘게 벗어나면 가중치를 줄여 방향을 양보
#    (방향만 고집하면 위치가 최대 20~28cm 벗어나 엉뚱한 곳으로 감 - 시뮬 확인)
POS_TOL = 0.03  # m
W_PITCH_LIST = (2.0, 0.2, 0.02)  # m/rad, 앞에서부터 시도 (한 번에 약 1ms. 5단계면 최악 4.8ms라 3단계로 줄임)
# 관절 명령 보호 (실물 SO-101에도 그대로 쓸 수 있게): 한계에서 여유를 두고, 관절별 속도 제한
JOINT_MARGIN = 0.05  # rad
JOINT_RATE = 3.0  # rad/s (IK 해는 10ms에 0.068rad = 6.8rad/s까지 튐)
# 힘 피드백 (2026-11-17): --force-src none(기본, 읽기 전용) / sim(MuJoCo 그리퍼 접촉력 정답) / na(NeuralActuator 추정)
FORCE_SCALE_DEFAULT = 0.3  # 로봇 힘 1N -> Minverse 0.3N (처음엔 약하게)
HAND_SPEED_CUT = 1.0  # m/s, Minverse 커서가 이보다 빠르면 힘 0 (비정상 움직임 보호)
CONTACT_K_DEFAULT = 300.0  # N/m (Minverse 기준): 손이 표면 안쪽으로 1mm 더 들어가면 0.3N
CONTACT_B_DEFAULT = 3.0  # N·s/m, 접촉 중 감쇠 (튕김 에너지 흡수)
MAPPING = os.path.join(HERE, "config", "teleop_mapping.json")
STYLUS_TIP = os.path.join(HERE, "config", "stylus_tip.json")  # 커서 -> 볼 끝 (calib/stylus_tip.py)
STYLUS = os.path.join(HERE, "config", "stylus_orientation.json")
STYLUS_AXIS = os.path.join(HERE, "config", "stylus_axis.json")
URI = "ws://localhost:10001"
MAX_TARGET_SPEED = 0.3  # m/s, IK 목표점 이동 속도 상한
# 큐브 패드가 몸통에서 떨어져 나가는 문제 (2026-11-18, 사용자 "잡으면 큐브가 잘려버림")
#  패드(3g)는 몸통에 미끄럼 관절(0~3mm)로 붙어 있는데, 한계가 MuJoCo 기본값(무른 제약 + 가벼운 질량)이라
#  잡고 흔들면 바깥으로 최대 22mm, 세게 누르면 안쪽으로 7.5mm까지 한계를 넘어감 (teleop_cube_20261118_145651.csv)
#  -> 불러온 뒤 한계 제약만 단단하게 + 바깥쪽(깊이 < 0)에 되돌림 스프링. FEM 힘 곡선(0~3mm)은 그대로
PAD_FIX = True
PAD_LIMIT_SOLREF = (0.002, 1.0)  # 시간 상수 = timestep x 2 (MuJoCo가 허용하는 가장 단단한 값)
PAD_LIMIT_SOLIMP = (0.99, 0.999, 0.001)
PAD_OUT_K, PAD_OUT_B = 2000.0, 2.0  # N/m, N·s/m
# 그리퍼 숙임 각도 범위 (0 = 수평 앞, 90° = 아래, - = 위로 듦). 스타일러스 볼을 위로 세우면(-80° 이하) IK가 팔을 앞으로
# 엎고 관절 3개를 한계까지 접은 자세로 넘어가 큐브를 쓸고 갇힘 (2026-11-18) -> 이 범위 밖 방향은 내보내지 않음
PITCH_MIN, PITCH_MAX = np.radians(-45.0), np.radians(130.0)
IK_SEED = np.array([0.0, 0.0, 0.5, 1.0])  # 갇혔을 때 다시 풀 기본 자세 (so101_reach_map 도달 지도와 같은 시작점)
FIXED_JAW_TIP = np.array([-0.0095, -0.0002, -0.1045])  # gripper 바디 좌표, 고정 턱 메시 끝 (원래 gripperframe은 0.012, -0.0002, -0.0981)
WEIGHT_GAIN_DEFAULT = 5.0
WEIGHT_HZ = 2.0    # 무게 신호 저역통과 (처음 10Hz는 손 떨림 관성이 그대로 들어와 덜덜)
WEIGHT_SLEW = 3.0  # N/s, Minverse로 보내는 무게 힘 변화 속도 상한


def cube_supported(m, d, cube_bodies, robot_bodies):
    """큐브(몸통·패드)가 로봇이 아닌 것(블록, 바닥)에 닿아 있는지"""
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in cube_bodies) != (b2 in cube_bodies):
            other = b2 if b1 in cube_bodies else b1
            if other not in robot_bodies:
                return True
    return False
REACH_DB = 0.005  # m, 방향을 다 양보한 IK가 이보다 더 못 닿으면 '갈 수 없는 곳' 벽 힘 (2026-11-18)
GRIP_OPEN, GRIP_CLOSED = 1.0, -0.17  # gripper 관절 (rad): 1.0 = 턱 사이 약 91mm, -0.17 = 약 4mm
REACH_LO, REACH_HI = np.array([XS[0], YS[0], ZS[0]]), np.array([XS[-1], YS[-1], ZS[-1]])
# 위쪽 상한: 도달 지도 격자(ZS)가 0.24에서 끝나서 생긴 인위적인 천장이었음. 실제로는 0.36m까지 도달 가능 (2026-11-18 확인)
# -> 이 천장 때문에 손을 조금만 들어도 벽 힘 2N이 걸렸음 (teleop_cube_20261118_151245.csv, 힘이 걸린 시간의 94%)
REACH_HI = REACH_HI.copy()
REACH_HI[2] = 0.36
GREEN = [0.1, 0.9, 0.2, 0.8]


def axis_sign(s):
    """'+y' -> (1, +1.0)"""
    return "xyz".index(s[1]), (1.0 if s[0] == "+" else -1.0)


# 1kHz 기록: 매 스텝 파일에 바로 씀 (메모리에 쌓지 않음). 종료 시 save_log가 닫음
# 패드 힘 합(Fx/Fy/Fz_world)은 패드가 큐브 바깥쪽(법선)으로 미는 힘 = 그리퍼가 받는 힘의 합
LOG_COLS = ["t", "goal_x", "goal_y", "goal_z", "tip_x", "tip_y", "tip_z", "cube_x", "cube_y", "cube_z",
            "top_mm", "top_N", "xneg_mm", "xneg_N", "xpos_mm", "xpos_N", "closure_mm", "grip_N",
            "Fx_world", "Fy_world", "Fz_world", "grip_btn",
            # 방향 확인용 (2026-11-17): 손목 각도 합 (원하는 값 / 속도 제한 후 / 실제), 스타일러스 볼 방향(로봇 좌표),
            # 팔 평면 밖으로 벗어난 각도(deg, SO-101은 반영 불가), 실제 그리퍼 축(로봇 좌표)
            "pitch_des", "pitch_cmd", "pitch_act", "ball_x", "ball_y", "ball_z", "out_of_plane_deg",
            "grip_ax_x", "grip_ax_y", "grip_ax_z",
            "w_pitch",  # 방향 우선 IK에서 실제로 쓴 가중치 (2.0 = 방향 그대로, 작을수록 위치 때문에 방향을 양보)
            # 힘 (2026-11-17): MuJoCo 정답(그리퍼가 받는 접촉력, 로봇 좌표 N), NeuralActuator 추정(기준 뺀 값 / 원래 값), 접촉 확률,
            # Minverse로 보낸 힘(Minverse 좌표 N, --force-src none이면 0)
            "fsim_x", "fsim_y", "fsim_z", "fna_x", "fna_y", "fna_z", "fna_raw_x", "fna_raw_y", "fna_raw_z", "na_gate",
            "fcmd_x", "fcmd_y", "fcmd_z",
            # 로컬 접촉 모델: 손 목표가 표면 안쪽으로 들어간 깊이(로봇 기준 mm), 접촉 상태(히스테리시스 적용)
            "contact_depth_mm", "in_contact",
            # 갈 수 없는 곳 벽 (2026-11-18): 손이 원하는 로봇 목표(작업 상자로 자르기 전), 벽 안으로 들어간 거리(로봇 mm)
            "hand_x", "hand_y", "hand_z", "wall_mm",
            # 강성 인지 파지 (2026-11-18): 상태(0 idle, 1 closing, 2 approach, 3 hold), 추정 강성(N/m), 목표 힘, 측정 힘(노이즈 평균)
            "grip_state", "k_est", "k_std", "f_target", "f_meas",
            # 무게 렌더링: 그리퍼가 큐브에서 받는 힘 z (걸러진 값, 로봇 N), Minverse로 보낸 무게 힘 z (Minverse N)
            "load_z", "fweight_z",
            "cube_id"]  # 지금 다루는(가장 가까운) 큐브: 0 soft, 1 medium, 2 hard (cube_x/y/z, 패드 열도 이 큐브 기준)
CUBE_ID = {"soft": 0, "medium": 1, "hard": 2}
LOG = {"fh": None, "path": None, "rows": 0}


def open_log():
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    LOG["path"] = os.path.join(HERE, "data", time.strftime("teleop_cube_%Y%m%d_%H%M%S.csv"))
    LOG["fh"] = open(LOG["path"], "w", encoding="utf-8", buffering=1 << 20)
    LOG["fh"].write(",".join(LOG_COLS) + "\n")


def write_log(row):
    LOG["fh"].write(",".join(f"{x:.6g}" for x in row) + "\n")
    LOG["rows"] += 1


def save_log():
    if LOG["fh"] is None:
        return
    LOG["fh"].close()
    LOG["fh"] = None
    print(f"기록 저장: {LOG['path']} ({LOG['rows']}행 = {LOG['rows'] / 1000:.1f}초)")
    try:  # 그래프 자동 생성 (같은 이름 _force.png, _haply.png)
        from teleop import plot as plot_na_force
        plot_na_force.teleop(LOG["path"])
    except Exception as e:  # 그래프 실패가 기록 저장을 막지 않게
        print(f"그래프 생성 실패 (python teleop/plot.py teleop 으로 다시 시도): {e!r}")


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def load_scene(level):
    """큐브 강성 선택: 장면의 cube_medium include를 바꾼 임시 장면을 models/에 써서 절대 경로로 불러옴"""
    if level == "medium":
        return mujoco.MjModel.from_xml_path(SCENE)
    text = open(SCENE, encoding="utf-8").read().replace("cube_medium.xml", f"cube_{level}.xml")
    path = os.path.join(HERE, "models", f"_teleop_scene_{level}.xml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return mujoco.MjModel.from_xml_path(path)


def main():
    cfg = json.load(open(MAPPING, encoding="utf-8"))
    s = arg("--scale", cfg.get("scale_user", cfg["scale"]), float)
    level = arg("--level", "medium")
    if level not in ("soft", "medium", "hard"):
        raise SystemExit("--level은 soft / medium / hard 중 하나")
    # 큐브 개수 (2026-11-18): 3 = 강성 3종을 좌(soft) 중(medium) 우(hard)에 (기본), 1 = 예전처럼 --level 하나만 가운데
    n_cubes = arg("--cubes", 3, int)
    if n_cubes not in (1, 3):
        raise SystemExit("--cubes는 1 또는 3")
    close_mm = arg("--close", 2.0, float)
    # 잡기 방식 (2026-11-18): stiff = 강성 인지 파지 (grasp/grip_control, 턱 힘 + 노이즈로 강성 추정 후 목표 힘까지만)
    #                        fixed = 예전 방식 (정답 패드 닫힘 --close mm에서 멈춤, 실물에선 불가)
    grip_mode = arg("--grip-mode", "stiff")
    if grip_mode not in ("stiff", "fixed"):
        raise SystemExit("--grip-mode는 stiff / fixed 중 하나")
    grip_noise = arg("--grip-noise", 0.5, float)  # 턱 힘 측정 노이즈 (N), 60Hz + 33ms 지연과 함께 주입
    # 강성 추정하며 닫는 속도 (rad/s). 빠를수록 판단 전에 더 찌그러짐 (노이즈 0.5N, 가장 깊은 패드 soft/medium/hard):
    #   0.1 -> 2.65/1.82/1.29mm, 0.2 -> 2.86/2.11/1.50mm, 0.3 -> 3.02/2.74/2.15mm (모두 들기 5/5, 2026-11-18)
    grip_rate = arg("--grip-rate", 0.2, float)
    # 들고 있는 물체 무게 렌더링 (2026-11-18): b로 잡고 있는 동안, 그리퍼가 큐브에서 받는 힘의 아래 성분 x 배율을 Minverse 아래로
    # (15g 큐브 = 0.15N이라 그대로는 안 느껴짐 -> 기본 5배 = 약 0.74N). 0이면 끔
    weight_gain = arg("--weight-gain", WEIGHT_GAIN_DEFAULT, float)
    log = "--no-log" not in sys.argv
    force_src = arg("--force-src", "none")
    if force_src not in ("none", "sim", "na"):
        raise SystemExit("--force-src는 none / sim / na 중 하나")
    force_scale = arg("--force-scale", FORCE_SCALE_DEFAULT, float)
    force_max = min(arg("--force-max", F_HARD_MAX, float), F_HARD_MAX)
    # contact(기본): 로컬 접촉 모델 (깊이 x 스프링 + 감쇠, sim 전용) / raw: 접촉력(또는 NA 추정)을 배율만 곱해 그대로
    force_mode = arg("--force-mode", "contact" if force_src == "sim" else "raw")
    if force_mode not in ("raw", "contact") or (force_mode == "contact" and force_src == "na"):
        raise SystemExit("--force-mode는 raw / contact (contact는 --force-src sim에서만)")
    k_h = arg("--k", CONTACT_K_DEFAULT, float)
    b_h = arg("--b", CONTACT_B_DEFAULT, float)
    R = np.array(cfg["R_haply_to_robot"], float)
    hc, rc = np.array(cfg["haply_center_m"]), np.array(cfg["robot_center_m"])
    h_half = np.array(cfg["haply_half_range_m"])
    box_lo = np.maximum(rc - s * np.abs(R) @ h_half, REACH_LO)
    box_hi = np.minimum(rc + s * np.abs(R) @ h_half, REACH_HI)
    ori = json.load(open(STYLUS, encoding="utf-8")) if os.path.exists(STYLUS) else None
    if ori:
        # 측정한 축 부호에 사용자 체감 보정(pitch_sign/roll_sign, teleop_mapping.json)을 곱함
        p_ax, p_sg = axis_sign(ori["pitch"]["axis"])
        r_ax, r_sg = axis_sign(ori["roll"]["axis"])
        p_sg *= cfg.get("pitch_sign", 1)
        r_sg *= cfg.get("roll_sign", 1)
    ax = json.load(open(STYLUS_AXIS, encoding="utf-8")) if os.path.exists(STYLUS_AXIS) else None
    if ax:
        ball_local = CANDIDATES[ax["ball_axis_local"]]
        up = np.array(ax["frame_up"]); up /= np.linalg.norm(up)
        fw = np.array(ax["frame_forward"]); fw -= (fw @ up) * up; fw /= np.linalg.norm(fw)
        B = np.vstack([np.cross(fw, up), fw, up])  # 자세 좌표 -> 사용자 기준 (오른쪽, 앞, 위) = Minverse 축 순서
    # 스타일러스 볼 끝 = 그리퍼 끝 (2026-11-17): 커서 위치 대신 볼 끝 위치를 매핑. 측정값이 없으면 커서 위치 그대로
    tip_cfg = json.load(open(STYLUS_TIP, encoding="utf-8")) if (ax and os.path.exists(STYLUS_TIP)) else None
    if tip_cfg and not tip_cfg.get("ok"):  # 측정이 불안정했으면 쓰지 않음
        tip_cfg = None
    p_tip = np.array(tip_cfg["p_local_m"]) if tip_cfg else np.zeros(3)
    if tip_cfg:
        # 매핑 중심은 커서 기준으로 잰 값 -> 스타일러스를 볼이 아래로 향하게 든 자세 기준으로 볼 끝 중심으로 옮김
        hc = hc - np.array([0, 0, tip_cfg["length_m"]])
    Mq = None  # 최근 스타일러스 자세 (자세 좌표 -> Minverse 축)

    def to_robot_raw(h):  # 작업 상자로 자르기 전 (벽 힘 계산용)
        return rc + s * R @ (h - hc)

    def to_robot(h):
        return np.clip(to_robot_raw(h), box_lo, box_hi)

    m = mujoco.MjModel.from_xml_path(SCENE3) if n_cubes == 3 else load_scene(level)
    d = mujoco.MjData(m)
    cubes = CubeForces(m, CUBE_PARAMS)
    j_lift = m.joint("shoulder_lift").id
    j_roll = m.joint("wrist_roll").id  # 손목 회전축 = 그리퍼 방향
    ball_log = np.full(3, np.nan)
    # 장면에 있는 큐브마다 관절/바디 정보. '지금 다루는 큐브'(level)는 그리퍼에 가장 가까운 것 (표시, 기록, 턱 힘, 무게)
    CUBE = {}
    for lv in (["soft", "medium", "hard"] if n_cubes == 3 else [level]):
        jf = m.joint(f"cube_{lv}_free")
        CUBE[lv] = dict(q=jf.qposadr[0], v=jf.dofadr[0], b=m.body(f"cube_{lv}").id,
                        pads=[m.joint(f"cube_{lv}_{f}") for f in ("top", "xneg", "xpos")],
                        bodies={m.body(f"cube_{lv}").id} | {m.body(f"cube_{lv}_pad_{f}").id for f in ("top", "xneg", "xpos")})
    cube_b = CUBE[level]["b"]
    pads = [j for c in CUBE.values() for j in c["pads"]]
    pad_adr = [(j.qposadr[0], j.dofadr[0]) for j in pads]
    if PAD_FIX:
        for j in pads:  # (큐브 전부) 패드 이동 한계를 단단하게 (모델 파일은 그대로, 불러온 뒤 값만 바꿈)
            m.jnt_solref[j.id] = PAD_LIMIT_SOLREF
            m.jnt_solimp[j.id, :3] = PAD_LIMIT_SOLIMP
    tip_site = m.site("gripperframe").id
    # 조종 기준점 = 고정 턱 끝 (2026-11-18 사용자 요청). 원래 gripperframe은 두 턱 사이 앞쪽 빈 공간(고정 턱 끝에서 약 2cm)
    # -> 불러온 뒤 site 위치만 옮김 (Menagerie 파일은 그대로). IK, 마커, 기록의 tip_* 모두 이 점 기준
    m.site_pos[tip_site] = FIXED_JAW_TIP
    # 접촉력: 그리퍼만이 아니라 팔 전체 (2026-11-18, 손목/아래팔이 블록에 걸려도 막히게). 받침(base)은 제외
    grip_bodies = {m.body(b).id for b in ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
                                          "moving_jaw_so101_v1")}
    tel = SimTelemetry(m) if force_src == "na" else None
    na = NAClient() if force_src == "na" else None
    renderer = ForceRenderer(cfg["R_haply_to_robot"], force_scale, force_max, m.opt.timestep)
    contact = ContactRenderer(cfg["R_haply_to_robot"], s, k_h, b_h, force_max, m.opt.timestep)
    hand_v = np.zeros(3)
    f_sim = f_src = f_cmd = np.zeros(3)
    f_na = None
    next_na = 0.0
    # 팔 평면 안의 수평 기준 방향 hz = (어깨 들기 축) x (위). 부호는 관절 축에 고정, 팔이 뻗는 쪽(+x, q=0일 때)이 되도록 한 번만 맞춤
    d.qpos[:6] = 0
    mujoco.mj_kinematics(m, d)
    hz_sign = 1.0 if np.cross(d.xaxis[j_lift], [0, 0, 1])[0] > 0 else -1.0
    ik = PosIK(m)
    ik.K_PITCH, ik.K_MID = 1.0, 0.02  # 기본값(0.3, 0.1)은 숙이기를 약하게만 따라감 -> 손목 각도 목표 비중을 키움
    marker_body = m.body("target_marker").mocapid[0]
    marker_geom = m.geom("target_marker").id
    goal = rc.copy()  # 로봇이 실제로 쫓는 목표 (속도 제한 적용)
    d.qpos[:] = ik.solve(goal, iters=200)
    d.qpos[5] = GRIP_OPEN
    d.ctrl[:] = d.qpos[:6]  # 로봇 관절 6개 (qpos 뒤쪽은 cube 자유 관절)
    mujoco.mj_forward(m, d)
    roll_cmd = roll_des = d.qpos[4]
    pitch_cmd = pitch_des = d.qpos[1:4].sum()  # 실제 IK에 넣는 값(속도 제한) / 스타일러스가 원하는 값
    grip_cmd = GRIP_OPEN
    jaw = JawGeometry(m)
    gc = GripController(jaw, sigma=max(grip_noise, 0.1))
    gc.RATE = grip_rate
    jq = m.joint("gripper").qposadr[0]
    cube_bodies = CUBE[level]["bodies"]
    g_rng = np.random.default_rng()
    g_queue, g_next = [], 0.0
    g_state_prev, f_meas = "idle", np.nan
    w_z = 0.0  # 걸러진 무게 신호 (로봇 z, N, 그리퍼가 큐브에서 받는 힘)
    a_w = m.opt.timestep / (1 / (2 * np.pi * WEIGHT_HZ) + m.opt.timestep)
    f_weight = np.zeros(3)  # Minverse 좌표
    w_out = 0.0  # 보내는 무게 힘 z (로봇 좌표 N, 변화 속도 제한 후)
    pad_max_mm = json.load(open(CUBE_PARAMS))["max_depth_mm"]
    over = False
    j_lo = m.jnt_range[:5, 0] + JOINT_MARGIN
    j_hi = m.jnt_range[:5, 1] - JOINT_MARGIN
    d.ctrl[:5] = np.clip(d.ctrl[:5], j_lo, j_hi)
    w_used = W_PITCH_LIST[0]
    q_ik = d.ctrl[:4].copy()
    # '로봇이 갈 수 없는 곳' 벽 (2026-11-18): 작업 상자 밖 + IK가 닿지 못하는 곳 -> Minverse를 되밀어 줌
    target_raw = target = rc.copy()
    pen_reach = np.zeros(3)  # 100Hz IK 결과 (20Hz로 거름)
    pen_reach_f = np.zeros(3)
    a_pen = m.opt.timestep / (1 / (2 * np.pi * 20) + m.opt.timestep)

    def solve_from(target, q0):
        for w in W_PITCH_LIST:
            ik.d.qpos[:4] = q0
            q = ik.solve_pose(target, w_pitch=w, margin=JOINT_MARGIN)
            mujoco.mj_kinematics(m, ik.d)
            e = ik.d.site_xpos[ik.sid] - target
            if np.linalg.norm(e) <= POS_TOL:
                break
        return q[:4].copy(), w, e

    def solve_prio(target):
        """방향 우선: 큰 가중치부터 풀고, 끝 위치가 POS_TOL 넘게 벗어나면 가중치를 줄임"""
        q0 = ik.d.qpos[:4].copy()
        # 그리퍼 끝은 손목 비틀기 축에서 12mm 떨어져 있음 -> 비틀면 끝이 최대 24mm 움직임. IK에 현재 비틀기 명령을 반영
        ik.d.qpos[4] = d.ctrl[4]
        q, w, e = solve_from(target, q0)
        # 갇힘 탈출 (2026-11-18): 이어 풀기만 하면 관절 3개가 한계에 붙은 접힌 자세(어깨 +97, 팔꿈치 -94, 손목 -92)에
        # 갇혀 목표가 11cm 떨어져도 못 나옴 (teleop_cube_20261118_150826.csv) -> 위치나 각도가 크게 어긋나면
        # 기본 자세에서 다시 풀어 보고 더 나은 쪽을 씀
        pe = abs(ik.pitch_target - q[1:4].sum())
        if np.linalg.norm(e) > POS_TOL or pe > 0.5:
            q2, w2, e2 = solve_from(target, IK_SEED)
            pe2 = abs(ik.pitch_target - q2[1:4].sum())
            ok, ok2 = np.linalg.norm(e) <= POS_TOL, np.linalg.norm(e2) <= POS_TOL
            if (ok2 and not ok) or (ok2 == ok and np.linalg.norm(e2) + 0.05 * pe2 < np.linalg.norm(e) + 0.05 * pe):
                q, w, e = q2, w2, e2
            ik.d.qpos[:4] = q  # 다음 번 이어 풀기 시작점
            mujoco.mj_kinematics(m, ik.d)
        return q, w, e  # e: IK 끝 - 목표 (닿지 못한 만큼)

    with connect(URI) as ws:
        devices = json.loads(ws.recv(timeout=5))["inverse3"]
        if not devices:
            raise SystemExit("Minverse가 서비스에 잡히지 않습니다. USB/전원 연결과 Haply Hub의 장치 표시를 확인하세요.")
        device_id = devices[0]["device_id"]
        probe = json.dumps({"inverse3": [{"device_id": device_id, "commands": {"probe_position": {}}}]})
        if force_src != "none":
            # 핸드셰이크가 없으면 힘 명령이 장치에 출력되지 않음 (2026-10-27 확인)
            handshake(ws, devices[0], "so101_teleop")
            print(f"힘 피드백 켬: 출처 {force_src}, 방식 {force_mode}, 상한 {renderer.f_max} N, "
                  f"손 속도 {HAND_SPEED_CUT} m/s 넘으면 0N, 종료 시 0N")
            if weight_gain > 0:
                print(f"  들고 있는 물체 무게: b로 잡고 있는 동안 x{weight_gain:g} 배로 아래 방향 (--weight-gain, 0이면 끔)")
            if force_mode == "contact":
                print(f"  로컬 접촉 모델: 스프링 {k_h} N/m (1mm당 {k_h/1000:.2f}N), 감쇠 {b_h} N·s/m (접촉 중에만)")
            else:
                print(f"  raw: 배율 {force_scale}, 20Hz 필터")
            if na:
                na.reset()
                print(f"NeuralActuator 서버 {'%s:%d' % ('127.0.0.1', 47101)} 사용 (먼저 na/server.py 실행). "
                      "처음 2초 평균을 기준으로 뺌 (그동안 아무것도 닿지 않게). 버튼 a = 기준 다시 잡기")
        print(f"배율 {s}, 로봇 목표 범위 {box_lo.round(3)} ~ {box_hi.round(3)} m")
        if n_cubes == 3:
            print("큐브 3개: 왼쪽(+y) soft, 가운데 medium, 오른쪽(-y) hard. 표시/기록은 그리퍼에 가장 가까운 큐브 기준")
        if grip_mode == "stiff":
            print(f"큐브 {level if n_cubes == 1 else '3종'}, 잡기 = 강성 인지 (턱 힘 노이즈 {grip_noise} N, 60Hz, 33ms 지연 주입). "
                  "b를 누르고 있으면 닫으며 강성 추정 -> 목표 힘까지만 조임, 놓으면 엶. 화면 왼쪽 위에 상태 표시")
        else:
            print(f"큐브 {level}, 잡을 때 닫힘 목표 {close_mm} mm (정답 패드 값 사용)")
        print("로봇은 항상 따라감 / 버튼 b: 누르는 동안 그리퍼 닫기 / 버튼 a: 각도 기준 다시 잡기 / "
              "버튼 c: 상자 되돌리기 / 창을 닫으면 종료")
        print("스타일러스 방향: " + ("볼 방향을 그리퍼 방향으로 (절대, stylus_axis.json)" if ax
                                 else "시작 자세 대비 상대 숙이기 (stylus_axis.json 없음)")
              + (", 비틀기는 상대값 (버튼 a로 기준 다시 잡기)" if ori else ""))
        print(f"방향 우선 (위치가 {POS_TOL*100:.0f}cm 넘게 벗어나면 방향을 양보), 관절 한계 여유 {JOINT_MARGIN} rad, "
              f"관절 속도 {JOINT_RATE} rad/s 이하")
        print("위치 기준: " + (f"스타일러스 볼 끝 (커서에서 {tip_cfg['length_m']*1e3:.0f}mm, stylus_tip.json)" if tip_cfg
                             else "커서 위치 (stylus_tip.json 없음 -> haply_stylus_tip_probe.py로 측정하면 볼 끝 기준)"))
        m.geom_rgba[marker_geom] = GREEN
        if log:
            open_log()

        with mujoco.viewer.launch_passive(m, d) as v:
            wall0, sim0, n, prev_grip, prev_a, prev_c, missing = time.perf_counter(), d.time, 0, None, False, False, 0
            q_ref = pitch_ref = roll_ref = None
            h = None
            gs_last, grip_miss, grip_warned = None, 0, False
            try:
                while v.is_running():
                    out = probe if force_src == "none" else force_msg(device_id, f_cmd)
                    msg = json.loads((ws.send(out), ws.recv(timeout=1))[1])
                    inv = msg.get("inverse3") or []
                    hand_fast = False
                    if inv:
                        h = np.array([inv[0]["state"]["cursor_position"][k] for k in "xyz"])
                        hv = inv[0]["state"].get("cursor_velocity")
                        hand_v = np.array([hv[k] for k in "xyz"]) if hv else np.zeros(3)
                        hand_fast = np.linalg.norm(hand_v) > HAND_SPEED_CUT
                        missing = 0
                    else:  # 응답에 Minverse가 빠진 프레임: 직전 위치 유지
                        missing += 1
                        if missing == 1000:
                            print("주의: 1초 넘게 Minverse 위치가 오지 않습니다. 연결 상태를 확인하세요.")
                    grips = msg.get("wireless_verse_grip") or []
                    # 스타일러스 신호 확인 (2026-11-18: 한 실행 내내 스타일러스 자세/버튼이 하나도 안 와서 회전·그리퍼가 안 움직였음)
                    # 짧게 빠진 프레임은 직전 값 유지(버튼이 풀려 그리퍼가 열리지 않게), 0.2초 넘으면 버튼 놓음, 1초 넘으면 안내
                    if grips and "orientation" in grips[0].get("state", {}):
                        gs_last, grip_miss = grips[0]["state"], 0
                        if grip_warned:
                            print("스타일러스 신호 다시 들어옴")
                            grip_warned = False
                    else:
                        grip_miss += 1
                        if grip_miss == 1000 and not grip_warned:
                            st = grips[0].get("status", {}) if grips else {}
                            print(f"주의: 1초 넘게 스타일러스 신호(자세/버튼)가 없습니다 (상태 {st or '스타일러스 없음'}) "
                                  "-> 회전·그리퍼가 안 움직임. 스타일러스 전원을 켜거나 버튼을 눌러 깨우세요")
                            grip_warned = True
                    gs = gs_last if (gs_last is not None and grip_miss < 200) else {"buttons": {}}
                    grip, a = bool(gs["buttons"].get("b")), bool(gs["buttons"].get("a"))
                    if grip != prev_grip:
                        print("그리퍼 닫음" if grip else "그리퍼 열림")
                        prev_grip = grip
                    if a and not prev_a:
                        q_ref = None  # 버튼 a를 누른 순간 각도 기준을 다시 잡음
                        print("각도 기준 다시 잡음")
                        if na:
                            na.rezero()
                            print("NeuralActuator 기준 다시 잡는 중 (2초, 아무것도 닿지 않게)")
                    prev_a = a
                    c_btn = bool(gs["buttons"].get("c"))
                    if c_btn and not prev_c:  # 상자를 처음 자리로 (위치/자세 = 모델 초기값, 속도 0, 패드도 원위치)
                        for c_ in CUBE.values():  # 큐브 전부
                            d.qpos[c_["q"]:c_["q"] + 7] = m.qpos0[c_["q"]:c_["q"] + 7]
                            d.qvel[c_["v"]:c_["v"] + 6] = 0
                        for j in pads:
                            d.qpos[j.qposadr[0]] = 0
                            d.qvel[j.dofadr[0]] = 0
                        mujoco.mj_forward(m, d)
                        print("상자 되돌림")
                    prev_c = c_btn

                    if h is not None:
                        target_raw = to_robot_raw(h + Mq @ p_tip if Mq is not None else h)  # 볼 끝 (측정값 없으면 p_tip = 0)
                        target = np.clip(target_raw, box_lo, box_hi)
                        d.mocap_pos[marker_body] = target
                        step = target - goal
                        lim = MAX_TARGET_SPEED * m.opt.timestep
                        goal += step if np.linalg.norm(step) <= lim else step / np.linalg.norm(step) * lim
                    if ori and "orientation" in gs:
                        o = gs["orientation"]
                        q = np.array([o["w"], o["x"], o["y"], o["z"]])
                        if q_ref is None:  # 시작 순간(또는 a를 누른 순간) 자세를 기준으로
                            q_ref, pitch_ref, roll_ref = q, d.qpos[1:4].sum(), d.qpos[4]
                        rv = rotvec(qmul(qconj(q_ref), q))
                        if ax:  # 절대: 스타일러스 볼 방향 -> 로봇 좌표 -> 팔 평면 안 각도
                            Mq = B @ qmat(q)
                            ball_dir = R @ (Mq @ ball_local)
                            hz = np.cross(d.xaxis[j_lift], [0, 0, 1])
                            hz = hz_sign * hz / np.linalg.norm(hz)
                            pitch_des = -np.arctan2(ball_dir[2], ball_dir @ hz)
                        else:
                            pitch_des = pitch_ref + p_sg * rv[p_ax]
                        roll_des = np.clip(roll_ref + r_sg * rv[r_ax], *m.jnt_range[4])
                    if ori and ax and "orientation" in gs:
                        ball_log = ball_dir / np.linalg.norm(ball_dir)
                    lift_n = d.xaxis[j_lift] / np.linalg.norm(d.xaxis[j_lift])
                    dt = m.opt.timestep
                    pitch_cmd += np.clip(np.clip(pitch_des, PITCH_MIN, PITCH_MAX) - pitch_cmd, -PITCH_RATE * dt, PITCH_RATE * dt)
                    roll_cmd += np.clip(roll_des - roll_cmd, -ROLL_RATE * dt, ROLL_RATE * dt)
                    ik.pitch_target = pitch_cmd

                    if len(CUBE) > 1 and n % 20 == 0 and gc.state == "idle" and not grip:
                        # 지금 다루는 큐브 = 그리퍼 끝에 가장 가까운 것 (잡는 동안에는 바꾸지 않음)
                        lv_near = min(CUBE, key=lambda k: np.linalg.norm(d.xpos[CUBE[k]["b"]] - d.site_xpos[tip_site]))
                        if lv_near != level:
                            level = lv_near
                            cube_b, cube_bodies = CUBE[level]["b"], CUBE[level]["bodies"]
                    closure, f_grip = cubes.grip(d)[level]
                    if grip_mode == "stiff":  # 강성 인지 파지: b를 누른 순간 시작, 놓으면 엶
                        if grip and gc.state == "idle" and grip_cmd == GRIP_OPEN:
                            gc.start(d.time, grip_cmd)
                        elif not grip and gc.state != "idle":
                            gc.release()
                        qc = gc.step(d.time, d.qpos[jq])
                        grip_cmd = float(np.clip(qc, GRIP_CLOSED, GRIP_OPEN)) if qc is not None else GRIP_OPEN
                    elif grip:  # 닫힘 목표까지 천천히 닫고, 넘으면 살짝 되돌림 (그리퍼 관절은 값이 작을수록 닫힘)
                        grip_cmd -= m.opt.timestep * GRIP_RATE * np.clip((close_mm - closure) / 0.5, -1, 1)
                        grip_cmd = float(np.clip(grip_cmd, GRIP_CLOSED, GRIP_OPEN))
                    else:
                        grip_cmd = GRIP_OPEN
                    if n % 10 == 0:  # 100Hz: 방향 우선 IK
                        q_ik, w_used, e_ik = solve_prio(goal)
                        # 방향을 끝까지 양보했는데도(마지막 가중치) REACH_DB 넘게 못 닿으면 = 로봇이 갈 수 없는 곳
                        ne = np.linalg.norm(e_ik)
                        pen_reach = (e_ik * (1 - REACH_DB / ne) if (w_used == W_PITCH_LIST[-1] and ne > REACH_DB)
                                     else np.zeros(3))
                    # 1kHz: 관절 한계 여유 -> 관절 속도 제한 (10ms마다 한꺼번에 움직이면 순간 속도가 5rad/s까지 튐)
                    q_new = np.clip(np.r_[q_ik, roll_cmd], j_lo, j_hi)
                    dq_max = JOINT_RATE * m.opt.timestep
                    d.ctrl[:5] = d.ctrl[:5] + np.clip(q_new - d.ctrl[:5], -dq_max, dq_max)
                    d.ctrl[5] = grip_cmd
                    cubes.apply(d)
                    if PAD_FIX:  # 패드가 바깥으로(깊이 < 0) 빠지면 되돌리는 스프링 (FEM 범위 0~3mm 안에서는 영향 없음)
                        for qa, da in pad_adr:
                            if d.qpos[qa] > 0:
                                d.qfrc_applied[da] -= PAD_OUT_K * d.qpos[qa] + PAD_OUT_B * d.qvel[da]
                    mujoco.mj_step(m, d)
                    n += 1
                    # 힘: MuJoCo 정답(그리퍼가 받는 접촉력, 1kHz) / NeuralActuator(62Hz, 최신값 유지)
                    f_sim = gripper_contact_force(m, d, grip_bodies)
                    if grip_mode == "stiff" and gc.state != "idle":
                        # 움직이는 턱 힘 (정답) -> 60Hz + 노이즈 + 33ms 지연 -> 강성 추정/그리퍼 제어 (실물: 서보 전류)
                        if d.time >= g_next:
                            f_true = jaw.jaw_force(m, d, cube_bodies)
                            g_queue.append((d.time + 0.033, d.time, f_true + g_rng.normal(0, grip_noise)))
                            g_next = d.time + 1 / 60
                        while g_queue and g_queue[0][0] <= d.time:
                            _, tc, fv = g_queue.pop(0)
                            gc.push_force(tc, fv)
                        f_meas = gc.f_now
                    else:
                        g_queue.clear()
                        f_meas = np.nan
                    if gc.state != g_state_prev:
                        if gc.state == "hold" and gc.k is not None:
                            print(f"잡기: 추정 강성 {gc.k:.0f} ± {gc.k_std:.0f} N/m -> 목표 힘 {gc.f_target:.2f} N "
                                  f"(정답 한쪽 힘 {f_grip:.2f} N, 패드 닫힘 {closure:.2f} mm)")
                            contact.set_object_stiffness(gc.k)  # 로컬 접촉 모델과 공유 (지금은 저장만)
                        elif gc.state == "closing" and g_state_prev == "hold":
                            print("잡기: 힘이 목표보다 많이 작음 -> 판단 취소, 다시 닫는 중")
                        g_state_prev = gc.state
                    if na:
                        tel.accumulate(d)
                        if d.time >= next_na:
                            na.send(tel.features(d))
                            next_na = d.time + DATA_DT
                        f_na, stale = na.poll()
                        if stale and n % 1000 == 0:
                            print("주의: NeuralActuator 서버 응답 없음 -> 힘 0 (na/server.py 실행 확인)")
                    f_src = f_sim if force_src == "sim" else (f_na if f_na is not None else np.zeros(3))
                    on = force_src != "none" and not hand_fast
                    pen_reach_f += a_pen * (pen_reach - pen_reach_f)
                    pen = (target - target_raw) + pen_reach_f  # 상자 밖 + 닿지 못함 (로봇 좌표, 갈 수 있는 쪽)
                    if force_mode == "contact":
                        f_cmd = contact.update(f_sim, d.site_xpos[tip_site].copy(), goal.copy(), hand_v, enable=on,
                                               hand=target.copy(), pen=pen)
                        f_cmd = f_cmd * min(1.0, d.time / 2.0)  # 시작 2초는 서서히 (스타일러스가 상자 밖에서 시작해도 갑자기 2N 안 나게)
                    else:
                        f_cmd = renderer.update(f_src, enable=on)
                    # 들고 있는 물체 무게: 그리퍼가 큐브에서 받는 힘의 아래 성분 (블록 위에 있으면 블록이 받쳐 0, 들면 무게 + 관성)
                    # 덜덜 떨림 보정 (2026-11-18, teleop_cube_20261118_164311.csv: 평균 0.68N, 표준편차 0.31N, 최대 2.62N)
                    #  - 큐브가 블록/바닥에 아직 닿아 있으면(끌리며 걸렸다 들렸다) 0: 완전히 떨어졌을 때만 무게
                    #  - 손 떨림이 관성으로 커져 들어옴 -> 2Hz 저역통과 + 변화 속도 WEIGHT_SLEW N/s 제한
                    w_z += a_w * (contact_force_between(m, d, grip_bodies, cube_bodies)[2] - w_z)
                    lifted = not cube_supported(m, d, cube_bodies, grip_bodies)
                    w_target = weight_gain * min(0.0, w_z) if (grip and weight_gain > 0 and on and lifted) else 0.0
                    w_out += np.clip(w_target - w_out, -WEIGHT_SLEW * m.opt.timestep, WEIGHT_SLEW * m.opt.timestep)
                    f_weight = contact.Rt @ np.array([0.0, 0.0, w_out]) * min(1.0, d.time / 2.0)
                    f_cmd = f_cmd + f_weight
                    nrm = np.linalg.norm(f_cmd)
                    if nrm > force_max:
                        f_cmd = f_cmd * (force_max / nrm)
                    faces = cubes.faces(d)[level]
                    if log:
                        fw = sum(cubes.force_world(d)[level].values())
                        write_log((d.time,*goal, *d.site_xpos[tip_site], *d.xpos[cube_b],
                                    faces["top"][0], faces["top"][1], faces["xneg"][0], faces["xneg"][1],
                                    faces["xpos"][0], faces["xpos"][1], closure, f_grip, *fw, int(grip),
                                    pitch_des, pitch_cmd, d.qpos[1:4].sum(), *ball_log,
                                    np.degrees(np.arcsin(np.clip(ball_log @ lift_n, -1, 1))),
                                    *d.xaxis[j_roll], w_used,
                                    *f_sim, *(f_na if f_na is not None else [np.nan] * 3),
                                    *(na.force_raw if na else [np.nan] * 3), na.gate if na else np.nan, *f_cmd,
                                contact.depth * 1e3, int(contact.in_contact), *target_raw, contact.wall * 1e3,
                                gc.STATE_CODE[gc.state], gc.k if gc.k is not None else np.nan,
                                gc.k_std if gc.k_std is not None else np.nan,
                                gc.f_target if gc.f_target is not None else np.nan, f_meas, w_z, f_weight[2], CUBE_ID[level]))
                    deepest = max(v[0] for v in faces.values())
                    if deepest > pad_max_mm + 0.05 and not over:
                        print(f"주의 t={d.time:.2f}s: 패드가 {deepest:.2f}mm 눌림 - {pad_max_mm}mm 한계를 넘어 FEM 범위 밖 (힘 값 부정확)")
                    over = deepest > pad_max_mm
                    # 큐브가 바닥에 굴러 패드가 바닥/블록에 살짝 닿은 경우(0.1N 미만, 그리퍼 열림)는 출력 안 함
                    if n % 250 == 0 and (faces["top"][1] > 0.1 or f_grip > 0.1 or (grip and f_grip > 0.02)):
                        print(f"t={d.time:6.2f}s  윗면 {faces['top'][0]:4.2f}mm {faces['top'][1]:4.2f}N  |  "
                              f"잡기 닫힘 {closure:4.2f}mm 한쪽 {f_grip:4.2f}N  |  큐브 높이 {d.xpos[cube_b][2]*1e3:5.1f}mm")
                    if n % 16 == 0:
                        if grip_mode == "stiff":  # 화면 왼쪽 위 (MuJoCo 글꼴은 한글 없음)
                            ks = (f"{gc.k:.0f} +- {gc.k_std:.0f} N/m" if gc.state in ("approach", "hold") and gc.k is not None
                                  else "estimating..." if gc.state == "closing" else "-")
                            v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                         "cube (nearest)\ngrip\nstiffness (est)\ntarget force\nforce (meas, noisy)\nforce (true, sim)\npad closure"
                                         "\nload (true)\nweight to haptic",
                                         f"{level}\n{gc.state}\n{ks}\n" + (f"{gc.f_target:.2f} N" if gc.f_target else "-") +
                                         f"\n{f_meas:.2f} N\n{f_grip:.2f} N\n{closure:.2f} mm"
                                         f"\n{-w_z:.2f} N\n{-f_weight[2]:.2f} N (x{weight_gain:g})"))
                        v.sync()
                        lag = (d.time - sim0) - (time.perf_counter() - wall0)
                        if lag > 0:
                            time.sleep(lag)
            finally:  # 창을 닫거나 오류가 나도 Minverse 힘을 0으로
                if force_src != "none":
                    zero = force_msg(device_id, np.zeros(3))
                    for _ in range(50):
                        try:
                            ws.send(zero); ws.recv(timeout=0.2)
                        except Exception:
                            break
                    print("Minverse 힘 0N 송신 완료")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl+C로 중단됨")
    except TimeoutError:
        print("접속은 됐지만 데이터가 오지 않습니다. Minverse 연결/보정 상태를 확인하세요.")
    except OSError as e:
        print(f"접속 실패: Haply 허브가 켜져 있는지 확인하세요 ({e})")
    finally:
        save_log()
