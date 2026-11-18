"""
NeuralActuator 힘 그래프 (cad2026 환경, matplotlib, 2026-11-17)

사용법
  python teleop/plot.py offline            # 오프라인 비교: 정답 / 실측 텔레메트리 추정 / 시뮬레이션 텔레메트리 추정 (들고 옮기기 빈손·300g·500g)
  python teleop/plot.py teleop [csv]       # 텔레오퍼레이션 기록: MuJoCo 정답 / NeuralActuator 추정 / Minverse로 보낸 힘 (csv 생략 시 최신)
결과: 같은 이름 .png (offline은 data/na_replay/na_force_offline.png)

offline 입력: external/.../NeuralActuator/outputs/real_<p>.npz, sim_<p>.npz (infer_actuator.py --force_only 결과)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = ROOT
NA_OUT = os.path.join(HERE, "external", "Dynamics-Modeling", "NeuralActuator", "outputs")
DT = 0.01605
# 차트 색 (dataviz 기본 팔레트: 1 파랑, 2 주황, 정답은 본문 잉크색)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "legend.frameon": False})


def offline():
    pays = [("empty", "빈손"), ("300g", "300 g"), ("500g", "500 g")]
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.2), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    for c, (p, title) in enumerate(pays):
        r, s = np.load(os.path.join(NA_OUT, f"real_{p}.npz")), np.load(os.path.join(NA_OUT, f"sim_{p}.npz"))
        t = np.arange(len(r["force_gt"])) * DT
        gt, fr, fs = r["force_gt"][:, 2], r["force_pred"][:, 2], s["force_pred"][:, 2]
        a = ax[0, c]
        a.plot(t, gt, color=INK, lw=2, label="정답 (페이로드 무게)")
        a.plot(t, fr, color=C1, lw=2, label="추정: 실측 서보 텔레메트리")
        a.plot(t, fs, color=C2, lw=2, label="추정: 시뮬레이션 텔레메트리")
        a.set_title(f"들고 옮기기 {title}", color=INK, loc="left", fontsize=10)
        a.axhline(0, color=INK2, lw=0.8)
        b = ax[1, c]
        er, es = np.abs(fr - gt), np.abs(fs - gt)
        b.plot(t, er, color=C1, lw=2, label=f"실측 텔레메트리 (평균 {er.mean():.2f} N)")
        b.plot(t, es, color=C2, lw=2, label=f"시뮬레이션 (평균 {es.mean():.2f} N)")
        b.set_xlabel("시간 (s)")
        b.legend(loc="upper left", fontsize=8)
        if c == 0:
            a.set_ylabel("Fz (N, 아래 = -)")
            b.set_ylabel("|오차| (N)")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("NeuralActuator z방향 외력 추정 (so101_weight_extended, --force_only)", y=1.05, color=INK, fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "data", "na_replay", "na_force_offline.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("저장:", out)


def teleop(path):
    D = np.genfromtxt(path, delimiter=",", names=True)
    t = D["t"] - D["t"][0]
    has_na = np.isfinite(D["fna_z"]).any()
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, k in enumerate("xyz"):
        a = ax[i]
        a.plot(t, D[f"fsim_{k}"], color=INK, lw=1.5, label="MuJoCo 정답 (그리퍼 접촉력)")
        if has_na:
            a.plot(t, D[f"fna_{k}"], color=C1, lw=2, label="NeuralActuator 추정 (기준 뺀 값)")
        a.set_ylabel(f"F{k} 로봇 좌표 (N)")
        a.axhline(0, color=INK2, lw=0.8)
    ax[-1].set_xlabel("시간 (s)")
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(os.path.basename(path), y=1.03, color=INK, fontsize=10)
    fig.tight_layout()
    out = os.path.splitext(path)[0] + "_force.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("저장:", out)
    if np.abs(D["fcmd_z"]).max() > 0:  # Minverse로 보낸 힘이 있으면 따로
        has_depth = "contact_depth_mm" in D.dtype.names
        fig, ax2 = plt.subplots(2 if has_depth else 1, 1, figsize=(12, 5.5 if has_depth else 3), sharex=True,
                                squeeze=False)
        a = ax2[0, 0]
        nrm = np.sqrt(D["fcmd_x"] ** 2 + D["fcmd_y"] ** 2 + D["fcmd_z"] ** 2)
        a.plot(t, nrm, color=C3, lw=2)
        a.set_ylabel("|Minverse 힘| (N)")
        a.set_title("Minverse로 보낸 힘 크기", loc="left", color=INK, fontsize=10)
        if has_depth:  # 로컬 접촉 모델: 손 목표가 표면 안쪽으로 들어간 깊이
            b = ax2[1, 0]
            b.plot(t, D["contact_depth_mm"], color=C1, lw=2)
            b.set_ylabel("접촉 깊이 (로봇 mm)")
            b.set_title("로컬 접촉 모델: 손 목표가 표면 안쪽으로 들어간 깊이", loc="left", color=INK, fontsize=10)
        ax2[-1, 0].set_xlabel("시간 (s)")
        fig.tight_layout()
        out2 = os.path.splitext(path)[0] + "_haply.png"
        fig.savefig(out2, dpi=120, bbox_inches="tight")
        print("저장:", out2)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "offline"
    if mode == "offline":
        offline()
    else:
        p = sys.argv[2] if len(sys.argv) > 2 else max(glob.glob(os.path.join(HERE, "data", "teleop_cube_*.csv")),
                                                      key=os.path.getmtime)
        teleop(p)
