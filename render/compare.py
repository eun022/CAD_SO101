"""
(3) 렌더링 4종 오프라인 비교 (cad2026, 2026-11-18)

입력: data/render/<level>_<scenario>.npz (render/dataset.py)
과정: 1kHz 정답 힘 -> 60Hz 샘플링(시작 위상 무작위) + 노이즈(축마다 정규분포 sigma) + 지연 D
      -> 각 방법이 1kHz로 렌더링 -> 정답(같은 시각의 1kHz 힘)과 비교
지표 (모두 힘 벡터 기준, N):
  rmse_all   전체 RMSE
  rmse_in    접촉 중 RMSE: 정답 |F| > 0.1N인 시간만 (자유 공간에서 0을 내는 것만으로 좋아지는 효과 제외)
  rmse_on    접촉 시작 구간 RMSE: 정답 |F|가 0.1N을 넘는 순간 -50ms ~ +200ms
  rmse_off   접촉 해제 구간 RMSE: 정답 |F|가 0.1N 아래로 내려가는 순간 -50ms ~ +200ms
  jerk       렌더링 힘의 변화율 RMS |dF/dt| (N/s). 계단(ZOH)이나 노이즈가 크면 커짐 = 손에 오는 덜컥거림
  overshoot  접촉마다 (렌더링 |F| 최댓값 - 정답 |F| 최댓값), 0 미만은 0. 평균 (N)
  free_rms   자유 공간(정답 |F| < 0.05N이고 접촉 구간에서 200ms 넘게 떨어진 곳) 렌더링 |F| RMS = 헛힘

사용법: python render/compare.py            # 전체 격자 (병렬) -> data/render/results.csv + 그래프
        python render/compare.py plot       # results.csv로 그래프만 다시
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from render import methods as FR

HERE = ROOT
DIR = os.path.join(HERE, "data", "render")
RATE = 60.0
SIGMAS = (0.0, 0.5, 0.7)
DELAYS_MS = (0, 16, 33, 50)
SEEDS = 5
EV_THR = 0.1
COLS = ["level", "scenario", "method", "sigma", "delay_ms", "seed",
        "rmse_all", "rmse_in", "rmse_on", "rmse_off", "jerk", "overshoot", "free_rms"]


def events(t, fm):
    on = fm > EV_THR
    d = np.diff(on.astype(int))
    return t[1:][d == 1], t[1:][d == -1]


def window_mask(t, times, a=-0.05, b=0.2):
    m = np.zeros(len(t), bool)
    for tt in times:
        m |= (t >= tt + a) & (t <= tt + b)
    return m


def render(method_cls, t, X, Fgt, sigma, delay, seed):
    rng = np.random.default_rng(seed)
    dt = t[1] - t[0]
    T = 1 / RATE
    tcap = np.arange(rng.uniform(0, T), t[-1], T)
    idx = np.minimum(np.round(tcap / dt).astype(int), len(t) - 1)
    meas = Fgt[idx] + rng.normal(0, sigma, (len(idx), 3))
    arrive = np.ceil((tcap + delay) / dt - 1e-9).astype(int)  # 도착 틱
    r = method_cls()
    r.reset()
    out = np.zeros_like(Fgt)
    j = 0
    for i in range(len(t)):
        sample = None
        while j < len(arrive) and arrive[j] <= i:  # 같은 틱에 여러 개면 마지막 것만 의미 있음 (모두 넣음)
            sample = (tcap[j], meas[j])
            if j + 1 < len(arrive) and arrive[j + 1] <= i:
                r.update(t[i], X[i], sample)
                sample = None
            j += 1
        out[i] = r.update(t[i], X[i], sample)
    return out


def metrics(t, Fgt, Fr):
    fm = np.linalg.norm(Fgt, axis=1)
    e = np.linalg.norm(Fr - Fgt, axis=1)
    on_t, off_t = events(t, fm)
    m_on, m_off = window_mask(t, on_t), window_mask(t, off_t)
    dFr = np.linalg.norm(np.diff(Fr, axis=0), axis=1) / (t[1] - t[0])
    ov = []
    for a in on_t:
        b = off_t[off_t > a]
        b = b[0] if len(b) else t[-1]
        w = (t >= a) & (t <= b + 0.2)
        ov.append(max(0.0, np.linalg.norm(Fr[w], axis=1).max() - fm[w].max()))
    near = window_mask(t, np.r_[on_t, off_t], -0.2, 0.2) | (fm >= 0.05)
    free = ~near
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else np.nan  # noqa: E731
    return dict(rmse_all=rms(e), rmse_in=rms(e[fm > EV_THR]), rmse_on=rms(e[m_on]), rmse_off=rms(e[m_off]), jerk=rms(dFr),
                overshoot=float(np.mean(ov)) if ov else np.nan,
                free_rms=rms(np.linalg.norm(Fr[free], axis=1)))


def run_file(path):
    Z = np.load(path)
    t, X, Fgt = Z["t"], Z["hand"], Z["F"]
    level, scen = os.path.basename(path)[:-4].split("_", 1)
    rows = []
    for sigma in SIGMAS:
        for dms in DELAYS_MS:
            for seed in range(SEEDS if sigma > 0 else 1):
                for cls in FR.ALL:
                    Fr = render(cls, t, X, Fgt, sigma, dms / 1000, 1000 * seed + 7)
                    rows.append(dict(level=level, scenario=scen, method=cls.name, sigma=sigma, delay_ms=dms,
                                     seed=seed, **metrics(t, Fgt, Fr)))
    gt = metrics(t, Fgt, Fgt)
    print(f"{os.path.basename(path):24s} 완료 ({len(rows)}개, 정답 자체 jerk {gt['jerk']:.1f} N/s)", flush=True)
    return rows


def summarize(rows):
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["method"], float(r["sigma"]), int(r["delay_ms"]))].append(r)
    keys = ["rmse_all", "rmse_in", "rmse_on", "rmse_off", "jerk", "overshoot", "free_rms"]
    out = {}
    for k, rs in g.items():
        out[k] = {m: float(np.nanmean([float(r[m]) for r in rs])) for m in keys}
    return out


def print_table(S, sigma, dms):
    print(f"\n노이즈 {sigma} N, 지연 {dms} ms (큐브 3종 x 시나리오 5종 x 시드 평균)")
    print(f"{'방법':14s} {'RMSE':>6s} {'접촉중':>6s} {'시작':>6s} {'해제':>6s} {'jerk':>7s} {'오버슈트':>7s} {'헛힘':>6s}")
    for cls in FR.ALL:
        v = S[(cls.name, sigma, dms)]
        print(f"{cls.name:14s} {v['rmse_all']:6.3f} {v['rmse_in']:6.3f} {v['rmse_on']:6.3f} {v['rmse_off']:6.3f} {v['jerk']:7.1f} "
              f"{v['overshoot']:7.3f} {v['free_rms']:6.3f}")


def plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from teleop import plot as P  # 같은 스타일 (맑은 고딕, 팔레트)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    S = summarize(rows)
    names = [c.name for c in FR.ALL]
    mets = [("rmse_in", "접촉 중 RMSE (N)"), ("rmse_on", "접촉 시작 RMSE (N)"), ("rmse_off", "접촉 해제 RMSE (N)"),
            ("jerk", "jerk: |dF/dt| RMS (N/s)"), ("overshoot", "오버슈트 (N)"), ("free_rms", "자유 공간 헛힘 RMS (N)")]
    for sigma in SIGMAS[1:]:
        fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
        for a, (mk, lab) in zip(axs.flat, mets):
            for c, nm in enumerate(names):
                a.plot(DELAYS_MS, [S[(nm, sigma, d)][mk] for d in DELAYS_MS], color=colors[c], lw=2, marker="o",
                       ms=5, label=nm)
            a.set_title(lab, loc="left", color=P.INK, fontsize=10)
            a.set_xlabel("지연 (ms)")
            a.set_xticks(DELAYS_MS)
            a.set_ylim(bottom=0)
        h, lab = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"렌더링 4종 비교: 60Hz, 노이즈 {sigma} N (큐브 3종 x 시나리오 5종 x 시드 {SEEDS} 평균)",
                     y=1.07, color=P.INK, fontsize=11)
        fig.tight_layout()
        out = os.path.join(DIR, f"compare_sigma{sigma}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("저장:", out)
    # 예시 시계열: medium 두드리기, 노이즈 0.5, 지연 33ms
    Z = np.load(os.path.join(DIR, "medium_tap.npz"))
    t, X, Fgt = Z["t"], Z["hand"], Z["F"]
    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True, sharey=True)
    for c, cls in enumerate(FR.ALL):
        Fr = render(cls, t, X, Fgt, 0.5, 0.033, 7)
        a = axs[c]
        a.plot(t, Fgt[:, 2], color=P.INK, lw=1.5, label="정답 (1kHz)")
        a.plot(t, Fr[:, 2], color=colors[c], lw=1.8, label=cls.name)
        a.set_ylabel("Fz (N)")
        a.legend(loc="upper right", ncol=2)
    axs[-1].set_xlabel("시간 (s)")
    fig.suptitle("예시: medium 큐브 두드리기 3번, 60Hz + 노이즈 0.5N + 지연 33ms", y=1.0, color=P.INK, fontsize=11)
    fig.tight_layout()
    out = os.path.join(DIR, "example_medium_tap.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("저장:", out)


def main():
    res = os.path.join(DIR, "results.csv")
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        rows = list(csv.DictReader(open(res, encoding="utf-8")))
    else:
        files = sorted(glob.glob(os.path.join(DIR, "*_*.npz")))
        with ProcessPoolExecutor(max_workers=min(15, os.cpu_count() or 4)) as ex:
            rows = [r for rs in ex.map(run_file, files) for r in rs]
        with open(res, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, COLS)
            w.writeheader()
            w.writerows(rows)
        print("저장:", res, len(rows), "행")
    S = summarize(rows)
    for sg, dm in ((0.5, 33), (0.7, 33), (0.5, 0), (0.7, 50)):
        print_table(S, sg, dm)
    plots(rows)


if __name__ == "__main__":
    main()

