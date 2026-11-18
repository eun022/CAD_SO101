"""
강성 추정 정확도 평가 (cad2026, 2026-11-18)

데이터
  누르기: data/render/<level>_<scenario>.npz (render/dataset.py) - p = -(고정 턱 끝 z), f = 그리퍼 접촉력 z
  잡기:   data/grasp/grip_<level>_<speed>.npz (grasp/grip_dataset.py) - p = 움직이는 턱 닫힘 거리, f = 그 턱 힘
정답 강성 (재료): 큐브 FEM 곡선 0~2mm를 직선 맞춤한 기울기
  누르기 = press 곡선 (윗면 패드 눌림 vs 힘), 잡기 = grip_total 곡선 (양쪽 패드 닫힘 합 vs 한쪽 힘)
참고 '시뮬 실효 강성': 노이즈 없는 1kHz 데이터에 같은 힌지 맞춤 (방법의 한계와 노이즈 영향을 나눠 보기 위함)
과정: 60Hz(위상 무작위) + 노이즈 sigma (축마다) + 지연 33ms (측정 시각은 앎) -> StiffnessEstimator
  -> 정답 접촉(노이즈 없는 힘 > 0.1N)과 겹치는 추정 접촉 구간마다 오차 %
결과: data/grasp/stiffness_results.csv, 표 출력, data/grasp/stiffness_fit_*.png
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from grasp.stiffness import StiffnessEstimator, hinge_fit  # noqa: E402

OUT = os.path.join(ROOT, "data", "grasp")
PARAMS = os.path.join(ROOT, "models", "cubes_teleop", "cube_params.json")
SIGMAS = (0.3, 0.5, 0.7)
SEEDS = 20
RATE, DELAY = 60.0, 0.033
GT_RANGE_MM = 2.0
EARLY_S = 0.25


def gt_stiffness():
    """레벨별 정답 강성 (N/m): FEM 곡선 0~2mm 직선 맞춤 기울기"""
    P = json.load(open(PARAMS))
    out = {}
    for lv, c in P["levels"].items():
        r = {}
        for mode, (curve, xkey) in {"press": (P["curves"]["press"], "depth_mm"),
                                    "grip": (P["grip_total"], "closure_mm")}.items():
            x = np.array(curve[xkey])
            y = np.array(curve["F1_N_per_MPa"]) * c["E_MPa"]
            s = x <= GT_RANGE_MM + 1e-9
            r[mode] = np.polyfit(x[s] * 1e-3, y[s], 1)[0]
        out[lv] = r
    return out


def load(path):
    Z = np.load(path)
    name = os.path.basename(path)[:-4]
    if name.startswith("grip_"):
        _, level, sc = name.split("_")
        return "grip", level, sc, Z["t"], Z["p_jaw"], Z["f_n"]
    level, sc = name.split("_", 1)
    return "press", level, sc, Z["t"], -Z["tip"][:, 2], Z["F"][:, 2]


def gt_episodes(t, f):
    on = f > 0.1
    d = np.diff(on.astype(int))
    a, b = t[1:][d == 1], t[1:][d == -1]
    if on[0]:
        a = np.r_[t[0], a]
    if len(b) < len(a):
        b = np.r_[b, t[-1]]
    return list(zip(a, b))


def run_one(t, p, f, sigma, seed):
    rng = np.random.default_rng(seed)
    dt = t[1] - t[0]
    tcap = np.arange(rng.uniform(0, 1 / RATE), t[-1], 1 / RATE)
    idx = np.minimum(np.round(tcap / dt).astype(int), len(t) - 1)
    meas = f[idx] + rng.normal(0, sigma, len(idx))  # 법선 방향 성분의 노이즈 = 축 노이즈와 같은 sigma
    arrive = np.ceil((tcap + DELAY) / dt - 1e-9).astype(int)
    est = StiffnessEstimator(sigma=max(sigma, 0.1))
    j = 0
    early = {}
    for i in range(len(t)):
        est.push_pos(t[i], p[i])
        while j < len(arrive) and arrive[j] <= i:
            est.push_force(tcap[j], meas[j])
            j += 1
        if est.active and est.t_on not in early and t[i] - est.t_on >= EARLY_S and est.k is not None:
            early[est.t_on] = est.k
    eps = list(est.episodes)
    if est.active:
        eps.append(dict(t_on=est.t_on, t_off=t[-1], k=est.k, p0=est.p0, k_std=est.k_std, n=len(est.ep)))
    for e in eps:
        e["k_early"] = early.get(e["t_on"], np.nan)
    return eps


def work(path):
    mode, level, sc, t, p, f = load(path)
    gt = gt_stiffness()[level][mode]
    ref = hinge_fit(p, f, min_in=20)  # 노이즈 없는 1kHz 전체 = 시뮬 실효 강성
    gte = gt_episodes(t, f)
    rows = []
    for sigma in SIGMAS:
        for seed in range(SEEDS):
            eps = run_one(t, p, f, sigma, 1000 * seed + 11)
            matched = set()
            for e in eps:
                hit = [g for g, (a, b) in enumerate(gte) if e["t_on"] <= b + 0.1 and e["t_off"] >= a - 0.1]
                if not hit:
                    rows.append(dict(mode=mode, level=level, scenario=sc, sigma=sigma, seed=seed, kind="false",
                                     k=e["k"], k_gt=gt, k_sim=ref[0]))
                    continue
                matched.update(hit)
                rows.append(dict(mode=mode, level=level, scenario=sc, sigma=sigma, seed=seed, kind="hit",
                                 k=e["k"], k_early=e["k_early"], k_gt=gt, k_sim=ref[0], k_std=e["k_std"],
                                 p0=e["p0"], p0_sim=ref[1]))
            for g in range(len(gte)):
                if g not in matched:
                    rows.append(dict(mode=mode, level=level, scenario=sc, sigma=sigma, seed=seed, kind="miss",
                                     k_gt=gt, k_sim=ref[0]))
    print(f"{os.path.basename(path):26s} 정답 {gt:7.0f} N/m, 노이즈 없는 1kHz 맞춤 {ref[0]:7.0f} N/m "
          f"({(ref[0]-gt)/gt*100:+5.1f}%)", flush=True)
    return rows


def table(rows):
    print("\n[실효 강성 대비] 추정기 정확도: 기준 = 노이즈 없는 1kHz 데이터에 같은 맞춤 (그리퍼가 실제로 느끼는 강성)")
    print(f"{'방식':5s} {'큐브':6s} {'노이즈':>5s} | {'실효 N/m':>8s} | {'치우침(중앙)':>10s} {'|오차| 중앙':>10s} {'|오차| 90%':>10s} "
          f"{'0.25s때':>8s} | {'감지율':>6s} {'헛감지':>6s}")
    for mode in ("press", "grip"):
        for level in ("soft", "medium", "hard"):
            for sigma in SIGMAS:
                R = [r for r in rows if r["mode"] == mode and r["level"] == level and float(r["sigma"]) == sigma]
                hit = [r for r in R if r["kind"] == "hit" and r["k"] not in (None, "", "None")]
                if not hit:
                    continue
                ks = float(np.mean([float(r["k_sim"]) for r in hit]))  # 표시용 평균 (오차는 시나리오별 실효 강성 기준)
                err = np.array([(float(r["k"]) - float(r["k_sim"])) / float(r["k_sim"]) * 100 for r in hit])
                ee = np.array([(float(r["k_early"]) - float(r["k_sim"])) / float(r["k_sim"]) * 100 for r in hit
                               if r.get("k_early") not in (None, "", "nan") and np.isfinite(float(r["k_early"]))])
                miss = [r for r in R if r["kind"] == "miss"]
                false = [r for r in R if r["kind"] == "false"]
                det = len(hit) / max(len(hit) + len(miss), 1) * 100
                print(f"{'누르기' if mode == 'press' else '잡기':5s} {level:6s} {sigma:5.1f} | {ks:8.0f} | {np.median(err):+9.1f}% "
                      f"{np.median(np.abs(err)):9.1f}% {np.percentile(np.abs(err), 90):9.1f}% "
                      f"{(np.median(np.abs(ee)) if len(ee) else np.nan):7.1f}% | {det:5.0f}% {len(false)/max(len({(r['scenario'], r['seed']) for r in R}), 1):5.2f}회")
    print("\n[재료 강성 대비] 정답 = 큐브 FEM 곡선 0~2mm 직선 기울기")
    print(f"{'방식':5s} {'큐브':6s} {'노이즈':>5s} | {'정답 N/m':>8s} {'시뮬실효':>8s} | {'평균오차':>7s} {'|오차|중앙':>9s} "
          f"{'0.25s때':>8s} | {'감지율':>6s} {'헛감지':>6s} {'꺾임점 오차':>10s}")
    for mode in ("press", "grip"):
        for level in ("soft", "medium", "hard"):
            for sigma in SIGMAS:
                R = [r for r in rows if r["mode"] == mode and r["level"] == level and float(r["sigma"]) == sigma]
                if not R:
                    continue
                hit = [r for r in R if r["kind"] == "hit" and r["k"] not in (None, "", "None")]
                miss = [r for r in R if r["kind"] == "miss"]
                false = [r for r in R if r["kind"] == "false"]
                gt, ks = float(R[0]["k_gt"]), float(R[0]["k_sim"])
                if hit:
                    err = np.array([(float(r["k"]) - gt) / gt * 100 for r in hit])
                    ee = np.array([(float(r["k_early"]) - gt) / gt * 100 for r in hit
                                   if r.get("k_early") not in (None, "", "nan") and np.isfinite(float(r["k_early"]))])
                    p0e = np.array([abs(float(r["p0"]) - float(r["p0_sim"])) * 1e3 for r in hit])
                    s_err = f"{np.median(err):+6.1f}% {np.median(np.abs(err)):8.1f}% " + \
                        (f"{np.median(np.abs(ee)):7.1f}%" if len(ee) else "    -   ")
                    s_p0 = f"{np.median(p0e):7.2f}mm"
                else:
                    s_err, s_p0 = "     -         -          -   ", "     -   "
                n_gt = len(hit) + len(miss)
                det = len(hit) / max(n_gt, 1) * 100
                print(f"{'누르기' if mode == 'press' else '잡기':5s} {level:6s} {sigma:5.1f} | {gt:8.0f} {ks:8.0f} | {s_err} | "
                      f"{det:5.0f}% {len(false)/max(len({(r['scenario'], r['seed']) for r in R}), 1):5.2f}회 {s_p0}")


def plot_fits():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from teleop import plot as P
    gt = gt_stiffness()
    for mode, pat in (("press", os.path.join(ROOT, "data", "render", "{lv}_press_slow.npz")),
                      ("grip", os.path.join(OUT, "grip_{lv}_slow.npz"))):
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        for a, lv in zip(axs, ("soft", "medium", "hard")):
            _, _, _, t, p, f = load(pat.format(lv=lv))
            rng = np.random.default_rng(3)
            tcap = np.arange(0, t[-1], 1 / RATE)
            idx = np.round(tcap / (t[1] - t[0])).astype(int)
            meas = f[idx] + rng.normal(0, 0.5, len(idx))
            g = gt_episodes(t, f)[0]
            sel = (tcap > g[0] - 0.3) & (tcap < g[1])
            ps, fs = p[idx][sel], meas[sel]
            k, p0, _, _ = hinge_fit(ps, fs)
            ref = hinge_fit(p, f, min_in=20)
            x = (ps - ref[1]) * 1e3
            a.scatter(x, fs, s=10, color=P.INK2, alpha=0.6, label="60Hz 측정 (노이즈 0.5N)")
            xx = np.linspace(x.min(), x.max(), 100)
            a.plot(xx, k * np.maximum(0, xx * 1e-3 + ref[1] - p0), color=P.C1, lw=2,
                   label=f"추정 k {k:.0f} N/m")
            a.plot(xx, gt[lv][mode] * np.maximum(0, xx * 1e-3), color=P.C2, lw=2, ls="--",
                   label=f"정답 (FEM 0~2mm) {gt[lv][mode]:.0f} N/m")
            a.set_title(f"{lv}", loc="left", color=P.INK, fontsize=10)
            a.set_xlabel("깊이 (mm, 실제 접촉 시작 = 0)" if mode == "press" else "턱 닫힘 (mm, 실제 접촉 시작 = 0)")
            a.set_ylabel("힘 (N)")
            a.legend(loc="upper left", fontsize=8)
        fig.suptitle(f"강성 추정 예시: {'누르기 (1cm/s)' if mode == 'press' else '잡기 (천천히 닫기)'}", color=P.INK,
                     fontsize=11)
        fig.tight_layout()
        out = os.path.join(OUT, f"stiffness_fit_{mode}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print("저장:", out)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "table":  # 저장된 결과로 표/그림만
        table(list(csv.DictReader(open(os.path.join(OUT, "stiffness_results.csv"), encoding="utf-8"))))
        plot_fits()
        return
    files =sorted(glob.glob(os.path.join(ROOT, "data", "render", "*_*.npz"))) + \
        sorted(glob.glob(os.path.join(OUT, "grip_*.npz")))
    with ProcessPoolExecutor(max_workers=min(15, os.cpu_count() or 4)) as ex:
        rows = [r for rs in ex.map(work, files) for r in rs]
    cols = ["mode", "level", "scenario", "sigma", "seed", "kind", "k", "k_early", "k_gt", "k_sim", "k_std", "p0", "p0_sim"]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "stiffness_results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    table(rows)
    plot_fits()


if __name__ == "__main__":
    main()

