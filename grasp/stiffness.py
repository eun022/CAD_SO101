"""
강성 추정 모듈 (2026-11-18, 강성 인지 파지 1단계). AI 없이 최소제곱

입력
  - 위치 p (1kHz): 접촉 방향으로 '들어간 정도' 스칼라 (m, 클수록 깊음)
      누르기: -(그리퍼 끝 z),  잡기: 움직이는 턱의 닫힘 거리 (그리퍼 관절 각도 -> 운동학)
  - 힘 f (저속, 예: 60Hz + 노이즈): 접촉 법선 방향 성분 (N), 측정 시각(t_cap)과 함께
모델 (힌지): f = k * max(0, p - p0)
  - p0 = 접촉(변형) 시작 위치 = 힘-깊이 곡선이 꺾이는 점, k = 강성 (N/m)
  - p0 후보 격자마다 k를 닫힌 해로 구해 제곱오차 최소 -> 접촉 전 샘플(힘 0)이 p0를 잡아 줌
접촉 구간: 최근 N_AVG개 평균 힘 > ON_N 이면 시작(히스테리시스), < OFF_N 이 OFF_S 넘게면 끝
  - 시작할 때 LOOKBACK_S 전 샘플부터 포함 (꺾이는 점 앞쪽 데이터)

    est = StiffnessEstimator()
    est.push_pos(t, p)            # 1kHz
    est.push_force(t_cap, f)      # 새 샘플이 올 때
    est.k, est.p0, est.k_std, est.active, est.episodes
"""
from collections import deque

import numpy as np


def hinge_fit(p, f, step=0.02e-3, min_in=3):
    """f = k*max(0, p-p0) 최소제곱. 반환 (k, p0, k_std, sse) 또는 None"""
    p, f = np.asarray(p, float), np.asarray(f, float)
    if len(p) < min_in + 2:
        return None
    n_c = int(min(600, max(2, (p.max() - p.min()) / step + 2)))
    P0 = np.linspace(p.min() - step, p.max(), n_c)
    H = np.maximum(0.0, p[None, :] - P0[:, None])      # 후보 x 샘플
    ok = (H > 0).sum(axis=1) >= min_in
    if not ok.any():
        return None
    H, P0 = H[ok], P0[ok]
    HH = (H * H).sum(axis=1)
    K = (H @ f) / HH
    SSE = float(f @ f) - K * (H @ f)                    # 최소제곱 해에서의 잔차 제곱합
    i = int(np.argmin(SSE))
    k, p0, hh, sse = float(K[i]), float(P0[i]), float(HH[i]), max(float(SSE[i]), 0.0)
    dof = max(len(p) - 2, 1)
    k_std = float(np.sqrt(sse / dof / hh))  # 잔차 분산 기준 k 표준오차 (p0 고정 가정)
    return k, p0, k_std, sse


class StiffnessEstimator:
    N_AVG = 6                  # 평균 낼 샘플 수 (60Hz면 100ms)
    OFF_S = 0.1
    LOOKBACK_S = 0.3
    HIST_S = 1.0

    def __init__(self, sigma=0.5):
        """sigma: 힘 노이즈 표준편차 추정값 (N). 시작 임계값 = 평균의 3시그마, 끝 = 그 절반
        (실제로는 아무것도 안 닿은 상태에서 잠깐 재서 정함)"""
        self.ON_N = max(0.2, 3 * sigma / np.sqrt(self.N_AVG))
        self.OFF_N = self.ON_N / 2
        self.reset()

    def reset(self):
        self.ph = deque()          # (t, p) 1kHz
        self.recent = deque()      # (t_cap, p_at_cap, f) 최근 HIST_S
        self.ep = []               # 지금 접촉 구간 샘플
        self.active = False
        self.t_low = None
        self.k = self.p0 = self.k_std = None
        self.episodes = []         # 끝난 접촉마다 dict(t_on, t_off, k, p0, k_std, n)
        self.t_on = None

    def _p_at(self, tc):
        for tt, pp in reversed(self.ph):
            if tt <= tc + 1e-9:
                return pp
        return self.ph[0][1] if self.ph else 0.0

    def push_pos(self, t, p):
        self.ph.append((t, float(p)))
        while self.ph and self.ph[0][0] < t - self.HIST_S:
            self.ph.popleft()

    def push_force(self, t_cap, f):
        s = (t_cap, self._p_at(t_cap), float(f))
        self.recent.append(s)
        while self.recent and self.recent[0][0] < t_cap - self.HIST_S:
            self.recent.popleft()
        favg = np.mean([r[2] for r in list(self.recent)[-self.N_AVG:]])
        if not self.active:
            if len(self.recent) >= self.N_AVG and favg > self.ON_N:
                self.active, self.t_low, self.t_on = True, None, t_cap
                self.ep = [r for r in self.recent if r[0] >= t_cap - self.LOOKBACK_S]
                self._update()
            return
        self.ep.append(s)
        if favg < self.OFF_N:
            self.t_low = t_cap if self.t_low is None else self.t_low
            if t_cap - self.t_low > self.OFF_S:
                self._update()
                self.episodes.append(dict(t_on=self.t_on, t_off=t_cap, k=self.k, p0=self.p0, k_std=self.k_std,
                                          n=len(self.ep)))
                self.active = False
                return
        else:
            self.t_low = None
        self._update()

    def _update(self):
        r = hinge_fit([e[1] for e in self.ep], [e[2] for e in self.ep])
        if r is not None:
            self.k, self.p0, self.k_std, _ = r
