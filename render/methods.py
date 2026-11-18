"""
(3) 저속 힘 스트림 렌더링 4종 (2026-11-18)

공통 인터페이스 (1kHz로 매 틱 호출):
    r.reset()
    F = r.update(t, x, sample)
      t: 현재 시각 (s)
      x: 현재 손 위치 (로봇 좌표 m, 1kHz) - 로컬 접촉 모델만 사용
      sample: 이번 틱에 새 저속 샘플이 도착했으면 (t_cap, F_meas), 아니면 None
              t_cap = 측정 시각 (도착 시각 - 지연), F_meas = 그때의 힘 + 노이즈 (로봇 좌표 N)
      F: 렌더링 힘 (로봇 좌표 N). Minverse로 보낼 때는 좌표 변환 + 배율 + 상한을 따로 적용

- ZOH: 마지막 샘플 유지
- Interp: 직전 두 샘플 사이를 한 주기 동안 직선으로 이음 (매끄럽지만 한 주기 더 늦음)
- Extrap: 직전 두 샘플의 기울기로 현재 시각까지 직선 연장 (지연 보상, 노이즈 증폭)
- ContactModel: 최근 샘플들로 '표면 위치 s0 + 강성 k + 법선 n'을 맞춘 뒤, 매 틱 현재 손 위치로 힘 계산
      F = n * k * max(0, s0 - x·n)   (힌지 모델, 측정 시각의 손 위치와 짝지어 맞춤 -> 지연이 위치로 보상됨)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from collections import deque

import numpy as np


class ZOH:
    name = "ZOH"

    def reset(self):
        self.F = np.zeros(3)

    def update(self, t, x, sample):
        if sample is not None:
            self.F = np.asarray(sample[1], float)
        return self.F


class Interp:
    name = "선형 보간"

    def reset(self):
        self.prev = self.cur = None
        self.t_arr = 0.0
        self.T = 1 / 60

    def update(self, t, x, sample):
        if sample is not None:
            tc, F = sample
            if self.cur is not None:
                self.T = max(tc - self.cur[0], 1e-3)
            self.prev, self.cur = self.cur, (tc, np.asarray(F, float))
            self.t_arr = t
        if self.cur is None:
            return np.zeros(3)
        if self.prev is None:
            return self.cur[1]
        u = min(1.0, (t - self.t_arr) / self.T)
        return self.prev[1] + u * (self.cur[1] - self.prev[1])


class Extrap:
    """최근 N_FIT개 샘플에 직선을 맞춰(최소제곱) 현재 시각까지 연장.
    (처음 버전처럼 두 점 기울기만 쓰면 노이즈 0.5N이 기울기 ±40N/s로 커져 RMSE 3.6N - 비교 의미 없음, 2026-11-18)"""
    name = "외삽"
    N_FIT = 4

    def __init__(self, horizon_max=0.1):
        self.hmax = horizon_max  # 너무 오래 샘플이 없으면 무한히 뻗지 않게

    def reset(self):
        self.buf = deque(maxlen=self.N_FIT)
        self.a = self.b = None
        self.tc = 0.0

    def update(self, t, x, sample):
        if sample is not None:
            self.buf.append((sample[0], np.asarray(sample[1], float)))
            ts = np.array([b[0] for b in self.buf])
            Fs = np.array([b[1] for b in self.buf])
            self.tc = ts[-1]
            if len(ts) >= 2:
                tt = ts - ts[-1]
                A = np.c_[np.ones_like(tt), tt]
                coef = np.linalg.lstsq(A, Fs, rcond=None)[0]
                self.a, self.b = coef[0], coef[1]  # 최근 시각의 값, 기울기
            else:
                self.a, self.b = Fs[-1], np.zeros(3)
        if self.a is None:
            return np.zeros(3)
        return self.a + self.b * min(t - self.tc, self.hmax)


class ContactModel:
    """로컬 접촉 모델 (힌지: 표면 s0 아래로 들어간 만큼 k배)
    - 버퍼: 최근 BUF_S초의 (측정 시각의 손 위치, 측정 힘)
    - 법선 n: 힘이 N_THR 넘는 샘플들의 방향 평균 (천천히 갱신)
    - s0, k: s0 후보 격자마다 k를 닫힌 해로 구하고 오차 최소 (이전 값 쪽으로 약하게 끌어당김 = 들뜸 방지)
    - 유효 조건: 표면 안쪽 샘플 MIN_IN개 이상 + 모델이 0보다 확실히 잘 설명 (오차 감소 > GAIN_MIN)
      무효면 0 (자유 공간에서 노이즈를 내보내지 않음)"""
    name = "로컬 접촉 모델"
    BUF_S = 0.5
    N_AVG = 6         # 접촉 판정/법선은 최근 6개(100ms) 평균 힘으로 (샘플 하나로 하면 노이즈 0.5N이 수시로 넘음)
    N_THR = 0.9       # N, 평균 힘이 이보다 크면 '접촉 증거'
    MIN_IN = 3
    GAIN_MIN = 4.0    # N^2, 힌지 모델이 줄인 제곱오차 합의 최소값
    KNOWN_MIN_IN = 6  # 기억할 표면: 안쪽 샘플 수 최소
    KNOWN_K = (0.1, 5.0)  # N/mm, 기억할 표면 강성 범위 (밖이면 노이즈로 잘못 맞춘 것)
    LAM_K, LAM_S = 2.0, 2.0   # 이전 k(N/mm), s0(mm) 쪽으로 끄는 세기 (샘플 몇 개분)
    K_RANGE = (0.05, 20.0)    # N/mm

    def __init__(self, hist_s=0.3):
        self.hist_s = hist_s

    RESET_S = 3.0  # 이만큼 접촉 증거가 없으면 기준점/법선/표면을 버림 (다른 곳에 닿을 수 있으니)
    # 표면 기억 (2026-11-18): 한 번 맞춘 표면은 기억해 두고, 지금 버퍼로는 아직 맞출 수 없어도(접촉 시작 직후,
    # 지연된 샘플이 아직 안 옴) 손이 그 표면 아래로 들어가면 바로 힘을 냄 = 로컬 접촉 모델의 핵심 이점(지연 보상)
    #  - 옆으로 R_LAT 넘게 벗어나면 안 씀 (다른 물체일 수 있음)
    #  - 반증: 측정 시각에 손이 표면 안쪽 1mm 넘게 있었는데 측정 힘이 예상의 30% 미만인 샘플이 2번 연속이면 버림
    R_LAT = 0.02  # m

    def reset(self):
        self.buf = deque()
        self.xhist = deque()  # (t, x) 1kHz
        self.n = None
        self.nsum = np.zeros(3)
        self.k = self.s0 = None
        self.valid = False
        self.anchor = None  # 위치 기준점 = 처음 힘이 보인 샘플의 손 위치
        self.t_strong = -1e9
        self.known = False
        self.miss = 0
        self.fit_nin, self.f_obs, self.f_known = 0, 0.0, 0.0
        # 처음 버전: 손 위치를 로봇 원점 기준으로 법선에 투영 -> 법선이 노이즈로 몇 도만 흔들려도 (0.27m x 각도)
        # 수십 mm가 튀어 표면 맞추기가 깨짐 (두드리기 2, 3번째를 통째로 놓침). 접촉 근처 기준점의 상대 위치로 바꿈

    def set_stiffness_prior(self, k_n_per_m):
        """강성 인지 파지(grasp/stiffness)의 추정 강성을 사전값으로 (2026-11-18): 처음 맞춤부터 이 값 쪽으로 정규화"""
        self.k = float(k_n_per_m) / 1000.0  # N/mm

    def _x_at(self, tc):
        for tt, xx in reversed(self.xhist):
            if tt <= tc + 1e-9:
                return xx
        return self.xhist[0][1]

    def _fit(self):
        S = np.array([(b[0] - self.anchor) @ self.n for b in self.buf]) * 1e3  # mm, 기준점 대비
        f = np.array([b[1] @ self.n for b in self.buf])
        sse0 = float(f @ f)
        lo, hi = S.min() - 1.0, S.max() + 3.0
        best = None
        for s0 in np.arange(lo, hi, 0.1):
            h = np.maximum(0.0, s0 - S)
            nin = int((h > 0).sum())
            if nin < self.MIN_IN:
                continue
            num, den = float(h @ f), float(h @ h)
            if self.k is not None:
                num += self.LAM_K * self.k
                den += self.LAM_K
            k = float(np.clip(num / max(den, 1e-9), *self.K_RANGE))
            r = f - k * h
            sse = float(r @ r)
            cost = sse + (self.LAM_S * (s0 - self.s0) ** 2 if self.s0 is not None else 0.0)
            if best is None or cost < best[0]:
                best = (cost, s0, k, sse)
        if best is None:
            return False
        _, s0, k, sse = best
        if sse0 - sse < self.GAIN_MIN:
            return False
        self.s0, self.k = s0, k
        h = np.maximum(0.0, s0 - S)
        self.fit_nin = int((h > 0).sum())
        self.f_obs = float(f[h > 0].max())
        return True

    def update(self, t, x, sample):
        x = np.asarray(x, float)
        self.xhist.append((t, x.copy()))
        while self.xhist and self.xhist[0][0] < t - self.hist_s:
            self.xhist.popleft()
        if sample is not None:
            tc, F = sample
            F = np.asarray(F, float)
            self.buf.append((self._x_at(tc), F, tc))
            while self.buf and self.buf[0][2] < tc - self.BUF_S:
                self.buf.popleft()
            Favg = np.mean([b[1] for b in list(self.buf)[-self.N_AVG:]], axis=0)
            if len(self.buf) >= self.N_AVG and np.linalg.norm(Favg) > self.N_THR:
                if self.anchor is None:  # 기준점 = 평균 구간 가운데쯤 샘플의 손 위치
                    self.anchor = list(self.buf)[-self.N_AVG // 2][0].copy()
                self.nsum = 0.8 * self.nsum + Favg
                self.t_strong = tc
            elif tc - self.t_strong > self.RESET_S and self.anchor is not None:
                self.anchor, self.n, self.s0 = None, None, None
                self.nsum[:] = 0
                self.known = False
            if self.known:  # 기억한 표면 반증 검사
                xc = self.buf[-1][0]
                h = self.s0 - ((xc - self.anchor) @ self.n) * 1e3
                self.miss = self.miss + 1 if (h > 1.0 and F @ self.n < 0.3 * self.k * h) else 0
                if self.miss >= 2:
                    self.known, self.miss = False, 0
            if self.anchor is not None and np.linalg.norm(self.nsum) > 1e-6:
                self.n = self.nsum / np.linalg.norm(self.nsum)
                self.valid = self._fit()
                if (self.valid and self.fit_nin >= self.KNOWN_MIN_IN
                        and self.KNOWN_K[0] <= self.k <= self.KNOWN_K[1]):
                    self.known, self.miss = True, 0
                    self.f_known = self.f_obs
            else:
                self.valid = False
        if self.n is None or self.s0 is None or not (self.valid or self.known):
            return np.zeros(3)
        rel = x - self.anchor
        depth_mm = max(0.0, self.s0 - (rel @ self.n) * 1e3)
        f = self.k * depth_mm
        if not self.valid:  # 기억한 표면만 있을 때: 옆으로 너무 벗어나면 안 쓰고, 관측했던 최대 힘 근처로 제한
            if np.linalg.norm(rel - (rel @ self.n) * self.n) > self.R_LAT:
                return np.zeros(3)
            f = min(f, 1.5 * self.f_known + 0.3)
        return self.n * f


ALL = [ZOH, Interp, Extrap, ContactModel]
