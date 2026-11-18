# CAD_SO101

NeuralActuator 기반 센서리스 햅틱 텔레오퍼레이션: SO-101 (MuJoCo 시뮬레이션) + Haply Minverse.
자세한 진행 기록과 결정 사항은 [CLAUDE.md](CLAUDE.md)에 있습니다.

## 폴더

| 폴더 | 내용 |
|---|---|
| `teleop/` | Minverse로 MuJoCo SO-101 조작 + 힘 피드백 (`run.py`) |
| `render/` | 저속 힘 렌더링 4종 오프라인 비교 (ZOH, 선형 보간, 외삽, 로컬 접촉 모델) |
| `grasp/` | 강성 인지 파지: 강성 추정, 강성 기반 그리퍼 제어 |
| `na/` | NeuralActuator 추론 서버, 시뮬레이션 텔레메트리 |
| `calib/` | Minverse/스타일러스 측정, 작업공간 매핑 |
| `models/` | MuJoCo 장면과 FEM 큐브 |
| `cube_pkg/` | FEM 큐브 패키지 (받은 파일, 수정하지 않음) |
| `config/` | 측정/보정 값 (JSON) |

`data/`(실행 기록, 그래프)는 저장소에 넣지 않습니다. 스크립트를 실행하면 다시 생깁니다.

## 따로 받아야 하는 것 (저장소에 없음)

```bash
# MuJoCo Menagerie SO-101 (Apache 2.0) -> models/mujoco_menagerie/robotstudio_so101/
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git models/mujoco_menagerie
git -C models/mujoco_menagerie sparse-checkout set robotstudio_so101

# NeuralActuator (MIT) -> external/Dynamics-Modeling/NeuralActuator/
git clone --filter=blob:none --sparse https://github.com/Frank-ZY-Dou/Dynamics-Modeling.git external/Dynamics-Modeling
git -C external/Dynamics-Modeling sparse-checkout set NeuralActuator
# 체크포인트: Hugging Face frankzydou/NeuralActuator 의 so101_weight*.pkl -> external/Dynamics-Modeling/NeuralActuator/checkpoints/
```

## 환경

- `cad2026` (Python 3.12): mujoco 3.17.0, numpy, matplotlib, websockets - 시뮬레이션, 텔레오퍼레이션, 비교
- `neural_actuator` (Python 3.10): jax (CPU), flax, mujoco-mjx - NeuralActuator 서버만

## 실행 (프로젝트 루트에서)

```powershell
python teleop/run.py --force-src sim   # 조작 + 힘 피드백 (Minverse 힘 출력 - 장치를 잡은 상태에서)
python render/dataset.py               # (3) 비교용 데이터 생성
python render/compare.py               # 렌더링 4종 비교
python grasp/eval_stiffness.py         # 강성 추정 정확도
python grasp/eval_grip.py              # 그리퍼 제어 비교
```
