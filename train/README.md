# 고스트 어댑터 (zero-init residual adapter) PoC

Whisper를 **건드리지 않은 채** 위에 얇은 보정 레이어를 얹어, 틀리는 케이스만
고치고 맞는 케이스·다른 언어는 보존하는 실험. 전체 재학습(불가능)도, 위험한
전체 파인튜닝(영어 붕괴)도 아닌 중간 길.

## 원리

- 디코더 출력과 lm_head 사이에 N겹 잔차 블록을 끼움: `x = x + Wup(GELU(Wdown(LN(x))))`
- **Wup을 0으로 초기화** → 시작 시 delta=0 → **원본과 비트 단위 동일 출력**
  (ControlNet zero-conv, LoRA B-0init과 같은 원리. `verify_identity.py`로 증명됨)
- base 전부 동결, 고스트 블록만 학습 (large-v3-turbo 기준 전체의 ~2%)
- **리허설**: 맞은 케이스도 정답과 함께 학습 → catastrophic forgetting 방지

이 두 가지(0-init + 리허설)가 ghost613 파인튜닝이 실패한 원인(영어 붕괴·과적합)을
구조적으로 막는다.

## 검증된 것 (이 PC, CPU)

- `verify_identity.py`: 0-init 고스트 = 원본과 출력 차이 **0.00e+00** (항등 보장 성립),
  가중치 교란 시 출력 변함 (학습 가능). **PASS**
- `poc_adapter.py` 스모크(whisper-tiny): 파이프라인 정상 동작, loss 하강 확인.
  낮은 LR에서 보존 목표(깨끗한 음성·영어 유지) 성립.
- **단**, 실제 "잡음 보정" 효과는 whisper-tiny로는 검증 불가 (한국어를 거의 못하는
  모델 + CPU 학습 한계). 아래 GPU 실전 설정 필요.

## GPU PC에서 실전 실행

```powershell
# 1) CUDA torch + 학습 의존성 (GPU PC venv)
uv pip install torch transformers --index-url https://download.pytorch.org/whl/cu128
uv pip install soundfile librosa

# 2) 데이터 준비 (없으면)
python test_audio\fetch_datasets.py 200          # FLEURS ko 200개
# (영어는 fetch_fleurs(N,"en_us","fleurs_en")로)

# 3) 항등 성질 먼저 확인
python train\verify_identity.py

# 4) 실전 학습 — large-v3-turbo, 충분한 데이터/스텝
python train\poc_adapter.py `
    --model openai/whisper-large-v3-turbo `
    --train 200 --heldout 40 --steps 2000 --lr 3e-4 --layers 4 `
    --snr 0 --save train\ghost_turbo_ko_noisy.pt
```

## 결과 해석 (held-out 3지표)

| 지표 | 목표 | 의미 |
|---|---|---|
| 잡음 한국어 CER | **↓ 내려감** | 보정 성공 |
| 깨끗 한국어 CER | = 유지 | 맞던 것 안 망가짐 |
| 영어 WER | = 유지 | forgetting 없음 |

**세 개가 동시에 만족돼야 성공.** 잡음 CER만 내려가고 나머지가 오르면 과적합 →
LR을 낮추거나(1e-4), 리허설 비중을 늘리거나, 학습 데이터를 늘릴 것.

## 주의 (실험 설계의 함정)

1. **과적합/암기**: 학습 발화가 적으면 일반화 대신 그 샘플을 외움. 반드시
   held-out(학습에 안 쓴 발화)으로 측정. 스모크에서 tiny+높은 LR이 발산한 게 그 예.
2. **추론 스택 분리**: faster-whisper/OpenVINO는 추론 전용(학습 불가). 학습은
   PyTorch transformers로, 배포는 고스트를 병합 후 CT2/OV로 변환.
3. **데이터가 본질**: 가장 가치 있는 건 *실제 사용 중 Whisper가 틀린 케이스 +
   사람이 고친 정답*. FLEURS는 이미 정확도가 높아 보정할 거리가 적다. 이 기법은
   범용 개선이 아니라 **도메인 특화 보정**(고유명사·전문용어·잡음)에서 빛난다.
