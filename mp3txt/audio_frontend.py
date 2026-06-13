# -*- coding: utf-8 -*-
"""음성 인식 전처리 front-end — 조용한 발화 증폭(AGC) + 선택적 노이즈 억제.

근거가 된 연구:
- AGC: 음성 구간을 추정해 선택적으로 레벨을 올리면 ASR WER이 낮아진다
  (Automatic gain control for speech recognition, US20160099007A1 등 표준 기법).
- 노이즈 억제: J.-M. Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time
  Full-Band Speech Enhancement" (RNNoise), MMSP 2018, arXiv:1709.08243.
- 주의: Whisper는 노이즈에 강건하지만 noise-invariant가 아니므로(Whisper-AT,
  arXiv:2307.03183) 공격적 노이즈 제거는 오히려 해가 될 수 있다. 그래서
  AGC를 기본으로 두고, 노이즈 억제는 보수적·선택적으로 적용한다.

모든 함수는 16kHz mono float32 ndarray를 받아 같은 형식을 돌려준다.
numpy만 쓰는 AGC는 항상 동작하고, 노이즈 억제는 noisereduce가 있을 때만 동작한다.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def apply_agc(audio: np.ndarray, target_dbfs: float = -20.0,
              max_gain_db: float = 30.0, noise_floor_dbfs: float = -45.0,
              ) -> np.ndarray:
    """발화 레벨을 목표 RMS(dBFS)로 맞추는 적응형 게인.

    조용히 녹음된 발화를 끌어올려 Whisper가 잘 듣게 한다. 무음/노이즈만 있는
    구간(RMS가 noise_floor 미만)은 증폭하지 않아 배경 잡음이 부풀지 않는다.
    증폭 후 피크가 1을 넘으면 클리핑 방지로 스케일을 낮춘다.

    인자:
        target_dbfs:     목표 RMS 레벨 (음성에 보편적으로 쓰는 -20 dBFS)
        max_gain_db:     한 번에 올릴 수 있는 최대 게인 (노이즈 폭주 방지)
        noise_floor_dbfs: 이보다 조용하면 발화로 보지 않고 증폭하지 않음
    """
    if audio.size == 0:
        return audio
    rms = _rms(audio)
    if rms <= 1e-8:
        return audio
    rms_dbfs = 20.0 * np.log10(rms)
    if rms_dbfs < noise_floor_dbfs:
        return audio  # 사실상 무음 — 잡음만 키우지 않는다

    gain_db = min(target_dbfs - rms_dbfs, max_gain_db)
    if gain_db <= 0.1:
        return audio  # 이미 충분히 크면 줄이지 않는다 (증폭 전용)
    gain = 10.0 ** (gain_db / 20.0)
    out = audio * gain
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:  # 클리핑 방지
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def reduce_noise(audio: np.ndarray, sr: int = SAMPLE_RATE,
                 strength: float = 0.7) -> np.ndarray:
    """정상 잡음(팬·에어컨·험 등) 억제 — noisereduce(스펙트럼 게이팅) 사용.

    noisereduce 미설치 시 원본을 그대로 돌려준다(no-op). 보수적으로 적용하기
    위해 prop_decrease를 1.0 미만으로 둔다(아티팩트 최소화).
    """
    if audio.size == 0:
        return audio
    try:
        import noisereduce as nr
    except ImportError:
        return audio
    try:
        out = nr.reduce_noise(y=audio, sr=sr, stationary=True,
                              prop_decrease=float(strength))
        return np.asarray(out, dtype=np.float32)
    except Exception:
        return audio  # 어떤 이유로든 실패하면 원본 유지


def enhance(audio: np.ndarray, agc: bool = True, denoise: bool = False,
            sr: int = SAMPLE_RATE) -> np.ndarray:
    """전처리 파이프라인: (선택)노이즈 억제 → (선택)AGC.

    노이즈 억제를 먼저 하고 AGC를 나중에 해야, 잡음을 키우지 않고 음성만 키운다.
    """
    out = audio
    if denoise:
        out = reduce_noise(out, sr=sr)
    if agc:
        out = apply_agc(out)
    return out


def noisereduce_available() -> bool:
    try:
        import noisereduce  # noqa: F401
        return True
    except ImportError:
        return False
