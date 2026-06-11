# -*- coding: utf-8 -*-
"""오디오 파일 디코드 — mp3/wav/m4a/flac/ogg 등을 16kHz mono float32 로 읽는다.

faster-whisper에 동봉된 PyAV 디코더를 재사용하므로 별도의 ffmpeg 호출이 없고,
전사(whisper)와 화자분리(pyannote)가 같은 파형을 공유한다 (디코드 1회).
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """오디오 파일을 sr Hz mono float32 ndarray로 디코드한다.

    지원 포맷은 PyAV(ffmpeg 라이브러리)가 읽을 수 있는 모든 것.
    실패 시 RuntimeError를 던진다.
    """
    from faster_whisper.audio import decode_audio

    try:
        audio = decode_audio(path, sampling_rate=sr)
    except Exception as e:
        raise RuntimeError(f"오디오 디코드 실패: {path}\n  ({e})") from e
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:  # 안전망: 혹시 다채널로 오면 평균으로 모노화
        audio = audio.mean(axis=0).astype(np.float32)
    if audio.size == 0:
        raise RuntimeError(f"오디오 트랙이 비어 있습니다: {path}")
    return audio


def duration_sec(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """파형 길이(초)."""
    return float(len(audio)) / float(sr)
