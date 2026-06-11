# -*- coding: utf-8 -*-
"""faster-whisper 래퍼 — CPU(int8) 전사.

- transcribe_long():  배치(파일) 변환용. 워드 타임스탬프 + VAD 필터 + beam 5.
- transcribe_array(): 실시간 모드용. 발화 조각을 빠르게 전사 (beam 1).
모델은 첫 호출 때 내려받아 %USERPROFILE%\\.cache\\huggingface 에 캐시된다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def _cpu_threads() -> int:
    """ctranslate2에 줄 스레드 수 — 코어를 다 쓰면 UI가 버벅이므로 2개 남긴다."""
    return max(2, (os.cpu_count() or 4) - 2)


class Transcriber:
    """Whisper 모델 1개를 감싸는 지연 로딩 래퍼. 인스턴스는 스레드 1개에서만 사용할 것."""

    def __init__(self, model_name: str, compute_type: str = "int8",
                 language: str | None = None, device: str = "cpu"):
        self.model_name = model_name
        self.device = device  # "cpu" 또는 "cuda" (NVIDIA GPU)
        # CUDA에서는 int8 가중치 + fp16 연산이 기본 — VRAM 적은 GPU(4GB)에서도 안전
        if device == "cuda" and compute_type == "int8":
            compute_type = "int8_float16"
        self.compute_type = compute_type
        # "auto"/"" 는 자동 감지(None)로 정규화
        self.language = None if language in (None, "", "auto") else language
        self._model = None

    def ensure_model(self) -> None:
        """모델을 로드(필요 시 다운로드)한다. 이미 로드됐으면 즉시 반환."""
        if self._model is not None:
            return
        if self.device == "cuda":
            from .engine_select import add_nvidia_dll_dirs
            add_nvidia_dll_dirs()  # pip로 설치한 cuBLAS/cuDNN DLL 경로 등록
        from faster_whisper import WhisperModel

        kwargs = {}
        if self.device == "cpu":
            kwargs["cpu_threads"] = _cpu_threads()
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            **kwargs,
        )

    def transcribe_long(self, audio: np.ndarray, word_timestamps: bool = True,
                        on_segment: Optional[Callable[[Segment], None]] = None,
                        ) -> tuple[list[Segment], str]:
        """긴 오디오(파일)를 전사한다. (세그먼트 리스트, 감지된 언어) 반환.

        on_segment 콜백으로 세그먼트가 나올 때마다 진행 상황을 알릴 수 있다.
        """
        self.ensure_model()
        raw_segments, info = self._model.transcribe(
            audio,
            language=self.language,
            word_timestamps=word_timestamps,
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,  # 반복 환각 방지 (긴 파일에서 중요)
        )
        results: list[Segment] = []
        for seg in raw_segments:  # 제너레이터 — 순회하면서 실제 연산이 일어난다
            words = [Word(w.start, w.end, w.word) for w in (seg.words or [])] \
                if word_timestamps else []
            segment = Segment(seg.start, seg.end, seg.text.strip(), words)
            if not segment.text:
                continue
            results.append(segment)
            if on_segment is not None:
                on_segment(segment)
        return results, info.language

    def transcribe_array(self, audio: np.ndarray, word_timestamps: bool = False,
                         ) -> tuple[list[Segment], str]:
        """짧은 발화(실시간 조각)를 빠르게 전사한다. (세그먼트 리스트, 언어) 반환."""
        self.ensure_model()
        raw_segments, info = self._model.transcribe(
            audio,
            language=self.language,
            word_timestamps=word_timestamps,
            beam_size=1,
            condition_on_previous_text=False,
        )
        results: list[Segment] = []
        for seg in raw_segments:
            words = [Word(w.start, w.end, w.word) for w in (seg.words or [])] \
                if word_timestamps else []
            text = seg.text.strip()
            if text:
                results.append(Segment(seg.start, seg.end, text, words))
        return results, info.language
