# -*- coding: utf-8 -*-
"""OpenVINO(인텔 GPU) Whisper 백엔드 — 실시간 모드용.

faster-whisper의 Transcriber와 같은 transcribe_array 계약을 구현해
RealtimeEngine이 엔진 종류와 무관하게 동작하도록 한다.

- 모델: optimum-cli로 변환한 large-v3-turbo int8 (models\\ 폴더, 약 780MB)
- 워드 타임스탬프는 미지원 → 발화 전체를 세그먼트 1개로 돌려준다
  (실시간 모드는 발화 단위 표시라 문제 없음)
- GPU 커널 컴파일 결과를 캐시해 2회차 로드부터 빨라진다
"""
from __future__ import annotations

import os

import numpy as np

from .transcribe import Segment

# Whisper 언어 토큰 (OpenVINO genai 형식)
_LANG_TOKEN = {
    "ko": "<|ko|>", "en": "<|en|>", "ja": "<|ja|>", "zh": "<|zh|>",
    "de": "<|de|>", "fr": "<|fr|>", "es": "<|es|>", "ru": "<|ru|>",
}


def _guess_lang(text: str) -> str:
    """언어 자동 감지 모드에서 번역 라우팅용 간이 판별 (문자 체계 기반)."""
    counts = {"ko": 0, "ja": 0, "zh": 0}
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:
            counts["ko"] += 1
        elif 0x3040 <= code <= 0x30FF:
            counts["ja"] += 1
        elif 0x4E00 <= code <= 0x9FFF:
            counts["zh"] += 1
    best = max(counts, key=counts.get)
    return best if counts[best] >= max(3, len(text) // 20) else "en"


class OvTranscriber:
    """OpenVINO WhisperPipeline 래퍼. 인스턴스는 스레드 1개에서만 사용할 것."""

    def __init__(self, model_dir: str, language: str | None = None,
                 device: str = "GPU"):
        self.model_dir = model_dir
        self.device = device
        self.language = None if language in (None, "", "auto") else language
        self._pipe = None
        self._config = None

    def ensure_model(self) -> None:
        """파이프라인을 로드한다 (GPU는 최초 커널 컴파일로 수십 초 걸릴 수 있음)."""
        if self._pipe is not None:
            return
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"OpenVINO 변환 모델이 없습니다: {self.model_dir}\n"
                "  optimum-cli export openvino --model openai/whisper-large-v3-turbo"
                " --weight-format int8 <폴더> 로 변환할 수 있습니다.")
        import openvino_genai as ov_genai

        cache_dir = os.path.join(os.path.expanduser("~"), ".mp3txt_local", "ov_cache")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            self._pipe = ov_genai.WhisperPipeline(
                self.model_dir, device=self.device, CACHE_DIR=cache_dir)
        except TypeError:  # 구버전 genai는 속성 kwargs 미지원
            self._pipe = ov_genai.WhisperPipeline(self.model_dir, device=self.device)
        config = self._pipe.get_generation_config()
        config.task = "transcribe"
        if self.language in _LANG_TOKEN:
            config.language = _LANG_TOKEN[self.language]
        self._config = config

    def transcribe_array(self, audio: np.ndarray, word_timestamps: bool = False,
                         ) -> tuple[list[Segment], str]:
        """짧은 발화를 전사한다. (세그먼트 리스트, 언어코드) 반환.

        word_timestamps는 받기만 하고 무시한다 (OpenVINO 경로 미지원).
        """
        self.ensure_model()
        result = self._pipe.generate(audio.astype(np.float32), self._config)
        text = str(result).strip()
        if not text:
            return [], self.language or "ko"
        lang = self.language or _guess_lang(text)
        duration = float(len(audio)) / 16000.0
        return [Segment(0.0, duration, text, [])], lang

    def transcribe_long(self, audio: np.ndarray, word_timestamps: bool = True,
                        on_segment=None) -> tuple[list[Segment], str]:
        """긴 오디오(파일)를 세그먼트 타임스탬프와 함께 전사한다.

        OpenVINO genai의 return_timestamps로 30초 초과 오디오를 자동 청크 처리하고
        각 청크의 시작/끝 시각을 얻는다. 워드 단위 타임스탬프는 없으므로
        (word_timestamps 인자는 무시) 화자 배정은 세그먼트 단위로 이뤄진다.
        """
        self.ensure_model()
        config = self._pipe.get_generation_config()
        config.task = "transcribe"
        config.return_timestamps = True
        if self.language in _LANG_TOKEN:
            config.language = _LANG_TOKEN[self.language]
        result = self._pipe.generate(audio.astype(np.float32), config)

        segments: list[Segment] = []
        chunks = getattr(result, "chunks", None)
        if chunks:
            for ch in chunks:
                text = (ch.text or "").strip()
                if not text:
                    continue
                start = float(getattr(ch, "start_ts", 0.0) or 0.0)
                end = float(getattr(ch, "end_ts", start) or start)
                if end < start:  # genai가 끝 시각을 -1로 줄 때 보정
                    end = float(len(audio)) / 16000.0
                seg = Segment(start, end, text, [])
                segments.append(seg)
                if on_segment is not None:
                    on_segment(seg)
        else:  # 타임스탬프 미지원 빌드 폴백 — 전체를 한 세그먼트로
            text = str(result).strip()
            if text:
                seg = Segment(0.0, float(len(audio)) / 16000.0, text, [])
                segments.append(seg)
                if on_segment is not None:
                    on_segment(seg)

        lang = self.language or (_guess_lang(segments[0].text) if segments else "ko")
        return segments, lang
