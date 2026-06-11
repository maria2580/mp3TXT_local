# -*- coding: utf-8 -*-
"""
segmenter — Silero VAD(faster_whisper 내장)로 오디오 스트림을 발화 단위로 나눈다.

스트리밍 VAD API가 없으므로 청크를 버퍼에 누적하고 주기적으로 버퍼 전체를
재스캔한다. 마지막 음성 구간 뒤로 충분한 침묵이 확인되면 그 구간(들)을
발화로 방출하고 버퍼에서 소비된 부분을 제거한다.
"""
from typing import Callable, Optional

import numpy as np


class UtteranceSegmenter:
    """오디오 청크를 누적하다 침묵으로 닫힌 음성 구간을 발화로 방출한다."""

    MIN_UTTERANCE_S = 0.3   # 이보다 짧은 발화는 버린다
    RESCAN_AFTER_S = 0.4    # 직전 스캔 이후 이만큼 쌓여야 VAD를 다시 돌린다
    SILENCE_KEEP_S = 1.0    # 무음만 가득 찼을 때 남겨 둘 버퍼 길이 (무음 메모리 방지)

    def __init__(self, tag: str,
                 on_utterance: Callable[[str, np.ndarray, float], None],
                 min_silence_ms: int = 600, max_utterance_s: float = 15.0,
                 sr: int = 16000) -> None:
        self.tag = tag
        self.on_utterance = on_utterance  # on_utterance(tag, audio, t_start_mono)
        self.min_silence_ms = min_silence_ms
        self.max_utterance_s = max_utterance_s
        self.sr = sr
        self._chunks: list[np.ndarray] = []
        self._length = 0                   # 버퍼 총 샘플 수
        self._t0: Optional[float] = None   # 버퍼 시작 시각 (monotonic)
        self._scanned_len = 0              # 마지막 VAD 스캔 시점의 버퍼 길이
        self._vad_options = None

    def feed(self, chunk: np.ndarray, t_mono: float) -> None:
        """청크를 누적하고, 충분히 쌓였으면 발화 경계를 찾는다."""
        if chunk is None or chunk.size == 0:
            return
        if not self._chunks:
            self._t0 = t_mono
        self._chunks.append(chunk)
        self._length += chunk.size
        if self._length - self._scanned_len < int(self.RESCAN_AFTER_S * self.sr):
            return  # 스캔 주기 미달 — VAD 호출 비용 절약
        self._scan()

    def flush(self) -> None:
        """남은 버퍼에서 음성 구간을 찾아 강제 방출한다 (음성이 있을 때만)."""
        if not self._chunks:
            return
        audio = np.concatenate(self._chunks)
        t0 = self._t0
        self._reset()
        for seg in self._detect(audio):
            self._emit(audio, t0, seg["start"], seg["end"])

    # ---- 내부 구현 ----

    def _detect(self, audio: np.ndarray) -> list:
        """faster_whisper에 내장된 Silero VAD로 음성 구간을 찾는다 (지연 import)."""
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        if self._vad_options is None:
            self._vad_options = VadOptions(
                min_silence_duration_ms=self.min_silence_ms, speech_pad_ms=120)
        return get_speech_timestamps(audio, vad_options=self._vad_options)

    def _scan(self) -> None:
        audio = np.concatenate(self._chunks)
        self._chunks = [audio]
        self._scanned_len = audio.size
        speeches = self._detect(audio)
        min_silence = int(self.min_silence_ms * self.sr / 1000)

        # 마지막 음성 구간이 침묵으로 닫혔으면 모든 구간을 방출하고 소비
        if speeches and audio.size - speeches[-1]["end"] >= min_silence:
            t0 = self._t0
            for seg in speeches:
                self._emit(audio, t0, seg["start"], seg["end"])
            self._consume(audio, speeches[-1]["end"])
            return

        # 버퍼가 최대 길이를 넘으면 강제 처리 (지연·메모리 상한)
        if audio.size > int(self.max_utterance_s * self.sr):
            if speeches:
                t0 = self._t0
                for seg in speeches:
                    self._emit(audio, t0, seg["start"], seg["end"])
                self._reset()
            else:
                self._consume(audio, audio.size - int(self.SILENCE_KEEP_S * self.sr))

    def _emit(self, audio: np.ndarray, t0: Optional[float],
              start: int, end: int) -> None:
        """음성 구간 하나를 발화로 콜백. 너무 짧으면 버린다."""
        if end - start < int(self.MIN_UTTERANCE_S * self.sr):
            return
        t_start = (t0 or 0.0) + start / self.sr
        self.on_utterance(self.tag, audio[start:end].copy(), t_start)

    def _consume(self, audio: np.ndarray, n: int) -> None:
        """버퍼 앞쪽 n샘플을 버리고 시작 시각을 그만큼 보정한다."""
        rest = audio[n:]
        if rest.size == 0:
            self._reset()
            return
        self._chunks = [rest]
        self._length = rest.size
        self._scanned_len = rest.size
        if self._t0 is not None:
            self._t0 += n / self.sr

    def _reset(self) -> None:
        self._chunks = []
        self._length = 0
        self._scanned_len = 0
        self._t0 = None
