# -*- coding: utf-8 -*-
"""
capture — pyaudiowpatch로 마이크와 WASAPI loopback(시스템 재생 소리)을 캡처한다.

WASAPI shared mode는 장치 기본 샘플레이트/채널과 일치해야 하므로 스트림은
장치 기본값으로 열고, 콜백 안에서 모노화·16kHz 리샘플 후
(tag, chunk, t_mono) 튜플을 큐에 넣는다.
"""
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioDevice:
    index: int
    name: str
    is_loopback: bool
    default_sr: int
    channels: int


_pa_instance = None
_pa_lock = threading.Lock()


def _pa():
    """PyAudio 인스턴스 싱글턴. 첫 호출 때 지연 생성한다 (무거운 import 회피)."""
    global _pa_instance
    with _pa_lock:
        if _pa_instance is None:
            import pyaudiowpatch as pyaudio
            _pa_instance = pyaudio.PyAudio()
    return _pa_instance


def _as_device(info: dict, is_loopback: bool) -> AudioDevice:
    return AudioDevice(
        index=int(info["index"]),
        name=str(info["name"]),
        is_loopback=is_loopback,
        default_sr=int(info["defaultSampleRate"]),
        channels=max(int(info["maxInputChannels"]), 1),
    )


def list_input_devices() -> list[AudioDevice]:
    """일반 마이크 입력 장치 목록 (loopback 제외, 입력 채널이 있는 것만)."""
    p = _pa()  # pyaudiowpatch 미설치 등의 오류는 호출자에게 그대로 전달
    devices: list[AudioDevice] = []
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
        except Exception:
            continue
        if int(info.get("maxInputChannels", 0)) <= 0:
            continue
        if info.get("isLoopbackDevice", False):
            continue
        devices.append(_as_device(info, is_loopback=False))
    return devices


def list_loopback_devices() -> list[AudioDevice]:
    """WASAPI loopback 장치 목록. 기본 스피커의 loopback이 맨 앞에 온다."""
    p = _pa()
    devices: list[AudioDevice] = []
    try:
        for info in p.get_loopback_device_info_generator():
            devices.append(_as_device(info, is_loopback=True))
    except Exception:
        return []  # loopback 장치가 없거나 WASAPI 미지원이면 빈 리스트
    # 기본 스피커의 loopback을 맨 앞으로. 실패하면 generator 순서(첫 항목) 유지.
    try:
        default_index = int(p.get_default_wasapi_loopback()["index"])
        devices.sort(key=lambda d: d.index != default_index)
    except Exception:
        pass
    return devices


class CaptureStream:
    """단일 장치 캡처 스트림. 콜백에서 가공한 청크를 out_queue로 보낸다."""

    def __init__(self, device: AudioDevice, tag: str,
                 out_queue: "queue.Queue", target_sr: int = 16000) -> None:
        self.device = device
        self.tag = tag
        self.out_queue = out_queue
        self.target_sr = target_sr
        self._stream = None

    def start(self) -> None:
        """장치 기본 샘플레이트/채널로 스트림을 열고 캡처를 시작한다."""
        if self._stream is not None:
            return
        import pyaudiowpatch as pyaudio

        rate = int(self.device.default_sr)
        channels = self.device.channels
        tag, out_queue, target_sr = self.tag, self.out_queue, self.target_sr

        def callback(in_data, frame_count, time_info, status):
            # pyaudio 내부 스레드에서 실행되므로 가볍게 유지하고 예외는 삼킨다
            try:
                audio = np.frombuffer(in_data, dtype=np.float32)
                if channels > 1:
                    audio = audio.reshape(-1, channels).mean(axis=1)
                if rate != target_sr and audio.size:
                    n_out = max(int(round(audio.size * target_sr / rate)), 1)
                    audio = np.interp(
                        np.linspace(0.0, audio.size - 1.0, n_out),
                        np.arange(audio.size), audio)
                out_queue.put((tag, np.asarray(audio, dtype=np.float32),
                               time.monotonic()))
            except Exception:
                pass
            return (None, pyaudio.paContinue)

        self._stream = _pa().open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=self.device.index,
            frames_per_buffer=int(rate * 0.1),  # 100ms
            stream_callback=callback,
        )

    def stop(self) -> None:
        """스트림을 닫는다. 여러 번 불러도 안전하다."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop_stream()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
