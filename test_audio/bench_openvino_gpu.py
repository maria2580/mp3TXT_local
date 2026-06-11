# -*- coding: utf-8 -*-
"""Iris Xe(OpenVINO GPU) vs CPU Whisper 벤치마크 — 44초 한국어 테스트 오디오 기준.

실행: .venv-ov\\Scripts\\python.exe test_audio\\bench_openvino_gpu.py <model_dir>
"""
import sys
import time
import wave

import numpy as np


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        data = w.readframes(w.getnframes())
    return (np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)


def bench(model_dir: str, device: str, audio: np.ndarray, runs: int = 2):
    import openvino_genai as ov_genai

    t0 = time.time()
    pipe = ov_genai.WhisperPipeline(model_dir, device=device)
    load_s = time.time() - t0

    config = pipe.get_generation_config()
    config.language = "<|ko|>"
    config.task = "transcribe"

    times, text = [], ""
    for _ in range(runs):  # 1회차는 GPU 커널 컴파일 포함 → 2회차가 실측
        t0 = time.time()
        result = pipe.generate(audio, config)
        times.append(time.time() - t0)
        text = str(result)
    return load_s, times, text


def main():
    model_dir = sys.argv[1]
    audio = load_wav(r"C:\Users\user\mp3TXT_local\test_audio\test_meeting_16k.wav")
    dur = len(audio) / 16000
    print(f"오디오 길이: {dur:.1f}초")
    for device in ("GPU", "CPU"):
        try:
            load_s, times, text = bench(model_dir, device, audio)
        except Exception as e:
            print(f"[{device}] 실패: {e}")
            continue
        best = min(times)
        print(f"[{device}] 로드 {load_s:.1f}초 | 전사 1회차 {times[0]:.1f}초, "
              f"2회차 {times[-1]:.1f}초 | RTF(2회차) {times[-1] / dur:.2f}")
        print(f"  결과 앞부분: {text[:80]}...")


if __name__ == "__main__":
    main()
