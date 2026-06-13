# -*- coding: utf-8 -*-
"""실제 데이터셋(FLEURS 한국어 + ESC-50 소음)으로 전처리 front-end 효과를 검증한다.

각 FLEURS 발화에 대해 조건별 CER을 측정하고 전체 평균을 낸다:
  - clean         : 원본 vs AGC (AGC가 깨끗한 음성을 해치는가?)
  - quiet(x0.05)  : 원본 vs AGC (조용한 발화에 AGC가 돕는가?)
  - noisy(SNR=5)  : 원본 vs AGC+denoise (시끄러운 환경에서 돕는가?)
  - noisy(SNR=0)  : 원본 vs AGC+denoise (더 심한 소음에서는?)

전사는 OpenVINO GPU(Iris Xe, turbo int8)로 가속한다.
실행: .venv-ov\\Scripts\\python.exe test_audio\\bench_datasets.py [발화수]
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\user\mp3TXT_local")
from mp3txt.audio_frontend import enhance

DATA = r"C:\Users\user\mp3TXT_local\test_audio\datasets"
MODEL = r"C:\Users\user\mp3TXT_local\models\whisper-large-v3-turbo-int8-ov"
SR = 16000


def cer(ref, hyp):
    import re
    norm = lambda s: re.sub(r"\s+", "", s)
    ref, hyp = norm(ref), norm(hyp)
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[m] / n * 100


def mix_noise(speech, noise, snr_db):
    """speech에 noise를 목표 SNR로 섞는다 (noise를 타일링/자름)."""
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[:len(speech)]
    s_p = np.mean(speech ** 2) + 1e-12
    n_p = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(s_p / (n_p * 10 ** (snr_db / 10)))
    return (speech + noise * scale).astype(np.float32)


class Pipe:
    def __init__(self):
        import openvino_genai as ov_genai
        cache = os.path.join(os.path.expanduser("~"), ".mp3txt_local", "ov_cache")
        os.makedirs(cache, exist_ok=True)
        try:
            self.p = ov_genai.WhisperPipeline(MODEL, device="GPU", CACHE_DIR=cache)
        except TypeError:
            self.p = ov_genai.WhisperPipeline(MODEL, device="GPU")
        self.cfg = self.p.get_generation_config()
        self.cfg.language = "<|ko|>"
        self.cfg.task = "transcribe"

    def __call__(self, audio):
        return str(self.p.generate(audio.astype(np.float32), self.cfg)).strip()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    speech_meta = json.load(open(os.path.join(DATA, "fleurs", "meta.json"), encoding="utf-8"))[:n]
    noise_files = sorted(glob.glob(os.path.join(DATA, "esc50", "*.npy")))
    noises = [np.load(f) for f in noise_files]
    print(f"발화 {len(speech_meta)}개, 소음 {len(noises)}종, OpenVINO GPU 로딩...")
    pipe = Pipe()
    rng = np.random.RandomState(0)

    # 조건: (이름, 오디오 변형 함수, front-end 인자)
    conditions = [
        ("clean",        lambda a, nz: a,                       {"agc": True}),
        ("quiet x0.05",  lambda a, nz: a * 0.05,                {"agc": True}),
        ("noisy SNR=5",  lambda a, nz: mix_noise(a, nz, 5),     {"agc": True, "denoise": True}),
        ("noisy SNR=0",  lambda a, nz: mix_noise(a, nz, 0),     {"agc": True, "denoise": True}),
    ]
    agg = {c[0]: {"raw": [], "enh": []} for c in conditions}

    for k, row in enumerate(speech_meta):
        speech = np.load(os.path.join(DATA, "fleurs", f"{row['id']:04d}.npy"))
        ref = row["text"]
        nz = noises[rng.randint(len(noises))]
        for name, transform, fe in conditions:
            audio = transform(speech, nz)
            agg[name]["raw"].append(cer(ref, pipe(audio)))
            agg[name]["enh"].append(cer(ref, pipe(enhance(audio, **fe))))
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(speech_meta)} 처리...")

    print(f"\n{'조건':<14}{'원본 CER':>10}{'전처리 CER':>12}{'변화':>10}  비고")
    print("-" * 60)
    for name, _, fe in conditions:
        raw = float(np.mean(agg[name]["raw"]))
        enh = float(np.mean(agg[name]["enh"]))
        opt = "+".join(k for k, v in fe.items() if v)
        delta = enh - raw
        mark = "개선" if delta < -0.3 else ("악화" if delta > 0.3 else "동일")
        print(f"{name:<14}{raw:>9.1f}%{enh:>11.1f}%{delta:>+9.1f}p  {opt} → {mark}")
    print(f"\n발화 수 {len(speech_meta)} · 모델 large-v3-turbo(OV GPU) · 소음 ESC-50")


if __name__ == "__main__":
    main()
