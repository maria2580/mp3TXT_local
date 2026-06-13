# -*- coding: utf-8 -*-
"""잡음 환경에서 turbo vs large-v3 full 비교 — large-v3의 여분 용량이
어려운 음성에서 실제로 도움이 되는지 검증.

FLEURS 한국어/영어 발화에 ESC-50 실제 소음을 SNR별로 섞어, 두 모델의
CER(한국어)/WER(영어)을 비교한다. 둘 다 OpenVINO GPU.

실행: .venv-ov\\Scripts\\python.exe test_audio\\bench_noisy_models.py [발화수]
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\user\mp3TXT_local\test_audio")
from bench_finetune_ov import Pipe, cer, wer  # noqa: E402

DATA = r"C:\Users\user\mp3TXT_local\test_audio\datasets"
TURBO = r"C:\Users\user\mp3TXT_local\models\whisper-large-v3-turbo-int8-ov"
LARGE = r"C:\Users\user\mp3TXT_local\models\ov-large-v3"


def mix_noise(speech, noise, snr_db):
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[:len(speech)]
    s_p = np.mean(speech ** 2) + 1e-12
    n_p = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(s_p / (n_p * 10 ** (snr_db / 10)))
    return (speech + noise * scale).astype(np.float32)


def eval_model(model_dir, subdir, lang, metric, snr, noises, rng, n):
    meta = json.load(open(os.path.join(DATA, subdir, "meta.json"), encoding="utf-8"))[:n]
    pipe = Pipe(model_dir, lang)
    scores = []
    for row in meta:
        a = np.load(os.path.join(DATA, subdir, f"{row['id']:04d}.npy"))
        nz = noises[rng.randint(len(noises))]
        audio = a if snr is None else mix_noise(a, nz, snr)
        scores.append(metric(row["text"], pipe(audio)))
    return float(np.mean(scores))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    noises = [np.load(f) for f in sorted(glob.glob(os.path.join(DATA, "esc50", "*.npy")))]

    sets = [
        ("한국어 CER", "fleurs", "ko", cer),
        ("영어 WER", "fleurs_en", "en", wer),
    ]
    snrs = [("SNR=10", 10), ("SNR=5", 5), ("SNR=0", 0)]

    print(f"잡음 환경 비교 (발화 {n} · ESC-50 소음 · OpenVINO GPU)")
    print(f"{'조건':<20}{'turbo':>10}{'large-v3':>11}{'차이':>9}")
    print("-" * 52)
    for set_label, subdir, lang, metric in sets:
        for snr_label, snr in snrs:
            rng = np.random.RandomState(0)  # 두 모델에 같은 소음 배치
            t = eval_model(TURBO, subdir, lang, metric, snr, noises, rng, n)
            rng = np.random.RandomState(0)
            l = eval_model(LARGE, subdir, lang, metric, snr, noises, rng, n)
            delta = l - t
            mark = "large-v3 우세" if delta < -0.5 else ("turbo 우세" if delta > 0.5 else "동급")
            print(f"{set_label+' '+snr_label:<20}{t:>9.1f}%{l:>10.1f}%{delta:>+8.1f}p {mark}")
    print("\nCER/WER 낮을수록 좋음. large-v3가 어려운 음성에서 이기는지 확인.")


if __name__ == "__main__":
    main()
