# -*- coding: utf-8 -*-
"""한국어 파인튜닝 vs 원본 turbo 비교 — OpenVINO GPU(Iris Xe)로 가속.

- 한국어: FLEURS ko (파인튜닝의 학습 도메인과 다른 held-out → 공정)
- 영어:   FLEURS en (catastrophic forgetting 확인)
두 모델 모두 OpenVINO GPU로 동일 조건 비교.

실행: .venv-ov\\Scripts\\python.exe test_audio\\bench_finetune_ov.py [발화수]
"""
import json
import os
import re
import sys

import numpy as np

DATA = r"C:\Users\user\mp3TXT_local\test_audio\datasets"
BASELINE = r"C:\Users\user\mp3TXT_local\models\whisper-large-v3-turbo-int8-ov"
CANDIDATE = r"C:\Users\user\mp3TXT_local\models\ov-turbo-korean"


def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[m], n


def cer(ref, hyp):
    d, n = edit_distance(re.sub(r"\s+", "", ref), re.sub(r"\s+", "", hyp))
    return d / max(n, 1) * 100


def wer(ref, hyp):
    norm = lambda s: re.sub(r"[^\w\s]", "", s.lower()).split()
    d, n = edit_distance(norm(ref), norm(hyp))
    return d / max(n, 1) * 100


class Pipe:
    def __init__(self, model_dir, lang):
        import openvino_genai as ov_genai
        cache = os.path.join(os.path.expanduser("~"), ".mp3txt_local", "ov_cache")
        os.makedirs(cache, exist_ok=True)
        try:
            self.p = ov_genai.WhisperPipeline(model_dir, device="GPU", CACHE_DIR=cache)
        except TypeError:
            self.p = ov_genai.WhisperPipeline(model_dir, device="GPU")
        self.cfg = self.p.get_generation_config()
        self.cfg.task = "transcribe"
        # 언어 강제는 다국어 매핑이 있는 모델에서만 가능. 없으면(파인튜닝이
        # 다국어 설정을 잃은 경우) 모델 기본값으로 전사한다 — 그 결과가 곧
        # 영어 보존 여부의 증거가 된다.
        self.lang_forced = True
        try:
            self.cfg.language = f"<|{lang}|>"
        except Exception:
            self.lang_forced = False

    def __call__(self, audio):
        audio = audio.astype(np.float32)
        try:
            return str(self.p.generate(audio, self.cfg)).strip()
        except RuntimeError:
            # 언어 매핑이 없어 generate가 실패하면 언어 미지정으로 재시도
            cfg = self.p.get_generation_config()
            cfg.task = "transcribe"
            self.lang_forced = False
            return str(self.p.generate(audio, cfg)).strip()


def run_set(model_dir, subdir, lang, metric, n):
    meta = json.load(open(os.path.join(DATA, subdir, "meta.json"), encoding="utf-8"))[:n]
    pipe = Pipe(model_dir, lang)
    scores = []
    for row in meta:
        audio = np.load(os.path.join(DATA, subdir, f"{row['id']:04d}.npy"))
        scores.append(metric(row["text"], pipe(audio)))
    return float(np.mean(scores)), len(scores)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rows = [
        ("FLEURS 한국어 (CER)", "fleurs", "ko", cer),
        ("FLEURS 영어 (WER)", "fleurs_en", "en", wer),
        ("FLEURS 영어 (CER)", "fleurs_en", "en", cer),
    ]
    print(f"{'데이터셋':<22}{'baseline':>12}{'finetune':>12}{'변화':>10}")
    print("-" * 58)
    for label, subdir, lang, metric in rows:
        if not os.path.isdir(os.path.join(DATA, subdir)):
            print(f"{label:<22}  (데이터 없음)")
            continue
        b, cnt = run_set(BASELINE, subdir, lang, metric, n)
        c, _ = run_set(CANDIDATE, subdir, lang, metric, n)
        delta = c - b
        mark = "개선" if delta < -0.3 else ("악화" if delta > 0.3 else "동일")
        print(f"{label:<22}{b:>11.1f}%{c:>11.1f}%{delta:>+9.1f}p {mark}")
    print(f"\n발화 {n} · 두 모델 OpenVINO GPU(Iris Xe) int8 · FLEURS held-out")
    print("한국어=개선 기대, 영어=보존(동일) 기대.")


if __name__ == "__main__":
    main()
