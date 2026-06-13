# -*- coding: utf-8 -*-
"""한국어 파인튜닝 Whisper vs 원본 turbo 비교 — 공정한 held-out 데이터로 측정.

- 한국어: FLEURS ko (파인튜닝이 Zeroth로 학습됐다면 FLEURS는 out-of-domain → 공정)
  → 파인튜닝이 우리 baseline보다 실제로 나은지
- 영어: FLEURS en → 파인튜닝이 영어를 망가뜨리는지 (catastrophic forgetting 확인)

두 모델 모두 faster-whisper(CPU int8)로 동일 조건 비교.
실행: .venv\\Scripts\\python.exe test_audio\\bench_finetune.py [발화수]
"""
import glob
import json
import os
import re
import sys

import numpy as np

DATA = r"C:\Users\user\mp3TXT_local\test_audio\datasets"
BASELINE = "large-v3-turbo"
CANDIDATE = r"C:\Users\user\mp3TXT_local\models\fw-turbo-korean"


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
    r = re.sub(r"\s+", "", ref)
    d, n = edit_distance(r, re.sub(r"\s+", "", hyp))
    return d / max(n, 1) * 100


def wer(ref, hyp):
    norm = lambda s: re.sub(r"[^\w\s]", "", s.lower()).split()
    d, n = edit_distance(norm(ref), norm(hyp))
    return d / max(n, 1) * 100


def load_model(name_or_path):
    from faster_whisper import WhisperModel
    return WhisperModel(name_or_path, device="cpu", compute_type="int8",
                        cpu_threads=max(2, (os.cpu_count() or 4) - 2))


def transcribe(model, audio, lang):
    segs, _ = model.transcribe(audio, language=lang, beam_size=5,
                               condition_on_previous_text=False)
    return " ".join(s.text for s in segs).strip()


def run_set(model, subdir, lang, metric, n):
    meta = json.load(open(os.path.join(DATA, subdir, "meta.json"), encoding="utf-8"))[:n]
    scores = []
    for row in meta:
        audio = np.load(os.path.join(DATA, subdir, f"{row['id']:04d}.npy"))
        scores.append(metric(row["text"], transcribe(model, audio, lang)))
    return float(np.mean(scores)), len(scores)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print("baseline(large-v3-turbo) 로딩...")
    base = load_model(BASELINE)
    print("candidate(fw-turbo-korean) 로딩...")
    cand = load_model(CANDIDATE)

    print(f"\n{'데이터셋':<22}{'baseline':>12}{'candidate':>12}{'변화':>10}")
    print("-" * 58)
    rows = [
        ("FLEURS 한국어 (CER)", "fleurs", "ko", cer),
        ("FLEURS 영어 (WER)", "fleurs_en", "en", wer),
        ("FLEURS 영어 (CER)", "fleurs_en", "en", cer),
    ]
    for label, subdir, lang, metric in rows:
        if not os.path.isdir(os.path.join(DATA, subdir)):
            print(f"{label:<22}  (데이터 없음 — 건너뜀)")
            continue
        b, cnt = run_set(base, subdir, lang, metric, n)
        c, _ = run_set(cand, subdir, lang, metric, n)
        delta = c - b
        mark = "개선" if delta < -0.3 else ("악화" if delta > 0.3 else "동일")
        print(f"{label:<22}{b:>11.1f}%{c:>11.1f}%{delta:>+9.1f}p {mark}")
    print(f"\n발화 수 {n} · 두 모델 모두 faster-whisper CPU int8 · FLEURS(held-out)")
    print("CER 낮을수록 좋음. 한국어=개선 기대, 영어=보존(동일) 기대.")


if __name__ == "__main__":
    main()
