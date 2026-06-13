# -*- coding: utf-8 -*-
"""전처리 front-end(AGC/노이즈억제)가 전사 정확도에 주는 영향을 CER로 측정한다.

clean / quiet(×0.08) / noisy(SNR~5dB 화이트노이즈) 세 조건에서,
front-end 적용 전후의 CER을 비교한다. 기준 텍스트는 TTS 대본 그대로.

실행: .venv\\Scripts\\python.exe test_audio\\bench_frontend.py
"""
import sys
import wave

import numpy as np

sys.path.insert(0, r"C:\Users\user\mp3TXT_local")

from mp3txt.audio_frontend import enhance
from mp3txt.transcribe import Transcriber

WAV = r"C:\Users\user\mp3TXT_local\test_audio\test_meeting_16k.wav"
REFERENCE = (
    "안녕하세요 오늘 회의를 시작하겠습니다 첫 번째 안건은 신제품 출시 일정입니다 "
    "네 안녕하세요 개발 진행 상황부터 말씀드리겠습니다 현재 베타 테스트 단계에 있습니다 "
    "좋습니다 그러면 다음 달까지 출시가 가능할까요 마케팅 팀에서 일정을 물어보고 있습니다 "
    "품질 검증이 끝나면 가능합니다 다음 주 금요일에 검증 결과를 공유 드리겠습니다 "
    "알겠습니다 그럼 오늘 회의는 여기까지 하겠습니다 수고하셨습니다"
)


def load_wav(path):
    with wave.open(path, "rb") as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def cer(ref, hyp):
    ref = ref.replace(" ", ""); hyp = hyp.replace(" ", "")
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[m] / max(n, 1) * 100


def transcribe(model, audio):
    segs, _ = model.transcribe_long(audio, word_timestamps=False)
    return " ".join(s.text for s in segs)


def main():
    rng = np.random.RandomState(42)
    clean = load_wav(WAV)
    quiet = clean * 0.08
    noise = rng.randn(len(clean)).astype(np.float32)
    sig_rms = np.sqrt(np.mean(clean ** 2))
    noise = noise / np.sqrt(np.mean(noise ** 2)) * sig_rms * 0.56  # SNR ~5dB
    noisy = clean + noise

    model = Transcriber("large-v3-turbo", "int8", "ko")
    model.ensure_model()

    conditions = [
        ("clean", clean, {}),
        ("quiet(x0.08)", quiet, {"agc": True}),
        ("noisy(SNR~5dB)", noisy, {"agc": True, "denoise": True}),
    ]
    print(f"{'조건':<16}{'원본 CER':>10}{'전처리 CER':>12}  비고")
    print("-" * 56)
    for name, audio, fe in conditions:
        base = cer(REFERENCE, transcribe(model, audio))
        if fe:
            enh = cer(REFERENCE, transcribe(model, enhance(audio, **fe)))
            opt = ", ".join(k for k, v in fe.items() if v)
            mark = "개선" if enh < base - 0.5 else ("악화" if enh > base + 0.5 else "동일")
            print(f"{name:<16}{base:>9.1f}%{enh:>11.1f}%  {opt} → {mark}")
        else:
            print(f"{name:<16}{base:>9.1f}%{'(기준)':>12}")


if __name__ == "__main__":
    main()
