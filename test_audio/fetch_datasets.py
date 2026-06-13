# -*- coding: utf-8 -*-
"""벤치마크용 공개 데이터셋 다운로드 — FLEURS(한국어 음성) + ESC-50(환경 소음).

- FLEURS ko_kr test: 실제 한국인 낭독 음성 + 정답 전사 (CC-BY)
- ESC-50: 실제 환경 소음 2000개 (CC-BY-NC)
둘 다 16kHz mono float32 npy + 메타로 저장해 벤치마크가 빠르게 읽게 한다.

실행: .venv-ov\\Scripts\\python.exe test_audio\\fetch_datasets.py [샘플수]
"""
import json
import os
import sys

import numpy as np

OUT = r"C:\Users\user\mp3TXT_local\test_audio\datasets"
SR = 16000


def resample(audio, sr_in):
    if sr_in == SR:
        return audio.astype(np.float32)
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=sr_in, target_sr=SR)


def decode_audio_field(a):
    """datasets의 audio 필드(decode=False)를 soundfile로 직접 디코드.

    torchcodec 없이 동작하도록 bytes 또는 path에서 읽는다. mono 16kHz 반환.
    """
    import io
    import soundfile as sf

    src = a.get("bytes")
    if src:
        data, sr = sf.read(io.BytesIO(src), dtype="float32", always_2d=False)
    else:
        data, sr = sf.read(a["path"], dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return resample(np.asarray(data, dtype=np.float32), sr)


def fetch_fleurs(n_samples: int):
    from datasets import Audio, load_dataset

    print(f"FLEURS ko_kr test 다운로드 중... (목표 {n_samples}개)")
    ds = load_dataset("google/fleurs", "ko_kr", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))  # torchcodec 우회
    d = os.path.join(OUT, "fleurs")
    os.makedirs(d, exist_ok=True)
    meta = []
    for i, row in enumerate(ds):
        if i >= n_samples:
            break
        audio = decode_audio_field(row["audio"])
        np.save(os.path.join(d, f"{i:04d}.npy"), audio)
        meta.append({"id": i, "text": row["transcription"],
                     "dur": round(len(audio) / SR, 1)})
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"  FLEURS 저장 완료: {len(meta)}개")


def fetch_esc50(n_samples: int):
    from datasets import Audio, load_dataset

    print(f"ESC-50 환경소음 다운로드 중... (목표 {n_samples}개)")
    ds = load_dataset("ashraq/esc50", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))  # torchcodec 우회
    d = os.path.join(OUT, "esc50")
    os.makedirs(d, exist_ok=True)
    # 음성과 겹치는 카테고리(아기울음 등)는 빼고 배경소음류만 고른다
    skip = {"crying_baby", "laughing", "sneezing", "coughing", "breathing",
            "snoring", "clapping", "footsteps"}
    meta, saved = [], 0
    for row in ds:
        if saved >= n_samples:
            break
        cat = row.get("category", "noise")
        if cat in skip:
            continue
        audio = decode_audio_field(row["audio"])
        np.save(os.path.join(d, f"{saved:04d}.npy"), audio)
        meta.append({"id": saved, "category": cat})
        saved += 1
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"  ESC-50 저장 완료: {len(meta)}개")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    os.makedirs(OUT, exist_ok=True)
    fetch_fleurs(n)
    fetch_esc50(40)
    print("완료.")


if __name__ == "__main__":
    main()
