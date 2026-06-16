# -*- coding: utf-8 -*-
"""고스트 어댑터 v4 — 지식 증류로 '원본 분포 보존'을 데이터 없이 달성.

연구 목표: 원본 학습 데이터 없이, 새 분포(잡음 한국어)는 개선하고 원본 분포는
유지(또는 개선)한다.

방법 (source-free continual learning, LwF 계열):
- 원본 데이터가 없으니 **원본 모델(교사)** 을 데이터 대신 쓴다.
- 학습 시작 전(고스트=0, 즉 모델==원본), 다국어 앵커 오디오를 교사가 전사한
  결과를 **의사 라벨**로 만든다 → 시퀀스 레벨 지식 증류.
- 학습 = 한국어 잡음 보정(정답 라벨) + 다국어 앵커 보존(교사 의사 라벨, CE).
  앵커가 원본 분포 전체에서의 행동을 고정한다.

평가 = 다국어·다도메인 배터리로 '퇴보'를 실제로 탐지 (FLEURS ko/en/ja/zh/de/fr).
FLEURS 영어 하나로는 원본 분포 보존을 주장할 수 없기 때문.

실행:
  python train/poc_distill.py --model openai/whisper-large-v3-turbo \\
      --ko_train 160 --ko_heldout 30 --anchor_per 40 --eval_per 30 \\
      --steps 2500 --lr 1e-4 --eval_every 250 --save train/ghost_v4.pt
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from ghost_whisper import attach_ghost, ghost_state, load_ghost_state  # noqa: E402

DATA = os.path.join(_ROOT, "test_audio", "datasets")

# FLEURS config → (서브폴더, Whisper 언어코드). en은 보존+한국어 외 언어들이 앵커.
LANGS = {
    "en_us": ("fleurs_en", "en"),
    "ja_jp": ("fleurs_ja", "ja"),
    "cmn_hans_cn": ("fleurs_zh", "zh"),
    "de_de": ("fleurs_de", "de"),
    "fr_fr": ("fleurs_fr", "fr"),
}


def cer(ref, hyp):
    r, h = re.sub(r"\s+", "", ref), re.sub(r"\s+", "", hyp)
    n, m = len(r), len(h)
    if n == 0:
        return 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return dp[m] / n * 100


def mix_noise(speech, noise, snr_db):
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[:len(speech)]
    s_p = np.mean(speech ** 2) + 1e-12
    n_p = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(s_p / (n_p * 10 ** (snr_db / 10)))
    return (speech + noise * scale).astype(np.float32)


def load_meta(subdir, n):
    path = os.path.join(DATA, subdir, "meta.json")
    return json.load(open(path, encoding="utf-8"))[:n]


def load_audio(subdir, idx):
    return np.load(os.path.join(DATA, subdir, f"{idx:04d}.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--ko_train", type=int, default=160)
    ap.add_argument("--ko_heldout", type=int, default=30)
    ap.add_argument("--anchor_per", type=int, default=40, help="앵커 언어당 학습 발화 수")
    ap.add_argument("--eval_per", type=int, default=30, help="평가 언어당 발화 수")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--snr", type=float, default=0.0)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"장치: {device} | 모델: {args.model} | 고스트 {args.layers}겹")
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device).float()
    model, trainable = attach_ghost(model, n_layers=args.layers)
    model = model.to(device)
    print(f"학습 파라미터(고스트만): {sum(p.numel() for p in trainable):,}")

    noises = [np.load(f) for f in sorted(glob.glob(os.path.join(DATA, "esc50", "*.npy")))]
    rng = np.random.RandomState(0)

    def feats_of(audio):
        return proc(audio, sampling_rate=16000, return_tensors="pt").input_features[0]

    def labels_of(text):
        return proc.tokenizer(text, return_tensors="pt").input_ids[0]

    def transcribe(audio, lang):
        model.eval()
        with torch.no_grad():
            ids = model.generate(feats_of(audio).unsqueeze(0).to(device),
                                 language=lang, task="transcribe", max_new_tokens=128)
        model.train()
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    # ---- 한국어 task 데이터 (정답 라벨) ----
    ko = load_meta("fleurs", args.ko_train + args.ko_heldout)
    ko_train, ko_held = ko[:args.ko_train], ko[args.ko_train:]
    examples = []
    for row in ko_train:
        a = load_audio("fleurs", row["id"])
        examples.append((feats_of(a), labels_of(row["text"])))                       # 리허설
        examples.append((feats_of(mix_noise(a, noises[rng.randint(len(noises))], args.snr)),
                         labels_of(row["text"])))                                      # 잡음 보정

    # ---- 다국어 앵커 (교사=현재 0-init 모델이 의사 라벨 생성) ----
    # 고스트가 아직 0이라 model==원본. 이 시점 전사가 곧 '원본 행동'.
    print("다국어 앵커 의사 라벨 생성(원본 모델 전사)...")
    n_anchor = 0
    for cfg_name, (subdir, lang) in LANGS.items():
        if not os.path.isdir(os.path.join(DATA, subdir)):
            print(f"  (없음: {subdir} 건너뜀)")
            continue
        rows = load_meta(subdir, args.eval_per + args.anchor_per)[args.eval_per:]
        for row in rows:
            a = load_audio(subdir, row["id"])
            pseudo = transcribe(a, lang)          # 교사 의사 라벨
            if pseudo:
                examples.append((feats_of(a), labels_of(pseudo)))
                n_anchor += 1
    rng.shuffle(examples)
    print(f"학습 예제 {len(examples)}개 (한국어 {len(ko_train) * 2} + 다국어 앵커 {n_anchor})")

    # ---- 평가 배터리 ----
    def eval_cer(rows, subdir, lang, noisy):
        ev = np.random.RandomState(123)
        scores = []
        for row in rows:
            a = load_audio(subdir, row["id"])
            if noisy:
                a = mix_noise(a, noises[ev.randint(len(noises))], args.snr)
            scores.append(cer(row["text"], transcribe(a, lang)))
        return float(np.mean(scores))

    eval_sets = [("잡음 ko", "fleurs", "ko", True, ko_held),
                 ("깨끗 ko", "fleurs", "ko", False, ko_held)]
    for cfg_name, (subdir, lang) in LANGS.items():
        if os.path.isdir(os.path.join(DATA, subdir)):
            eval_sets.append((f"{lang}", subdir, lang, False,
                              load_meta(subdir, args.eval_per)))

    def evaluate():
        return {name: eval_cer(rows, subdir, lang, noisy)
                for name, subdir, lang, noisy, rows in eval_sets}

    base = evaluate()
    print("[base] " + " | ".join(f"{k} {v:.1f}" for k, v in base.items()))

    # 페널티: 잡음 한국어 개선을 보되, 원본 분포(잡음 ko 제외 전부) 퇴보엔 큰 벌점
    preserve = [name for name, *_ in eval_sets if name != "잡음 ko"]

    def penalty(m):
        p = m["잡음 ko"]
        for k in preserve:
            p += 3 * max(0.0, m[k] - base[k])  # 어느 분포든 퇴보하면 벌점
        return p

    # ---- 학습 ----
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    best_state, best_pen, best_step = ghost_state(model), penalty(base), 0
    model.train()
    for step in range(1, args.steps + 1):
        feats, labels = examples[step % len(examples)]
        out = model(input_features=feats.unsqueeze(0).to(device),
                    labels=labels.unsqueeze(0).to(device))
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate()
            pen = penalty(m)
            tag = ""
            if pen < best_pen:
                best_pen, best_step, best_state = pen, step, ghost_state(model)
                tag = " *best"
            print(f"  step {step:4d} loss {out.loss.item():.3f} | "
                  + " ".join(f"{k} {m[k]:.1f}" for k in m) + tag)

    print(f"\nbest = step {best_step}")
    load_ghost_state(model, best_state)
    ghost = evaluate()

    print(f"\n{'분포':<10}{'base':>8}{'+고스트':>9}{'변화':>8}")
    print("-" * 36)
    for k in base:
        d = ghost[k] - base[k]
        goal = "(목표↓)" if k == "잡음 ko" else "(보존)"
        print(f"{k:<10}{base[k]:>7.1f}{ghost[k]:>8.1f}{d:>+7.1f}  {goal}")

    if args.save:
        torch.save(best_state, args.save)
        print(f"\n고스트 저장: {args.save}")


if __name__ == "__main__":
    main()
