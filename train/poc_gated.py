# -*- coding: utf-8 -*-
"""고스트 어댑터 v5 — 언어 게이트로 원본 분포 보존을 *구조적으로 보장*.

v4에서 다국어 배터리가 드러낸 문제: 한국어 보정이 일·중(문자 공유)을 망가뜨린다.
손실/증류로 막으려 했지만 실패(best=step0).

v5 아이디어: 손실이 아니라 **아키텍처**로 해결한다.
- 고스트를 **한국어일 때만 켠다**. 다른 언어는 어댑터를 통째로 우회.
- 그러면 en/ja/zh/de/fr는 원본과 비트 단위 동일 → 보존이 *측정*이 아니라 *보장*.
- 학습은 한국어만(잡음 보정 + 깨끗 리허설). 다국어 앵커 불필요.
- Whisper는 디코딩 첫 토큰으로 언어를 식별하므로, 추론 시 감지된 언어가 ko면
  게이트를 켜고 아니면 끈다 (PoC에선 평가셋 언어를 알고 있으니 그대로 게이트).

평가 배터리(ko/en/ja/zh/de/fr): 비한국어는 게이트 off → base와 0.0 차이를 *증명*.

실행:
  python train/poc_gated.py --model openai/whisper-large-v3-turbo \\
      --ko_train 160 --ko_heldout 30 --eval_per 30 --steps 2000 \\
      --lr 1e-4 --eval_every 200 --save train/ghost_v5.pt
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
from ghost_whisper import (attach_ghost, ghost_state, load_ghost_state,  # noqa: E402
                           set_ghost_active)

DATA = os.path.join(_ROOT, "test_audio", "datasets")
LANGS = {"en_us": ("fleurs_en", "en"), "ja_jp": ("fleurs_ja", "ja"),
         "cmn_hans_cn": ("fleurs_zh", "zh"), "de_de": ("fleurs_de", "de"),
         "fr_fr": ("fleurs_fr", "fr")}


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
    return json.load(open(os.path.join(DATA, subdir, "meta.json"), encoding="utf-8"))[:n]


def load_audio(subdir, idx):
    return np.load(os.path.join(DATA, subdir, f"{idx:04d}.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--ko_train", type=int, default=160)
    ap.add_argument("--ko_heldout", type=int, default=30)
    ap.add_argument("--eval_per", type=int, default=30)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--snr", type=float, default=0.0)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"장치: {device} | 모델: {args.model} | 고스트 {args.layers}겹 (언어 게이트)")
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

    # ---- 한국어 학습 데이터만 (잡음 보정 + 깨끗 리허설) ----
    ko = load_meta("fleurs", args.ko_train + args.ko_heldout)
    ko_train, ko_held = ko[:args.ko_train], ko[args.ko_train:]
    examples = []
    for row in ko_train:
        a = load_audio("fleurs", row["id"])
        examples.append((feats_of(a), labels_of(row["text"])))
        examples.append((feats_of(mix_noise(a, noises[rng.randint(len(noises))], args.snr)),
                         labels_of(row["text"])))
    rng.shuffle(examples)
    print(f"학습 예제 {len(examples)}개 (한국어 전용: 리허설 {len(ko_train)} + 잡음보정 {len(ko_train)})")

    def transcribe(audio, lang, ghost_on):
        set_ghost_active(model, ghost_on)   # 언어 게이트
        model.eval()
        with torch.no_grad():
            ids = model.generate(feats_of(audio).unsqueeze(0).to(device),
                                 language=lang, task="transcribe", max_new_tokens=128)
        model.train()
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def eval_cer(rows, subdir, lang, noisy, ghost_on):
        ev = np.random.RandomState(123)
        scores = []
        for row in rows:
            a = load_audio(subdir, row["id"])
            if noisy:
                a = mix_noise(a, noises[ev.randint(len(noises))], args.snr)
            scores.append(cer(row["text"], transcribe(a, lang, ghost_on)))
        return float(np.mean(scores))

    # 평가셋: (이름, subdir, lang, noisy, rows, ghost_on)
    # 한국어만 게이트 on, 나머지는 off(우회 → 원본과 동일해야 함)
    eval_sets = [("잡음 ko", "fleurs", "ko", True, ko_held, True),
                 ("깨끗 ko", "fleurs", "ko", False, ko_held, True)]
    for cfg_name, (subdir, lang) in LANGS.items():
        if os.path.isdir(os.path.join(DATA, subdir)):
            eval_sets.append((lang, subdir, lang, False,
                              load_meta(subdir, args.eval_per), False))

    def evaluate():
        return {name: eval_cer(rows, subdir, lang, noisy, ghost_on)
                for name, subdir, lang, noisy, rows, ghost_on in eval_sets}

    base = evaluate()
    print("[base] " + " | ".join(f"{k} {v:.1f}" for k, v in base.items()))

    # 비한국어는 게이트 off라 항상 base와 동일 → 페널티는 한국어 2지표만
    def penalty(m):
        return m["잡음 ko"] + 3 * max(0.0, m["깨끗 ko"] - base["깨끗 ko"])

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    best_state, best_pen, best_step = ghost_state(model), penalty(base), 0
    set_ghost_active(model, True)
    model.train()
    for step in range(1, args.steps + 1):
        set_ghost_active(model, True)  # 평가가 게이트를 off로 남겼을 수 있어 매 스텝 복원
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

    print(f"\n{'분포':<10}{'base':>8}{'+게이트고스트':>14}{'변화':>8}")
    print("-" * 42)
    for k in base:
        d = ghost[k] - base[k]
        goal = "(목표↓)" if k == "잡음 ko" else ("(보존)" if k == "깨끗 ko" else "(우회=보장)")
        print(f"{k:<10}{base[k]:>7.1f}{ghost[k]:>13.1f}{d:>+7.1f}  {goal}")

    if args.save:
        torch.save(best_state, args.save)
        print(f"\n고스트 저장: {args.save}")


if __name__ == "__main__":
    main()
