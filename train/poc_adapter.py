# -*- coding: utf-8 -*-
"""고스트 어댑터 PoC — 잡음 한국어를 보정하되 깨끗한 음성·영어는 보존하는지 검증.

학습 데이터(모두 정답 라벨과 함께):
  - 깨끗한 한국어 (correct/rehearsal): "맞은 건 그대로 유지" 학습
  - 잡음 한국어 (error): "틀린 건 정답으로 보정" 학습
검증(held-out, 학습에 안 쓴 발화):
  - 잡음 한국어 CER  → 내려가야 성공 (보정됨)
  - 깨끗한 한국어 CER → 유지돼야 성공 (안 망가짐)
  - 영어 WER         → 유지돼야 성공 (catastrophic forgetting 없음)

CPU 스모크: --model openai/whisper-tiny --train 12 --steps 30 (원리 확인용)
실전(GPU PC): --model openai/whisper-large-v3-turbo --train 200 --steps 2000

실행: .venv-ov\\Scripts\\python.exe train\\poc_adapter.py [옵션]
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
from ghost_whisper import attach_ghost  # noqa: E402

DATA = os.path.join(_ROOT, "test_audio", "datasets")


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


def wer(ref, hyp):
    norm = lambda s: re.sub(r"[^\w\s]", "", s.lower()).split()
    r, h = norm(ref), norm(hyp)
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
    meta = json.load(open(os.path.join(DATA, subdir, "meta.json"), encoding="utf-8"))
    return meta[:n]


def load_audio(subdir, idx):
    return np.load(os.path.join(DATA, subdir, f"{idx:04d}.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny")
    ap.add_argument("--train", type=int, default=12, help="학습 발화 수(한국어)")
    ap.add_argument("--heldout", type=int, default=8, help="검증 발화 수")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)  # 보수적 — 과적합/발산 방지
    ap.add_argument("--clip", type=float, default=1.0, help="grad clip norm")
    ap.add_argument("--eval_every", type=int, default=200, help="held-out 평가 주기")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--snr", type=float, default=0.0, help="잡음 SNR(dB)")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"장치: {device} | 모델: {args.model} | 고스트 {args.layers}겹")
    proc = WhisperProcessor.from_pretrained(args.model)
    # float32로 통일 — 체크포인트가 fp16이면 입력(float32)과 dtype이 안 맞아 에러난다.
    # 어댑터 학습 안정성에도 fp32가 유리 (turbo 809M은 24GB GPU에 여유).
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device).float()
    model, trainable = attach_ghost(model, n_layers=args.layers)
    model = model.to(device)  # 새로 부착한 고스트 블록도 GPU로 (기본 CPU 생성)
    n_train_p = sum(p.numel() for p in trainable)
    print(f"학습 파라미터(고스트만): {n_train_p:,}")

    noises = [np.load(f) for f in sorted(glob.glob(os.path.join(DATA, "esc50", "*.npy")))]
    rng = np.random.RandomState(0)

    # 학습 셋: 깨끗한 한국어(리허설) + 잡음 한국어(보정), 모두 정답 라벨
    ko = load_meta("fleurs", args.train + args.heldout)
    train_ko, held_ko = ko[:args.train], ko[args.train:args.train + args.heldout]
    held_en = load_meta("fleurs_en", args.heldout)

    def make_example(subdir, row, noisy):
        audio = load_audio(subdir, row["id"])
        if noisy:
            audio = mix_noise(audio, noises[rng.randint(len(noises))], args.snr)
        feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
        labels = proc.tokenizer(row["text"], return_tensors="pt").input_ids[0]
        return feats, labels

    examples = []
    for row in train_ko:
        examples.append(make_example("fleurs", row, noisy=False))  # 리허설
        examples.append(make_example("fleurs", row, noisy=True))   # 보정
    print(f"학습 예제: {len(examples)}개 (리허설 {len(train_ko)} + 보정 {len(train_ko)})")

    # ---- 평가 함수 (학습 전후 공용) ----
    def transcribe(feats):
        with torch.no_grad():
            ids = model.generate(feats.unsqueeze(0).to(device), max_new_tokens=128)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def eval_set(rows, subdir, noisy, metric):
        ev = np.random.RandomState(123)  # 평가용 고정 노이즈 (학습 rng와 분리)
        scores = []
        for row in rows:
            audio = load_audio(subdir, row["id"])
            if noisy:
                audio = mix_noise(audio, noises[ev.randint(len(noises))], args.snr)
            feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
            scores.append(metric(row["text"], transcribe(feats)))
        return float(np.mean(scores))

    def evaluate():
        model.eval()
        m = {"잡음 한국어 CER": eval_set(held_ko, "fleurs", True, cer),
             "깨끗 한국어 CER": eval_set(held_ko, "fleurs", False, cer),
             "영어 WER": eval_set(held_en, "fleurs_en", False, wer)}
        model.train()
        return m

    # base = 고스트 0-init(=학습 전) 성능
    base = evaluate()
    print(f"[base] 잡음ko {base['잡음 한국어 CER']:.1f} | 깨끗ko "
          f"{base['깨끗 한국어 CER']:.1f} | 영어 {base['영어 WER']:.1f}")

    def penalty(m):  # 잡음 보정을 보되, 깨끗·영어 퇴보엔 큰 벌점 (보존 우선)
        return (m["잡음 한국어 CER"]
                + 3 * max(0.0, m["깨끗 한국어 CER"] - base["깨끗 한국어 CER"])
                + 3 * max(0.0, m["영어 WER"] - base["영어 WER"]))

    # 학습 (배치 1, teacher forcing, grad clip, 주기적 평가로 best 선택)
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    best_state = {k: v.clone() for k, v in model.proj_out.blocks.state_dict().items()}
    best_pen, best_step = penalty(base), 0
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
                best_pen, best_step = pen, step
                best_state = {k: v.clone()
                              for k, v in model.proj_out.blocks.state_dict().items()}
                tag = " *best"
            print(f"  step {step:4d} loss {out.loss.item():.3f} | 잡음ko "
                  f"{m['잡음 한국어 CER']:.1f} 깨끗ko {m['깨끗 한국어 CER']:.1f} "
                  f"영어 {m['영어 WER']:.1f}{tag}")

    # best 고스트 복원 → 최종 보고
    print(f"\nbest = step {best_step}")
    model.proj_out.blocks.load_state_dict(best_state)
    ghost_state = best_state
    ghost = evaluate()

    print(f"\n{'지표':<18}{'base':>9}{'+고스트':>10}{'변화':>9}  목표")
    print("-" * 56)
    goals = {"잡음 한국어 CER": "↓ 보정", "깨끗 한국어 CER": "= 유지",
             "영어 WER": "= 보존"}
    for k in base:
        d = ghost[k] - base[k]
        print(f"{k:<18}{base[k]:>8.1f}%{ghost[k]:>9.1f}%{d:>+8.1f}p  {goals[k]}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save(ghost_state, args.save)
        print(f"\n고스트 가중치 저장: {args.save}")


if __name__ == "__main__":
    main()
