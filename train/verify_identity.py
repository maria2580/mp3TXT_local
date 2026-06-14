# -*- coding: utf-8 -*-
"""고스트 레이어의 핵심 보장 검증: 0-init 상태에서 원본과 출력이 동일한가?

이 설계의 전제 — "건드리기 전엔 원본을 절대 안 망친다" — 가 수학적으로
성립하는지 실제 모델로 확인한다. whisper-tiny로 CPU에서 빠르게 검증.

실행: .venv-ov\\Scripts\\python.exe train\\verify_identity.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghost_whisper import attach_ghost


def main():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    name = "openai/whisper-tiny"
    print(f"모델 로드: {name}")
    proc = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name)
    model.eval()

    # 더미 오디오 3초
    rng = np.random.RandomState(0)
    audio = rng.randn(16000 * 3).astype(np.float32) * 0.1
    feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
    dec_ids = torch.tensor([[model.config.decoder_start_token_id, 50364, 50259]])

    with torch.no_grad():
        base_logits = model(feats, decoder_input_ids=dec_ids).logits.clone()

    # 고스트 부착 (0-init)
    model, trainable = attach_ghost(model, n_layers=4, bottleneck=256)
    model.eval()
    with torch.no_grad():
        ghost_logits = model(feats, decoder_input_ids=dec_ids).logits

    diff = (base_logits - ghost_logits).abs().max().item()
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"고스트 학습 파라미터: {n_train:,} ({n_train / n_total * 100:.2f}% of {n_total:,})")
    print(f"원본 vs 고스트(0-init) 로짓 최대 차이: {diff:.2e}")
    assert diff < 1e-5, f"항등 성질 실패! 차이 {diff}"
    print("PASS — 0-init 고스트는 원본과 동일한 출력 (항등 보장 성립)")

    # 비-0 가중치를 주면 출력이 바뀌는지(학습 여지가 있는지)도 확인
    with torch.no_grad():
        blk = model.proj_out.blocks[0]
        blk.up.weight.add_(torch.randn_like(blk.up.weight) * 0.01)
        changed_logits = model(feats, decoder_input_ids=dec_ids).logits
    diff2 = (base_logits - changed_logits).abs().max().item()
    print(f"가중치 교란 후 차이: {diff2:.2e} (>0이어야 학습 가능)")
    assert diff2 > 1e-6, "고스트가 출력에 영향을 못 줌 — 학습 불가 구조"
    print("PASS — 고스트가 출력을 바꿀 수 있음 (보정 학습 가능)")


if __name__ == "__main__":
    main()
