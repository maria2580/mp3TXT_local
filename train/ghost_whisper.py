# -*- coding: utf-8 -*-
"""고스트 레이어(zero-init residual adapter)를 Whisper에 끼우는 모듈.

설계:
- 디코더 출력 hidden state와 lm_head(proj_out) 사이에 N겹의 잔차 블록을 끼운다.
- 각 블록은 `x = x + Wup(GELU(Wdown(LayerNorm(x))))` 이고, **Wup을 0으로 초기화**해
  학습 시작 시점엔 delta=0 → 완벽한 항등(원본과 비트 단위 동일 출력)이다.
  (ControlNet의 zero-conv, LoRA의 B행렬 0-초기화와 같은 원리)
- base 가중치는 전부 동결하고 고스트 블록만 학습한다.
- 틀린 케이스는 보정하도록, 맞은 케이스는 유지(리허설)하도록 학습한다.

inference 스택(faster-whisper/OpenVINO)은 학습 불가라, 학습은 여기 PyTorch
transformers로 하고 끝나면 병합·변환해 배포한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GhostBlock(nn.Module):
    """0-init 잔차 보정 블록. 초기 상태에서 입력을 그대로 통과시킨다."""

    def __init__(self, d_model: int, bottleneck: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, d_model)
        # 핵심: up을 0으로 초기화 → delta=0 → 시작 시 완벽한 항등
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


class GhostStack(nn.Module):
    """N겹 고스트 블록 + 원본 proj_out. 고스트를 통과시킨 뒤 어휘로 사영한다."""

    def __init__(self, proj_out: nn.Module, d_model: int,
                 n_layers: int = 4, bottleneck: int = 256):
        super().__init__()
        self.blocks = nn.ModuleList(
            [GhostBlock(d_model, bottleneck) for _ in range(n_layers)])
        self.proj_out = proj_out  # 원본 lm_head (동결)

    def forward(self, hidden):
        for block in self.blocks:
            hidden = block(hidden)
        return self.proj_out(hidden)


def attach_ghost(model, n_layers: int = 4, bottleneck: int = 256):
    """WhisperForConditionalGeneration에 고스트 스택을 끼우고 base를 동결한다.

    proj_out을 GhostStack으로 교체하므로 generate()도 자동으로 고스트를 거친다.
    학습 대상은 고스트 블록뿐. 돌려주는 것은 (model, 학습가능 파라미터 리스트).
    """
    d_model = model.config.d_model
    ghost = GhostStack(model.proj_out, d_model, n_layers, bottleneck)
    model.proj_out = ghost

    # base 전부 동결, 고스트 블록만 학습
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for block in ghost.blocks:
        for p in block.parameters():
            p.requires_grad = True
            trainable.append(p)
    return model, trainable


def ghost_state_dict(model) -> dict:
    """학습된 고스트 블록 가중치만 추출 (배포·재로딩용, 수 MB)."""
    return {k: v.cpu() for k, v in model.proj_out.blocks.state_dict().items()}


def load_ghost(model, state: dict) -> None:
    model.proj_out.blocks.load_state_dict(state)
