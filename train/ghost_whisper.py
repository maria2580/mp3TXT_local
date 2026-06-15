# -*- coding: utf-8 -*-
"""고스트 어댑터(zero-init residual adapter)를 Whisper 디코더 레이어에 끼우는 모듈.

설계 (v2 — 레이어 내부 배치):
- 마지막 N개 디코더 레이어를 감싸, 각 레이어 출력 hidden state에 잔차 보정을 더한다:
  `h = layer(...); h = h + Wup(GELU(Wdown(LayerNorm(h))))`
- **Wup을 0으로 초기화** → 시작 시 delta=0 → 완벽한 항등(원본과 비트 단위 동일).
- v1은 어댑터를 lm_head 직전(출력 사영 바로 앞)에 뒀는데, 그 자리가 자기회귀
  생성에 지나치게 민감해 작은 변화도 반복 루프를 유발했다(PoC에서 확인). 그래서
  표준 어댑터 위치인 **디코더 레이어 내부(FFN 뒤)** 로 옮겼다.
- base 가중치는 전부 동결, 어댑터만 학습. KV 캐시는 원본 레이어가 계산하고
  어댑터는 hidden state만 후처리하므로 generation 캐싱과 호환된다.
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
        nn.init.zeros_(self.up.weight)  # 핵심: delta=0 → 시작 시 완벽한 항등
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


class LayerGhost(nn.Module):
    """디코더 레이어를 감싸 출력 hidden state에 고스트 보정을 더한다."""

    def __init__(self, layer: nn.Module, d_model: int, bottleneck: int):
        super().__init__()
        self.layer = layer
        self.adapter = GhostBlock(d_model, bottleneck)

    def forward(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        if isinstance(out, tuple):
            return (self.adapter(out[0]),) + out[1:]
        return self.adapter(out)


def attach_ghost(model, n_layers: int = 4, bottleneck: int = 256):
    """마지막 N개 디코더 레이어를 고스트로 감싸고 base를 동결한다.

    돌려주는 것: (model, 학습가능 파라미터 리스트). 학습 대상은 어댑터뿐.
    """
    d_model = model.config.d_model
    layers = model.model.decoder.layers
    n = min(n_layers, len(layers))
    wrapped = []
    for i in range(len(layers) - n, len(layers)):
        lg = LayerGhost(layers[i], d_model, bottleneck)
        layers[i] = lg
        wrapped.append(lg)
    model._ghost_wrapped = wrapped  # 상태 저장/복원 핸들

    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for lg in wrapped:
        for p in lg.adapter.parameters():
            p.requires_grad = True
            trainable.append(p)
    return model, trainable


def ghost_state(model) -> dict:
    """학습된 어댑터 가중치만 추출 (배포·재로딩용, 수 MB)."""
    return {f"{i}.{k}": v.detach().cpu().clone()
            for i, lg in enumerate(model._ghost_wrapped)
            for k, v in lg.adapter.state_dict().items()}


def load_ghost_state(model, state: dict) -> None:
    for i, lg in enumerate(model._ghost_wrapped):
        prefix = f"{i}."
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        lg.adapter.load_state_dict(sub)


def first_up_weight(model) -> "torch.Tensor":
    """검증용: 첫 어댑터의 up.weight 핸들 (교란 테스트에 사용)."""
    return model._ghost_wrapped[0].adapter.up.weight
