# -*- coding: utf-8 -*-
"""추론 엔진 자동 감지 — 어떤 하드웨어로 전사할지 런타임에 실측으로 정한다.

우선순위 (auto):
  1. "cuda"         : NVIDIA GPU + CUDA 사용 가능 (faster-whisper, 가장 빠르고
                      워드 타임스탬프 지원)
  2. "openvino-gpu" : 인텔 GPU (OpenVINO, large-v3-turbo int8 전용 변환 모델)
  3. "cpu"          : faster-whisper CPU int8 (어디서나 동작)

감지는 추정(CPU 브랜드 등)이 아니라 라이브러리에 직접 장치 수를 물어서 한다.
감지에 성공해도 모델 로드가 실패할 수 있으므로(예: cuDNN 미설치) 호출 측은
build 실패 시 다음 순위로 강등해야 한다.
"""
from __future__ import annotations

ENGINE_LABELS = {
    "cuda": "NVIDIA GPU",
    "openvino-gpu": "인텔 GPU",
    "cpu": "CPU",
}


def cuda_device_count() -> int:
    """ctranslate2(faster-whisper 엔진)가 보는 CUDA 장치 수. 실패하면 0."""
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def openvino_gpu_available() -> bool:
    """OpenVINO 런타임이 GPU 장치(인텔 내장/외장)를 인식하는지."""
    try:
        import openvino
        return "GPU" in openvino.Core().available_devices
    except Exception:
        return False


def resolve_chain(preference: str = "auto") -> list[str]:
    """선호 엔진을 받아 시도 순서(폴백 체인)를 돌려준다. 마지막은 항상 cpu."""
    if preference == "cuda":
        chain = ["cuda"]
    elif preference == "openvino-gpu":
        chain = ["openvino-gpu"]
    elif preference == "cpu":
        return ["cpu"]
    else:  # auto — 실측 감지 순서대로
        chain = []
        if cuda_device_count() > 0:
            chain.append("cuda")
        if openvino_gpu_available():
            chain.append("openvino-gpu")
    if "cpu" not in chain:
        chain.append("cpu")
    return chain


def label(engine: str) -> str:
    return ENGINE_LABELS.get(engine, engine)
