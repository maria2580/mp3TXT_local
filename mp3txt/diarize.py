# -*- coding: utf-8 -*-
"""pyannote 화자분리 래퍼 — speaker-diarization-community-1 (완전 로컬 추론).

모델 가중치는 Hugging Face에서 최초 1회 내려받으며, 게이트 모델이라
hf.co/pyannote/speaker-diarization-community-1 에서 약관 동의 후 발급한
토큰이 필요하다. 토큰이 없으면 None을 돌려주고 호출 측은 화자 없이 진행한다.
"""
from __future__ import annotations

import numpy as np

MODEL_ID = "pyannote/speaker-diarization-community-1"

# (start_sec, end_sec, speaker_label) 목록
Turn = tuple[float, float, str]


def diarize(audio: np.ndarray, hf_token: str | None, sr: int = 16000,
            num_speakers: int | None = None) -> list[Turn] | None:
    """화자분리를 수행해 시간순 (시작, 끝, 화자라벨) 목록을 돌려준다.

    토큰이 없거나 모델 로드/추론에 실패하면 사유를 출력하고 None을 돌려준다
    (배치 변환은 화자 표기 없이 계속 진행할 수 있도록).
    """
    if not hf_token:
        print("안내: Hugging Face 토큰이 없어 화자분리를 건너뜁니다.")
        print("  화자분리를 쓰려면:")
        print(f"  1) https://huggingface.co/{MODEL_ID} 에서 약관 동의")
        print("  2) https://huggingface.co/settings/tokens 에서 토큰(Read) 발급")
        print('  3) mp3txt_local.py --set-token <토큰> 으로 저장')
        return None

    try:
        import warnings
        # pyannote 내장 디코더(torchcodec)용 경고 억제 — 우리는 파형을 메모리로
        # 직접 전달하므로 해당 디코더를 쓰지 않는다 (경고만 길고 실해 없음)
        warnings.filterwarnings(
            "ignore", message=r"\s*torchcodec is not installed", category=UserWarning)
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        print(f"안내: pyannote.audio를 불러올 수 없어 화자분리를 건너뜁니다. ({e})")
        return None

    try:
        try:
            pipeline = Pipeline.from_pretrained(MODEL_ID, token=hf_token)
        except TypeError:  # pyannote.audio 3.x는 인자 이름이 다르다
            pipeline = Pipeline.from_pretrained(MODEL_ID, use_auth_token=hf_token)
        if pipeline is None:
            raise RuntimeError(
                "파이프라인 로드 결과가 비어 있습니다. 모델 약관 동의 여부를 확인하세요.")
    except Exception as e:
        print(f"안내: 화자분리 모델 로드 실패 — 화자 표기 없이 진행합니다.\n  ({e})")
        return None

    # NVIDIA GPU가 있으면 파이프라인을 GPU로 (CUDA 빌드 torch 필요 — README 참고)
    try:
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            print("화자분리: NVIDIA GPU 사용")
    except Exception as e:
        print(f"안내: 화자분리 GPU 전환 실패 — CPU로 진행합니다. ({e})")

    try:
        torch.set_num_threads(max(2, (torch.get_num_threads() or 4)))
        waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)  # (1, N)
        inputs = {"waveform": waveform, "sample_rate": sr}
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = int(num_speakers)
        result = _run_with_progress(pipeline, inputs, kwargs)
        annotation = getattr(result, "speaker_diarization", result)  # 4.x / 3.x 겸용
        turns: list[Turn] = [
            (float(segment.start), float(segment.end), str(label))
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: (t[0], t[1]))
        return turns
    except Exception as e:
        print(f"안내: 화자분리 실패 — 화자 표기 없이 진행합니다.\n  ({e})")
        return None


def _run_with_progress(pipeline, inputs, kwargs):
    """가능하면 진행률 훅을 붙여 실행한다 (pyannote 버전에 따라 없을 수 있음)."""
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
    except ImportError:
        return pipeline(inputs, **kwargs)
    with ProgressHook() as hook:
        return pipeline(inputs, hook=hook, **kwargs)
