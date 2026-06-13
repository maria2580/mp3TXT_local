# -*- coding: utf-8 -*-
"""사용자 설정 관리 — %USERPROFILE%\\.mp3txt_local\\config.json 에 저장.

배치/실시간 공용 설정과 Hugging Face 토큰(화자분리 모델 다운로드용)을 관리한다.
토큰은 환경변수 HF_TOKEN 또는 HUGGING_FACE_HUB_TOKEN 으로도 줄 수 있다.
"""
import json
import os

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".mp3txt_local")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_DIR, ".venv", "Scripts", "python.exe")

DEFAULTS = {
    "hf_token": "",                    # pyannote 화자분리용 Hugging Face 토큰
    "batch_model": "large-v3-turbo",   # 파일 변환용 Whisper 모델
    "realtime_model": "small",         # 실시간 모드용 Whisper 모델
    "compute_type": "int8",            # CPU 추론: int8 권장
    "language": "auto",                # "auto" 또는 "ko"/"en"/"ja"/...
    "translation_target": "ko",        # 실시간 번역 대상 언어
    "num_speakers": None,              # 화자 수를 알면 지정 (None=자동)
    "save_mp3_from_video": False,      # 동영상 변환 시 mp3도 같이 저장
    "realtime_engine": "auto",         # "auto"|"cuda"|"openvino-gpu"|"cpu"
    "openvino_model_dir": "",          # 비우면 <프로젝트>\models\whisper-large-v3-turbo-int8-ov
    "caption_opacity": 0.85,           # 자막 모드 투명도 (0.3~1.0)
    "caption_font_size": 14,           # 자막 모드 글자 크기
    "caption_geometry": "",            # 자막 모드 창 위치/크기 기억
    "frontend_agc": True,              # 적응형 게인 (조용한 발화 증폭, 무해)
    "frontend_denoise": False,         # 노이즈 제거 (노이즈 환경에서 켜기, 지연 증가)
}


def openvino_model_dir(cfg: dict | None = None) -> str:
    """OpenVINO 변환 모델 폴더 경로 (설정이 비어 있으면 기본 위치)."""
    if cfg is None:
        cfg = load()
    path = (cfg.get("openvino_model_dir") or "").strip()
    if path:
        return path
    return os.path.join(PROJECT_DIR, "models", "whisper-large-v3-turbo-int8-ov")


def load() -> dict:
    """기본값 위에 설정 파일을 덮어쓴 dict를 돌려준다. 파일이 없거나 깨져도 동작한다."""
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in DEFAULTS:
                if key in data:
                    cfg[key] = data[key]
    except (OSError, ValueError):
        pass
    return cfg


def save(cfg: dict) -> None:
    """알려진 키만 골라 설정 파일에 저장한다."""
    data = {key: cfg[key] for key in DEFAULTS if key in cfg}
    os.makedirs(CONFIG_DIR, exist_ok=True)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CONFIG_PATH)


def get_hf_token(cfg: dict | None = None) -> str | None:
    """설정 파일 → 환경변수 순으로 Hugging Face 토큰을 찾는다. 없으면 None."""
    if cfg is None:
        cfg = load()
    token = (cfg.get("hf_token") or "").strip()
    if token:
        return token
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = (os.environ.get(name) or "").strip()
        if token:
            return token
    return None
