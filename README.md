# mp3TXT_local

OpenAI **Whisper**(오픈소스)의 강건한 한국어·영어 인식 성능을 회의 녹음과 영상 자료의
텍스트 추출에 활용하기 위해 만든 **완전 로컬** 변환 도구입니다.

파일 탐색기 **우클릭 컨텍스트 메뉴에 "TXT 추출하기"** 를 추가하며, 클릭하면
**pyannote(community-1)** 로 음성에서 화자를 분리한 뒤 Whisper 전사 결과와 정렬해
화자별 회의 텍스트 파일로 정리합니다. 모든 추론이 PC 안에서 일어나 녹음이 외부로
전송되지 않고, 실시간 모드에서는 마이크와 **컴퓨터 재생 소리**를 즉시 받아쓰며
선택적으로 번역까지 합니다.

| 기능 | 엔진 | 비고 |
|---|---|---|
| 전사(STT) | faster-whisper (CPU int8) | 배치: large-v3-turbo, 실시간: small (변경 가능) |
| 화자분리 | pyannote speaker-diarization-community-1 | Hugging Face 토큰 필요 (무료) |
| 번역 | Argos Translate | 완전 로컬, 첫 사용 시 언어팩 다운로드 |
| 시스템 소리 캡처 | WASAPI loopback (pyaudiowpatch) | 마이크와 동시 사용 가능 |

## 1. 배치 변환 — 우클릭 "TXT 추출하기"

1. 음성 파일(mp3/wav/m4a/flac/ogg/opus/wma/aac 등) 또는 **동영상 파일**
   (mp4/mkv/avi/mov/wmv 등) 우클릭 → **TXT 추출하기(mp3TXT_local)**

   > **윈도우 11 주의**: 기본 우클릭 메뉴에는 안 보입니다. 메뉴 맨 아래
   > **`추가 옵션 표시`** 를 누르거나, **Shift를 누른 채 우클릭**하면 나타납니다.
2. 같은 폴더에 같은 이름의 `.txt` 저장 (이미 있으면 `이름 (1).txt` … 자동 회피)

동영상은 음성 트랙만 추출해 같은 파이프라인으로 변환합니다 (중간 mp3 불필요).
mp3 파일도 따로 남기려면 `--save-mp3` 옵션 또는 설정 `save_mp3_from_video: true`
(ffmpeg 필요 — 동영상과 같은 폴더에 mp3 저장).

출력 형식:

```
참가자 1 (00:00:05): 안녕하세요. 오늘 회의를 시작하겠습니다.

참가자 2 (00:00:12): 네, 개발 진행 상황부터 말씀드리겠습니다.
```

화자분리를 사용할 수 없으면(토큰 미설정 등) `(00:00:05): 내용` 형식으로 저장됩니다.

## 2. 실시간 모드 — GUI 앱

```powershell
.venv\Scripts\python.exe realtime_app.py
```

- **입력 소스**: 마이크, 시스템 소리(컴퓨터에서 재생되는 모든 소리), 또는 둘 다
- 발화가 끝날 때마다(무음 0.6초) 텍스트가 한 줄씩 추가됩니다
- **번역**: `→ 한국어` 또는 `→ 영어` 선택 시 원문 아래에 번역문 표시 (Argos, 로컬)
- [복사] 버튼으로 전체 기록을 클립보드에, [저장] 버튼으로 txt로 저장
- **엔진 자동 선택**: NVIDIA GPU → 인텔 GPU(OpenVINO) → CPU 순으로 감지

### 자막 모드

[자막 모드] 버튼을 누르면 윈도우 라이브 캡션 스타일의 **항상-위 오버레이**로 바뀝니다.
라이브 캡션과의 차이:

| | 윈도우 라이브 캡션 | mp3TXT_local 자막 모드 |
|---|---|---|
| 지나간 내용 | 못 봄 | **스크롤로 전체 히스토리** |
| 내보내기 | 없음 | **복사·txt 저장** |
| 번역 | Copilot+ 전용, 영어로만 | 어느 PC든 ko↔en |
| 인식 모델 | ~120MB 경량 | Whisper (GPU면 turbo급) |

- 투명도 슬라이더(30~100%), 글자 크기 A−/A+, 위치·크기 드래그 조절 (다음 실행 시 기억)
- 컨트롤 바: 설정(전체 모드 복귀) / 시작·중지 / 복사 / 저장 / 닫기

## 설치

```powershell
# 1) 가상환경 + 의존성 (이미 완료된 상태로 제공)
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --python .venv\Scripts\python.exe

# 2) 우클릭 메뉴 등록 (관리자 권한 불필요 - HKCU 사용)
.venv\Scripts\python.exe mp3txt_local.py

# 3) 화자분리용 Hugging Face 토큰 (무료, 최초 1회)
#    a. https://huggingface.co/pyannote/speaker-diarization-community-1 약관 동의
#    b. https://huggingface.co/settings/tokens 에서 Read 토큰 발급
.venv\Scripts\python.exe mp3txt_local.py --set-token hf_xxxxxxxx

# 4) 모델 미리 다운로드 (선택, 권장 - 첫 변환이 바로 시작되게)
.venv\Scripts\python.exe mp3txt_local.py --setup
```

## 명령어

```powershell
.venv\Scripts\python.exe mp3txt_local.py                      # 우클릭 메뉴 등록
.venv\Scripts\python.exe mp3txt_local.py --uninstall          # 메뉴 제거
.venv\Scripts\python.exe mp3txt_local.py --set-token <토큰>   # HF 토큰 저장
.venv\Scripts\python.exe mp3txt_local.py --setup              # 모델 사전 다운로드
.venv\Scripts\python.exe mp3txt_local.py "회의.mp3"           # 수동 변환 (여러 파일 가능)
.venv\Scripts\python.exe mp3txt_local.py "회의.mp3" --no-diarization  # 화자분리 없이
.venv\Scripts\python.exe mp3txt_local.py "강의.mp4" --save-mp3 # 동영상 변환 + mp3 저장
.venv\Scripts\python.exe realtime_app.py                      # 실시간 GUI
```

## 설정

`%USERPROFILE%\.mp3txt_local\config.json` (없으면 기본값):

| 키 | 기본값 | 설명 |
|---|---|---|
| `hf_token` | `""` | 화자분리용 토큰 (환경변수 HF_TOKEN도 인식) |
| `batch_model` | `large-v3-turbo` | 파일 변환 모델. 정확도 우선이면 `large-v3` (훨씬 느림) |
| `realtime_model` | `small` | 실시간 모델. 품질 우선이면 `large-v3-turbo` (지연 증가) |
| `compute_type` | `int8` | CPU 권장값 |
| `language` | `auto` | `ko` 고정 시 감지 오류 방지 + 약간 빨라짐 |
| `num_speakers` | `null` | 화자 수를 알면 지정 (정확도 향상) |

## NVIDIA GPU 가속 (선택)

NVIDIA GPU가 있는 PC에서는 다음 두 가지를 설치하면 전사·화자분리가 GPU로 동작합니다.
설치하지 않으면 GPU가 있어도 CUDA 장치 0개로 감지되어 CPU로 동작합니다 (정상 폴백).

```powershell
# 1) CUDA 빌드 torch (화자분리용 — 기본 PyPI torch는 Windows에서 CPU 전용)
uv pip install --python .venv\Scripts\python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2) cuBLAS/cuDNN DLL (전사용 — faster-whisper의 CUDA 인식에 필요)
uv pip install --python .venv\Scripts\python.exe nvidia-cublas-cu12 nvidia-cudnn-cu12

# 3) 확인 (둘 다 0/False가 아니어야 함)
.venv\Scripts\python.exe -c "from mp3txt import engine_select; print('CUDA 장치:', engine_select.cuda_device_count())"
.venv\Scripts\python.exe -c "import torch; print('torch CUDA:', torch.cuda.is_available())"
```

이후 변환하면 "전사 중... [NVIDIA GPU]"와 "화자분리: NVIDIA GPU 사용"이 표시됩니다.

## 동작 세부 사항

- **CPU 전용 설계**: 이 PC(GPU 없음, i5-1335U) 실측 — 44초 음성을 large-v3-turbo로
  65초에 변환 (약 1.5배). 1시간 녹음이면 약 1.5시간 + 화자분리 시간이 듭니다.
  첫 실행 시 모델 다운로드(수백 MB~1.5GB)가 있습니다.
- **실시간 모드 처리 한계**: 전사가 실시간을 못 따라가면 오래된 발화부터 자동 생략하고
  안내를 표시합니다. 기본 모델(small)은 이 PC에서 실시간 처리가 가능합니다.
- **메뉴 등록 위치**: `HKCU\Software\Classes\*\shell\mp3TXT_local` + `AppliesTo` 확장자
  필터를 씁니다. (`SystemFileAssociations`의 HKCU 브랜치는 이 PC의 Windows 11에서
  Explorer가 메뉴에 표시해 주지 않아 위치를 옮겼습니다.)
- **다중 선택**: 여러 파일을 한꺼번에 변환하면 뮤텍스로 한 번에 하나씩 차례로 처리합니다.
- **저장 위치 대체**: 음성 파일 폴더에 쓸 수 없으면 자동으로 `내 문서`에 저장합니다.
  변환 시작 전에 출력 파일을 먼저 확보합니다.
- **화자분리 폴백**: 토큰이 없거나 모델 로드에 실패하면 변환을 중단하지 않고
  타임스탬프만으로 저장합니다.
- **시스템 소리 캡처**: WASAPI loopback은 소리가 재생 중일 때만 프레임이 들어옵니다.
  아무것도 재생되지 않으면 (당연히) 아무것도 전사되지 않습니다.
- **개인정보**: 모든 추론이 이 PC 안에서 일어납니다. 모델/언어팩 다운로드 시에만 인터넷을 씁니다.

## 구성 요소

| 경로 | 설명 |
|---|---|
| `mp3txt_local.py` | 배치 엔트리 (설치/제거/변환) |
| `realtime_app.py` | 실시간 GUI 런처 |
| `mp3txt/config.py` | 설정 관리 |
| `mp3txt/audio_io.py` | 오디오 디코드 (PyAV) |
| `mp3txt/transcribe.py` | faster-whisper 래퍼 |
| `mp3txt/diarize.py` | pyannote 화자분리 래퍼 |
| `mp3txt/formatter.py` | 화자 배정 + "참가자 N (hh:mm:ss):" 포맷 |
| `mp3txt/translate.py` | 번역 모듈 (Argos) |
| `mp3txt/realtime/` | 캡처(WASAPI)·VAD 분할·전사 엔진·tkinter GUI |
| `.venv\` | uv로 만든 Python 3.12 가상환경 |
