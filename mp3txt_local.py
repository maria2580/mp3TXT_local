# -*- coding: utf-8 -*-
"""mp3TXT_local — 음성 파일을 완전 로컬에서 화자분리 + 전사해 .txt로 저장한다.

엔진: faster-whisper(전사, CPU int8) + pyannote community-1(화자분리).
출력: 음성 파일과 같은 폴더에 같은 이름의 .txt
      "참가자 1 (00:00:05): ..." 형식 (화자분리 불가 시 타임스탬프만).

사용법:
    python mp3txt_local.py                    # 우클릭 컨텍스트 메뉴 등록 (설치)
    python mp3txt_local.py --uninstall        # 컨텍스트 메뉴 제거
    python mp3txt_local.py --set-token <토큰> # 화자분리용 Hugging Face 토큰 저장
    python mp3txt_local.py --setup            # 모델 미리 다운로드 (최초 1회 권장)
    python mp3txt_local.py <음성파일> [--no-diarization]  # 변환 실행
"""
import os
import sys
import time
import traceback

# 콘솔 코드페이지(cp949 등)에 없는 문자가 출력에 섞여도 죽지 않게 한다
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None:
        try:
            _stream.reconfigure(errors="replace")
        except (AttributeError, OSError):
            pass

SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(SCRIPT_PATH)
VENV_PYTHON = os.path.join(PROJECT_DIR, ".venv", "Scripts", "python.exe")
sys.path.insert(0, PROJECT_DIR)  # 어디서 실행돼도 mp3txt 패키지를 찾도록

MENU_LABEL = "TXT 추출하기(mp3TXT_local)"
MENU_NAME = "mp3TXT_local"
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus",
              ".wma", ".aac", ".webm", ".mka", ".amr")
# 동영상도 음성 트랙만 추출해 같은 파이프라인으로 변환한다 (PyAV가 직접 디코드)
VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".wmv", ".m4v",
              ".ts", ".flv", ".mpg", ".mpeg")
ALL_EXTS = AUDIO_EXTS + VIDEO_EXTS
MUTEX_NAME = "Local\\mp3TXT_local_engine"  # 동시 변환 1개로 제한 (다중 선택 대비)


# 우클릭 메뉴 등록 위치: 모든 파일(*) 키 + AppliesTo 확장자 필터.
# SystemFileAssociations의 HKCU 브랜치는 이 PC(Win11 26200)에서 Explorer가
# 메뉴에 그려 주지 않는 것이 확인되어, 실제로 렌더링되는 위치를 쓴다.
MENU_KEY = rf"Software\Classes\*\shell\{MENU_NAME}"


def applies_to_query() -> str:
    """음성/동영상 확장자에만 메뉴가 보이도록 하는 AQS 필터."""
    return " OR ".join(f'System.FileExtension:="{ext}"' for ext in ALL_EXTS)


def legacy_menu_key(ext: str) -> str:
    """구버전(동작 안 함) 등록 위치 — 설치/제거 시 청소용."""
    return rf"Software\Classes\SystemFileAssociations\{ext}\shell\{MENU_NAME}"


def notify_assoc_changed() -> None:
    """셸에 파일 연결 변경을 알린다 — 이게 없으면 Explorer가 캐시된 메뉴를 계속 쓴다."""
    import ctypes

    SHCNE_ASSOCCHANGED = 0x08000000
    SHCNF_IDLIST = 0x0
    try:
        ctypes.windll.shell32.SHChangeNotify(
            SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None)
    except Exception:
        pass  # 실패해도 등록 자체는 유효 (재부팅/Explorer 재시작 시 반영)


def remove_legacy_keys() -> int:
    import winreg

    removed = 0
    for ext in ALL_EXTS:
        key_path = legacy_menu_key(ext)
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path + r"\command")
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)
            removed += 1
        except FileNotFoundError:
            pass
    return removed


def open_unique(path):
    """'name.txt' → 'name (1).txt' → ... 순으로 새 파일을 만들어 (핸들, 경로)를 돌려준다."""
    base, ext = os.path.splitext(path)
    candidate, n = path, 0
    while True:
        try:
            return open(candidate, "x", encoding="utf-8-sig", newline="\r\n"), candidate
        except FileExistsError:
            n += 1
            candidate = f"{base} ({n}){ext}"


def open_output(audio_path):
    """출력 txt를 음성 파일 옆에 만들고, 쓰기 불가 폴더면 내 문서 폴더로 대체한다."""
    target = os.path.splitext(audio_path)[0] + ".txt"
    try:
        return open_unique(target)
    except OSError:
        docs = os.path.join(os.path.expanduser("~"), "Documents")
        fallback = os.path.join(docs, os.path.basename(target))
        print(f"음성 파일 폴더에 쓸 수 없어 내 문서 폴더에 저장합니다: {docs}")
        return open_unique(fallback)


class ConvertSlot:
    """이름 있는 뮤텍스로 변환 동시 실행을 1개로 제한한다.

    여러 파일을 한꺼번에 우클릭 변환하면 파일 수만큼 프로세스가 뜨는데,
    Whisper/pyannote 모델을 전부 동시에 올리지 않도록 차례로 실행한다.
    """

    def __enter__(self):
        import ctypes
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._handle = self._k32.CreateMutexW(None, False, MUTEX_NAME)
        if not self._handle:
            return self  # 뮤텍스 생성 실패 시 제한 없이 진행
        WAIT_OBJECT_0, WAIT_ABANDONED, INFINITE = 0x0, 0x80, 0xFFFFFFFF
        rc = self._k32.WaitForSingleObject(self._handle, 0)
        if rc not in (WAIT_OBJECT_0, WAIT_ABANDONED):
            print("다른 변환 작업이 진행 중입니다. 차례를 기다리는 중...")
            self._k32.WaitForSingleObject(self._handle, INFINITE)
        return self

    def __exit__(self, *exc):
        if self._handle:
            self._k32.ReleaseMutex(self._handle)
            self._k32.CloseHandle(self._handle)
        return False


def extract_mp3(video_path: str) -> str | None:
    """동영상에서 음성을 mp3로 추출해 같은 폴더에 저장한다 (ffmpeg 필요).

    성공하면 mp3 경로, ffmpeg가 없거나 실패하면 경고 출력 후 None.
    """
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("경고: ffmpeg를 찾을 수 없어 mp3 추출을 건너뜁니다 (변환은 계속 진행).")
        return None
    base = os.path.splitext(video_path)[0]
    target, n = base + ".mp3", 0
    while os.path.exists(target):
        n += 1
        target = f"{base} ({n}).mp3"
    print(f"mp3 추출 중: {os.path.basename(target)}")
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", video_path,
         "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k", target],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"경고: mp3 추출 실패 (변환은 계속 진행).\n  {result.stderr.strip()}")
        return None
    print(f"mp3 저장 완료: {target}")
    return target


def transcribe_file(audio_path: str, no_diarization: bool = False,
                    save_mp3: bool = False) -> str:
    """음성/동영상 파일 1개를 전사 + 화자분리해 txt로 저장하고 경로를 돌려준다."""
    from mp3txt import config

    audio_path = os.path.abspath(audio_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"파일이 없습니다: {audio_path}")

    cfg = config.load()
    is_video = os.path.splitext(audio_path)[1].lower() in VIDEO_EXTS
    if is_video:
        print(f"동영상 파일: {audio_path}")
        print("음성 트랙만 추출해 변환합니다.")
        if save_mp3 or cfg.get("save_mp3_from_video"):
            extract_mp3(audio_path)
    else:
        print(f"음성 파일: {audio_path}")

    # 변환 전에 출력 파일부터 확보해, 다 끝낸 뒤 저장에 실패하는 일이 없게 한다
    out_file, out_path = open_output(audio_path)

    try:
        turns = _convert(audio_path, cfg, no_diarization, out_file)
    except BaseException:
        # 실패 시 빈 자리표시 파일을 지워, 재시도 결과가 "(1).txt"로 밀리지 않게 한다
        try:
            if os.path.getsize(out_path) == 0:
                os.remove(out_path)
        except OSError:
            pass
        raise

    print(f"\n저장 완료: {out_path}")
    if turns is None and not no_diarization:
        print("주의: 화자분리 없이 저장되었습니다 (위 안내 참조).")
    return out_path


def _convert(audio_path, cfg, no_diarization, out_file):
    """디코드 → 화자분리 → 전사 → 포맷 저장. 화자 구간(turns)을 돌려준다."""
    from mp3txt import audio_io, config, diarize, formatter
    from mp3txt.transcribe import Transcriber

    with out_file, ConvertSlot():
        t0 = time.time()
        print("오디오 디코드 중...")
        audio = audio_io.load_audio(audio_path)
        total_sec = audio_io.duration_sec(audio)
        from mp3txt import engine_select
        device = "cuda" if engine_select.cuda_device_count() > 0 else "cpu"
        engine_label = "NVIDIA GPU" if device == "cuda" else f"CPU {cfg['compute_type']}"
        print(f"길이: {formatter.fmt_ts(total_sec)}  "
              f"(모델: {cfg['batch_model']}, {engine_label})")

        # 1) 화자분리 (토큰 없으면 None → 타임스탬프만)
        turns = None
        if no_diarization:
            print("화자분리: 사용 안 함 (--no-diarization)")
        else:
            print("화자분리 중... (최초 실행 시 모델 다운로드)")
            turns = diarize.diarize(audio, config.get_hf_token(cfg),
                                    num_speakers=cfg.get("num_speakers"))
            if turns:
                n = len({t[2] for t in turns})
                print(f"화자분리 완료: 화자 {n}명, 구간 {len(turns)}개")

        # 2) 전사 (세그먼트가 나올 때마다 진행 상황 출력)
        # NVIDIA GPU(CUDA)가 있으면 사용, 로드 실패 시 CPU로 강등.
        # (인텔 GPU는 워드 타임스탬프 미지원이라 배치에는 쓰지 않는다 —
        #  화자분리 정렬 정밀도가 우선)
        if device == "cuda":
            print("전사 중... [NVIDIA GPU] (최초 실행 시 모델 다운로드)")
        else:
            print("전사 중... (최초 실행 시 모델 다운로드)")
        transcriber = Transcriber(cfg["batch_model"], cfg["compute_type"],
                                  cfg["language"], device=device)
        try:
            transcriber.ensure_model()
        except Exception as e:
            if device != "cuda":
                raise
            print(f"안내: NVIDIA GPU 로드 실패 — CPU로 전환합니다. ({e})")
            transcriber = Transcriber(cfg["batch_model"], cfg["compute_type"],
                                      cfg["language"])
            transcriber.ensure_model()

        def on_segment(seg):
            preview = seg.text if len(seg.text) <= 60 else seg.text[:57] + "..."
            print(f"  [{formatter.fmt_ts(seg.start)}/{formatter.fmt_ts(total_sec)}] {preview}")

        segments, lang = transcriber.transcribe_long(audio, word_timestamps=True,
                                                     on_segment=on_segment)
        print(f"전사 완료: 세그먼트 {len(segments)}개, 언어: {lang}")

        # 3) 화자 배정 + 포맷 + 저장
        utterances = formatter.assign_speakers(segments, turns)
        out_file.write(formatter.render(utterances))
        out_file.write("\n")
        print(f"소요 시간: {time.time() - t0:.1f}초")
    return turns


def setup_models():
    """Whisper/화자분리 모델을 미리 내려받아 첫 변환이 바로 시작되게 한다."""
    from mp3txt import config, diarize
    from mp3txt.transcribe import Transcriber
    import numpy as np

    cfg = config.load()
    print(f"[1/2] Whisper 모델 다운로드: {cfg['batch_model']} ...")
    transcriber = Transcriber(cfg["batch_model"], cfg["compute_type"])
    transcriber.ensure_model()
    print("  완료.")

    print("[2/2] 화자분리 모델 다운로드 확인...")
    silent = np.zeros(16000, dtype=np.float32)  # 1초 무음으로 로드만 검증
    result = diarize.diarize(silent, config.get_hf_token(cfg))
    if result is not None:
        print("  완료: 화자분리 사용 가능.")
    print("\n준비 끝. 음성 파일을 우클릭해 변환할 수 있습니다.")
    print("(우클릭 메뉴가 안 보이면 등록부터: 인자 없이 python mp3txt_local.py 실행)")


def set_token(token: str):
    from mp3txt import config
    cfg = config.load()
    cfg["hf_token"] = token.strip()
    config.save(cfg)
    print("Hugging Face 토큰을 저장했습니다.")
    print("모델 다운로드 확인을 위해 다음을 실행하세요:")
    print(f'  "{VENV_PYTHON}" "{SCRIPT_PATH}" --setup')


def install():
    """HKCU에 음성 파일 우클릭 메뉴를 등록한다 (관리자 권한 불필요)."""
    import winreg

    if not os.path.isfile(VENV_PYTHON):
        raise FileNotFoundError(f"venv 파이썬이 없습니다: {VENV_PYTHON}")

    command = f'"{VENV_PYTHON}" "{SCRIPT_PATH}" "%1"'
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, MENU_KEY) as key:
        winreg.SetValueEx(key, None, 0, winreg.REG_SZ, MENU_LABEL)
        winreg.SetValueEx(key, "Icon", 0, winreg.REG_SZ, "imageres.dll,-102")
        # 기본 모델은 16개 이상 다중 선택 시 메뉴를 숨긴다 → 한도를 ~100개로
        winreg.SetValueEx(key, "MultiSelectModel", 0, winreg.REG_SZ, "Player")
        # 모든 파일(*)에 등록하되 음성/동영상 확장자에만 표시
        winreg.SetValueEx(key, "AppliesTo", 0, winreg.REG_SZ, applies_to_query())
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, MENU_KEY + r"\command") as key:
        winreg.SetValueEx(key, None, 0, winreg.REG_SZ, command)
    remove_legacy_keys()  # 구버전 등록 위치 청소
    notify_assoc_changed()

    print("우클릭 컨텍스트 메뉴 등록 완료!")
    print(f"  메뉴 이름: {MENU_LABEL}")
    print(f"  음성 확장자: {', '.join(AUDIO_EXTS)}")
    print(f"  동영상 확장자: {', '.join(VIDEO_EXTS)} (음성 트랙만 추출)")
    print(f"  실행 명령: {command}")
    print("\n음성 파일을 우클릭하면 메뉴가 보입니다.")
    print("(윈도우 11에서는 [추가 옵션 표시]를 누르면 나타납니다)")
    print("제거하려면:  python mp3txt_local.py --uninstall")


def uninstall():
    import winreg

    removed = remove_legacy_keys()
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, MENU_KEY + r"\command")
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, MENU_KEY)
        removed += 1
    except FileNotFoundError:
        pass
    notify_assoc_changed()
    if removed:
        print("컨텍스트 메뉴를 제거했습니다.")
    else:
        print("등록된 컨텍스트 메뉴가 없습니다.")


def main():
    args = sys.argv[1:]
    try:
        if not args:
            install()
        elif args[0] == "--uninstall":
            uninstall()
        elif args[0] == "--set-token":
            if len(args) < 2 or not args[1].strip():
                print("사용법: python mp3txt_local.py --set-token <토큰>")
                sys.exit(2)
            set_token(args[1])
        elif args[0] == "--setup":
            setup_models()
        else:
            paths = [a for a in args if not a.startswith("--")]
            no_diar = "--no-diarization" in args
            save_mp3 = "--save-mp3" in args
            if not paths:
                print(f"알 수 없는 옵션: {' '.join(args)}")
                sys.exit(2)
            for i, path in enumerate(paths):
                if len(paths) > 1:
                    print(f"\n===== 파일 {i + 1} / {len(paths)} =====")
                transcribe_file(path, no_diarization=no_diar, save_mp3=save_mp3)
            time.sleep(3)  # 창이 바로 닫히지 않게 결과를 잠시 보여준다
    except Exception:
        traceback.print_exc()
        if sys.stdin is not None and sys.stdin.isatty():
            print("\n오류가 발생했습니다. Enter를 누르면 닫힙니다.")
            try:
                input()
            except EOFError:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
