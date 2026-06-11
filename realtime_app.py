# -*- coding: utf-8 -*-
"""mp3TXT_local 실시간 모드 런처 — 마이크/시스템 소리를 실시간 전사하는 GUI를 띄운다.

사용법:
    .venv\\Scripts\\python.exe realtime_app.py
(콘솔 창 없이 띄우려면 pythonw.exe 사용 — 단, 모델 다운로드 진행 표시가 안 보인다)
"""
import os
import sys
import traceback

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None:
        try:
            _stream.reconfigure(errors="replace")
        except (AttributeError, OSError):
            pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)


def main():
    try:
        from mp3txt.realtime.gui import run_app
        run_app()
    except Exception:
        traceback.print_exc()
        try:
            import tkinter.messagebox as mb
            mb.showerror("mp3TXT_local", "실행 중 오류가 발생했습니다.\n"
                         "콘솔 출력을 확인해 주세요.")
        except Exception:
            pass
        if sys.stdin is not None and sys.stdin.isatty():
            print("\n오류가 발생했습니다. Enter를 누르면 닫힙니다.")
            try:
                input()
            except EOFError:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
