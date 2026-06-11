# -*- coding: utf-8 -*-
"""실시간 엔진 통합 테스트 — 시스템 소리(loopback)를 캡처하면서 테스트 mp3를
ffplay로 재생하고, TranscriptEvent가 실제로 도착하는지 확인한다.

GUI 없이 엔진만 돌리는 헤드리스 검증용. 수동 실행:
    .venv\\Scripts\\python.exe test_audio\\test_realtime.py
"""
import os
import queue
import subprocess
import sys
import time

sys.path.insert(0, r"C:\Users\user\mp3TXT_local")

from mp3txt.realtime.capture import list_loopback_devices
from mp3txt.realtime.engine import RealtimeEngine, TranscriptEvent

MP3 = r"C:\Users\user\mp3TXT_local\test_audio\test_meeting.mp3"


def main():
    loops = list_loopback_devices()
    if not loops:
        print("FAIL: loopback 장치 없음")
        sys.exit(1)
    print(f"loopback 장치: {loops[0].name} ({loops[0].default_sr}Hz)")

    events: queue.Queue = queue.Queue()
    engine = RealtimeEngine("small", "ko", None, events)
    engine.start([(loops[0], "시스템")])
    print("엔진 시작. 테스트 오디오 재생...")

    player = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", MP3])

    transcripts = []
    deadline = time.monotonic() + 120  # 재생 46초 + 전사 여유
    while time.monotonic() < deadline:
        try:
            ev = events.get(timeout=1.0)
        except queue.Empty:
            if player.poll() is not None and transcripts and \
                    time.monotonic() - transcripts[-1][0] > 20:
                break  # 재생 끝났고 20초간 새 결과 없으면 종료
            continue
        if isinstance(ev, TranscriptEvent):
            transcripts.append((time.monotonic(), ev))
            print(f"  [{ev.tag}] {ev.text}")
        else:
            print(f"  (상태) {ev[1]}")

    print("엔진 중지 중...")
    t0 = time.monotonic()
    engine.stop()
    stop_sec = time.monotonic() - t0
    print(f"중지 완료 ({stop_sec:.1f}초)")
    if player.poll() is None:
        player.kill()

    # 중지 후 늦은 이벤트가 새지 않는지 2초 관찰
    time.sleep(2)
    late = 0
    while True:
        try:
            events.get_nowait()
            late += 1
        except queue.Empty:
            break

    texts = " ".join(ev.text for _, ev in transcripts)
    print(f"\n전사 이벤트 {len(transcripts)}개, 중지 시간 {stop_sec:.1f}초, 늦은 이벤트 {late}개")
    ok = len(transcripts) >= 3 and stop_sec < 15
    keyword_hit = any(k in texts for k in ("회의", "출시", "검증", "안녕"))
    print(f"키워드 일치: {keyword_hit}")
    print("PASS" if ok and keyword_hit else "FAIL")
    sys.exit(0 if ok and keyword_hit else 1)


if __name__ == "__main__":
    main()
