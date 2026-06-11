# -*- coding: utf-8 -*-
"""
engine — 캡처(콜백) → 발화 분할(펌프 스레드) → 전사·번역(워커 스레드)을 엮는다.

GUI와는 event_queue 하나로 통신한다:
  - TranscriptEvent            : 전사 결과
  - ("status", 메시지 문자열)  : 상태바 표시용
  - ("notice", 메시지 문자열)  : 대본 영역에 남길 안내 (캡처 실패, 번역 불가 등)
GUI는 엔진(세션)마다 event_queue를 새로 만들어 넘긴다 — 중지 후 늦게 도착하는
이벤트가 다음 세션 화면을 오염시키지 않도록.
"""
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .capture import AudioDevice, CaptureStream
from .segmenter import UtteranceSegmenter

_SENTINEL = None  # 큐 종료 신호
# 전사 대기 발화 상한 — CPU가 실시간을 못 따라가면 오래된 발화부터 버린다
MAX_PENDING_UTTERANCES = 50


@dataclass
class TranscriptEvent:
    tag: str                    # "마이크" 또는 "시스템"
    wall_time: float            # 발화 시작 시각 (time.time() 기준)
    text: str
    lang: str
    translation: Optional[str]


class RealtimeEngine:
    """실시간 전사 파이프라인. start/stop은 GUI 스레드에서 호출된다."""

    def __init__(self, model_name: str, language: Optional[str],
                 translator, event_queue: "queue.Queue",
                 engine_pref: str = "auto") -> None:
        self.model_name = model_name
        self.language = language          # None이면 자동 감지
        self.translator = translator      # Translator 인스턴스 또는 None(번역 끔)
        self.engine_pref = engine_pref    # "auto"|"cuda"|"openvino-gpu"|"cpu"
        self.event_queue = event_queue
        self._chunk_queue: queue.Queue = queue.Queue()
        self._utterance_queue: queue.Queue = queue.Queue(maxsize=MAX_PENDING_UTTERANCES)
        self._streams: list[CaptureStream] = []
        self._segmenters: dict[str, UtteranceSegmenter] = {}
        self._pump_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._abort = threading.Event()   # 중지 후 잔여 작업·이벤트 방출 차단
        self._state_lock = threading.Lock()
        self._mono_to_wall = 0.0          # monotonic → wall time 오프셋
        self._extra_opts: dict = {}       # 실시간용 전사 옵션 (워커에서 결정)
        self._last_text = ""
        self._last_emit = 0.0
        self._last_drop_notice = 0.0      # 발화 드롭 안내 속도 제한
        self._translate_checked: set[str] = set()  # 언어팩 확인을 시작한 언어
        self._translate_failed: set[str] = set()   # 준비 실패한 언어 (재안내 방지)
        self._status_listening = "듣는 중..."      # 엔진 라벨 포함 상태 문구

    def start(self, devices: list[tuple[AudioDevice, str]]) -> None:
        """(장치, tag) 쌍들로 캡처를 시작한다. 모델은 워커 스레드에서 로드된다.

        선택한 장치를 하나도 열지 못하면 내부 정리 후 RuntimeError를 던진다.
        """
        with self._state_lock:
            if self._running:
                return
            self._running = True
        self._mono_to_wall = time.time() - time.monotonic()
        self._worker_thread = threading.Thread(
            target=self._worker, name="rt-worker", daemon=True)
        self._pump_thread = threading.Thread(
            target=self._pump, name="rt-pump", daemon=True)
        self._worker_thread.start()
        self._pump_thread.start()

        started, last_err = 0, None
        for device, tag in devices:
            if tag not in self._segmenters:
                self._segmenters[tag] = UtteranceSegmenter(tag, self._on_utterance)
            stream = CaptureStream(device, tag, self._chunk_queue)
            try:
                stream.start()
                started += 1
            except Exception as e:
                last_err = e
                self._emit(("notice", f"{tag} 캡처 시작 실패: {e}"))
                continue
            self._streams.append(stream)

        if devices and started == 0:
            self.stop()
            raise RuntimeError(f"선택한 장치에서 캡처를 시작하지 못했습니다. ({last_err})")

    def stop(self) -> None:
        """캡처 중지 → 잔여 큐 폐기 → 워커 종료까지 정리. 여러 번 불러도 안전.

        중지 시점 이후의 잔여 발화는 전사하지 않고 버린다 (수 분 걸릴 수 있으므로).
        """
        with self._state_lock:
            if not self._running:
                return
            self._running = False
        self._abort.set()
        for stream in self._streams:
            stream.stop()
        self._streams = []
        self._chunk_queue.put(_SENTINEL)       # 펌프 종료
        if self._pump_thread is not None:
            self._pump_thread.join(timeout=5)
            self._pump_thread = None
        self._segmenters = {}
        # 전사 백로그를 비워 워커가 sentinel에 바로 도달하게 한다
        while True:
            try:
                self._utterance_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._utterance_queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass  # 이론상 도달 불가 (생산자가 모두 멈춘 뒤 비웠음)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=10)
            self._worker_thread = None

    # ---- 내부 구현 ----

    def _emit(self, event) -> None:
        """중지된 세션의 늦은 이벤트가 GUI로 새지 않게 한 곳에서 방출한다."""
        if not self._abort.is_set():
            self.event_queue.put(event)

    def _on_utterance(self, tag: str, audio, t_start_mono: float) -> None:
        if self._abort.is_set():
            return
        item = (tag, audio, t_start_mono)
        try:
            self._utterance_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        # 가장 오래된 발화를 버리고 새 발화를 넣는다 (실시간성 우선)
        try:
            self._utterance_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._utterance_queue.put_nowait(item)
        except queue.Full:
            return
        now = time.monotonic()
        if now - self._last_drop_notice > 30.0:
            self._last_drop_notice = now
            self._emit(("notice", "처리 지연: 전사가 따라가지 못해 오래된 발화 일부를 생략했습니다. "
                                  "더 작은 모델을 권장합니다."))

    def _pump(self) -> None:
        """캡처 큐의 청크를 tag별 세그먼터에 먹인다."""
        while True:
            item = self._chunk_queue.get()
            if item is _SENTINEL:
                break
            if self._abort.is_set():
                continue
            tag, chunk, t_mono = item
            seg = self._segmenters.get(tag)
            if seg is None:
                continue
            try:
                seg.feed(chunk, t_mono)
            except Exception:
                pass  # VAD 오류 등으로 펌프 스레드가 죽지 않게 한다

    def _build_transcriber(self, engine: str):
        """엔진 종류에 맞는 전사기 인스턴스를 만든다 (로드는 호출 측에서)."""
        if engine == "openvino-gpu":
            from .. import config as config_mod
            from ..transcribe_ov import OvTranscriber
            return OvTranscriber(config_mod.openvino_model_dir(),
                                 language=self.language)
        from ..transcribe import Transcriber
        device = "cuda" if engine == "cuda" else "cpu"
        return Transcriber(self.model_name, language=self.language, device=device)

    def _worker(self) -> None:
        """엔진 폴백 체인으로 모델을 로드한 뒤 발화를 전사·번역해 GUI로 보낸다."""
        from .. import engine_select

        transcriber = None
        for engine in engine_select.resolve_chain(self.engine_pref):
            name = engine_select.label(engine)
            self._emit(("status", f"모델 준비 중... [{name}] "
                                  "(최초 실행은 다운로드/컴파일로 수 분 걸릴 수 있습니다)"))
            try:
                candidate = self._build_transcriber(engine)
                candidate.ensure_model()
                transcriber = candidate
                if engine == "openvino-gpu":
                    self._emit(("notice", "인텔 GPU 엔진: large-v3-turbo(int8) 모델로 "
                                          "전사합니다 (모델 선택 무시)."))
                break
            except Exception as e:
                self._emit(("notice", f"{name} 엔진 사용 불가 — 다음 순위로 넘어갑니다. ({e})"))
            if self._abort.is_set():
                return
        if transcriber is None:
            self._emit(("status", "모델 로딩 실패: 사용 가능한 엔진이 없습니다."))
            return
        if self._abort.is_set():
            return
        self._extra_opts = self._transcribe_kwargs(transcriber)
        self._status_listening = f"듣는 중... [{engine_select.label(engine)}]"
        self._emit(("status", self._status_listening))
        while True:
            item = self._utterance_queue.get()
            if item is _SENTINEL:
                break
            if self._abort.is_set():
                continue  # 중지 이후 잔여 발화는 버린다
            tag, audio, t_start_mono = item
            try:
                self._transcribe_one(transcriber, tag, audio, t_start_mono)
            except Exception as e:
                self._emit(("status", f"전사 오류: {e}"))

    @staticmethod
    def _transcribe_kwargs(transcriber) -> dict:
        """Transcriber가 받아 주는 범위에서 실시간용 전사 옵션을 고른다.

        beam_size=1, condition_on_previous_text=False — 속도·환각 방지.
        시그니처에 없으면 조용히 생략한다 (기본 구현이 내부에서 같은 값을 쓴다).
        """
        import inspect
        wanted = {"beam_size": 1, "condition_on_previous_text": False}
        try:
            params = inspect.signature(transcriber.transcribe_array).parameters
        except (TypeError, ValueError):
            return {}
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return wanted
        return {k: v for k, v in wanted.items() if k in params}

    def _transcribe_one(self, transcriber, tag: str, audio,
                        t_start_mono: float) -> None:
        segments, lang = transcriber.transcribe_array(audio, **self._extra_opts)
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            return
        now = time.monotonic()
        if text == self._last_text and now - self._last_emit < 1.0:
            self._last_emit = now  # 반복 환각은 계속 누른다
            return
        self._last_text, self._last_emit = text, now
        self._emit(TranscriptEvent(
            tag=tag,
            wall_time=t_start_mono + self._mono_to_wall,
            text=text,
            lang=lang,
            translation=self._translate(text, lang),
        ))

    def _translate(self, text: str, lang: str) -> Optional[str]:
        """번역기가 있고 준비되면 번역. 준비 실패는 1회만 안내하고 이후 건너뛴다."""
        if self.translator is None or lang in self._translate_failed:
            return None
        try:
            first_check = lang not in self._translate_checked
            if first_check:
                self._translate_checked.add(lang)
                self._emit(("status", "번역 언어팩 확인 중... (최초 1회 다운로드일 수 있음)"))
            if not self.translator.ensure_ready(lang):
                self._translate_failed.add(lang)
                self._emit(("notice", f"번역 사용 불가({lang}): 언어팩 준비에 실패해 "
                                      "원문만 표시합니다."))
                self._emit(("status", self._status_listening))
                return None
            if first_check:
                self._emit(("status", self._status_listening))
            return self.translator.translate(text, lang)
        except Exception:
            return None
