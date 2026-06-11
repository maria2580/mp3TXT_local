# -*- coding: utf-8 -*-
"""
gui — tkinter로 만든 실시간 음성 → 텍스트 변환 창.

장치·모델·언어·번역을 고르고 시작을 누르면 RealtimeEngine이 event_queue로
보내 주는 결과를 100ms 주기로 폴링해 화면에 붙인다. 표준 라이브러리만 사용.
"""
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Optional

from .capture import AudioDevice, list_input_devices, list_loopback_devices
from .engine import RealtimeEngine, TranscriptEvent

TITLE = "mp3TXT_local — 실시간 음성 텍스트 변환"
FONT = ("Malgun Gothic", 10)
MODELS = ["small", "base", "medium", "large-v3-turbo"]
LANGS = [("자동", None), ("한국어", "ko"), ("영어", "en"),
         ("일본어", "ja"), ("중국어", "zh")]
TRANSLATIONS = [("끔", None), ("→ 한국어", "ko"), ("→ 영어", "en")]
ENGINES = [("자동", "auto"), ("NVIDIA GPU", "cuda"),
           ("인텔 GPU", "openvino-gpu"), ("CPU", "cpu")]
POLL_MS = 100


class RealtimeApp:
    """실시간 STT 메인 창."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.engine: Optional[RealtimeEngine] = None
        self.event_queue: queue.Queue = queue.Queue()
        self._stopping = False
        self._closing = False
        self._mics: list[AudioDevice] = []
        self._loops: list[AudioDevice] = []
        self._cfg = self._load_config()

        root.title(TITLE)
        root.geometry("720x560")
        root.minsize(560, 400)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_widgets()
        self._apply_config_defaults()
        self._load_devices()
        self.root.after(POLL_MS, self._poll)

    # ---- 구성 ----

    @staticmethod
    def _load_config() -> dict:
        try:
            from .. import config
            return config.load()
        except Exception:
            return {}

    def _save_config(self, model: str, lang: Optional[str],
                     target: Optional[str], engine: str = "auto") -> None:
        """선택한 설정을 영속화한다. 실패해도 동작을 막지 않는다."""
        try:
            from .. import config
            cfg = config.load()
            cfg["realtime_model"] = model
            cfg["language"] = lang or "auto"
            cfg["translation_target"] = target or ""  # 끔이면 빈 문자열
            cfg["realtime_engine"] = engine
            config.save(cfg)
        except Exception:
            pass

    def _build_widgets(self) -> None:
        style = ttk.Style(self.root)
        for name in ("TLabel", "TButton", "TCheckbutton", "TCombobox"):
            style.configure(name, font=FONT)

        # 상단 설정 프레임
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        top.columnconfigure(1, weight=1)

        self.mic_on = tk.BooleanVar(value=True)
        self.mic_check = ttk.Checkbutton(top, text="마이크", variable=self.mic_on)
        self.mic_check.grid(row=0, column=0, sticky="w")
        self.mic_combo = ttk.Combobox(top, state="readonly", font=FONT)
        self.mic_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=2)

        self.loop_on = tk.BooleanVar(value=True)
        self.loop_check = ttk.Checkbutton(top, text="시스템 소리", variable=self.loop_on)
        self.loop_check.grid(row=1, column=0, sticky="w")
        self.loop_combo = ttk.Combobox(top, state="readonly", font=FONT)
        self.loop_combo.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=2)

        opts = ttk.Frame(top)
        opts.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(opts, text="모델").pack(side="left")
        self.model_combo = ttk.Combobox(opts, state="readonly", values=MODELS,
                                        width=14, font=FONT)
        self.model_combo.pack(side="left", padx=(4, 0))
        ttk.Label(opts, text="언어").pack(side="left", padx=(10, 0))
        self.lang_combo = ttk.Combobox(opts, state="readonly",
                                       values=[lbl for lbl, _ in LANGS],
                                       width=8, font=FONT)
        self.lang_combo.pack(side="left", padx=(4, 0))
        ttk.Label(opts, text="번역").pack(side="left", padx=(10, 0))
        self.trans_combo = ttk.Combobox(opts, state="readonly",
                                        values=[lbl for lbl, _ in TRANSLATIONS],
                                        width=10, font=FONT)
        self.trans_combo.pack(side="left", padx=(4, 0))
        ttk.Label(opts, text="엔진").pack(side="left", padx=(10, 0))
        self.engine_combo = ttk.Combobox(opts, state="readonly",
                                         values=[lbl for lbl, _ in ENGINES],
                                         width=11, font=FONT)
        self.engine_combo.pack(side="left", padx=(4, 0))
        self.toggle_btn = ttk.Button(opts, text="시작", command=self._on_toggle)
        self.toggle_btn.pack(side="right")

        # 하단 상태바 (텍스트 영역보다 먼저 pack해서 공간 확보)
        bottom = ttk.Frame(self.root, padding=(8, 2, 8, 6))
        bottom.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(bottom, textvariable=self.status_var, anchor="w").pack(
            side="left", fill="x", expand=True)
        self.topmost_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bottom, text="항상 위", variable=self.topmost_var,
                        command=self._on_topmost).pack(side="right")
        ttk.Button(bottom, text="지우기", command=self._on_clear).pack(
            side="right", padx=(0, 8))
        ttk.Button(bottom, text="저장", command=self._on_save).pack(
            side="right", padx=(0, 8))

        # 중앙 텍스트 영역 (읽기 전용)
        self.text = ScrolledText(self.root, wrap="word", state="disabled", font=FONT)
        self.text.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.text.tag_configure("마이크", foreground="#1d4ed8")  # 파란색 계열
        self.text.tag_configure("시스템", foreground="#15803d")  # 초록색 계열
        self.text.tag_configure("번역", foreground="#6b7280")    # 회색

    def _apply_config_defaults(self) -> None:
        """config 값을 콤보박스에 반영한다.

        GUI 목록에 없는 값(예: 손으로 넣은 language='de')은 목록에 동적으로
        추가해 그대로 표시·보존한다 — '자동'으로 폴백했다가 시작 시
        config을 덮어써 버리는 사고 방지.
        """
        model = self._cfg.get("realtime_model", "small")
        self.model_combo.set(model if model in MODELS else "small")

        self._langs = list(LANGS)
        lang_code = self._cfg.get("language", "auto") or "auto"
        if lang_code != "auto" and lang_code not in [c for _, c in self._langs]:
            self._langs.append((lang_code, lang_code))
            self.lang_combo["values"] = [lbl for lbl, _ in self._langs]
        self.lang_combo.set(next(
            (lbl for lbl, code in self._langs if (code or "auto") == lang_code), "자동"))

        self._trans = list(TRANSLATIONS)
        target = self._cfg.get("translation_target", "ko")
        if target and target not in [c for _, c in self._trans]:
            self._trans.append((f"→ {target}", target))
            self.trans_combo["values"] = [lbl for lbl, _ in self._trans]
        self.trans_combo.set(next(
            (lbl for lbl, code in self._trans if code == target), "끔"))

        engine = self._cfg.get("realtime_engine", "auto") or "auto"
        self.engine_combo.set(next(
            (lbl for lbl, code in ENGINES if code == engine), "자동"))

    def _load_devices(self) -> None:
        try:
            self._mics = list_input_devices()
            self._loops = list_loopback_devices()
        except Exception as e:
            messagebox.showerror(
                "장치 오류",
                "오디오 장치 목록을 불러오지 못했습니다.\n"
                "pyaudiowpatch가 설치되어 있는지 확인해 주세요.\n\n"
                f"상세: {e}")
            self._mics, self._loops = [], []
        self.mic_combo["values"] = [d.name for d in self._mics]
        if self._mics:
            self.mic_combo.current(0)
        else:
            self.mic_on.set(False)
            self.mic_check.config(state="disabled")
            self.mic_combo.config(state="disabled")
        self.loop_combo["values"] = [d.name for d in self._loops]
        if self._loops:
            self.loop_combo.current(0)
        else:
            self.loop_on.set(False)
            self.loop_check.config(state="disabled")
            self.loop_combo.config(state="disabled")

    def _set_settings_enabled(self, enabled: bool) -> None:
        combo = "readonly" if enabled else "disabled"
        check = "normal" if enabled else "disabled"
        self.mic_check.config(state=check if self._mics else "disabled")
        self.mic_combo.config(state=combo if self._mics else "disabled")
        self.loop_check.config(state=check if self._loops else "disabled")
        self.loop_combo.config(state=combo if self._loops else "disabled")
        for w in (self.model_combo, self.lang_combo, self.trans_combo,
                  self.engine_combo):
            w.config(state=combo)

    # ---- 시작/중지 ----

    def _on_toggle(self) -> None:
        if self.engine is None:
            self._on_start()
        else:
            self._on_stop()

    def _on_start(self) -> None:
        devices: list[tuple[AudioDevice, str]] = []
        if self.mic_on.get() and self._mics and self.mic_combo.current() >= 0:
            devices.append((self._mics[self.mic_combo.current()], "마이크"))
        if self.loop_on.get() and self._loops and self.loop_combo.current() >= 0:
            devices.append((self._loops[self.loop_combo.current()], "시스템"))
        if not devices:
            messagebox.showwarning("선택 필요", "캡처할 소스를 최소 1개 선택해 주세요.")
            return

        model = self.model_combo.get() or "small"
        lang = dict(self._langs).get(self.lang_combo.get())
        target = dict(self._trans).get(self.trans_combo.get())
        engine = dict(ENGINES).get(self.engine_combo.get(), "auto")
        self._save_config(model, lang, target, engine)

        translator = None
        if target:
            try:
                from ..translate import Translator
                translator = Translator(target_lang=target)
            except Exception as e:
                messagebox.showwarning(
                    "번역 비활성",
                    f"번역기를 준비하지 못해 번역 없이 진행합니다.\n상세: {e}")

        # 세션마다 새 큐를 쓴다 — 이전 세션의 늦은 이벤트가 새 화면에 섞이지 않게
        self.event_queue = queue.Queue()
        self.engine = RealtimeEngine(model, lang, translator, self.event_queue,
                                     engine_pref=engine)
        try:
            self.engine.start(devices)
        except Exception as e:
            self.engine = None
            messagebox.showerror("시작 실패", f"캡처를 시작하지 못했습니다.\n상세: {e}")
            return
        self.toggle_btn.config(text="중지")
        self._set_settings_enabled(False)
        self.status_var.set("시작됨")

    def _on_stop(self) -> None:
        engine = self.engine
        if engine is None or self._stopping:
            return
        self._stopping = True
        self.toggle_btn.config(state="disabled")
        self.status_var.set("중지 중...")

        def work():
            try:
                engine.stop()  # 워커 join까지 블로킹 — GUI 스레드 밖에서 수행
            finally:
                self.root.after(0, self._on_stopped)

        threading.Thread(target=work, daemon=True).start()

    def _on_stopped(self) -> None:
        self._stopping = False
        self.engine = None
        if self._closing:
            self.root.destroy()
            return
        self.toggle_btn.config(text="시작", state="normal")
        self._set_settings_enabled(True)
        self.status_var.set("대기 중")

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self.engine is not None:
            if not self._stopping:
                self._on_stop()  # 완료되면 _on_stopped가 destroy한다
        else:
            self.root.destroy()

    # ---- 이벤트 폴링·표시 ----

    def _poll(self) -> None:
        try:
            while True:
                ev = self.event_queue.get_nowait()
                if isinstance(ev, TranscriptEvent):
                    self._append(ev)
                elif isinstance(ev, tuple) and len(ev) == 2 and ev[0] == "status":
                    if not self._stopping:
                        self.status_var.set(str(ev[1]))
                elif isinstance(ev, tuple) and len(ev) == 2 and ev[0] == "notice":
                    self._append_notice(str(ev[1]))
        except queue.Empty:
            pass
        self.root.after(POLL_MS, self._poll)

    def _append(self, ev: TranscriptEvent) -> None:
        at_bottom = self.text.yview()[1] >= 0.999  # 맨 아래를 보고 있을 때만 자동 스크롤
        stamp = time.strftime("%H:%M:%S", time.localtime(ev.wall_time))
        tag = ev.tag if ev.tag in ("마이크", "시스템") else "마이크"
        self.text.config(state="normal")
        self.text.insert("end", f"[{stamp}][{ev.tag}] {ev.text}\n", tag)
        if ev.translation:
            self.text.insert("end", f"    ▶ {ev.translation}\n", "번역")
        self.text.config(state="disabled")
        if at_bottom:
            self.text.see("end")

    def _append_notice(self, msg: str) -> None:
        """캡처 실패·번역 불가 같은 안내를 대본 영역에 회색으로 남긴다."""
        at_bottom = self.text.yview()[1] >= 0.999
        self.text.config(state="normal")
        self.text.insert("end", f"[안내] {msg}\n", "번역")
        self.text.config(state="disabled")
        if at_bottom:
            self.text.see("end")

    # ---- 하단 버튼 ----

    def _on_save(self) -> None:
        if not self.text.get("1.0", "end-1c").strip():
            messagebox.showinfo("저장", "저장할 내용이 없습니다.")
            return
        path = filedialog.asksaveasfilename(
            title="대본 저장",
            defaultextension=".txt",
            initialfile=time.strftime("실시간대본_%Y%m%d_%H%M%S.txt"),
            filetypes=[("텍스트 파일", "*.txt"), ("모든 파일", "*.*")])
        if not path:
            return
        # 다이얼로그가 열려 있는 동안 추가된 전사도 빠지지 않게 지금 다시 읽는다
        content = self.text.get("1.0", "end-1c")
        try:
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(content)
        except OSError as e:
            messagebox.showerror("저장 실패", f"파일을 저장하지 못했습니다.\n상세: {e}")
            return
        self.status_var.set(f"저장 완료: {path}")

    def _on_clear(self) -> None:
        if self.text.get("1.0", "end-1c").strip():
            if not messagebox.askyesno(
                    "지우기", "전사 내용을 모두 지울까요?\n"
                    "저장하지 않은 내용은 복구할 수 없습니다."):
                return
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")

    def _on_topmost(self) -> None:
        self.root.attributes("-topmost", self.topmost_var.get())


def run_app() -> None:
    """실시간 STT 창을 띄우고 mainloop를 돈다."""
    root = tk.Tk()
    RealtimeApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_app()
