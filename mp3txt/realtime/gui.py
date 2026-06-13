# -*- coding: utf-8 -*-
"""
gui — tkinter로 만든 실시간 음성 → 텍스트 변환 창.

두 가지 화면 모드:
  - 전체 모드: 장치·모델·언어·번역·엔진 설정 + 대본 영역 + 하단 버튼
  - 자막 모드: 테두리 없는 항상-위 오버레이 (투명도/글자 크기 조절,
    드래그 이동, 모서리 크기 조절). 윈도우 라이브 캡션과 달리
    지나간 내용을 스크롤로 볼 수 있고 복사·저장이 된다.

엔진과는 event_queue 하나로 통신하며 100ms 주기로 폴링한다.
"""
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Optional

from .capture import AudioDevice, list_input_devices, list_loopback_devices
from .engine import RealtimeEngine, TranscriptEvent

TITLE = "mp3TXT_local — 실시간 음성 텍스트 변환"
FONT_FAMILY = "Malgun Gothic"
MODELS = ["small", "base", "medium", "large-v3-turbo"]
LANGS = [("자동", None), ("한국어", "ko"), ("영어", "en"),
         ("일본어", "ja"), ("중국어", "zh")]
TRANSLATIONS = [("끔", None), ("→ 한국어", "ko"), ("→ 영어", "en")]
ENGINES = [("자동", "auto"), ("NVIDIA GPU", "cuda"),
           ("인텔 GPU", "openvino-gpu"), ("CPU", "cpu")]
POLL_MS = 100

# 일반 모드 색
LIGHT_COLORS = {"bg": "#ffffff", "fg": "#000000",
                "마이크": "#1d4ed8", "시스템": "#15803d", "번역": "#6b7280"}
# 자막 모드 색 (어두운 배경에 밝은 글씨)
DARK_COLORS = {"bg": "#101010", "fg": "#f0f0f0",
               "마이크": "#7eb6ff", "시스템": "#7fe3a0", "번역": "#a8a8a8"}
CAPTION_BAR_BG = "#1c1c1c"
CAPTION_BTN = {"bg": "#2a2a2a", "fg": "#e8e8e8",
               "activebackground": "#3a3a3a", "activeforeground": "#ffffff",
               "relief": "flat", "bd": 0, "padx": 8, "pady": 2}


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

        # 자막 모드 상태
        self._caption_mode = False
        self._saved_geometry = ""
        self._drag_origin = None
        self._resize_origin = None
        self.caption_opacity = float(self._cfg.get("caption_opacity") or 0.85)
        self.caption_font_size = int(self._cfg.get("caption_font_size") or 14)
        self.text_font = tkfont.Font(family=FONT_FAMILY, size=10)

        root.title(TITLE)
        root.geometry("760x560")
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

    def _save_config(self, **updates) -> None:
        """주어진 키만 갱신해 영속화한다. 실패해도 동작을 막지 않는다."""
        try:
            from .. import config
            cfg = config.load()
            cfg.update(updates)
            config.save(cfg)
        except Exception:
            pass

    def _build_widgets(self) -> None:
        ui_font = (FONT_FAMILY, 10)
        style = ttk.Style(self.root)
        for name in ("TLabel", "TButton", "TCheckbutton", "TCombobox"):
            style.configure(name, font=ui_font)

        # 상단 설정 프레임
        self.top = ttk.Frame(self.root, padding=8)
        self.top.pack(fill="x")
        self.top.columnconfigure(1, weight=1)

        self.mic_on = tk.BooleanVar(value=True)
        self.mic_check = ttk.Checkbutton(self.top, text="마이크", variable=self.mic_on)
        self.mic_check.grid(row=0, column=0, sticky="w")
        self.mic_combo = ttk.Combobox(self.top, state="readonly", font=ui_font)
        self.mic_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=2)

        self.loop_on = tk.BooleanVar(value=True)
        self.loop_check = ttk.Checkbutton(self.top, text="시스템 소리", variable=self.loop_on)
        self.loop_check.grid(row=1, column=0, sticky="w")
        self.loop_combo = ttk.Combobox(self.top, state="readonly", font=ui_font)
        self.loop_combo.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=2)

        self.denoise_on = tk.BooleanVar(value=bool(self._cfg.get("frontend_denoise")))
        self.denoise_check = ttk.Checkbutton(
            self.top, text="노이즈 제거 (시끄러운 환경)", variable=self.denoise_on)
        self.denoise_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        opts = ttk.Frame(self.top)
        opts.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(opts, text="모델").pack(side="left")
        self.model_combo = ttk.Combobox(opts, state="readonly", values=MODELS,
                                        width=13, font=ui_font)
        self.model_combo.pack(side="left", padx=(4, 0))
        ttk.Label(opts, text="언어").pack(side="left", padx=(8, 0))
        self.lang_combo = ttk.Combobox(opts, state="readonly",
                                       values=[lbl for lbl, _ in LANGS],
                                       width=7, font=ui_font)
        self.lang_combo.pack(side="left", padx=(4, 0))
        ttk.Label(opts, text="번역").pack(side="left", padx=(8, 0))
        self.trans_combo = ttk.Combobox(opts, state="readonly",
                                        values=[lbl for lbl, _ in TRANSLATIONS],
                                        width=9, font=ui_font)
        self.trans_combo.pack(side="left", padx=(4, 0))
        ttk.Label(opts, text="엔진").pack(side="left", padx=(8, 0))
        self.engine_combo = ttk.Combobox(opts, state="readonly",
                                         values=[lbl for lbl, _ in ENGINES],
                                         width=10, font=ui_font)
        self.engine_combo.pack(side="left", padx=(4, 0))
        self.toggle_btn = ttk.Button(opts, text="시작", command=self._on_toggle)
        self.toggle_btn.pack(side="right")

        # 하단 상태바 (텍스트 영역보다 먼저 pack해서 공간 확보)
        self.bottom = ttk.Frame(self.root, padding=(8, 2, 8, 6))
        self.bottom.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(self.bottom, textvariable=self.status_var, anchor="w").pack(
            side="left", fill="x", expand=True)
        self.topmost_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.bottom, text="항상 위", variable=self.topmost_var,
                        command=self._on_topmost).pack(side="right")
        ttk.Button(self.bottom, text="자막 모드", command=self._enter_caption).pack(
            side="right", padx=(0, 8))
        ttk.Button(self.bottom, text="지우기", command=self._on_clear).pack(
            side="right", padx=(0, 8))
        ttk.Button(self.bottom, text="저장", command=self._on_save).pack(
            side="right", padx=(0, 8))
        ttk.Button(self.bottom, text="복사", command=self._on_copy).pack(
            side="right", padx=(0, 8))

        # 자막 모드 컨트롤 바 (자막 모드에서만 pack)
        self._build_caption_bar()

        # 중앙 텍스트 영역 (읽기 전용, 두 모드가 공유 — 히스토리 끊김 없음)
        self.text = ScrolledText(self.root, wrap="word", state="disabled",
                                 font=self.text_font, borderwidth=0)
        self.text.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self._apply_colors(LIGHT_COLORS)

    def _build_caption_bar(self) -> None:
        """자막 모드 상단의 얇은 컨트롤 바 — 잡고 끌면 창이 움직인다."""
        bar = tk.Frame(self.root, bg=CAPTION_BAR_BG, height=30)
        self.caption_bar = bar

        def btn(text, cmd, width=None):
            b = tk.Button(bar, text=text, command=cmd,
                          font=(FONT_FAMILY, 9), **CAPTION_BTN)
            if width:
                b.config(width=width)
            b.pack(side="left", padx=(4, 0), pady=3)
            return b

        btn("설정", self._exit_caption)
        self.caption_toggle_btn = btn("시작", self._on_toggle)
        btn("복사", self._on_copy)
        btn("저장", self._on_save)
        btn("A−", lambda: self._caption_font_step(-2), width=3)
        btn("A+", lambda: self._caption_font_step(+2), width=3)

        tk.Label(bar, text="투명도", bg=CAPTION_BAR_BG, fg="#cccccc",
                 font=(FONT_FAMILY, 9)).pack(side="left", padx=(10, 2))
        self.opacity_var = tk.DoubleVar(value=self.caption_opacity)
        scale = tk.Scale(bar, variable=self.opacity_var, from_=0.3, to=1.0,
                         resolution=0.05, orient="horizontal", showvalue=False,
                         length=90, bg=CAPTION_BAR_BG, fg="#cccccc",
                         troughcolor="#333333", highlightthickness=0, bd=0,
                         command=self._on_opacity)
        scale.pack(side="left", pady=3)

        close = tk.Button(bar, text="✕", command=self._on_close,
                          font=(FONT_FAMILY, 9), **CAPTION_BTN)
        close.pack(side="right", padx=(0, 4), pady=3)
        # 크기 조절 그립 (오른쪽 아래 방향으로 드래그)
        grip = tk.Label(bar, text="◢", bg=CAPTION_BAR_BG, fg="#777777",
                        cursor="size_nw_se", font=(FONT_FAMILY, 9))
        grip.pack(side="right", padx=(0, 2))
        grip.bind("<ButtonPress-1>", self._resize_press)
        grip.bind("<B1-Motion>", self._resize_drag)

        # 빈 영역 드래그 → 창 이동
        for widget in (bar,):
            widget.bind("<ButtonPress-1>", self._drag_press)
            widget.bind("<B1-Motion>", self._drag_motion)

        self.caption_status = tk.Label(bar, text="", bg=CAPTION_BAR_BG,
                                       fg="#999999", font=(FONT_FAMILY, 9),
                                       anchor="e")
        self.caption_status.pack(side="right", fill="x", expand=True, padx=6)
        self.caption_status.bind("<ButtonPress-1>", self._drag_press)
        self.caption_status.bind("<B1-Motion>", self._drag_motion)

    def _apply_colors(self, colors: dict) -> None:
        self.text.configure(bg=colors["bg"], fg=colors["fg"])
        for tag in ("마이크", "시스템", "번역"):
            self.text.tag_configure(tag, foreground=colors[tag])

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
        self.denoise_check.config(state=check)
        for w in (self.model_combo, self.lang_combo, self.trans_combo,
                  self.engine_combo):
            w.config(state=combo)

    # ---- 자막 모드 ----

    def _enter_caption(self) -> None:
        if self._caption_mode:
            return
        self._caption_mode = True
        self._saved_geometry = self.root.geometry()

        self.top.pack_forget()
        self.bottom.pack_forget()
        self.text.pack_forget()
        self.caption_bar.pack(fill="x", side="top")
        self.text.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", self.caption_opacity)
        self._apply_colors(DARK_COLORS)
        self.root.configure(bg=CAPTION_BAR_BG)
        self.text_font.configure(size=self.caption_font_size)

        geometry = (self._cfg.get("caption_geometry") or "").strip()
        if not geometry:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            width = int(screen_w * 0.62)
            height = 190
            geometry = f"{width}x{height}+{(screen_w - width) // 2}+{screen_h - height - 90}"
        self.root.geometry(geometry)
        self._sync_caption_buttons()
        self.text.see("end")

    def _exit_caption(self) -> None:
        if not self._caption_mode:
            return
        self._save_caption_prefs()
        self._caption_mode = False

        self.caption_bar.pack_forget()
        self.text.pack_forget()
        self.top.pack(fill="x")
        self.bottom.pack(fill="x", side="bottom")
        self.text.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        self.root.overrideredirect(False)
        self.root.attributes("-alpha", 1.0)
        self.root.attributes("-topmost", self.topmost_var.get())
        self._apply_colors(LIGHT_COLORS)
        self.text_font.configure(size=10)
        if self._saved_geometry:
            self.root.geometry(self._saved_geometry)
        self.text.see("end")

    def _save_caption_prefs(self) -> None:
        self._save_config(
            caption_opacity=round(float(self.opacity_var.get()), 2),
            caption_font_size=self.caption_font_size,
            caption_geometry=self.root.geometry() if self._caption_mode else
            (self._cfg.get("caption_geometry") or ""),
        )
        self._cfg = self._load_config()

    def _on_opacity(self, _value=None) -> None:
        self.caption_opacity = float(self.opacity_var.get())
        if self._caption_mode:
            self.root.attributes("-alpha", self.caption_opacity)

    def _caption_font_step(self, delta: int) -> None:
        self.caption_font_size = max(10, min(30, self.caption_font_size + delta))
        if self._caption_mode:
            self.text_font.configure(size=self.caption_font_size)

    def _sync_caption_buttons(self) -> None:
        running = self.engine is not None
        label = "중지" if running else "시작"
        self.caption_toggle_btn.config(text=label)
        self.toggle_btn.config(text=label)

    # 드래그 이동 / 크기 조절 (테두리 없는 창용)
    def _drag_press(self, event) -> None:
        self._drag_origin = (event.x_root - self.root.winfo_x(),
                             event.y_root - self.root.winfo_y())

    def _drag_motion(self, event) -> None:
        if not self._caption_mode or self._drag_origin is None:
            return
        dx, dy = self._drag_origin
        self.root.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    def _resize_press(self, event) -> None:
        self._resize_origin = (event.x_root, event.y_root,
                               self.root.winfo_width(), self.root.winfo_height())

    def _resize_drag(self, event) -> None:
        if not self._caption_mode or self._resize_origin is None:
            return
        x0, y0, w0, h0 = self._resize_origin
        width = max(380, w0 + (event.x_root - x0))
        height = max(110, h0 + (event.y_root - y0))
        self.root.geometry(f"{width}x{height}")

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
        denoise = bool(self.denoise_on.get())
        self._save_config(realtime_model=model, language=lang or "auto",
                          translation_target=target or "",
                          realtime_engine=engine, frontend_denoise=denoise)

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
                                     engine_pref=engine,
                                     agc=bool(self._cfg.get("frontend_agc", True)),
                                     denoise=denoise)
        try:
            self.engine.start(devices)
        except Exception as e:
            self.engine = None
            messagebox.showerror("시작 실패", f"캡처를 시작하지 못했습니다.\n상세: {e}")
            return
        self._set_settings_enabled(False)
        self.status_var.set("시작됨")
        self._sync_caption_buttons()

    def _on_stop(self) -> None:
        engine = self.engine
        if engine is None or self._stopping:
            return
        self._stopping = True
        self.toggle_btn.config(state="disabled")
        self.caption_toggle_btn.config(state="disabled")
        self.status_var.set("중지 중...")
        self._set_caption_status("중지 중...")

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
        self.toggle_btn.config(state="normal")
        self.caption_toggle_btn.config(state="normal")
        self._set_settings_enabled(True)
        self.status_var.set("대기 중")
        self._set_caption_status("대기 중")
        self._sync_caption_buttons()

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._caption_mode:
            self._save_caption_prefs()
        if self.engine is not None:
            if not self._stopping:
                self._on_stop()  # 완료되면 _on_stopped가 destroy한다
        else:
            self.root.destroy()

    # ---- 이벤트 폴링·표시 ----

    def _set_caption_status(self, msg: str) -> None:
        try:
            self.caption_status.config(text=msg)
        except Exception:
            pass

    def _poll(self) -> None:
        try:
            while True:
                ev = self.event_queue.get_nowait()
                if isinstance(ev, TranscriptEvent):
                    self._append(ev)
                elif isinstance(ev, tuple) and len(ev) == 2 and ev[0] == "status":
                    if not self._stopping:
                        self.status_var.set(str(ev[1]))
                        self._set_caption_status(str(ev[1]))
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

    # ---- 복사/저장/지우기 ----

    def _on_copy(self) -> None:
        content = self.text.get("1.0", "end-1c")
        if not content.strip():
            self._flash_status("복사할 내용이 없습니다")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._flash_status("클립보드에 복사됨")

    def _flash_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self._set_caption_status(msg)

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
        self._flash_status(f"저장 완료: {path}")

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
