# -*- coding: utf-8 -*-
"""
translate — Argos Translate(CTranslate2 기반 완전 로컬 번역)를 감싸는 모듈.

첫 사용 시에만 인터넷으로 언어팩을 내려받고(언어당 수십~수백 MB),
설치 후에는 완전히 오프라인으로 동작한다. 직행 팩이 없는 언어쌍은
영어(en) 피벗 팩 2개(src→en, en→tgt)를 설치해 자동으로 경유 번역한다.

사용 예:
    from mp3txt.translate import Translator

    tr = Translator(target_lang="ko")
    if tr.ensure_ready("en"):            # 언어팩 확인/설치 (최초 1회만 다운로드)
        result = tr.translate("Hello, world.", "en")
        if result is not None:
            print(result)
"""
import threading

# Whisper 언어코드 → Argos 코드. 대부분 동일하므로 다른 것만 적는다.
# 여기 없는 코드는 그대로 Argos에 시도한다.
_LANG_MAP = {
    "jw": "jv",  # 자바어 (Whisper는 옛 표기 jw 사용)
    "no": "nb",  # 노르웨이어 (Argos는 보크몰 nb)
}


def _to_argos_code(lang: str) -> str:
    """Whisper 언어코드를 Argos 코드로 바꾼다. 모르는 코드는 그대로 돌려준다."""
    lang = (lang or "").strip().lower()
    return _LANG_MAP.get(lang, lang)


class Translator:
    """Argos Translate 래퍼. 언어팩 준비(ensure_ready)와 번역(translate)을 담당한다.

    전사 워커 스레드에서 호출되므로 ensure_ready는 락으로 직렬화한다.
    실패한 언어쌍은 세션 내에서 다시 시도하지 않는다 (매 세그먼트마다
    네트워크 오류를 반복하지 않기 위함).
    """

    def __init__(self, target_lang: str = "ko"):
        self.target_lang = _to_argos_code(target_lang)
        self._lock = threading.Lock()
        self._ready: set[tuple[str, str]] = set()   # 이번 세션에서 확인된 언어쌍
        self._failed: set[tuple[str, str]] = set()  # 실패한 언어쌍 (재시도 금지)

    def ensure_ready(self, src_lang: str) -> bool:
        """src_lang → target_lang 번역 경로(직행 또는 en 피벗)를 준비한다.

        설치된 팩에 경로가 있으면 그대로 True. 없으면 패키지 인덱스를 갱신해
        언어팩을 내려받아 설치한다. 모든 실패는 경고 출력 후 False.
        """
        src = _to_argos_code(src_lang)
        tgt = self.target_lang
        if not src or src == tgt:
            return True  # 번역 불필요 (translate가 None을 돌려준다)

        pair = (src, tgt)
        with self._lock:
            if pair in self._ready:
                return True
            if pair in self._failed:
                return False

            try:
                import argostranslate.package as argos_pkg  # 지연 import (무거움)
            except ImportError:
                print("경고: argostranslate 패키지가 설치되어 있지 않습니다. 번역을 건너뜁니다.")
                self._failed.add(pair)
                return False

            try:
                if self._has_path(argos_pkg, src, tgt):
                    self._ready.add(pair)
                    return True

                print(f"언어팩 준비 중: {src} -> {tgt} "
                      "(최초 1회 다운로드, 언어당 수십~수백 MB)")
                argos_pkg.update_package_index()
                for p in self._pick_packages(argos_pkg, src, tgt):
                    print(f"  언어팩 다운로드 중: {p.from_code} -> {p.to_code} ...")
                    path = p.download()
                    argos_pkg.install_from_path(path)
                    print(f"  설치 완료: {p.from_code} -> {p.to_code}")

                if self._has_path(argos_pkg, src, tgt):
                    self._ready.add(pair)
                    return True
                print(f"경고: {src} -> {tgt} 번역 경로를 만들지 못했습니다. 번역을 건너뜁니다.")
                self._failed.add(pair)
                return False
            except Exception as e:
                print(f"경고: 언어팩 준비 실패 ({src} -> {tgt}): {e}")
                self._failed.add(pair)
                return False

    def translate(self, text: str, src_lang: str) -> str | None:
        """text를 target_lang으로 번역한다.

        빈/공백 입력, 원문과 같은 언어, 번역 실패면 None을 돌려준다.
        설치된 팩 그래프에서 자동으로 en 피벗을 거친다.
        """
        if not text or not text.strip():
            return None
        src = _to_argos_code(src_lang)
        if not src or src == self.target_lang:
            return None  # 번역 불필요
        try:
            import argostranslate.translate as argos_tr  # 지연 import (무거움)
            result = argos_tr.translate(text, src, self.target_lang)
        except Exception:
            return None
        result = (result or "").strip()
        return result or None

    @staticmethod
    def _has_path(argos_pkg, src: str, tgt: str) -> bool:
        """설치된 팩에 src→tgt 경로가 있는지 본다 (직행 또는 en 피벗)."""
        pairs = {(p.from_code, p.to_code) for p in argos_pkg.get_installed_packages()}
        if (src, tgt) in pairs:
            return True
        return (src, "en") in pairs and ("en", tgt) in pairs

    @staticmethod
    def _pick_packages(argos_pkg, src: str, tgt: str) -> list:
        """내려받을 팩 목록을 고른다. 직행 팩 우선, 없으면 en 피벗 팩 2개.

        이미 설치된 팩은 목록에서 뺀다. 사용 가능한 팩이 없으면 RuntimeError.
        """
        installed = {(p.from_code, p.to_code) for p in argos_pkg.get_installed_packages()}
        by_pair = {(p.from_code, p.to_code): p for p in argos_pkg.get_available_packages()}

        direct = by_pair.get((src, tgt))
        if direct is not None:
            return [direct]

        to_en = by_pair.get((src, "en"))
        from_en = by_pair.get(("en", tgt))
        if to_en is None or from_en is None:
            raise RuntimeError(f"사용 가능한 언어팩이 없습니다: {src} -> {tgt}")
        return [p for p in (to_en, from_en)
                if (p.from_code, p.to_code) not in installed]
