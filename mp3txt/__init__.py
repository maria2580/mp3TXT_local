# -*- coding: utf-8 -*-
"""mp3TXT_local — 완전 로컬 음성 → 텍스트 변환 패키지.

- 배치 변환: faster-whisper(전사) + pyannote(화자분리) → "참가자 N (hh:mm:ss): ..." txt
- 실시간 모드: 마이크/시스템 소리 캡처 → 발화 단위 전사 (+ 선택적 번역)
"""
