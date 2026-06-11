# -*- coding: utf-8 -*-
"""전사 세그먼트 + 화자분리 결과를 "참가자 N (hh:mm:ss): ..." 텍스트로 만든다.

화자 배정은 워드 타임스탬프 단위 다수결(겹침 길이 가중)로 하고,
같은 화자의 연속 발화는 한 줄로 합친다.
"""
from __future__ import annotations

from dataclasses import dataclass

from .diarize import Turn
from .transcribe import Segment

# 같은 화자라도 이 이상 쉬면 줄을 나눈다 (가독성)
MERGE_MAX_GAP_SEC = 4.0
# 화자분리 결과가 없을 때 세그먼트를 줄로 묶는 기준 간격 (이어 말한 것만 병합)
PLAIN_MERGE_GAP_SEC = 1.0
# 화자분리 없이 병합할 때 한 줄의 최대 길이 (벽글 방지)
PLAIN_MERGE_MAX_SEC = 30.0
# 화자 구간과 전혀 겹치지 않는 워드를 가장 가까운 구간에 붙일 때 허용 거리
NEAREST_TURN_MAX_SEC = 2.0


@dataclass
class Utterance:
    speaker: int | None   # 1부터 시작하는 참가자 번호, 화자분리 없으면 None
    start: float
    end: float
    text: str


def fmt_ts(seconds: float) -> str:
    """초 → hh:mm:ss (음수/결측은 00:00:00)."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class _TurnIndex:
    """시간순 정렬된 화자 구간에서, 시간순 질의에 대해 최적 화자를 찾는 스위프 인덱스."""

    def __init__(self, turns: list[Turn]):
        self.turns = turns
        self._lo = 0

    def best_speaker(self, start: float, end: float) -> str | None:
        """[start, end)와 가장 많이 겹치는 화자. 안 겹치면 근접(2초 이내) 화자."""
        turns = self.turns
        # 질의가 시간순이므로 완전히 지나간 구간은 건너뛴다
        # (겹치는 화자 구간이 있을 수 있어 5초 여유를 둔다)
        while self._lo < len(turns) and turns[self._lo][1] < start - 5.0:
            self._lo += 1

        best_label, best_overlap = None, 0.0
        nearest_label, nearest_dist = None, float("inf")
        for i in range(self._lo, len(turns)):
            t_start, t_end, label = turns[i]
            if t_start >= end + NEAREST_TURN_MAX_SEC:
                break
            overlap = min(end, t_end) - max(start, t_start)
            if overlap > best_overlap:
                best_label, best_overlap = label, overlap
            elif overlap <= 0:
                dist = max(t_start - end, start - t_end)
                if dist < nearest_dist:
                    nearest_label, nearest_dist = label, dist
        if best_label is not None:
            return best_label
        if nearest_dist <= NEAREST_TURN_MAX_SEC:
            return nearest_label
        return None


def assign_speakers(segments: list[Segment], turns: list[Turn] | None,
                    ) -> list[Utterance]:
    """세그먼트마다 화자를 배정하고 연속 발화를 합쳐 Utterance 목록을 만든다."""
    if not segments:
        return []
    if not turns:
        return _merge_plain(segments)

    index = _TurnIndex(turns)
    labeled: list[tuple[str | None, Segment]] = []
    prev_label: str | None = None
    for seg in segments:
        votes: dict[str, float] = {}
        if seg.words:
            for w in seg.words:
                label = index.best_speaker(w.start, w.end)
                if label is not None:
                    votes[label] = votes.get(label, 0.0) + max(w.end - w.start, 0.01)
        else:
            label = index.best_speaker(seg.start, seg.end)
            if label is not None:
                votes[label] = 1.0
        if votes:
            label = max(votes.items(), key=lambda kv: kv[1])[0]
        else:
            label = prev_label  # 화자분리가 놓친 구간은 직전 화자를 잇는다
        labeled.append((label, seg))
        if label is not None:
            prev_label = label

    # 앞부분에 화자 미상이 남았으면 처음 등장한 화자로 채운다
    first_known = next((lab for lab, _ in labeled if lab is not None), None)
    labeled = [(lab if lab is not None else first_known, seg) for lab, seg in labeled]

    # 등장 순서대로 참가자 번호 부여
    order: dict[str, int] = {}
    for lab, _ in labeled:
        if lab is not None and lab not in order:
            order[lab] = len(order) + 1

    # 같은 화자의 연속 세그먼트 병합
    utterances: list[Utterance] = []
    for lab, seg in labeled:
        speaker = order.get(lab) if lab is not None else None
        if (utterances
                and utterances[-1].speaker == speaker
                and seg.start - utterances[-1].end < MERGE_MAX_GAP_SEC):
            last = utterances[-1]
            last.text = f"{last.text} {seg.text}".strip()
            last.end = seg.end
        else:
            utterances.append(Utterance(speaker, seg.start, seg.end, seg.text))
    return utterances


def _merge_plain(segments: list[Segment]) -> list[Utterance]:
    """화자분리 없이 시간 간격만으로 줄을 묶는다 (speaker=None)."""
    utterances: list[Utterance] = []
    for seg in segments:
        if (utterances
                and seg.start - utterances[-1].end < PLAIN_MERGE_GAP_SEC
                and seg.end - utterances[-1].start <= PLAIN_MERGE_MAX_SEC):
            last = utterances[-1]
            last.text = f"{last.text} {seg.text}".strip()
            last.end = seg.end
        else:
            utterances.append(Utterance(None, seg.start, seg.end, seg.text))
    return utterances


def render(utterances: list[Utterance]) -> str:
    """"참가자 N (hh:mm:ss): 내용" 줄들을 빈 줄로 구분해 이어붙인다."""
    lines = []
    for u in utterances:
        ts = fmt_ts(u.start)
        if u.speaker is not None:
            lines.append(f"참가자 {u.speaker} ({ts}): {u.text}")
        else:
            lines.append(f"({ts}): {u.text}")
    return "\n\n".join(lines)
