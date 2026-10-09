"""Gộp quyết định nhiều khung hình của CÙNG một quả (theo track ID) → một quyết định khi quả qua vạch.

Trên băng chuyền, cơ cấu gạt đặt sau camera; quyết định phát ra đúng một lần khi tâm quả vượt `line_x`
(pixel, theo chiều băng chạy +x). Khi nối phần cứng thật: thời điểm kích van/gạt = khoảng cách từ vạch tới
cơ cấu gạt / tốc độ băng.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

from .pipeline import Decision


@dataclass
class _Track:
    probs: list[float] = field(default_factory=list)  # khung nhìn trọn quả
    partial_probs: list[float] = field(default_factory=list)  # khung quả chạm mép (chỉ dùng khi không có khung trọn)
    fruits: Counter = field(default_factory=Counter)
    last_x: float | None = None
    decided: bool = False
    age: int = 0


class EmaSmoother:
    """Làm mượt P(hư) theo thời gian cho từng track (EMA) → nhãn hiển thị không nhảy qua lại ở quả lưng chừng.

    Chỉ dùng để HIỂN THỊ từng khung; quyết định gạt vẫn do TrackVoter (trung bình cả track khi qua vạch).
    """

    def __init__(self, alpha: float = 0.3, threshold: float = 0.5, band: float = 0.1, max_age: int = 60):
        self.alpha, self.threshold, self.band, self.max_age = alpha, threshold, band, max_age
        self.ema: dict[int, float] = {}
        self.state: dict[int, bool] = {}  # True = đang báo hư (vùng trễ: chỉ đổi khi vượt ngưỡng ± band)
        self.age: dict[int, int] = {}

    def update(self, decisions: list[Decision]) -> list[Decision]:
        out = []
        for d in decisions:
            if d.track_id is None or d.partial:  # quả đang vào khung: không cập nhật, để nguyên (vẽ xám)
                out.append(d)
                continue
            prev = self.ema.get(d.track_id)
            p = d.p_defect if prev is None else self.alpha * d.p_defect + (1 - self.alpha) * prev
            self.ema[d.track_id] = p
            self.age[d.track_id] = 0
            was = self.state.get(d.track_id)
            if was is None:
                bad = p >= self.threshold
            else:
                bad = p >= self.threshold - self.band if was else p >= self.threshold + self.band
            self.state[d.track_id] = bad
            out.append(replace(d, p_defect=p, quality="defect" if bad else "good", action="reject" if bad else "accept"))
        seen = {d.track_id for d in decisions}
        for tid in list(self.age):
            if tid not in seen:
                self.age[tid] += 1
                if self.age[tid] > self.max_age:
                    del self.age[tid], self.ema[tid], self.state[tid]
        return out


class TrackVoter:
    def __init__(self, line_x: float, threshold: float = 0.5, min_frames: int = 2, max_age: int = 60):
        self.line_x = line_x
        self.threshold = threshold
        self.min_frames = min_frames
        self.max_age = max_age
        self.tracks: dict[int, _Track] = {}

    def update(self, decisions: list[Decision]) -> list[dict]:
        events = []
        seen = set()
        for d in decisions:
            if d.track_id is None:
                continue
            t = self.tracks.setdefault(d.track_id, _Track())
            seen.add(d.track_id)
            t.age = 0
            cx = (d.box[0] + d.box[2]) / 2
            (t.partial_probs if d.partial else t.probs).append(d.p_defect)
            t.fruits[d.fruit] += 1
            if t.last_x is None and cx >= self.line_x:
                # track sinh ra khi đã qua vạch = tracker đổi ID của quả đã quyết định → bỏ qua, tránh gạt 2 lần
                t.decided = True
            crossed = cx >= self.line_x and t.last_x is not None and t.last_x < self.line_x
            t.last_x = cx
            if crossed and not t.decided:
                t.decided = True
                ps = t.probs or t.partial_probs
                p = sum(ps) / len(ps)
                events.append({
                    "track_id": d.track_id,
                    "fruit": t.fruits.most_common(1)[0][0],
                    "action": "reject" if p >= self.threshold else "accept",
                    "p_defect": round(p, 4),
                    "frames": len(ps),
                    "low_evidence": len(t.probs) < self.min_frames,
                })
        for tid in list(self.tracks):  # dọn track đã mất
            if tid not in seen:
                self.tracks[tid].age += 1
                if self.tracks[tid].age > self.max_age:
                    del self.tracks[tid]
        return events
