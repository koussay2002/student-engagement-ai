"""Turns per-frame visual features into an engagement label, score and forecast.

This is a transparent, rule-based model. It is a starting point: swap `instant`
for a classifier trained on labelled data (e.g. DAiSEE) without touching the rest.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

FOCUSED, DISTRACTED, CONFUSED, DROWSY, UNKNOWN = (
    "Focused", "Distracted", "Confused", "Drowsy", "Unknown")

EAR_CLOSED = 0.17      # eye aspect ratio below this = eyes closed
YAW_AWAY = 35.0        # degrees of head turn counted as looking away
ROLL_CONFUSED = 20.0   # head tilt counted as a confusion cue
MISSING_LIMIT = 8      # face-less updates before a student counts as distracted


@dataclass
class Features:
    face: bool = False
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    ear: float = 0.30
    phone: bool = False


@dataclass
class Result:
    label: str
    score: float        # smoothed engagement, 0-100
    confidence: float   # 0-1, agreement of recent frames with the label
    predicted: float    # forecast score at the horizon
    trend: float        # score change per minute


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def instant(f: Features):
    """(score, label) for a single observation, or None if nothing was seen."""
    if not f.face and not f.phone:
        return None
    score = 100.0
    score -= min(50.0, max(0.0, abs(f.yaw) - 15.0) * 1.2)
    score -= min(25.0, max(0.0, abs(f.pitch) - 25.0) * 0.8)
    closed = f.face and f.ear < EAR_CLOSED
    if closed:
        score -= 35.0
    if f.phone:
        score -= 45.0
    score = _clamp(score)

    if f.phone:
        label = DISTRACTED
    elif closed:
        label = DROWSY
    elif abs(f.yaw) > YAW_AWAY:
        label = DISTRACTED
    elif abs(f.roll) > ROLL_CONFUSED:
        label = CONFUSED
    elif score >= 70:
        label = FOCUSED
    else:
        label = DISTRACTED
    return score, label


@dataclass
class _State:
    ema: float | None = None
    labels: deque = field(default_factory=deque)
    hist: deque = field(default_factory=deque)
    missing: int = 0
    last: Result | None = None


class EngagementModel:
    def __init__(self, alpha: float = 0.15, window: int = 15,
                 horizon_min: float = 5.0, hist_len: int = 120):
        self.alpha, self.window, self.horizon = alpha, window, horizon_min
        self.hist_len = hist_len
        self.states: dict[str, _State] = {}

    def update(self, sid: str, f: Features, t: float) -> Result:
        st = self.states.get(sid)
        if st is None:
            st = self.states[sid] = _State(labels=deque(maxlen=self.window),
                                           hist=deque(maxlen=self.hist_len))
        inst = instant(f)
        if inst is None:
            st.missing += 1
            if st.missing < MISSING_LIMIT:
                return st.last or Result(UNKNOWN, 0.0, 0.0, 0.0, 0.0)
            inst = (40.0, DISTRACTED)
        else:
            st.missing = 0

        score, label = inst
        st.ema = score if st.ema is None else self.alpha * score + (1 - self.alpha) * st.ema
        st.labels.append(label)
        top, n = Counter(st.labels).most_common(1)[0]
        st.hist.append((t, st.ema))

        trend = 0.0
        if len(st.hist) >= 10 and st.hist[-1][0] - st.hist[0][0] > 1.0:
            ts = np.array([h[0] for h in st.hist]) - st.hist[0][0]
            ys = np.array([h[1] for h in st.hist])
            trend = float(np.polyfit(ts, ys, 1)[0]) * 60.0
        delta = max(-25.0, min(25.0, trend * self.horizon * 0.5))
        st.last = Result(top, st.ema, n / len(st.labels), _clamp(st.ema + delta), trend)
        return st.last

    def prune(self, active_ids) -> None:
        for sid in list(self.states):
            if sid not in active_ids:
                del self.states[sid]


def summarize(results) -> dict:
    results = list(results)
    counts = Counter(r.label for r in results)
    avg = float(np.mean([r.score for r in results])) if results else 0.0
    return {"count": len(results), "average": avg, "counts": dict(counts)}
