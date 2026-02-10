"""
Bridge Guardian — Central Risk Scoring Engine

Combines pre-computed sub-module scores (stationary, proximity, pose)
into a final risk level per tracked person.
"""

from collections import deque
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class RiskLevel(IntEnum):
    SAFE = 0
    OBSERVE = 1
    WARNING = 2
    DANGER = 3
    CRITICAL = 4


@dataclass
class PersonRiskState:
    track_id: int
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.SAFE
    raw_score: float = 0.0
    stationary_score: float = 0.0
    proximity_score: float = 0.0
    pose_danger_score: float = 0.0
    climbing_detected: bool = False
    facing_outward: bool = False
    stationary_duration_s: float = 0.0
    in_roi_zone: str = ""
    risk_trend: str = "stable"       # "rising" | "falling" | "stable"
    timestamp: float = 0.0
    _history: deque = field(default_factory=lambda: deque(maxlen=10), repr=False)


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """Aggregates per-person sub-scores into a unified risk assessment."""

    def __init__(self, config: dict) -> None:
        # --- weights ---
        weights = config.get("weights", {})
        self._w_stationary: float = weights.get("stationary_time", 0.30)
        self._w_proximity: float = weights.get("railing_proximity", 0.35)
        self._w_pose: float = weights.get("dangerous_pose", 0.35)

        # --- risk-level thresholds ---
        levels = config.get("risk_levels", {})
        self._thresh_observe: float = levels.get("observe", 0.3)
        self._thresh_warning: float = levels.get("warning", 0.5)
        self._thresh_danger: float = levels.get("danger", 0.7)
        self._thresh_critical: float = levels.get("critical", 0.85)

        # --- time multipliers ---
        tm = config.get("time_multipliers", {})
        dawn_cfg = tm.get("dawn", {})
        self._dawn_start: int = dawn_cfg.get("start", 4)
        self._dawn_end: int = dawn_cfg.get("end", 6)
        self._dawn_mult: float = dawn_cfg.get("multiplier", 1.5)

        night_cfg = tm.get("night", {})
        self._night_start: int = night_cfg.get("start", 22)
        self._night_end: int = night_cfg.get("end", 4)
        self._night_mult: float = night_cfg.get("multiplier", 1.3)

        self._default_mult: float = tm.get("default", 1.0)

        # --- per-track state ---
        self._states: Dict[int, PersonRiskState] = {}

        logger.info(
            "RiskEngine initialised  weights=(stat=%.2f, prox=%.2f, pose=%.2f)  "
            "thresholds=(obs=%.2f, warn=%.2f, dang=%.2f, crit=%.2f)",
            self._w_stationary, self._w_proximity, self._w_pose,
            self._thresh_observe, self._thresh_warning,
            self._thresh_danger, self._thresh_critical,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        track_id: int,
        stationary_score: float,
        proximity_score: float,
        pose_danger_score: float,
        climbing_detected: bool,
        facing_outward: bool,
        stationary_duration_s: float,
        in_roi_zone: str,
        timestamp: float,
    ) -> PersonRiskState:
        """Compute / update risk state for *track_id* and return it."""

        state = self._states.get(track_id)
        if state is None:
            state = PersonRiskState(track_id=track_id)
            self._states[track_id] = state

        # ---- weighted sum ----
        raw_score = (
            stationary_score * self._w_stationary
            + proximity_score * self._w_proximity
            + pose_danger_score * self._w_pose
        )

        # ---- time-of-day multiplier ----
        score = min(raw_score * self._get_time_multiplier(), 1.0)

        # ---- override rules ----
        if climbing_detected:
            score = max(score, 0.9)  # force CRITICAL

        if (
            stationary_duration_s > 60
            and facing_outward
            and in_roi_zone != ""
        ):
            score = max(score, 0.75)  # force minimum DANGER

        # ---- risk level ----
        level = self._score_to_level(score)

        # ---- trend ----
        state._history.append(score)
        trend = self._compute_trend(list(state._history))

        # ---- update state ----
        state.risk_score = score
        state.risk_level = level
        state.raw_score = raw_score
        state.stationary_score = stationary_score
        state.proximity_score = proximity_score
        state.pose_danger_score = pose_danger_score
        state.climbing_detected = climbing_detected
        state.facing_outward = facing_outward
        state.stationary_duration_s = stationary_duration_s
        state.in_roi_zone = in_roi_zone
        state.risk_trend = trend
        state.timestamp = timestamp

        return state

    def get_state(self, track_id: int) -> Optional[PersonRiskState]:
        """Return the current risk state for *track_id*, or ``None``."""
        return self._states.get(track_id)

    def get_all_states(self) -> List[PersonRiskState]:
        """Return a snapshot list of all tracked risk states."""
        return list(self._states.values())

    def remove_track(self, track_id: int) -> None:
        """Remove tracked state when a track is deleted."""
        self._states.pop(track_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_time_multiplier(self) -> float:
        """Return a risk multiplier based on current time-of-day."""
        hour = datetime.now().hour

        # dawn range: [dawn_start, dawn_end)
        if self._dawn_start <= hour < self._dawn_end:
            return self._dawn_mult

        # night range: hour >= night_start OR hour < night_end
        if hour >= self._night_start or hour < self._night_end:
            return self._night_mult

        return self._default_mult

    def _compute_trend(self, history: List[float]) -> str:
        """Simple linear-regression trend on the last ≤10 scores."""
        if len(history) < 3:
            return "stable"

        recent = history[-10:]
        x = np.arange(len(recent), dtype=np.float64)
        coeffs = np.polyfit(x, recent, 1)
        slope = coeffs[0]

        if slope > 0.02:
            return "rising"
        if slope < -0.02:
            return "falling"
        return "stable"

    def _score_to_level(self, score: float) -> RiskLevel:
        """Map a 0-1 score to the appropriate :class:`RiskLevel`."""
        if score >= self._thresh_critical:
            return RiskLevel.CRITICAL
        if score >= self._thresh_danger:
            return RiskLevel.DANGER
        if score >= self._thresh_warning:
            return RiskLevel.WARNING
        if score >= self._thresh_observe:
            return RiskLevel.OBSERVE
        return RiskLevel.SAFE
