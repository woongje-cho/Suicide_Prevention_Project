"""
Bridge Guardian — Stationary Detection Module
Detects how long a tracked person has been stationary (not moving).
"""

import math
from collections import deque
from utils.logger import get_logger

logger = get_logger(__name__)


class StationaryDetector:
    """Detects stationary duration and computes risk score based on immobility."""

    def __init__(self, config: dict) -> None:
        """
        Initialize StationaryDetector with configuration.

        Args:
            config: Dictionary containing risk_assessment configuration with keys:
                - stationary_time_threshold_s (float): Duration threshold in seconds (default: 30)
                - stationary_movement_threshold_px (float): Movement threshold in pixels (default: 25)
        """
        self.stationary_time_threshold_s = config.get("stationary_time_threshold_s", 30)
        self.stationary_movement_threshold_px = config.get("stationary_movement_threshold_px", 25)

        logger.info(
            f"StationaryDetector initialized: "
            f"time_threshold={self.stationary_time_threshold_s}s, "
            f"movement_threshold={self.stationary_movement_threshold_px}px"
        )

    def get_stationary_duration(self, positions_deque: deque) -> float:
        """
        Calculate how long the person has been stationary.

        Args:
            positions_deque: Deque of (cx, cy, timestamp) tuples representing position history.

        Returns:
            Duration in seconds the person has been stationary. Returns 0.0 if fewer than 2 entries.
        """
        if len(positions_deque) < 2:
            return 0.0

        # Latest position
        cx_now, cy_now, t_now = positions_deque[-1]

        # Scan backwards to find earliest stationary timestamp
        earliest_stationary_t = t_now
        for i in range(len(positions_deque) - 2, -1, -1):
            cx_i, cy_i, t_i = positions_deque[i]

            # Compute displacement from current position
            displacement = math.sqrt((cx_now - cx_i) ** 2 + (cy_now - cy_i) ** 2)

            # If displacement exceeds threshold, stop scanning
            if displacement > self.stationary_movement_threshold_px:
                break

            # Update earliest stationary timestamp
            earliest_stationary_t = t_i

        # Calculate duration
        duration = t_now - earliest_stationary_t
        return duration

    def get_stationary_score(self, positions_deque: deque) -> float:
        """
        Compute stationary risk score (0.0 to 1.0).

        Args:
            positions_deque: Deque of (cx, cy, timestamp) tuples representing position history.

        Returns:
            Score between 0.0 and 1.0, where 1.0 indicates stationary for >= threshold duration.
        """
        duration = self.get_stationary_duration(positions_deque)
        score = min(duration / self.stationary_time_threshold_s, 1.0)
        return score
