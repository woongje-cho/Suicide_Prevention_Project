"""
Bridge Guardian - Pose Danger Analyzer
YOLOv8-pose COCO 17 keypoints를 분석하여 climbing, leaning, orientation 위험을 감지한다.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# COCO 17 Keypoint Indices
# ---------------------------------------------------------------------------
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


# ---------------------------------------------------------------------------
# Result Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ClimbingResult:
    detected: bool = False
    knee_above_hip: bool = False
    leg_angle_abnormal: bool = False
    confidence: float = 0.0


@dataclass
class OrientationResult:
    facing_outward: bool = False
    angle_deg: float = 0.0
    method: str = ""          # "ear_visibility" | "shoulder_nose"
    confidence: float = 0.0


@dataclass
class LeanResult:
    detected: bool = False
    lean_angle_deg: float = 0.0
    confidence: float = 0.0


@dataclass
class PoseDangerResult:
    danger_score: float = 0.0
    climbing: ClimbingResult = field(default_factory=ClimbingResult)
    orientation: OrientationResult = field(default_factory=OrientationResult)
    lean: LeanResult = field(default_factory=LeanResult)


# ---------------------------------------------------------------------------
# PoseAnalyzer
# ---------------------------------------------------------------------------
class PoseAnalyzer:
    """2D keypoint 기반 위험 자세 분석기.

    Parameters
    ----------
    config : dict
        ``risk_assessment`` 섹션 딕셔너리. 필요 키:
        - climbing_knee_hip_ratio (float)
        - climbing_leg_angle_threshold (float, degrees)
        - climbing_temporal_window (int, frames)
        - torso_lean_threshold_deg (float)
        - shoulder_width_ratio_side (float)
        - keypoint_confidence_min (float)
    """

    def __init__(self, config: dict) -> None:
        self._knee_hip_ratio = float(config.get("climbing_knee_hip_ratio", 0.15))
        self._leg_angle_thresh = float(config.get("climbing_leg_angle_threshold", 160))
        self._temporal_window = int(config.get("climbing_temporal_window", 15))
        self._lean_thresh_deg = float(config.get("torso_lean_threshold_deg", 25))
        self._side_ratio = float(config.get("shoulder_width_ratio_side", 0.08))
        self._kp_conf_min = float(config.get("keypoint_confidence_min", 0.3))

        logger.info(
            "PoseAnalyzer initialized: knee_hip_ratio=%.2f, leg_angle=%.0f, "
            "lean_thresh=%.0f, side_ratio=%.2f, kp_conf_min=%.2f",
            self._knee_hip_ratio, self._leg_angle_thresh,
            self._lean_thresh_deg, self._side_ratio, self._kp_conf_min,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        keypoints_history: Optional[Deque] = None,
    ) -> PoseDangerResult:
        """키포인트를 분석하여 종합 위험 결과를 반환한다.

        Parameters
        ----------
        keypoints : np.ndarray  (17, 2)
        kp_conf : np.ndarray    (17,)
        keypoints_history : deque of (keypoints, kp_conf), optional
            시계열 분석용 과거 프레임.

        Returns
        -------
        PoseDangerResult
        """
        climbing = self._detect_climbing(keypoints, kp_conf, keypoints_history)
        orientation = self._detect_orientation(keypoints, kp_conf)
        lean = self._detect_lean(keypoints, kp_conf)

        # 종합 danger score (0-1)
        score = self._compute_danger_score(climbing, orientation, lean)

        return PoseDangerResult(
            danger_score=score,
            climbing=climbing,
            orientation=orientation,
            lean=lean,
        )

    # ------------------------------------------------------------------
    # Climbing Detection
    # ------------------------------------------------------------------

    def _detect_climbing(
        self,
        kp: np.ndarray,
        conf: np.ndarray,
        history: Optional[Deque] = None,
    ) -> ClimbingResult:
        """무릎이 엉덩이 위로 올라가거나, 다리 각도가 비정상인 경우 감지."""
        result = ClimbingResult()

        # 1) Knee-above-hip check (양쪽 중 하나라도)
        knee_above = False
        if self._kp_ok(conf, L_KNEE, L_HIP):
            if kp[L_KNEE][1] < kp[L_HIP][1] - self._knee_hip_ratio * self._bbox_height(kp, conf):
                knee_above = True
        if not knee_above and self._kp_ok(conf, R_KNEE, R_HIP):
            if kp[R_KNEE][1] < kp[R_HIP][1] - self._knee_hip_ratio * self._bbox_height(kp, conf):
                knee_above = True
        result.knee_above_hip = knee_above

        # 2) Leg angle check — hip-knee-ankle 각도가 비정상적으로 펴짐
        leg_abnormal = False
        left_angle = self._angle_3pt(kp, conf, L_HIP, L_KNEE, L_ANKLE)
        right_angle = self._angle_3pt(kp, conf, R_HIP, R_KNEE, R_ANKLE)
        if left_angle is not None and left_angle > self._leg_angle_thresh:
            leg_abnormal = True
        if right_angle is not None and right_angle > self._leg_angle_thresh:
            leg_abnormal = True
        result.leg_angle_abnormal = leg_abnormal

        # 3) Temporal consistency — 최근 N 프레임에서 knee_above 빈도
        temporal_score = 0.0
        if history and len(history) >= 2:
            window = list(history)[-self._temporal_window:]
            above_count = 0
            for hist_kp, hist_conf in window:
                if self._kp_ok(hist_conf, L_KNEE, L_HIP):
                    if hist_kp[L_KNEE][1] < hist_kp[L_HIP][1] - self._knee_hip_ratio * self._bbox_height(hist_kp, hist_conf):
                        above_count += 1
                        continue
                if self._kp_ok(hist_conf, R_KNEE, R_HIP):
                    if hist_kp[R_KNEE][1] < hist_kp[R_HIP][1] - self._knee_hip_ratio * self._bbox_height(hist_kp, hist_conf):
                        above_count += 1
            temporal_score = above_count / len(window) if window else 0.0

        # Combine
        frame_score = 0.0
        if knee_above:
            frame_score += 0.5
        if leg_abnormal:
            frame_score += 0.3

        result.confidence = min(frame_score + temporal_score * 0.5, 1.0)
        result.detected = result.confidence >= 0.4
        return result

    # ------------------------------------------------------------------
    # Orientation Detection (facing direction)
    # ------------------------------------------------------------------

    def _detect_orientation(
        self,
        kp: np.ndarray,
        conf: np.ndarray,
    ) -> OrientationResult:
        """몸이 바깥쪽(난간 쪽)을 향하는지 판별.

        방법 1 (ear_visibility): 양쪽 귀가 모두 보이면 뒷모습 → 바깥 향함.
        방법 2 (shoulder_nose): 어깨 중심 대비 코 위치로 전면/후면 판별.
        """
        result = OrientationResult()

        # --- Method 1: Ear visibility ---
        l_ear_vis = conf[L_EAR] >= self._kp_conf_min
        r_ear_vis = conf[R_EAR] >= self._kp_conf_min
        nose_vis = conf[NOSE] >= self._kp_conf_min

        if l_ear_vis and r_ear_vis:
            # 양쪽 귀 모두 보임 → 뒷모습(outward)
            result.facing_outward = True
            result.method = "ear_visibility"
            result.confidence = min(conf[L_EAR], conf[R_EAR])
            # 귀 간 거리 대비 어깨 폭으로 각도 추정
            if self._kp_ok(conf, L_SHOULDER, R_SHOULDER):
                ear_dist = abs(kp[L_EAR][0] - kp[R_EAR][0])
                shoulder_dist = abs(kp[L_SHOULDER][0] - kp[R_SHOULDER][0])
                if shoulder_dist > 1:
                    ratio = ear_dist / shoulder_dist
                    # ratio가 작을수록 정면을 등지고 있음
                    result.angle_deg = max(0, 180 - ratio * 180)
            return result

        if not l_ear_vis and not r_ear_vis and nose_vis:
            # 귀 안 보이고 코 보임 → 앞모습(inward)
            result.facing_outward = False
            result.method = "ear_visibility"
            result.confidence = conf[NOSE]
            result.angle_deg = 0.0
            return result

        # --- Method 2: Shoulder-Nose lateral offset ---
        if nose_vis and self._kp_ok(conf, L_SHOULDER, R_SHOULDER):
            shoulder_mid_x = (kp[L_SHOULDER][0] + kp[R_SHOULDER][0]) / 2
            shoulder_width = abs(kp[L_SHOULDER][0] - kp[R_SHOULDER][0])
            nose_x = kp[NOSE][0]

            if shoulder_width > 1:
                ratio = abs(nose_x - shoulder_mid_x) / shoulder_width
                # 코가 어깨 중심에서 많이 벗어나면 측면/후면
                if ratio < self._side_ratio:
                    # 코가 정중앙 → 정면 또는 정후면
                    # 귀 하나라도 보이면 측면
                    if l_ear_vis or r_ear_vis:
                        result.facing_outward = False
                        result.method = "shoulder_nose"
                        result.confidence = 0.5
                    else:
                        # 귀도 안 보이고 코가 중앙 → 정면
                        result.facing_outward = False
                        result.method = "shoulder_nose"
                        result.confidence = 0.4
                else:
                    # 코가 옆으로 벗어남 → 측면 보기
                    result.facing_outward = False
                    result.method = "shoulder_nose"
                    result.confidence = 0.3

                result.angle_deg = ratio * 90  # 대략적 각도
            return result

        # 판별 불가
        result.method = "unknown"
        result.confidence = 0.0
        return result

    # ------------------------------------------------------------------
    # Lean Detection (torso forward lean)
    # ------------------------------------------------------------------

    def _detect_lean(
        self,
        kp: np.ndarray,
        conf: np.ndarray,
    ) -> LeanResult:
        """상체가 앞으로 기울어졌는지 감지.

        어깨 중점 → 엉덩이 중점 벡터의 수직 대비 기울기 각도를 측정한다.
        """
        result = LeanResult()

        # 어깨, 엉덩이 중점 계산
        if not (self._kp_ok(conf, L_SHOULDER, R_SHOULDER) and
                self._kp_ok(conf, L_HIP, R_HIP)):
            return result

        shoulder_mid = (
            (kp[L_SHOULDER][0] + kp[R_SHOULDER][0]) / 2,
            (kp[L_SHOULDER][1] + kp[R_SHOULDER][1]) / 2,
        )
        hip_mid = (
            (kp[L_HIP][0] + kp[R_HIP][0]) / 2,
            (kp[L_HIP][1] + kp[R_HIP][1]) / 2,
        )

        # 수직(y축)과 torso 벡터 간 각도
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]  # 이미지 좌표계: y 아래로 증가

        # atan2로 수직 대비 기울기 계산 (수직 = dx=0, dy<0)
        angle_rad = math.atan2(abs(dx), abs(dy))
        angle_deg = math.degrees(angle_rad)

        result.lean_angle_deg = angle_deg
        result.confidence = min(
            conf[L_SHOULDER], conf[R_SHOULDER],
            conf[L_HIP], conf[R_HIP],
        )
        result.detected = angle_deg > self._lean_thresh_deg

        return result

    # ------------------------------------------------------------------
    # Danger Score Aggregation
    # ------------------------------------------------------------------

    def _compute_danger_score(
        self,
        climbing: ClimbingResult,
        orientation: OrientationResult,
        lean: LeanResult,
    ) -> float:
        """개별 분석 결과를 0-1 종합 위험 점수로 합산."""
        score = 0.0

        # Climbing은 가장 위험 — 감지 시 높은 점수
        if climbing.detected:
            score += 0.6 * climbing.confidence
        elif climbing.knee_above_hip:
            score += 0.3 * climbing.confidence

        # 바깥 향함
        if orientation.facing_outward:
            score += 0.2 * orientation.confidence

        # 기울어짐
        if lean.detected:
            # 기울기 비례 가중
            lean_ratio = min(lean.lean_angle_deg / 45.0, 1.0)
            score += 0.2 * lean_ratio * lean.confidence

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _kp_ok(self, conf: np.ndarray, *indices: int) -> bool:
        """지정한 keypoint들이 모두 confidence 임계 이상인지 확인."""
        return all(conf[i] >= self._kp_conf_min for i in indices)

    def _bbox_height(self, kp: np.ndarray, conf: np.ndarray) -> float:
        """보이는 keypoint들로 대략적 bbox 높이를 추정."""
        visible = [kp[i][1] for i in range(17) if conf[i] >= self._kp_conf_min]
        if len(visible) < 2:
            return 100.0  # fallback
        return max(visible) - min(visible)

    def _angle_3pt(
        self,
        kp: np.ndarray,
        conf: np.ndarray,
        a: int, b: int, c: int,
    ) -> Optional[float]:
        """세 keypoint(a-b-c)가 이루는 각도(도)를 반환. confidence 부족 시 None."""
        if not self._kp_ok(conf, a, b, c):
            return None

        ba = kp[a] - kp[b]  # b→a 벡터
        bc = kp[c] - kp[b]  # b→c 벡터

        cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return math.degrees(math.acos(float(cos_val)))
