"""
Bridge Guardian — Proximity Detector
Detects whether a person is in a defined ROI (railing zone) and validates via depth.
"""

import cv2
import numpy as np
from typing import Any, List, Optional, Tuple
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProximityResult:
    """Result of proximity detection"""
    in_roi: bool
    zone_name: str
    proximity_score: float
    depth_m: Optional[float]
    depth_valid: bool
    lidar_distance_m: Optional[float] = None
    lidar_confirmed: bool = False


class ProximityDetector:
    """Detects proximity to railing zones using ROI masks and depth validation"""

    def __init__(self, config: dict, frame_width: int, frame_height: int) -> None:
        """
        Initialize proximity detector with ROI zones and depth thresholds.

        Args:
            config: Dictionary from the ``risk_assessment`` section of settings.yaml.
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        # ROI zones
        zones_config = config.get("railing_roi_zones", [])

        # Depth thresholds
        self.railing_depth_min_m = config.get("railing_depth_min_m", 0.5)
        self.railing_depth_max_m = config.get("railing_depth_max_m", 3.0)

        # Pre-compute ROI masks
        self.roi_zones = []
        for zone in zones_config:
            name = zone.get("name", "unknown")
            polygon_norm = zone.get("polygon", [])
            
            if not polygon_norm:
                logger.warning(f"Zone '{name}' has empty polygon, skipping")
                continue
            
            # Convert normalized polygon to pixel coordinates
            pixel_polygon = self._normalize_to_pixel(polygon_norm)
            
            # Create binary mask
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.fillPoly(mask, [pixel_polygon], 255)
            
            self.roi_zones.append((name, mask, pixel_polygon))
            logger.info(f"Loaded ROI zone '{name}' with {len(polygon_norm)} vertices")

        logger.info(f"ProximityDetector initialized with {len(self.roi_zones)} zones, "
                   f"depth range [{self.railing_depth_min_m}, {self.railing_depth_max_m}]m")

    def _normalize_to_pixel(self, polygon_norm: list) -> np.ndarray:
        """
        Convert normalized polygon coordinates to pixel coordinates.

        Args:
            polygon_norm: List of [x, y] normalized coordinates (0-1 range)

        Returns:
            np.ndarray of shape (N, 2) with int32 pixel coordinates
        """
        polygon_pixel = []
        for x_norm, y_norm in polygon_norm:
            x_pixel = int(x_norm * self.frame_width)
            y_pixel = int(y_norm * self.frame_height)
            polygon_pixel.append([x_pixel, y_pixel])
        
        return np.array(polygon_pixel, dtype=np.int32)

    def get_foot_point(
        self, 
        bbox: np.ndarray, 
        keypoints: np.ndarray, 
        kp_conf: np.ndarray
    ) -> Tuple[int, int]:
        """
        Get foot point from keypoints or bbox.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            keypoints: Keypoint coordinates array of shape (17, 2)
            kp_conf: Keypoint confidence array of shape (17,)

        Returns:
            Tuple of (x, y) foot point coordinates
        """
        # Ankle keypoint indices (COCO format)
        L_ANKLE = 15
        R_ANKLE = 16
        CONF_THRESHOLD = 0.3

        left_ankle_conf = kp_conf[L_ANKLE] if len(kp_conf) > L_ANKLE else 0.0
        right_ankle_conf = kp_conf[R_ANKLE] if len(kp_conf) > R_ANKLE else 0.0

        # Try to use ankle keypoints
        if left_ankle_conf > CONF_THRESHOLD and right_ankle_conf > CONF_THRESHOLD:
            # Both ankles visible - use midpoint
            left_ankle = keypoints[L_ANKLE]
            right_ankle = keypoints[R_ANKLE]
            foot_x = int((left_ankle[0] + right_ankle[0]) / 2)
            foot_y = int((left_ankle[1] + right_ankle[1]) / 2)
            return (foot_x, foot_y)
        elif left_ankle_conf > CONF_THRESHOLD:
            # Only left ankle visible
            left_ankle = keypoints[L_ANKLE]
            return (int(left_ankle[0]), int(left_ankle[1]))
        elif right_ankle_conf > CONF_THRESHOLD:
            # Only right ankle visible
            right_ankle = keypoints[R_ANKLE]
            return (int(right_ankle[0]), int(right_ankle[1]))
        
        # Fallback: bottom-center of bbox
        x1, y1, x2, y2 = bbox
        foot_x = int((x1 + x2) / 2)
        foot_y = int(y2)
        return (foot_x, foot_y)

    def check_proximity(
        self,
        bbox: np.ndarray,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        camera_handler: Optional[Any] = None,
        lidar_distance_m: Optional[float] = None,
    ) -> ProximityResult:
        """
        Check if person is in proximity to railing zone.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            keypoints: Keypoint coordinates array of shape (17, 2)
            kp_conf: Keypoint confidence array of shape (17,)
            depth_frame: Optional depth frame for depth validation
            camera_handler: Optional camera handler with get_median_depth_in_bbox method
            lidar_distance_m: Optional LiDAR distance from sensor fusion (metres)

        Returns:
            ProximityResult with proximity information
        """
        # Get foot point
        foot_x, foot_y = self.get_foot_point(bbox, keypoints, kp_conf)

        # Clamp to frame bounds
        foot_x = max(0, min(foot_x, self.frame_width - 1))
        foot_y = max(0, min(foot_y, self.frame_height - 1))

        # Check each ROI zone
        for zone_name, mask, pixel_polygon in self.roi_zones:
            if mask[foot_y, foot_x] > 0:
                # Person is in ROI
                base_score = 0.7
                depth_m = None
                depth_valid = False
                lidar_confirmed = False

                # Validate with depth if available
                if depth_frame is not None and camera_handler is not None:
                    # Get small bbox around foot point
                    x1_depth = max(0, foot_x - 5)
                    y1_depth = max(0, foot_y - 5)
                    x2_depth = min(self.frame_width - 1, foot_x + 5)
                    y2_depth = min(self.frame_height - 1, foot_y + 5)

                    # Get median depth
                    depth_m = camera_handler.get_median_depth_in_bbox(
                        depth_frame, x1_depth, y1_depth, x2_depth, y2_depth
                    )

                    if depth_m > 0:
                        # Check if depth is in valid range
                        if self.railing_depth_min_m <= depth_m <= self.railing_depth_max_m:
                            base_score = 1.0
                            depth_valid = True
                        else:
                            base_score = 0.4
                            depth_valid = False
                    # else: depth_m == 0, keep base_score = 0.7, depth_valid = False

                # LiDAR confirmation: validate with lidar distance if available
                if lidar_distance_m is not None:
                    if self.railing_depth_min_m <= lidar_distance_m <= self.railing_depth_max_m:
                        lidar_confirmed = True
                        # Boost score when LiDAR confirms proximity
                        if depth_valid:
                            base_score = min(1.0, base_score + 0.1)
                        else:
                            # LiDAR alone can raise score even without depth
                            base_score = max(base_score, 0.85)

                return ProximityResult(
                    in_roi=True,
                    zone_name=zone_name,
                    proximity_score=base_score,
                    depth_m=depth_m,
                    depth_valid=depth_valid,
                    lidar_distance_m=lidar_distance_m,
                    lidar_confirmed=lidar_confirmed,
                )

        # Not in any ROI
        return ProximityResult(
            in_roi=False,
            zone_name="",
            proximity_score=0.0,
            depth_m=None,
            depth_valid=False,
            lidar_distance_m=lidar_distance_m,
            lidar_confirmed=False,
        )

    def get_roi_masks(self) -> List[Tuple[str, np.ndarray]]:
        """
        Get ROI masks for visualization.

        Returns:
            List of (zone_name, mask) tuples
        """
        return [(name, mask) for name, mask, _ in self.roi_zones]

    def get_roi_polygons(self) -> List[Tuple[str, np.ndarray]]:
        """
        Get ROI polygons for visualization.

        Returns:
            List of (zone_name, pixel_polygon) tuples
        """
        return [(name, polygon) for name, _, polygon in self.roi_zones]
