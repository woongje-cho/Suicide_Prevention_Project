"""
Bridge Guardian — Main Entry Point
Real-time suicide prevention monitoring pipeline with OpenCV visualization.
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from utils.logger import get_logger, setup_logging
from perception.realsense_handler import CameraHandler, FrameData
from perception.person_detector import PersonDetector, DetectedPerson
from perception.person_tracker import PersonTracker, TrackedPerson
from risk_assessment.pose_analyzer import PoseAnalyzer
from risk_assessment.stationary_detector import StationaryDetector
from risk_assessment.proximity_detector import ProximityDetector
from risk_assessment.risk_engine import RiskEngine, RiskLevel
from communication.alert_service import AlertService

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global shutdown signal
# ---------------------------------------------------------------------------
_running = True


def _signal_handler(sig, frame):
    """Handle Ctrl+C / SIGINT for graceful shutdown."""
    global _running
    logger.info("Shutdown signal received")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)

# ---------------------------------------------------------------------------
# COCO 17 skeleton connections for visualization
# ---------------------------------------------------------------------------
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 11), (6, 12),                          # torso
    (11, 12),                                  # hips
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

# Risk-level → BGR colour mapping
_RISK_COLORS = {
    RiskLevel.SAFE:     (0, 200, 0),       # green
    RiskLevel.OBSERVE:  (0, 200, 0),       # green
    RiskLevel.WARNING:  (0, 255, 255),     # yellow
    RiskLevel.DANGER:   (0, 0, 255),       # red
    RiskLevel.CRITICAL: (255, 0, 255),     # magenta
}


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    """Load a YAML configuration file and return it as a dict."""
    logger.info("Loading config from: %s", path)
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_visualization(
    frame: np.ndarray,
    tracked_persons: list,
    risk_states: list,
    proximity_detector: ProximityDetector,
    config: dict,
) -> np.ndarray:
    """Draw ROI zones, bounding boxes, skeletons, and risk labels on a frame copy.

    Parameters
    ----------
    frame : np.ndarray
        BGR colour image.
    tracked_persons : list[TrackedPerson]
        Currently tracked persons.
    risk_states : list[PersonRiskState]
        All current risk states from the engine.
    proximity_detector : ProximityDetector
        Used for ROI polygon retrieval.
    config : dict
        ``visualization`` section from settings.yaml.

    Returns
    -------
    np.ndarray
        Annotated frame (copy of original).
    """
    vis = frame.copy()
    show_skeleton = config.get("show_skeleton", True)
    show_roi = config.get("show_roi_zones", True)
    show_labels = config.get("show_risk_labels", True)
    show_fps = config.get("show_fps", True)

    # --- ROI zone overlays ---
    if show_roi:
        for zone_name, polygon in proximity_detector.get_roi_polygons():
            cv2.polylines(vis, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
            # Semi-transparent fill
            overlay = vis.copy()
            cv2.fillPoly(overlay, [polygon], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

    # Build risk-state lookup by track_id
    risk_map = {rs.track_id: rs for rs in risk_states}

    # --- Per-person overlays ---
    for person in tracked_persons:
        rs = risk_map.get(person.track_id)
        if rs is None:
            continue

        color = _RISK_COLORS.get(rs.risk_level, (0, 200, 0))
        x1, y1, x2, y2 = person.bbox.astype(int)

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Skeleton
        if show_skeleton:
            kp = person.keypoints
            kc = person.kp_conf
            for i, j in SKELETON_CONNECTIONS:
                if kc[i] > 0.3 and kc[j] > 0.3:
                    pt1 = (int(kp[i][0]), int(kp[i][1]))
                    pt2 = (int(kp[j][0]), int(kp[j][1]))
                    cv2.line(vis, pt1, pt2, color, 2)
            # Keypoint dots
            for idx in range(17):
                if kc[idx] > 0.3:
                    cx_kp, cy_kp = int(kp[idx][0]), int(kp[idx][1])
                    cv2.circle(vis, (cx_kp, cy_kp), 3, color, -1)

        # Risk label
        if show_labels:
            label = f"ID:{person.track_id} {rs.risk_level.name} {rs.risk_score:.2f}"
            label_y = max(y1 - 10, 15)
            cv2.putText(vis, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Extra flags
            flag_y = label_y
            if rs.climbing_detected:
                flag_y -= 18
                cv2.putText(vis, "[CLIMBING]", (x1, max(flag_y, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if rs.facing_outward:
                flag_y -= 18
                cv2.putText(vis, "[OUTWARD]", (x1, max(flag_y, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

    return vis


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the Bridge Guardian real-time monitoring pipeline."""
    global _running

    # ---- CLI arguments ----
    parser = argparse.ArgumentParser(
        description="Bridge Guardian — Real-time suicide prevention monitoring",
    )
    parser.add_argument(
        "--config", type=str, default="config/settings.yaml",
        help="Path to YAML config file (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Override camera source (file path or 'realsense')",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable OpenCV visualization window",
    )
    parser.add_argument(
        "--save-video", type=str, default=None,
        help="Output path for saving annotated video",
    )
    args = parser.parse_args()

    # ---- Setup ----
    setup_logging()
    config = load_config(args.config)

    # Source override
    if args.source is not None:
        config["camera"]["source"] = args.source
        logger.info("Camera source overridden to: %s", args.source)

    display_enabled = not args.no_display

    # ---- Initialize modules ----
    logger.info("Initialising pipeline modules …")

    # 1. Camera
    camera = CameraHandler(config["camera"])
    frame_width = config["camera"].get("width", 640)
    frame_height = config["camera"].get("height", 480)

    # 2. Detector
    detector = PersonDetector(config["detection"])

    # 3. Tracker
    tracker = PersonTracker(config["tracking"])

    # 4. Risk sub-modules
    pose_analyzer = PoseAnalyzer(config["risk_assessment"])
    stationary_detector = StationaryDetector(config["risk_assessment"])
    proximity_detector = ProximityDetector(
        config["risk_assessment"], frame_width, frame_height,
    )

    # 5. Risk engine
    risk_engine = RiskEngine(config["risk_assessment"])

    # 6. Alert service
    alert_service = AlertService(config["alerts"])

    logger.info("All modules initialised. Starting main loop.")

    # ---- Video writer (optional) ----
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = config["camera"].get("fps", 30)
        video_writer = cv2.VideoWriter(
            args.save_video, fourcc, fps_out, (frame_width, frame_height),
        )
        logger.info("Video writer opened: %s (%dx%d @ %d fps)",
                     args.save_video, frame_width, frame_height, fps_out)

    # ---- Main loop ----
    frame_count = 0
    fps_start_time = time.time()
    last_cleanup_time = time.time()
    vis_config = config.get("visualization", {})

    try:
        while _running:
            # 1. Acquire frame
            frame_data = camera.get_frames()
            if frame_data is None:
                logger.info("No more frames — exiting loop")
                break

            # 2. Detect persons
            detections = detector.detect(frame_data.color_image)

            # 3. Update tracker
            tracked = tracker.update(
                detections, frame_data.timestamp,
                frame_data.depth_frame, camera,
            )

            # 4. Risk assessment per person
            for person in tracked:
                # Pose analysis
                pose_result = pose_analyzer.analyze(
                    person.keypoints, person.kp_conf, person.keypoints_history,
                )

                # Stationary analysis
                stationary_score = stationary_detector.get_stationary_score(
                    person.positions,
                )
                stationary_duration = stationary_detector.get_stationary_duration(
                    person.positions,
                )

                # Proximity analysis
                prox_result = proximity_detector.check_proximity(
                    person.bbox, person.keypoints, person.kp_conf,
                    frame_data.depth_frame, camera,
                )

                # Central risk assessment
                risk_state = risk_engine.assess(
                    track_id=person.track_id,
                    stationary_score=stationary_score,
                    proximity_score=prox_result.proximity_score,
                    pose_danger_score=pose_result.danger_score,
                    climbing_detected=pose_result.climbing.detected,
                    facing_outward=pose_result.orientation.facing_outward,
                    stationary_duration_s=stationary_duration,
                    in_roi_zone=prox_result.zone_name,
                    timestamp=frame_data.timestamp,
                )

                # Alert if WARNING or above
                if risk_state.risk_level >= RiskLevel.WARNING:
                    alert_service.send_alert(
                        track_id=person.track_id,
                        risk_level=risk_state.risk_level.name,
                        risk_score=risk_state.risk_score,
                        details={
                            "stationary_duration_s": stationary_duration,
                            "in_roi_zone": prox_result.zone_name,
                            "climbing_detected": pose_result.climbing.detected,
                            "facing_outward": pose_result.orientation.facing_outward,
                            "risk_trend": risk_state.risk_trend,
                        },
                        frame=frame_data.color_image,
                    )

            # 5. Frame counter (always increment for FPS reporting)
            frame_count += 1

            # 6. Periodic cooldown cleanup (every 30 s)
            now = time.time()
            if now - last_cleanup_time >= 30.0:
                alert_service.cleanup_cooldowns()
                # Sync risk_engine with tracker — remove stale tracks
                active_ids = {p.track_id for p in tracked}
                for rs in risk_engine.get_all_states():
                    if rs.track_id not in active_ids:
                        risk_engine.remove_track(rs.track_id)
                last_cleanup_time = now

            # 7. Visualization
            vis_frame = None
            if display_enabled:
                vis_frame = draw_visualization(
                    frame_data.color_image,
                    tracked,
                    risk_engine.get_all_states(),
                    proximity_detector,
                    vis_config,
                )

                # FPS overlay
                if vis_config.get("show_fps", True):
                    elapsed = time.time() - fps_start_time
                    if elapsed > 0:
                        fps_val = frame_count / elapsed
                        cv2.putText(
                            vis_frame,
                            f"FPS: {fps_val:.1f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                cv2.imshow("Bridge Guardian", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    _running = False

            # 8. Save video
            if video_writer is not None:
                if vis_frame is not None:
                    video_writer.write(vis_frame)
                else:
                    # No display → save raw frame
                    video_writer.write(frame_data.color_image)

            # 9. Terminal FPS report every 100 frames
            if frame_count > 0 and frame_count % 100 == 0:
                elapsed = time.time() - fps_start_time
                if elapsed > 0:
                    logger.info(
                        "Frame %d — %.1f FPS  |  Tracked: %d",
                        frame_count, frame_count / elapsed, len(tracked),
                    )

    except Exception:
        logger.exception("Unhandled exception in main loop")
    finally:
        # ---- Cleanup ----
        logger.info("Shutting down …")
        camera.stop()
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            logger.info("Video writer released")
        logger.info("Bridge Guardian stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
