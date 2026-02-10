"""
Bridge Guardian — Main Entry Point
자살 예방 순찰 로봇 통합 파이프라인.

모듈 구성:
  - Perception : RealSense 카메라 + YOLOv8 자세 감지 + 2D LiDAR + 센서 퓨전
  - Risk       : 자세 분석 + 정지 감지 + 난간 근접 감지 + 종합 위험도 엔진
  - Navigation : GPS + 모터 제어 + 경로 계획
  - Behavior   : 상태 머신 (순찰→감지→접근→개입→알림)
  - Communication : TTS 음성 안내 + Twilio SMS/전화 (01088405390)
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

# Perception
from perception.realsense_handler import CameraHandler, FrameData
from perception.person_detector import PersonDetector, DetectedPerson
from perception.person_tracker import PersonTracker, TrackedPerson
from perception.lidar_handler import LidarHandler
from perception.sensor_fusion import SensorFusion

# Risk Assessment
from risk_assessment.pose_analyzer import PoseAnalyzer
from risk_assessment.stationary_detector import StationaryDetector
from risk_assessment.proximity_detector import ProximityDetector
from risk_assessment.risk_engine import RiskEngine, RiskLevel

# Navigation
from navigation.gps_handler import GPSHandler
from navigation.motor_controller import MotorController
from navigation.path_planner import PathPlanner

# Behavior
from behavior.state_machine import BehaviorStateMachine, RobotState

# Communication
from communication.alert_service import AlertService
from communication.tts_service import TTSService

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
    state_machine=None,
    gps_data=None,
    lidar_handler=None,
) -> np.ndarray:
    """Draw ROI zones, bounding boxes, skeletons, risk labels,
    robot state, and LiDAR minimap on a frame copy."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    show_skeleton = config.get("show_skeleton", True)
    show_roi = config.get("show_roi_zones", True)
    show_labels = config.get("show_risk_labels", True)

    # --- ROI zone overlays ---
    if show_roi:
        for zone_name, polygon in proximity_detector.get_roi_polygons():
            cv2.polylines(vis, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
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

            flag_y = label_y
            if rs.climbing_detected:
                flag_y -= 18
                cv2.putText(vis, "[CLIMBING]", (x1, max(flag_y, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if rs.facing_outward:
                flag_y -= 18
                cv2.putText(vis, "[OUTWARD]", (x1, max(flag_y, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

    # --- Robot state overlay (top-right) ---
    if state_machine is not None:
        state_text = f"State: {state_machine.state.name}"
        dur_text = f"Duration: {state_machine.state_duration:.1f}s"
        target_text = f"Target: {state_machine.target_track_id or '-'}"
        y_off = 30
        for txt in [state_text, dur_text, target_text]:
            cv2.putText(vis, txt, (w - 280, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_off += 22

    # --- GPS overlay (bottom-left) ---
    if gps_data is not None and gps_data.valid:
        gps_text = f"GPS: {gps_data.latitude:.5f}, {gps_data.longitude:.5f}"
        cv2.putText(vis, gps_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # --- LiDAR minimap (bottom-right) ---
    if lidar_handler is not None:
        scan = lidar_handler.get_scan()
        if scan is not None and scan.valid:
            minimap_size = 120
            minimap = np.zeros((minimap_size, minimap_size, 3), dtype=np.uint8)
            center = minimap_size // 2
            max_range = 6.0  # meters visible on minimap

            for deg in range(360):
                r = scan.ranges[deg] if deg < len(scan.ranges) else float('inf')
                if r <= 0 or r > max_range or r == float('inf'):
                    continue
                rad = np.radians(deg)
                px = int(center + (r / max_range) * (minimap_size // 2) * np.sin(rad))
                py = int(center - (r / max_range) * (minimap_size // 2) * np.cos(rad))
                if 0 <= px < minimap_size and 0 <= py < minimap_size:
                    cv2.circle(minimap, (px, py), 1, (0, 255, 0), -1)

            # Robot position marker
            cv2.circle(minimap, (center, center), 3, (0, 0, 255), -1)
            cv2.rectangle(minimap, (0, 0), (minimap_size - 1, minimap_size - 1),
                          (100, 100, 100), 1)

            # Place minimap on main frame
            x_off = w - minimap_size - 10
            y_off = h - minimap_size - 10
            vis[y_off:y_off + minimap_size, x_off:x_off + minimap_size] = minimap

    return vis


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the Bridge Guardian real-time monitoring pipeline."""
    global _running

    # ---- CLI arguments ----
    parser = argparse.ArgumentParser(
        description="Bridge Guardian — 자살 예방 순찰 로봇",
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
    parser.add_argument(
        "--no-motor", action="store_true",
        help="Disable motor control (camera-only mode)",
    )
    parser.add_argument(
        "--no-lidar", action="store_true",
        help="Disable LiDAR (camera-only perception)",
    )
    parser.add_argument(
        "--no-gps", action="store_true",
        help="Disable GPS module",
    )
    parser.add_argument(
        "--no-tts", action="store_true",
        help="Disable TTS voice output",
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

    # 1. Camera (핵심 — 항상 필요)
    camera = CameraHandler(config["camera"])
    frame_width = config["camera"].get("width", 640)
    frame_height = config["camera"].get("height", 480)

    # 2. Person Detector + Tracker
    detector = PersonDetector(config["detection"])
    tracker = PersonTracker(config["tracking"])

    # 3. Risk assessment sub-modules
    pose_analyzer = PoseAnalyzer(config["risk_assessment"])
    stationary_detector = StationaryDetector(config["risk_assessment"])
    proximity_detector = ProximityDetector(
        config["risk_assessment"], frame_width, frame_height,
    )
    risk_engine = RiskEngine(config["risk_assessment"])

    # 4. LiDAR (선택적)
    lidar_handler = None
    if not args.no_lidar and config.get("lidar", {}).get("enabled", False):
        lidar_cfg = config.get("lidar", {})
        lidar_handler = LidarHandler(lidar_cfg)
        if lidar_handler.start():
            logger.info("LiDAR started successfully")
        else:
            logger.warning("LiDAR failed to start — continuing without LiDAR")

    # 5. Sensor Fusion
    fusion_cfg = config.get("sensor_fusion", {})
    fusion_cfg["frame_width"] = frame_width
    sensor_fusion = SensorFusion(fusion_cfg)

    # 6. GPS (선택적)
    gps_handler = None
    if not args.no_gps and config.get("gps", {}).get("enabled", False):
        gps_cfg = config.get("gps", {})
        gps_handler = GPSHandler(
            port=gps_cfg.get("port", "/dev/ttyACM1"),
            baudrate=gps_cfg.get("baudrate", 9600),
            update_interval=gps_cfg.get("update_rate_hz", 1.0),
        )
        if gps_handler.start():
            logger.info("GPS started successfully")
        else:
            logger.warning("GPS failed to start — continuing without GPS")

    # 7. Motor Controller (선택적)
    motor = None
    if not args.no_motor and config.get("motor", {}).get("enabled", False):
        motor_cfg = config.get("motor", {})
        motor = MotorController(
            port=motor_cfg.get("port", "/dev/ttyACM0"),
            baudrate=motor_cfg.get("baudrate", 9600),
        )
        if motor.connect():
            logger.info("Motor controller connected")
        else:
            logger.warning("Motor failed to connect — continuing without motor")

    # 8. Path Planner
    nav_cfg = config.get("navigation", {})
    path_planner = PathPlanner(nav_cfg)

    # 9. TTS (선택적)
    tts_service = None
    if not args.no_tts:
        tts_cfg = config.get("tts", {})
        tts_service = TTSService(tts_cfg)
        logger.info("TTS service initialized")

    # 10. Alert Service
    alert_service = AlertService(config["alerts"])
    if tts_service is not None:
        alert_service.set_tts_service(tts_service)
    if gps_handler is not None:
        alert_service.set_gps_handler(gps_handler)

    # 11. Behavior State Machine
    behavior_cfg = config.get("behavior", {})
    state_machine = BehaviorStateMachine(
        intervention_tts_interval=8.0,
        detection_to_approach_delay=behavior_cfg.get("detection_to_approach_delay", 3.0),
        approach_risk_threshold=behavior_cfg.get("approach_risk_threshold", 0.5),
        emergency_risk_threshold=behavior_cfg.get("emergency_risk_threshold", 0.85),
        scan_duration=behavior_cfg.get("scanning_duration_s", 5.0),
    )

    logger.info("All modules initialised. Starting main loop.")
    logger.info(
        "Modules: camera=ON, lidar=%s, gps=%s, motor=%s, tts=%s",
        "ON" if lidar_handler else "OFF",
        "ON" if gps_handler else "OFF",
        "ON" if motor else "OFF",
        "ON" if tts_service else "OFF",
    )

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

    # ---- Start patrol if motor enabled ----
    if motor is not None:
        state_machine.start_patrol()
        logger.info("Patrol started")

    # ---- Main loop ----
    frame_count = 0
    fps_start_time = time.time()
    last_cleanup_time = time.time()
    vis_config = config.get("visualization", {})
    patrol_waypoint_idx = 0

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

            # 4. Sensor Fusion (camera + LiDAR + depth)
            fused_persons = sensor_fusion.fuse(
                detections,
                lidar_handler=lidar_handler,
                depth_frame=frame_data.depth_frame,
                camera_handler=camera,
            )

            # Build fused distance lookup by track_id
            fused_dist_map = {}
            for fp in fused_persons:
                if fp.track_id is not None:
                    fused_dist_map[fp.track_id] = fp.fused_distance_m

            # 5. GPS data
            gps_data = gps_handler.get_current() if gps_handler else None

            # 6. Risk assessment per person
            highest_risk_score = 0.0
            highest_risk_level = "SAFE"
            highest_risk_track_id = None

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

                # Proximity analysis (with LiDAR distance if available)
                lidar_dist = fused_dist_map.get(person.track_id)
                prox_result = proximity_detector.check_proximity(
                    person.bbox, person.keypoints, person.kp_conf,
                    frame_data.depth_frame, camera,
                    lidar_distance_m=lidar_dist,
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

                # Track highest risk person
                if risk_state.risk_score > highest_risk_score:
                    highest_risk_score = risk_state.risk_score
                    highest_risk_level = risk_state.risk_level.name
                    highest_risk_track_id = person.track_id

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
                            "lidar_distance_m": lidar_dist,
                            "lidar_confirmed": prox_result.lidar_confirmed,
                        },
                        frame=frame_data.color_image,
                    )

            # 7. Behavior State Machine update
            action = state_machine.update(
                risk_level=highest_risk_level,
                risk_score=highest_risk_score,
                track_id=highest_risk_track_id,
                waypoint_arrived=False,  # TODO: waypoint detection from GPS/odometry
                patrol_steering=0,
                patrol_speed=config.get("motor", {}).get("patrol_speed", 80),
            )

            # Execute motor command
            if motor is not None and action.get("motor_command") is not None:
                motor.send(action["motor_command"])

            # Execute TTS message
            if tts_service is not None and action.get("tts_message"):
                tts_service.speak_text(action["tts_message"], priority=2)

            # Execute alert/call from state machine
            if action.get("make_call") and highest_risk_track_id is not None:
                alert_service.send_alert(
                    track_id=highest_risk_track_id,
                    risk_level="CRITICAL",
                    risk_score=highest_risk_score,
                    details={"state_machine_call": True},
                    frame=frame_data.color_image,
                )

            # 8. Frame counter
            frame_count += 1

            # 9. Periodic cleanup (every 30s)
            now = time.time()
            if now - last_cleanup_time >= 30.0:
                alert_service.cleanup_cooldowns()
                # Sync risk_engine with tracker — remove stale tracks
                active_ids = {p.track_id for p in tracked}
                for rs in risk_engine.get_all_states():
                    if rs.track_id not in active_ids:
                        risk_engine.remove_track(rs.track_id)
                last_cleanup_time = now

            # 10. Visualization
            vis_frame = None
            if display_enabled:
                vis_frame = draw_visualization(
                    frame_data.color_image,
                    tracked,
                    risk_engine.get_all_states(),
                    proximity_detector,
                    vis_config,
                    state_machine=state_machine,
                    gps_data=gps_data,
                    lidar_handler=lidar_handler,
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
                elif key == ord("e"):  # 'e' = emergency stop
                    state_machine.emergency_stop()
                    if motor is not None:
                        motor.stop()
                    logger.warning("Emergency stop triggered by user")
                elif key == ord("p"):  # 'p' = start patrol
                    state_machine.start_patrol()

            # 11. Save video
            if video_writer is not None:
                if vis_frame is not None:
                    video_writer.write(vis_frame)
                else:
                    video_writer.write(frame_data.color_image)

            # 12. Terminal FPS report every 100 frames
            if frame_count > 0 and frame_count % 100 == 0:
                elapsed = time.time() - fps_start_time
                if elapsed > 0:
                    state_info = state_machine.get_status()
                    logger.info(
                        "Frame %d — %.1f FPS | Tracked: %d | State: %s | "
                        "Target: %s | Risk: %.2f",
                        frame_count, frame_count / elapsed, len(tracked),
                        state_info["state"],
                        state_info["target_track_id"] or "-",
                        highest_risk_score,
                    )

    except Exception:
        logger.exception("Unhandled exception in main loop")
    finally:
        # ---- Cleanup ----
        logger.info("Shutting down …")

        # 모터 정지
        if motor is not None:
            motor.stop()
            motor.disconnect()

        # LiDAR 정지
        if lidar_handler is not None:
            lidar_handler.stop()

        # GPS 정지
        if gps_handler is not None:
            gps_handler.stop()

        # TTS 정지
        if tts_service is not None:
            tts_service.stop()

        # Camera 정지
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
