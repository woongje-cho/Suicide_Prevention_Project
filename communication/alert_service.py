"""
Bridge Guardian — 경고 발송 서비스
콘솔/로그/Twilio 알림을 통해 위험 임계값 초과 시 경고를 전달한다.
"""

import os
import time
import logging
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# Twilio는 선택적 의존성 — 없어도 정상 동작
try:
    from twilio.rest import Client as TwilioClient
    _TWILIO_AVAILABLE = True
except ImportError:
    _TWILIO_AVAILABLE = False

# 위험 수준 비교용 순서 매핑
_LEVEL_ORDER: Dict[str, int] = {
    "SAFE": 0,
    "OBSERVE": 1,
    "WARNING": 2,
    "DANGER": 3,
    "CRITICAL": 4,
}

# ANSI 색상 — 콘솔 출력용
_ANSI_COLORS: Dict[str, str] = {
    "WARNING":  "\033[93m",      # 노란색
    "DANGER":   "\033[91m",      # 빨간색
    "CRITICAL": "\033[1;91m",    # 굵은 빨간색
}
_ANSI_RESET = "\033[0m"


class AlertService:
    """위험 수준에 따라 콘솔·로그·Twilio 알림을 발송하는 서비스."""

    def __init__(self, config: dict) -> None:
        """
        Parameters
        ----------
        config : dict
            settings.yaml의 ``alerts`` 섹션에 대응하는 딕셔너리.
            키: enabled, console, log_file, min_alert_level, cooldown_seconds,
                save_alert_frames, save_dir,
                twilio (enabled, account_sid, auth_token,
                        from_number, to_numbers,
                        call_on_critical, sms_on_warning)
        """
        self._config = config
        self._enabled: bool = config.get("enabled", True)
        self._console: bool = config.get("console", True)
        self._cooldown_seconds: float = float(config.get("cooldown_seconds", 60))
        self._save_alert_frames: bool = config.get("save_alert_frames", False)
        self._save_dir: str = config.get("save_dir", "output/alert_frames")

        # 위험 수준 최소 임계
        min_level_str: str = config.get("min_alert_level", "WARNING").upper()
        self._min_alert_level: int = _LEVEL_ORDER.get(min_level_str, 2)

        # 쿨다운: track_id → (last_alert_time, last_level_str)
        self._cooldowns: Dict[int, tuple] = {}

        # ----- 별도 파일 로거 설정 -----
        self._alert_logger: Optional[logging.Logger] = None
        log_file: str = config.get("log_file", "")
        if log_file:
            self._alert_logger = logging.getLogger("bridge_guardian.alerts")
            self._alert_logger.setLevel(logging.DEBUG)
            # 기존 핸들러 중복 방지
            if not self._alert_logger.handlers:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(str(log_path), encoding="utf-8")
                fh.setLevel(logging.DEBUG)
                fmt = "[%(asctime)s] [%(levelname)-8s] %(message)s"
                fh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
                self._alert_logger.addHandler(fh)

        # ----- Twilio 설정 -----
        self._twilio_client: Optional[object] = None
        self._twilio_enabled: bool = False
        self._twilio_from: str = ""
        self._twilio_to_numbers: List[str] = []
        self._call_on_critical: bool = False
        self._sms_on_warning: bool = False

        twilio_cfg: dict = config.get("twilio", {})
        if twilio_cfg.get("enabled", False):
            if not _TWILIO_AVAILABLE:
                logger.warning(
                    "Twilio가 설정에서 활성화되었으나 twilio 패키지가 설치되지 않았습니다. "
                    "SMS/전화 알림이 비활성화됩니다."
                )
            else:
                # 환경 변수 우선, config fallback
                account_sid = os.environ.get(
                    "TWILIO_ACCOUNT_SID",
                    twilio_cfg.get("account_sid", ""),
                )
                auth_token = os.environ.get(
                    "TWILIO_AUTH_TOKEN",
                    twilio_cfg.get("auth_token", ""),
                )
                self._twilio_from = os.environ.get(
                    "TWILIO_FROM_NUMBER",
                    twilio_cfg.get("from_number", ""),
                )
                self._twilio_to_numbers = twilio_cfg.get("to_numbers", [])
                self._call_on_critical = twilio_cfg.get("call_on_critical", False)
                self._sms_on_warning = twilio_cfg.get("sms_on_warning", True)

                if account_sid and auth_token:
                    try:
                        self._twilio_client = TwilioClient(account_sid, auth_token)
                        self._twilio_enabled = True
                        logger.info("Twilio 클라이언트 초기화 완료 (수신 번호 %d개)", len(self._twilio_to_numbers))
                    except Exception as exc:
                        logger.error("Twilio 클라이언트 생성 실패: %s", exc)
                else:
                    logger.warning("Twilio 자격 증명(account_sid/auth_token)이 비어 있습니다.")

    # ------------------------------------------------------------------
    # 내부: 알림 발송 여부 판단
    # ------------------------------------------------------------------

    def _should_alert(self, track_id: int, risk_level: str) -> bool:
        """알림을 보내야 하는지 판단한다.

        Returns
        -------
        bool
            True면 알림 발송, False면 억제.
        """
        if not self._enabled:
            return False

        level_val = _LEVEL_ORDER.get(risk_level.upper(), 0)
        if level_val < self._min_alert_level:
            return False

        # 쿨다운 검사
        if track_id in self._cooldowns:
            last_time, last_level = self._cooldowns[track_id]
            elapsed = time.time() - last_time
            if elapsed < self._cooldown_seconds:
                last_val = _LEVEL_ORDER.get(last_level, 0)
                if level_val <= last_val:
                    # 동일하거나 낮은 수준이면 억제
                    return False

        return True

    # ------------------------------------------------------------------
    # 공개 API: 알림 발송
    # ------------------------------------------------------------------

    def send_alert(
        self,
        track_id: int,
        risk_level: str,
        risk_score: float,
        details: dict,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """위험 수준에 따라 알림을 발송한다.

        Parameters
        ----------
        track_id : int
            추적 대상 ID.
        risk_level : str
            SAFE | OBSERVE | WARNING | DANGER | CRITICAL.
        risk_score : float
            0.0 ~ 1.0 위험 점수.
        details : dict
            키: stationary_duration_s, in_roi_zone, climbing_detected,
                facing_outward, risk_trend.
        frame : np.ndarray, optional
            현재 프레임 이미지(경고 프레임 저장용).
        """
        risk_level = risk_level.upper()

        if not self._should_alert(track_id, risk_level):
            return

        # ---- 메시지 생성 ----
        dur = details.get("stationary_duration_s", 0)
        zone = details.get("in_roi_zone", "N/A")
        climb = details.get("climbing_detected", False)
        facing = details.get("facing_outward", False)
        trend = details.get("risk_trend", "stable")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = (
            f"[{risk_level}] Person ID:{track_id} | "
            f"Risk: {risk_score:.2f} | "
            f"Stationary: {dur}s | "
            f"Zone: {zone} | "
            f"Climbing: {climb} | "
            f"Facing: {facing} | "
            f"Trend: {trend} | "
            f"Time: {timestamp}"
        )

        # ---- 콘솔 출력 ----
        if self._console:
            color = _ANSI_COLORS.get(risk_level, "")
            reset = _ANSI_RESET if color else ""
            print(f"{color}{message}{reset}")

        # ---- 로그 파일 기록 ----
        if self._alert_logger is not None:
            log_level = {
                "WARNING": logging.WARNING,
                "DANGER": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }.get(risk_level, logging.INFO)
            self._alert_logger.log(log_level, message)

        # ---- 프레임 저장 ----
        if self._save_alert_frames and frame is not None:
            self._save_frame(frame, risk_level, track_id)

        # ---- Twilio SMS ----
        if (
            self._twilio_enabled
            and self._sms_on_warning
            and risk_level in ("WARNING", "DANGER", "CRITICAL")
        ):
            self._send_sms(message)

        # ---- Twilio 전화 ----
        if (
            self._twilio_enabled
            and self._call_on_critical
            and risk_level == "CRITICAL"
        ):
            self._make_call(message)

        # ---- 쿨다운 갱신 ----
        self._cooldowns[track_id] = (time.time(), risk_level)

    # ------------------------------------------------------------------
    # Twilio SMS
    # ------------------------------------------------------------------

    def _send_sms(self, message: str) -> None:
        """등록된 모든 수신 번호에 SMS를 발송한다."""
        for number in self._twilio_to_numbers:
            try:
                self._twilio_client.messages.create(
                    body=message,
                    from_=self._twilio_from,
                    to=number,
                )
                logger.info("SMS 발송 완료 → %s", number)
            except Exception as exc:
                logger.error("SMS 발송 실패 → %s: %s", number, exc)

    # ------------------------------------------------------------------
    # Twilio 음성 전화
    # ------------------------------------------------------------------

    def _make_call(self, message: str) -> None:
        """등록된 모든 수신 번호에 TwiML 음성 전화를 건다."""
        twiml = (
            '<Response>'
            f'<Say language="ko-KR">{message}</Say>'
            '</Response>'
        )
        for number in self._twilio_to_numbers:
            try:
                self._twilio_client.calls.create(
                    twiml=twiml,
                    from_=self._twilio_from,
                    to=number,
                )
                logger.info("음성 전화 발신 완료 → %s", number)
            except Exception as exc:
                logger.error("음성 전화 발신 실패 → %s: %s", number, exc)

    # ------------------------------------------------------------------
    # 프레임 저장
    # ------------------------------------------------------------------

    def _save_frame(self, frame: np.ndarray, risk_level: str, track_id: int) -> None:
        """경고 시점의 프레임을 이미지로 저장한다."""
        save_dir = Path(self._save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{risk_level}_track{track_id}_{ts}.jpg"
        filepath = save_dir / filename

        try:
            cv2.imwrite(str(filepath), frame)
            logger.debug("경고 프레임 저장: %s", filepath)
        except Exception as exc:
            logger.error("경고 프레임 저장 실패: %s", exc)

    # ------------------------------------------------------------------
    # 쿨다운 정리
    # ------------------------------------------------------------------

    def cleanup_cooldowns(self) -> None:
        """만료된 쿨다운 항목(2× cooldown_seconds 경과)을 제거한다."""
        now = time.time()
        cutoff = self._cooldown_seconds * 2
        expired = [
            tid for tid, (ts, _) in self._cooldowns.items()
            if now - ts > cutoff
        ]
        for tid in expired:
            del self._cooldowns[tid]
        if expired:
            logger.debug("쿨다운 정리: %d건 제거", len(expired))
