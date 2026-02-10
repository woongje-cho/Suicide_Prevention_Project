"""
Bridge Guardian — 행동 상태 머신
로봇의 전체 행동 상태를 관리하고, 센서/위험도 입력에 따라 행동을 결정한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from navigation.motor_controller import MotorCommand

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Robot state enum
# ---------------------------------------------------------------------------


class RobotState(Enum):
    """로봇 행동 상태"""

    IDLE = auto()              # 대기 (시스템 시작 전 / 수동 정지)
    PATROL = auto()            # 순찰 중
    SCANNING = auto()          # 웨이포인트 도착 후 주변 스캔
    DETECTED = auto()          # 위험 대상 감지됨 (관찰 단계)
    APPROACHING = auto()       # 대상에게 접근 중
    INTERVENING = auto()       # 개입 (TTS 음성 재생, 대화 시도)
    ALERTING = auto()          # 관제센터 알림 전송 중
    EMERGENCY_STOP = auto()    # 비상 정지


# ---------------------------------------------------------------------------
# Korean TTS messages for each intervention phase
# ---------------------------------------------------------------------------

_TTS_MESSAGES: Dict[str, List[str]] = {
    "greeting": [
        "안녕하세요. 저는 다리 위를 순찰하는 안전 로봇입니다.",
        "혹시 괜찮으세요? 도움이 필요하시면 말씀해 주세요.",
    ],
    "concern": [
        "많이 힘드셨나요? 잠시 이야기를 나눠볼까요?",
        "혼자 계신 것 같아 걱정이 됩니다. 괜찮으신가요?",
    ],
    "support": [
        "당신은 소중한 사람입니다. 도움을 드리고 싶습니다.",
        "자살예방상담전화 1393으로 연결해 드릴까요?",
        "잠시만요. 전문 상담사와 통화할 수 있도록 도와드리겠습니다.",
    ],
    "emergency": [
        "위급 상황이 감지되었습니다. 119에 연락하고 있습니다.",
        "제발 조금만 기다려 주세요. 도움이 곧 도착합니다.",
    ],
    "waiting": [
        "제가 여기 있겠습니다. 천천히 이야기해 주세요.",
    ],
}


# ---------------------------------------------------------------------------
# State context (내부 상태 추적)
# ---------------------------------------------------------------------------


@dataclass
class _StateContext:
    """상태 머신 내부 컨텍스트"""

    state: RobotState = RobotState.IDLE
    previous_state: RobotState = RobotState.IDLE
    state_enter_time: float = 0.0

    # 감지 대상 정보
    target_track_id: Optional[int] = None
    target_risk_level: str = ""
    target_risk_score: float = 0.0

    # 개입 단계 추적
    intervention_phase: int = 0       # 0=greeting, 1=concern, 2=support
    last_tts_time: float = 0.0
    alert_sent: bool = False
    call_made: bool = False

    # 스캔 타이머
    scan_start_time: float = 0.0
    scan_duration: float = 5.0        # 스캔 시간 (초)


# ---------------------------------------------------------------------------
# Behavior State Machine
# ---------------------------------------------------------------------------


class BehaviorStateMachine:
    """
    Bridge Guardian 행동 상태 머신.

    ``update()``를 매 프레임 호출하면, 현재 센서 입력에 따라
    상태를 전이하고 실행할 행동(action dict)을 반환한다.

    Parameters
    ----------
    intervention_tts_interval : float
        TTS 메시지 간 최소 간격 (초).
    detection_to_approach_delay : float
        DETECTED → APPROACHING 전이 전 대기 시간 (초).
    approach_risk_threshold : float
        APPROACHING으로 전이하는 최소 위험 점수.
    emergency_risk_threshold : float
        EMERGENCY/ALERTING으로 전이하는 위험 점수.
    scan_duration : float
        SCANNING 상태 유지 시간 (초).
    """

    def __init__(
        self,
        intervention_tts_interval: float = 8.0,
        detection_to_approach_delay: float = 3.0,
        approach_risk_threshold: float = 0.5,
        emergency_risk_threshold: float = 0.85,
        scan_duration: float = 5.0,
    ) -> None:
        self._ctx = _StateContext()
        self._ctx.scan_duration = scan_duration

        self._tts_interval = intervention_tts_interval
        self._detect_delay = detection_to_approach_delay
        self._approach_threshold = approach_risk_threshold
        self._emergency_threshold = emergency_risk_threshold

        logger.info(
            "BehaviorStateMachine 초기화: 접근 임계값=%.2f, 비상 임계값=%.2f",
            self._approach_threshold,
            self._emergency_threshold,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> RobotState:
        """현재 로봇 상태"""
        return self._ctx.state

    @property
    def previous_state(self) -> RobotState:
        """이전 로봇 상태"""
        return self._ctx.previous_state

    @property
    def state_duration(self) -> float:
        """현재 상태 유지 시간 (초)"""
        return time.time() - self._ctx.state_enter_time

    @property
    def target_track_id(self) -> Optional[int]:
        """현재 추적 중인 대상 ID"""
        return self._ctx.target_track_id

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _transition(self, new_state: RobotState) -> None:
        """상태를 전이한다."""
        if new_state == self._ctx.state:
            return
        old = self._ctx.state
        self._ctx.previous_state = old
        self._ctx.state = new_state
        self._ctx.state_enter_time = time.time()

        logger.info("상태 전이: %s → %s", old.name, new_state.name)

        # 상태 진입 시 초기화
        if new_state == RobotState.INTERVENING:
            self._ctx.intervention_phase = 0
            self._ctx.last_tts_time = 0.0
        elif new_state == RobotState.SCANNING:
            self._ctx.scan_start_time = time.time()
        elif new_state == RobotState.DETECTED:
            self._ctx.alert_sent = False
            self._ctx.call_made = False

    def force_state(self, state: RobotState) -> None:
        """상태를 강제 전이한다 (외부 제어용)."""
        logger.warning("강제 상태 전이: %s → %s", self._ctx.state.name, state.name)
        self._transition(state)

    def emergency_stop(self) -> dict:
        """비상 정지 상태로 전이하고 정지 명령을 반환한다."""
        self._transition(RobotState.EMERGENCY_STOP)
        return self._make_action(
            motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
            tts_message="비상 정지가 활성화되었습니다.",
        )

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        risk_level: str = "",
        risk_score: float = 0.0,
        track_id: Optional[int] = None,
        waypoint_arrived: bool = False,
        patrol_steering: int = 0,
        patrol_speed: int = 0,
        approach_steering: int = 0,
        approach_speed: int = 0,
        approach_reached: bool = False,
    ) -> dict:
        """
        매 프레임 호출되어 현재 입력에 따라 상태를 전이하고 행동을 결정한다.

        Parameters
        ----------
        risk_level : str
            최고 위험 수준 이름 ("SAFE", "OBSERVE", "WARNING", "DANGER", "CRITICAL").
        risk_score : float
            최고 위험 점수 (0.0–1.0).
        track_id : int or None
            최고 위험 대상의 추적 ID.
        waypoint_arrived : bool
            순찰 웨이포인트 도착 여부.
        patrol_steering : int
            순찰 조향값 (-7~+7).
        patrol_speed : int
            순찰 속도 (0–255).
        approach_steering : int
            접근 조향값 (-7~+7).
        approach_speed : int
            접근 속도 (0–255).
        approach_reached : bool
            대상 접근 완료 여부.

        Returns
        -------
        dict
            ``{"motor_command": MotorCommand, "tts_message": str|None,
              "send_alert": bool, "make_call": bool,
              "target_track_id": int|None}``
        """
        now = time.time()
        state = self._ctx.state

        # ---- 비상 정지: 외부에서만 해제 가능 ----
        if state == RobotState.EMERGENCY_STOP:
            return self._make_action(
                motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
            )

        # ---- 위험 감지 시 상태 전이 (어느 상태에서든) ----
        if risk_score >= self._emergency_threshold and track_id is not None:
            if state not in (RobotState.ALERTING, RobotState.EMERGENCY_STOP):
                self._ctx.target_track_id = track_id
                self._ctx.target_risk_level = risk_level
                self._ctx.target_risk_score = risk_score
                self._transition(RobotState.ALERTING)

        elif risk_score >= self._approach_threshold and track_id is not None:
            if state in (RobotState.PATROL, RobotState.SCANNING, RobotState.IDLE):
                self._ctx.target_track_id = track_id
                self._ctx.target_risk_level = risk_level
                self._ctx.target_risk_score = risk_score
                self._transition(RobotState.DETECTED)

        # ---- 대상이 사라진 경우 순찰 복귀 ----
        if state in (RobotState.DETECTED, RobotState.APPROACHING, RobotState.INTERVENING):
            if track_id is None or risk_score < 0.1:
                logger.info("대상 소실 또는 위험 해소 — 순찰 복귀")
                self._ctx.target_track_id = None
                self._transition(RobotState.PATROL)

        # ---- 상태별 행동 ----
        state = self._ctx.state  # 전이 후 재확인

        if state == RobotState.IDLE:
            return self._handle_idle()

        elif state == RobotState.PATROL:
            return self._handle_patrol(
                waypoint_arrived, patrol_steering, patrol_speed,
            )

        elif state == RobotState.SCANNING:
            return self._handle_scanning(now)

        elif state == RobotState.DETECTED:
            return self._handle_detected(now)

        elif state == RobotState.APPROACHING:
            return self._handle_approaching(
                approach_steering, approach_speed, approach_reached,
            )

        elif state == RobotState.INTERVENING:
            return self._handle_intervening(now)

        elif state == RobotState.ALERTING:
            return self._handle_alerting(now)

        # fallback
        return self._make_action()

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_idle(self) -> dict:
        """IDLE: 정지 상태."""
        return self._make_action(
            motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
        )

    def _handle_patrol(
        self,
        waypoint_arrived: bool,
        steering: int,
        speed: int,
    ) -> dict:
        """PATROL: 순찰 이동."""
        if waypoint_arrived:
            self._transition(RobotState.SCANNING)
            return self._make_action(
                motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
            )

        return self._make_action(
            motor_command=MotorCommand(
                steering=steering,
                left_speed=speed,
                right_speed=speed,
            ),
        )

    def _handle_scanning(self, now: float) -> dict:
        """SCANNING: 웨이포인트 도착 후 주변 스캔."""
        elapsed = now - self._ctx.scan_start_time
        if elapsed >= self._ctx.scan_duration:
            self._transition(RobotState.PATROL)
            return self._make_action()

        # 제자리 정지하며 스캔
        return self._make_action(
            motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
        )

    def _handle_detected(self, now: float) -> dict:
        """DETECTED: 위험 대상 감지 — 잠시 관찰 후 접근."""
        elapsed = self.state_duration
        if elapsed >= self._detect_delay:
            self._transition(RobotState.APPROACHING)

        # 정지하며 관찰
        return self._make_action(
            motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
            target_track_id=self._ctx.target_track_id,
        )

    def _handle_approaching(
        self,
        steering: int,
        speed: int,
        reached: bool,
    ) -> dict:
        """APPROACHING: 대상에게 접근."""
        if reached:
            self._transition(RobotState.INTERVENING)
            return self._make_action(
                motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
                target_track_id=self._ctx.target_track_id,
            )

        return self._make_action(
            motor_command=MotorCommand(
                steering=steering,
                left_speed=speed,
                right_speed=speed,
            ),
            target_track_id=self._ctx.target_track_id,
        )

    def _handle_intervening(self, now: float) -> dict:
        """INTERVENING: 대상에게 음성 개입."""
        tts_message = None
        phase = self._ctx.intervention_phase

        # TTS 간격 확인
        if now - self._ctx.last_tts_time >= self._tts_interval:
            if phase == 0:
                messages = _TTS_MESSAGES["greeting"]
            elif phase == 1:
                messages = _TTS_MESSAGES["concern"]
            elif phase == 2:
                messages = _TTS_MESSAGES["support"]
            else:
                messages = _TTS_MESSAGES["waiting"]

            # 단계 내 순환 메시지 선택
            msg_index = min(
                int((now - self._ctx.state_enter_time) / self._tts_interval) % len(messages),
                len(messages) - 1,
            )
            tts_message = messages[msg_index]
            self._ctx.last_tts_time = now

            # 단계 진행 (각 단계에서 메시지를 모두 재생하면 다음 단계)
            if phase < 3:
                self._ctx.intervention_phase = phase + 1

        # 장시간 개입 시 알림 전송
        send_alert = False
        if self.state_duration > 30.0 and not self._ctx.alert_sent:
            send_alert = True
            self._ctx.alert_sent = True
            self._transition(RobotState.ALERTING)

        return self._make_action(
            motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
            tts_message=tts_message,
            send_alert=send_alert,
            target_track_id=self._ctx.target_track_id,
        )

    def _handle_alerting(self, now: float) -> dict:
        """ALERTING: 관제센터 알림 및 긴급 전화."""
        tts_message = None
        send_alert = False
        make_call = False

        # 첫 진입 시 알림 + 전화
        if not self._ctx.alert_sent:
            send_alert = True
            self._ctx.alert_sent = True
            logger.warning(
                "관제센터 알림 전송: track_id=%s, risk=%s (%.2f)",
                self._ctx.target_track_id,
                self._ctx.target_risk_level,
                self._ctx.target_risk_score,
            )

        if not self._ctx.call_made:
            make_call = True
            self._ctx.call_made = True
            logger.warning("긴급 전화 발신: 119 / 1393")

        # 긴급 TTS
        if now - self._ctx.last_tts_time >= self._tts_interval:
            messages = _TTS_MESSAGES["emergency"]
            msg_index = int(
                (now - self._ctx.state_enter_time) / self._tts_interval,
            ) % len(messages)
            tts_message = messages[msg_index]
            self._ctx.last_tts_time = now

        return self._make_action(
            motor_command=MotorCommand(steering=0, left_speed=0, right_speed=0),
            tts_message=tts_message,
            send_alert=send_alert,
            make_call=make_call,
            target_track_id=self._ctx.target_track_id,
        )

    # ------------------------------------------------------------------
    # Action builder
    # ------------------------------------------------------------------

    @staticmethod
    def _make_action(
        motor_command: Optional[MotorCommand] = None,
        tts_message: Optional[str] = None,
        send_alert: bool = False,
        make_call: bool = False,
        target_track_id: Optional[int] = None,
    ) -> dict:
        """행동 딕셔너리를 생성한다."""
        if motor_command is None:
            motor_command = MotorCommand(steering=0, left_speed=0, right_speed=0)

        return {
            "motor_command": motor_command,
            "tts_message": tts_message,
            "send_alert": send_alert,
            "make_call": make_call,
            "target_track_id": target_track_id,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def start_patrol(self) -> None:
        """순찰을 시작한다 (IDLE → PATROL)."""
        if self._ctx.state == RobotState.IDLE:
            self._transition(RobotState.PATROL)
        else:
            logger.warning(
                "IDLE 상태가 아닙니다 (현재: %s). force_state() 사용을 고려하세요.",
                self._ctx.state.name,
            )

    def reset(self) -> None:
        """상태 머신을 IDLE로 초기화한다."""
        self._ctx = _StateContext()
        logger.info("상태 머신 초기화됨 (IDLE)")

    def get_status(self) -> dict:
        """현재 상태 요약 딕셔너리를 반환한다."""
        return {
            "state": self._ctx.state.name,
            "previous_state": self._ctx.previous_state.name,
            "state_duration_s": round(self.state_duration, 1),
            "target_track_id": self._ctx.target_track_id,
            "target_risk_level": self._ctx.target_risk_level,
            "target_risk_score": round(self._ctx.target_risk_score, 3),
            "intervention_phase": self._ctx.intervention_phase,
            "alert_sent": self._ctx.alert_sent,
            "call_made": self._ctx.call_made,
        }
