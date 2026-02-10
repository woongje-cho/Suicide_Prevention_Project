<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Pose-FF6F00?style=for-the-badge&logo=yolo&logoColor=white"/>
  <img src="https://img.shields.io/badge/Intel-RealSense%20D435-0071C5?style=for-the-badge&logo=intel&logoColor=white"/>
  <img src="https://img.shields.io/badge/RPLidar-A2-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NVIDIA-Jetson%20Orin%20Nano-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/Twilio-SMS%2FCall-F22F46?style=for-the-badge&logo=twilio&logoColor=white"/>
</p>

<h1 align="center">Bridge Guardian</h1>
<h3 align="center">다리 위의 친구 &mdash; AI 자살 예방 자율 순찰 로봇</h3>

<p align="center">
  <b>YOLOv8-pose</b> + <b>2D LiDAR</b> + <b>Intel RealSense D435</b> 센서 퓨전으로<br/>
  다리 난간의 위험 행동을 감지하고, 자율 접근 → 음성 개입 → 긴급 전화까지 수행하는 로봇 시스템
</p>

---

## Overview

매년 한국에서 자살로 사망하는 인원 중 **투신**은 주요 수단 중 하나이며, 다리에서의 자살 시도는 사전 징후가 포착되면 충분히 개입 가능합니다.

**Bridge Guardian**은 어린이 전동차를 개조한 자율 순찰 로봇으로, 다리 위를 순찰하며 **감지 → 접근 → 음성 개입 → 긴급 알림** 전체 파이프라인을 자동으로 수행합니다.

### 핵심 기능

| 기능 | 설명 | 모듈 |
|------|------|------|
| **카메라 감지** | YOLOv8-pose 17 키포인트 + BoT-SORT 추적 | `perception/` |
| **2D LiDAR** | RPLidar A2 360° 스캔 + 사람 클러스터 감지 | `perception/lidar_handler` |
| **센서 퓨전** | 카메라-LiDAR 각도 매칭 + 깊이 카메라 거리 통합 | `perception/sensor_fusion` |
| **위험도 평가** | 자세(Climbing/Lean/Orientation) + 정지시간 + 근접도 | `risk_assessment/` |
| **자율 순찰** | GPS 기반 웨이포인트 순찰 + 위험 감지 시 자동 접근 | `navigation/` |
| **행동 상태 머신** | IDLE → PATROL → DETECT → APPROACH → INTERVENE → ALERT | `behavior/` |
| **TTS 음성 개입** | 한국어 음성 안내 (pyttsx3/gTTS/espeak) | `communication/tts_service` |
| **긴급 전화/SMS** | Twilio로 01088405390 자동 전화 + SMS | `communication/alert_service` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              main.py                                    │
│                  통합 파이프라인 + 상태 머신 + OpenCV 시각화               │
├──────────────────┬──────────────────┬──────────────────┬────────────────┤
│   Perception     │  Risk Assessment │    Navigation    │ Communication  │
├──────────────────┼──────────────────┼──────────────────┼────────────────┤
│ RealSense D435   │ Pose Analyzer    │ GPS Handler      │ TTS Service    │
│ YOLOv8-pose      │ Stationary Det.  │ Motor Controller │ Alert Service  │
│ Person Tracker   │ Proximity Det.   │ Path Planner     │ (Twilio SMS    │
│ RPLidar A2       │ Risk Engine      │                  │  + 전화)       │
│ Sensor Fusion    │ (+ LiDAR 검증)   │                  │                │
├──────────────────┴──────────────────┴──────────────────┴────────────────┤
│                      Behavior State Machine                             │
│        IDLE → PATROL → SCANNING → DETECTED → APPROACHING               │
│                    → INTERVENING → ALERTING → EMERGENCY_STOP            │
└─────────────────────────────────────────────────────────────────────────┘
```

```
bridge-guardian/
├── config/
│   └── settings.yaml              # 전체 설정 (7개 모듈 통합)
├── perception/
│   ├── realsense_handler.py       # RealSense D435 / 비디오 파일 입력
│   ├── person_detector.py         # YOLOv8n-pose + BoT-SORT 추적
│   ├── person_tracker.py          # 다중 인물 시계열 상태 관리
│   ├── lidar_handler.py           # RPLidar A2 드라이버 + 클러스터 감지
│   └── sensor_fusion.py           # 카메라-LiDAR-깊이 센서 퓨전
├── risk_assessment/
│   ├── pose_analyzer.py           # Climbing / Lean / Orientation 감지
│   ├── stationary_detector.py     # 정지 시간 기반 위험도
│   ├── proximity_detector.py      # ROI + Depth + LiDAR 근접 감지
│   └── risk_engine.py             # 가중합산 + 시간대 보정 + Override
├── navigation/
│   ├── gps_handler.py             # GPS NMEA 파싱 (pynmea2)
│   ├── motor_controller.py        # Arduino 시리얼 모터 제어
│   └── path_planner.py            # 순찰 웨이포인트 + 접근 경로
├── behavior/
│   └── state_machine.py           # 로봇 행동 상태 머신 (8 states)
├── communication/
│   ├── alert_service.py           # 콘솔/로그/Twilio SMS+전화
│   └── tts_service.py             # 한국어 TTS 음성 안내
├── utils/
│   └── logger.py                  # 컬러 로깅 유틸리티
├── main.py                        # 메인 파이프라인 (진입점)
└── requirements.txt               # 전체 의존성
```

---

## How It Works

### 1. Perception (인식)

```
카메라 프레임 → YOLOv8-pose → 사람 검출 + 17개 COCO 키포인트 + BoT-SORT ID
                                            ↕ (센서 퓨전)
RPLidar A2 → 360° 스캔 → 클러스터 감지 → 각도 매칭 → 거리 검증
```

- **RealSense D435**: BGR 컬러 + 정렬된 깊이맵 동시 취득
- **RPLidar A2**: 360° 스캔으로 난간 부근 사람 거리 측정 (0.15~12m)
- **센서 퓨전**: 카메라 픽셀 x → 각도 변환 후 LiDAR 클러스터와 매칭, 가중 평균 거리 산출
- **비디오 파일**: 녹화 영상으로 오프라인 테스트 가능

### 2. Risk Assessment (위험도 평가)

```
최종 점수 = (정지 x 0.30 + 근접 x 0.35 + 자세 x 0.35) x 시간대 보정
```

| 위험 등급 | 점수 범위 | 색상 | 조치 |
|----------|----------|------|------|
| SAFE | 0.0 ~ 0.3 | 초록 | - |
| OBSERVE | 0.3 ~ 0.5 | 초록 | 모니터링 |
| WARNING | 0.5 ~ 0.7 | 노랑 | TTS 인사 + SMS |
| DANGER | 0.7 ~ 0.85 | 빨강 | TTS 지원 안내 + SMS |
| CRITICAL | 0.85 ~ 1.0 | 마젠타 | TTS 긴급 + SMS + **전화** |

**Override 규칙:**
- Climbing 감지 → 즉시 CRITICAL
- 60초 정지 + 바깥 향함 + ROI 내 → 최소 DANGER
- LiDAR 근접 확인 시 proximity_score 부스트

### 3. Behavior (행동 상태 머신)

```
IDLE → PATROL → SCANNING → DETECTED → APPROACHING → INTERVENING → ALERTING
  ↑                                                                    │
  └──────────── EMERGENCY_STOP ←───────────────────────────────────────┘
```

| 상태 | 행동 |
|------|------|
| **PATROL** | 웨이포인트 순찰 이동 |
| **SCANNING** | 정지 후 주변 스캔 (5초) |
| **DETECTED** | 위험 대상 감지, 관찰 (3초) |
| **APPROACHING** | 대상에게 접근 이동 |
| **INTERVENING** | TTS 음성 개입 ("괜찮으세요?", "1393 상담전화") |
| **ALERTING** | Twilio 긴급 전화 (01088405390) + SMS |

### 4. Communication (소통)

**TTS 음성 안내** (한국어):
- "안녕하세요, 괜찮으신가요?"
- "혼자가 아닙니다. 자살예방상담전화 1393으로 연락하실 수 있습니다."
- "긴급 상황입니다. 119에 연락하고 있습니다."

**긴급 전화/SMS**:
- 01088405390으로 자동 전화 발신 (Twilio, 한국어 TwiML)
- GPS 좌표 포함 SMS 발송

---

## Quick Start

### Requirements

- Python 3.10+
- CUDA GPU (권장) 또는 CPU
- Intel RealSense D435 (선택)
- RPLidar A2 (선택)
- GPS 모듈 (선택)
- Arduino + 모터 드라이버 (선택)

### Installation

```bash
git clone https://github.com/woongje-cho/Suicide_Prevention_Project.git
cd Suicide_Prevention_Project

pip install -r requirements.txt
```

### Run

```bash
# 전체 시스템 (모든 센서 연결 시)
python main.py

# 카메라만 (LiDAR/GPS/모터 없이)
python main.py --no-lidar --no-gps --no-motor

# 비디오 파일 테스트
python main.py --source test_video.mp4 --no-lidar --no-gps --no-motor

# 화면 없이 + 결과 영상 저장
python main.py --source test.mp4 --no-display --save-video output.mp4

# TTS 비활성화
python main.py --no-tts
```

### CLI Options

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--config` | 설정 파일 경로 | `config/settings.yaml` |
| `--source` | 카메라 소스 오버라이드 | 설정 파일 값 |
| `--no-display` | 시각화 창 비활성화 | `False` |
| `--save-video` | 결과 영상 저장 경로 | 없음 |
| `--no-lidar` | LiDAR 비활성화 | `False` |
| `--no-gps` | GPS 비활성화 | `False` |
| `--no-motor` | 모터 제어 비활성화 | `False` |
| `--no-tts` | TTS 음성 비활성화 | `False` |

### Keyboard Controls

| 키 | 기능 |
|----|------|
| `q` / `ESC` | 종료 |
| `e` | 비상 정지 |
| `p` | 순찰 시작 |

---

## Hardware Setup

| 부품 | 사양 | 용도 | 연결 |
|------|------|------|------|
| **Jetson Orin Nano** | 8GB RAM, 40 TOPS | 엣지 추론 | - |
| **Intel RealSense D435** | RGB+Depth, 30fps | 깊이 카메라 | USB 3.0 |
| **RPLidar A2** | 360°, 0.15-12m, 10Hz | 2D 거리 측정 | USB → /dev/ttyUSB0 (115200) |
| **GPS Module** | NMEA $GPGGA/$GPRMC | 위치 추적 | Serial → /dev/ttyACM1 (9600) |
| **Arduino** | Motor Driver | 모터 제어 | Serial → /dev/ttyACM0 (9600) |
| **어린이 전동차** | 개조 플랫폼 | 로봇 본체 | - |
| **스피커** | 3.5mm / USB | TTS 출력 | Audio out |

### 모터 시리얼 프로토콜 (H-Mobility 호환)

```
s{steering}l{left_speed}r{right_speed}\n

steering : -7 (좌) ~ 0 (직진) ~ +7 (우)
speed    : 0 ~ 255 (PWM)

예: s0l100r100\n  → 직진 100 속도
    s3l80r80\n   → 우회전 80 속도
```

---

## Configuration

모든 설정은 `config/settings.yaml` 하나로 관리됩니다 (7개 모듈 통합).

<details>
<summary><b>주요 설정 항목 (클릭하여 펼치기)</b></summary>

```yaml
camera:
  source: "realsense"     # 또는 "video.mp4"
  width: 640
  height: 480

detection:
  model_path: "yolov8n-pose.pt"
  confidence: 0.5

lidar:
  enabled: true
  port: "/dev/ttyUSB0"
  baudrate: 115200

gps:
  enabled: true
  port: "/dev/ttyACM1"

motor:
  enabled: true
  port: "/dev/ttyACM0"
  patrol_speed: 80

behavior:
  approach_risk_threshold: 2      # WARNING
  intervention_duration_s: 30.0

tts:
  enabled: true
  engine: "pyttsx3"               # pyttsx3, gtts, espeak

alerts:
  twilio:
    enabled: false
    # 전화번호 01088405390은 코드에 하드코딩됨
```

</details>

---

## Visualization

실행 시 OpenCV 창에 실시간 오버레이가 표시됩니다:

- **ROI 영역**: 초록색 반투명 다각형
- **바운딩 박스**: 위험 등급별 색상 (초록→노랑→빨강→마젠타)
- **스켈레톤**: COCO 17 키포인트 연결선
- **위험 라벨**: `ID:1 WARNING 0.65`
- **특수 플래그**: `[CLIMBING]`, `[OUTWARD]`
- **로봇 상태**: 우측 상단 (State / Duration / Target)
- **GPS 좌표**: 좌측 하단
- **LiDAR 미니맵**: 우측 하단 (360° 탑뷰)
- **FPS**: 좌측 상단

---

## Graceful Degradation

모든 센서 모듈은 하드웨어 미연결 시에도 **크래시 없이** 동작합니다:

| 모듈 | 하드웨어 없을 때 |
|------|------------------|
| RealSense | 비디오 파일로 대체, 깊이 데이터 없이 동작 |
| RPLidar | 더미 모드 (None 반환), 카메라만으로 판단 |
| GPS | 위치 정보 없이 동작, 알림에 좌표 미포함 |
| Motor | 명령 로그만 출력, 물리적 이동 없음 |
| TTS | 텍스트 로그만 출력, 음성 없음 |
| Twilio | SMS/전화 미발송, 콘솔/로그 알림만 |

---

## Roadmap

- [x] YOLOv8-pose 기반 사람 감지 + BoT-SORT 추적
- [x] Climbing / Lean / Orientation 자세 분석
- [x] ROI + Depth 난간 근접 감지
- [x] 가중합산 위험도 엔진 + 시간대 보정
- [x] 콘솔/로그/Twilio 알림 시스템
- [x] 비디오 파일 입력 모드
- [x] 2D LiDAR 센서 퓨전 (RPLidar A2)
- [x] 카메라-LiDAR-깊이 센서 퓨전
- [x] GPS 위치 추적
- [x] 모터 제어 (H-Mobility 시리얼 프로토콜)
- [x] 순찰 경로 관리 (웨이포인트 기반)
- [x] 행동 상태 머신 (8 states)
- [x] TTS 한국어 음성 개입
- [x] 긴급 전화 자동 발신 (01088405390)
- [ ] ROS 2 통합
- [ ] 야간 IR 카메라 지원
- [ ] 클라우드 대시보드

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [RPLidar Python](https://github.com/SkoltechRobotics/rplidar)
- [Twilio Voice API](https://www.twilio.com/docs/voice)
- H-Mobility Autonomous Vehicle Project (성균관대학교 자동화연구실)

---

## License

This project is for educational purposes (창의제품설계 수업 프로젝트).