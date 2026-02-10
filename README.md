<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Pose-FF6F00?style=for-the-badge&logo=yolo&logoColor=white"/>
  <img src="https://img.shields.io/badge/Intel-RealSense%20D435-0071C5?style=for-the-badge&logo=intel&logoColor=white"/>
  <img src="https://img.shields.io/badge/NVIDIA-Jetson%20Orin%20Nano-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
</p>

<h1 align="center">Bridge Guardian</h1>
<h3 align="center">다리 위의 친구 &mdash; AI 기반 자살 예방 실시간 감시 시스템</h3>

<p align="center">
  <b>YOLOv8-pose 키포인트 분석</b>과 <b>Intel RealSense 깊이 카메라</b>를 활용하여<br/>
  다리 난간 부근의 위험 행동을 실시간으로 감지하고 즉각 경고를 발송하는 시스템
</p>

---

## Overview

매년 한국에서 자살로 사망하는 인원 중 **투신**은 주요 수단 중 하나이며, 다리에서의 자살 시도는 사전 징후가 포착되면 충분히 개입 가능합니다.

**Bridge Guardian**은 다리 위에 설치된 카메라를 통해 아래 세 가지 위험 행동을 실시간으로 감지합니다:

| 감지 유형 | 방법 | 위험 가중치 |
|-----------|------|------------|
| **난간 타넘기 (Climbing)** | 무릎-엉덩이 역전 + 다리 각도 + 시계열 패턴 | 35% |
| **난간 근접 + 장시간 정지** | ROI 영역 판정 + 깊이 검증 + 체류 시간 | 30% + 35% |
| **바깥 방향 응시** | 귀 가시성 + 어깨-코 오프셋 기반 방향 추정 | (복합 판정) |

위험도가 임계값을 초과하면 **콘솔 경고**, **로그 기록**, **SMS/전화 알림**(Twilio)을 자동 발송합니다.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       main.py                           │
│              통합 파이프라인 + OpenCV 시각화                │
├──────────┬──────────┬───────────────────┬───────────────┤
│ Camera   │ Detector │    Risk Engine    │    Alert      │
│ Handler  │ + Track  │                   │    Service    │
├──────────┼──────────┼───────────────────┼───────────────┤
│RealSense │ YOLOv8n  │ Pose Analyzer     │ Console/Log   │
│D435 or   │ -pose +  │ Stationary Det.   │ Twilio SMS    │
│Video File│ BoT-SORT │ Proximity Det.    │ Twilio Call    │
└──────────┴──────────┴───────────────────┴───────────────┘
```

```
bridge-guardian/
├── config/
│   └── settings.yaml              # 전체 설정 (카메라, 감지, 위험도, 알림)
├── perception/
│   ├── realsense_handler.py       # RealSense D435 / 비디오 파일 입력
│   ├── person_detector.py         # YOLOv8n-pose 감지 + BoT-SORT 추적
│   └── person_tracker.py          # 다중 인물 시계열 상태 관리
├── risk_assessment/
│   ├── pose_analyzer.py           # Climbing / Lean / Orientation 감지
│   ├── stationary_detector.py     # 정지 시간 기반 위험도 산출
│   ├── proximity_detector.py      # ROI + Depth 기반 난간 근접 감지
│   └── risk_engine.py             # 가중합산 + 시간대 보정 + Override 규칙
├── communication/
│   └── alert_service.py           # 콘솔/로그/Twilio 경고 발송
├── utils/
│   └── logger.py                  # 컬러 로깅 유틸리티
├── main.py                        # 메인 파이프라인 (진입점)
└── requirements.txt
```

---

## How It Works

### 1. Perception (인식)

```
카메라 프레임 → YOLOv8n-pose → 사람 검출 + 17개 COCO 키포인트 + BoT-SORT ID
```

- **RealSense D435**: BGR 컬러 + 정렬된 깊이맵 동시 취득
- **비디오 파일**: 녹화 영상으로 오프라인 테스트 가능
- **PersonTracker**: track_id별 위치/키포인트 이력을 deque로 관리

### 2. Risk Assessment (위험도 평가)

세 개의 독립적 분석 모듈이 각각 **0.0 ~ 1.0** 점수를 산출합니다:

**Pose Analyzer** — 자세 위험도
- 무릎이 엉덩이 위로 올라감 → Climbing 감지
- 상체 기울기 > 25도 → Lean 감지
- 양쪽 귀 모두 보임 → 뒷모습(바깥 향함) 판별

**Stationary Detector** — 정지 시간
- 최근 위치 이력에서 이동량 < 25px인 구간의 지속 시간 측정
- `score = min(duration / 30초, 1.0)`

**Proximity Detector** — 난간 근접도
- 설정된 ROI 다각형 내에 발 위치가 포함되는지 판정
- RealSense 깊이 데이터로 이중 검증 (0.5m ~ 3.0m)

### 3. Risk Engine (종합 판정)

```
최종 점수 = (정지 x 0.30 + 근접 x 0.35 + 자세 x 0.35) x 시간대 보정
```

| 위험 등급 | 점수 범위 | 색상 | 조치 |
|----------|----------|------|------|
| SAFE | 0.0 ~ 0.3 | 초록 | - |
| OBSERVE | 0.3 ~ 0.5 | 초록 | 모니터링 |
| WARNING | 0.5 ~ 0.7 | 노랑 | SMS 발송 |
| DANGER | 0.7 ~ 0.85 | 빨강 | SMS + 긴급 |
| CRITICAL | 0.85 ~ 1.0 | 마젠타 | SMS + 전화 |

**Override 규칙:**
- Climbing 감지 → 즉시 **CRITICAL** (score >= 0.9)
- 60초 이상 정지 + 바깥 향함 + ROI 내 → 최소 **DANGER** (score >= 0.75)

**시간대 보정:**
- 새벽 04~06시: x1.5
- 야간 22~04시: x1.3

---

## Quick Start

### Requirements

- Python 3.10+
- CUDA GPU (권장) 또는 CPU
- Intel RealSense D435 (선택 — 비디오 파일로도 동작)

### Installation

```bash
git clone https://github.com/woongje-cho/Suicide_Prevention_Project.git
cd Suicide_Prevention_Project

pip install -r requirements.txt

# RealSense 카메라 사용 시 (선택)
pip install pyrealsense2
```

### Run

```bash
# RealSense 카메라 실시간
python main.py

# 비디오 파일 테스트
python main.py --source test_video.mp4

# 화면 없이 + 결과 영상 저장
python main.py --source test.mp4 --no-display --save-video output.mp4

# 커스텀 설정 파일
python main.py --config my_settings.yaml
```

### CLI Options

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--config` | 설정 파일 경로 | `config/settings.yaml` |
| `--source` | 카메라 소스 오버라이드 | 설정 파일 값 |
| `--no-display` | 시각화 창 비활성화 | `False` |
| `--save-video` | 결과 영상 저장 경로 | 없음 |

---

## Configuration

모든 설정은 `config/settings.yaml` 하나로 관리됩니다.

<details>
<summary><b>주요 설정 항목 (클릭하여 펼치기)</b></summary>

```yaml
# 카메라
camera:
  source: "realsense"     # 또는 "video.mp4"
  width: 640
  height: 480
  fps: 30

# YOLO 모델
detection:
  model_path: "yolov8n-pose.pt"
  confidence: 0.5
  device: 0               # GPU index 또는 "cpu"

# 난간 ROI (정규화 좌표 0~1)
risk_assessment:
  railing_roi_zones:
    - name: "left_railing"
      polygon: [[0.0, 0.3], [0.15, 0.3], [0.15, 0.9], [0.0, 0.9]]

  # 위험도 가중치
  weights:
    stationary_time: 0.30
    railing_proximity: 0.35
    dangerous_pose: 0.35

# 알림
alerts:
  enabled: true
  console: true
  log_file: "alerts.log"
  min_alert_level: "WARNING"
  twilio:
    enabled: false         # Twilio SMS/전화 사용 시 true
```

</details>

---

## Hardware Setup

| 부품 | 사양 | 용도 |
|------|------|------|
| **Jetson Orin Nano** | 8GB RAM, 40 TOPS | 엣지 추론 |
| **Intel RealSense D435** | RGB + Depth, 30fps | 깊이 카메라 |
| **2D LiDAR** (2차) | RPLidar 등 | 근접 센서 퓨전 |

### Jetson 배포 (TensorRT 가속)

```bash
# YOLOv8 모델을 TensorRT 엔진으로 변환
yolo export model=yolov8n-pose.pt format=engine device=0

# settings.yaml에서 엔진 경로 지정
# model_path: "yolov8n-pose.engine"
```

---

## Visualization

실행 시 OpenCV 창에 실시간 오버레이가 표시됩니다:

- **ROI 영역**: 초록색 반투명 다각형
- **바운딩 박스**: 위험 등급별 색상
- **스켈레톤**: COCO 17 키포인트 연결선
- **위험 라벨**: `ID:1 WARNING 0.65`
- **특수 플래그**: `[CLIMBING]`, `[OUTWARD]`
- **FPS**: 좌측 상단

`q` 또는 `ESC`로 종료합니다.

---

## Roadmap

- [x] YOLOv8-pose 기반 사람 감지 + 추적
- [x] Climbing / Lean / Orientation 자세 분석
- [x] ROI + Depth 난간 근접 감지
- [x] 가중합산 위험도 엔진 + 시간대 보정
- [x] 콘솔/로그/Twilio 알림 시스템
- [x] 비디오 파일 입력 모드
- [ ] 2D LiDAR 센서 퓨전
- [ ] 순찰 경로 관리 (Navigation)
- [ ] 모터 제어 (Arduino 시리얼)
- [ ] TTS 음성 개입 시스템
- [ ] ROS 2 통합

---

## License

This project is for educational purposes (창의제품설계 수업 프로젝트).