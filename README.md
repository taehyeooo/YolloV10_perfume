# Facial Expression Recognition for Perfume Recommendation
### Comparative Performance Analysis of YOLOv8 and YOLOv10

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv10-00BFFF)
![License](https://img.shields.io/badge/License-AGPL--3.0-green)
![Dataset](https://img.shields.io/badge/Dataset-Roboflow%20CC%20BY%204.0-orange)

---

## Overview

본 프로젝트는 **향수 추천 시스템**의 핵심 모듈인 실시간 표정 인식 기능의 정확도를 향상시키기 위해 진행되었습니다.

기존 시스템에서 사용하던 **YOLOv8**은 빠른 추론 속도라는 장점이 있었지만, 표정 인식 정확도에 한계가 있었습니다. 이를 극복하기 위해 최신 모델인 **YOLOv10**을 도입하고, 두 모델 간의 성능을 정량적으로 비교·분석하였습니다.

**인식 대상 표정 클래스 (5종)**

| 클래스 | 설명 |
|--------|------|
| `anger` | 분노 |
| `fear` | 공포 |
| `happy` | 기쁨 |
| `neutral` | 무표정 |
| `sad` | 슬픔 |

---

## Research Background

사람의 감정 상태에 맞는 향수를 추천하기 위해서는 얼굴 표정을 정확하게 인식하는 것이 핵심입니다. 기존에 사용하던 YOLOv8은 실시간 처리에 적합한 속도를 제공했지만, 세부 표정 클래스를 구분하는 정밀도가 아쉬웠습니다.

YOLOv10은 NMS(Non-Maximum Suppression)를 제거한 **One-to-One 헤드(one2one head)** 구조를 도입하여 추론 속도와 정확도를 동시에 개선한 모델입니다. 본 연구에서는 다양한 하이퍼파라미터(Epoch, Batch Size) 조합으로 두 모델을 학습시켜 최적의 설정을 도출하는 것을 목표로 하였습니다.

---

## Experimental Results

### 모델별 성능 비교 (Accuracy)

| 모델 | Epoch | Batch Size | Accuracy |
|------|-------|------------|----------|
| YOLOv8 | 100 | 16 | **0.612** |
| YOLOv8 | 100 | 32 | 0.542 |
| YOLOv10 | 100 | 16 | 0.641 |
| YOLOv10 | 200 | 16 | **0.673** ✅ |
| YOLOv10 | 200 | 32 | 0.618 |

> **Best Model:** YOLOv10 · Epoch 200 · Batch Size 16 → Accuracy **0.673**

### Confusion Matrix 분석 (YOLOv10 Best Model)

| 클래스 | 정확한 예측 건수 | 주요 오분류 |
|--------|----------------|------------|
| `happy` | **130건** | - |
| `anger` | **86건** | - |
| `neutral` | 중간 수준 | `sad`로 혼동 |
| `sad` | 중간 수준 | `neutral`로 혼동 |
| `fear` | 낮음 | **16건 오답 발생** |

### Precision-Confidence Curve 분석

- **`happy` 클래스**: 전 신뢰도 구간에서 높은 정밀도 유지
- **`neutral` / `sad` 클래스**: 신뢰도가 낮은 구간에서 정밀도 상대적으로 저하
- **`fear` 클래스**: 다른 표정과의 특징 유사성으로 인해 오분류 발생 빈도 높음

---

## Conclusion

본 실험을 통해 다음의 결론을 도출하였습니다.

1. **YOLOv10이 YOLOv8 대비 성능 우위** — 동일한 Batch Size 16 조건에서 YOLOv10(0.673)이 YOLOv8(0.612)보다 약 **6.1%p 높은 정확도**를 달성하였습니다.
2. **배치 크기는 작을수록 유리** — 두 모델 모두 Batch Size 16일 때 32보다 높은 정확도를 기록하였으며, 이는 소규모 배치 학습이 세부 패턴 학습과 과적합 방지에 효과적임을 보여줍니다.
3. **에포크 수 증가의 효과** — YOLOv10의 경우 Epoch 100 대비 200에서 성능이 유의미하게 향상되었습니다.
4. **하이퍼파라미터 튜닝의 중요성** — 실시간 표정 인식 시스템 설계 시 모델 선택뿐만 아니라 최적의 하이퍼파라미터 설정이 성능에 결정적 영향을 미칩니다.

---

## Tech Stack

| 분류 | 기술 |
|------|------|
| Language | Python 3.8+ |
| Deep Learning | PyTorch 2.0+ |
| Object Detection | Ultralytics YOLOv10 |
| Model Hub | HuggingFace Hub (`huggingface_hub`) |
| Dataset | Roboflow (CC BY 4.0) |
| Pretrained Weights | `yolov10n.pt` |

---

## Project Structure

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/9d4015b6-841f-4a9a-babd-e6cda08b4c23" />


---

## Installation

```bash
# 1. 저장소 클론
git clone https://github.com/taehyeooo/YolloV10_perfume.git
cd YolloV10_perfume

# 2. 의존성 설치
pip install ultralytics huggingface_hub torch torchvision
```

---

## Usage

### Training (학습)

```python
from model import YOLOv10

# 사전 학습 가중치로 모델 로드
model = YOLOv10("yolov10n.pt")

# 학습 실행 (Best 설정: epoch=200, batch=16)
model.train(
    data="dataset/data.yaml",
    epochs=200,
    batch=16,
    imgsz=640,
)
```

### Validation (검증)

```python
from model import YOLOv10

model = YOLOv10("yolov10n.pt")

results = model.val(
    data="dataset/data.yaml",
    batch=16,
)
print(results)
```

### Prediction (추론)

```python
from model import YOLOv10

model = YOLOv10("yolov10n.pt")

# 이미지 파일 또는 웹캠(0) 입력 가능
results = model.predict(
    source="your_image.jpg",  # 또는 source=0 (웹캠)
    conf=0.5,
    save=True,
)
```

---

## Dataset

- **출처**: [Roboflow Universe — facial-expression-gtvqk](https://universe.roboflow.com/fisrt-one/facial-expression-gtvqk/dataset/1)
- **라이선스**: CC BY 4.0
- **클래스 수**: 5 (`anger`, `fear`, `happy`, `neutral`, `sad`)
- **구성**: train / valid / test 분리

---

## Citation

본 프로젝트는 YOLOv10 공식 구현체를 기반으로 합니다.

```bibtex
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```

- Paper: [arXiv:2405.14458](https://arxiv.org/abs/2405.14458v1)
- Official Repo: [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
