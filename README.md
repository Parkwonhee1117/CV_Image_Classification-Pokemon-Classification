# 🧠 Pokemon Classification with Transfer Learning

## 📌 1. 프로젝트 개요
<img width="747" height="1128" alt="image" src="https://github.com/user-attachments/assets/17d5a051-756b-4112-963b-ff13cb9298ab" />
<img width="740" height="1092" alt="image" src="https://github.com/user-attachments/assets/bb4a4929-5b61-4a4e-bb42-d791982312c7" />


본 프로젝트는 Transfer Learning을 활용하여 포켓몬 이미지를 분류하는 딥러닝 모델을 구축하는 것을 목표로 한다.
Kaggle의 Pokemon Dataset (약 7,000장, 150 클래스)을 사용하여 다양한 실험 설정을 비교 분석하였다.

---

## 📌 2. 데이터셋

* Dataset: Pokemon Image Dataset
* 이미지 수: 약 7,000장
* 클래스 수: 150개
* 데이터 구조:

```
archive/PokemonData/
├── Bulbasaur/
├── Charmander/
├── Pikachu/
...
```

---

## 📌 3. 실험 설계

총 4가지 실험을 통해 모델 성능을 비교하였다.

### 🔧 실험 변수

* Backbone 모델: ResNet50, EfficientNet-B0
* Pretrained 사용 여부
* Fine-tuning 범위 (FC layer만 vs 전체 학습)

---

## 📊 4. 실험 설정

| Experiment | Model           | Pretrained | Fine-tuning |
| ---------- | --------------- | ---------- | ----------- |
| Exp1       | ResNet50        | O          | FC only     |
| Exp2       | ResNet50        | O          | All layers  |
| Exp3       | EfficientNet-B0 | O          | All layers  |
| Exp4       | ResNet50        | X          | All layers  |

---

## 📈 5. 실험 결과

> results.json 기반으로 작성

| Experiment | Accuracy |
| ---------- | -------- |
| Exp1       | 0.8123167155425219   |
| Exp2       | 0.8922287390029325   |
| Exp3       | 0.9303519061583577   |
| Exp4       | 0.22067448680351906   |

---

## 🔍 6. 결과 분석

### 1️⃣ Transfer Learning 효과

* Pretrained 모델(Exp1, Exp2, Exp3)은 높은 성능을 보인 반면,
  Random Initialization(Exp4)은 **Accuracy 22%로 매우 낮은 성능**을 기록하였다.
* 이는 데이터셋 규모(약 7,000장)가 충분히 크지 않기 때문에,
  사전 학습된 특징을 활용하는 **Transfer Learning이 필수적**임을 보여준다.

---

### 2️⃣ Fine-tuning 범위 영향

* FC layer만 학습한 Exp1(81%) 대비,
  전체 레이어를 학습한 Exp2(89%)는 **약 8% 성능 향상**을 보였다.
* 이는 포켓몬 이미지가 ImageNet과 도메인이 다르기 때문에
  feature extractor까지 재학습하는 것이 중요함을 의미한다.

---

### 3️⃣ 모델 구조 영향

* EfficientNet 기반 모델(Exp3)이 **93%로 가장 높은 성능**을 기록하였다.
* 이는 ResNet50보다 더 효율적인 feature extraction 구조 덕분으로 해석된다.

---

## 🎯 최종 결론

* Transfer Learning은 필수적이며,
* Full Fine-tuning이 성능 향상에 크게 기여하고,
* EfficientNet이 가장 우수한 성능을 보였다.

👉 최적 조합:
**EfficientNet + Pretrained + Full Fine-tuning**

---

## 📈 7. 학습 과정

* Epoch: 5
* Optimizer: Adam (lr=0.001)
* Loss: CrossEntropyLoss

👉 Validation Accuracy 기준으로 Best 성능 기록

---

## 🖥️ 8. Demo (Streamlit GUI)

Streamlit을 이용하여 이미지 업로드 기반 포켓몬 분류 시스템 구현

### 기능

* 이미지 업로드
* Top-5 예측 결과 출력
* 확률 기반 결과 표시

---

## 🚀 실행 방법

### 1️⃣ 학습 실행

```
python train.py
```

👉 실행 후: (다소 시간 걸림 - 약 1시간)

* exp1.pth ~ exp4.pth 생성
* classes.txt 생성
* results.json 생성

---

### 2️⃣ GUI 실행

```
streamlit run app.py
```

---

## 📁 프로젝트 구조

```
project/
│── train.py
│── app.py
│── classes.txt
│── exp1.pth
│── exp2.pth
│── exp3.pth
│── exp4.pth
│── results.json
│── archive/
│    └── PokemonData/
```

---

## 📌 9. 결론

본 프로젝트를 통해 Transfer Learning이 이미지 분류 문제에서 매우 효과적임을 확인하였다.
특히 pretrained 모델과 전체 fine-tuning을 함께 적용할 경우 가장 높은 성능을 달성할 수 있었다.

---

## ✨ 10. 향후 개선 방향

* 더 많은 epoch 학습
* 데이터 augmentation 강화
* Confusion Matrix 및 추가 분석
* Top-1 / Top-5 정확도 비교

---
