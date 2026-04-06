# AIS 선박 항적 이상 탐지 시스템

Maritime Vessel Anomaly Detection using AIS Data

## Overview

해양빅데이터AI센터에서 연구선 관측자료를 다루며 AIS 데이터도 비슷한 방식으로 분석해볼 수 있겠다 싶어서 시작한 사이드 프로젝트입니다.

해양에서 선박이 AIS 신호를 통해 위치/속도/침로를 주기적으로 송신하는데, 이걸 역으로 활용하면 **"이 배 뭔가 이상한데?"** 를 데이터로 잡아낼 수 있지 않을까? 라는 생각이 출발점이었습니다.

실제로 불법조업 선박은 단속 해역에서 급격한 방향 전환을 하거나, AIS 신호를 꺼버리는 패턴을 보입니다.
이런 행동 패턴을 비지도 학습으로 탐지하는 게 프로젝트의 핵심입니다.

## 용어 정리

분석에서 자주 나오는 용어들을 정리해두었습니다.

**해양/AIS 관련**
| 용어 | 설명 |
|------|------|
| AIS | Automatic Identification System. 선박이 자신의 위치·속도·방향 등을 자동으로 송신하는 장치. 일정 톤수 이상 선박은 의무 탑재 |
| MMSI | Maritime Mobile Service Identity. 선박마다 부여된 9자리 고유 식별번호. 주민등록번호 같은 것 |
| SOG | Speed Over Ground. 해저면 기준 실제 이동 속도 (단위: knot, 1knot ≈ 1.85km/h) |
| COG | Course Over Ground. 해저면 기준 실제 진행 방향 (0°=북, 90°=동, 180°=남, 270°=서) |
| Heading | 선수 방향. 선박의 앞머리가 가리키는 방향. 조류나 바람 때문에 COG와 다를 수 있음 |
| EEZ | Exclusive Economic Zone. 배타적 경제수역. 영해 밖 200해리까지의 해역 |

**모델/분석 관련**
| 용어 | 설명 |
|------|------|
| Isolation Forest | 데이터를 랜덤하게 분할해서, 빨리 고립되는 점을 이상치로 판단하는 알고리즘. 라벨 없이 작동 |
| LOF | Local Outlier Factor. 주변 데이터 밀도를 비교해서, 혼자 밀도가 낮은 점을 이상치로 판단 |
| HDBSCAN | Hierarchical DBSCAN. 계층적 밀도 기반 클러스터링으로, eps 파라미터 없이 가변 밀도 클러스터에 적응적으로 동작. 노이즈 포인트를 이상치로 분류 |
| SHAP | SHapley Additive exPlanations. 게임 이론 기반으로 각 피처가 개별 예측에 얼마나 기여했는지 정량적으로 분해하는 모델 해석 방법 |
| 앙상블 | 여러 모델의 결과를 종합해서 최종 판단을 내리는 방법. 여기서는 3개 중 2개 이상이 이상이라고 하면 최종 이상으로 판정 |
| 비지도 학습 | 정답 라벨 없이 데이터 패턴만으로 학습하는 방식. 이상 탐지에서는 "정상이 이렇게 생겼으니, 여기서 벗어나면 이상" 접근 |

## 뭘 했나

- NOAA에서 미국 연안 AIS 데이터 **1개월치(2022년 1월, 31일간)**를 가져와서
- 활동량 상위 500척으로 추려서
- 가속도, 선수-침로 괴리, 정박 패턴 등 **7가지 행동 피처**를 만들고
- Isolation Forest / LOF / HDBSCAN 세 가지 모델로 이상 탐지 돌려서
- 3개 중 2개 이상이 "이상"이라고 판단한 건만 최종 이상으로 잡고
- **SHAP으로 "왜 이 선박이 이상한지"** 피처별 기여도까지 분석했습니다

## Tech Stack

![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-6C3483?style=for-the-badge)
![Folium](https://img.shields.io/badge/folium-77B829?style=for-the-badge&logo=leaflet&logoColor=white)

## Project Structure

```
ais-anomaly-detection/
├── notebooks/
│   ├── 01_data_collection.ipynb      # 데이터 수집 및 전처리
│   ├── 02_eda.ipynb                  # 탐색적 데이터 분석
│   ├── 03_feature_engineering.ipynb  # 피처 엔지니어링
│   ├── 04_anomaly_detection.ipynb    # 이상 탐지 모델링 + SHAP 해석
│   └── 05_visualization.ipynb        # 지도 기반 시각화
├── src/
│   ├── config.py                     # config.yaml 로더
│   ├── data_loader.py                # 데이터 로드
│   ├── preprocessing.py              # 전처리 (좌표, 속도, 타임스탬프)
│   ├── features.py                   # 7가지 행동 피처 생성
│   ├── models.py                     # IF / LOF / HDBSCAN 앙상블
│   ├── explainer.py                  # SHAP 기반 모델 해석
│   └── visualize.py                  # 시각화 유틸리티
├── tests/                            # 단위 테스트
├── scripts/
│   ├── download_ais.py               # NOAA AIS 데이터 자동 다운로드
│   └── build_pipeline.py             # 전체 파이프라인 일괄 실행
├── data/
│   └── raw/                          # 원본 데이터 (git 미추적)
├── results/
│   └── figures/
├── config.yaml                       # 하이퍼파라미터 설정
├── requirements.txt
└── README.md
```

## Data

| 데이터 | 출처 | 비고 |
|--------|------|------|
| NOAA AIS (2022-01) | [MarineCadastre](https://marinecadastre.gov/accessais/) | 미국 연안, 31일간 |

원본 CSV는 일별 ~284MB(총 ~8.8GB)라 git에 올리지 않았습니다. 다운로드 스크립트로 자동 수집할 수 있습니다.

```bash
python scripts/download_ais.py
```

## Feature Engineering

AIS 원시 필드(SOG, COG, Heading)에서 7가지 행동 피처를 뽑아냈습니다.

| 피처 | 설명 | 이상 의미 |
|------|------|----------|
| speed_deviation | 선박별 평균 속도 대비 편차 (z-score) | 급가속/급감속 |
| acceleration | 속도 변화율 (knots/sec) | 급격한 가감속 |
| course_change | 연속 레코드 간 침로 변화량 | 급격한 방향 전환 |
| heading_cog_diff | 선수방향과 대지침로 간 각도 차이 | 비정상 조종, 조류/바람 영향 |
| signal_gap_sec | AIS 신호 간 시간 간격(초) | AIS 끄기 (은닉 행동) |
| stop_duration_min | 정박/저속 지속 시간(분) | 의심 정박 패턴 |
| is_night | 야간 활동 여부 (22~06시) | 야간 불법 활동 |

특히 `heading_cog_diff`는 AIS Heading 필드(511=사용불가)를 NaN 처리한 뒤 COG와의 차이를 계산합니다. 선수방향과 실제 진행방향이 크게 다르면 조류에 떠밀리거나 비정상 조종 상태를 의미합니다.

## Analysis Pipeline

```
원본 (31일간) → 전처리 → 상위 500척 → 7가지 피처 생성
                                         ↓
                              ┌── Isolation Forest
                              ├── LOF (n_neighbors=30)
                              └── HDBSCAN
                                         ↓
                              앙상블 (2/3 동의) → 최종 이상 판정
                                         ↓
                              SHAP 기반 피처 기여도 분석
```

---

## Results

### 0. 선박 교통량 지도

![Traffic Map](results/figures/traffic_map.png)

2022년 1월 한 달간 미국 연안에서 AIS 신호를 보낸 선박 500척의 위치입니다. 동해안, 서해안, 멕시코만 연안을 따라 항적이 분포하고, 주요 항구 근처에 밀집되어 있는 게 보입니다.

### 1. 어떤 배들이 있나

![Vessel Type Distribution](results/figures/vessel_type_distribution.png)

500척 중 절반 이상이 어선(Fishing)입니다. 미국 연안 데이터라 어선 비중이 높고, 화물선(Cargo)과 여객선(Passenger)이 뒤를 잇습니다. 어선이 많다는 건 이상 탐지 관점에서 좋은 소식인데, 실제로 불법조업 같은 이상 행동이 가장 빈번한 선종이 어선이기 때문입니다.

### 2. 속도 분포

![Speed Distribution](results/figures/speed_distribution.png)

대부분의 선박이 0~5노트(정박 또는 저속)에 몰려 있습니다. 선종별 평균 속도 차이가 확연한데, HSC(고속선)가 가장 빠르고 어선이 느린 편입니다. 이 차이가 이상 탐지에서 중요합니다 — 어선이 갑자기 고속으로 달리면 의심할 만하니까요.

### 3. 이상 탐지용 피처 분포

![Feature Distributions](results/figures/feature_distributions.png)

7가지 피처를 만들었습니다.
- **Speed Deviation**: 해당 선박의 평균 속도 대비 얼마나 벗어났는지. 대부분 0 근처에 몰려 있고 양 끝단이 이상 후보입니다.
- **Acceleration**: 속도 변화율. 급가속/급감속 구간을 잡아냅니다.
- **Course Change**: 연속 레코드 간 침로 변화량(도). 30도 이상이면 꽤 급격한 방향 전환입니다.
- **Heading-COG Diff**: 선수방향과 대지침로의 괴리. 45도 이상이면 조류/바람 영향 또는 비정상 상태를 의심합니다.
- **Signal Gap**: AIS 신호 간 시간 간격(분). 30분 이상 끊기면 의도적 신호 차단을 의심할 수 있습니다.
- **Stop Duration**: 정박/저속 지속 시간. 특이한 장소에서 장시간 정박하면 의심 대상입니다.
- **Night Activity**: 선박별 야간(22시~06시) 활동 비율.

### 4. 모델 비교

![Model Comparison](results/figures/model_comparison.png)

3가지 모델을 돌렸는데, Isolation Forest와 LOF는 contamination 파라미터로 이상 비율을 직접 설정하고, HDBSCAN은 밀도 기반으로 자체 판단합니다. HDBSCAN은 샘플링 없이 전체 데이터를 직접 처리하므로 편향이 없습니다.

오른쪽 Model Agreement를 보면 대부분이 0(정상)이고, 2개 이상 모델이 동의한 건이 최종 이상으로 잡힙니다. 단일 모델만 이상이라고 한 건은 false positive일 가능성이 높아서 걸러냈습니다.

### 4-1. contamination 민감도 분석

![Sensitivity Analysis](results/figures/sensitivity_analysis.png)

contamination(오염 비율)을 1%~10%로 바꿔보면서 결과가 얼마나 달라지는지 확인했습니다. HDBSCAN은 contamination 파라미터가 없으므로 고정값이며, 앙상블에서 IF/LOF의 변화에 따라 최종 결과가 조절됩니다.

### 5. 정상 vs 이상 — 뭐가 다른가

![Normal vs Anomaly](results/figures/normal_vs_anomaly.png)

이상으로 판정된 레코드(빨강)가 정상(파랑)과 어떻게 다른지 7가지 피처 차원에서 비교합니다.
- **SOG**: 이상 레코드는 고속 구간에 뚜렷하게 몰려 있습니다.
- **Acceleration**: 급가속/급감속 구간에서 이상이 집중됩니다.
- **Course Change**: 160~180도 급회전(유턴 수준)에서 이상이 튑니다.
- **Heading-COG Diff**: 선수방향과 진행방향의 큰 괴리가 이상 레코드에서 두드러집니다.

### 6. SHAP 기반 피처 기여도 분석

![SHAP Summary](results/figures/shap_summary.png)

SHAP으로 Isolation Forest의 이상 판정 근거를 분석했습니다. 각 점은 하나의 레코드이고, 색상은 피처 값의 크기(빨강=높음, 파랑=낮음)를 나타냅니다.

![SHAP Bar](results/figures/shap_bar.png)

평균 |SHAP| 기준 피처 중요도 순위입니다. 어떤 피처가 이상 판정에 가장 큰 영향을 미치는지 한눈에 볼 수 있습니다.

![SHAP Waterfall](results/figures/shap_waterfall.png)

개별 이상 레코드에 대한 waterfall plot입니다. **"이 배가 왜 이상한지"** 를 피처별로 분해해서 보여줍니다. 예를 들어 "이 레코드는 course_change가 높아서 이상 점수의 30%를 차지했고, SOG도 높아서 25%를 기여했다"는 식으로 해석할 수 있습니다.

### 7. 이상 항적 지도

![Anomaly Map](results/figures/anomaly_map_static.png)

실제 지도 위에 정상 항적(파랑)과 이상 항적(빨강)을 찍었습니다. 이상이 연안 전체에 분포하지만, 특히 항구 입출항 구간과 항로 전환 지점에 집중되는 게 보입니다.

### 8. 이상 상위 3척 항적 추적

![Top Vessels](results/figures/top_vessels_map.png)

이상 레코드가 가장 많은 선박 3척의 항적을 추적했습니다. 장거리 이동하면서 이상 구간(점)이 반복적으로 나타나는 패턴이 보입니다. 이런 선박은 실제로 정밀 조사 대상이 될 수 있습니다.

### 9. 이상 1위 선박 딥다이브

![Deep Dive Timeseries](results/figures/deep_dive_timeseries.png)

이상 레코드가 가장 많은 선박 한 척을 추적한 시계열입니다. 위에서부터 SOG(속도), Acceleration(가속도), COG(침로), Course Change(침로 변화량)입니다. 빨간 점이 이상으로 탐지된 시점인데, 급가속 구간에서 이상이 집중되는 패턴을 볼 수 있습니다.

![Deep Dive Map](results/figures/deep_dive_map.png)

같은 선박의 항적을 지도에 찍으면 이렇게 됩니다. 초록 삼각형이 출발, 빨간 사각형이 도착. 빨간 점이 이상 구간인데, 출발 직후와 중간 항로 전환 지점에 몰려 있습니다.

### 10. 시간대별 이상 빈도

![Hourly Anomaly Rate](results/figures/hourly_anomaly_rate.png)

UTC 기준 시간대별 이상 발생률입니다. 1개월 데이터로 분석하면 요일별 패턴도 함께 확인할 수 있습니다. 한국 해역 데이터로 분석하면 야간 집중 패턴이 더 뚜렷하게 나올 수 있습니다.

### 11. 선박 유형별 이상 비율

![Anomaly Rate by Type](results/figures/anomaly_rate_by_type.png)

선종별 이상 비율입니다. HSC(고속선)가 가장 높고, 어선이 낮은 편인데, 이건 미국 연안 데이터라 한국처럼 불법조업 이슈가 반영되지 않아서입니다. 고속선과 여객선이 높은 건 운항 특성상 급가속/급감속이 잦기 때문으로 보입니다.

---

## Conclusion

### 핵심 발견

1개월 AIS 레코드에서 3개 모델 앙상블로 이상 행동을 탐지했고, SHAP으로 뜯어보니 이상 판정에 가장 크게 먹히는 피처는 **속도 계열(SOG, acceleration, speed_deviation)**과 **방향 전환(course_change)**이었습니다.

### 실용적 의미

이 시스템의 핵심은 **스크리닝**입니다. 대량의 AIS 레코드에서 감시 대상을 좁혀주고, SHAP으로 **"왜 이상한지"**까지 설명할 수 있으니 해양 감시 인력이 단순 목록이 아닌 **근거 기반 우선순위**로 판단할 수 있습니다.

### 한국 해역 적용 시

현재는 미국 연안 데이터로 방법론을 검증한 단계입니다.

- **불법조업 탐지**: 한국 EEZ 내 중국 어선의 불법조업이 실제 이슈인 만큼, 어선의 이상 비율이 크게 달라질 수 있음
- **신호 단절 패턴**: 불법 선박은 단속 회피를 위해 AIS를 의도적으로 끄는 경우가 많아 핵심 피처가 될 수 있음
- **야간 집중 패턴**: 단일 시간대인 한국 해역에서는 야간 이상 집중이 더 뚜렷하게 나올 것으로 보임

---

## 한계점 및 향후 계획

- 현재는 미국 연안 데이터를 썼지만, 최종 목표는 **한국 해역(해양수산부 공공데이터)** 적용입니다
- 상위 500척만 추출했기 때문에 신호 단절 패턴이 잡히지 않을 수 있습니다. 전체 선박 대상 분석이 필요합니다
- 라벨이 없는 비지도 학습이라 정확도 측정이 어렵습니다. 실제 불법조업 적발 데이터와 매칭하면 검증이 가능합니다

## Getting Started

```bash
git clone https://github.com/yangbeomseok/ais-anomaly-detection.git
cd ais-anomaly-detection
pip install -r requirements.txt

# 1개월 AIS 데이터 다운로드 (~8.8GB)
python scripts/download_ais.py

# 노트북 실행
jupyter notebook
```

노트북을 01번부터 순서대로 실행하면 됩니다.

## License

MIT
