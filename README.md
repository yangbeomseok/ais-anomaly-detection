# AIS 선박 항적 이상 탐지 시스템

Maritime Vessel Anomaly Detection using AIS Data

## Overview

AIS(선박자동식별장치) 데이터를 활용하여 선박의 이상 행동 패턴을 탐지하는 데이터 분석 프로젝트입니다.
불법조업, 항로 이탈, AIS 신호 단절 등 해양 안전을 위협하는 이상 행동을 머신러닝 기반으로 식별합니다.

## Key Features

- **탐색적 데이터 분석**: 선박 유형별 분포, 교통량 패턴, 주요 항로 시각화
- **피처 엔지니어링**: 속도 이상, 침로 급변, 신호 단절, 항로 이탈, 야간 활동 등
- **이상 탐지 모델링**: Isolation Forest, DBSCAN, LOF 비교 분석
- **지도 기반 시각화**: folium을 활용한 정상/이상 항적 시각화

## Tech Stack

![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Folium](https://img.shields.io/badge/folium-77B829?style=for-the-badge&logo=leaflet&logoColor=white)
![Plotly](https://img.shields.io/badge/plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

## Project Structure

```
ais-anomaly-detection/
├── notebooks/
│   ├── 01_data_collection.ipynb      # 데이터 수집 및 전처리
│   ├── 02_eda.ipynb                  # 탐색적 데이터 분석
│   ├── 03_feature_engineering.ipynb  # 피처 엔지니어링
│   ├── 04_anomaly_detection.ipynb    # 이상 탐지 모델링
│   └── 05_visualization.ipynb        # 지도 기반 시각화
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── visualize.py
├── data/
│   └── raw/                          # 원본 데이터 (git 미추적)
├── results/
│   └── figures/                      # 분석 결과 시각화
├── requirements.txt
└── README.md
```

## Data Sources

| 데이터 | 출처 | 용도 |
|--------|------|------|
| AIS Dataset | [Kaggle](https://www.kaggle.com/datasets/eminserkanerdonmez/ais-dataset) | 프로토타이핑 |
| AIS 동적 정보 | [해양수산부 공공데이터](https://www.data.go.kr/data/15129186/fileData.do) | 본 분석 |

## Analysis Pipeline

```
데이터 수집 → 전처리 → EDA → 피처 엔지니어링 → 이상 탐지 → 시각화
```

1. **데이터 수집/전처리**: 결측치 처리, 좌표 범위 필터링, MMSI 기준 그룹화
2. **EDA**: 선박 유형 분포, 시간대별 교통량, 주요 항로 히트맵
3. **피처 엔지니어링**: 속도 편차, 침로 변화량, 신호 간격, 항로 이탈 거리
4. **이상 탐지**: Isolation Forest / DBSCAN / LOF 모델 비교
5. **시각화**: 지도 위 정상 항적(blue) vs 이상 항적(red) 오버레이

## Results

> 분석 완료 후 주요 결과 시각화가 추가됩니다.

## Getting Started

```bash
git clone https://github.com/yangbeomseok/ais-anomaly-detection.git
cd ais-anomaly-detection
pip install -r requirements.txt
jupyter notebook
```

## License

MIT
