"""1개월 AIS 데이터 전체 파이프라인 (메모리 최적화 버전)
1차: 일별 CSV에서 선박별 레코드 수 카운트 → 상위 500척 확정
2차: 해당 선박만 필터링하며 로드 → 전처리 → 피처 → 모델
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from src.preprocessing import clean_coordinates, clean_speed, parse_timestamp, sort_by_vessel_time
from src.features import build_features
from src.models import (
    detect_isolation_forest, detect_lof, detect_hdbscan,
    ensemble_anomaly, FEATURE_COLS
)

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

USE_COLS = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading',
            'VesselType', 'VesselName', 'Length', 'Width', 'Status']
DTYPES = {'MMSI': 'int64', 'VesselType': 'float32', 'SOG': 'float32',
          'COG': 'float32', 'Heading': 'float32'}

raw_files = sorted(RAW_DIR.glob("AIS_2022_01_*.csv"))
print(f"CSV 파일 수: {len(raw_files)}")

# ── PASS 1: 선박별 레코드 수 카운트 (MMSI 컬럼만 읽음) ──
print("\n=== PASS 1: 상위 500척 선별 ===")
vessel_counts = Counter()

for i, f in enumerate(raw_files, 1):
    mmsi_series = pd.read_csv(f, usecols=['MMSI'], dtype={'MMSI': 'int64'})['MMSI']
    vessel_counts.update(mmsi_series.value_counts().to_dict())
    print(f"  [{i}/{len(raw_files)}] {f.name} ({len(mmsi_series):,} rows)")

top_500 = set(mmsi for mmsi, _ in vessel_counts.most_common(500))
print(f"\n  상위 500척 확정 (최소 레코드: {vessel_counts.most_common(500)[-1][1]:,})")

# ── PASS 2: 상위 500척만 필터링하며 로드 + 전처리 ──
print("\n=== PASS 2: 데이터 로드 + 전처리 ===")
chunks = []

for i, f in enumerate(raw_files, 1):
    chunk = pd.read_csv(f, usecols=USE_COLS, dtype=DTYPES, parse_dates=['BaseDateTime'])
    chunk = chunk[chunk['MMSI'].isin(top_500)]

    # 전처리
    chunk = chunk[
        (chunk['LAT'].between(-90, 90)) &
        (chunk['LON'].between(-180, 180)) &
        (chunk['LAT'] != 0) & (chunk['LON'] != 0)
    ]
    chunk = chunk[(chunk['SOG'] >= 0) & (chunk['SOG'] < 102.3)]
    chunk = chunk.dropna(subset=['BaseDateTime'])

    chunks.append(chunk)
    print(f"  [{i}/{len(raw_files)}] {f.name} → {len(chunk):,} rows")

df = pd.concat(chunks, ignore_index=True)
del chunks
df = df.sort_values(['MMSI', 'BaseDateTime']).reset_index(drop=True)

print(f"\n  총: {len(df):,} rows, {df['MMSI'].nunique()} vessels")
print(f"  기간: {df['BaseDateTime'].min()} ~ {df['BaseDateTime'].max()}")

df.to_parquet(DATA_DIR / "ais_cleaned.parquet", index=False)
print(f"  저장: ais_cleaned.parquet ({(DATA_DIR / 'ais_cleaned.parquet').stat().st_size / 1e6:.1f} MB)")

# ── 피처 생성 ──
print("\n=== 피처 생성 ===")
df = build_features(df)
df.to_parquet(DATA_DIR / "ais_featured.parquet", index=False)
print(f"  저장: ais_featured.parquet ({(DATA_DIR / 'ais_featured.parquet').stat().st_size / 1e6:.1f} MB)")

# ── 모델 실행 ──
print("\n=== 모델 실행 ===")
df_model = df.dropna(subset=FEATURE_COLS).copy()
del df
print(f"  모델 입력: {len(df_model):,} rows")

print("  Isolation Forest...")
df_model, if_model = detect_isolation_forest(df_model, contamination=0.05)
print(f"    → {df_model['anomaly_if'].sum():,}")

print("  LOF...")
df_model, lof_model = detect_lof(df_model, contamination=0.05, n_neighbors=30)
print(f"    → {df_model['anomaly_lof'].sum():,}")

print("  HDBSCAN...")
df_model, hdb_model = detect_hdbscan(df_model, min_cluster_size=50, min_samples=10)
print(f"    → {df_model['anomaly_hdbscan'].sum():,}")

print("  앙상블...")
df_model = ensemble_anomaly(df_model, threshold=2)
n_final = df_model['anomaly_final'].sum()
print(f"    → Ensemble: {n_final:,} ({n_final/len(df_model)*100:.2f}%)")

df_model.to_parquet(DATA_DIR / "ais_results.parquet", index=False)
print(f"  저장: ais_results.parquet ({(DATA_DIR / 'ais_results.parquet').stat().st_size / 1e6:.1f} MB)")

print("\n=== 완료! ===")
