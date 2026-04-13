"""1개월 AIS 데이터 전체 파이프라인 (메모리 최적화 버전)
1차: 일별 CSV에서 선박별 레코드 수 카운트 → 상위 500척 확정
2차: 해당 선박만 필터링하며 로드 → 전처리 → 피처 → 모델

v2 개선: NaN imputation, IF auto threshold (elbow), per-vessel LOF/HDBSCAN
"""

import pandas as pd
import numpy as np
import gc
import sys
from pathlib import Path
from collections import Counter

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))          # scripts/ 디렉토리도 추가
sys.stdout.reconfigure(line_buffering=True)

from src.preprocessing import clean_coordinates, clean_speed, parse_timestamp, sort_by_vessel_time
from src.features import build_features
from src.models import ensemble_anomaly, FEATURE_COLS
from run_models import impute_features, find_elbow_threshold

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

# ── 모델 실행 (v2: imputation, auto IF threshold, per-vessel LOF/HDBSCAN) ──
print("\n=== 모델 실행 ===")

# NaN imputation (dropna 대신) — 전체 행 보존
print("  NaN imputation (선박별 중앙값)...")
df_model = impute_features(df)
del df
gc.collect()
n_total = len(df_model)
print(f"  모델 입력: {n_total:,} rows (100% preserved)")

# 피처 스케일링
X_raw = df_model[FEATURE_COLS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
del X_raw
gc.collect()

# 미세 노이즈 (LOF 동일 거리 방지)
rng = np.random.RandomState(42)
X_scaled += rng.normal(0, 1e-8, X_scaled.shape)

# 선박별 인덱스 사전
vessels = df_model["MMSI"].unique()
vessel_idx = {
    mmsi: np.where(df_model["MMSI"].values == mmsi)[0]
    for mmsi in vessels
}
print(f"  선박 수: {len(vessels)}")

# ── Isolation Forest (글로벌, 자동 임계값 - elbow method) ──
print("  Isolation Forest (auto threshold)...")
model_if = IsolationForest(random_state=42, n_jobs=-1)
model_if.fit(X_scaled)
scores_if = model_if.score_samples(X_scaled)

RESULTS_DIR = DATA_DIR.parent / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
threshold_if = find_elbow_threshold(
    scores_if, plot_path=str(RESULTS_DIR / "if_auto_threshold.png")
)
df_model["anomaly_if"] = (scores_if < threshold_if).astype(int)
n_if = int(df_model["anomaly_if"].sum())
print(f"    → {n_if:,} ({n_if/n_total*100:.2f}%, auto)")
del model_if
gc.collect()

# ── LOF (선박별) ──
print(f"  LOF (per-vessel, {len(vessels)} vessels)...")
lof_arr = np.zeros(n_total, dtype=np.int8)

for i, mmsi in enumerate(vessels):
    idx = vessel_idx[mmsi]
    X_v = X_scaled[idx]
    n_pts = len(X_v)
    if n_pts < 10:
        continue
    n_neighbors = min(30, n_pts - 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
    preds = lof.fit_predict(X_v)
    lof_arr[idx] = (preds == -1).astype(np.int8)
    if (i + 1) % 100 == 0:
        print(f"    LOF progress: {i+1}/{len(vessels)} vessels")

df_model["anomaly_lof"] = lof_arr
n_lof = int(lof_arr.sum())
print(f"    → {n_lof:,} ({n_lof/n_total*100:.2f}%)")
del lof_arr
gc.collect()

# ── HDBSCAN (선박별) ──
print(f"  HDBSCAN (per-vessel, {len(vessels)} vessels)...")
hdb_arr = np.zeros(n_total, dtype=np.int8)

for i, mmsi in enumerate(vessels):
    idx = vessel_idx[mmsi]
    X_v = X_scaled[idx]
    n_pts = len(X_v)
    mcs = max(10, min(50, n_pts // 20))
    ms = min(10, mcs)
    if n_pts < mcs + 1:
        continue
    hdb = HDBSCAN(min_cluster_size=mcs, min_samples=ms)
    labels = hdb.fit_predict(X_v)
    hdb_arr[idx] = (labels == -1).astype(np.int8)
    if (i + 1) % 100 == 0:
        print(f"    HDBSCAN progress: {i+1}/{len(vessels)} vessels")

df_model["anomaly_hdbscan"] = hdb_arr
n_hdb = int(hdb_arr.sum())
print(f"    → {n_hdb:,} ({n_hdb/n_total*100:.2f}%)")
del hdb_arr, X_scaled
gc.collect()

# ── 앙상블 ──
print("  앙상블...")
df_model = ensemble_anomaly(df_model, threshold=2)
n_final = int(df_model['anomaly_final'].sum())
print(f"    → Ensemble: {n_final:,} ({n_final/n_total*100:.2f}%)")

df_model.to_parquet(DATA_DIR / "ais_results.parquet", index=False)
print(f"  저장: ais_results.parquet ({(DATA_DIR / 'ais_results.parquet').stat().st_size / 1e6:.1f} MB)")

print("\n=== 완료! ===")
