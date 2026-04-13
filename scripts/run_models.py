"""
AIS Anomaly Detection - 모델 실행 스크립트
ais_featured.parquet → IF(글로벌) + LOF(선박별) + HDBSCAN(선박별) → 앙상블 → ais_results.parquet
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
import logging
import gc
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "SOG", "speed_deviation", "acceleration", "course_change",
    "heading_cog_diff", "signal_gap_sec", "stop_duration_min", "is_night",
]


def main():
    t_start = time.time()

    # ── 1. 데이터 로드 ──────────────────────────────────────
    logger.info("Loading ais_featured.parquet...")
    df = pd.read_parquet("data/ais_featured.parquet")
    logger.info("Loaded: %d rows, %d cols", *df.shape)

    df_model = df.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)
    del df
    gc.collect()
    n_total = len(df_model)
    logger.info("After dropna: %d rows", n_total)

    # ── 2. 피처 스케일링 ────────────────────────────────────
    X_raw = df_model[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    del X_raw
    gc.collect()

    # 미세 노이즈 추가 (동일 거리로 인한 LOF 0-div 방지)
    rng = np.random.RandomState(42)
    X_scaled += rng.normal(0, 1e-8, X_scaled.shape)

    # 선박별 인덱스 사전 구축
    vessels = df_model["MMSI"].unique()
    vessel_idx = {
        mmsi: np.where(df_model["MMSI"].values == mmsi)[0]
        for mmsi in vessels
    }
    logger.info("Vessels: %d", len(vessels))

    # ── 3. Isolation Forest (글로벌) ────────────────────────
    logger.info("=" * 60)
    logger.info("[1/3] Isolation Forest — global, %d rows", n_total)
    t0 = time.time()
    model_if = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    df_model["anomaly_if"] = (model_if.fit_predict(X_scaled) == -1).astype(int)
    n_if = int(df_model["anomaly_if"].sum())
    logger.info("  → %d anomalies (%.2f%%) in %.1fs",
                n_if, n_if / n_total * 100, time.time() - t0)
    del model_if
    gc.collect()

    # ── 4. LOF (선박별) ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[2/3] LOF — per-vessel (%d vessels)", len(vessels))
    t0 = time.time()
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
            logger.info("  LOF progress: %d/%d vessels", i + 1, len(vessels))

    df_model["anomaly_lof"] = lof_arr
    n_lof = int(lof_arr.sum())
    logger.info("  → %d anomalies (%.2f%%) in %.1fs",
                n_lof, n_lof / n_total * 100, time.time() - t0)
    del lof_arr
    gc.collect()

    # ── 5. HDBSCAN (선박별) ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("[3/3] HDBSCAN — per-vessel (%d vessels)", len(vessels))
    t0 = time.time()
    hdb_arr = np.zeros(n_total, dtype=np.int8)

    for i, mmsi in enumerate(vessels):
        idx = vessel_idx[mmsi]
        X_v = X_scaled[idx]
        n_pts = len(X_v)

        # 적응적 파라미터: 선박당 행 수에 비례
        mcs = max(10, min(50, n_pts // 20))
        ms = min(10, mcs)
        if n_pts < mcs + 1:
            continue

        hdb = HDBSCAN(min_cluster_size=mcs, min_samples=ms)
        labels = hdb.fit_predict(X_v)
        hdb_arr[idx] = (labels == -1).astype(np.int8)
        if (i + 1) % 100 == 0:
            logger.info("  HDBSCAN progress: %d/%d vessels", i + 1, len(vessels))

    df_model["anomaly_hdbscan"] = hdb_arr
    n_hdb = int(hdb_arr.sum())
    logger.info("  → %d anomalies (%.2f%%) in %.1fs",
                n_hdb, n_hdb / n_total * 100, time.time() - t0)
    del hdb_arr, X_scaled
    gc.collect()

    # ── 6. 앙상블 투표 ──────────────────────────────────────
    logger.info("=" * 60)
    anomaly_cols = ["anomaly_if", "anomaly_lof", "anomaly_hdbscan"]
    df_model["anomaly_score"] = df_model[anomaly_cols].sum(axis=1)
    df_model["anomaly_final"] = (df_model["anomaly_score"] >= 2).astype(int)
    n_final = int(df_model["anomaly_final"].sum())

    logger.info("Ensemble (threshold ≥ 2/3): %d anomalies (%.2f%%)",
                n_final, n_final / n_total * 100)

    # ── 요약 ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SUMMARY")
    for c in anomaly_cols + ["anomaly_final"]:
        n = int(df_model[c].sum())
        logger.info("  %-20s %7d  (%.2f%%)", c, n, n / n_total * 100)

    # ── 7. 저장 ─────────────────────────────────────────────
    output_path = "data/ais_results.parquet"
    df_model.to_parquet(output_path, index=False)
    sz_mb = round(pd.io.common.file_exists(output_path) and
                  __import__("os").path.getsize(output_path) / 1e6, 1)
    logger.info("Saved → %s  (%d rows, %d cols, ~%.0f MB)",
                output_path, *df_model.shape, sz_mb)
    logger.info("Total elapsed: %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
