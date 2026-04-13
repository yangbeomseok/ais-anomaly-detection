"""
AIS Anomaly Detection - 모델 실행 스크립트 (v2)

개선 사항:
  1. NaN 복구: dropna 대신 선박별 중앙값 대체 → 전체 430만 행 유지
  2. IF 자동 임계값: elbow method로 contamination 자동 결정
  3. LOF/HDBSCAN: 선박별 실행 유지

ais_featured.parquet → 모델 3개 → 앙상블 → ais_results.parquet
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUT_DIR = "results/figures"


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """NaN을 선박별 중앙값으로 대체한다. 중앙값도 없으면 0으로 채운다."""
    df = df.copy()
    for col in FEATURE_COLS:
        if df[col].isna().any():
            n_na = df[col].isna().sum()
            # 선박별 중앙값 대체
            df[col] = df.groupby("MMSI")[col].transform(
                lambda s: s.fillna(s.median())
            )
            # 그래도 남은 NaN (해당 선박이 전부 NaN인 경우) → 전체 중앙값
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
            # 최후의 수단
            df[col] = df[col].fillna(0)
            logger.info("  %s: %d NaN → 선박별 중앙값 대체 완료", col, n_na)
    return df


def find_elbow_threshold(scores: np.ndarray, plot_path: str = None) -> float:
    """정렬된 anomaly score에서 elbow point를 찾아 임계값을 반환한다.
    scores: 낮을수록 이상 (IF의 score_samples 결과)
    """
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)

    # 서브샘플링 (효율)
    sample_size = min(50000, n)
    idx = np.linspace(0, n - 1, sample_size, dtype=int)
    y = sorted_scores[idx]

    # 정규화
    x = np.linspace(0, 1, sample_size)
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min + 1e-10)

    # 시작-끝 직선 대비 최대 거리 = elbow
    line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x
    diff = y_norm - line
    elbow_idx = np.argmax(diff)
    threshold = y[elbow_idx]

    # 시각화
    if plot_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # (a) Score 분포 + threshold
        ax1.hist(scores, bins=200, color="#3498db", alpha=0.7, edgecolor="none")
        ax1.axvline(threshold, color="red", linestyle="--", linewidth=2,
                    label=f"Auto threshold = {threshold:.4f}")
        anomaly_pct = (scores < threshold).mean() * 100
        ax1.set_title(f"IF Anomaly Score Distribution (auto: {anomaly_pct:.1f}%)")
        ax1.set_xlabel("Anomaly Score")
        ax1.set_ylabel("Count")
        ax1.legend()

        # (b) Elbow curve
        ax2.plot(np.linspace(0, 100, sample_size), y, color="#3498db", linewidth=0.5)
        ax2.axhline(threshold, color="red", linestyle="--", linewidth=2,
                    label=f"Elbow point")
        ax2.scatter([elbow_idx / sample_size * 100], [threshold],
                    color="red", s=100, zorder=5)
        ax2.set_title("Sorted Scores — Elbow Detection")
        ax2.set_xlabel("Percentile (%)")
        ax2.set_ylabel("Anomaly Score")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  → %s", plot_path)

    return threshold


def main():
    t_start = time.time()

    # ── 1. 데이터 로드 ──────────────────────────────────────
    logger.info("Loading ais_featured.parquet...")
    df = pd.read_parquet("data/ais_featured.parquet")
    logger.info("Loaded: %d rows, %d cols", *df.shape)

    # ── 2. NaN 복구 (dropna 대신 imputation) ────────────────
    logger.info("Imputing NaN values...")
    df_model = impute_features(df)
    del df
    gc.collect()
    n_total = len(df_model)
    logger.info("After imputation: %d rows (100%% preserved)", n_total)

    # ── 3. 피처 스케일링 ────────────────────────────────────
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
    logger.info("Vessels: %d", len(vessels))

    # ── 4. Isolation Forest (글로벌, 자동 임계값) ────────────
    logger.info("=" * 60)
    logger.info("[1/3] Isolation Forest — global, auto threshold")
    t0 = time.time()

    model_if = IsolationForest(random_state=42, n_jobs=-1)
    model_if.fit(X_scaled)
    scores_if = model_if.score_samples(X_scaled)

    # Elbow method로 임계값 자동 결정
    threshold_if = find_elbow_threshold(
        scores_if, plot_path=f"{OUT_DIR}/if_auto_threshold.png"
    )
    df_model["anomaly_if"] = (scores_if < threshold_if).astype(int)
    n_if = int(df_model["anomaly_if"].sum())
    auto_pct = n_if / n_total * 100
    logger.info("  → %d anomalies (%.2f%%, auto-determined) in %.1fs",
                n_if, auto_pct, time.time() - t0)
    del model_if
    gc.collect()

    # ── 5. LOF (선박별) ─────────────────────────────────────
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

    # ── 6. HDBSCAN (선박별) ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("[3/3] HDBSCAN — per-vessel (%d vessels)", len(vessels))
    t0 = time.time()
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
            logger.info("  HDBSCAN progress: %d/%d vessels", i + 1, len(vessels))

    df_model["anomaly_hdbscan"] = hdb_arr
    n_hdb = int(hdb_arr.sum())
    logger.info("  → %d anomalies (%.2f%%) in %.1fs",
                n_hdb, n_hdb / n_total * 100, time.time() - t0)
    del hdb_arr, X_scaled
    gc.collect()

    # ── 7. 앙상블 투표 ──────────────────────────────────────
    logger.info("=" * 60)
    anomaly_cols = ["anomaly_if", "anomaly_lof", "anomaly_hdbscan"]
    df_model["anomaly_score"] = df_model[anomaly_cols].sum(axis=1)
    df_model["anomaly_final"] = (df_model["anomaly_score"] >= 2).astype(int)
    n_final = int(df_model["anomaly_final"].sum())
    logger.info("Ensemble (threshold >= 2/3): %d anomalies (%.2f%%)",
                n_final, n_final / n_total * 100)

    # ── 요약 ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SUMMARY  (total rows: %d)", n_total)
    for c in anomaly_cols + ["anomaly_final"]:
        n = int(df_model[c].sum())
        logger.info("  %-20s %7d  (%.2f%%)", c, n, n / n_total * 100)

    # ── 8. 저장 ─────────────────────────────────────────────
    import os
    output_path = "data/ais_results.parquet"
    df_model.to_parquet(output_path, index=False)
    sz_mb = os.path.getsize(output_path) / 1e6
    logger.info("Saved → %s  (%d rows, %d cols, ~%.0f MB)",
                output_path, *df_model.shape, sz_mb)
    logger.info("Total elapsed: %.1f min", (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
