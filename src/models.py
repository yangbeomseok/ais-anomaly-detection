"""이상 탐지 모델 모듈"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from .config import load_config

logger = logging.getLogger(__name__)

_cfg = load_config()
FEATURE_COLS = _cfg.get("features", {}).get("columns", [
    "SOG", "speed_deviation", "acceleration",
    "course_change", "heading_cog_diff",
    "signal_gap_sec", "stop_duration_min", "is_night",
])


def _validate_features(df: pd.DataFrame, feature_cols: list) -> None:
    """피처 컬럼 존재 여부를 검증한다."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame에 필요한 컬럼이 없습니다: {missing}")


def prepare_features(df: pd.DataFrame, feature_cols: list = None) -> tuple[np.ndarray, StandardScaler]:
    """모델 입력용 피처 행렬을 준비한다. (스케일링된 배열, 학습된 스케일러)를 반환한다."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    _validate_features(df, feature_cols)
    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def detect_isolation_forest(df: pd.DataFrame, contamination: float = None,
                            feature_cols: list = None) -> tuple[pd.DataFrame, IsolationForest]:
    """Isolation Forest 기반 이상 탐지. (결과 DataFrame, 학습된 모델)을 반환한다."""
    if contamination is None:
        contamination = _cfg.get("models", {}).get("isolation_forest", {}).get("contamination", 0.05)
    random_state = _cfg.get("models", {}).get("isolation_forest", {}).get("random_state", 42)

    df = df.copy()
    X, _ = prepare_features(df, feature_cols)
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    df["anomaly_if"] = model.fit_predict(X)
    df["anomaly_if"] = (df["anomaly_if"] == -1).astype(int)
    logger.info("Isolation Forest: %d anomalies (%.2f%%)", df["anomaly_if"].sum(),
                df["anomaly_if"].mean() * 100)
    return df, model


def detect_lof(df: pd.DataFrame, contamination: float = None,
               n_neighbors: int = None, feature_cols: list = None) -> tuple[pd.DataFrame, LocalOutlierFactor]:
    """Local Outlier Factor 기반 이상 탐지. (결과 DataFrame, 학습된 모델)을 반환한다."""
    lof_cfg = _cfg.get("models", {}).get("lof", {})
    if contamination is None:
        contamination = lof_cfg.get("contamination", 0.05)
    if n_neighbors is None:
        n_neighbors = lof_cfg.get("n_neighbors", 30)

    df = df.copy()
    X, _ = prepare_features(df, feature_cols)
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1,
    )
    preds = model.fit_predict(X)
    df["anomaly_lof"] = (preds == -1).astype(int)
    logger.info("LOF: %d anomalies (%.2f%%)", df["anomaly_lof"].sum(),
                df["anomaly_lof"].mean() * 100)
    return df, model


def detect_hdbscan(df: pd.DataFrame, min_cluster_size: int = None,
                   min_samples: int = None, feature_cols: list = None) -> tuple[pd.DataFrame, HDBSCAN]:
    """HDBSCAN 기반 이상 탐지. noise label(-1)을 이상으로 판정한다.
    (결과 DataFrame, 학습된 모델)을 반환한다."""
    hdb_cfg = _cfg.get("models", {}).get("hdbscan", {})
    if min_cluster_size is None:
        min_cluster_size = hdb_cfg.get("min_cluster_size", 50)
    if min_samples is None:
        min_samples = hdb_cfg.get("min_samples", 10)

    df = df.copy()
    X, _ = prepare_features(df, feature_cols)
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        copy=True,
        n_jobs=-1,
    )
    labels = model.fit_predict(X)
    df["anomaly_hdbscan"] = (labels == -1).astype(int)
    logger.info("HDBSCAN: %d anomalies (%.2f%%)", df["anomaly_hdbscan"].sum(),
                df["anomaly_hdbscan"].mean() * 100)
    return df, model


def ensemble_anomaly(df: pd.DataFrame, threshold: int = None) -> pd.DataFrame:
    """앙상블: threshold개 이상 모델이 이상으로 판단하면 최종 이상."""
    if threshold is None:
        threshold = _cfg.get("models", {}).get("ensemble", {}).get("threshold", 2)

    df = df.copy()
    anomaly_cols = [c for c in df.columns if c.startswith("anomaly_") and c != "anomaly_final" and c != "anomaly_score"]
    if not anomaly_cols:
        raise ValueError("앙상블할 이상 탐지 결과 컬럼이 없습니다 (anomaly_* 컬럼 필요)")
    df["anomaly_score"] = df[anomaly_cols].sum(axis=1)
    df["anomaly_final"] = (df["anomaly_score"] >= threshold).astype(int)
    logger.info("Ensemble (%d/%d): %d anomalies (%.2f%%)", threshold, len(anomaly_cols),
                df["anomaly_final"].sum(), df["anomaly_final"].mean() * 100)
    return df
