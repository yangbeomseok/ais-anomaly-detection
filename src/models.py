"""이상 탐지 모델 모듈"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["SOG", "speed_deviation", "course_change", "signal_gap_sec", "is_night"]


def prepare_features(df: pd.DataFrame, feature_cols: list = None) -> np.ndarray:
    """모델 입력용 피처 행렬을 준비한다."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def detect_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Isolation Forest 기반 이상 탐지."""
    df = df.copy()
    X, _ = prepare_features(df)
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    df["anomaly_if"] = model.fit_predict(X)
    df["anomaly_if"] = (df["anomaly_if"] == -1).astype(int)
    return df


def detect_lof(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Local Outlier Factor 기반 이상 탐지."""
    df = df.copy()
    X, _ = prepare_features(df)
    model = LocalOutlierFactor(contamination=contamination, n_jobs=-1)
    preds = model.fit_predict(X)
    df["anomaly_lof"] = (preds == -1).astype(int)
    return df


def detect_dbscan(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 10,
                  max_samples: int = 100000) -> pd.DataFrame:
    """DBSCAN 기반 이상 탐지 (클러스터 미소속 = 이상). 대용량 시 샘플링."""
    df = df.copy()
    X_full, scaler = prepare_features(df)

    if len(X_full) > max_samples:
        sample_idx = np.random.RandomState(42).choice(len(X_full), max_samples, replace=False)
        X_sample = X_full[sample_idx]
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        sample_labels = model.fit_predict(X_sample)

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        nn.fit(X_sample)
        dists, indices = nn.kneighbors(X_full)
        labels = sample_labels[indices.ravel()]
        labels[dists.ravel() > eps] = -1
    else:
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X_full)

    df["anomaly_dbscan"] = (labels == -1).astype(int)
    return df


def ensemble_anomaly(df: pd.DataFrame, threshold: int = 2) -> pd.DataFrame:
    """앙상블: threshold개 이상 모델이 이상으로 판단하면 최종 이상."""
    df = df.copy()
    anomaly_cols = [c for c in df.columns if c.startswith("anomaly_")]
    df["anomaly_score"] = df[anomaly_cols].sum(axis=1)
    df["anomaly_final"] = (df["anomaly_score"] >= threshold).astype(int)
    return df
