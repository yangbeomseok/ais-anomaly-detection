"""이상 탐지 모델 모듈 테스트"""

import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import HDBSCAN

from src.models import (
    prepare_features, detect_isolation_forest, detect_lof,
    detect_hdbscan, ensemble_anomaly, FEATURE_COLS,
)


@pytest.fixture
def model_df():
    """모델 입력용 테스트 데이터 (200행, 랜덤)."""
    np.random.seed(42)
    n = 200
    data = {
        "MMSI": np.random.choice([1, 2, 3], n),
        "SOG": np.random.exponential(3, n),
        "speed_deviation": np.random.normal(0, 1, n),
        "acceleration": np.random.normal(0, 0.1, n),
        "course_change": np.abs(np.random.normal(0, 30, n)),
        "heading_cog_diff": np.abs(np.random.normal(0, 20, n)),
        "signal_gap_sec": np.random.exponential(60, n),
        "stop_duration_min": np.random.exponential(5, n),
        "is_night": np.random.binomial(1, 0.3, n),
    }
    return pd.DataFrame(data)


def test_prepare_features_shape(model_df):
    X, scaler = prepare_features(model_df)
    assert X.shape == (len(model_df), len(FEATURE_COLS))


def test_prepare_features_returns_scaler(model_df):
    X, scaler = prepare_features(model_df)
    assert hasattr(scaler, "mean_")
    assert hasattr(scaler, "scale_")


def test_prepare_features_scaled(model_df):
    X, _ = prepare_features(model_df)
    assert np.abs(X.mean(axis=0)).max() < 0.1  # 평균 ~0


def test_prepare_features_missing_col():
    df = pd.DataFrame({"SOG": [1, 2, 3]})
    with pytest.raises(ValueError, match="필요한 컬럼"):
        prepare_features(df)


def test_detect_isolation_forest(model_df):
    result, model = detect_isolation_forest(model_df, contamination=0.1)
    assert "anomaly_if" in result.columns
    assert isinstance(model, IsolationForest)
    assert result["anomaly_if"].isin([0, 1]).all()
    assert result["anomaly_if"].sum() > 0


def test_detect_lof_returns_model(model_df):
    result, model = detect_lof(model_df, contamination=0.1)
    assert "anomaly_lof" in result.columns
    assert isinstance(model, LocalOutlierFactor)
    assert result["anomaly_lof"].sum() > 0


def test_detect_hdbscan_returns_model(model_df):
    result, model = detect_hdbscan(model_df, min_cluster_size=10, min_samples=5)
    assert "anomaly_hdbscan" in result.columns
    assert isinstance(model, HDBSCAN)


def test_ensemble_anomaly(model_df):
    model_df["anomaly_if"] = 0
    model_df["anomaly_lof"] = 0
    model_df["anomaly_hdbscan"] = 0
    # 처음 10개를 IF+LOF 이상으로 설정
    model_df.loc[:9, "anomaly_if"] = 1
    model_df.loc[:9, "anomaly_lof"] = 1

    result = ensemble_anomaly(model_df, threshold=2)
    assert "anomaly_final" in result.columns
    assert result["anomaly_final"].sum() == 10


def test_ensemble_no_anomaly_cols():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="앙상블할"):
        ensemble_anomaly(df)


def test_detect_isolation_forest_does_not_mutate(model_df):
    """원본 DataFrame이 변경되지 않는지 확인."""
    original_cols = set(model_df.columns)
    detect_isolation_forest(model_df, contamination=0.1)
    assert set(model_df.columns) == original_cols
