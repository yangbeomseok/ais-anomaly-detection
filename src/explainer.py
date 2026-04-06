"""SHAP 기반 모델 해석 모듈"""

import logging
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from .config import load_config
from .models import FEATURE_COLS, prepare_features

logger = logging.getLogger(__name__)

_cfg = load_config()


def compute_shap_values(model, df: pd.DataFrame, feature_cols: list = None,
                        max_samples: int = None) -> tuple[shap.Explanation, np.ndarray]:
    """Isolation Forest에 대한 SHAP 값을 계산한다.
    TreeExplainer를 사용하여 각 피처의 이상 판정 기여도를 산출한다."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if max_samples is None:
        max_samples = _cfg.get("shap", {}).get("max_samples", 5000)

    X, _ = prepare_features(df, feature_cols)
    X_df = pd.DataFrame(X, columns=feature_cols)

    if len(X_df) > max_samples:
        X_df = X_df.sample(max_samples, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_df)
    logger.info("SHAP values computed for %d samples", len(X_df))
    return shap_values, X_df.values


def plot_shap_summary(shap_values, feature_names: list = None,
                      save_path: str = None, max_display: int = 10):
    """SHAP summary plot: 전체 피처의 이상 탐지 기여도를 시각화한다."""
    if feature_names is None:
        feature_names = FEATURE_COLS
    dpi = _cfg.get("visualization", {}).get("dpi", 150)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return plt.gcf()


def plot_shap_bar(shap_values, feature_names: list = None,
                  save_path: str = None, max_display: int = 10):
    """SHAP bar plot: 평균 절대 SHAP 값 기반 피처 중요도."""
    if feature_names is None:
        feature_names = FEATURE_COLS
    dpi = _cfg.get("visualization", {}).get("dpi", 150)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return plt.gcf()


def plot_shap_waterfall(shap_values, index: int = 0, save_path: str = None):
    """개별 레코드에 대한 SHAP waterfall plot.
    특정 선박의 이상 판정 이유를 설명할 때 사용한다."""
    dpi = _cfg.get("visualization", {}).get("dpi", 150)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[index], show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return plt.gcf()


def get_top_anomaly_explanations(shap_values, df_sampled: pd.DataFrame,
                                 feature_cols: list = None, top_n: int = 10) -> pd.DataFrame:
    """SHAP 값 기반으로 가장 이상한 레코드 top_n개와 주요 원인 피처를 반환한다."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    abs_shap = np.abs(shap_values.values).sum(axis=1)
    top_indices = np.argsort(abs_shap)[-top_n:][::-1]

    results = []
    for idx in top_indices:
        row_shap = shap_values.values[idx]
        top_feat_idx = np.argmax(np.abs(row_shap))
        results.append({
            "sample_index": idx,
            "total_shap": abs_shap[idx],
            "top_feature": feature_cols[top_feat_idx],
            "top_feature_shap": row_shap[top_feat_idx],
        })
    return pd.DataFrame(results)
