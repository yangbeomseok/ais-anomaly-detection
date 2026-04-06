"""시각화 모듈"""

import logging
import folium
from folium.plugins import HeatMap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import load_config

logger = logging.getLogger(__name__)
_cfg = load_config()


def plot_trajectory_map(df: pd.DataFrame, lat_col: str = "LAT", lon_col: str = "LON",
                        anomaly_col: str = "anomaly_final", center: list = None) -> folium.Map:
    """정상(파랑) vs 이상(빨강) 항적을 지도에 표시한다."""
    for col in [lat_col, lon_col, anomaly_col]:
        if col not in df.columns:
            raise ValueError(f"DataFrame에 '{col}' 컬럼이 없습니다")

    if center is None:
        center = [df[lat_col].mean(), df[lon_col].mean()]

    vis_cfg = _cfg.get("visualization", {})
    normal_sample = vis_cfg.get("normal_sample_size", 5000)
    anomaly_radius = vis_cfg.get("anomaly_marker_radius", 3)

    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB dark_matter")

    normal = df[df[anomaly_col] == 0].sample(min(normal_sample, len(df[df[anomaly_col] == 0])), random_state=42)
    for _, row in normal.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=1, color="blue", fill=True, opacity=0.3
        ).add_to(m)

    anomalies = df[df[anomaly_col] == 1]
    for _, row in anomalies.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=anomaly_radius, color="red", fill=True, opacity=0.8
        ).add_to(m)

    return m


def plot_traffic_heatmap(df: pd.DataFrame, lat_col: str = "LAT", lon_col: str = "LON",
                         center: list = None) -> folium.Map:
    """선박 교통량 히트맵을 생성한다."""
    if center is None:
        center = [df[lat_col].mean(), df[lon_col].mean()]

    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB dark_matter")
    heat_data = df[[lat_col, lon_col]].dropna().values.tolist()
    HeatMap(heat_data, radius=10, blur=15).add_to(m)
    return m


def plot_anomaly_distribution(df: pd.DataFrame, anomaly_col: str = "anomaly_final",
                              save_path: str = None):
    """이상 탐지 결과 분포를 시각화한다."""
    if anomaly_col not in df.columns:
        raise ValueError(f"DataFrame에 '{anomaly_col}' 컬럼이 없습니다")

    dpi = _cfg.get("visualization", {}).get("dpi", 150)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    counts = df[anomaly_col].value_counts()
    labels = ["Normal", "Anomaly"]
    colors = ["#3498db", "#e74c3c"]
    ax.bar(labels, counts.values, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Normal vs Anomaly Distribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
