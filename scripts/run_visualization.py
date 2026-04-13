"""
AIS Anomaly Detection - 시각화 스크립트
ais_results.parquet → 지도/차트 → results/figures/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import logging
import time
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "results/figures"
DPI = 150


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    logger.info("Loading ais_results.parquet...")
    df = pd.read_parquet("data/ais_results.parquet")
    logger.info("Loaded: %d rows", len(df))
    logger.info("  Normal: %d  |  Anomaly: %d",
                (df["anomaly_final"] == 0).sum(), (df["anomaly_final"] == 1).sum())
    return df


# ═══════════════════════════════════════════════════════════
# 5.1  정상 vs 이상 항적 지도 (folium, 샘플링)
# ═══════════════════════════════════════════════════════════
def viz_trajectory_map(df):
    logger.info("5.1 Trajectory map (folium)...")
    center = [df["LAT"].mean(), df["LON"].mean()]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB dark_matter")

    # 정상: 5000개 샘플
    normal = df[df["anomaly_final"] == 0].sample(5000, random_state=42)
    for _, r in normal.iterrows():
        folium.CircleMarker([r["LAT"], r["LON"]], radius=1, color="blue",
                            fill=True, opacity=0.3).add_to(m)

    # 이상: 최대 5000개 샘플
    anomaly = df[df["anomaly_final"] == 1]
    if len(anomaly) > 5000:
        anomaly = anomaly.sample(5000, random_state=42)
    for _, r in anomaly.iterrows():
        folium.CircleMarker([r["LAT"], r["LON"]], radius=3, color="red",
                            fill=True, opacity=0.8).add_to(m)

    path = f"{OUT_DIR}/anomaly_map.html"
    m.save(path)
    logger.info("  → %s", path)


# ═══════════════════════════════════════════════════════════
# 5.2  이상 선박 Top 10 + Top 3 항적 (folium)
# ═══════════════════════════════════════════════════════════
def viz_top_vessels_folium(df):
    logger.info("5.2 Top anomaly vessels map (folium)...")
    anomaly_df = df[df["anomaly_final"] == 1]
    top10 = anomaly_df["MMSI"].value_counts().head(10)
    logger.info("  Top 10 anomaly vessels:\n%s", top10.to_string())

    colors = ["red", "orange", "yellow"]
    center = [df["LAT"].mean(), df["LON"].mean()]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB dark_matter")

    for i, mmsi in enumerate(top10.index[:3]):
        vessel = df[df["MMSI"] == mmsi].sort_values("BaseDateTime")
        coords = vessel[["LAT", "LON"]].values.tolist()
        folium.PolyLine(coords, color=colors[i], weight=2, opacity=0.7,
                        tooltip=f"MMSI: {mmsi}").add_to(m)
        anom = vessel[vessel["anomaly_final"] == 1]
        for _, r in anom.iterrows():
            folium.CircleMarker(
                [r["LAT"], r["LON"]], radius=5, color=colors[i],
                fill=True, opacity=0.9,
                tooltip=f'MMSI: {mmsi}<br>SOG: {r["SOG"]:.1f}<br>Time: {r["BaseDateTime"]}'
            ).add_to(m)

    path = f"{OUT_DIR}/top_anomaly_vessels.html"
    m.save(path)
    logger.info("  → %s", path)
    return top10


# ═══════════════════════════════════════════════════════════
# 5.3  정상 vs 이상 분포 바 차트
# ═══════════════════════════════════════════════════════════
def viz_distribution(df):
    logger.info("5.3 Anomaly distribution bar chart...")
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["anomaly_final"].value_counts().sort_index()
    labels = ["Normal", "Anomaly"]
    colors = ["#3498db", "#e74c3c"]
    bars = ax.bar(labels, counts.values, color=colors)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f"{val:,}", ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("Count")
    ax.set_title("Normal vs Anomaly Distribution")
    plt.tight_layout()
    path = f"{OUT_DIR}/anomaly_distribution.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


# ═══════════════════════════════════════════════════════════
# 5.4  모델별 비교 차트
# ═══════════════════════════════════════════════════════════
def viz_model_comparison(df):
    logger.info("5.4 Model comparison chart...")
    models = {
        "Isolation Forest": "anomaly_if",
        "LOF": "anomaly_lof",
        "HDBSCAN": "anomaly_hdbscan",
        "Ensemble (≥2/3)": "anomaly_final",
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(models.keys())
    counts = [int(df[c].sum()) for c in models.values()]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
    bars = ax.bar(names, counts, color=colors)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Anomaly Count")
    ax.set_title("Anomaly Detection — Model Comparison")
    plt.tight_layout()
    path = f"{OUT_DIR}/model_comparison.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


# ═══════════════════════════════════════════════════════════
# 5.5  이상 유형별 공간 분포
# ═══════════════════════════════════════════════════════════
def viz_anomaly_types_spatial(df):
    logger.info("5.5 Anomaly types spatial distribution...")
    anom = df[df["anomaly_final"] == 1]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    configs = [
        ("speed_deviation", lambda x: x.abs() > 3, "Speed Anomaly", "red"),
        ("course_change", lambda x: x > 90, "Course Change > 90°", "orange"),
        ("signal_gap_sec", lambda x: x > 3600, "Signal Gap > 1hr", "yellow"),
    ]
    for ax, (col, cond, title, color) in zip(axes, configs):
        subset = anom[cond(anom[col])] if col in anom.columns else pd.DataFrame()
        ax.scatter(subset["LON"], subset["LAT"], s=1, c=color, alpha=0.3)
        ax.set_title(f"{title} ({len(subset):,})")
        ax.set_xlabel("LON")
        ax.set_ylabel("LAT")

    plt.tight_layout()
    path = f"{OUT_DIR}/anomaly_types_spatial.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


# ═══════════════════════════════════════════════════════════
# 5.6  시간대별 이상 빈도
# ═══════════════════════════════════════════════════════════
def viz_hourly_rate(df):
    logger.info("5.6 Hourly anomaly rate...")
    df = df.copy()
    df["hour"] = df["BaseDateTime"].dt.hour

    hourly_total = df.groupby("hour").size()
    hourly_anom = df[df["anomaly_final"] == 1].groupby("hour").size()
    rate = (hourly_anom / hourly_total * 100).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(rate.index, rate.values, color="coral", alpha=0.8)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Anomaly Rate (%)")
    ax.set_title("Anomaly Rate by Hour")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    path = f"{OUT_DIR}/hourly_anomaly_rate.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


# ═══════════════════════════════════════════════════════════
# 5.7  정상 vs 이상 항적 (matplotlib + contextily)
# ═══════════════════════════════════════════════════════════
def viz_static_map(df):
    logger.info("5.7 Static map (matplotlib)...")
    try:
        import contextily as ctx
    except ImportError:
        logger.warning("  contextily not installed, skipping basemap")
        ctx = None

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    normal = df[df["anomaly_final"] == 0].sample(
        min(10000, (df["anomaly_final"] == 0).sum()), random_state=42)
    anomaly = df[df["anomaly_final"] == 1].sample(
        min(10000, (df["anomaly_final"] == 1).sum()), random_state=42)

    ax.scatter(normal["LON"], normal["LAT"], s=0.2, c="#3498db", alpha=0.2, label="Normal")
    ax.scatter(anomaly["LON"], anomaly["LAT"], s=1.5, c="#e74c3c", alpha=0.5, label="Anomaly")

    if ctx:
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.DarkMatter)
        except Exception as e:
            logger.warning("  basemap failed: %s", e)

    ax.set_title("Normal (blue) vs Anomaly (red)", color="white", fontsize=14)
    ax.set_xlabel("Longitude", color="#8b949e")
    ax.set_ylabel("Latitude", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.legend(loc="lower left", facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    path = f"{OUT_DIR}/anomaly_map_static.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info("  → %s", path)


# ═══════════════════════════════════════════════════════════
# 5.8  Top 3 선박 항적 (matplotlib + contextily)
# ═══════════════════════════════════════════════════════════
def viz_top_vessels_static(df):
    logger.info("5.8 Top 3 vessel trajectories (static)...")
    try:
        import contextily as ctx
    except ImportError:
        ctx = None

    anomaly_df = df[df["anomaly_final"] == 1]
    top3 = anomaly_df["MMSI"].value_counts().head(3).index
    colors_top = ["#ff6b6b", "#ffa94d", "#69db7c"]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    for i, mmsi in enumerate(top3):
        vessel = df[df["MMSI"] == mmsi].sort_values("BaseDateTime")
        ax.plot(vessel["LON"], vessel["LAT"], "-", color=colors_top[i],
                linewidth=1.5, alpha=0.8, label=f"MMSI: {mmsi}")
        anom = vessel[vessel["anomaly_final"] == 1]
        ax.scatter(anom["LON"], anom["LAT"], s=8, color=colors_top[i], zorder=5)

    if ctx:
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.DarkMatter)
        except Exception:
            pass

    ax.set_title("Top 3 Anomalous Vessel Trajectories", color="white", fontsize=14)
    ax.set_xlabel("Longitude", color="#8b949e")
    ax.set_ylabel("Latitude", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.legend(loc="lower left", facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=9)
    plt.tight_layout()
    path = f"{OUT_DIR}/top_vessels_map.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info("  → %s", path)


# ═══════════════════════════════════════════════════════════
# 5.9  선박 유형별 이상 비율
# ═══════════════════════════════════════════════════════════
def viz_vessel_type_rate(df):
    logger.info("5.9 Anomaly rate by vessel type...")
    VESSEL_TYPE_MAP = {
        30: "Fishing", 31: "Towing", 32: "Towing (large)",
        33: "Dredging", 34: "Diving ops", 35: "Military",
        36: "Sailing", 37: "Pleasure craft",
        40: "HSC", 50: "Pilot", 51: "SAR", 52: "Tug",
        60: "Passenger", 70: "Cargo", 80: "Tanker",
    }

    df = df.copy()
    df["VesselTypeGroup"] = df["VesselType"].apply(
        lambda x: VESSEL_TYPE_MAP.get(int(x // 10) * 10, "Other") if pd.notna(x) else "Unknown"
    )
    stats = df.groupby("VesselTypeGroup").agg(
        total=("anomaly_final", "count"),
        anomaly=("anomaly_final", "sum"),
    )
    stats["rate"] = stats["anomaly"] / stats["total"] * 100
    stats = stats[stats["total"] >= 100].sort_values("rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(stats.index, stats["rate"], color="coral", alpha=0.8)
    ax.set_xlabel("Anomaly Rate (%)")
    ax.set_title("Anomaly Rate by Vessel Type")
    for i, (idx, row) in enumerate(stats.iterrows()):
        ax.text(row["rate"] + 0.05, i, f'{row["rate"]:.1f}%', va="center")
    plt.tight_layout()
    path = f"{OUT_DIR}/anomaly_rate_by_type.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


# ═══════════════════════════════════════════════════════════
# 5.10  이상 1위 선박 딥다이브
# ═══════════════════════════════════════════════════════════
def viz_deep_dive(df):
    logger.info("5.10 Deep dive — top anomaly vessel...")
    try:
        import contextily as ctx
    except ImportError:
        ctx = None

    top_mmsi = df[df["anomaly_final"] == 1]["MMSI"].value_counts().index[0]
    vessel = df[df["MMSI"] == top_mmsi].sort_values("BaseDateTime").copy()
    vessel["minutes"] = (vessel["BaseDateTime"] - vessel["BaseDateTime"].min()).dt.total_seconds() / 60
    normal_v = vessel[vessel["anomaly_final"] == 0]
    anomaly_v = vessel[vessel["anomaly_final"] == 1]
    logger.info("  MMSI: %d  |  Total: %d  |  Anomaly: %d", top_mmsi, len(vessel), len(anomaly_v))

    # (a) 시계열
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"MMSI: {top_mmsi} — Deep Dive Analysis", fontsize=14)

    series = [
        ("SOG", "SOG (knots)", "#79c0ff"),
        ("COG", "COG (degrees)", "#7ee787"),
        ("course_change", "Course Change (deg)", "#ffa657"),
    ]
    for ax, (col, ylabel, color) in zip(axes, series):
        ax.plot(vessel["minutes"], vessel[col], "-", color=color, linewidth=0.8, label=col)
        ax.scatter(anomaly_v["minutes"], anomaly_v[col], s=20, c="red", zorder=5, label="Anomaly")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.2)
    if "course_change" in vessel.columns:
        axes[2].axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90°")
    axes[2].set_xlabel("Time (minutes from start)")
    plt.tight_layout()
    path = f"{OUT_DIR}/deep_dive_timeseries.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)

    # (b) 지도
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(vessel["LON"], vessel["LAT"], "-", color="cyan", linewidth=1, alpha=0.7, label="Normal")
    ax.scatter(anomaly_v["LON"], anomaly_v["LAT"], s=15, c="red", zorder=5, label="Anomaly")
    start, end = vessel.iloc[0], vessel.iloc[-1]
    ax.scatter(start["LON"], start["LAT"], s=80, c="green", marker="^", zorder=10, label="Start")
    ax.scatter(end["LON"], end["LAT"], s=80, c="red", marker="s", zorder=10, label="End")

    if ctx:
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.DarkMatter)
        except Exception:
            pass

    ax.set_title(f"MMSI: {top_mmsi} — Trajectory", color="white", fontsize=14)
    ax.set_xlabel("Longitude", color="#8b949e")
    ax.set_ylabel("Latitude", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.legend(loc="lower left", facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=9)
    plt.tight_layout()
    path = f"{OUT_DIR}/deep_dive_map.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info("  → %s", path)


# ═══════════════════════════════════════════════════════════
# 5.11  앙상블 벤 다이어그램 스타일 비교
# ═══════════════════════════════════════════════════════════
def viz_ensemble_overlap(df):
    logger.info("5.11 Model overlap analysis...")
    cols = ["anomaly_if", "anomaly_lof", "anomaly_hdbscan"]
    labels = ["IF", "LOF", "HDBSCAN"]

    # 조합별 카운트
    combos = {}
    for i in range(8):
        bits = [(i >> j) & 1 for j in range(3)]
        mask = np.ones(len(df), dtype=bool)
        key_parts = []
        for k, b in enumerate(bits):
            if b:
                mask &= (df[cols[k]] == 1).values
                key_parts.append(labels[k])
            else:
                mask &= (df[cols[k]] == 0).values
        key = " ∩ ".join(key_parts) if key_parts else "None"
        combos[key] = int(mask.sum())

    fig, ax = plt.subplots(figsize=(10, 5))
    items = [(k, v) for k, v in combos.items() if v > 0 and k != "None"]
    items.sort(key=lambda x: -x[1])
    names = [x[0] for x in items]
    vals = [x[1] for x in items]
    colors = plt.cm.Set2(np.linspace(0, 1, len(items)))
    bars = ax.barh(names, vals, color=colors)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=10)
    ax.set_xlabel("Count")
    ax.set_title("Model Agreement — Anomaly Overlap")
    plt.tight_layout()
    path = f"{OUT_DIR}/model_overlap.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", path)


# ═══════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    ensure_dirs()
    df = load_data()

    viz_distribution(df)
    viz_model_comparison(df)
    viz_ensemble_overlap(df)
    viz_anomaly_types_spatial(df)
    viz_hourly_rate(df)
    viz_static_map(df)
    viz_top_vessels_static(df)
    viz_vessel_type_rate(df)
    viz_deep_dive(df)
    viz_trajectory_map(df)
    viz_top_vessels_folium(df)

    logger.info("=" * 60)
    logger.info("All visualizations saved to %s/", OUT_DIR)
    logger.info("Total elapsed: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
