"""지도 기반 시각화 재생성 — cartopy 해안선 포함, 선박별 자동 줌"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()
OUT = "results/figures"
DPI = 150


def make_map_ax(fig, extent=None, facecolor="#0d1117", subplot_spec=None):
    """cartopy 기반 지도 축 생성 (해안선 + 국경선 포함)"""
    if subplot_spec:
        ax = fig.add_subplot(*subplot_spec, projection=ccrs.PlateCarree(), facecolor=facecolor)
    else:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(), facecolor=facecolor)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#1a1a2e", edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#0d1117")
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.5, edgecolor="#444444")
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.4, edgecolor="#1a3a5c")
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#0d1117", edgecolor="#444444", linewidth=0.3)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.3, edgecolor="#333333", linestyle="--")
    ax.add_feature(cfeature.STATES.with_scale("10m"), linewidth=0.2, edgecolor="#2a2a2a")
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#333333", alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"color": "#8b949e", "fontsize": 9}
    return ax


def vessel_extent(vessel_df, pad=0.5):
    """선박 데이터 범위 + 여유(pad) 계산. 최소 1도 범위 보장."""
    lon_min, lon_max = vessel_df["LON"].min(), vessel_df["LON"].max()
    lat_min, lat_max = vessel_df["LAT"].min(), vessel_df["LAT"].max()
    # 최소 1도 범위 보장
    if lon_max - lon_min < 1:
        mid = (lon_min + lon_max) / 2
        lon_min, lon_max = mid - 0.5, mid + 0.5
    if lat_max - lat_min < 1:
        mid = (lat_min + lat_max) / 2
        lat_min, lat_max = mid - 0.5, mid + 0.5
    return [lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad]


logger.info("Loading data...")
df = pd.read_parquet("data/ais_results.parquet")
logger.info("Rows: %d", len(df))

# 전체 지도 범위
lon_min, lon_max = df["LON"].quantile(0.01) - 2, df["LON"].quantile(0.99) + 2
lat_min, lat_max = df["LAT"].quantile(0.01) - 2, df["LAT"].quantile(0.99) + 2
full_extent = [lon_min, lon_max, lat_min, lat_max]

# ═══════════════════════════════════════════════════════════
# 1. traffic_map — 전체 항적
# ═══════════════════════════════════════════════════════════
logger.info("1. traffic_map.png")
fig = plt.figure(figsize=(14, 9), facecolor="#0d1117")
ax = make_map_ax(fig, full_extent)
sample = df.sample(min(80000, len(df)), random_state=42)
ax.scatter(sample["LON"], sample["LAT"], s=0.3, c="#3498db", alpha=0.3,
           transform=ccrs.PlateCarree())
ax.set_title("AIS Traffic Map — 500 vessels, 7 days", color="white", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig(f"{OUT}/traffic_map.png", dpi=DPI, bbox_inches="tight", facecolor="#0d1117")
plt.close()

# ═══════════════════════════════════════════════════════════
# 2. anomaly_map_static — 정상 vs 이상
# ═══════════════════════════════════════════════════════════
logger.info("2. anomaly_map_static.png")
fig = plt.figure(figsize=(14, 9), facecolor="#0d1117")
ax = make_map_ax(fig, full_extent)

normal = df[df["anomaly_final"] == 0].sample(
    min(15000, (df["anomaly_final"] == 0).sum()), random_state=42)
anomaly = df[df["anomaly_final"] == 1].sample(
    min(15000, (df["anomaly_final"] == 1).sum()), random_state=42)

ax.scatter(normal["LON"], normal["LAT"], s=0.3, c="#3498db", alpha=0.2,
           label="Normal", transform=ccrs.PlateCarree())
ax.scatter(anomaly["LON"], anomaly["LAT"], s=2, c="#e74c3c", alpha=0.5,
           label="Anomaly", transform=ccrs.PlateCarree())
ax.set_title("Normal (blue) vs Anomaly (red)", color="white", fontsize=14, pad=12)
ax.legend(loc="lower left", facecolor="#161b22", edgecolor="#30363d",
          labelcolor="white", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/anomaly_map_static.png", dpi=DPI, bbox_inches="tight", facecolor="#0d1117")
plt.close()

# ═══════════════════════════════════════════════════════════
# 3. top_vessels_map — 이상 상위 3척 (각 선박 줌인 서브플롯)
# ═══════════════════════════════════════════════════════════
logger.info("3. top_vessels_map.png")
anomaly_df = df[df["anomaly_final"] == 1]
top3 = anomaly_df["MMSI"].value_counts().head(3).index
top3_counts = anomaly_df["MMSI"].value_counts().head(3).values
colors_top = ["#ff6b6b", "#ffa94d", "#69db7c"]

fig = plt.figure(figsize=(18, 6), facecolor="#0d1117")

for i, (mmsi, count) in enumerate(zip(top3, top3_counts)):
    vessel = df[df["MMSI"] == mmsi].sort_values("BaseDateTime")
    anom = vessel[vessel["anomaly_final"] == 1]
    ext = vessel_extent(vessel, pad=0.3)

    ax = make_map_ax(fig, ext, subplot_spec=(1, 3, i + 1))
    ax.plot(vessel["LON"], vessel["LAT"], "-", color=colors_top[i],
            linewidth=1.5, alpha=0.6, transform=ccrs.PlateCarree())
    ax.scatter(anom["LON"], anom["LAT"], s=8, color=colors_top[i],
               zorder=5, transform=ccrs.PlateCarree())
    # 출발/도착
    ax.scatter(vessel.iloc[0]["LON"], vessel.iloc[0]["LAT"], s=60, c="green",
               marker="^", zorder=10, transform=ccrs.PlateCarree())
    ax.scatter(vessel.iloc[-1]["LON"], vessel.iloc[-1]["LAT"], s=60, c="white",
               marker="s", zorder=10, transform=ccrs.PlateCarree())
    ax.set_title(f"MMSI: {mmsi}\n({count} anomalies)", color="white", fontsize=11, pad=8)

fig.suptitle("Top 3 Anomalous Vessel Trajectories", color="white", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/top_vessels_map.png", dpi=DPI, bbox_inches="tight", facecolor="#0d1117")
plt.close()

# ═══════════════════════════════════════════════════════════
# 4. deep_dive_map — 이상 1위 선박 (줌인)
# ═══════════════════════════════════════════════════════════
logger.info("4. deep_dive_map.png")
top_mmsi = anomaly_df["MMSI"].value_counts().index[0]
vessel = df[df["MMSI"] == top_mmsi].sort_values("BaseDateTime")
anomaly_v = vessel[vessel["anomaly_final"] == 1]
ext = vessel_extent(vessel, pad=0.2)

fig = plt.figure(figsize=(10, 10), facecolor="#0d1117")
ax = make_map_ax(fig, ext)

ax.plot(vessel["LON"], vessel["LAT"], "-", color="cyan", linewidth=1.2,
        alpha=0.7, label="Track", transform=ccrs.PlateCarree())
ax.scatter(anomaly_v["LON"], anomaly_v["LAT"], s=12, c="red",
           zorder=5, label="Anomaly", transform=ccrs.PlateCarree())

start, end = vessel.iloc[0], vessel.iloc[-1]
ax.scatter(start["LON"], start["LAT"], s=100, c="green", marker="^",
           zorder=10, label="Start", transform=ccrs.PlateCarree())
ax.scatter(end["LON"], end["LAT"], s=100, c="red", marker="s",
           zorder=10, label="End", transform=ccrs.PlateCarree())

n_anom = len(anomaly_v)
n_total = len(vessel)
ax.set_title(f"MMSI: {top_mmsi} — {n_total:,} records, {n_anom:,} anomalies",
             color="white", fontsize=14, pad=12)
ax.legend(loc="lower left", facecolor="#161b22", edgecolor="#30363d",
          labelcolor="white", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/deep_dive_map.png", dpi=DPI, bbox_inches="tight", facecolor="#0d1117")
plt.close()

logger.info("All map plots regenerated with coastlines!")
