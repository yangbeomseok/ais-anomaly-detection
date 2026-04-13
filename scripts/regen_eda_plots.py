"""Stale EDA plots (Apr 9) 재생성 스크립트 — 현재 데이터 기반"""
import matplotlib
matplotlib.use("Agg")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()
OUT = "results/figures"
DPI = 150

FEATURE_COLS = [
    "SOG", "speed_deviation", "acceleration", "course_change",
    "heading_cog_diff", "signal_gap_sec", "stop_duration_min", "is_night",
]
VESSEL_TYPE_MAP = {
    30: "Fishing", 31: "Towing", 32: "Towing(L)", 33: "Dredging",
    34: "Diving", 35: "Military", 36: "Sailing", 37: "Pleasure",
    40: "HSC", 50: "Pilot", 51: "SAR", 52: "Tug",
    60: "Passenger", 70: "Cargo", 80: "Tanker",
}

logger.info("Loading data...")
df = pd.read_parquet("data/ais_results.parquet")
logger.info("Rows: %d, Vessels: %d", len(df), df["MMSI"].nunique())

# 1. vessel_type_distribution
df["VTG"] = df["VesselType"].apply(
    lambda x: VESSEL_TYPE_MAP.get(int(x // 10) * 10, "Other") if pd.notna(x) else "Unknown"
)
vt = df.drop_duplicates("MMSI")["VTG"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
vt.plot.bar(ax=ax, color="steelblue", alpha=0.8)
ax.set_title(f"Vessel Type Distribution (n={df['MMSI'].nunique()})")
ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/vessel_type_distribution.png", dpi=DPI, bbox_inches="tight")
plt.close()
logger.info("vessel_type_distribution.png")

# 2. speed_distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["SOG"].clip(0, 30), bins=100, color="steelblue", alpha=0.7, edgecolor="none")
ax.set_title("Speed Over Ground Distribution")
ax.set_xlabel("SOG (knots)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT}/speed_distribution.png", dpi=DPI, bbox_inches="tight")
plt.close()
logger.info("speed_distribution.png")

# 3. cog_distribution
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
cog_rad = np.deg2rad(df["COG"].dropna().sample(min(50000, len(df)), random_state=42))
ax.hist(cog_rad, bins=72, color="steelblue", alpha=0.7)
ax.set_title("COG Distribution", pad=20)
plt.tight_layout()
plt.savefig(f"{OUT}/cog_distribution.png", dpi=DPI, bbox_inches="tight")
plt.close()
logger.info("cog_distribution.png")

# 4. records_per_vessel
recs = df.groupby("MMSI").size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(recs)), recs.values, color="steelblue", alpha=0.7, width=1.0)
ax.set_title(f"Records per Vessel (n={len(recs)})")
ax.set_xlabel("Vessel (sorted)")
ax.set_ylabel("Records")
plt.tight_layout()
plt.savefig(f"{OUT}/records_per_vessel.png", dpi=DPI, bbox_inches="tight")
plt.close()
logger.info("records_per_vessel.png")

# 5. hourly_traffic
df["hour"] = df["BaseDateTime"].dt.hour
hourly = df.groupby("hour").size()
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(hourly.index, hourly.values, color="steelblue", alpha=0.8)
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Record Count")
ax.set_title("Hourly AIS Traffic")
ax.set_xticks(range(24))
plt.tight_layout()
plt.savefig(f"{OUT}/hourly_traffic.png", dpi=DPI, bbox_inches="tight")
plt.close()
logger.info("hourly_traffic.png")

# 6. feature_distributions
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, col in zip(axes.flat, FEATURE_COLS):
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    data = df[col].dropna().clip(lo, hi)
    ax.hist(data.sample(min(100000, len(data)), random_state=42), bins=80,
            color="steelblue", alpha=0.7, edgecolor="none")
    ax.set_title(col)
    ax.set_ylabel("Count")
plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/feature_distributions.png", dpi=DPI, bbox_inches="tight")
plt.close()
logger.info("feature_distributions.png")

# 7. feature_correlation
corr = df[FEATURE_COLS].sample(min(100000, len(df)), random_state=42).corr()
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Feature Correlation")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_correlation.png", dpi=DPI, bbox_inches="tight")
plt.close()
logger.info("feature_correlation.png")

# 8. normal_vs_anomaly
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
n_norm = (df["anomaly_final"] == 0).sum()
n_anom = (df["anomaly_final"] == 1).sum()
normal = df[df["anomaly_final"] == 0].sample(min(50000, n_norm), random_state=42)
anomaly = df[df["anomaly_final"] == 1].sample(min(50000, n_anom), random_state=42)
for ax, col in zip(axes.flat, FEATURE_COLS):
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    bins = np.linspace(lo, hi, 60)
    ax.hist(normal[col].clip(lo, hi), bins=bins, alpha=0.5, color="#3498db",
            label="Normal", density=True)
    ax.hist(anomaly[col].clip(lo, hi), bins=bins, alpha=0.5, color="#e74c3c",
            label="Anomaly", density=True)
    ax.set_title(col)
    ax.legend(fontsize=8)
plt.suptitle("Normal vs Anomaly Feature Distributions", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/normal_vs_anomaly.png", dpi=DPI, bbox_inches="tight")
plt.close()
logger.info("normal_vs_anomaly.png")

# 9. feature_importance
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

X = df[FEATURE_COLS].sample(50000, random_state=42)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
mdl = IsolationForest(random_state=42, n_jobs=-1)
mdl.fit(X_sc)
scores = mdl.score_samples(X_sc)
importance = [abs(np.corrcoef(X_sc[:, i], scores)[0, 1]) for i in range(len(FEATURE_COLS))]
imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importance}).sort_values(
    "importance", ascending=True
)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(imp_df["feature"], imp_df["importance"], color="coral", alpha=0.8)
ax.set_xlabel("|Correlation with IF Score|")
ax.set_title("Feature Importance (IF Score Correlation)")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=DPI, bbox_inches="tight")
plt.close()
logger.info("feature_importance.png")

# 10. sensitivity_analysis
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import HDBSCAN as HDBSCAN_

contams = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
X_sens = X_sc[:20000]
results_if, results_lof = [], []
for c in contams:
    m1 = IsolationForest(contamination=c, random_state=42, n_jobs=-1)
    results_if.append((m1.fit_predict(X_sens) == -1).sum())
    m2 = LocalOutlierFactor(contamination=c, n_neighbors=30, n_jobs=-1)
    results_lof.append((m2.fit_predict(X_sens) == -1).sum())
hdb = HDBSCAN_(min_cluster_size=50, min_samples=10)
hdb_count = (hdb.fit_predict(X_sens) == -1).sum()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot([c * 100 for c in contams], [r / len(X_sens) * 100 for r in results_if],
        "o-", label="IF", color="#3498db")
ax.plot([c * 100 for c in contams], [r / len(X_sens) * 100 for r in results_lof],
        "s-", label="LOF", color="#2ecc71")
ax.axhline(hdb_count / len(X_sens) * 100, color="#e67e22", linestyle="--",
           label=f"HDBSCAN ({hdb_count / len(X_sens) * 100:.1f}%)")
ax.set_xlabel("Contamination (%)")
ax.set_ylabel("Anomaly Rate (%)")
ax.set_title("Sensitivity Analysis")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/sensitivity_analysis.png", dpi=DPI, bbox_inches="tight")
plt.close()
logger.info("sensitivity_analysis.png")

# 11. traffic_map
fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0d1117")
ax.set_facecolor("#0d1117")
sample = df.sample(min(50000, len(df)), random_state=42)
ax.scatter(sample["LON"], sample["LAT"], s=0.3, c="#3498db", alpha=0.3)
ax.set_title("AIS Traffic Map (500 vessels)", color="white", fontsize=14)
ax.set_xlabel("Longitude", color="#8b949e")
ax.set_ylabel("Latitude", color="#8b949e")
ax.tick_params(colors="#8b949e")
plt.tight_layout()
plt.savefig(f"{OUT}/traffic_map.png", dpi=DPI, bbox_inches="tight", facecolor="#0d1117")
plt.close()
logger.info("traffic_map.png")

logger.info("All stale plots regenerated!")
