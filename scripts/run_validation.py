"""
AIS Anomaly Detection - 도메인 기반 검증 스크립트

라벨 없는 비지도 학습 결과를 해양 도메인 규칙으로 검증한다.
알려진 이상 패턴과 모델 탐지 결과의 일치도를 측정한다.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUT_DIR = "results/figures"


def define_domain_rules(df: pd.DataFrame) -> dict:
    """해양 도메인 기반 이상 규칙을 정의하고 각 규칙에 해당하는 행을 반환한다."""

    rules = {}

    # 1. Dark Shipping: AIS 신호 1시간 이상 단절
    rules["Dark Shipping\n(signal gap > 1hr)"] = (
        df["signal_gap_sec"] > 3600
    )

    # 2. 급격한 방향 전환: 90도 이상 침로 변화
    rules["Erratic Movement\n(course change > 90°)"] = (
        df["course_change"] > 90
    )

    # 3. 속도 이상: 선박별 평균 대비 3σ 초과
    rules["Speed Anomaly\n(speed deviation > 3σ)"] = (
        df["speed_deviation"].abs() > 3
    )

    # 4. 선수-침로 괴리: 45도 이상 차이 (비정상 조종)
    rules["Heading-COG Mismatch\n(diff > 45°)"] = (
        df["heading_cog_diff"] > 45
    )

    # 5. 의심 정박: 비항구 해역에서 장시간 정박 (2시간 이상)
    rules["Suspicious Anchoring\n(stop > 2hrs)"] = (
        df["stop_duration_min"] > 120
    )

    # 6. 야간 고속 운항: 야간에 15노트 이상
    rules["Night High-Speed\n(night + SOG > 15kt)"] = (
        (df["is_night"] == 1) & (df["SOG"] > 15)
    )

    return rules


def compute_overlap(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """각 도메인 규칙과 모델 탐지 결과의 일치도를 계산한다."""
    results = []

    for rule_name, rule_mask in rules.items():
        n_rule = int(rule_mask.sum())
        if n_rule == 0:
            continue

        # 규칙에 해당하는 행 중 모델이 이상으로 탐지한 비율 (Recall)
        detected = int((rule_mask & (df["anomaly_final"] == 1)).sum())
        recall = detected / n_rule * 100

        # 모델별 탐지율
        recall_if = int((rule_mask & (df["anomaly_if"] == 1)).sum()) / n_rule * 100
        recall_lof = int((rule_mask & (df["anomaly_lof"] == 1)).sum()) / n_rule * 100
        recall_hdb = int((rule_mask & (df["anomaly_hdbscan"] == 1)).sum()) / n_rule * 100

        results.append({
            "Rule": rule_name,
            "Rule Count": n_rule,
            "Detected (Ensemble)": detected,
            "Recall (%)": round(recall, 1),
            "IF Recall (%)": round(recall_if, 1),
            "LOF Recall (%)": round(recall_lof, 1),
            "HDBSCAN Recall (%)": round(recall_hdb, 1),
        })

    return pd.DataFrame(results)


def plot_validation_recall(validation_df: pd.DataFrame, save_path: str):
    """도메인 규칙별 탐지율(recall) 차트를 그린다."""
    fig, ax = plt.subplots(figsize=(12, 6))

    rules = validation_df["Rule"].values
    x = np.arange(len(rules))
    width = 0.2

    ax.bar(x - 1.5 * width, validation_df["IF Recall (%)"],
           width, label="IF", color="#3498db", alpha=0.8)
    ax.bar(x - 0.5 * width, validation_df["LOF Recall (%)"],
           width, label="LOF", color="#2ecc71", alpha=0.8)
    ax.bar(x + 0.5 * width, validation_df["HDBSCAN Recall (%)"],
           width, label="HDBSCAN", color="#e67e22", alpha=0.8)
    ax.bar(x + 1.5 * width, validation_df["Recall (%)"],
           width, label="Ensemble", color="#e74c3c", alpha=0.8)

    ax.set_xlabel("Domain Rule")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Domain Validation — How well do models catch known anomaly patterns?")
    ax.set_xticks(x)
    ax.set_xticklabels(rules, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 105)

    # 앙상블 recall 값 표시
    for i, v in enumerate(validation_df["Recall (%)"].values):
        ax.text(i + 1.5 * width, v + 1, f"{v:.0f}%", ha="center", fontsize=9,
                fontweight="bold", color="#e74c3c")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", save_path)


def plot_validation_summary(df: pd.DataFrame, rules: dict, save_path: str):
    """앙상블 이상 중 도메인 규칙에 해당하는 비율 (설명 가능성)."""
    anomaly_df = df[df["anomaly_final"] == 1]
    n_anomaly = len(anomaly_df)

    any_rule = pd.Series(False, index=df.index)
    rule_counts = {}
    for rule_name, rule_mask in rules.items():
        any_rule |= rule_mask
        short_name = rule_name.split("\n")[0]
        rule_counts[short_name] = int((rule_mask & (df["anomaly_final"] == 1)).sum())

    # 도메인 규칙 하나도 해당 안 되는 이상
    n_unexplained = int((~any_rule & (df["anomaly_final"] == 1)).sum())
    rule_counts["Unexplained"] = n_unexplained

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # (a) 도메인 규칙별 이상 구성
    labels = list(rule_counts.keys())
    values = list(rule_counts.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    colors[-1] = [0.8, 0.8, 0.8, 1.0]  # Unexplained은 회색

    bars = ax1.barh(labels, values, color=colors)
    for bar, val in zip(bars, values):
        pct = val / n_anomaly * 100
        ax1.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                 f"{val:,} ({pct:.1f}%)", va="center", fontsize=10)
    ax1.set_xlabel("Count")
    ax1.set_title(f"Ensemble Anomalies by Domain Rule (total: {n_anomaly:,})")

    # (b) 설명 가능 비율 파이 차트
    n_explained = n_anomaly - n_unexplained
    ax2.pie([n_explained, n_unexplained],
            labels=[f"Explained\n{n_explained:,} ({n_explained/n_anomaly*100:.1f}%)",
                    f"Unexplained\n{n_unexplained:,} ({n_unexplained/n_anomaly*100:.1f}%)"],
            colors=["#2ecc71", "#bdc3c7"],
            startangle=90, textprops={"fontsize": 12})
    ax2.set_title("Domain Explainability of Ensemble Anomalies")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", save_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info("Loading ais_results.parquet...")
    df = pd.read_parquet("data/ais_results.parquet")
    logger.info("Loaded: %d rows", len(df))
    logger.info("  Anomaly (ensemble): %d (%.2f%%)",
                (df["anomaly_final"] == 1).sum(),
                (df["anomaly_final"] == 1).mean() * 100)

    # 도메인 규칙 정의
    logger.info("Defining domain rules...")
    rules = define_domain_rules(df)

    # 일치도 계산
    logger.info("Computing overlap...")
    validation_df = compute_overlap(df, rules)
    logger.info("\n%s", validation_df.to_string(index=False))

    # 시각화
    logger.info("Generating validation plots...")
    plot_validation_recall(validation_df, f"{OUT_DIR}/domain_validation_recall.png")
    plot_validation_summary(df, rules, f"{OUT_DIR}/domain_validation_summary.png")

    # 텍스트 리포트 저장
    report_path = "results/validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("AIS Anomaly Detection — Domain Validation Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"Ensemble anomalies: {(df['anomaly_final']==1).sum():,} "
                f"({(df['anomaly_final']==1).mean()*100:.2f}%)\n\n")
        f.write(validation_df.to_string(index=False))
        f.write("\n\n")

        # 설명 가능 비율
        any_rule = pd.Series(False, index=df.index)
        for mask in rules.values():
            any_rule |= mask
        anomaly_mask = df["anomaly_final"] == 1
        n_explained = int((any_rule & anomaly_mask).sum())
        n_total_anom = int(anomaly_mask.sum())
        f.write(f"Explained by domain rules: {n_explained:,} / {n_total_anom:,} "
                f"({n_explained/n_total_anom*100:.1f}%)\n")
        f.write(f"Unexplained: {n_total_anom - n_explained:,} "
                f"({(n_total_anom - n_explained)/n_total_anom*100:.1f}%)\n")

    logger.info("  → %s", report_path)
    logger.info("Validation complete!")


if __name__ == "__main__":
    main()
