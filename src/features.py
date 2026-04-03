"""피처 엔지니어링 모듈"""

import pandas as pd
import numpy as np


def calc_speed_deviation(df: pd.DataFrame, vessel_col: str = "MMSI", speed_col: str = "SOG") -> pd.DataFrame:
    """선박별 평균 속도 대비 편차를 계산한다."""
    df = df.copy()
    vessel_stats = df.groupby(vessel_col)[speed_col].agg(["mean", "std"]).reset_index()
    vessel_stats.columns = [vessel_col, "speed_mean", "speed_std"]
    df = df.merge(vessel_stats, on=vessel_col, how="left")
    df["speed_deviation"] = (df[speed_col] - df["speed_mean"]) / df["speed_std"].replace(0, 1)
    return df


def calc_course_change(df: pd.DataFrame, vessel_col: str = "MMSI", course_col: str = "COG") -> pd.DataFrame:
    """연속 레코드 간 침로 변화량을 계산한다."""
    df = df.copy()
    df["course_change"] = df.groupby(vessel_col)[course_col].diff().abs()
    df["course_change"] = df["course_change"].apply(lambda x: min(x, 360 - x) if pd.notna(x) else x)
    return df


def calc_signal_gap(df: pd.DataFrame, vessel_col: str = "MMSI", time_col: str = "BaseDateTime") -> pd.DataFrame:
    """AIS 신호 간 시간 간격(초)을 계산한다."""
    df = df.copy()
    df["signal_gap_sec"] = df.groupby(vessel_col)[time_col].diff().dt.total_seconds()
    return df


def calc_night_activity(df: pd.DataFrame, time_col: str = "BaseDateTime") -> pd.DataFrame:
    """야간 활동 여부를 계산한다 (22시~06시)."""
    df = df.copy()
    hour = df[time_col].dt.hour
    df["is_night"] = ((hour >= 22) | (hour < 6)).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """전체 피처 엔지니어링 파이프라인을 실행한다."""
    df = calc_speed_deviation(df)
    df = calc_course_change(df)
    df = calc_signal_gap(df)
    df = calc_night_activity(df)
    return df
