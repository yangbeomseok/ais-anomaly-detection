"""피처 엔지니어링 모듈"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"MMSI", "BaseDateTime", "SOG", "COG", "Heading"}


def _validate_columns(df: pd.DataFrame, required: set) -> None:
    """필수 컬럼 존재 여부를 검증한다."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame에 필요한 컬럼이 없습니다: {missing}")


def calc_speed_deviation(df: pd.DataFrame, vessel_col: str = "MMSI", speed_col: str = "SOG") -> pd.DataFrame:
    """선박별 평균 속도 대비 편차를 계산한다."""
    df = df.copy()
    vessel_stats = df.groupby(vessel_col)[speed_col].agg(["mean", "std"]).reset_index()
    vessel_stats.columns = [vessel_col, "speed_mean", "speed_std"]
    df = df.merge(vessel_stats, on=vessel_col, how="left")
    df["speed_deviation"] = (df[speed_col] - df["speed_mean"]) / df["speed_std"].replace(0, 1)
    return df


def calc_acceleration(df: pd.DataFrame, vessel_col: str = "MMSI",
                      speed_col: str = "SOG", time_col: str = "BaseDateTime") -> pd.DataFrame:
    """선박별 가속도(knots/sec)를 계산한다. 급가속/급감속 탐지용."""
    df = df.copy()
    dt = df.groupby(vessel_col)[time_col].diff().dt.total_seconds()
    ds = df.groupby(vessel_col)[speed_col].diff()
    df["acceleration"] = (ds / dt.replace(0, np.nan)).astype(float)
    return df


def calc_course_change(df: pd.DataFrame, vessel_col: str = "MMSI", course_col: str = "COG") -> pd.DataFrame:
    """연속 레코드 간 침로 변화량을 계산한다."""
    df = df.copy()
    df["course_change"] = df.groupby(vessel_col)[course_col].diff().abs()
    df["course_change"] = df["course_change"].apply(lambda x: min(x, 360 - x) if pd.notna(x) else x)
    return df


def calc_heading_cog_diff(df: pd.DataFrame, heading_col: str = "Heading",
                          course_col: str = "COG") -> pd.DataFrame:
    """선수방향(Heading)과 대지침로(COG) 간 각도 차이를 계산한다.
    큰 차이는 조류/바람 영향 또는 비정상 조종을 의미한다.
    AIS에서 Heading=511은 사용 불가이므로 NaN 처리한다."""
    df = df.copy()
    heading = df[heading_col].replace(511.0, np.nan)
    diff = (heading - df[course_col]).abs()
    df["heading_cog_diff"] = diff.where(diff <= 180, 360 - diff)
    return df


def calc_signal_gap(df: pd.DataFrame, vessel_col: str = "MMSI", time_col: str = "BaseDateTime") -> pd.DataFrame:
    """AIS 신호 간 시간 간격(초)을 계산한다."""
    df = df.copy()
    df["signal_gap_sec"] = df.groupby(vessel_col)[time_col].diff().dt.total_seconds()
    return df


def calc_stop_duration(df: pd.DataFrame, vessel_col: str = "MMSI",
                       speed_col: str = "SOG", time_col: str = "BaseDateTime",
                       threshold: float = 0.5) -> pd.DataFrame:
    """선박별 정박/저속 지속 시간(분)을 계산한다.
    SOG < threshold인 연속 구간의 누적 시간을 측정한다."""
    df = df.copy()
    df["is_stopped"] = (df[speed_col] < threshold).astype(int)
    df["stop_group"] = df.groupby(vessel_col)["is_stopped"].transform(
        lambda s: (s != s.shift()).cumsum()
    )
    df["stop_duration_min"] = df.groupby([vessel_col, "stop_group"])[time_col].transform(
        lambda t: (t - t.iloc[0]).dt.total_seconds() / 60
    )
    df.loc[df["is_stopped"] == 0, "stop_duration_min"] = 0.0
    df.drop(columns=["is_stopped", "stop_group"], inplace=True)
    return df


def calc_night_activity(df: pd.DataFrame, time_col: str = "BaseDateTime") -> pd.DataFrame:
    """야간 활동 여부를 계산한다 (22시~06시)."""
    df = df.copy()
    hour = df[time_col].dt.hour
    df["is_night"] = ((hour >= 22) | (hour < 6)).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """전체 피처 엔지니어링 파이프라인을 실행한다."""
    _validate_columns(df, _REQUIRED_COLS)
    logger.info("피처 생성 시작: %d rows", len(df))
    df = calc_speed_deviation(df)
    df = calc_acceleration(df)
    df = calc_course_change(df)
    df = calc_heading_cog_diff(df)
    df = calc_signal_gap(df)
    df = calc_stop_duration(df)
    df = calc_night_activity(df)
    logger.info("피처 생성 완료: 7개 피처 추가됨")
    return df
