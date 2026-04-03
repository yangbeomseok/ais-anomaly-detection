"""AIS 데이터 전처리 모듈"""

import pandas as pd
import numpy as np


def clean_coordinates(df: pd.DataFrame, lat_col: str = "LAT", lon_col: str = "LON") -> pd.DataFrame:
    """유효하지 않은 좌표를 제거한다."""
    mask = (
        (df[lat_col].between(-90, 90)) &
        (df[lon_col].between(-180, 180)) &
        (df[lat_col] != 0) &
        (df[lon_col] != 0)
    )
    return df[mask].copy()


def clean_speed(df: pd.DataFrame, speed_col: str = "SOG", max_speed: float = 50.0) -> pd.DataFrame:
    """비정상 속도를 제거한다 (음수, 과도한 속도)."""
    mask = df[speed_col].between(0, max_speed)
    return df[mask].copy()


def parse_timestamp(df: pd.DataFrame, time_col: str = "BaseDateTime") -> pd.DataFrame:
    """타임스탬프를 datetime으로 변환한다."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df.dropna(subset=[time_col])


def filter_korea_eez(df: pd.DataFrame, lat_col: str = "LAT", lon_col: str = "LON") -> pd.DataFrame:
    """한국 EEZ 근처 데이터만 필터링한다."""
    mask = (
        df[lat_col].between(32.0, 40.0) &
        df[lon_col].between(124.0, 132.0)
    )
    return df[mask].copy()


def sort_by_vessel_time(df: pd.DataFrame, vessel_col: str = "MMSI", time_col: str = "BaseDateTime") -> pd.DataFrame:
    """선박별 시간순 정렬한다."""
    return df.sort_values([vessel_col, time_col]).reset_index(drop=True)
