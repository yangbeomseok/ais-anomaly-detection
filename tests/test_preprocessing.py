"""전처리 모듈 테스트"""

import pandas as pd
import numpy as np
import pytest

from src.preprocessing import (
    clean_coordinates, clean_speed, parse_timestamp, sort_by_vessel_time
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "MMSI": [1, 1, 2, 2, 2],
        "BaseDateTime": [
            "2022-01-01 00:00:00", "2022-01-01 00:01:00",
            "2022-01-01 00:00:00", "2022-01-01 00:02:00", "2022-01-01 00:01:00",
        ],
        "LAT": [37.0, 0.0, -91.0, 36.5, 36.0],
        "LON": [-122.0, 0.0, 130.0, -121.0, -120.0],
        "SOG": [5.0, 10.0, -1.0, 60.0, 3.0],
    })


def test_clean_coordinates_removes_zero(sample_df):
    result = clean_coordinates(sample_df)
    assert (result["LAT"] == 0).sum() == 0
    assert (result["LON"] == 0).sum() == 0


def test_clean_coordinates_removes_out_of_range(sample_df):
    result = clean_coordinates(sample_df)
    assert result["LAT"].between(-90, 90).all()


def test_clean_speed_removes_negative(sample_df):
    result = clean_speed(sample_df)
    assert (result["SOG"] < 0).sum() == 0


def test_clean_speed_removes_above_max(sample_df):
    result = clean_speed(sample_df, max_speed=50.0)
    assert (result["SOG"] > 50.0).sum() == 0


def test_clean_speed_custom_max():
    df = pd.DataFrame({"SOG": [1.0, 10.0, 100.0, 200.0]})
    result = clean_speed(df, max_speed=150.0)
    assert len(result) == 3


def test_parse_timestamp(sample_df):
    result = parse_timestamp(sample_df)
    assert pd.api.types.is_datetime64_any_dtype(result["BaseDateTime"])


def test_parse_timestamp_drops_invalid():
    df = pd.DataFrame({"BaseDateTime": ["2022-01-01", "not-a-date", None]})
    result = parse_timestamp(df)
    assert len(result) == 1


def test_sort_by_vessel_time(sample_df):
    sample_df = parse_timestamp(sample_df)
    result = sort_by_vessel_time(sample_df)
    for mmsi in result["MMSI"].unique():
        vessel = result[result["MMSI"] == mmsi]
        assert vessel["BaseDateTime"].is_monotonic_increasing


def test_clean_coordinates_preserves_valid():
    df = pd.DataFrame({
        "LAT": [37.5, 40.0, -33.8],
        "LON": [-122.4, -74.0, 151.2],
    })
    result = clean_coordinates(df)
    assert len(result) == 3


def test_clean_speed_empty_result():
    df = pd.DataFrame({"SOG": [-5.0, -1.0]})
    result = clean_speed(df)
    assert len(result) == 0
