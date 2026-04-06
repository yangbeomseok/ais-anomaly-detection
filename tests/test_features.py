"""피처 엔지니어링 모듈 테스트"""

import pandas as pd
import numpy as np
import pytest

from src.features import (
    calc_speed_deviation, calc_acceleration, calc_course_change,
    calc_heading_cog_diff, calc_signal_gap, calc_stop_duration,
    calc_night_activity, build_features,
)


@pytest.fixture
def ais_df():
    """최소한의 AIS 데이터프레임."""
    return pd.DataFrame({
        "MMSI": [1, 1, 1, 2, 2],
        "BaseDateTime": pd.to_datetime([
            "2022-01-01 00:00:00", "2022-01-01 00:01:00", "2022-01-01 00:02:00",
            "2022-01-01 00:00:00", "2022-01-01 00:05:00",
        ]),
        "SOG": [5.0, 10.0, 5.0, 3.0, 3.0],
        "COG": [90.0, 180.0, 270.0, 45.0, 50.0],
        "Heading": [90.0, 175.0, 511.0, 45.0, 55.0],
    })


def test_speed_deviation(ais_df):
    result = calc_speed_deviation(ais_df)
    assert "speed_deviation" in result.columns
    assert not result["speed_deviation"].isna().all()


def test_speed_deviation_zero_std():
    """표준편차 0인 선박 (속도 일정) — division by zero 방지 확인."""
    df = pd.DataFrame({
        "MMSI": [1, 1, 1],
        "SOG": [5.0, 5.0, 5.0],
    })
    result = calc_speed_deviation(df)
    assert (result["speed_deviation"] == 0).all()


def test_acceleration(ais_df):
    result = calc_acceleration(ais_df)
    assert "acceleration" in result.columns
    # 첫 레코드는 NaN (diff 불가)
    vessel_1 = result[result["MMSI"] == 1]
    assert pd.isna(vessel_1.iloc[0]["acceleration"])
    assert pd.notna(vessel_1.iloc[1]["acceleration"])


def test_course_change(ais_df):
    result = calc_course_change(ais_df)
    assert "course_change" in result.columns
    # 90→180 = 90도 변화
    vessel_1 = result[result["MMSI"] == 1]
    assert vessel_1.iloc[1]["course_change"] == pytest.approx(90.0)


def test_course_change_wraparound():
    """350→10도는 실제 20도 변화 (360도 래핑 처리 확인)."""
    df = pd.DataFrame({
        "MMSI": [1, 1],
        "COG": [350.0, 10.0],
    })
    result = calc_course_change(df)
    assert result.iloc[1]["course_change"] == pytest.approx(20.0)


def test_heading_cog_diff(ais_df):
    result = calc_heading_cog_diff(ais_df)
    assert "heading_cog_diff" in result.columns


def test_heading_511_becomes_nan(ais_df):
    """Heading=511은 NaN 처리되어야 한다."""
    result = calc_heading_cog_diff(ais_df)
    vessel_1 = result[result["MMSI"] == 1]
    assert pd.isna(vessel_1.iloc[2]["heading_cog_diff"])


def test_signal_gap(ais_df):
    result = calc_signal_gap(ais_df)
    assert "signal_gap_sec" in result.columns
    vessel_1 = result[result["MMSI"] == 1]
    assert vessel_1.iloc[1]["signal_gap_sec"] == pytest.approx(60.0)


def test_stop_duration(ais_df):
    result = calc_stop_duration(ais_df)
    assert "stop_duration_min" in result.columns
    # SOG가 threshold 이상이면 stop_duration_min은 0
    moving = result[result["SOG"] >= 0.5]
    assert (moving["stop_duration_min"] == 0).all()


def test_night_activity():
    df = pd.DataFrame({
        "BaseDateTime": pd.to_datetime([
            "2022-01-01 03:00:00",  # 야간
            "2022-01-01 12:00:00",  # 주간
            "2022-01-01 23:00:00",  # 야간
        ]),
    })
    result = calc_night_activity(df)
    assert result.iloc[0]["is_night"] == 1
    assert result.iloc[1]["is_night"] == 0
    assert result.iloc[2]["is_night"] == 1


def test_build_features(ais_df):
    result = build_features(ais_df)
    expected_cols = [
        "speed_deviation", "acceleration", "course_change",
        "heading_cog_diff", "signal_gap_sec", "stop_duration_min", "is_night",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_build_features_missing_column():
    """필수 컬럼이 없으면 ValueError."""
    df = pd.DataFrame({"MMSI": [1], "SOG": [5.0]})
    with pytest.raises(ValueError, match="필요한 컬럼"):
        build_features(df)
