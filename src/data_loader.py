"""AIS 데이터 로더 모듈"""

import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_ais_csv(filename: str) -> pd.DataFrame:
    """AIS CSV 파일을 로드한다."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {filepath}")
    return pd.read_csv(filepath)


def load_all_ais(pattern: str = "*.csv") -> pd.DataFrame:
    """data/raw/ 내 모든 AIS CSV를 병합 로드한다."""
    files = list(DATA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_DIR}/{pattern}")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)
