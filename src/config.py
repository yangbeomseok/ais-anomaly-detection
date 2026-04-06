"""설정 관리 모듈 - config.yaml을 로드하여 각 모듈에 제공한다."""

import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_config = None


def _defaults() -> dict:
    """config.yaml이 없을 때 사용할 기본값."""
    return {
        "data": {
            "raw_dir": "data/raw",
            "cleaned_path": "data/ais_cleaned.parquet",
            "featured_path": "data/ais_featured.parquet",
            "results_path": "data/ais_results.parquet",
        },
        "preprocessing": {
            "max_speed": 50.0,
            "stop_threshold": 0.5,
            "top_n_vessels": 500,
        },
        "features": {
            "columns": [
                "SOG", "speed_deviation", "acceleration",
                "course_change", "heading_cog_diff",
                "signal_gap_sec", "stop_duration_min", "is_night",
            ]
        },
        "models": {
            "isolation_forest": {"contamination": 0.05, "random_state": 42},
            "lof": {"contamination": 0.05, "n_neighbors": 30},
            "hdbscan": {"min_cluster_size": 50, "min_samples": 10},
            "ensemble": {"threshold": 2},
        },
        "shap": {"max_samples": 5000},
        "visualization": {
            "normal_sample_size": 5000,
            "anomaly_marker_radius": 3,
            "dpi": 150,
        },
    }


def load_config(path: str = None) -> dict:
    """config.yaml을 로드한다. 파일이 없으면 기본값을 반환한다."""
    global _config
    if _config is not None and path is None:
        return _config

    config_path = Path(path) if path else _CONFIG_PATH

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    else:
        _config = _defaults()

    return _config


def get(section: str, key: str = None, default=None):
    """설정값을 가져온다. 예: get("models", "isolation_forest")"""
    cfg = load_config()
    value = cfg.get(section, {})
    if key is not None:
        return value.get(key, default) if isinstance(value, dict) else default
    return value
