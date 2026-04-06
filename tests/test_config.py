"""설정 모듈 테스트"""

import pytest
from src.config import load_config, get, _defaults


def test_load_config_returns_dict():
    cfg = load_config()
    assert isinstance(cfg, dict)


def test_config_has_required_sections():
    cfg = load_config()
    for section in ["data", "preprocessing", "features", "models", "shap", "visualization"]:
        assert section in cfg, f"Missing config section: {section}"


def test_get_section():
    result = get("models")
    assert isinstance(result, dict)
    assert "isolation_forest" in result


def test_get_key():
    result = get("preprocessing", "max_speed")
    assert isinstance(result, (int, float))
    assert result > 0


def test_get_missing_key_returns_default():
    result = get("models", "nonexistent_key", default=999)
    assert result == 999


def test_defaults_complete():
    d = _defaults()
    assert "features" in d
    assert "columns" in d["features"]
    assert len(d["features"]["columns"]) == 8
