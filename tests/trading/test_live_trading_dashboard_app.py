from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_streamlit_dashboard_module_imports_without_starting_app() -> None:
    app_path = ROOT / "apps/live_trading_dashboard.py"
    spec = importlib.util.spec_from_file_location("live_trading_dashboard_app", app_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.main)
    assert len(module.FEATURE_LABELS) >= 36
