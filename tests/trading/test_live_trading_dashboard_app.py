from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def test_streamlit_dashboard_module_imports_without_starting_app() -> None:
    module = _load_dashboard_module()

    assert callable(module.main)
    assert len(module.FEATURE_LABELS) >= 36


def test_day4_artifact_paths_fall_back_for_legacy_output_settings(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    config = SimpleNamespace(
        outputs=SimpleNamespace(
            live_trading_dir=tmp_path,
        )
    )

    paths = module._day4_artifact_paths(config)

    assert paths["portfolio_snapshot.csv"] == tmp_path / "portfolio_snapshot.csv"
    assert paths["risk_decisions.csv"] == tmp_path / "risk_decisions.csv"


def _load_dashboard_module():
    app_path = ROOT / "apps/live_trading_dashboard.py"
    spec = importlib.util.spec_from_file_location("live_trading_dashboard_app", app_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
