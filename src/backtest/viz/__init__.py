"""Reusable backtest visualization & diagnostics suite.

Quick start::

    from src.backtest.viz import BacktestResult, create_backtest_report
    result = BacktestResult.from_ledger(trades_df)
    figures = create_backtest_report(result, output_path="report.html")

Every plot is independently callable and registered via
`src.backtest.viz.registry.register_plot`; see `src/backtest/viz/demo.py`
for a full synthetic example.
"""
from src.backtest.viz.schema import BacktestResult, load_result_from_dir

from src.backtest.viz import (  # noqa: F401  (import registers the plots)
    plots_performance,
    plots_risk,
    plots_trades,
    plots_edge,
    plots_execution,
    plots_robustness,
)

from src.backtest.viz.registry import (
    SECTION_ORDER,
    build_all_figures,
    get_registered_plots,
    register_plot,
    run_plot,
)
from src.backtest.viz.report import (
    build_report_figures,
    create_backtest_report,
    render_overview_cards,
    run_single_plot,
)
from src.backtest.viz.compare import (
    comparison_table,
    comparison_table_figure,
    create_comparison_report,
    plot_compare_drawdown,
    plot_compare_equity,
    plot_compare_exposure,
    plot_compare_rolling_sharpe,
)

__all__ = [
    "BacktestResult",
    "load_result_from_dir",
    "register_plot",
    "get_registered_plots",
    "run_plot",
    "run_single_plot",
    "build_all_figures",
    "build_report_figures",
    "create_backtest_report",
    "render_overview_cards",
    "create_comparison_report",
    "comparison_table",
    "comparison_table_figure",
    "plot_compare_equity",
    "plot_compare_drawdown",
    "plot_compare_rolling_sharpe",
    "plot_compare_exposure",
    "SECTION_ORDER",
]
