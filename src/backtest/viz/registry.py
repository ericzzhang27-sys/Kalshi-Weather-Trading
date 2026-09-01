"""Extensible registry of diagnostic plots.

New plots are added by decorating a function with `@register_plot(section=...)`;
reports and dashboards pick them up automatically without any other change.
Each registered function receives a `BacktestResult` and returns a
`plotly.graph_objects.Figure`, or None when required optional data is missing.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

SECTION_ORDER = ["overview", "performance", "risk", "trades", "edge", "execution", "robustness"]

_REGISTRY: "OrderedDict[str, list[PlotSpec]]" = OrderedDict((s, []) for s in SECTION_ORDER)


@dataclass(frozen=True)
class PlotSpec:
    name: str
    section: str
    title: str
    func: Callable
    requires: tuple[str, ...] = field(default=())


def register_plot(section: str, name: str | None = None, title: str | None = None,
                  requires: tuple[str, ...] = ()):
    """Register a plotting function.

    `requires` may contain: 'trades', 'prices', 'benchmark', 'cash',
    'positions', 'probabilities' or a trade-column name; the plot is skipped
    (returning None) when unmet.
    """
    def decorator(func: Callable):
        spec = PlotSpec(
            name=name or func.__name__.removeprefix("plot_"),
            section=section,
            title=title or (func.__doc__.strip().splitlines()[0] if func.__doc__ else name or func.__name__),
            func=func,
            requires=tuple(requires),
        )
        bucket = _REGISTRY.setdefault(section, [])
        bucket.append(spec)
        return func
    return decorator


def get_registered_plots(section: str | None = None) -> list[PlotSpec]:
    sections = [section] if section else SECTION_ORDER
    out: list[PlotSpec] = []
    for s in sections:
        out.extend(_REGISTRY.get(s, []))
    return out


def _has_data(res, col: str) -> bool:
    if res.has_trades:
        t = res.prepared_trades()
        if col in t.columns and pd_notna_any(t[col]):
            return True
    return res.has_column(col)


def pd_notna_any(series) -> bool:
    import pandas as pd
    return bool(pd.to_numeric(series, errors="coerce").notna().any() or series.notna().any())


def run_plot(spec: PlotSpec, result) -> "Figure | None":
    from src.backtest.viz.schema import BacktestResult  # local import avoids cycle at module load
    res: BacktestResult = result
    for req in spec.requires:
        if req == "trades" and not res.has_trades:
            return None
        if req == "prices" and not res.has_prices:
            return None
        if req == "benchmark" and not res.has_benchmark:
            return None
        if req == "cash" and res.cash_series() is None:
            return None
        if req == "positions" and not res.has_positions:
            return None
        if req == "probabilities" and not _has_data(res, "model_probability"):
            return None
        if req.startswith("col:"):
            if not _has_data(res, req[4:]):
                return None
        elif req not in ("trades", "prices", "benchmark", "cash", "positions", "probabilities") \
                and not _has_data(res, req):
            return None
    try:
        return spec.func(result)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Plot '%s' failed for result '%s'", spec.name, result.name)
        return None


def build_all_figures(result) -> dict[str, "OrderedDict[str, Figure]"]:
    """Run every registered plot for a result, grouped by section."""
    figures: dict[str, OrderedDict] = {s: OrderedDict() for s in SECTION_ORDER}
    for spec in get_registered_plots():
        fig = run_plot(spec, result)
        if fig is not None:
            figures.setdefault(spec.section, OrderedDict())[spec.name] = fig
    return figures
