"""Closed-loop weather and Kalshi research infrastructure.

The package is deliberately separate from the live trading loop.  It may
produce shadow decisions and historical-proxy evidence, but it never submits
orders or changes the repository's live-trading safety flags.
"""

from .gates import CompetenceThresholds, evaluate_competence_gates
from .interfaces import (
    ExperimentRecord,
    FixedPointLevel,
    ForecastDistribution,
    ForecastRequest,
    MarketSnapshot,
    TemperatureBucket,
    TradeDecision,
)

__all__ = [
    "CompetenceThresholds",
    "ExperimentRecord",
    "FixedPointLevel",
    "ForecastDistribution",
    "ForecastRequest",
    "MarketSnapshot",
    "TemperatureBucket",
    "TradeDecision",
    "evaluate_competence_gates",
]
