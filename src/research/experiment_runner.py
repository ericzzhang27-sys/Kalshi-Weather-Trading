from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class SearchResult:
    study_name: str
    best_params: Mapping[str, Any]
    best_value: float
    trials: tuple[Mapping[str, Any], ...]


def run_optuna_successive_halving(
    *,
    study_name: str,
    storage_path: str | Path,
    objective: Callable[[Any], float],
    n_trials: int,
    seed: int = 42,
    direction: str = "minimize",
    min_resource: int = 1,
    reduction_factor: int = 5,
) -> SearchResult:
    """Deterministic Optuna search that retains pruned, failed, and complete trials."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required for the research runner; install requirements.txt") from exc
    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be minimize or maximize")
    database = Path(storage_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{database.resolve().as_posix()}",
        load_if_exists=False,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=min_resource,
            reduction_factor=reduction_factor,
        ),
    )
    study.optimize(objective, n_trials=int(n_trials), catch=(Exception,))
    records = []
    for trial in study.trials:
        records.append(
            {
                "number": trial.number,
                "state": trial.state.name.lower(),
                "params": dict(trial.params),
                "value": trial.value,
                "intermediate_values": dict(trial.intermediate_values),
                "duration_seconds": trial.duration.total_seconds() if trial.duration else 0.0,
                "system_attrs": json.loads(json.dumps(trial.system_attrs, default=str)),
                "user_attrs": json.loads(json.dumps(trial.user_attrs, default=str)),
            }
        )
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE" and trial.value is not None]
    if not complete:
        raise RuntimeError("all Optuna trials failed or were pruned")
    return SearchResult(study_name, dict(study.best_params), float(study.best_value), tuple(records))
