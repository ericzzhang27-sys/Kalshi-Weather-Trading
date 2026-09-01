from __future__ import annotations

"""
NGBoost forecast-error ensembles for smoother, better-calibrated bucket prices.

Members are independently trained NGBoost distributional models (varying seed,
base distribution, column subsample, and bootstrap resample). Predictions are
combined at the moment level:

    mu_combined     = mean_i(mu_i)
    sigma_combined  = sqrt( mean_i(sigma_i^2) + var_i(mu_i) )

which is the law-of-total-variance moment match of the member mixture. The
variance term across members widens sigma wherever members disagree, directly
attenuating spurious large edges caused by one member's overconfident sigma.

The combined (mu, sigma) pair is emitted as a Normal distribution so the entire
existing pricing pipeline (`src/distribution_pricing.price_buckets_for_dataframe`)
works unchanged.
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.distributional_model import (  # noqa: E402
    normalize_distribution_name,
    predict_distribution_details,
    train_ngboost_distribution,
)


@dataclass(frozen=True)
class EnsembleMemberSpec:
    """Deterministic recipe for one ensemble member."""

    name: str
    distribution: str
    random_state: int
    col_sample: float
    bootstrap_fraction: float
    bootstrap_seed: int


@dataclass
class EnsembleConfig:
    """Hyperparameters shared by all members plus diversity knobs."""

    distributions: tuple[str, ...] = ("laplace", "student_t", "normal")
    seeds: tuple[int, ...] = (11, 42)
    n_estimators: int = 120
    learning_rate: float = 0.05
    max_depth: int = 2
    min_samples_leaf: int = 50
    minibatch_frac: float = 1.0
    natural_gradient: bool = True
    early_stopping_rounds: int | None = 20
    col_sample: float = 0.8
    bootstrap_fraction: float = 0.85

    def __post_init__(self) -> None:
        if not self.distributions:
            raise ValueError("EnsembleConfig requires at least one distribution")
        if not self.seeds:
            raise ValueError("EnsembleConfig requires at least one seed")
        normalized = tuple(normalize_distribution_name(str(d)) for d in self.distributions)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Duplicate distributions in ensemble config: {self.distributions}")
        object.__setattr__(self, "distributions", normalized)
        if not 0.0 < self.col_sample <= 1.0:
            raise ValueError(f"col_sample must be in (0, 1], got {self.col_sample}")
        if not 0.0 < self.bootstrap_fraction <= 1.0:
            raise ValueError(
                f"bootstrap_fraction must be in (0, 1], got {self.bootstrap_fraction}"
            )


def build_member_specs(config: EnsembleConfig) -> list[EnsembleMemberSpec]:
    """Deterministic member list: one spec per (distribution, seed) pair."""
    specs = []
    for distribution in config.distributions:
        for seed_index, seed in enumerate(config.seeds):
            specs.append(
                EnsembleMemberSpec(
                    name=f"{distribution}_seed{seed}",
                    distribution=distribution,
                    random_state=int(seed),
                    col_sample=float(config.col_sample),
                    bootstrap_fraction=float(config.bootstrap_fraction),
                    bootstrap_seed=1000 + 97 * seed_index,
                )
            )
    return specs


def bootstrap_row_indices(n_rows: int, fraction: float, seed: int) -> np.ndarray:
    """Sample-with-replacement row indices of length round(fraction * n_rows)."""
    if n_rows <= 0:
        raise ValueError("Cannot bootstrap an empty training frame")
    size = max(1, int(round(float(fraction) * n_rows)))
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_rows, size=size)


def train_ensemble_member(
    spec: EnsembleMemberSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    *,
    config: EnsembleConfig,
) -> dict[str, Any]:
    """Train a single ensemble member on its bootstrap resample."""
    indices = bootstrap_row_indices(len(X_train), spec.bootstrap_fraction, spec.bootstrap_seed)
    X_resample = X_train.iloc[indices].reset_index(drop=True)
    y_resample = np.asarray(y_train, dtype=float)[indices]
    model = train_ngboost_distribution(
        X_train=X_resample,
        y_train=y_resample,
        X_val=X_val,
        y_val=y_val,
        distribution=spec.distribution,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        minibatch_frac=config.minibatch_frac,
        natural_gradient=config.natural_gradient,
        random_state=spec.random_state,
        col_sample=spec.col_sample,
        early_stopping_rounds=config.early_stopping_rounds,
    )
    return {
        "spec": spec,
        "model": model,
        "n_training_rows": int(len(indices)),
        "n_unique_training_rows": int(pd.unique(indices).size),
    }


def combine_moment_predictions(mus: list[np.ndarray], sigmas: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Moment-match the member mixture: mean of mus, total-variance sigma."""
    if not mus or not sigmas:
        raise ValueError("At least one member prediction is required")
    mu_arrays = [np.asarray(mu, dtype=float) for mu in mus]
    sigma_arrays = [np.asarray(sigma, dtype=float) for sigma in sigmas]
    lengths = {array.shape[0] for array in mu_arrays} | {array.shape[0] for array in sigma_arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError(f"Member predictions must share one non-zero length, got {lengths}")
    stacked_mu = np.vstack(mu_arrays)
    stacked_sigma = np.vstack(sigma_arrays)
    if not np.isfinite(stacked_mu).all():
        raise ValueError("Member mu values contain non-finite entries")
    if not np.isfinite(stacked_sigma).all() or (stacked_sigma <= 0).any():
        raise ValueError("Member sigma values must be finite and positive")
    mu_combined = stacked_mu.mean(axis=0)
    within = np.mean(stacked_sigma**2, axis=0)
    between = stacked_mu.var(axis=0, ddof=0)
    sigma_combined = np.sqrt(within + between)
    if not np.isfinite(sigma_combined).all() or (sigma_combined <= 0).any():
        raise AssertionError("Combined sigma must remain finite and positive")
    return mu_combined, sigma_combined


def predict_ensemble_moments(
    members: list[dict[str, Any]],
    X: pd.DataFrame,
    *,
    sigma_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-member predictions -> combined Normal moments."""
    if not members:
        raise ValueError("Ensemble has no members to predict with")
    mus = []
    sigmas = []
    for member in members:
        details = predict_distribution_details(member["model"], X, member["spec"].distribution)
        mus.append(np.asarray(details["mu"], dtype=float))
        sigmas.append(np.asarray(details["sigma"], dtype=float) * float(sigma_scale))
    return combine_moment_predictions(mus, sigmas)


@dataclass
class EnsembleModel:
    """Serializable container for a trained member ensemble."""

    members: list[dict[str, Any]] = field(default_factory=list)
    config: EnsembleConfig = field(default_factory=EnsembleConfig)
    feature_columns: list[str] = field(default_factory=list)
    sigma_scale: float = 1.0
    name: str = "ngboost_ensemble_v1"
    created_at_utc: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.config, EnsembleConfig):
            raise TypeError("config must be an EnsembleConfig")
        if self.members:
            seen = set()
            for member in self.members:
                key = member["spec"].name
                if key in seen:
                    raise ValueError(f"Duplicate ensemble member name: {key}")
                seen.add(key)

    def predict_distribution_params(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = [column for column in self.feature_columns if column not in X.columns]
        if missing:
            raise ValueError(f"Feature columns missing from input frame: {missing}")
        return predict_ensemble_moments(
            self.members,
            X[self.feature_columns],
            sigma_scale=self.sigma_scale,
        )

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "members": self.members,
            "config": vars(self.config),
            "feature_columns": self.feature_columns,
            "sigma_scale": float(self.sigma_scale),
            "name": self.name,
            "created_at_utc": self.created_at_utc or datetime.now(timezone.utc).isoformat(),
        }
        artifact_path = directory / f"{self.name}.pkl"
        with artifact_path.open("wb") as file:
            pickle.dump(payload, file)
        manifest = {
            "model_type": "ngboost_moment_ensemble",
            "name": self.name,
            "combined_distribution_type": "normal",
            "member_count": len(self.members),
            "members": [
                {
                    "name": m["spec"].name,
                    "distribution": m["spec"].distribution,
                    "random_state": m["spec"].random_state,
                    "col_sample": m["spec"].col_sample,
                    "bootstrap_fraction": m["spec"].bootstrap_fraction,
                    "bootstrap_seed": m["spec"].bootstrap_seed,
                    "n_training_rows": m["n_training_rows"],
                    "n_unique_training_rows": m["n_unique_training_rows"],
                }
                for m in self.members
            ],
            "combination_rule": (
                "mu=mean(member mu); sigma=sqrt(mean(sigma^2)+var(mu)) "
                "(law of total variance moment match)"
            ),
            "sigma_scale": float(self.sigma_scale),
            "feature_count": len(self.feature_columns),
            "created_at_utc": payload["created_at_utc"],
        }
        (directory / f"{self.name}_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return artifact_path

    @classmethod
    def load(cls, directory: Path, name: str = "ngboost_ensemble_v1") -> "EnsembleModel":
        artifact_path = Path(directory) / f"{name}.pkl"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Ensemble artifact not found: {artifact_path}")
        with artifact_path.open("rb") as file:
            payload = pickle.load(file)
        return cls(
            members=payload["members"],
            config=EnsembleConfig(**payload["config"]),
            feature_columns=list(payload["feature_columns"]),
            sigma_scale=float(payload["sigma_scale"]),
            name=str(payload["name"]),
            created_at_utc=str(payload.get("created_at_utc", "")),
        )


def train_full_ensemble(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None,
    y_val: np.ndarray | None,
    config: EnsembleConfig,
    *,
    feature_columns: list[str],
    sigma_scale: float = 1.0,
    name: str = "ngboost_ensemble_v1",
    on_member_done=None,
) -> EnsembleModel:
    """Train every member spec and wrap them in an EnsembleModel."""
    members = []
    specs = build_member_specs(config)
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] Training ensemble member {spec.name} ...", flush=True)
        member = train_ensemble_member(spec, X_train, y_train, X_val, y_val, config=config)
        members.append(member)
        if on_member_done is not None:
            on_member_done(index, len(specs), spec, member)
    return EnsembleModel(
        members=members,
        config=config,
        feature_columns=list(feature_columns),
        sigma_scale=float(sigma_scale),
        name=name,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
