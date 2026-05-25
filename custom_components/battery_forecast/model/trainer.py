"""Train and persist the load forecast model."""

from __future__ import annotations

import base64
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .features import load_training_data

_LOGGER = logging.getLogger(__name__)

HOLDOUT_HOURS = 14 * 24

SKLEARN_INSTALL_HINT = (
    "scikit-learn is not installed in Home Assistant. "
    "Install it once via SSH/Terminal add-on, e.g.: "
    "pip install scikit-learn  (inside the HA Python venv), then restart HA."
)


def _require_sklearn() -> tuple[Any, Any, Any, Any, Any]:
    """Import sklearn lazily so config flow works without it."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.pipeline import Pipeline

        return (
            HistGradientBoostingRegressor,
            SimpleImputer,
            Pipeline,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
    except ImportError as err:
        raise ImportError(SKLEARN_INSTALL_HINT) from err


@dataclass
class ModelBundle:
    """Serialized model artifacts."""

    pipeline: Any
    feature_names: list[str]
    active_features: list[str]
    feature_importances: dict[str, float]
    trained_at: str
    n_samples: int
    mae_kwh: float
    rmse_kwh: float
    r2: float


def _build_pipeline() -> Any:
    (
        HistGradientBoostingRegressor,
        SimpleImputer,
        Pipeline,
        _mae,
        _mse,
        _r2,
    ) = _require_sklearn()
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=12,
                    learning_rate=0.08,
                    max_iter=300,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
        ]
    )


def _prune_features(
    feature_names: list[str],
    importances: np.ndarray,
    threshold: float,
) -> list[str]:
    total = float(importances.sum()) or 1.0
    active = []
    for name, imp in zip(feature_names, importances, strict=True):
        if imp / total >= threshold:
            active.append(name)
    if not active:
        return list(feature_names)
    return active


def _fit_impl(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    min_samples: int,
    threshold: float,
    feature_names: list[str],
) -> ModelBundle:
    (
        HistGradientBoostingRegressor,
        _SimpleImputer,
        _Pipeline,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    ) = _require_sklearn()

    if len(y) < min_samples:
        raise ValueError(
            f"Not enough training samples: {len(y)} < {min_samples}. "
            "Check statistics retention and entity selection."
        )

    if len(y) > HOLDOUT_HOURS + min_samples:
        X_train, y_train, w_train = X[:-HOLDOUT_HOURS], y[:-HOLDOUT_HOURS], weights[:-HOLDOUT_HOURS]
        X_val, y_val = X[-HOLDOUT_HOURS:], y[-HOLDOUT_HOURS:]
    else:
        X_train, y_train, w_train = X, y, weights
        X_val, y_val = X, y

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train, model__sample_weight=w_train)

    pred_val = pipeline.predict(X_val)
    mae = float(mean_absolute_error(y_val, pred_val))
    rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
    r2 = float(r2_score(y_val, pred_val)) if len(y_val) > 1 else 0.0

    model: Any = pipeline.named_steps["model"]
    imp = getattr(model, "feature_importances_", np.ones(len(feature_names)))
    importances = {n: float(v) for n, v in zip(feature_names, imp, strict=True)}
    active = _prune_features(feature_names, imp, threshold)

    _LOGGER.info(
        "Trained battery forecast model: samples=%s mae=%.3f rmse=%.3f r2=%.3f",
        len(y),
        mae,
        rmse,
        r2,
    )

    return ModelBundle(
        pipeline=pipeline,
        feature_names=feature_names,
        active_features=active,
        feature_importances=importances,
        trained_at=datetime.now(timezone.utc).isoformat(),
        n_samples=len(y),
        mae_kwh=mae,
        rmse_kwh=rmse,
        r2=r2,
    )


async def async_train_model(hass: Any, config: dict[str, Any]) -> ModelBundle:
    """Load data and train model."""
    min_samples = int(config.get("min_training_samples", 168))
    threshold = float(config.get("importance_threshold", 0.005))
    X, y, weights, feature_names, _hours = await load_training_data(hass, config)
    return _fit_impl(X, y, weights, min_samples, threshold, feature_names)


def bundle_to_storage(bundle: ModelBundle) -> dict[str, Any]:
    """Convert bundle to JSON-storable dict."""
    return {
        "model_b64": base64.b64encode(pickle.dumps(bundle.pipeline)).decode("ascii"),
        "feature_names": bundle.feature_names,
        "active_features": bundle.active_features,
        "feature_importances": bundle.feature_importances,
        "trained_at": bundle.trained_at,
        "n_samples": bundle.n_samples,
        "mae_kwh": bundle.mae_kwh,
        "rmse_kwh": bundle.rmse_kwh,
        "r2": bundle.r2,
    }


def bundle_from_storage(data: dict[str, Any]) -> ModelBundle:
    """Restore bundle from storage dict."""
    _require_sklearn()  # needed to unpickle pipeline
    pipeline = pickle.loads(base64.b64decode(data["model_b64"]))
    return ModelBundle(
        pipeline=pipeline,
        feature_names=list(data["feature_names"]),
        active_features=list(data.get("active_features", data["feature_names"])),
        feature_importances=dict(data.get("feature_importances", {})),
        trained_at=data["trained_at"],
        n_samples=int(data["n_samples"]),
        mae_kwh=float(data["mae_kwh"]),
        rmse_kwh=float(data.get("rmse_kwh", 0)),
        r2=float(data.get("r2", 0)),
    )


def predict_load_kwh(
    bundle: ModelBundle,
    X: np.ndarray,
) -> np.ndarray:
    """Predict hourly net load in kWh."""
    return np.maximum(0.0, bundle.pipeline.predict(X))
