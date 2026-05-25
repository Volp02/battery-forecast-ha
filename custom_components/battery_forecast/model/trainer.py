"""Train and persist the load forecast model."""

from __future__ import annotations

import base64
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np

from .features import load_training_data

_LOGGER = logging.getLogger(__name__)

HOLDOUT_HOURS = 14 * 24
MODEL_SKLEARN = "sklearn"
MODEL_NUMPY = "numpy"

SKLEARN_INSTALL_HINT = (
    "For better accuracy install scikit-learn in Home Assistant: "
    "pip install scikit-learn (SSH/Terminal add-on), then retrain."
)


@dataclass
class NumpyRegressionModel:
    """Weighted linear regression fallback (no scikit-learn)."""

    imputer_medians: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_imp = np.array(X, dtype=np.float64, copy=True)
        for col in range(X_imp.shape[1]):
            mask = np.isnan(X_imp[:, col])
            if mask.any():
                X_imp[mask, col] = self.imputer_medians[col]
        return self.intercept + X_imp @ self.coef


@dataclass
class ModelBundle:
    """Serialized model artifacts."""

    model_type: Literal["sklearn", "numpy"]
    feature_names: list[str]
    active_features: list[str]
    feature_importances: dict[str, float]
    trained_at: str
    n_samples: int
    mae_kwh: float
    rmse_kwh: float
    r2: float
    pipeline: Any | None = None
    numpy_model: NumpyRegressionModel | None = None


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if len(y_true) > 1 and np.var(y_true) > 0:
        r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    else:
        r2 = 0.0
    return mae, rmse, r2


def _split_train_val(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, min_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y) > HOLDOUT_HOURS + min_samples:
        return (
            X[:-HOLDOUT_HOURS],
            y[:-HOLDOUT_HOURS],
            weights[:-HOLDOUT_HOURS],
            X[-HOLDOUT_HOURS:],
            y[-HOLDOUT_HOURS:],
        )
    return X, y, weights, X, y


def _prune_features(
    feature_names: list[str],
    importances: np.ndarray,
    threshold: float,
) -> list[str]:
    total = float(np.abs(importances).sum()) or 1.0
    active = [
        name
        for name, imp in zip(feature_names, importances, strict=True)
        if abs(imp) / total >= threshold
    ]
    return active or list(feature_names)


def _fit_numpy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    threshold: float,
    n_samples: int,
) -> ModelBundle:
    imputer_medians = np.nanmedian(X_train, axis=0)
    imputer_medians = np.where(np.isnan(imputer_medians), 0.0, imputer_medians)

    X_imp = np.array(X_train, dtype=np.float64, copy=True)
    for col in range(X_imp.shape[1]):
        mask = np.isnan(X_imp[:, col])
        if mask.any():
            X_imp[mask, col] = imputer_medians[col]

    sw = np.sqrt(np.maximum(w_train, 1e-6))
    X_aug = np.hstack([np.ones((len(X_imp), 1)), X_imp * sw[:, None]])
    y_w = y_train * sw
    solution, _, _, _ = np.linalg.lstsq(X_aug, y_w, rcond=None)
    intercept = float(solution[0])
    coef = solution[1:].astype(np.float64)

    numpy_model = NumpyRegressionModel(
        imputer_medians=imputer_medians,
        coef=coef,
        intercept=intercept,
    )
    pred_val = numpy_model.predict(X_val)
    mae, rmse, r2 = _metrics(y_val, pred_val)

    imp = np.abs(coef)
    importances = {n: float(v) for n, v in zip(feature_names, imp, strict=True)}
    active = _prune_features(feature_names, imp, threshold)

    _LOGGER.info(
        "Trained battery forecast model (numpy fallback): samples=%s mae=%.3f rmse=%.3f r2=%.3f. %s",
        n_samples,
        mae,
        rmse,
        r2,
        SKLEARN_INSTALL_HINT,
    )

    return ModelBundle(
        model_type=MODEL_NUMPY,
        pipeline=None,
        numpy_model=numpy_model,
        feature_names=feature_names,
        active_features=active,
        feature_importances=importances,
        trained_at=datetime.now(timezone.utc).isoformat(),
        n_samples=n_samples,
        mae_kwh=mae,
        rmse_kwh=rmse,
        r2=r2,
    )


def _fit_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    threshold: float,
    n_samples: int,
) -> ModelBundle:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
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
    pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    pred_val = pipeline.predict(X_val)
    mae, rmse, r2 = _metrics(y_val, pred_val)

    model = pipeline.named_steps["model"]
    imp = getattr(model, "feature_importances_", np.ones(len(feature_names)))
    importances = {n: float(v) for n, v in zip(feature_names, imp, strict=True)}
    active = _prune_features(feature_names, imp, threshold)

    _LOGGER.info(
        "Trained battery forecast model (sklearn): samples=%s mae=%.3f rmse=%.3f r2=%.3f",
        n_samples,
        mae,
        rmse,
        r2,
    )

    return ModelBundle(
        model_type=MODEL_SKLEARN,
        pipeline=pipeline,
        numpy_model=None,
        feature_names=feature_names,
        active_features=active,
        feature_importances=importances,
        trained_at=datetime.now(timezone.utc).isoformat(),
        n_samples=n_samples,
        mae_kwh=mae,
        rmse_kwh=rmse,
        r2=r2,
    )


def _fit_impl(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    min_samples: int,
    threshold: float,
    feature_names: list[str],
) -> ModelBundle:
    if len(y) < min_samples:
        raise ValueError(
            f"Not enough training samples: {len(y)} < {min_samples}. "
            "Check statistics retention and entity selection."
        )

    X_train, y_train, w_train, X_val, y_val = _split_train_val(X, y, weights, min_samples)

    if _sklearn_available():
        return _fit_sklearn(
            X_train, y_train, w_train, X_val, y_val, feature_names, threshold, len(y)
        )

    _LOGGER.warning(
        "scikit-learn not found — using numpy linear model. %s", SKLEARN_INSTALL_HINT
    )
    return _fit_numpy(
        X_train, y_train, w_train, X_val, y_val, feature_names, threshold, len(y)
    )


async def async_train_model(hass: Any, config: dict[str, Any]) -> ModelBundle:
    """Load data and train model."""
    min_samples = int(config.get("min_training_samples", 168))
    threshold = float(config.get("importance_threshold", 0.005))
    X, y, weights, feature_names, _hours = await load_training_data(hass, config)
    return _fit_impl(X, y, weights, min_samples, threshold, feature_names)


def bundle_to_storage(bundle: ModelBundle) -> dict[str, Any]:
    """Convert bundle to JSON-storable dict."""
    if bundle.model_type == MODEL_SKLEARN:
        model_blob = bundle.pipeline
    else:
        model_blob = bundle.numpy_model
    return {
        "model_type": bundle.model_type,
        "model_b64": base64.b64encode(pickle.dumps(model_blob)).decode("ascii"),
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
    model_type = data.get("model_type", MODEL_SKLEARN)
    blob = pickle.loads(base64.b64decode(data["model_b64"]))

    pipeline = None
    numpy_model = None
    if model_type == MODEL_NUMPY:
        numpy_model = blob
    else:
        pipeline = blob

    return ModelBundle(
        model_type=model_type,
        pipeline=pipeline,
        numpy_model=numpy_model,
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
    if bundle.model_type == MODEL_NUMPY and bundle.numpy_model is not None:
        return np.maximum(0.0, bundle.numpy_model.predict(X))
    if bundle.pipeline is not None:
        return np.maximum(0.0, bundle.pipeline.predict(X))
    raise RuntimeError("No trained model loaded")
