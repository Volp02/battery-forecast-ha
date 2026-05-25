"""Data update coordinator for Battery Forecast."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .helpers import get_config
from .const import (
    ATTR_AUTO_RETRAIN_LAST,
    ATTR_FEATURE_IMPORTANCES,
    ATTR_FORECAST_SOC_MAE,
    ATTR_MAE_KWH,
    ATTR_MODEL_SAMPLES,
    ATTR_MODEL_TRAINED_AT,
    ATTR_R2,
    ATTR_RMSE_KWH,
    CONF_AUTO_RETRAIN_ENABLED,
    CONF_AUTO_RETRAIN_EVAL_HOURS,
    CONF_AUTO_RETRAIN_MIN_HOURS,
    CONF_AUTO_RETRAIN_SOC_MAE,
    DEFAULT_AUTO_RETRAIN_ENABLED,
    DEFAULT_AUTO_RETRAIN_EVAL_HOURS,
    DEFAULT_AUTO_RETRAIN_MIN_HOURS,
    DEFAULT_AUTO_RETRAIN_SOC_MAE,
    DOMAIN,
    STORAGE_EVAL_KEY,
    STORAGE_EVAL_VERSION,
    STORAGE_KEY,
    STORAGE_VERSION,
)
from .model.evaluation import (
    append_snapshot,
    compute_forecast_soc_mae,
    should_auto_retrain,
)
from .model.simulator import ForecastResult, run_forecast
from .model.trainer import (
    ModelBundle,
    async_train_model,
    bundle_from_storage,
    bundle_to_storage,
    sklearn_environment,
)

_LOGGER = logging.getLogger(__name__)


class BatteryForecastCoordinator(DataUpdateCoordinator[ForecastResult]):
    """Fetch forecast data."""

    config_entry: ConfigEntry

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.config_entry = entry
        config = get_config(hass, entry)
        update_interval = timedelta(
            minutes=int(config.get("update_interval_minutes", 5))
        )
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=update_interval,
        )
        self._store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._eval_store = Store(hass, STORAGE_EVAL_VERSION, STORAGE_EVAL_KEY)
        self._bundle: ModelBundle | None = None
        self._forecast_soc_mae: float | None = None
        self._auto_retrain_last_at: str | None = None
        self._auto_retrain_task: asyncio.Task[None] | None = None
        self._retrain_lock = asyncio.Lock()
        self._sklearn_env: dict[str, Any] = sklearn_environment()

    @property
    def model_bundle(self) -> ModelBundle | None:
        return self._bundle

    async def async_load_model(self) -> None:
        """Load persisted model from storage."""
        data = await self._store.async_load()
        if data:
            try:
                self._bundle = bundle_from_storage(data)
            except Exception as err:  # noqa: BLE001
                _LOGGER.error("Failed to load model: %s", err)
                self._bundle = None
        eval_data = await self._eval_store.async_load()
        if eval_data:
            self._auto_retrain_last_at = eval_data.get("last_auto_train_at")

    async def async_save_model(self, bundle: ModelBundle) -> None:
        """Persist model bundle."""
        await self._store.async_save(bundle_to_storage(bundle))
        self._bundle = bundle

    async def async_train(self, *, auto: bool = False) -> ModelBundle:
        """Train and store a new model."""
        if auto:
            _LOGGER.info("Battery Forecast: auto-retrain started (forecast deviation)")
        else:
            _LOGGER.info("Battery Forecast: train service started")
        config = get_config(self.hass, self.config_entry)
        bundle = await async_train_model(self.hass, config)
        await self.async_save_model(bundle)
        _LOGGER.info(
            "Battery Forecast: train complete — model=%s mae=%.3f kWh samples=%s",
            bundle.model_type,
            bundle.mae_kwh,
            bundle.n_samples,
        )
        return bundle

    def model_attributes(self) -> dict[str, Any]:
        """Expose model metadata for sensors."""
        if not self._bundle:
            attrs: dict[str, Any] = {}
        else:
            b = self._bundle
            attrs = {
                ATTR_MODEL_TRAINED_AT: b.trained_at,
                ATTR_MODEL_SAMPLES: b.n_samples,
                ATTR_MAE_KWH: b.mae_kwh,
                ATTR_RMSE_KWH: b.rmse_kwh,
                ATTR_R2: b.r2,
                ATTR_FEATURE_IMPORTANCES: b.feature_importances,
                "model_type": b.model_type,
            }
        if self._forecast_soc_mae is not None:
            attrs[ATTR_FORECAST_SOC_MAE] = round(self._forecast_soc_mae, 2)
        if self._auto_retrain_last_at:
            attrs[ATTR_AUTO_RETRAIN_LAST] = self._auto_retrain_last_at
        self._sklearn_env = sklearn_environment()
        attrs.update(self._sklearn_env)
        return attrs

    async def _async_update_data(self) -> ForecastResult:
        if self._bundle is None:
            await self.async_load_model()
        if self._bundle is None:
            raise UpdateFailed(
                "No trained model. Call service battery_forecast.train first."
            )
        config = get_config(self.hass, self.config_entry)
        try:
            result = await self.hass.async_add_executor_job(
                run_forecast, self.hass, config, self._bundle
            )
        except Exception as err:
            raise UpdateFailed(f"Forecast failed: {err}") from err

        await self._evaluate_and_maybe_retrain(result, config)
        return result

    async def _evaluate_and_maybe_retrain(
        self, result: ForecastResult, config: dict[str, Any]
    ) -> None:
        if not config.get(CONF_AUTO_RETRAIN_ENABLED, DEFAULT_AUTO_RETRAIN_ENABLED):
            return

        eval_data = await self._eval_store.async_load() or {
            "snapshots": [],
            "last_auto_train_at": None,
        }

        def _append() -> None:
            append_snapshot(eval_data, result)

        await self.hass.async_add_executor_job(_append)

        self._forecast_soc_mae = await compute_forecast_soc_mae(
            self.hass,
            battery_soc=config["battery_soc"],
            eval_data=eval_data,
            eval_hours=int(
                config.get(CONF_AUTO_RETRAIN_EVAL_HOURS, DEFAULT_AUTO_RETRAIN_EVAL_HOURS)
            ),
        )

        await self._eval_store.async_save(eval_data)

        threshold = float(
            config.get(CONF_AUTO_RETRAIN_SOC_MAE, DEFAULT_AUTO_RETRAIN_SOC_MAE)
        )
        min_hours = float(
            config.get(CONF_AUTO_RETRAIN_MIN_HOURS, DEFAULT_AUTO_RETRAIN_MIN_HOURS)
        )

        if not should_auto_retrain(
            mae_percent=self._forecast_soc_mae,
            threshold_percent=threshold,
            last_auto_train_at=eval_data.get("last_auto_train_at"),
            min_interval_hours=min_hours,
        ):
            return

        _LOGGER.info(
            "Battery Forecast: auto-retrain scheduled — forecast SOC MAE %.1f%% >= %.1f%%",
            self._forecast_soc_mae or 0,
            threshold,
        )
        self._schedule_auto_retrain(eval_data)

    def _schedule_auto_retrain(self, eval_data: dict[str, Any]) -> None:
        if self._auto_retrain_task and not self._auto_retrain_task.done():
            return

        async def _run() -> None:
            from datetime import datetime, timezone

            async with self._retrain_lock:
                try:
                    await self.async_train(auto=True)
                    now = datetime.now(timezone.utc).isoformat()
                    eval_data["last_auto_train_at"] = now
                    self._auto_retrain_last_at = now
                    await self._eval_store.async_save(eval_data)
                    await self.async_request_refresh()
                except Exception as err:  # noqa: BLE001
                    _LOGGER.error("Battery Forecast: auto-retrain failed: %s", err)

        self._auto_retrain_task = self.hass.async_create_task(_run())
