"""Data update coordinator for Battery Forecast."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .helpers import get_config
from .const import (
    ATTR_FEATURE_IMPORTANCES,
    ATTR_MAE_KWH,
    ATTR_MODEL_SAMPLES,
    ATTR_MODEL_TRAINED_AT,
    ATTR_R2,
    ATTR_RMSE_KWH,
    DOMAIN,
    STORAGE_KEY,
    STORAGE_VERSION,
)
from .model.simulator import ForecastResult, run_forecast
from .model.trainer import (
    ModelBundle,
    async_train_model,
    bundle_from_storage,
    bundle_to_storage,
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
        self._bundle: ModelBundle | None = None

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

    async def async_save_model(self, bundle: ModelBundle) -> None:
        """Persist model bundle."""
        await self._store.async_save(bundle_to_storage(bundle))
        self._bundle = bundle

    async def async_train(self) -> ModelBundle:
        """Train and store a new model."""
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
            return {}
        b = self._bundle
        return {
            ATTR_MODEL_TRAINED_AT: b.trained_at,
            ATTR_MODEL_SAMPLES: b.n_samples,
            ATTR_MAE_KWH: b.mae_kwh,
            ATTR_RMSE_KWH: b.rmse_kwh,
            ATTR_R2: b.r2,
            ATTR_FEATURE_IMPORTANCES: b.feature_importances,
            "model_type": b.model_type,
        }

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

        return result
