"""Battery Forecast integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError

from .const import DOMAIN, SERVICE_RELOAD_MODEL, SERVICE_TRAIN
from .coordinator import BatteryForecastCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR]


async def async_setup(hass: HomeAssistant, base_config: dict) -> bool:
    """Set up Battery Forecast from YAML (not used)."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Battery Forecast from a config entry."""
    coordinator = BatteryForecastCoordinator(hass, entry)
    await coordinator.async_load_model()

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    if not hass.services.has_service(DOMAIN, SERVICE_TRAIN):

        async def handle_train(call: ServiceCall) -> None:
            target = call.data.get("config_entry_id") or next(
                iter(hass.data.get(DOMAIN, {})), None
            )
            if not target:
                raise HomeAssistantError("No config entry found")
            coord: BatteryForecastCoordinator = hass.data[DOMAIN][target]
            try:
                await coord.async_train()
                await coord.async_request_refresh()
            except ImportError as err:
                raise HomeAssistantError(str(err)) from err
            except Exception as err:
                raise HomeAssistantError(f"Training failed: {err}") from err

        async def handle_reload(call: ServiceCall) -> None:
            target = call.data.get("config_entry_id") or next(
                iter(hass.data.get(DOMAIN, {})), None
            )
            if not target:
                raise HomeAssistantError("No config entry found")
            coord: BatteryForecastCoordinator = hass.data[DOMAIN][target]
            await coord.async_load_model()
            await coord.async_request_refresh()

        hass.services.async_register(DOMAIN, SERVICE_TRAIN, handle_train)
        hass.services.async_register(DOMAIN, SERVICE_RELOAD_MODEL, handle_reload)

    if coordinator.model_bundle is not None:
        await coordinator.async_config_entry_first_refresh()
    else:
        _LOGGER.warning(
            "Battery Forecast: no model yet. Run service %s.train after setup.",
            DOMAIN,
        )
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
        if not hass.data[DOMAIN]:
            hass.data.pop(DOMAIN)
            hass.services.async_remove(DOMAIN, SERVICE_TRAIN)
            hass.services.async_remove(DOMAIN, SERVICE_RELOAD_MODEL)
    return unload_ok
