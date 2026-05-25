"""Shared helpers (avoid circular imports with config_flow)."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback

from .const import CONF_BATTERY_POWER_INVERT, DEFAULT_BATTERY_POWER_INVERT


@callback
def get_config(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    """Merge config entry data and options."""
    return {**entry.data, **entry.options}


def _parse_float(state: Any) -> float | None:
    if state is None:
        return None
    try:
        return float(state)
    except (TypeError, ValueError):
        return None


def read_power_w(hass: HomeAssistant, entity_id: str | None) -> float:
    """Read a power sensor state as watts (handles W / kW unit)."""
    if not entity_id:
        return 0.0
    state = hass.states.get(entity_id)
    if state is None:
        return 0.0
    val = _parse_float(state.state)
    if val is None:
        return 0.0
    unit = (state.attributes.get("unit_of_measurement") or "").lower()
    if unit in ("kw", "kilowatt", "kilowatts"):
        return val * 1000.0
    return val


def read_battery_power_w(hass: HomeAssistant, config: dict[str, Any]) -> float:
    """Battery power in W; optional sign invert (positive = charging by default)."""
    entity_id = config.get("battery_power")
    if not entity_id:
        return 0.0
    watts = read_power_w(hass, entity_id)
    if config.get(CONF_BATTERY_POWER_INVERT, DEFAULT_BATTERY_POWER_INVERT):
        watts = -watts
    return watts


def read_battery_power_kw(hass: HomeAssistant, config: dict[str, Any]) -> float:
    return read_battery_power_w(hass, config) / 1000.0
