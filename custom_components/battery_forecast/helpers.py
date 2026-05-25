"""Shared helpers (avoid circular imports with config_flow)."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback


@callback
def get_config(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    """Merge config entry data and options."""
    return {**entry.data, **entry.options}
