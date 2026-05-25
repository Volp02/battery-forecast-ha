"""Config flow for Battery Forecast."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
)

from .const import (
    CONF_BATTERY_CAPACITY_KWH,
    CONF_BATTERY_CHARGE_ENERGY,
    CONF_BATTERY_DISCHARGE_ENERGY,
    CONF_BATTERY_POWER,
    CONF_BATTERY_SOC,
    CONF_EMPTY_SOC_PERCENT,
    CONF_FEATURE_ENTITIES,
    CONF_FORECAST_HORIZON_HOURS,
    CONF_HEAT_PUMP_POWER,
    CONF_HOUSE_POWER,
    CONF_IMPORTANCE_THRESHOLD,
    CONF_MAX_FEATURE_ENTITIES,
    CONF_MIN_TRAINING_SAMPLES,
    CONF_OUTDOOR_TEMP,
    CONF_PV_FORECAST,
    CONF_PV_FORECAST_TOMORROW,
    CONF_PV_POWER,
    CONF_SAMPLE_HALF_LIFE_DAYS,
    CONF_TRAINING_DAYS,
    CONF_UPDATE_INTERVAL_MINUTES,
    CONF_USE_RECORDER_FALLBACK,
    CONF_WEATHER_ENTITY,
    DEFAULT_EMPTY_SOC_PERCENT,
    DEFAULT_FORECAST_HORIZON_HOURS,
    DEFAULT_IMPORTANCE_THRESHOLD,
    DEFAULT_MAX_FEATURE_ENTITIES,
    DEFAULT_MIN_TRAINING_SAMPLES,
    DEFAULT_SAMPLE_HALF_LIFE_DAYS,
    DEFAULT_TRAINING_DAYS,
    DEFAULT_UPDATE_INTERVAL_MINUTES,
    DEFAULT_USE_RECORDER_FALLBACK,
    DOMAIN,
)

POWER_SENSOR = EntitySelector(
    EntitySelectorConfig(domain="sensor", device_class="power")
)
ENERGY_SENSOR = EntitySelector(
    EntitySelectorConfig(domain="sensor", device_class="energy")
)
SOC_SENSOR = EntitySelector(EntitySelectorConfig(domain="sensor"))
TEMP_SENSOR = EntitySelector(
    EntitySelectorConfig(domain="sensor", device_class="temperature")
)
WEATHER_ENTITY = EntitySelector(EntitySelectorConfig(domain="weather"))
FEATURE_ENTITIES = EntitySelector(
    EntitySelectorConfig(domain="sensor", multiple=True)
)


def _optional_entity(value: str | None) -> str | None:
    return value if value else None


def _validate_entity(hass: HomeAssistant, entity_id: str | None) -> None:
    if entity_id and hass.states.get(entity_id) is None:
        registry = er.async_get(hass)
        if registry.async_get(entity_id) is None:
            raise ValueError(f"invalid_entity: {entity_id}")


def _is_likely_grid_not_house(entity_id: str) -> bool:
    """Warn when users pick grid import instead of total house load."""
    eid = entity_id.lower()
    if any(
        hint in eid
        for hint in (
            "momentanleistung",
            "hausverbrauch",
            "home_consumption",
            "house_consumption",
        )
    ):
        return False
    return any(
        hint in eid
        for hint in (
            "gesamtverbrauch",
            "grid_import",
            "netzbezug",
            "strombezug",
            "grid_power",
            "net_power",
        )
    )


class BatteryForecastConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Battery Forecast."""

    VERSION = 1

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if self._async_current_entries():
            return self.async_abort(reason="already_configured")
        return await self.async_step_battery(user_input)

    async def async_step_battery(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_solar()
        return self.async_show_form(
            step_id="battery",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_BATTERY_SOC): SOC_SENSOR,
                    vol.Required(
                        CONF_BATTERY_CAPACITY_KWH,
                        default=10.0,
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0.1,
                            max=500,
                            step=0.1,
                            mode=NumberSelectorMode.BOX,
                            unit_of_measurement="kWh",
                        )
                    ),
                    vol.Required(CONF_BATTERY_POWER): POWER_SENSOR,
                    vol.Optional(CONF_BATTERY_CHARGE_ENERGY): ENERGY_SENSOR,
                    vol.Optional(CONF_BATTERY_DISCHARGE_ENERGY): ENERGY_SENSOR,
                }
            ),
        )

    async def async_step_solar(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            self._data.update({k: _optional_entity(v) for k, v in user_input.items()})
            return await self.async_step_loads()
        return self.async_show_form(
            step_id="solar",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_PV_POWER): POWER_SENSOR,
                    vol.Optional(CONF_PV_FORECAST): SOC_SENSOR,
                    vol.Optional(CONF_PV_FORECAST_TOMORROW): SOC_SENSOR,
                }
            ),
        )

    async def async_step_loads(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            house = user_input.get(CONF_HOUSE_POWER)
            if house and _is_likely_grid_not_house(house):
                errors[CONF_HOUSE_POWER] = "house_power_is_grid"
            if not errors:
                self._data.update({k: _optional_entity(v) for k, v in user_input.items()})
                return await self.async_step_environment()
        return self.async_show_form(
            step_id="loads",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_HOUSE_POWER): POWER_SENSOR,
                    vol.Optional(CONF_HEAT_PUMP_POWER): POWER_SENSOR,
                }
            ),
            errors=errors,
        )

    async def async_step_environment(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if user_input is not None:
            self._data.update({k: _optional_entity(v) for k, v in user_input.items()})
            return await self.async_step_features()
        return self.async_show_form(
            step_id="environment",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_OUTDOOR_TEMP): TEMP_SENSOR,
                    vol.Optional(CONF_WEATHER_ENTITY): WEATHER_ENTITY,
                }
            ),
        )

    async def async_step_features(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            entities = user_input.get(CONF_FEATURE_ENTITIES) or []
            max_f = DEFAULT_MAX_FEATURE_ENTITIES
            self._data[CONF_FEATURE_ENTITIES] = list(entities)[:max_f]
            return await self.async_step_ml()
        return self.async_show_form(
            step_id="features",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_FEATURE_ENTITIES, default=[]): FEATURE_ENTITIES,
                }
            ),
        )

    async def async_step_ml(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            self._data.update(user_input)
            return await self._create_entry()
        return self.async_show_form(
            step_id="ml",
            data_schema=vol.Schema(self._ml_schema()),
        )

    def _ml_schema(self) -> dict:
        return {
            vol.Optional(
                CONF_EMPTY_SOC_PERCENT, default=DEFAULT_EMPTY_SOC_PERCENT
            ): NumberSelector(
                NumberSelectorConfig(min=0, max=50, step=1, mode=NumberSelectorMode.SLIDER)
            ),
            vol.Optional(CONF_TRAINING_DAYS, default=DEFAULT_TRAINING_DAYS): NumberSelector(
                NumberSelectorConfig(
                    min=7, max=365, step=1, mode=NumberSelectorMode.BOX, unit_of_measurement="d"
                )
            ),
            vol.Optional(
                CONF_SAMPLE_HALF_LIFE_DAYS, default=DEFAULT_SAMPLE_HALF_LIFE_DAYS
            ): NumberSelector(
                NumberSelectorConfig(
                    min=7, max=365, step=1, mode=NumberSelectorMode.BOX, unit_of_measurement="d"
                )
            ),
            vol.Optional(
                CONF_USE_RECORDER_FALLBACK, default=DEFAULT_USE_RECORDER_FALLBACK
            ): bool,
            vol.Optional(
                CONF_FORECAST_HORIZON_HOURS, default=DEFAULT_FORECAST_HORIZON_HOURS
            ): NumberSelector(
                NumberSelectorConfig(min=12, max=96, step=1, mode=NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_UPDATE_INTERVAL_MINUTES, default=DEFAULT_UPDATE_INTERVAL_MINUTES
            ): NumberSelector(
                NumberSelectorConfig(min=1, max=60, step=1, mode=NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_MIN_TRAINING_SAMPLES, default=DEFAULT_MIN_TRAINING_SAMPLES
            ): NumberSelector(
                NumberSelectorConfig(min=48, max=8760, step=1, mode=NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_MAX_FEATURE_ENTITIES, default=DEFAULT_MAX_FEATURE_ENTITIES
            ): NumberSelector(
                NumberSelectorConfig(min=0, max=50, step=1, mode=NumberSelectorMode.BOX)
            ),
            vol.Optional(
                CONF_IMPORTANCE_THRESHOLD, default=DEFAULT_IMPORTANCE_THRESHOLD
            ): NumberSelector(
                NumberSelectorConfig(min=0.0, max=0.1, step=0.001, mode=NumberSelectorMode.BOX)
            ),
        }

    async def _create_entry(self) -> FlowResult:
        hass = self.hass
        required = [
            self._data[CONF_BATTERY_SOC],
            self._data[CONF_BATTERY_POWER],
            self._data[CONF_HOUSE_POWER],
        ]
        try:
            for entity_id in required:
                _validate_entity(hass, entity_id)
            for entity_id in self._data.get(CONF_FEATURE_ENTITIES, []):
                _validate_entity(hass, entity_id)
        except ValueError:
            return self.async_abort(reason="cannot_connect")

        return self.async_create_entry(title="Battery Forecast", data=self._data)

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> BatteryForecastOptionsFlow:
        return BatteryForecastOptionsFlow()


class BatteryForecastOptionsFlow(config_entries.OptionsFlow):
    """Handle options."""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        entry = self.config_entry
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_EMPTY_SOC_PERCENT,
                        default=entry.options.get(
                            CONF_EMPTY_SOC_PERCENT,
                            entry.data.get(CONF_EMPTY_SOC_PERCENT, DEFAULT_EMPTY_SOC_PERCENT),
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(min=0, max=50, step=1, mode=NumberSelectorMode.SLIDER)
                    ),
                    vol.Optional(
                        CONF_TRAINING_DAYS,
                        default=entry.options.get(
                            CONF_TRAINING_DAYS,
                            entry.data.get(CONF_TRAINING_DAYS, DEFAULT_TRAINING_DAYS),
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=7, max=365, step=1, mode=NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional(
                        CONF_SAMPLE_HALF_LIFE_DAYS,
                        default=entry.options.get(
                            CONF_SAMPLE_HALF_LIFE_DAYS,
                            entry.data.get(
                                CONF_SAMPLE_HALF_LIFE_DAYS, DEFAULT_SAMPLE_HALF_LIFE_DAYS
                            ),
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=7, max=365, step=1, mode=NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional(
                        CONF_USE_RECORDER_FALLBACK,
                        default=entry.options.get(
                            CONF_USE_RECORDER_FALLBACK,
                            entry.data.get(
                                CONF_USE_RECORDER_FALLBACK, DEFAULT_USE_RECORDER_FALLBACK
                            ),
                        ),
                    ): bool,
                    vol.Optional(
                        CONF_FORECAST_HORIZON_HOURS,
                        default=entry.options.get(
                            CONF_FORECAST_HORIZON_HOURS,
                            entry.data.get(
                                CONF_FORECAST_HORIZON_HOURS, DEFAULT_FORECAST_HORIZON_HOURS
                            ),
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(min=12, max=96, step=1, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_UPDATE_INTERVAL_MINUTES,
                        default=entry.options.get(
                            CONF_UPDATE_INTERVAL_MINUTES,
                            entry.data.get(
                                CONF_UPDATE_INTERVAL_MINUTES,
                                DEFAULT_UPDATE_INTERVAL_MINUTES,
                            ),
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(min=1, max=60, step=1, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_MIN_TRAINING_SAMPLES,
                        default=entry.options.get(
                            CONF_MIN_TRAINING_SAMPLES,
                            entry.data.get(
                                CONF_MIN_TRAINING_SAMPLES, DEFAULT_MIN_TRAINING_SAMPLES
                            ),
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(min=48, max=8760, step=1, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_MAX_FEATURE_ENTITIES,
                        default=entry.options.get(
                            CONF_MAX_FEATURE_ENTITIES,
                            entry.data.get(
                                CONF_MAX_FEATURE_ENTITIES, DEFAULT_MAX_FEATURE_ENTITIES
                            ),
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(min=0, max=50, step=1, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_IMPORTANCE_THRESHOLD,
                        default=entry.options.get(
                            CONF_IMPORTANCE_THRESHOLD,
                            entry.data.get(
                                CONF_IMPORTANCE_THRESHOLD, DEFAULT_IMPORTANCE_THRESHOLD
                            ),
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0.0, max=0.1, step=0.001, mode=NumberSelectorMode.BOX
                        )
                    ),
                }
            ),
        )


