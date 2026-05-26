"""Battery Forecast sensor platform."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    ATTR_CONFIDENCE,
    ATTR_BATTERY_POWER_KW,
    ATTR_EMPTY_AT_EXTRAPOLATED,
    ATTR_EMPTY_WITHIN_HORIZON,
    ATTR_FORECAST_HORIZON_HOURS,
    ATTR_HOUSE_POWER_ENTITY,
    ATTR_NET_ENERGY_NEXT_HOUR_KWH,
    ATTR_SIMULATION_STEPS,
    ATTR_SOC_AT_HORIZON,
    DOMAIN,
    SENSOR_TYPE_EMPTY_AT,
    SENSOR_TYPE_FULL_AT,
    SENSOR_TYPE_HOURS_REMAINING,
    SENSOR_TYPE_MIN_SOC_12H,
    SENSOR_TYPE_NET_LOAD,
    SENSOR_TYPE_PREDICTED_SOC,
    SENSOR_TYPE_PREDICTED_SOC_2H,
    SENSOR_TYPE_PREDICTED_SOC_4H,
    SENSOR_TYPE_PREDICTED_SOC_6H,
)
from .coordinator import BatteryForecastCoordinator
from .helpers import get_config
from .model.simulator import ForecastResult

# Stable entity_id slugs (sensor.battery_empty_at, …)
ENTITY_OBJECT_IDS: dict[str, str] = {
    SENSOR_TYPE_EMPTY_AT: "battery_empty_at",
    SENSOR_TYPE_FULL_AT: "battery_full_at",
    SENSOR_TYPE_HOURS_REMAINING: "battery_hours_remaining",
    SENSOR_TYPE_PREDICTED_SOC: "battery_predicted_soc_1h",
    SENSOR_TYPE_PREDICTED_SOC_2H: "battery_predicted_soc_2h",
    SENSOR_TYPE_PREDICTED_SOC_4H: "battery_predicted_soc_4h",
    SENSOR_TYPE_PREDICTED_SOC_6H: "battery_predicted_soc_6h",
    SENSOR_TYPE_NET_LOAD: "battery_net_load_next_hour",
    SENSOR_TYPE_MIN_SOC_12H: "battery_min_soc_12h",
}

SENSORS: tuple[SensorEntityDescription, ...] = (
    SensorEntityDescription(
        key=SENSOR_TYPE_EMPTY_AT,
        translation_key=SENSOR_TYPE_EMPTY_AT,
        device_class=SensorDeviceClass.TIMESTAMP,
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_FULL_AT,
        translation_key=SENSOR_TYPE_FULL_AT,
        device_class=SensorDeviceClass.TIMESTAMP,
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_HOURS_REMAINING,
        translation_key=SENSOR_TYPE_HOURS_REMAINING,
        native_unit_of_measurement="h",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_PREDICTED_SOC,
        translation_key=SENSOR_TYPE_PREDICTED_SOC,
        native_unit_of_measurement="%",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_PREDICTED_SOC_2H,
        translation_key=SENSOR_TYPE_PREDICTED_SOC_2H,
        native_unit_of_measurement="%",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_PREDICTED_SOC_4H,
        translation_key=SENSOR_TYPE_PREDICTED_SOC_4H,
        native_unit_of_measurement="%",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_PREDICTED_SOC_6H,
        translation_key=SENSOR_TYPE_PREDICTED_SOC_6H,
        native_unit_of_measurement="%",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_NET_LOAD,
        translation_key=SENSOR_TYPE_NET_LOAD,
        native_unit_of_measurement="kWh",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key=SENSOR_TYPE_MIN_SOC_12H,
        translation_key=SENSOR_TYPE_MIN_SOC_12H,
        native_unit_of_measurement="%",
        state_class=SensorStateClass.MEASUREMENT,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: BatteryForecastCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        BatteryForecastSensor(coordinator, description) for description in SENSORS
    )


class BatteryForecastSensor(CoordinatorEntity[BatteryForecastCoordinator], SensorEntity):
    """Battery forecast sensor."""

    entity_description: SensorEntityDescription
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: BatteryForecastCoordinator,
        description: SensorEntityDescription,
    ) -> None:
        super().__init__(coordinator)
        self.entity_description = description
        object_id = ENTITY_OBJECT_IDS[description.key]
        entry_id = coordinator.config_entry.entry_id
        self._attr_unique_id = f"{entry_id}_{object_id}"
        self._attr_suggested_object_id = object_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry_id)},
            name="Battery Forecast",
            manufacturer="Battery Forecast",
            model="ML Forecast",
        )

    @property
    def available(self) -> bool:
        return (
            self.coordinator.model_bundle is not None
            and self.coordinator.last_update_success
        )

    @property
    def native_value(self) -> StateType:
        data: ForecastResult | None = self.coordinator.data
        if data is None:
            return None
        key = self.entity_description.key
        if key == SENSOR_TYPE_EMPTY_AT:
            return data.empty_at
        if key == SENSOR_TYPE_FULL_AT:
            return data.full_at
        if key == SENSOR_TYPE_HOURS_REMAINING:
            if data.hours_remaining is not None:
                return round(min(data.hours_remaining, 24.0), 1)
            return 24.0
        if key == SENSOR_TYPE_PREDICTED_SOC:
            return data.predicted_soc_1h
        if key == SENSOR_TYPE_PREDICTED_SOC_2H:
            return data.predicted_soc_2h
        if key == SENSOR_TYPE_PREDICTED_SOC_4H:
            return data.predicted_soc_4h
        if key == SENSOR_TYPE_PREDICTED_SOC_6H:
            return data.predicted_soc_6h
        if key == SENSOR_TYPE_NET_LOAD:
            return data.net_load_next_hour_kwh
        if key == SENSOR_TYPE_MIN_SOC_12H:
            return data.min_soc_next_12h
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        data = self.coordinator.data
        attrs = self.coordinator.model_attributes()
        if data is None:
            return attrs or None
        attrs[ATTR_CONFIDENCE] = data.confidence
        attrs[ATTR_NET_ENERGY_NEXT_HOUR_KWH] = data.net_load_next_hour_kwh
        attrs[ATTR_EMPTY_WITHIN_HORIZON] = data.empty_within_horizon
        attrs[ATTR_EMPTY_AT_EXTRAPOLATED] = data.empty_at_extrapolated
        attrs[ATTR_BATTERY_POWER_KW] = data.battery_power_kw
        attrs[ATTR_SOC_AT_HORIZON] = data.soc_at_horizon
        config = get_config(self.coordinator.hass, self.coordinator.config_entry)
        attrs[ATTR_FORECAST_HORIZON_HOURS] = int(
            config.get("forecast_horizon_hours", 48)
        )
        attrs[ATTR_HOUSE_POWER_ENTITY] = config.get("house_power")
        if data.pv_forecast_today_kwh is not None:
            attrs["pv_forecast_today_kwh"] = data.pv_forecast_today_kwh
        if data.pv_forecast_tomorrow_kwh is not None:
            attrs["pv_forecast_tomorrow_kwh"] = data.pv_forecast_tomorrow_kwh
        if self.entity_description.key == SENSOR_TYPE_EMPTY_AT:
            attrs[ATTR_SIMULATION_STEPS] = data.simulation_steps[:24]
        return attrs
