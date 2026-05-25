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
    ATTR_NET_ENERGY_NEXT_HOUR_KWH,
    ATTR_SIMULATION_STEPS,
    DOMAIN,
    SENSOR_TYPE_EMPTY_AT,
    SENSOR_TYPE_HOURS_REMAINING,
    SENSOR_TYPE_NET_LOAD,
    SENSOR_TYPE_PREDICTED_SOC,
)
from .coordinator import BatteryForecastCoordinator
from .model.simulator import ForecastResult

# Stable entity_id slugs (sensor.battery_empty_at, …)
ENTITY_OBJECT_IDS: dict[str, str] = {
    SENSOR_TYPE_EMPTY_AT: "battery_empty_at",
    SENSOR_TYPE_HOURS_REMAINING: "battery_hours_remaining",
    SENSOR_TYPE_PREDICTED_SOC: "battery_predicted_soc_1h",
    SENSOR_TYPE_NET_LOAD: "battery_net_load_next_hour",
}

SENSORS: tuple[SensorEntityDescription, ...] = (
    SensorEntityDescription(
        key=SENSOR_TYPE_EMPTY_AT,
        translation_key=SENSOR_TYPE_EMPTY_AT,
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
        key=SENSOR_TYPE_NET_LOAD,
        translation_key=SENSOR_TYPE_NET_LOAD,
        native_unit_of_measurement="kWh",
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
        if key == SENSOR_TYPE_HOURS_REMAINING:
            return data.hours_remaining
        if key == SENSOR_TYPE_PREDICTED_SOC:
            return data.predicted_soc_1h
        if key == SENSOR_TYPE_NET_LOAD:
            return data.net_load_next_hour_kwh
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        data = self.coordinator.data
        attrs = self.coordinator.model_attributes()
        if data is None:
            return attrs or None
        attrs[ATTR_CONFIDENCE] = data.confidence
        attrs[ATTR_NET_ENERGY_NEXT_HOUR_KWH] = data.net_load_next_hour_kwh
        if self.entity_description.key == SENSOR_TYPE_EMPTY_AT:
            attrs[ATTR_SIMULATION_STEPS] = data.simulation_steps[:24]
        return attrs
