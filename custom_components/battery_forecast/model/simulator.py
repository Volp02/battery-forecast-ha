"""SOC forward simulation and empty-at prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from .features import build_inference_features
from .trainer import ModelBundle, predict_load_kwh

_LOGGER = logging.getLogger(__name__)

PV_DAYLIGHT_START = 8
PV_DAYLIGHT_END = 18


@dataclass
class ForecastResult:
    """Coordinator data object."""

    empty_at: datetime | None
    hours_remaining: float | None
    empty_within_horizon: bool
    soc_at_horizon: float | None
    predicted_soc_1h: float | None
    net_load_next_hour_kwh: float | None
    confidence: float
    simulation_steps: list[dict[str, Any]]


def _parse_float(state: Any) -> float | None:
    if state is None:
        return None
    try:
        return float(state)
    except (TypeError, ValueError):
        return None


def _read_entity_kw(hass: Any, entity_id: str | None) -> float:
    if not entity_id:
        return 0.0
    state = hass.states.get(entity_id)
    if state is None:
        return 0.0
    val = _parse_float(state.state)
    if val is None:
        return 0.0
    if abs(val) > 500:
        return val / 1000.0
    return val / 1000.0


def _read_entity_percent(hass: Any, entity_id: str) -> float:
    state = hass.states.get(entity_id)
    if state is None:
        return 0.0
    val = _parse_float(state.state)
    return val if val is not None else 0.0


def _distribute_daily_pv_kwh(daily_kwh: float, hour: int, profile: np.ndarray | None) -> float:
    """Spread daily PV kWh across daylight hours."""
    if daily_kwh <= 0:
        return 0.0
    if profile is not None and len(profile) == 24:
        share = profile[hour]
        total = profile[PV_DAYLIGHT_START:PV_DAYLIGHT_END].sum()
        if total > 0:
            return daily_kwh * (share / total) if PV_DAYLIGHT_START <= hour < PV_DAYLIGHT_END else 0.0
    if PV_DAYLIGHT_START <= hour < PV_DAYLIGHT_END:
        span = PV_DAYLIGHT_END - PV_DAYLIGHT_START
        return daily_kwh / span
    return 0.0


def build_pv_hourly_profile(hass: Any, pv_power: str | None) -> np.ndarray | None:
    """Build normalized 24h PV shape from recent states (optional)."""
    if not pv_power:
        return None
    profile = np.zeros(24)
    state = hass.states.get(pv_power)
    if state is None:
        return None
    # Use last hour attribute pattern if unavailable — flat fallback in distribute
    for h in range(PV_DAYLIGHT_START, PV_DAYLIGHT_END):
        profile[h] = 1.0
    if profile.sum() > 0:
        profile /= profile.sum()
    return profile


def get_pv_forecast_kwh_by_hour(
    hass: Any,
    base_time: datetime,
    horizon_hours: int,
    *,
    pv_forecast_today: str | None,
    pv_forecast_tomorrow: str | None,
    pv_power: str | None,
    profile: np.ndarray | None,
) -> list[float]:
    """Hourly PV kWh for simulation horizon."""
    today_kwh = _read_entity_kw(hass, pv_forecast_today) if pv_forecast_today else 0.0
    tomorrow_kwh = (
        _read_entity_kw(hass, pv_forecast_tomorrow) if pv_forecast_tomorrow else 0.0
    )
    if today_kwh <= 0 and pv_power:
        today_kwh = max(0.0, _read_entity_kw(hass, pv_power)) * 0.001

    result: list[float] = []
    t = base_time.replace(minute=0, second=0, microsecond=0)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)

    for i in range(horizon_hours):
        hour_dt = t + timedelta(hours=i)
        day_offset = (hour_dt.date() - t.date()).days
        daily = tomorrow_kwh if day_offset >= 1 else today_kwh
        result.append(_distribute_daily_pv_kwh(daily, hour_dt.hour, profile))
    return result


def simulate_soc(
    *,
    current_soc: float,
    capacity_kwh: float,
    empty_soc_percent: float,
    horizon_hours: int,
    load_kwh_per_hour: list[float],
    pv_kwh_per_hour: list[float],
    start_time: datetime,
) -> ForecastResult:
    """Run forward SOC simulation."""
    soc = current_soc
    steps: list[dict[str, Any]] = []
    empty_at: datetime | None = None

    if capacity_kwh <= 0:
        return ForecastResult(
            empty_at=None,
            hours_remaining=None,
            empty_within_horizon=False,
            soc_at_horizon=None,
            predicted_soc_1h=None,
            net_load_next_hour_kwh=load_kwh_per_hour[0] if load_kwh_per_hour else None,
            confidence=0.0,
            simulation_steps=[],
        )

    for i in range(horizon_hours):
        load = load_kwh_per_hour[i] if i < len(load_kwh_per_hour) else 0.0
        pv = pv_kwh_per_hour[i] if i < len(pv_kwh_per_hour) else 0.0
        soc_change = ((pv - load) / capacity_kwh) * 100.0
        soc = max(0.0, min(100.0, soc + soc_change))
        step_time = start_time + timedelta(hours=i)
        steps.append(
            {
                "hour": step_time.isoformat(),
                "soc": round(soc, 2),
                "load_kwh": round(load, 3),
                "pv_kwh": round(pv, 3),
                "soc_change": round(soc_change, 3),
            }
        )
        if empty_at is None and soc <= empty_soc_percent:
            empty_at = step_time

    soc_at_horizon = steps[-1]["soc"] if steps else current_soc
    empty_within_horizon = empty_at is not None
    if empty_at is not None:
        hours_remaining = (empty_at - start_time).total_seconds() / 3600.0
    else:
        # Battery still above threshold after horizon — show horizon as lower bound.
        hours_remaining = float(horizon_hours)

    predicted_soc_1h = steps[0]["soc"] if len(steps) > 1 else (steps[0]["soc"] if steps else current_soc)
    if len(steps) >= 2:
        predicted_soc_1h = steps[1]["soc"]

    return ForecastResult(
        empty_at=empty_at,
        hours_remaining=hours_remaining,
        empty_within_horizon=empty_within_horizon,
        soc_at_horizon=soc_at_horizon,
        predicted_soc_1h=predicted_soc_1h,
        net_load_next_hour_kwh=load_kwh_per_hour[0] if load_kwh_per_hour else None,
        confidence=0.0,
        simulation_steps=steps,
    )


def run_forecast(
    hass: Any,
    config: dict[str, Any],
    bundle: ModelBundle,
) -> ForecastResult:
    """Full inference + simulation pipeline."""
    horizon = int(config.get("forecast_horizon_hours", 48))
    now = datetime.now(timezone.utc)

    current_soc = _read_entity_percent(hass, config["battery_soc"])
    capacity = float(config.get("battery_capacity_kwh", 10.0))
    empty_soc = float(config.get("empty_soc_percent", 10.0))

    outdoor_temp = None
    if config.get("outdoor_temp"):
        state = hass.states.get(config["outdoor_temp"])
        if state:
            outdoor_temp = _parse_float(state.state)

    hp_kw = _read_entity_kw(hass, config.get("heat_pump_power"))
    pv_kw = _read_entity_kw(hass, config.get("pv_power"))

    feature_entities: list[str] = list(config.get("feature_entities") or [])

    X = build_inference_features(
        now,
        horizon,
        outdoor_temp=outdoor_temp,
        hourly_house_kw=bundle.hourly_house_kw,
        heat_pump_kw=hp_kw,
        pv_kw=pv_kw,
        feature_entities=feature_entities,
        feature_names=bundle.feature_names,
    )

    loads = predict_load_kwh(bundle, X).tolist()
    profile = build_pv_hourly_profile(hass, config.get("pv_power"))
    pv_series = get_pv_forecast_kwh_by_hour(
        hass,
        now,
        horizon,
        pv_forecast_today=config.get("pv_forecast"),
        pv_forecast_tomorrow=config.get("pv_forecast_tomorrow"),
        pv_power=config.get("pv_power"),
        profile=profile,
    )

    result = simulate_soc(
        current_soc=current_soc,
        capacity_kwh=capacity,
        empty_soc_percent=empty_soc,
        horizon_hours=horizon,
        load_kwh_per_hour=loads,
        pv_kwh_per_hour=pv_series,
        start_time=now.replace(minute=0, second=0, microsecond=0),
    )

    # Confidence from model metrics
    mae = max(bundle.mae_kwh, 0.01)
    confidence = max(0.0, min(1.0, 1.0 - (mae / 2.0)))
    result.confidence = round(confidence, 3)
    return result
