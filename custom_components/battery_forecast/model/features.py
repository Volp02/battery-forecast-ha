"""Build hourly training features from HA statistics and recorder."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from ..const import SHORT_TERM_STATISTICS_DAYS

_LOGGER = logging.getLogger(__name__)

FEATURE_NAMES_BASE = [
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "month",
    "is_weekend",
    "outdoor_temp",
    "heat_pump_kw",
    "pv_kw",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _floor_hour(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def _entity_column(entity_id: str) -> str:
    return "f_" + entity_id.replace(".", "_").replace("-", "_")


def _parse_state_power(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_statistics_start(start_ts: Any) -> datetime | None:
    """Parse StatisticsRow start field (unix float in HA 2025+)."""
    if start_ts is None:
        return None
    if isinstance(start_ts, (int, float)):
        return datetime.fromtimestamp(start_ts, tz=timezone.utc)
    if isinstance(start_ts, str):
        parsed = datetime.fromisoformat(start_ts)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    if isinstance(start_ts, datetime):
        if start_ts.tzinfo is None:
            return start_ts.replace(tzinfo=timezone.utc)
        return start_ts
    return None


def _row_value(row: dict[str, Any]) -> float | None:
    for key in ("mean", "sum", "state", "max"):
        val = row.get(key)
        if val is not None:
            return float(val)
    return None


def _statistics_to_hourly(
    stats: dict[str, list[dict[str, Any]]],
    entity_ids: list[str],
) -> dict[str, dict[datetime, float]]:
    """Convert statistics rows to hourly buckets."""
    result: dict[str, dict[datetime, float]] = {eid: {} for eid in entity_ids}
    for eid, rows in stats.items():
        accum: dict[datetime, list[float]] = defaultdict(list)
        for row in rows:
            start_dt = _parse_statistics_start(row.get("start"))
            if start_dt is None:
                continue
            val = _row_value(row)
            if val is None:
                continue
            accum[_floor_hour(start_dt)].append(val)
        for hour, values in accum.items():
            result[eid][hour] = float(np.mean(values))
    return result


def _fetch_statistics_sync(
    hass: Any,
    entity_ids: list[str],
    start: datetime,
    end: datetime,
    period: str,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch statistics (sync — run in executor)."""
    from homeassistant.components.recorder import get_instance

    if not entity_ids:
        return {}

    if get_instance(hass) is None:
        _LOGGER.warning("Recorder not available")
        return {}

    try:
        from homeassistant.components.recorder.statistics import statistics_during_period
    except ImportError as err:
        raise ImportError(
            "Home Assistant recorder API statistics_during_period not found. "
            "Requires Home Assistant 2025.5 or newer."
        ) from err

    return statistics_during_period(
        hass,
        start_time=start,
        end_time=end,
        statistic_ids=set(entity_ids),
        period=period,  # type: ignore[arg-type]
        units=None,
        types={"mean", "sum", "state"},
    )


async def _fetch_statistics_hourly(
    hass: Any,
    entity_ids: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, dict[datetime, float]]:
    """Fetch statistics and normalize to hourly values."""
    if not entity_ids:
        return {}

    result: dict[str, dict[datetime, float]] = {eid: {} for eid in entity_ids}

    # Long-term / hourly statistics for full training window
    stats_hour = await hass.async_add_executor_job(
        _fetch_statistics_sync, hass, entity_ids, start, end, "hour"
    )
    result = _statistics_to_hourly(stats_hour, entity_ids)

    # Short-term fine statistics (~10 days) override hourly buckets
    short_start = max(start, end - timedelta(days=SHORT_TERM_STATISTICS_DAYS))
    if short_start < end:
        try:
            stats_fine = await hass.async_add_executor_job(
                _fetch_statistics_sync,
                hass,
                entity_ids,
                short_start,
                end,
                "5minute",
            )
            fine_hourly = _statistics_to_hourly(stats_fine, entity_ids)
            for eid in entity_ids:
                if eid not in result:
                    result[eid] = {}
                for hour, val in fine_hourly.get(eid, {}).items():
                    result[eid][hour] = val
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("5-minute statistics unavailable: %s", err)

    return result


def _state_time_and_value(state: Any) -> tuple[datetime | None, Any]:
    """Extract timestamp and value from State or compressed history dict."""
    if isinstance(state, dict):
        ts = state.get("last_changed") or state.get("last_updated")
        if ts is None and "lu" in state:
            ts = datetime.fromtimestamp(state["lu"], tz=timezone.utc)
        if ts is None and "lc" in state:
            ts = datetime.fromtimestamp(state["lc"], tz=timezone.utc)
        raw = state.get("state", state.get("s"))
        return ts, raw
    ts = getattr(state, "last_changed", None) or getattr(state, "last_updated", None)
    return ts, getattr(state, "state", None)


def _fetch_recorder_sync(
    hass: Any,
    entity_ids: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, list[Any]]:
    """Fetch recorder states (sync — run in executor)."""
    from homeassistant.components.recorder import history

    return history.get_significant_states(
        hass,
        start,
        end,
        entity_ids,
        None,
        include_start_time_state=True,
        significant_changes_only=False,
        minimal_response=False,
        no_attributes=True,
    )


async def _fetch_recorder_hourly(
    hass: Any,
    entity_ids: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, dict[datetime, float]]:
    """Resample recorder states to hourly mean power (W)."""
    result: dict[str, dict[datetime, float]] = {eid: {} for eid in entity_ids}
    if not entity_ids:
        return result

    states = await hass.async_add_executor_job(
        _fetch_recorder_sync, hass, entity_ids, start, end
    )

    accum: dict[str, dict[datetime, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for eid, state_list in states.items():
        for state in state_list:
            ts, raw = _state_time_and_value(state)
            if ts is None:
                continue
            if isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            val = _parse_state_power(raw)
            if val is None:
                continue
            accum[eid][_floor_hour(ts)].append(val)

    for eid, hours in accum.items():
        for hour, values in hours.items():
            result[eid][hour] = float(np.mean(values))

    return result


async def merge_power_series(
    hass: Any,
    entity_ids: list[str],
    start: datetime,
    end: datetime,
    *,
    use_recorder_fallback: bool,
    recorder_days: int = 30,
) -> dict[str, dict[datetime, float]]:
    """Merge statistics (1y) with optional recorder override for recent days."""
    stats = await _fetch_statistics_hourly(hass, entity_ids, start, end)
    if not use_recorder_fallback:
        return stats

    rec_start = max(start, end - timedelta(days=recorder_days))
    try:
        recorder = await _fetch_recorder_hourly(hass, entity_ids, rec_start, end)
    except Exception as err:  # noqa: BLE001
        _LOGGER.warning("Recorder history fallback skipped: %s", err)
        return stats

    for eid in entity_ids:
        if eid not in stats:
            stats[eid] = {}
        for hour, val in recorder.get(eid, {}).items():
            stats[eid][hour] = val
    return stats


def _w_to_kw(v: float | None) -> float:
    if v is None:
        return 0.0
    # Energy statistics may already be kWh sums — heuristic: large values = energy
    if abs(v) > 500:
        return v / 1000.0
    return v / 1000.0


def build_hourly_dataset(
    power_series: dict[str, dict[datetime, float]],
    *,
    house_power: str,
    pv_power: str | None,
    heat_pump_power: str | None,
    outdoor_temp: str | None,
    feature_entities: list[str],
    start: datetime,
    end: datetime,
) -> tuple[np.ndarray, np.ndarray, list[str], list[datetime]]:
    """Return X, y, feature_names, hour_index."""
    hours: list[datetime] = []
    cursor = _floor_hour(start)
    end_floor = _floor_hour(end)
    while cursor <= end_floor:
        hours.append(cursor)
        cursor += timedelta(hours=1)

    feature_cols = list(FEATURE_NAMES_BASE)
    for fe in feature_entities:
        col = _entity_column(fe)
        if col not in feature_cols:
            feature_cols.append(col)

    rows: list[list[float]] = []
    targets: list[float] = []

    house = power_series.get(house_power, {})
    pv = power_series.get(pv_power, {}) if pv_power else {}
    hp = power_series.get(heat_pump_power, {}) if heat_pump_power else {}
    temp = power_series.get(outdoor_temp, {}) if outdoor_temp else {}

    for hour in hours:
        house_w = house.get(hour)
        if house_w is None:
            continue

        pv_kw = _w_to_kw(pv.get(hour)) if pv_power else 0.0
        hp_kw = _w_to_kw(hp.get(hour)) if heat_pump_power else 0.0
        temp_c = temp.get(hour) if outdoor_temp else float("nan")

        feature_kw = 0.0
        row = [
            math.sin(2 * math.pi * hour.hour / 24),
            math.cos(2 * math.pi * hour.hour / 24),
            float(hour.weekday()),
            float(hour.month),
            1.0 if hour.weekday() >= 5 else 0.0,
            temp_c if temp_c is not None else float("nan"),
            hp_kw,
            pv_kw,
        ]

        for fe in feature_entities:
            col_val = _w_to_kw(power_series.get(fe, {}).get(hour))
            row.append(col_val)
            feature_kw += col_val

        house_kw = _w_to_kw(house_w)
        net_load = max(0.0, house_kw + hp_kw + feature_kw - pv_kw)
        rows.append(row)
        targets.append(net_load)

    if not rows:
        return (
            np.empty((0, len(feature_cols))),
            np.empty(0),
            feature_cols,
            [],
        )

    return (
        np.array(rows, dtype=np.float64),
        np.array(targets, dtype=np.float64),
        feature_cols,
        hours,
    )


def compute_sample_weights(
    hours: list[datetime],
    *,
    half_life_days: float,
    reference: datetime | None = None,
) -> np.ndarray:
    """Exponential decay: recent hours weighted higher."""
    if not hours:
        return np.empty(0)
    ref = reference or _utc_now()
    weights = []
    for hour in hours:
        age_days = max(0.0, (ref - hour).total_seconds() / 86400.0)
        weights.append(math.exp(-age_days / max(half_life_days, 1.0)))
    return np.array(weights, dtype=np.float64)


def build_inference_features(
    base_time: datetime,
    horizon_hours: int,
    *,
    outdoor_temp: float | None,
    heat_pump_kw: float,
    pv_kw: float,
    feature_kw_map: dict[str, float],
    feature_entities: list[str],
    feature_names: list[str],
) -> np.ndarray:
    """Build feature matrix for future hours."""
    rows = []
    t = _floor_hour(base_time)
    for i in range(horizon_hours):
        hour = t + timedelta(hours=i)
        row_dict = {
            "hour_sin": math.sin(2 * math.pi * hour.hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour.hour / 24),
            "day_of_week": float(hour.weekday()),
            "month": float(hour.month),
            "is_weekend": 1.0 if hour.weekday() >= 5 else 0.0,
            "outdoor_temp": outdoor_temp if outdoor_temp is not None else float("nan"),
            "heat_pump_kw": heat_pump_kw,
            "pv_kw": pv_kw,
        }
        for fe in feature_entities:
            row_dict[_entity_column(fe)] = feature_kw_map.get(fe, 0.0)

        row = [row_dict.get(name, float("nan")) for name in feature_names]
        rows.append(row)
    return np.array(rows, dtype=np.float64)


async def load_training_data(
    hass: Any,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[datetime]]:
    """Load and merge all sources into training matrices."""
    training_days = int(config.get("training_days", 365))
    end = _utc_now()
    start = end - timedelta(days=training_days)

    entity_ids: list[str] = [config["house_power"]]
    if config.get("pv_power"):
        entity_ids.append(config["pv_power"])
    if config.get("heat_pump_power"):
        entity_ids.append(config["heat_pump_power"])
    if config.get("outdoor_temp"):
        entity_ids.append(config["outdoor_temp"])
    feature_entities: list[str] = list(config.get("feature_entities") or [])[
        : int(config.get("max_feature_entities", 30))
    ]
    entity_ids.extend(feature_entities)

    power_series = await merge_power_series(
        hass,
        list(dict.fromkeys(entity_ids)),
        start,
        end,
        use_recorder_fallback=config.get("use_recorder_fallback", True),
        recorder_days=min(30, training_days),
    )

    X, y, feature_names, hour_list = build_hourly_dataset(
        power_series,
        house_power=config["house_power"],
        pv_power=config.get("pv_power"),
        heat_pump_power=config.get("heat_pump_power"),
        outdoor_temp=config.get("outdoor_temp"),
        feature_entities=feature_entities,
        start=start,
        end=end,
    )

    weights = compute_sample_weights(
        hour_list,
        half_life_days=float(config.get("sample_half_life_days", 90)),
        reference=end,
    )
    return X, y, weights, feature_names, hour_list
