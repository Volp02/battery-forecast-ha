"""Compare past forecasts to actual SOC and trigger auto-retrain."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

_LOGGER = logging.getLogger(__name__)

MAX_SNAPSHOTS = 96


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_hour(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (TypeError, ValueError):
        return None


def snapshot_from_result(result: Any, recorded_at: datetime | None = None) -> dict[str, Any]:
    """Build a storable forecast snapshot."""
    when = recorded_at or _utc_now()
    steps = []
    for step in result.simulation_steps[:48]:
        steps.append(
            {
                "hour": step["hour"],
                "soc": float(step["soc"]),
                "load_kwh": float(step.get("load_kwh", 0)),
            }
        )
    return {
        "recorded_at": when.isoformat(),
        "steps": steps,
    }


def append_snapshot(eval_data: dict[str, Any], result: Any) -> None:
    """Add snapshot; keep one entry per forecast hour start."""
    snapshots: list[dict[str, Any]] = list(eval_data.get("snapshots") or [])
    new_snap = snapshot_from_result(result)
    if not new_snap["steps"]:
        return
    first_hour = _parse_hour(new_snap["steps"][0]["hour"])
    if first_hour is not None:
        snapshots = [
            s
            for s in snapshots
            if not (
                s.get("steps")
                and _parse_hour(s["steps"][0]["hour"]) == first_hour
            )
        ]
    snapshots.append(new_snap)
    eval_data["snapshots"] = snapshots[-MAX_SNAPSHOTS:]


def _fetch_hourly_soc_sync(
    hass: Any,
    entity_id: str,
    start: datetime,
    end: datetime,
) -> dict[datetime, float]:
    from .features import _fetch_statistics_sync, _statistics_to_hourly

    stats = _fetch_statistics_sync(hass, [entity_id], start, end, "hour")
    hourly = _statistics_to_hourly(stats, [entity_id])
    return hourly.get(entity_id, {})


async def compute_forecast_soc_mae(
    hass: Any,
    *,
    battery_soc: str,
    eval_data: dict[str, Any],
    eval_hours: int,
    min_samples: int = 6,
) -> float | None:
    """Mean absolute error (%) between predicted and actual SOC for past hours."""
    now = _utc_now()
    window_start = now - timedelta(hours=eval_hours)
    settle_before = now - timedelta(hours=1)

    actual = await hass.async_add_executor_job(
        _fetch_hourly_soc_sync,
        hass,
        battery_soc,
        window_start - timedelta(hours=1),
        now,
    )

    errors: list[float] = []
    for snapshot in eval_data.get("snapshots") or []:
        for step in snapshot.get("steps") or []:
            hour = _parse_hour(step.get("hour", ""))
            if hour is None:
                continue
            if hour < window_start or hour >= settle_before:
                continue
            actual_soc = actual.get(hour.replace(minute=0, second=0, microsecond=0))
            if actual_soc is None:
                hour_key = hour.replace(minute=0, second=0, microsecond=0)
                for ah, val in actual.items():
                    if abs((ah - hour_key).total_seconds()) < 1800:
                        actual_soc = val
                        break
            if actual_soc is None:
                continue
            errors.append(abs(float(step["soc"]) - float(actual_soc)))

    if len(errors) < min_samples:
        return None
    return float(sum(errors) / len(errors))


def hours_since(iso_ts: str | None) -> float | None:
    if not iso_ts:
        return None
    try:
        then = datetime.fromisoformat(iso_ts)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return (_utc_now() - then).total_seconds() / 3600.0
    except (TypeError, ValueError):
        return None


def should_auto_retrain(
    *,
    mae_percent: float | None,
    threshold_percent: float,
    last_auto_train_at: str | None,
    min_interval_hours: float,
) -> bool:
    if mae_percent is None:
        return False
    if mae_percent < threshold_percent:
        return False
    elapsed = hours_since(last_auto_train_at)
    if elapsed is not None and elapsed < min_interval_hours:
        return False
    return True
