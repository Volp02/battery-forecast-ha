# Battery Forecast (Home Assistant)

HACS-ready custom integration that predicts when your home battery will reach a low state of charge, using **machine learning** (numpy) on Home Assistant statistics (short-term + long-term) and optional recorder data.

## Features

- **Config flow** with entity selectors for battery, PV, house load, heat pump, weather, and up to 30 optional feature sensors
- **ML model** trained on up to **365 days** of hourly data (weighted linear regression via numpy)
- **Weighted training**: recent weeks count more (configurable half-life, default 90 days)
- **Hybrid forecast**: ML predicts hourly net load → SOC simulation with solar forecast entities
- **Sensors**: empty-at timestamp, hours remaining, min SOC in 12h, predicted SOC in 1h, net load next hour
- **Auto-retrain** when forecast SOC error exceeds a threshold (no external automation required)
- **Services**: `battery_forecast.train`, `battery_forecast.reload_model`

## Requirements

- Home Assistant **2025.5+**
- Working **Recorder** and **Statistics** for your entities
- Long-term statistics retention ≥ your training window (default **365 days**)
- Python package (installed automatically by HA): **numpy**

## Installation

### HACS

1. **HACS → Settings → Custom repositories** → URL: `https://github.com/Volp02/battery-forecast-ha`, Category: **Integration**
2. **HACS → Integrations** → Battery Forecast → **Download** → **`v1.0`**
3. Restart Home Assistant
4. **Settings → Devices & services → Add integration → Battery Forecast**

If the version list is empty: **HACS → Integrations → Battery Forecast** → three dots → **Update repository information**, then redownload. The GitHub release must be **published** (not draft).

Fallback: download branch **`master`**.

### Manual

Copy `custom_components/battery_forecast` into your `config/custom_components/` folder and restart HA.

## Configuration

| Step | Description |
|------|-------------|
| Battery | SOC (%), capacity (kWh), power (W); optional charge/discharge energy |
| Solar | PV power; optional forecast today/tomorrow (e.g. Forecast.Solar, Solcast) |
| Loads | **House** instant power (required, not grid import); optional heat pump |
| Environment | Outdoor temperature; optional weather entity |
| Features | Optional list of power/energy sensors (EV, boiler, …) |
| ML | Empty SOC %, training days (365), half-life, horizon, intervals |

### Example entities

| Role | Example |
|------|---------|
| SOC | `sensor.battery_soc` |
| Power | `sensor.battery_power` (W, + = charging) |
| House load | `sensor.momentanleistung_haus_3` (total home consumption) |
| Grid (do **not** use for house) | `sensor.gesamtverbrauch_haus_2` — near 0 W when PV/battery cover load |
| PV | `sensor.solar_power` |
| Forecast | `sensor.forecast_today` / `sensor.solcast_pv_forecast_forecast_today` |
| Heat pump | `sensor.heat_pump_power` |
| Outdoor temp | `sensor.outdoor_temperature` |

## Training

After setup, train the model (can take several minutes):

```yaml
service: battery_forecast.train
```

Or use **Developer tools → Services**.

Training uses:

1. **Short-term statistics** (~10 days, fine resolution)
2. **Long-term statistics** (up to `training_days`, default 365)
3. **Recorder** (optional, recent high-resolution override)

### Auto-retrain (built-in, no automation required)

Enabled by default. After each forecast update the integration compares **predicted vs. actual SOC** for past hours. If the mean error exceeds the threshold (default **12 %**) and the last auto-train was at least **24 h** ago, it runs `battery_forecast.train` in the background.

Configure under **Battery Forecast → Options** (or the ML step on first setup):

| Option | Default |
|--------|---------|
| Auto-retrain enabled | on |
| SOC error threshold | 12 % |
| Min. interval | 24 h |
| Evaluation window | 24 h |

Sensor attributes: `forecast_soc_mae_24h`, `auto_retrain_last_at`.

Manual `battery_forecast.train` still works anytime.

## Sensors

One device **Battery Forecast** with five entities (friendly names are translated):

| Entity ID | Friendly name (EN) | Description |
|-----------|-------------------|-------------|
| `sensor.battery_empty_at` | Empty at | First timestamp when simulated SOC ≤ threshold |
| `sensor.battery_hours_remaining` | Hours remaining | Hours until empty (within horizon) |
| `sensor.battery_predicted_soc_1h` | Predicted SOC in 1h | SOC after one simulated hour of simulation |
| `sensor.battery_net_load_next_hour` | Net load next hour | ML estimate (kWh) |
| `sensor.battery_min_soc_12h` | Min SOC in 12 h | Lowest simulated SOC in the next 12 hours |

PV forecast today/tomorrow (kWh) is used hourly in the SOC simulation (charging effect during daylight). Attributes `pv_forecast_today_kwh` / `pv_forecast_tomorrow_kwh` show the values used.

After upgrading from older builds, remove stale `sensor.battery_forecast` / `_2` … entities in **Settings → Devices & services** if they remain as orphaned entities.

Attributes include `model_type` (always `numpy`), `confidence`, `mae_kwh`, `r2`, `model_trained_at`, `feature_importances`, and `simulation_steps` (first 24h).

**ML note:** Training uses the **house consumption** sensor (`house_power`), **not** grid import. Optional feature sensors (washer, dryer, EV, …) are **supplementary** — you can use zero or a few; the model must not depend on them. They are **not** added on top of house power (no double counting). At **forecast** time those optional sensors are treated as **off (0 W)** for all future hours (we do not know if the washer will run); **house power**, time-of-day, heat pump, PV, and temperature drive the prediction.

### Outdoor temperature and weather

| Input | Used in ML? | How |
|-------|-------------|-----|
| **Outdoor temperature** (`outdoor_temp`) | **Yes** | Hourly values from **statistics** during training; at forecast time the **current** temperature is applied to all future hours (simple; no hourly weather forecast yet). |
| **Weather entity** (`weather_entity`) | **No** | Stored in config for a future release (e.g. forecast temperature). Pick a sensor with `device_class: temperature` or similar for `outdoor_temp`. |

If `outdoor_temp` is omitted, the column is imputed (median / NaN handling) and time-of-day features still apply.

### Training progress in logs

Filter logs with `battery_forecast`. Typical sequence:

```text
Battery Forecast: train service started
Battery Forecast: loading training data (365 days, …)
Battery Forecast: dataset ready — N samples, M features
Battery Forecast: fitting weighted linear model (numpy)
Battery Forecast: train complete — model=numpy mae=… kWh
```

## Data tips

- Ensure **statistics** are enabled for power sensors (default for `state_class: measurement`).
- **1 year** of long-term data is enough for seasonal patterns; older data is not required.
- **Feature sensors are optional.** House consumption power alone is enough. Add at most a few large loads (heat pump, EV) if you want; omit washer/dryer unless you need them for training hints — they must not be required for a sensible forecast.
- If training fails with “not enough samples”, reduce `min_training_samples` or check entity history in **History** graph.

## Manual test checklist

- [ ] Config flow completes with your entities
- [ ] `battery_forecast.train` finishes without error in logs
- [ ] All five sensors become available
- [ ] `hours_remaining` plausible during evening discharge
- [ ] Changing **empty SOC %** in options shifts `empty_at`
- [ ] `feature_importances` attribute lists sensible sensors after train

## License

MIT
