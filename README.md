# Battery Forecast (Home Assistant)

HACS-ready custom integration that predicts when your home battery will reach a low state of charge, using **machine learning** on Home Assistant statistics (short-term + long-term) and optional recorder data.

## Features

- **Config flow** with entity selectors for battery, PV, house load, heat pump, weather, and up to 30 optional feature sensors
- **ML model** (`HistGradientBoostingRegressor`) trained on up to **365 days** of hourly data
- **Weighted training**: recent weeks count more (configurable half-life, default 90 days)
- **Hybrid forecast**: ML predicts hourly net load → SOC simulation with solar forecast entities
- **Sensors**: empty-at timestamp, hours remaining, predicted SOC in 1h, net load next hour
- **Services**: `battery_forecast.train`, `battery_forecast.reload_model`

## Requirements

- Home Assistant **2024.1+**
- Working **Recorder** and **Statistics** for your entities
- Long-term statistics retention ≥ your training window (default **365 days**)
- Python package (installed automatically): `numpy`
- **`scikit-learn`** optional but recommended (better accuracy; see below)

## Installation

### HACS

1. **HACS → Settings → Custom repositories** → URL: `https://github.com/Volp02/battery-forecast-ha`, Category: **Integration**
2. **HACS → Integrations** → Battery Forecast → Download version **`v0.1b`** (pre-release)
3. Restart Home Assistant

If download fails, remove the repo from HACS, clear cache (**HACS → Settings → Advanced → Clear data**), re-add the custom repository, and download again.
4. **Settings → Devices & services → Add integration → Battery Forecast**

### Manual

Copy `custom_components/battery_forecast` into your `config/custom_components/` folder and restart HA.

### scikit-learn (optional, recommended)

Training works **without** scikit-learn using a built-in numpy linear model (attribute `model_type: numpy`).

For **better accuracy** (gradient boosting, attribute `model_type: sklearn`), install scikit-learn once:

**Home Assistant OS** (SSH & Terminal add-on):

```bash
pip install scikit-learn
```

Then restart Home Assistant and run `battery_forecast.train` again.

**Docker:**

```bash
docker exec -it homeassistant pip install scikit-learn
```

## Configuration

| Step | Description |
|------|-------------|
| Battery | SOC (%), capacity (kWh), power (W); optional charge/discharge energy |
| Solar | PV power; optional forecast today/tomorrow (e.g. Forecast.Solar, Solcast) |
| Loads | House/grid power (required); optional heat pump |
| Environment | Outdoor temperature; optional weather entity |
| Features | Optional list of power/energy sensors (EV, boiler, …) |
| ML | Empty SOC %, training days (365), half-life, horizon, intervals |

### Example entities

| Role | Example |
|------|---------|
| SOC | `sensor.battery_soc` |
| Power | `sensor.battery_power` (W, + = charging) |
| House | `sensor.grid_import_power` or `sensor.home_consumption` |
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

Automation example (weekly retrain):

```yaml
automation:
  - alias: Retrain battery forecast
    trigger:
      - platform: time
        at: "03:00:00"
    action:
      - service: battery_forecast.train
```

## Sensors

| Sensor | Description |
|--------|-------------|
| Battery empty at | First timestamp when simulated SOC ≤ threshold |
| Battery hours remaining | Hours until empty (within horizon) |
| Predicted SOC (1h) | SOC after one simulated hour |
| Predicted net load next hour | ML estimate (kWh) |

Attributes include `confidence`, `mae_kwh`, `model_trained_at`, `feature_importances`, and `simulation_steps` (first 24h).

## Data tips

- Ensure **statistics** are enabled for power sensors (default for `state_class: measurement`).
- **1 year** of long-term data is enough for seasonal patterns; older data is not required.
- Add **feature sensors** only for large, stable loads (heat pump, EV, pool) — not dozens of switches.
- If training fails with “not enough samples”, reduce `min_training_samples` or check entity history in **History** graph.

## Manual test checklist

- [ ] Config flow completes with your entities
- [ ] `battery_forecast.train` finishes without error in logs
- [ ] All four sensors become available
- [ ] `hours_remaining` plausible during evening discharge
- [ ] Changing **empty SOC %** in options shifts `empty_at`
- [ ] `feature_importances` attribute lists sensible sensors after train

## License

MIT
