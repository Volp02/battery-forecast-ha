# Battery Forecast (Home Assistant)

HACS-ready custom integration that predicts when your home battery will reach a low state of charge, using **machine learning** on Home Assistant statistics (short-term + long-term) and optional recorder data.

## Features

- **Config flow** with entity selectors for battery, PV, house load, heat pump, weather, and up to 30 optional feature sensors
- **ML model** trained on up to **365 days** of hourly data (numpy by default, optional sklearn)
- **Weighted training**: recent weeks count more (configurable half-life, default 90 days)
- **Hybrid forecast**: ML predicts hourly net load → SOC simulation with solar forecast entities
- **Sensors**: empty-at timestamp, hours remaining, predicted SOC in 1h, net load next hour
- **Services**: `battery_forecast.train`, `battery_forecast.reload_model`

## Requirements

- Home Assistant **2025.5+**
- Working **Recorder** and **Statistics** for your entities
- Long-term statistics retention ≥ your training window (default **365 days**)
- Python package (installed automatically): `numpy`
- **`scikit-learn`** optional but recommended (better accuracy; see below)

## Installation

### HACS

1. **HACS → Settings → Custom repositories** → URL: `https://github.com/Volp02/battery-forecast-ha`, Category: **Integration**
2. **HACS → Settings** → enable **“Show beta versions”** (required for pre-releases)
3. **HACS → Integrations** → Battery Forecast → **Download** → choose version **`v0.1b`**
4. Restart Home Assistant
5. **Settings → Devices & services → Add integration → Battery Forecast**

**“No releases found”** / only commit hashes (e.g. `565e661`) in HACS:

HACS needs **at least one normal GitHub release** (not pre-release) before it shows a beta/pre-release picker. This repo therefore has:

| Tag | Type | Use |
|-----|------|-----|
| **`v0.1.0`** | stable release | Default in HACS |
| **`v0.1b`** | pre-release | Beta — enable **Show beta versions** |

If the version list is empty:

1. **HACS → Settings** → enable **Show beta versions**
2. Open **Battery Forecast** → three dots → **Redownload** / **Update repository information**
3. The GitHub release must **not** be a **Draft** (only published releases count)
4. Pick **`v0.1b`** (beta) or **`v0.1.0`** (stable) — not a bare commit hash

Fallback: download branch **`master`**.

### Manual

Copy `custom_components/battery_forecast` into your `config/custom_components/` folder and restart HA.

### ML backends: numpy (default) vs scikit-learn (optional)

| Backend | When | Sensor attribute | Accuracy |
|---------|------|------------------|----------|
| **numpy** | Always available (no extra install) | `model_type: numpy` | Linear model, good for beta/testing |
| **sklearn** | After manual `pip install scikit-learn` | `model_type: sklearn` | Gradient boosting, better for complex loads |

Training works **out of the box** with numpy. Check logs and sensor attributes after `battery_forecast.train`.

### scikit-learn (optional, recommended)

**Home Assistant OS** (SSH & Terminal add-on):

```bash
pip install scikit-learn
```

Restart Home Assistant, then run `battery_forecast.train` again.

**Docker:**

```bash
docker exec -it homeassistant pip install scikit-learn
```

---

## TODO: scikit-learn documentation (planned)

> **Status:** Not finished yet — numpy is the default for v0.1b. Full sklearn guide to be added before a stable release.

Planned content for a future `docs/SKLEARN.md` + README section:

- [ ] **Why install sklearn?** — When numpy is enough vs when sklearn is worth it (heat pump, EV, many feature sensors)
- [ ] **Install step-by-step** — HA OS, Docker, supervised vs SSH add-on, how to verify: `python -c "import sklearn; print(sklearn.__version__)"`
- [ ] **Retrain workflow** — `battery_forecast.train` after install; confirm `model_type: sklearn` in sensor attributes
- [ ] **Troubleshooting** — `RequirementsNotFound`, wrong Python venv, container rebuild after HA OS update
- [ ] **Performance notes** — Training duration on Raspberry Pi / NUC, RAM use, recommended `training_days` with sklearn
- [ ] **Comparison** — Example MAE before/after sklearn on same entity setup (documented with real numbers once tested)

Until then: use numpy and watch **Settings → System → Logs** (filter `battery_forecast`) for training progress.

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

One device **Battery Forecast** with four entities (friendly names are translated):

| Entity ID | Friendly name (EN) | Description |
|-----------|-------------------|-------------|
| `sensor.battery_empty_at` | Empty at | First timestamp when simulated SOC ≤ threshold |
| `sensor.battery_hours_remaining` | Hours remaining | Hours until empty (within horizon) |
| `sensor.battery_predicted_soc_1h` | Predicted SOC in 1h | SOC after one simulated hour of simulation |
| `sensor.battery_net_load_next_hour` | Net load next hour | ML estimate (kWh) |

After upgrading from older builds, remove stale `sensor.battery_forecast` / `_2` … entities in **Settings → Devices & services** if they remain as orphaned entities.

Attributes include `model_type` (`numpy` / `sklearn`), `confidence`, `mae_kwh`, `model_trained_at`, `feature_importances`, and `simulation_steps` (first 24h).

**ML note:** Training uses the **house/grid meter** as total load. Optional feature sensors (washer, dryer, EV, …) are **supplementary** — you can use zero or a few; the model must not depend on them. They are **not** added on top of house power (no double counting). At **forecast** time those optional sensors are treated as **off (0 W)** for all future hours (we do not know if the washer will run); **house power**, time-of-day, heat pump, PV, and temperature drive the prediction.

### Outdoor temperature and weather

| Input | Used in ML? | How |
|-------|-------------|-----|
| **Outdoor temperature** (`outdoor_temp`) | **Yes** | Hourly values from **statistics** during training; at forecast time the **current** temperature is applied to all future hours (simple; no hourly weather forecast yet). |
| **Weather entity** (`weather_entity`) | **No (v0.1.1)** | Stored in config for a future release (e.g. forecast temperature). Pick a sensor with `device_class: temperature` or similar for `outdoor_temp`. |

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
- **Feature sensors are optional.** House/grid power alone is enough. Add at most a few large loads (heat pump, EV) if you want; omit washer/dryer unless you need them for training hints — they must not be required for a sensible forecast.
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
