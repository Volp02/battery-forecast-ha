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

**“No releases found”** in HACS usually means:

- Pre-releases are hidden → turn on **Show beta versions** in HACS settings, or
- HACS fell back to a commit hash (e.g. `bd4f536`) → pick **`v0.1b`** explicitly in the version list, or
- Use **Redownload** / clear HACS cache (**Settings → Advanced → Clear data**) after the GitHub release is published

If no version list appears, download from branch **`master`** (same as latest dev).

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

| Sensor | Description |
|--------|-------------|
| Battery empty at | First timestamp when simulated SOC ≤ threshold |
| Battery hours remaining | Hours until empty (within horizon) |
| Predicted SOC (1h) | SOC after one simulated hour |
| Predicted net load next hour | ML estimate (kWh) |

Attributes include `model_type` (`numpy` / `sklearn`), `confidence`, `mae_kwh`, `model_trained_at`, `feature_importances`, and `simulation_steps` (first 24h).

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
