# Battery Forecast (Home Assistant)

HACS-ready custom integration that predicts when your home battery will reach a low state of charge, using **machine learning** on Home Assistant statistics (short-term + long-term) and optional recorder data.

## Features

- **Config flow** with entity selectors for battery, PV, house load, heat pump, weather, and up to 30 optional feature sensors
- **ML model** trained on up to **365 days** of hourly data (numpy by default, optional sklearn)
- **Weighted training**: recent weeks count more (configurable half-life, default 90 days)
- **Hybrid forecast**: ML predicts hourly net load → SOC simulation with solar forecast entities
- **Sensors**: empty-at timestamp, hours remaining, min SOC in 12h, predicted SOC in 1h, net load next hour
- **Services**: `battery_forecast.train`, `battery_forecast.reload_model`

## Requirements

- Home Assistant **2025.5+**
- Working **Recorder** and **Statistics** for your entities
- Long-term statistics retention ≥ your training window (default **365 days**)
- Python package (installed automatically by HA): `numpy`
- **`scikit-learn`** optional — HA often **cannot** auto-install it (Python 3.14 / HA OS missing wheels). Manual install via SSH if you want `model_type: sklearn`

## Installation

### HACS

1. **HACS → Settings → Custom repositories** → URL: `https://github.com/Volp02/battery-forecast-ha`, Category: **Integration**
2. **HACS → Settings** → enable **“Show beta versions”** (required for pre-releases)
3. **HACS → Integrations** → Battery Forecast → **Download** → choose latest beta (e.g. **`v0.2.6b`**)
4. Restart Home Assistant
5. **Settings → Devices & services → Add integration → Battery Forecast**

**“No releases found”** / only commit hashes (e.g. `565e661`) in HACS:

HACS needs **at least one normal GitHub release** (not pre-release) before it shows a beta/pre-release picker. This repo therefore has:

| Tag | Type | Use |
|-----|------|-----|
| **`v0.1.0`** | stable release | HACS anchor / default picker |
| **`v0.2.6b`** | pre-release | Current beta — enable **Show beta versions** |

If the version list is empty:

1. **HACS → Settings** → enable **Show beta versions**
2. Open **Battery Forecast** → three dots → **Redownload** / **Update repository information**
3. The GitHub release must **not** be a **Draft** (only published releases count)
4. Pick **`v0.2.6b`** (beta) or **`v0.1.0`** (stable) — not a bare commit hash

Fallback: download branch **`master`**.

### Manual

Copy `custom_components/battery_forecast` into your `config/custom_components/` folder and restart HA.

### ML backends: numpy (default) vs scikit-learn (optional)

| Backend | When | Sensor attribute | Accuracy |
|---------|------|------------------|----------|
| **numpy** | Always (installed by HA via `manifest.json`) | `model_type: numpy` | Weighted linear regression — works out of the box |
| **sklearn** | Only after **manual** install (see below) | `model_type: sklearn` | Gradient boosting — often better for variable loads |

**Why is scikit-learn not in `manifest.json`?**  
On many Home Assistant installs (especially **Python 3.14**), HA reports:

`Setup failed … Requirements not found: ['scikit-learn']`

So the integration ships **numpy only**. sklearn is an optional upgrade you install yourself **into Home Assistant’s Python environment**.

**Do you need sklearn?** Often **no** — if `model_type: numpy` already gives plausible `empty_at` / `hours_remaining` and `mae_kwh` is low (e.g. &lt; 0.35), stay on numpy. Try sklearn if you want to squeeze more accuracy out of complex patterns (PC at night, heat pump, weekends).

---

## Installing scikit-learn (optional)

### Are these two commands enough?

```bash
pip install scikit-learn
python -c "import sklearn; print(sklearn.__version__)"
```

**Partly.** They are the core install + check, but you **must also**:

1. Run them in **Home Assistant’s Python** (not your laptop, not a random venv).
2. **Restart Home Assistant** (full restart, not only reload integration).
3. Run **`battery_forecast.train`** again.
4. Confirm **`model_type: sklearn`** on a Battery Forecast sensor (not `numpy`).

If you skip the restart or train step, the integration will keep using the old **numpy** model.

---

### Step-by-step (recommended order)

1. Install and configure **Battery Forecast** (HACS), train once with numpy (works without sklearn).
2. Install scikit-learn using **one** of the methods below.
3. Verify import prints a version (e.g. `1.5.2`) **without error**.
4. **Settings → System → Restart Home Assistant**.
5. **Developer tools → Services** → `battery_forecast.train` (can take several minutes with sklearn).
6. Open any Battery Forecast sensor → attributes:
   - `model_type` → must be **`sklearn`**
   - `mae_kwh` / `r2` → compare to your previous numpy run

---

### Home Assistant OS (important: two different Pythons)

On **HA OS**, the **Terminal add-on** often uses a **different Python** than **Home Assistant Core**.

| Where you run `pip` | Used for training? |
|---------------------|-------------------|
| Terminal add-on (`python3` in `/config`) | **No** — only tests the add-on |
| **Home Assistant Core** container | **Yes** — this must have sklearn |

**Symptom:** `python3 -c "import sklearn"` works in the terminal, but sensor attribute stays `model_type: numpy` after retrain.

**Fix — install into the Core container (HA OS host shell):**

1. **Terminal & SSH** add-on: enable **SSH on port 22**, set a password.
2. From your PC:

   ```bash
   ssh -p 22 root@YOUR_HA_IP
   ```

3. If you see the **Home Assistant CLI** banner (`[core-ssh ~]$`, `docker: command not found`):

   **Do not use `login`** — on current HA OS that only asks for the add-on password (unrelated).

   Use one of these instead:

   **A) Advanced SSH & Web Terminal** (recommended, community add-on):

   - Install add-on **Advanced SSH & Web Terminal** (search in add-on store).
   - Configuration: disable **protection mode**, enable **Docker access** / host access (wording varies by version).
   - Open the add-on **web terminal** or SSH into it, then:

     ```bash
     docker ps
     docker exec -it homeassistant python3 -m pip install scikit-learn
     docker exec -it homeassistant python3 -c "import sklearn; print(sklearn.__version__)"
     ```

   **B) HA OS debug SSH on port 22222** (expert, requires SSH key on boot partition):

   - See [Home Assistant debugging](https://www.home-assistant.io/docs/locked/lost_password/) / community guides for `authorized_keys` on the boot partition and `ssh -p 22222 root@YOUR_HA_IP`, then `login` at the `ha >` prompt and `docker exec …`.

   The official **Terminal & SSH** add-on (port 22, HA CLI) **cannot** run `docker` — installing `pip` there does **not** affect Core training.

4. Restart Home Assistant from the UI (or `ha core restart` from CLI), then `battery_forecast.train`.

**Python 3.14 (HA Core 2026.5+):** Even with `docker exec` into the `homeassistant` container, `pip install scikit-learn` often **fails** (no wheel, builds scipy via meson → `Permission denied: meson`). Your Terminal add-on may use Python 3.13 where sklearn installs fine — that still does **not** help Core training. **Stay on `model_type: numpy`** until scikit-learn publishes cp314 wheels or HA ships an install path.

Optional check in the Core container:

```bash
docker exec -it homeassistant python3 --version
docker exec -it homeassistant python3 -m pip install scikit-learn --only-binary=:all:
```

If that fails, numpy is the correct backend for your system.

4. **Restart Home Assistant** (UI) → `battery_forecast.train` → `model_type: sklearn`.

**Check in HA:** Battery Forecast sensor attributes now include `core_python`, `sklearn_importable`, `sklearn_version` (or `sklearn_import_error`).

Installing only in the add-on shell:

```bash
python3 -m pip install scikit-learn
```

is **not enough** on HA OS unless `sklearn_importable: true` appears on the sensor.

---

### Home Assistant Container / Docker

```bash
docker exec -it homeassistant python3 -m pip install scikit-learn
docker exec -it homeassistant python3 -c "import sklearn; print(sklearn.__version__)"
```

Then restart the container / HA as you usually do.

---

### Home Assistant Supervised / generic Linux venv

Activate the **same** virtualenv Home Assistant uses, then:

```bash
pip install scikit-learn
python -c "import sklearn; print(sklearn.__version__)"
```

---

### After install: retrain (required)

```yaml
service: battery_forecast.train
```

Or **Developer tools → Services** → `battery_forecast.train`.

Watch **Settings → System → Logs**, filter `battery_forecast`. Success looks like:

```text
Battery Forecast: fitting sklearn HistGradientBoostingRegressor
Battery Forecast: train complete — model=sklearn mae=… kWh
```

---

### Troubleshooting

| Problem | What to do |
|---------|------------|
| `pip install` succeeds but `model_type` stays **numpy** | Wrong Python environment and/or **HA not restarted** → use `docker exec … homeassistant python3` and full restart |
| `import sklearn` fails | No wheel for your Python version (common on **3.14**) → stay on **numpy**; retry after HA/sklearn updates |
| `Setup failed … Requirements not found: scikit-learn` | You are on an old build with sklearn in `manifest.json` → update to **v0.2.6b+** (numpy only in manifest) |
| Training very slow / high CPU | Normal for sklearn; run at night; reduce `training_days` in options |
| Integration worked before, broke after experiment | Install **v0.2.6b**, restart HA — do **not** add sklearn to `manifest.json` yourself |

---

### Uninstall / back to numpy only

```bash
docker exec -it homeassistant python3 -m pip uninstall scikit-learn -y
```

Restart HA → `battery_forecast.train` → `model_type: numpy`.

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

Attributes include `model_type` (`numpy` / `sklearn`), `confidence`, `mae_kwh`, `model_trained_at`, `feature_importances`, and `simulation_steps` (first 24h).

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
