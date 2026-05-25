"""Constants for the Battery Forecast integration."""

DOMAIN = "battery_forecast"
STORAGE_KEY = "battery_forecast_model"
STORAGE_VERSION = 1

CONF_BATTERY_SOC = "battery_soc"
CONF_BATTERY_CAPACITY_KWH = "battery_capacity_kwh"
CONF_BATTERY_POWER = "battery_power"
CONF_BATTERY_CHARGE_ENERGY = "battery_charge_energy"
CONF_BATTERY_DISCHARGE_ENERGY = "battery_discharge_energy"

CONF_PV_POWER = "pv_power"
CONF_PV_FORECAST = "pv_forecast"
CONF_PV_FORECAST_TOMORROW = "pv_forecast_tomorrow"

CONF_HOUSE_POWER = "house_power"
CONF_HEAT_PUMP_POWER = "heat_pump_power"

CONF_OUTDOOR_TEMP = "outdoor_temp"
CONF_WEATHER_ENTITY = "weather_entity"

CONF_FEATURE_ENTITIES = "feature_entities"

CONF_EMPTY_SOC_PERCENT = "empty_soc_percent"
CONF_TRAINING_DAYS = "training_days"
CONF_SAMPLE_HALF_LIFE_DAYS = "sample_half_life_days"
CONF_USE_RECORDER_FALLBACK = "use_recorder_fallback"
CONF_FORECAST_HORIZON_HOURS = "forecast_horizon_hours"
CONF_UPDATE_INTERVAL_MINUTES = "update_interval_minutes"
CONF_MIN_TRAINING_SAMPLES = "min_training_samples"
CONF_MAX_FEATURE_ENTITIES = "max_feature_entities"
CONF_IMPORTANCE_THRESHOLD = "importance_threshold"

DEFAULT_EMPTY_SOC_PERCENT = 10.0
DEFAULT_TRAINING_DAYS = 365
DEFAULT_SAMPLE_HALF_LIFE_DAYS = 90
DEFAULT_USE_RECORDER_FALLBACK = True
DEFAULT_FORECAST_HORIZON_HOURS = 48
DEFAULT_UPDATE_INTERVAL_MINUTES = 5
DEFAULT_MIN_TRAINING_SAMPLES = 168
DEFAULT_MAX_FEATURE_ENTITIES = 30
DEFAULT_IMPORTANCE_THRESHOLD = 0.005

SHORT_TERM_STATISTICS_DAYS = 10

ATTR_CONFIDENCE = "confidence"
ATTR_MODEL_TRAINED_AT = "model_trained_at"
ATTR_MODEL_SAMPLES = "model_samples"
ATTR_MAE_KWH = "mae_kwh"
ATTR_RMSE_KWH = "rmse_kwh"
ATTR_R2 = "r2"
ATTR_FEATURE_IMPORTANCES = "feature_importances"
ATTR_SIMULATION_STEPS = "simulation_steps"
ATTR_NET_ENERGY_NEXT_HOUR_KWH = "net_energy_next_hour_kwh"

SENSOR_TYPE_EMPTY_AT = "empty_at"
SENSOR_TYPE_HOURS_REMAINING = "hours_remaining"
SENSOR_TYPE_PREDICTED_SOC = "predicted_soc"
SENSOR_TYPE_NET_LOAD = "net_load_next_hour"

SERVICE_TRAIN = "train"
SERVICE_RELOAD_MODEL = "reload_model"
