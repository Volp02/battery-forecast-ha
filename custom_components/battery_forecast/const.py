"""Constants for the Battery Forecast integration."""

DOMAIN = "battery_forecast"
STORAGE_KEY = "battery_forecast_model"
STORAGE_VERSION = 1

CONF_BATTERY_SOC = "battery_soc"
CONF_BATTERY_CAPACITY_KWH = "battery_capacity_kwh"
CONF_BATTERY_POWER = "battery_power"
CONF_BATTERY_CHARGE_ENERGY = "battery_charge_energy"
CONF_BATTERY_DISCHARGE_ENERGY = "battery_discharge_energy"
CONF_BATTERY_POWER_INVERT = "battery_power_invert"

DEFAULT_BATTERY_POWER_INVERT = False

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
CONF_AUTO_RETRAIN_ENABLED = "auto_retrain_enabled"
CONF_AUTO_RETRAIN_SOC_MAE = "auto_retrain_soc_mae_percent"
CONF_AUTO_RETRAIN_MIN_HOURS = "auto_retrain_min_hours"
CONF_AUTO_RETRAIN_EVAL_HOURS = "auto_retrain_eval_hours"

STORAGE_EVAL_KEY = "battery_forecast_eval"
STORAGE_EVAL_VERSION = 1

DEFAULT_EMPTY_SOC_PERCENT = 10.0
DEFAULT_TRAINING_DAYS = 365
DEFAULT_SAMPLE_HALF_LIFE_DAYS = 90
DEFAULT_USE_RECORDER_FALLBACK = True
DEFAULT_FORECAST_HORIZON_HOURS = 48
DEFAULT_UPDATE_INTERVAL_MINUTES = 5
DEFAULT_MIN_TRAINING_SAMPLES = 168
DEFAULT_MAX_FEATURE_ENTITIES = 30
DEFAULT_IMPORTANCE_THRESHOLD = 0.005
DEFAULT_AUTO_RETRAIN_ENABLED = True
DEFAULT_AUTO_RETRAIN_SOC_MAE = 12.0
DEFAULT_AUTO_RETRAIN_MIN_HOURS = 24
DEFAULT_AUTO_RETRAIN_EVAL_HOURS = 24

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
ATTR_EMPTY_WITHIN_HORIZON = "empty_within_horizon"
ATTR_SOC_AT_HORIZON = "soc_at_horizon"
ATTR_FORECAST_HORIZON_HOURS = "forecast_horizon_hours"
ATTR_HOUSE_POWER_ENTITY = "house_power_entity"
ATTR_EMPTY_AT_EXTRAPOLATED = "empty_at_extrapolated"
ATTR_BATTERY_POWER_KW = "battery_power_kw"
ATTR_FORECAST_SOC_MAE = "forecast_soc_mae_24h"
ATTR_AUTO_RETRAIN_LAST = "auto_retrain_last_at"

SENSOR_TYPE_EMPTY_AT = "empty_at"
SENSOR_TYPE_HOURS_REMAINING = "hours_remaining"
SENSOR_TYPE_PREDICTED_SOC = "predicted_soc"
SENSOR_TYPE_NET_LOAD = "net_load_next_hour"
SENSOR_TYPE_MIN_SOC_12H = "min_soc_12h"

ATTR_MIN_SOC_12H = "min_soc_next_12h"
ATTR_PV_FORECAST_TODAY_KWH = "pv_forecast_today_kwh"
ATTR_PV_FORECAST_TOMORROW_KWH = "pv_forecast_tomorrow_kwh"

SERVICE_TRAIN = "train"
SERVICE_RELOAD_MODEL = "reload_model"
