# -*- coding: utf-8 -*-
"""
================================================================================
  Open-Meteo Live Solar and Weather Forecast Fetcher for KMUTT CB4
  Coordinates: 13.6515 N, 100.4952 E
  Provides 7-Day (168h) hourly solar radiation and atmospheric data
================================================================================
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CB4_LAT = 13.6515
CB4_LON = 100.4952
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FORECAST_CACHE_FILE = os.path.join(DATA_DIR, "openmeteo_forecast_cb4.json")
CACHE_EXPIRY_SECONDS = 3600  # 1 hour cache

def fetch_openmeteo_forecast(days=7, force_reload=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not force_reload and os.path.exists(FORECAST_CACHE_FILE):
        file_age = time.time() - os.path.getmtime(FORECAST_CACHE_FILE)
        if file_age < CACHE_EXPIRY_SECONDS:
            try:
                with open(FORECAST_CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return parse_forecast_json(data)
            except Exception as e:
                print(f"[CACHE] Error reading cache: {e}, fetching fresh data...")

    print(f"[API] Fetching Open-Meteo Live Forecast for CB4 ({days} days)...")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": CB4_LAT,
        "longitude": CB4_LON,
        "hourly": [
            "shortwave_radiation",
            "direct_normal_irradiance",
            "diffuse_radiation",
            "temperature_2m",
            "relative_humidity_2m",
            "cloud_cover",
            "precipitation_probability",
            "weather_code"
        ],
        "daily": [
            "shortwave_radiation_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "sunrise",
            "sunset",
            "precipitation_probability_max",
            "weather_code"
        ],
        "forecast_days": days,
        "timezone": "Asia/Bangkok"
    }

    try:
        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()
        data = res.json()
        
        with open(FORECAST_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[API] Successfully fetched and cached forecast to {FORECAST_CACHE_FILE}")
        return parse_forecast_json(data)
    except Exception as e:
        print(f"[ERROR] Failed to fetch Open-Meteo forecast: {e}")
        if os.path.exists(FORECAST_CACHE_FILE):
            with open(FORECAST_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return parse_forecast_json(data)
        return None

def parse_forecast_json(data):
    hourly = data.get("hourly", {})
    daily = data.get("daily", {})
    times = hourly.get("time", [])
    if not times:
        return None
        
    df_hourly = pd.DataFrame({
        "timestamp": pd.to_datetime(times),
        "time_str": times,
        "ghi": hourly.get("shortwave_radiation", []),
        "dni": hourly.get("direct_normal_irradiance", []),
        "dhi": hourly.get("diffuse_radiation", []),
        "ambient_temp": hourly.get("temperature_2m", []),
        "humidity": hourly.get("relative_humidity_2m", []),
        "cloud_cover": hourly.get("cloud_cover", []),
        "rain_prob": hourly.get("precipitation_probability", []),
        "weather_code": hourly.get("weather_code", [])
    })
    
    df_hourly['date'] = df_hourly['timestamp'].dt.strftime('%Y-%m-%d')
    df_hourly['hour'] = df_hourly['timestamp'].dt.hour
    df_hourly['minute'] = df_hourly['timestamp'].dt.minute
    df_hourly['dow'] = df_hourly['timestamp'].dt.dayofweek
    
    time_frac = (df_hourly['hour'] * 60 + df_hourly['minute']) / 1440.0
    df_hourly['tod_sin'] = np.sin(2 * np.pi * time_frac)
    df_hourly['tod_cos'] = np.cos(2 * np.pi * time_frac)
    
    df_hourly['diffuse_ratio'] = np.where(
        df_hourly['ghi'] > 10.0,
        np.clip(df_hourly['dhi'] / df_hourly['ghi'], 0.0, 1.0),
        0.5
    )
    
    t_cell_est = df_hourly['ambient_temp'] + (45.0 - 20.0) / 800.0 * df_hourly['ghi']
    df_hourly['temp_factor'] = np.clip(1.0 - 0.0035 * (t_cell_est - 25.0), 0.70, 1.05)
    
    eta_system = 0.82
    df_hourly['theoretical_power_kw'] = (
        18.72 * (df_hourly['ghi'] / 1000.0) * df_hourly['temp_factor'] * eta_system
    ).clip(lower=0.0)
    
    df_hourly['cloud_shading_prob'] = np.where(
        df_hourly['cloud_cover'] > 40.0,
        (df_hourly['cloud_cover'] / 100.0) * (1.0 - df_hourly['ghi'] / 1000.0).clip(0, 1),
        0.0
    )
    
    df_hourly['irradiance'] = df_hourly['ghi']
    
    df_hourly['irr_lag1'] = df_hourly['irradiance'].shift(1).fillna(0)
    df_hourly['irr_lag2'] = df_hourly['irradiance'].shift(2).fillna(0)
    df_hourly['irr_lag3'] = df_hourly['irradiance'].shift(3).fillna(0)
    df_hourly['irr_lag6'] = df_hourly['irradiance'].shift(6).fillna(0)
    df_hourly['irr_lag12'] = df_hourly['irradiance'].shift(12).fillna(0)
    df_hourly['irr_rollmean_6'] = df_hourly['irradiance'].rolling(window=6, min_periods=1).mean()
    df_hourly['irr_rollstd_6'] = df_hourly['irradiance'].rolling(window=6, min_periods=1).std().fillna(0)
    
    daily_dates = daily.get("time", [])
    df_daily = pd.DataFrame({
        "date": daily_dates,
        "solar_radiation_mj": daily.get("shortwave_radiation_sum", []),
        "temp_max": daily.get("temperature_2m_max", []),
        "temp_min": daily.get("temperature_2m_min", []),
        "sunrise": [s[-5:] if s else "" for s in daily.get("sunrise", [])],
        "sunset": [s[-5:] if s else "" for s in daily.get("sunset", [])],
        "rain_prob_max": daily.get("precipitation_probability_max", []),
        "weather_code": daily.get("weather_code", [])
    })
    
    df_daily['peak_sun_hours'] = df_daily['solar_radiation_mj'] / 3.6
    
    return {
        "hourly": df_hourly,
        "daily": df_daily,
        "raw_json": data
    }

if __name__ == "__main__":
    res = fetch_openmeteo_forecast(days=7)
    if res:
        print(f"Hourly shape: {res['hourly'].shape}")
        print(res['hourly'][['time_str', 'ghi', 'cloud_cover', 'ambient_temp', 'theoretical_power_kw']].head(12))
        print("\nDaily forecast:")
        print(res['daily'])
