"""
================================================================================
  CB4 Dataset Builder: Telemetry + Historical Weather Fusion
================================================================================
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "mydashbord", "solar_telemetry.db")
WEATHER_CSV = os.path.join(DATA_DIR, "weather_history_cb4.csv")
MERGED_CSV = os.path.join(DATA_DIR, "cb4_weather_telemetry_merged.csv")

INSTALLED_KWP = 18.720

def build_merged_dataset(save_csv=True):
    print("=" * 60)
    print("  Building Merged Dataset: Telemetry + Historical Weather")
    print("=" * 60)

    # 1. Load Weather CSV
    if not os.path.exists(WEATHER_CSV):
        from weather_archive_fetcher import fetch_historical_weather
        df_weather = fetch_historical_weather()
    else:
        df_weather = pd.read_csv(WEATHER_CSV)
    
    df_weather['dt'] = pd.to_datetime(df_weather['time'])
    df_weather = df_weather.sort_values('dt').reset_index(drop=True)
    print(f"[1/4] Weather rows loaded: {len(df_weather):,}")

    # 2. Load Telemetry from SQLite
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    df_telem = pd.read_sql("SELECT * FROM telemetry ORDER BY timestamp ASC", conn)
    conn.close()
    print(f"[2/4] Telemetry rows loaded: {len(df_telem):,}")

    df_telem['dt'] = pd.to_datetime(df_telem['timestamp'])
    df_telem = df_telem.sort_values('dt').reset_index(drop=True)

    # 3. Merge asof (nearest time within 45 mins)
    print("[3/4] Performing time-series merge (merge_asof)...")
    merged = pd.merge_asof(
        df_telem,
        df_weather[['dt', 'weather_ghi', 'weather_dni', 'weather_dhi', 'weather_temp', 'weather_cloud']],
        on='dt',
        direction='nearest',
        tolerance=pd.Timedelta('45m')
    )

    # Fill any missing weather with forward/back fill
    merged[['weather_ghi', 'weather_dni', 'weather_dhi', 'weather_temp', 'weather_cloud']] = (
        merged[['weather_ghi', 'weather_dni', 'weather_dhi', 'weather_temp', 'weather_cloud']]
        .ffill().bfill()
    )

    # 4. Feature Engineering
    print("[4/4] Engineering solar & environmental features...")
    irr = pd.to_numeric(merged['Irradiance (W/m2)'], errors='coerce').fillna(0)
    p_ac = pd.to_numeric(merged['Active power(kW)'], errors='coerce').fillna(0)
    cloud = merged['weather_cloud'].astype(float)
    temp = merged['weather_temp'].astype(float)
    ghi_w = merged['weather_ghi'].astype(float)
    dni_w = merged['weather_dni'].astype(float)
    dhi_w = merged['weather_dhi'].astype(float)

    # Temperature coefficient derating (-0.4% per °C above 25°C)
    temp_factor = 1.0 - 0.004 * (temp - 25.0)
    temp_factor = np.clip(temp_factor, 0.80, 1.10)

    # Theoretical maximum power (80% PR under STC)
    theo_power = (irr / 1000.0) * INSTALLED_KWP * 0.82 * temp_factor
    theo_power = np.clip(theo_power, 0.0, 20.0)

    # Diffuse ratio
    diffuse_ratio = dhi_w / (ghi_w + 1e-4)
    diffuse_ratio = np.clip(diffuse_ratio, 0.0, 1.0)

    # Cloud shading indicator: when cloud is high and diffuse ratio is high
    cloud_shading_prob = (cloud / 100.0) * (0.5 + 0.5 * diffuse_ratio)

    merged['irradiance'] = irr
    merged['actual_power_kw'] = p_ac
    merged['pv1_power'] = pd.to_numeric(merged['PV1 power(kW)'], errors='coerce').fillna(0)
    merged['pv2_power'] = pd.to_numeric(merged['PV2 power(kW)'], errors='coerce').fillna(0)
    merged['pv3_power'] = pd.to_numeric(merged['PV3 power(kW)'], errors='coerce').fillna(0)
    merged['pv4_power'] = pd.to_numeric(merged['PV4 power(kW)'], errors='coerce').fillna(0)
    merged['cloud_cover'] = cloud
    merged['ambient_temp'] = temp
    merged['temp_factor'] = temp_factor
    merged['diffuse_ratio'] = diffuse_ratio
    merged['cloud_shading_prob'] = cloud_shading_prob
    merged['theoretical_power_kw'] = theo_power

    # Pro_2 / mydashbord Tabular Features: Cyclical Time & Lags
    merged['hour'] = merged['dt'].dt.hour
    merged['minute'] = merged['dt'].dt.minute
    merged['dow'] = merged['dt'].dt.dayofweek
    mins = merged['hour'] * 60 + merged['minute']
    merged['tod_sin'] = np.sin(2 * np.pi * mins / (24 * 60))
    merged['tod_cos'] = np.cos(2 * np.pi * mins / (24 * 60))

    # Lags (5, 10, 15, 30, 60 mins) & Rolling 30-min window
    for k in [1, 2, 3, 6, 12]:
        merged[f'irr_lag{k}'] = merged['irradiance'].shift(k).fillna(0)
    merged['irr_rollmean_6'] = merged['irradiance'].rolling(6, min_periods=1).mean().fillna(0)
    merged['irr_rollstd_6'] = merged['irradiance'].rolling(6, min_periods=1).std().fillna(0)

    # Filter for daylight hours (irradiance >= 15 W/m² or power > 0.05 kW)
    daylight_mask = (merged['irradiance'] >= 15) | (merged['actual_power_kw'] > 0.05)
    df_clean = merged[daylight_mask].copy().reset_index(drop=True)

    print(f"[DONE] Filtered daylight records: {len(df_clean):,} rows (from {len(merged):,} total)")

    if save_csv:
        os.makedirs(DATA_DIR, exist_ok=True)
        df_clean.to_csv(MERGED_CSV, index=False)
        print(f"[SAVED] Merged dataset saved to: {MERGED_CSV}")

    return df_clean

if __name__ == "__main__":
    df = build_merged_dataset()
    print(df[['timestamp', 'irradiance', 'actual_power_kw', 'cloud_cover', 'ambient_temp', 'theoretical_power_kw']].head(5))
