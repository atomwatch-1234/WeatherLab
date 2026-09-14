"""
================================================================================
  Open-Meteo Historical Weather Archive Fetcher for KMUTT CB4
  Fetches hourly GHI, DNI, DHI, Temperature, and Cloud Cover (2026)
================================================================================
"""

import os
import sys
import requests
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CB4_LAT = 13.6515
CB4_LON = 100.4952
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_FILE = os.path.join(DATA_DIR, "weather_history_cb4.csv")

def fetch_historical_weather(start_date="2026-01-22", end_date="2026-09-10", force_reload=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CACHE_FILE) and not force_reload:
        print(f"[CACHE] Loading cached weather from: {CACHE_FILE}")
        return pd.read_csv(CACHE_FILE)

    print(f"[API] Fetching Open-Meteo Archive for CB4 ({start_date} -> {end_date})...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": CB4_LAT,
        "longitude": CB4_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "shortwave_radiation",         # GHI in W/m²
            "direct_normal_irradiance",    # DNI in W/m²
            "diffuse_radiation",           # DHI in W/m²
            "temperature_2m",              # Ambient Temp in °C
            "cloud_cover"                  # Total Cloud Cover 0-100%
        ],
        "timezone": "Asia/Bangkok"
    }

    try:
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()
        hourly = data.get("hourly", {})
        
        df = pd.DataFrame({
            "time": hourly.get("time", []),
            "weather_ghi": hourly.get("shortwave_radiation", []),
            "weather_dni": hourly.get("direct_normal_irradiance", []),
            "weather_dhi": hourly.get("diffuse_radiation", []),
            "weather_temp": hourly.get("temperature_2m", []),
            "weather_cloud": hourly.get("cloud_cover", [])
        })
        
        df.to_csv(CACHE_FILE, index=False)
        print(f"[SAVED] Saved {len(df):,} hourly weather records to {CACHE_FILE}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch historical weather: {e}")
        return None

if __name__ == "__main__":
    df_w = fetch_historical_weather()
    if df_w is not None:
        print(df_w.head())
