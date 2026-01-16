import requests
import pandas as pd
from io import StringIO
import numpy as np
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
LAT, LON = 28.6139, 77.2090  # Delhi
START = "2015-01-01"
yesterday = datetime.now() - timedelta(days=1)
END = yesterday.strftime('%Y-%m-%d')
OUTPUT_FILENAME = "Unified_Weather_Dataset_Latest.json"

def fetch_weather_data():
    print(f"--- Fetching Data: {START} to {END} ---")
    
    # NASA POWER API
    nasa_params = "ALLSKY_SFC_UV_INDEX,T2M,RH2M,WS10M,PRECTOTCORR"
    nasa_url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={nasa_params}&start={START.replace('-', '')}&end={END.replace('-', '')}&latitude={LAT}&longitude={LON}&community=RE&format=CSV"
    
    try:
        r_nasa = requests.get(nasa_url, timeout=120)
        lines = r_nasa.text.splitlines()
        header = next(i for i, line in enumerate(lines) if line.startswith("YEAR"))
        df_nasa = pd.read_csv(StringIO("\n".join(lines[header:]))).replace(-999, np.nan)
        df_nasa["Date"] = pd.to_datetime(df_nasa[["YEAR", "MO", "DY"]].rename(columns={"YEAR":"year","MO":"month","DY":"day"}))
        df_nasa = df_nasa.rename(columns={"ALLSKY_SFC_UV_INDEX":"UV_Index","T2M":"T_N","RH2M":"H_N","WS10M":"W_N","PRECTOTCORR":"R_N"})
    except: df_nasa = pd.DataFrame()

    # Open-Meteo API
    om_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={START}&end_date={END}&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,windspeed_10m_mean&timezone=auto"
    try:
        r_om = requests.get(om_url).json()
        df_om = pd.DataFrame(r_om["daily"])
        df_om["Date"] = pd.to_datetime(df_om["time"])
        df_om = df_om.rename(columns={"temperature_2m_mean":"T_O","relative_humidity_2m_mean":"H_O","precipitation_sum":"R_O","windspeed_10m_mean":"W_O"})
    except: df_om = pd.DataFrame()

    # Unify
    df = pd.merge(df_nasa, df_om, on="Date", how="outer").sort_values("Date")
    df["Temperature_C"] = df[["T_N", "T_O"]].mean(axis=1)
    df["Humidity_%"] = df[["H_N", "H_O"]].mean(axis=1)
    df["Rainfall_mm"] = df[["R_N", "R_O"]].mean(axis=1)
    df["WindSpeed_m/s"] = df[["W_N", "W_O"]].mean(axis=1)
    
    final_cols = ["Date", "UV_Index", "Temperature_C", "Humidity_%", "Rainfall_mm", "WindSpeed_m/s"]
    df = df[final_cols].interpolate().bfill().ffill()
    
    df.to_json(OUTPUT_FILENAME, orient="records", indent=4, date_format="iso")
    print(f"✅ Data saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    fetch_weather_data()