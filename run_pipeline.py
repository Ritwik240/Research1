import hashlib, json, os, requests, random, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping

# 1. PROOF OF STAKE BLOCKCHAIN
class Block:
    def __init__(self, index, data, previous_hash, validator, timestamp=None):
        self.index, self.timestamp = index, timestamp or str(datetime.now())
        self.data, self.previous_hash, self.validator = data, previous_hash, validator
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class WeatherBlockchainPoS:
    def __init__(self):
        self.file_path = "weather_ledger_pos.json"
        self.validators = {"Node_Alpha": 60, "Node_Beta": 40}
        self.load_chain()

    def load_chain(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                self.chain = [Block(**b) for b in json.load(f)]
        else: self.chain = [Block(0, {"msg": "Genesis"}, "0", "System")]

    def add_block(self, data):
        val = random.choices(list(self.validators.keys()), weights=list(self.validators.values()))[0]
        new_block = Block(len(self.chain), data, self.chain[-1].hash, val)
        self.chain.append(new_block)
        with open(self.file_path, "w") as f:
            json.dump([b.__dict__ for b in self.chain], f, indent=4)
        print(f"⛓️ Block {new_block.index} forged by {val}")

# 2. UTILITIES
def get_anomaly_scores(data):
    inputs = Input(shape=(data.shape[1],))
    enc = Dense(8, activation='relu')(inputs); btn = Dense(4, activation='relu')(enc)
    dec = Dense(8, activation='relu')(btn); out = Dense(data.shape[1], activation='sigmoid')(dec)
    ae = Model(inputs, out); ae.compile(optimizer='adam', loss='mse')
    ae.fit(data, data, epochs=20, verbose=0)
    return np.mean(np.power(data - ae.predict(data), 2), axis=1)

def build_cnn_lstm(win, feat, forecast, targets):
    model = Sequential([
        Input(shape=(win, feat)),
        Conv1D(64, kernel_size=3, activation='relu'),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(forecast * targets),
        Reshape((forecast, targets))
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

# 3. MAIN PIPELINE
def run_pipeline():
    URL = "https://raw.githubusercontent.com/Ritwik240/Weather-Dataset/refs/heads/main/Unified_Weather_Dataset_Latest.json"
    df = pd.read_json(URL)
    data_hash = hashlib.sha256(requests.get(URL).content).hexdigest()
    
    METEO = ["Temperature_C", "Humidity_%", "WindSpeed_m/s", "Rainfall_mm", "UV_Index"]
    df['Month_Sin'] = np.sin(2 * np.pi * pd.to_datetime(df['Date']).dt.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * pd.to_datetime(df['Date']).dt.month / 12)
    
    scaler_ae = MinMaxScaler()
    df['Anomaly_Score'] = get_anomaly_scores(scaler_ae.fit_transform(df[METEO]))
    
    FEATS = METEO + ["Month_Sin", "Month_Cos", "Anomaly_Score"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATS])
    
    # Save Individual Scalers (Photo 2 Format)
    for i, col in enumerate(METEO):
        col_scaler = MinMaxScaler()
        col_scaler.min_, col_scaler.scale_ = [scaler.min_[i]], [scaler.scale_[i]]
        joblib.dump(col_scaler, f'scaler_{col.replace("/", "_").replace("%", "pct")}.joblib')
    
    WIN, FOR = 14, 7
    X, y = [], []
    for i in range(len(scaled) - WIN - FOR):
        X.append(scaled[i:i+WIN])
        y.append(scaled[i+WIN:i+WIN+FOR, :len(METEO)])
    X, y = np.array(X), np.array(y)
    
    model = build_cnn_lstm(WIN, len(FEATS), FOR, len(METEO))
    model.fit(X, y, epochs=15, verbose=0)
    
    # Generate 7-Day Forecast (Photo 3 Format)
    last_window = scaled[-WIN:].reshape(1, WIN, len(FEATS))
    preds = model.predict(last_window).reshape(FOR, len(METEO))
    
    # Manual Inverse Scaling for Forecast
    unscaled_preds = []
    for i in range(FOR):
        row = np.zeros(len(FEATS))
        row[:len(METEO)] = preds[i]
        unscaled_preds.append(scaler.inverse_transform([row])[0][:len(METEO)])
    
    forecast_out = []
    base_date = pd.to_datetime(df['Date'].iloc[-1])
    for i, p in enumerate(unscaled_preds):
        prob = min(100, max(0, p[3] * 15)) # Rainfall logic
        forecast_out.append({
            "Date": (base_date + timedelta(days=i+1)).strftime('%Y-%m-%d'),
            "Temperature_C": float(p[0]), "Humidity_%": float(p[1]),
            "UV_Index": float(p[4]), "WindSpeed_m/s": float(p[2]),
            "Rainfall_mm": float(p[3]), "Rain_Probability": round(prob, 1),
            "Rain_Alert": "Rain Likely 🌧️" if prob > 50 else "No Rain ☀️"
        })
    
    with open("forecast_7days_with_alerts.json", "w") as f:
        json.dump(forecast_out, f, indent=4)
    
    # Update Blockchain
    WeatherBlockchainPoS().add_block({
        "data_hash": data_hash,
        "forecast_hash": hashlib.sha256(str(forecast_out).encode()).hexdigest()
    })
    print("📈 Forecast and Scalers updated successfully.")

if __name__ == "__main__":
    run_pipeline()