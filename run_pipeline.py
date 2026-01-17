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

# --- 1. PROOF OF STAKE BLOCKCHAIN ---
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

# --- 2. MODELS & UTILITIES ---
def get_anomaly_scores(data):
    inputs = Input(shape=(data.shape[1],))
    enc = Dense(8, activation='relu')(inputs)
    btn = Dense(4, activation='relu')(enc)
    dec = Dense(8, activation='relu')(btn)
    out = Dense(data.shape[1], activation='sigmoid')(dec)
    ae = Model(inputs, out)
    ae.compile(optimizer='adam', loss='mse')
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

def visualize_results(y_true, y_pred, cols):
    fig, axs = plt.subplots(len(cols), 1, figsize=(10, 15), sharex=True)
    fig.suptitle('AI Model Verification: Actual vs Predicted (Last 30 Samples)', fontsize=14)
    for i, col in enumerate(cols):
        # Apply visual clipping for the graph
        display_pred = y_pred[-30:, i]
        if col in ["Rainfall_mm", "UV_Index"]:
            display_pred = np.maximum(0, display_pred)
            
        axs[i].plot(y_true[-30:, i], label='Actual', color='blue', marker='o', markersize=3)
        axs[i].plot(display_pred, label='AI Forecast', color='orange', linestyle='--', marker='x')
        axs[i].set_ylabel(col.replace('_',' '))
        axs[i].legend(loc='upper right')
    plt.xlabel('Timeline (Days)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('forecast_verification.png')
    print("📈 Verification plot saved as 'forecast_verification.png'")

# --- 3. MAIN PIPELINE ---
def run_pipeline():
    URL = "https://raw.githubusercontent.com/Ritwik240/Weather-Dataset/refs/heads/main/Unified_Weather_Dataset_Latest.json"
    df = pd.read_json(URL)
    data_content = requests.get(URL).content
    data_hash = hashlib.sha256(data_content).hexdigest()
    
    METEO = ["Temperature_C", "Humidity_%", "WindSpeed_m/s", "Rainfall_mm", "UV_Index"]
    df['Month_Sin'] = np.sin(2 * np.pi * pd.to_datetime(df['Date']).dt.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * pd.to_datetime(df['Date']).dt.month / 12)
    
    scaler_ae = MinMaxScaler()
    df['Anomaly_Score'] = get_anomaly_scores(scaler_ae.fit_transform(df[METEO]))
    
    FEATS = METEO + ["Month_Sin", "Month_Cos", "Anomaly_Score"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATS])
    
    # Save Individual Scalers (Photo 2 structure)
    for i, col in enumerate(METEO):
        col_scaler = MinMaxScaler()
        col_scaler.min_, col_scaler.scale_ = [scaler.min_[i]], [scaler.scale_[i]]
        clean_name = col.replace("/", "_").replace("%", "pct")
        joblib.dump(col_scaler, f'scaler_{clean_name}.joblib')
    
    WIN, FOR = 14, 7
    X, y = [], []
    for i in range(len(scaled) - WIN - FOR):
        X.append(scaled[i:i+WIN])
        y.append(scaled[i+WIN:i+WIN+FOR, :len(METEO)])
    X, y = np.array(X), np.array(y)
    
    model = build_cnn_lstm(WIN, len(FEATS), FOR, len(METEO))
    model.fit(X, y, epochs=20, verbose=0, callbacks=[EarlyStopping(patience=3)])
    model.save('weather_model_cnn_lstm.h5')
    
    # E. Evaluation & Accuracy Metrics
    test_preds = model.predict(X[-100:])
    y_true_real, y_pred_real = [], []
    
    for i in range(len(test_preds)):
        row_true, row_pred = np.zeros((FOR, len(FEATS))), np.zeros((FOR, len(FEATS)))
        row_true[:, :len(METEO)] = y[-100+i]
        row_pred[:, :len(METEO)] = test_preds[i]
        y_true_real.append(scaler.inverse_transform(row_true)[0, :len(METEO)])
        y_pred_real.append(scaler.inverse_transform(row_pred)[0, :len(METEO)])
        
    y_true_real, y_pred_real = np.array(y_true_real), np.array(y_pred_real)
    
    metrics = {}
    for i, col in enumerate(METEO):
        metrics[col] = {
            "MAE": round(float(mean_absolute_error(y_true_real[:, i], y_pred_real[:, i])), 4),
            "R2": round(float(r2_score(y_true_real[:, i], y_pred_real[:, i])), 4)
        }

    visualize_results(y_true_real, y_pred_real, METEO)

    # G. 7-Day Forecast with Clipping & Confidence Intervals
    last_window = scaled[-WIN:].reshape(1, WIN, len(FEATS))
    future_preds = model.predict(last_window).reshape(FOR, len(METEO))
    
    forecast_list = []
    base_date = pd.to_datetime(df['Date'].iloc[-1])
    
    for i in range(FOR):
        row = np.zeros(len(FEATS))
        row[:len(METEO)] = future_preds[i]
        real_vals = scaler.inverse_transform([row])[0]
        
        # Applying np.maximum(0, ...) to ensure physical realism
        temp = float(real_vals[0])
        temp_mae = metrics["Temperature_C"]["MAE"]
        rain_mm = max(0.0, float(real_vals[3]))
        uv_val = max(0.0, float(real_vals[4]))
        
        prob = min(100, max(0, rain_mm * 15)) 
        
        forecast_list.append({
            "Date": (base_date + timedelta(days=i+1)).strftime('%Y-%m-%d'),
            "Temperature_C": round(temp, 2),
            "Confidence_Range": f"{round(temp - temp_mae, 1)}°C to {round(temp + temp_mae, 1)}°C",
            "Humidity_%": round(float(real_vals[1]), 2),
            "UV_Index": round(uv_val, 2),
            "WindSpeed_m/s": round(float(real_vals[2]), 2),
            "Rainfall_mm": round(rain_mm, 2),
            "Rain_Probability": round(prob, 1),
            "Rain_Alert": "Rain Likely 🌧️" if prob > 40 else "No Rain ☀️"
        })
    
    with open("forecast_7days_with_alerts.json", "w") as f:
        json.dump(forecast_list, f, indent=4)
        
    WeatherBlockchainPoS().add_block({
        "data_hash": data_hash,
        "model_accuracy": metrics,
        "forecast_snapshot_hash": hashlib.sha256(str(forecast_list).encode()).hexdigest()
    })
    print("✅ Pipeline updated with Physical Constraints and Confidence Intervals.")

if __name__ == "__main__":
    run_pipeline()
