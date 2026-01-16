import hashlib
import json
import os
import requests
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------
# 1. PROOF OF STAKE BLOCKCHAIN
# ---------------------------------
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

# ---------------------------------
# 2. MODELS & UTILITIES
# ---------------------------------
def get_anomaly_scores(data):
    # Autoencoder for Extreme Weather
    inputs = Input(shape=(data.shape[1],))
    enc = Dense(8, activation='relu')(inputs)
    btn = Dense(4, activation='relu')(enc)
    dec = Dense(8, activation='relu')(btn)
    out = Dense(data.shape[1], activation='sigmoid')(dec)
    ae = Model(inputs, out)
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(data, data, epochs=30, batch_size=16, verbose=0)
    return np.mean(np.power(data - ae.predict(data), 2), axis=1)

def build_cnn_lstm(win, feat, forecast, targets):
    model = Sequential([
        Input(shape=(win, feat)),
        Conv1D(64, kernel_size=3, activation='relu'),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(forecast * targets),
        Reshape((forecast, targets))
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

# ---------------------------------
# 3. VISUALIZATION WORKFLOW
# ---------------------------------
def visualize_results(y_true, y_pred, cols):
    fig, axs = plt.subplots(len(cols), 1, figsize=(10, 15), sharex=True)
    fig.suptitle('Actual vs. Predicted: 7-Day Forecast (Last 30 Samples)', fontsize=14)
    for i, col in enumerate(cols):
        axs[i].plot(y_true[-30:, i], label='Actual', color='blue', marker='o', markersize=4)
        axs[i].plot(y_pred[-30:, i], label='AI Forecast', color='orange', linestyle='--', marker='x')
        axs[i].set_ylabel(col.replace('_',' '))
        axs[i].legend(loc='upper right')
    plt.xlabel('Timeline (Days)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('forecast_verification.png')
    print("📈 Visualization saved as 'forecast_verification.png'")

# ---------------------------------
# 4. MAIN PIPELINE
# ---------------------------------
def run_pipeline():
    # A. Data Loading
    URL = "https://raw.githubusercontent.com/Ritwik240/Weather-Dataset/refs/heads/main/Unified_Weather_Dataset_Latest.json"
    df = pd.read_json(URL)
    data_hash = hashlib.sha256(requests.get(URL).content).hexdigest()
    
    # B. Feature Engineering (Meteo, Temporal, Behavioral, Derived)
    METEO = ["Temperature_C", "Humidity_%", "WindSpeed_m/s", "Rainfall_mm", "UV_Index"]
    df['Month_Sin'] = np.sin(2 * np.pi * pd.to_datetime(df['Date']).dt.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * pd.to_datetime(df['Date']).dt.month / 12)
    df['Solar_Block'] = (df['UV_Index'] > 0).astype(int)
    df['Rain_Binary'] = (df['Rainfall_mm'] > 0.5).astype(int)
    
    scaler_ae = MinMaxScaler()
    df['Anomaly_Score'] = get_anomaly_scores(scaler_ae.fit_transform(df[METEO]))
    
    # C. Data Preparation
    FEATS = METEO + ["Month_Sin", "Month_Cos", "Solar_Block", "Rain_Binary", "Anomaly_Score"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATS])
    
    WIN, FOR = 14, 7
    X, y = [], []
    for i in range(len(scaled) - WIN - FOR):
        X.append(scaled[i:i+WIN])
        y.append(scaled[i+WIN:i+WIN+FOR, :len(METEO)])
    X, y = np.array(X), np.array(y)
    
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    
    # D. Training & Prediction
    model = build_cnn_lstm(WIN, len(FEATS), FOR, len(METEO))
    model.fit(X_train, y_train, epochs=20, verbose=0, callbacks=[EarlyStopping(patience=3)])
    
    # E. Accuracy Test & Inverse Scaling
    preds = model.predict(X_test)
    y_test_2d, preds_2d = y_test.reshape(-1, len(METEO)), preds.reshape(-1, len(METEO))
    
    def inv(data):
        dummy = np.zeros((len(data), len(FEATS)))
        dummy[:, :len(METEO)] = data
        return scaler.inverse_transform(dummy)[:, :len(METEO)]
        
    y_true_real, y_pred_real = inv(y_test_2d), inv(preds_2d)
    
    print(f"\n{'Variable':<15} | {'MAE':<8} | {'R2 Score':<8}")
    print("-" * 40)
    metrics = {}
    for i, col in enumerate(METEO):
        m_mae = mean_absolute_error(y_true_real[:, i], y_pred_real[:, i])
        m_r2 = r2_score(y_true_real[:, i], y_pred_real[:, i])
        metrics[col] = {"MAE": round(m_mae, 4), "R2": round(m_r2, 4)}
        print(f"{col:<15} | {m_mae:<8.2f} | {m_r2:<8.2f}")
    
    # F. Visualization & Blockchain
    visualize_results(y_true_real, y_pred_real, METEO)
    WeatherBlockchainPoS().add_block({
        "data_hash": data_hash,
        "model_hash": hashlib.sha256(str(model.get_weights()).encode()).hexdigest(),
        "accuracy": metrics
    })

if __name__ == "__main__":
    run_pipeline()