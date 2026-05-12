import numpy as np
import pandas as pd
import pickle, os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ─── Config ────────────────────────────────────────────────────────────────────
WINDOW       = 24        # look-back hours
BATCH_SIZE   = 64
EPOCHS       = 100       # early stopping will cut this short
TRAIN_RATIO  = 0.8
SEED         = 42
MODEL_PATH   = "renewflow_lstm_model.keras"
SCALER_X     = "scaler_X.pkl"
SCALER_Y     = "scaler_y.pkl"

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("Loading data...")
df = pd.read_csv("renewflow_lstm_ready.csv")
print(f"  Shape: {df.shape}")

# Cyclical time encoding (hour 23 → hour 0 is a small step)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df = df.drop(columns=["timestamp", "hour", "month"])

# Split before scaling to avoid leakage
split  = int(TRAIN_RATIO * len(df))
X_cols = [c for c in df.columns if c != "generation"]
y_col  = "generation"

X_all = df[X_cols].values
y_all = df[y_col].values.reshape(-1, 1)

X_train_raw, X_test_raw = X_all[:split], X_all[split:]
y_train_raw, y_test_raw = y_all[:split], y_all[split:]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_X.fit_transform(X_train_raw)
y_train_sc = scaler_y.fit_transform(y_train_raw)
X_test_sc  = scaler_X.transform(X_test_raw)
y_test_sc  = scaler_y.transform(y_test_raw)

with open(SCALER_X, "wb") as f: pickle.dump(scaler_X, f)
with open(SCALER_Y, "wb") as f: pickle.dump(scaler_y, f)

def make_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i : i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)

X_tr, y_tr = make_sequences(X_train_sc, y_train_sc, WINDOW)
X_te, y_te = make_sequences(X_test_sc,  y_test_sc,  WINDOW)
print(f"Sequences — train: {X_tr.shape}  test: {X_te.shape}")

n_features = X_tr.shape[2]

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(WINDOW, n_features)),
    Dropout(0.2),
    BatchNormalization(),

    LSTM(64, return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),

    LSTM(32),
    Dropout(0.1),

    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)           # linear output for regression
], name="RenewFlow_LSTM_v2")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mae",
    metrics=["mse"]
)
model.summary()

print("Training...")
callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
    ),
    ModelCheckpoint(
        filepath=MODEL_PATH, monitor="val_loss",
        save_best_only=True, verbose=0
    )
]

history = model.fit(
    X_tr, y_tr,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("Evaluating...")
y_pred_sc = model.predict(X_te)
y_pred = np.clip(scaler_y.inverse_transform(y_pred_sc), 0, None)
y_true = scaler_y.inverse_transform(y_te.reshape(-1, 1))

mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")
print(f"  Model saved → {MODEL_PATH}")
