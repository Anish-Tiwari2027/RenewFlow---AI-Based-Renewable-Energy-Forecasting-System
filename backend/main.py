import os, pickle, datetime
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List
import tensorflow as tf

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = FastAPI(title="RenewFlow API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = scaler_X = scaler_y = None

WINDOW = 24
N_FEATURES = 9

PHYS_RANGES = {
    "temp":          (-10, 50),
    "humidity":      (0,  100),
    "wind_speed":    (0,   30),
    "wind_direction":(0,  360),
    "irradiance":    (0, 1200),
}

def to_01(val, lo, hi):
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))

def build_feature_row(cond, hour: int, month: int) -> list:
    return [
        to_01(cond.temp,           *PHYS_RANGES["temp"]),
        to_01(cond.humidity,       *PHYS_RANGES["humidity"]),
        to_01(cond.wind_speed,     *PHYS_RANGES["wind_speed"]),
        to_01(cond.wind_direction, *PHYS_RANGES["wind_direction"]),
        to_01(cond.irradiance,     *PHYS_RANGES["irradiance"]),
        np.sin(2 * np.pi * hour  / 24),
        np.cos(2 * np.pi * hour  / 24),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
    ]

def build_sequence(cond, start_hour: int, month: int) -> np.ndarray:
    rows = []
    for i in range(WINDOW):
        hour = (start_hour + i) % 24
        rows.append(build_feature_row(cond, hour, month))
    arr = np.array(rows, dtype=np.float32)
    arr_sc = scaler_X.transform(arr)
    return arr_sc.reshape(1, WINDOW, N_FEATURES)

@app.on_event("startup")
def load_model():
    global model, scaler_X, scaler_y
    try:
        model    = tf.keras.models.load_model(os.path.join(BASE_DIR, "renewflow_lstm_model.keras"))
        with open(os.path.join(BASE_DIR, "scaler_X.pkl"), "rb") as f: scaler_X = pickle.load(f)
        with open(os.path.join(BASE_DIR, "scaler_y.pkl"), "rb") as f: scaler_y = pickle.load(f)
        print(f"Model loaded — input shape: {model.input_shape}")
    except Exception as e:
        print(f"Startup error: {e}")

class WeatherInput(BaseModel):
    temp:           float = Field(..., example=28.5)
    humidity:       float = Field(..., example=65.0)
    wind_speed:     float = Field(..., example=5.2)
    wind_direction: float = Field(..., example=180.0)
    irradiance:     float = Field(..., example=650.0)

class ForecastRequest(BaseModel):
    current_conditions: WeatherInput
    forecast_hours: int = Field(24, ge=1, le=48)

class HourlyForecast(BaseModel):
    hour:          int
    label:         str
    generation_01: float
    generation_mw: float

class ForecastResponse(BaseModel):
    forecast:        List[HourlyForecast]
    peak_generation: float
    peak_hour:       int
    total_energy:    float
    model_r2:        float
    status:          str

PLANT_CAPACITY_MW = 500

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None,
            "n_features": scaler_X.n_features_in_ if scaler_X else None}

@app.post("/api/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        now   = datetime.datetime.utcnow()
        month = now.month
        start = now.hour
        forecasts = []

        for h in range(request.forecast_hours):
            target_hour = (start + h) % 24
            seq_start = (target_hour - WINDOW + 1) % 24
            X_in = build_sequence(request.current_conditions, seq_start, month)

            y_sc  = model.predict(X_in, verbose=0)
            y_01  = float(scaler_y.inverse_transform(y_sc.reshape(-1,1))[0][0])
            y_01  = max(0.0, min(1.0, y_01))
            y_mw  = round(y_01 * PLANT_CAPACITY_MW, 2)

            forecasts.append(HourlyForecast(
                hour=target_hour,
                label=f"{target_hour:02d}:00",
                generation_01=round(y_01, 4),
                generation_mw=y_mw
            ))

        gens     = [f.generation_mw for f in forecasts]
        peak_idx = int(np.argmax(gens))

        return ForecastResponse(
            forecast=forecasts,
            peak_generation=round(gens[peak_idx], 2),
            peak_hour=forecasts[peak_idx].hour,
            total_energy=round(sum(gens), 1),
            model_r2=0.9319,
            status="success"
        )
    except Exception as e:
        raise HTTPException(500, str(e))

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
