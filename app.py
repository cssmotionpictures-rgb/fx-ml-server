from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import yfinance as yf
from typing import Optional

app = FastAPI()

# Load model once at startup
ort_session = ort.InferenceSession("fx_beast_lstm.onnx")

SCALER_MEAN = np.array([2.54e-6, 1.746e-5, 50.49797161, 0.03999965, 0.02667172, 0.05687337,
                        0.98074726, 0.00463077, 0.0, 1.028e-5, 2.981e-5, 7.396e-5,
                        0.00153139, 0.49979699], dtype=np.float32)
SCALER_STD = np.array([0.00098456, 0.00215526, 11.9659512, 1.21774518, 0.67419372,
                       0.9680574, 0.45361827, 0.57239448, 1.0, 0.00119222,
                       0.00206979, 0.00320913, 0.00103524, 0.28858137], dtype=np.float32)

SEQ_LEN = 48
N_FEATURES = 14

YAHOO_SYMBOLS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X", "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X", "AUD/JPY": "AUDJPY=X", "CHF/JPY": "CHFJPY=X",
    "EUR/AUD": "EURAUD=X", "EUR/CAD": "EURCAD=X", "EUR/CHF": "EURCHF=X",
    "GBP/AUD": "GBPAUD=X", "GBP/CAD": "GBPCAD=X", "GBP/CHF": "GBPCHF=X",
    "AUD/CAD": "AUDCAD=X", "AUD/CHF": "AUDCHF=X", "CAD/CHF": "CADCHF=X",
    "NZD/JPY": "NZDJPY=X", "CAD/JPY": "CADJPY=X", "GBP/NZD": "GBPNZD=X",
    "EUR/NZD": "EURNZD=X", "AUD/NZD": "AUDNZD=X",
}

def ema(arr, period):
    k = 2 / (period + 1)
    out = np.zeros_like(arr); out[0] = arr[0]
    for i in range(1, len(arr)): out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out

def rsi_series(arr, period=14):
    delta = np.diff(arr, prepend=arr[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_g = np.zeros_like(arr); avg_l = np.zeros_like(arr)
    for i in range(1, len(arr)):
        avg_g[i] = (avg_g[i-1] * (period-1) + gain[i]) / period if i > 1 else gain[i]
        avg_l[i] = (avg_l[i-1] * (period-1) + loss[i]) / period if i > 1 else loss[i]
    rs = avg_g / (avg_l + 1e-9)
    return 100 - 100 / (1 + rs)

def atr_series(h, l, c, period=14):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    out = np.zeros_like(c); out[0] = tr[0]
    for i in range(1, len(c)): out[i] = (out[i-1] * (period-1) + tr[i]) / period
    return out

def build_features(df):
    o = df["Open"].values.astype(np.float32)
    h = df["High"].values.astype(np.float32)
    l = df["Low"].values.astype(np.float32)
    c = df["Close"].values.astype(np.float32)
    v = df["Volume"].values.astype(np.float32)
    e8, e21, e50 = ema(c, 8), ema(c, 21), ema(c, 50)
    r14 = rsi_series(c, 14)
    a14 = atr_series(h, l, c, 14)
    n = len(c)
    vol_mean = v.mean()
    vol_std = v.std() or 1e-9
    rows = []
    for i in range(n):
        ret1 = (c[i] - c[i-1]) / c[i-1] if i > 0 else 0
        ret5 = (c[i] - c[i-5]) / c[i-5] if i >= 5 else 0
        rows.append([
            ret1, ret5, r14[i],
            (c[i] - e21[i]) / (a14[i] + 1e-9),
            (e8[i] - e21[i]) / (a14[i] + 1e-9),
            (e21[i] - e50[i]) / (a14[i] + 1e-9),
            (h[i] - l[i]) / (a14[i] + 1e-9),
            (c[i] - o[i]) / (a14[i] + 1e-9),
            (v[i] - vol_mean) / vol_std,
            c[i] / e8[i] - 1, c[i] / e21[i] - 1, c[i] / e50[i] - 1,
            a14[i] / (c[i] + 1e-9),
            i / n,
        ])
    return np.array(rows, dtype=np.float32)

class PredictRequest(BaseModel):
    pair: Optional[str] = None

@app.get("/")
def health():
    return {"status": "ok", "model": "fx_beast_lstm.onnx"}

@app.post("/predict")
def predict(req: PredictRequest):
    pairs = [req.pair] if req.pair else list(YAHOO_SYMBOLS.keys())
    results = []
    for pair in pairs:
        try:
            symbol = YAHOO_SYMBOLS.get(pair)
            if not symbol: continue
            df = yf.download(symbol, period="2mo", interval="1h", progress=False, auto_adjust=False)
            if len(df) < SEQ_LEN + 10: continue
            if hasattr(df.columns, "get_level_values"):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            F = build_features(df)
            seq = F[-SEQ_LEN:]
            scaled = (seq - SCALER_MEAN) / (SCALER_STD + 1e-9)
            inp = scaled.reshape(1, SEQ_LEN, N_FEATURES).astype(np.float32)
            input_name = ort_session.get_inputs()[0].name
            outputs = ort_session.run(None, {input_name: inp})
            prob_up = float(outputs[0][0][0])
            results.append({
                "pair": pair,
                "prob_up": prob_up,
                "prob_down": 1 - prob_up,
                "confidence": round(max(prob_up, 1 - prob_up) * 100),
                "direction": "BUY" if prob_up > 0.5 else "SELL",
            })
        except Exception as e:
            results.append({"pair": pair, "error": str(e)})
    return {"predictions": [r for r in results if "error" not in r], "errors": [r for r in results if "error" in r]}
