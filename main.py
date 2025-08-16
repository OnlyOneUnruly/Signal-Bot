import os, time, threading, requests
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask
from datetime import datetime, timezone

# ===== ENV (set these on Render dashboard) =====
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

# Your markets (Yahoo tickers)
SYMBOLS = {
    "AUDCAD": "AUDCAD=X",
    "AEDCNY": "AEDCNY=X",
    "AUDUSD": "AUDUSD=X",
}

# Check frequency (seconds). 20–30s is fine for catching new 1m candles.
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "25"))

# Low-volatility filter (True to reduce false signals)
LOW_VOL_MODE = os.environ.get("LOW_VOL_MODE", "true").lower() == "true"
LOW_VOL_MULT = float(os.environ.get("LOW_VOL_MULT", "0.9"))  # ATR < 0.9 * ATR avg

# ===== Tiny web server to keep Render free tier awake =====
app = Flask(__name__)

@app.get("/")
def home():
    return "ok"

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

def run_web():
    port = int(os.environ.get("PORT", "10000"))
    # threaded=True so it doesn't block our bot loop
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)

# ===== Indicators (no TA-Lib needed) =====
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

# ===== Data + signals =====
def get_minute_data(y_symbol: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(y_symbol).history(period="1d", interval="1m", auto_adjust=False)
        if df is None or df.empty:
            return None
        # yfinance returns index as datetime; standardize columns
        df = df.rename(columns=str)  # ensure regular strings
        return df[["Open", "High", "Low", "Close"]].dropna()
    except Exception as e:
        print(f"[ERR] fetch {y_symbol}: {e}")
        return None

def calc_signal(df: pd.DataFrame) -> str | None:
    if df is None or len(df) < 30:
        return None
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ema_fast = ema(close, 5)
    ema_slow = ema(close, 20)
    rsi14 = rsi(close, 14)
    atr14 = atr(high, low, close, 14)

    last = -1
    if any(pd.isna(x.iloc[last]) for x in (ema_fast, ema_slow, rsi14, atr14)):
        return None

    trend_up = ema_fast.iloc[last] > ema_slow.iloc[last]
    trend_down = ema_fast.iloc[last] < ema_slow.iloc[last]
    rsi_buy = rsi14.iloc[last] > 55
    rsi_sell = rsi14.iloc[last] < 45

    vol_ok = True
    if LOW_VOL_MODE:
        recent = atr14.iloc[-20:].dropna()
        if len(recent) >= 5:
            vol_ok = atr14.iloc[last] < recent.mean() * LOW_VOL_MULT

    if trend_up and rsi_buy and vol_ok:
        return "BUY (1m expiry)"
    if trend_down and rsi_sell and vol_ok:
        return "SELL (1m expiry)"
    return None

def send_signal(symbol_name: str, text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("[WARN] BOT_TOKEN/CHAT_ID missing")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    msg = f"📢 {symbol_name} M1 Signal: {text}"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        print(f"[ERR] telegram: {e}")

def loop():
    last_candle_sent: dict[str, str] = {}
    while True:
        try:
            for name, ysym in SYMBOLS.items():
                df = get_minute_data(ysym)
                if df is None or df.empty:
                    print(f"[{name}] no data")
                    continue

                # last complete candle time (as string)
                candle_time = df.index[-1].isoformat()
                if last_candle_sent.get(name) == candle_time:
                    # already processed this candle
                    continue

                sig = calc_signal(df)
                if sig:
                    send_signal(name, sig)
                    print(f"[{name}] sent: {sig} @ {candle_time}")
                else:
                    print(f"[{name}] no signal @ {candle_time}")

                last_candle_sent[name] = candle_time
        except Exception as e:
            print("[LOOP ERR]", e)

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    # start tiny web server so Render keeps the service alive
    threading.Thread(target=run_web, daemon=True).start()
    loop()
