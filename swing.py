# =====================================================================
#   SWING SCREENER — Correct Code v5.0
#   Trend + Momentum + Volume + Breakout + Zones + Fakeout Detection
#   + Telegram Trigger (/scan)
# =====================================================================

import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from openpyxl import Workbook
from concurrent.futures import ThreadPoolExecutor


# =====================================================================
# TELEGRAM CONFIG
# =====================================================================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = int(os.environ.get("TELEGRAM_CHAT_ID"))

def telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    for chunk in [msg[i:i+3500] for i in range(0, len(msg), 3500)]:
        try:
            requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": chunk,
                    "parse_mode": "HTML"
                },
                timeout=10
            )
        except:
            pass


# =====================================================================
# BLACKLIST SYSTEM
# =====================================================================
BLACKLIST_FILE = None
failed_attempts = {}

def load_blacklist():
    return set()

def save_to_blacklist(symbol):
    pass


# =====================================================================
# SAFE DATA DOWNLOAD (.history — clean)
# =====================================================================
def safe_download(sym, period="1mo"):
    try:
        df = yf.Ticker(sym + ".NS").history(
            period=period,
            interval="1d",
            auto_adjust=False
        )
        if df is None or df.empty:
            return None

        df.index = pd.to_datetime(df.index)

        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in needed):
            return None

        return df.astype(float).dropna()

    except Exception:
        return None


# =====================================================================
# INDICATORS
# =====================================================================
def ema(series, length):
    return float(series.ewm(span=length, adjust=False).mean().iloc[-1])

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1])


# =====================================================================
# RESISTANCE ZONE DETECTION
# =====================================================================
def find_resistance_zones(df, lookback=60, sensitivity=0.01):
    high = df["High"].tail(lookback)

    swing_highs = []
    for i in range(1, len(high)-1):
        if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i+1]:
            swing_highs.append(high.iloc[i])

    if not swing_highs:
        return None

    swing_highs = sorted(swing_highs)
    zones = []
    current = [swing_highs[0]]

    for price in swing_highs[1:]:
        if abs(price - np.mean(current)) / np.mean(current) <= sensitivity:
            current.append(price)
        else:
            zones.append(current)
            current = [price]

    zones.append(current)

    strong = [z for z in zones if len(z) >= 2]
    if not strong:
        return None

    zone = strong[-1]
    return float(min(zone)), float(max(zone)), len(zone)


# =====================================================================
# FAKEOUT DETECTION ENGINE
# =====================================================================
def detect_fakeout(df):
    high = df["High"].iloc[-1]
    low = df["Low"].iloc[-1]
    close = df["Close"].iloc[-1]
    open_ = df["Open"].iloc[-1]

    candle_range = high - low
    upper_wick = high - max(close, open_)

    fake_wick = upper_wick >= 0.4 * candle_range

    avg_vol20 = df["Volume"].tail(20).mean()
    today_vol = df["Volume"].iloc[-1]
    fake_low_vol = today_vol < 1.2 * avg_vol20

    rsi_val = rsi(df["Close"])
    fake_rsi = rsi_val < 55

    fake_score = fake_wick + fake_low_vol + fake_rsi

    return fake_score, fake_wick, fake_low_vol, fake_rsi


# =====================================================================
# EVALUATION
# =====================================================================
def evaluate(df):
    try:
        close = df["Close"]
        volume = df["Volume"]
        high = df["High"]

        cmp_price = float(close.iloc[-1])

        ema20_val = ema(close, 20)
        ema50_val = ema(close, 50)
        rsi_val = rsi(close)

        cond_ema = ema20_val > ema50_val
        cond_rsi = 55 <= rsi_val <= 70

        avg_vol20 = float(volume.tail(20).mean())
        today_vol = float(volume.iloc[-1])
        cond_vol = today_vol > 1.2 * avg_vol20 and avg_vol20 > 200000

        high20 = float(high.tail(20).max())
        breakout20 = cmp_price > high20

        zone_low = zone_high = touches = None
        breakout_zone = False

        zone = find_resistance_zones(df)
        if zone:
            zone_low, zone_high, touches = zone
            breakout_zone = cmp_price > zone_high

        fake_score, fake_wick, fake_low_vol, fake_rsi = detect_fakeout(df)

        passed = cond_ema and cond_rsi and cond_vol

        return {
            "pass": passed,
            "CMP": cmp_price,
            "EMA20": round(ema20_val,2),
            "EMA50": round(ema50_val,2),
            "RSI": round(rsi_val,2),
            "AvgVol20": int(avg_vol20),
            "TodayVol": int(today_vol),
            "Breakout20D": breakout20,
            "ZoneBreakout": breakout_zone,
            "ZoneLow": zone_low,
            "ZoneHigh": zone_high,
            "ZoneTouches": touches,
            "FakeScore": int(fake_score),
            "FakeWick": fake_wick,
            "FakeLowVol": fake_low_vol,
            "FakeRSI": fake_rsi
        }

    except Exception:
        return None


# =====================================================================
# PROCESS SYMBOL
# =====================================================================
def process_symbol(sym):
    df = safe_download(sym, "6mo")
    if df is None or len(df) < 60:
        return None
    result = evaluate(df)
    if result and result["pass"]:
        return (sym, result)
    return None


# =====================================================================
# UNIVERSE
# =====================================================================
def fetch_nse_symbols():
    try:
        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10).json()
        return [x["symbol"].upper() for x in r["data"]]
    except:
        return None

def build_universe():
    blacklist = load_blacklist()
    symbols = fetch_nse_symbols()

    # 🔐 Cloud fallback if NSE blocks
    if not symbols:
        symbols = [
            "RELIANCE", "HDFCBANK", "ICICIBANK",
            "TCS", "INFY", "LT",
            "SBIN", "AXISBANK", "ITC",
            "BAJFINANCE", "ADANIENT"
        ]

    bad = {"", " ", "-", "NIFTY", "BANKNIFTY"}
    symbols = [s for s in symbols if s not in bad and s not in blacklist]

    valid = []
    for sym in symbols:
        df = safe_download(sym, "1mo")
        if df is not None:
            valid.append(sym)
        else:
            failed_attempts[sym] = failed_attempts.get(sym, 0) + 1
            if failed_attempts[sym] >= 2:
                save_to_blacklist(sym)

    return valid


# =====================================================================
# SCAN
# =====================================================================
def scan(universe):
    with ThreadPoolExecutor(max_workers=20) as ex:
        return [x for x in ex.map(process_symbol, universe) if x]


# =====================================================================
# MAIN SCAN EXECUTION
# =====================================================================
def main_scan():
    telegram("🔍 Cloud scan started (debug mode)")

    universe = build_universe()

    telegram(f"🧪 Universe size: {len(universe)}")

    if not universe:
        telegram("❌ Universe is EMPTY on cloud")
        return

    universe = build_universe()
    picks = scan(universe)

    ranked = []
    for sym, d in picks:
        ema_score = (d["EMA20"] - d["EMA50"]) / d["EMA50"]
        rsi_score = 1 - abs(d["RSI"] - 60) / 15
        ranked.append((ema_score + rsi_score, sym, d))

    ranked.sort(reverse=True)
    top = ranked[:15]

    if top:
        msg = "<b>🔥 Top Swing Picks</b>\n\n"

        for rank, sym, d in top:

            trend = "Strong Uptrend" if d["EMA20"] > d["EMA50"] else "Weak Trend"

            rsi_txt = ("Overbought" if d["RSI"] >= 65 
                       else "Healthy Momentum" if d["RSI"] >= 55 
                       else "Weak Momentum")

            vol_factor = d["TodayVol"] / d["AvgVol20"]
            if vol_factor >= 2:
                vol_txt = "Very Strong Volume"
            elif vol_factor >= 1.2:
                vol_txt = "Strong Volume"
            else:
                vol_txt = "Normal Volume"

            fake_label = ["LOW", "MEDIUM", "HIGH", "VERY HIGH"][d["FakeScore"]]

            zone_txt = ""
            if d["ZoneLow"] is not None:
                zone_txt = (
                    f"🟦 <b>Res Zone:</b> ₹{round(d['ZoneLow'])}-₹{round(d['ZoneHigh'])} "
                    f"(Touches: {d['ZoneTouches']})\n"
                )

            tags = []
            if d["Breakout20D"]:
                tags.append("🚀 20D Breakout")
            if d["ZoneBreakout"]:
                tags.append("🔥 Zone Breakout")

            msg += (
                f"<b>{sym}</b> {' | '.join(tags)}\n"
                f"💰 CMP: ₹{round(d['CMP'])}\n"
                f"📈 Trend: {trend}\n"
                f"💪 RSI: {round(d['RSI'])} ({rsi_txt})\n"
                f"📊 Volume: {vol_txt}\n"
                f"{zone_txt}"
                f"⚡ Fakeout Risk: {fake_label}\n"
                f"------------------------\n"
            )

        telegram(msg)
    else:
        telegram("⚠ No Swing Setups Today")

    # Excel Export
    wb = Workbook()
    ws = wb.active
    ws.append([
        "Symbol","CMP","EMA20","EMA50","RSI","AvgVol20","TodayVol",
        "Breakout20D","ZoneBreakout","ZoneLow","ZoneHigh","ZoneTouches",
        "FakeScore","FakeWick","FakeLowVol","FakeRSI"
    ])

    for rank, sym, d in top:
        ws.append([
            sym, d["CMP"], d["EMA20"], d["EMA50"], d["RSI"],
            d["AvgVol20"], d["TodayVol"], d["Breakout20D"],
            d["ZoneBreakout"], d["ZoneLow"], d["ZoneHigh"],
            d["ZoneTouches"], d["FakeScore"], d["FakeWick"],
            d["FakeLowVol"], d["FakeRSI"]
        ])

    wb.save("Top_Swing_Picks_v5.xlsx")