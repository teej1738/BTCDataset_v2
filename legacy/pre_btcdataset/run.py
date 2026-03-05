# -*- coding: utf-8 -*-
"""
TRADING DATA DOWNLOADER
Futures: Hyperliquid API (BTC, ETH, SOL perps)
Spot:    Binance US API (no geo-restrictions for US users)

No API key needed. No VPN needed. Works in the US.
Saves to: a "data" folder next to this script.

HOW TO USE:
1. In terminal run:  pip install requests pandas
2. Navigate to folder:  cd C:\Users\tjall\Desktop\Trading
3. Run:  python binance_data_downloader.py
4. A "data" folder will appear with all CSV files inside.
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone

# -----------------------------------------------
# CONFIGURATION - Edit these as needed
# -----------------------------------------------

# Hyperliquid coin names for futures/perps
FUTURES_SYMBOLS = ["BTC", "ETH", "SOL"]

# Timeframes to download
INTERVALS = [
    "15m",   # ICT entry timeframe
    "1h",    # Mid-timeframe bias
    "4h",    # Higher timeframe structure
    "1d",    # Daily bias / Elliott Wave
]

# How far back to pull (Hyperliquid launched late 2023)
FUTURES_START_DATE = "2023-10-01"

# Where to save CSV files - creates a "data" folder next to this script
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Pause between API calls in seconds - keeps us under rate limits
SLEEP_BETWEEN_REQUESTS = 0.3


# -----------------------------------------------
# HYPERLIQUID FUTURES DOWNLOADER
# No API key. No geo-restrictions. Works in US.
# -----------------------------------------------

HYPERLIQUID_URL = "https://api.hyperliquid.xyz/info"

# Map interval strings to Hyperliquid format
HL_INTERVAL_MAP = {
    "1m":  "1m",
    "3m":  "3m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",
    "12h": "12h",
    "1d":  "1d",
    "1w":  "1w",
}


def date_to_ms(date_str):
    """Convert 'YYYY-MM-DD' string to Unix timestamp in milliseconds (UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_date(ms):
    """Convert Unix milliseconds to readable date string."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_interval_ms(interval):
    """Return how many milliseconds one candle covers for a given interval."""
    mapping = {
        "1m":  60000,
        "3m":  180000,
        "5m":  300000,
        "15m": 900000,
        "30m": 1800000,
        "1h":  3600000,
        "2h":  7200000,
        "4h":  14400000,
        "12h": 43200000,
        "1d":  86400000,
        "1w":  604800000,
    }
    return mapping.get(interval, 3600000)


def fetch_hyperliquid_candles(coin, interval, start_ms, end_ms):
    """
    Fetch candle data from Hyperliquid for a given coin and interval.
    Loops through batches until all data from start to end is downloaded.
    """
    hl_interval = HL_INTERVAL_MAP.get(interval, interval)
    all_candles = []
    current_start = start_ms
    batch_window_ms = get_interval_ms(interval) * 5000

    print("  Fetching " + coin + " FUTURES " + interval + " from " + ms_to_date(start_ms) + " ...")

    while current_start < end_ms:
        batch_end = min(current_start + batch_window_ms, end_ms)

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin":      coin,
                "interval":  hl_interval,
                "startTime": current_start,
                "endTime":   batch_end,
            }
        }

        try:
            response = requests.post(HYPERLIQUID_URL, json=payload, timeout=15)
            response.raise_for_status()
            batch = response.json()
        except requests.exceptions.RequestException as e:
            print("  ERROR: " + str(e))
            print("  Retrying in 5 seconds...")
            time.sleep(5)
            continue

        if not batch:
            break

        all_candles.extend(batch)

        last_candle_time = batch[-1]["t"]
        current_start = last_candle_time + get_interval_ms(interval)

        if len(all_candles) % 5000 == 0 and len(all_candles) > 0:
            print("    ... " + str(len(all_candles)) + " candles so far (up to " + ms_to_date(last_candle_time) + ")")

        if len(batch) < 10:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return all_candles


def hyperliquid_to_dataframe(candles):
    """
    Convert Hyperliquid candle data into a clean pandas DataFrame.

    Hyperliquid candle fields:
      t = open time (ms), T = close time (ms)
      o = open, h = high, l = low, c = close
      v = volume (base currency), n = number of trades
    """
    rows = []
    for c in candles:
        rows.append({
            "open_time":  c["t"],
            "open":       float(c["o"]),
            "high":       float(c["h"]),
            "low":        float(c["l"]),
            "close":      float(c["c"]),
            "volume":     float(c["v"]),
            "close_time": c["T"],
            "trades":     int(c["n"]),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["open_time_utc"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time_utc"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df.sort_values("open_time", inplace=True)
    df.drop_duplicates(subset=["open_time"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# -----------------------------------------------
# SAVE TO CSV
# -----------------------------------------------

def save_to_csv(df, symbol, market_type, interval, start_date):
    """Save DataFrame to a CSV file. Returns the full file path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    filename = symbol + "_" + market_type + "_" + interval + "_" + start_date + "_to_" + today + ".csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    df.to_csv(filepath, index=False)
    return filepath


# -----------------------------------------------
# MAIN - Controls what gets downloaded
# -----------------------------------------------

def main():
    print("=" * 55)
    print("  TRADING DATA DOWNLOADER")
    print("  Futures: Hyperliquid  |  Spot: Binance US")
    print("=" * 55)
    print("  Saving to: " + OUTPUT_DIR)
    print("  Symbols:   " + str(FUTURES_SYMBOLS))
    print("  Intervals: " + str(INTERVALS))
    print("")

    end_ms           = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    futures_start_ms = date_to_ms(FUTURES_START_DATE)

    total = len(FUTURES_SYMBOLS) * len(INTERVALS)
    count = 0

    for coin in FUTURES_SYMBOLS:
        for interval in INTERVALS:
            count += 1
            print("[" + str(count) + "/" + str(total) + "] " + coin + " FUTURES - " + interval)

            candles = fetch_hyperliquid_candles(coin, interval, futures_start_ms, end_ms)

            if not candles:
                print("  WARNING: No data returned for " + coin + " " + interval + ". Skipping.")
                print("")
                continue

            df       = hyperliquid_to_dataframe(candles)
            filepath = save_to_csv(df, coin, "FUTURES_HL", interval, FUTURES_START_DATE)

            print("  SAVED: " + str(len(df)) + " candles -> " + filepath)
            print("  Range: " + str(df["open_time_utc"].iloc[0]) + "  to  " + str(df["open_time_utc"].iloc[-1]))
            print("")

    print("=" * 55)
    print("  ALL DONE! Files are in: " + OUTPUT_DIR)
    print("=" * 55)


if __name__ == "__main__":
    main()
