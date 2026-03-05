"""
Binance Historical Data Downloader — 5m and 30m intervals
Downloads OHLCV kline data and saves to CSV files.
Output format matches existing 1h/4h/1d files exactly.
Requires: pip install requests pandas
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOLS = ["BTCUSDT"]
INTERVALS = ["5m", "30m"]
OUTPUT_DIR = r"C:\Users\tjall\Desktop\Trading\data"
LIMIT = 1000                        # max candles per request (Binance cap)

BASE_URL = "https://api.binance.com/api/v3/klines"

START_DATE = "2017-08-17"           # Binance launch date
END_DATE   = "2023-01-01"           # fills gap up to your existing data

# ── Helpers ───────────────────────────────────────────────────────────────────

RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades",
    "taker_base", "taker_quote", "ignore"
]

def date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

def interval_to_ms(interval: str) -> int:
    mapping = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000,
        "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
        "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
        "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
        "3d": 259_200_000, "1w": 604_800_000,
    }
    return mapping[interval]

# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current_start,
            "endTime":   end_ms,
            "limit":     LIMIT,
        }

        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        all_rows.extend(data)

        last_open_time = data[-1][0]
        current_start = last_open_time + interval_to_ms(interval)

        print(f"  {symbol} {interval} — fetched up to "
              f"{ms_to_dt(last_open_time).strftime('%Y-%m-%d %H:%M')} "
              f"({len(all_rows):,} candles so far)")

        time.sleep(0.25)

        if len(data) < LIMIT:
            break

    df = pd.DataFrame(all_rows, columns=RAW_COLUMNS)
    df.drop(columns=["ignore"], inplace=True)

    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_vol", "taker_base", "taker_quote"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["trades"] = df["trades"].astype(int)

    # Keep raw ms timestamps + add human-readable UTC columns (matches 1h format)
    df["open_time_utc"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time_utc"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Exact column order matching your 1h file
    df = df[[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades",
        "taker_base", "taker_quote",
        "open_time_utc", "close_time_utc"
    ]]

    return df

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_ms = date_to_ms(START_DATE) if START_DATE else 0
    end_ms   = date_to_ms(END_DATE)   if END_DATE   else int(time.time() * 1000)

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            print(f"\n{'─'*55}")
            print(f"Downloading {symbol} | {interval}")
            print(f"{'─'*55}")

            try:
                df = fetch_klines(symbol, interval, start_ms, end_ms)

                # Filename matches your convention e.g. BTCUSDT_BINANCE_5m_2017-08-17_to_2023-01-01.csv
                filename = os.path.join(
                    OUTPUT_DIR,
                    f"{symbol}_BINANCE_{interval}_{START_DATE}_to_{END_DATE}.csv"
                )
                df.to_csv(filename, index=False)

                print(f"  ✓ Saved {len(df):,} rows → {filename}")

            except requests.HTTPError as e:
                print(f"  ✗ HTTP error for {symbol} {interval}: {e}")
            except Exception as e:
                print(f"  ✗ Unexpected error for {symbol} {interval}: {e}")

    print(f"\nAll done. Files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
