"""
Refresh the local price + macro caches up to the analysis date 2026-09-10.

Run once WITH internet:   BIST_ALLOW_NETWORK=1 python scripts/refresh_data_2026.py
The old caches are copied to data/_backup_before_refresh/ first. Full history is
re-downloaded (not appended) because yfinance auto_adjust retroactively changes
older prices after splits/dividends.
"""
import os
import shutil
import sys
import time

import pandas as pd
import yfinance as yf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

START, END_EXCL, LAST = "2023-01-01", "2026-09-11", pd.Timestamp("2026-09-10")
CACHE = os.path.join(ROOT, "data", "universe_cache")
BACKUP = os.path.join(ROOT, "data", "_backup_before_refresh")


def download(symbol):
    raw = yf.download(symbol, start=START, end=END_EXCL, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df[df.index <= LAST]


def main():
    from data.universe import UNIVERSE
    os.makedirs(BACKUP, exist_ok=True)
    for f in os.listdir(CACHE):
        if f.endswith(".csv"):
            shutil.copy2(os.path.join(CACHE, f), os.path.join(BACKUP, f))
    macro = os.path.join(ROOT, "data", "macro", "macro_daily.csv")
    if os.path.exists(macro):
        shutil.copy2(macro, os.path.join(BACKUP, "macro_daily.csv"))
    print(f"Backed up old caches to {os.path.abspath(BACKUP)}")

    targets = [(t, s) for t, s in UNIVERSE[["ticker", "yahoo_symbol"]].values.tolist()] + [("XU100", "XU100.IS")]
    bad = []
    for ticker, sym in targets:
        try:
            df = download(sym)
            if len(df) < 800 or df.index[-1] < pd.Timestamp("2026-09-01"):
                raise ValueError(f"suspicious result: {len(df)} rows, last {df.index[-1].date() if len(df) else None}")
            df.to_csv(os.path.join(CACHE, f"{ticker}.csv"))
            print(f"{ticker:7s} rows={len(df):4d}  {df.index[0].date()} -> {df.index[-1].date()}")
        except Exception as exc:
            bad.append(ticker)
            print(f"{ticker:7s} FAILED: {exc}")
        time.sleep(0.5)

    from analysis.macro_context import build_daily_macro_series
    m = build_daily_macro_series("2024-01-01", "2026-09-10")
    os.makedirs(os.path.dirname(macro), exist_ok=True)
    m.to_csv(macro)
    print(f"macro_daily.csv rows={len(m)} last={m.index[-1].date()}")
    print(m.tail(3).to_string())
    print("FAILED tickers:", bad if bad else "none")


if __name__ == "__main__":
    main()
