"""
Data collection module for BIST100 MCP Trader.
Fetches historical OHLCV data using yfinance.
"""

import os
import time
import yfinance as yf
import pandas as pd
from config import SYMBOL, START_DATE, END_DATE
from universe import UNIVERSE


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV data for a given ticker from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. 'ASELS.IS')
        start:  Start date string in YYYY-MM-DD format (inclusive)
        end:    End date string in YYYY-MM-DD format (exclusive)

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] indexed by Date.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns if present (e.g. ('Close', 'ASELS.IS') -> 'Close')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only standard OHLCV columns
    ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    df = raw[ohlcv_cols].copy()

    df.index.name = "Date"
    return df


def summarise(df: pd.DataFrame) -> None:
    """Print a concise summary of the DataFrame."""
    print("=== Shape ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n=== Columns & dtypes ===")
    print(df.dtypes)

    print("\n=== Missing values per column ===")
    print(df.isnull().sum())

    print("\n=== First 10 rows ===")
    print(df.head(10).to_string())


def fetch_universe_history(
    start: str = START_DATE,
    end: str = END_DATE,
    cache_dir: str = None,
) -> dict:
    """
    Download daily OHLCV data for all 30 universe stocks and cache each to disk.

    Skips tickers whose CSV already exists in cache_dir (simple file-based cache).
    Sleeps 0.5 s between requests to stay within yfinance rate limits.

    Args:
        start:     Start date string in YYYY-MM-DD format.
        end:       End date string in YYYY-MM-DD format.
        cache_dir: Directory for cached CSV files. Defaults to data/universe_cache/
                   relative to the project root.

    Returns:
        dict mapping ticker -> DataFrame for every successfully downloaded ticker.
    """
    if cache_dir is None:
        _src = os.path.dirname(__file__)
        cache_dir = os.path.join(_src, "..", "data", "universe_cache")
    os.makedirs(cache_dir, exist_ok=True)

    results: dict = {}
    errors: dict = {}

    tickers = UNIVERSE[["ticker", "yahoo_symbol"]].values.tolist()

    for ticker, yahoo_sym in tickers:
        csv_path = os.path.join(cache_dir, f"{ticker}.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            results[ticker] = df
            continue

        try:
            df = fetch_ohlcv(yahoo_sym, start, end)
            if df.empty:
                errors[ticker] = "Empty DataFrame returned (no data for date range)"
            else:
                df.to_csv(csv_path)
                results[ticker] = df
        except Exception as exc:
            errors[ticker] = str(exc)

        time.sleep(0.5)

    # ---- Summary report ----
    total_rows = sum(len(v) for v in results.values())
    print(f"\n{'='*60}")
    print(f"Universe fetch complete: {len(results)}/{len(tickers)} tickers succeeded")
    print(f"Total rows downloaded : {total_rows:,}")
    print(f"Cache directory       : {os.path.abspath(cache_dir)}")

    if errors:
        print(f"\nFailed tickers ({len(errors)}):")
        for t, msg in errors.items():
            print(f"  {t:8s}  {msg}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    df = fetch_ohlcv(SYMBOL, START_DATE, END_DATE)
    summarise(df)
