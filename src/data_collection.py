"""
Data collection module for BIST100 MCP Trader.
Fetches historical OHLCV data using yfinance.
"""

import yfinance as yf
import pandas as pd


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


if __name__ == "__main__":
    TICKER = "ASELS.IS"
    START  = "2023-01-01"
    END    = "2026-01-01"

    df = fetch_ohlcv(TICKER, START, END)
    summarise(df)
