"""
Technical indicators module for BIST100 MCP Trader.
All calculations are performed on a Close-price Series from an OHLCV DataFrame.
"""

import pandas as pd
import numpy as np


def calculate_sma(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Simple Moving Average of the Close price.

    Returns NaN for the first (window - 1) rows because there are not
    enough preceding observations to fill the rolling window.
    """
    return df["Close"].rolling(window=window).mean().rename(f"SMA{window}")


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Exponential Moving Average of the Close price.

    pandas uses adjust=True by default, so the first value is the simple
    mean of the first `window` rows; values before that are NaN.
    """
    return df["Close"].ewm(span=window, adjust=False).mean().rename(f"EMA{window}")


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder smoothing via EWM).

    Formula: RSI = 100 - 100 / (1 + RS), where RS = avg_gain / avg_loss.
    First (window) rows are NaN because the initial average requires
    exactly `window` price changes.
    """
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(f"RSI{window}")


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD indicator: MACD line, signal line, and histogram.

    - MACD line  = EMA(fast) - EMA(slow)  → first (slow - 1) rows are NaN
    - Signal line = EMA(signal) of MACD   → an additional (signal - 1) rows NaN
    - Histogram  = MACD - Signal

    Returns a DataFrame with columns [MACD, MACD_Signal, MACD_Hist].
    """
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"MACD": macd_line, "MACD_Signal": signal_line, "MACD_Hist": histogram}
    )


if __name__ == "__main__":
    from data_collection import fetch_ohlcv

    df = fetch_ohlcv("ASELS.IS", "2023-01-01", "2026-01-01")

    df["SMA20"] = calculate_sma(df, 20)
    df["SMA50"] = calculate_sma(df, 50)
    df["EMA20"] = calculate_ema(df, 20)
    df[f"RSI14"] = calculate_rsi(df, 14)
    df[["MACD", "MACD_Signal", "MACD_Hist"]] = calculate_macd(df)

    print("=== NaN counts per indicator column ===")
    indicator_cols = ["SMA20", "SMA50", "EMA20", "RSI14", "MACD", "MACD_Signal", "MACD_Hist"]
    print(df[indicator_cols].isnull().sum().to_string())

    print("\n=== First row where each indicator becomes valid ===")
    for col in indicator_cols:
        first_valid = df[col].first_valid_index()
        first_valid_pos = df.index.get_loc(first_valid)
        print(f"  {col:12s}: first valid at row {first_valid_pos} ({first_valid.date()})")

    print("\n=== Last 10 rows (all indicator columns) ===")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(df[["Close"] + indicator_cols].tail(10).to_string())
