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


def calculate_technical_score(
    df: pd.DataFrame,
    pivot_lows: pd.DataFrame | None = None,
    support_proximity_pct: float = 0.05,
) -> pd.DataFrame:
    """
    Score each trading day on 0-5 technical quality criteria.

    Criteria (each worth +1 point)
    --------------------------------
    1. crit_trend      : SMA20 > SMA50  (short-term MA above long-term MA)
    2. crit_price_ma   : Close > SMA20  (price above short-term trend)
    3. crit_rsi        : 30 < RSI14 < 70  (healthy momentum, not extreme)
    4. crit_macd       : MACD > MACD_Signal  (bullish MACD crossover)
    5. crit_support    : Close is within `support_proximity_pct` (default 5%)
                         above the most recent pivot low at that date.
                         Requires `pivot_lows` DataFrame from
                         support_resistance.find_support_resistance().
                         Criterion is NaN (skipped) if pivot_lows is None or
                         no prior pivot low exists for a given row.

    Prerequisite columns in df: SMA20, SMA50, RSI14, MACD, MACD_Signal.
    Call calculate_sma/ema/rsi/macd first.

    Args:
        df:                    OHLCV DataFrame with all indicator columns added.
        pivot_lows:            DataFrame of pivot low rows (subset of df) from
                               find_support_resistance(). Index must be DatetimeIndex.
        support_proximity_pct: Maximum fractional distance above the nearest
                               prior support level to award the support criterion.

    Returns:
        df with added columns:
            crit_trend, crit_price_ma, crit_rsi, crit_macd, crit_support (bool/NaN)
            technical_score (int 0-5)
    """
    df = df.copy()

    df["crit_trend"]    = df["SMA20"] > df["SMA50"]
    df["crit_price_ma"] = df["Close"] > df["SMA20"]
    df["crit_rsi"]      = (df["RSI14"] > 30) & (df["RSI14"] < 70)
    df["crit_macd"]     = df["MACD"] > df["MACD_Signal"]

    if pivot_lows is not None and not pivot_lows.empty:
        # For each trading day find the most recent prior pivot low price
        pivot_low_series = pivot_lows["Close"].sort_index()

        def _nearest_prior_support(date):
            prior = pivot_low_series[pivot_low_series.index <= date]
            return prior.iloc[-1] if not prior.empty else float("nan")

        nearest_support = df.index.map(_nearest_prior_support)
        nearest_support = pd.Series(nearest_support, index=df.index, dtype=float)

        # Within 5% above support: support <= Close <= support * (1 + pct)
        df["crit_support"] = (
            (df["Close"] >= nearest_support) &
            (df["Close"] <= nearest_support * (1 + support_proximity_pct))
        )
    else:
        df["crit_support"] = float("nan")

    crit_cols = ["crit_trend", "crit_price_ma", "crit_rsi", "crit_macd", "crit_support"]
    df["technical_score"] = (
        df[crit_cols]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1, skipna=True)
        .astype(int)
    )

    return df


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

    # --- Technical score ---
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from support_resistance import find_support_resistance

    _, pivot_lows = find_support_resistance(df, order=5)
    df = calculate_technical_score(df, pivot_lows=pivot_lows)

    score_cols = [
        "Close", "SMA20", "SMA50", "RSI14", "MACD", "MACD_Signal",
        "crit_trend", "crit_price_ma", "crit_rsi", "crit_macd", "crit_support",
        "technical_score",
    ]
    print("\n=== Technical score — last 10 rows ===")
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 160)
    print(df[score_cols].tail(10).to_string())
