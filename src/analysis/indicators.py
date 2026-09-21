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


def calculate_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2,
) -> pd.DataFrame:
    """
    Bollinger Bands around the Close price.

    BB_Middle = SMA(window)
    BB_Upper  = BB_Middle + num_std * rolling_std(window)
    BB_Lower  = BB_Middle - num_std * rolling_std(window)

    First (window - 1) rows are NaN.
    """
    df = df.copy()
    middle = df["Close"].rolling(window=window).mean()
    std    = df["Close"].rolling(window=window).std()
    df["BB_Middle"] = middle
    df["BB_Upper"]  = middle + num_std * std
    df["BB_Lower"]  = middle - num_std * std
    return df


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average True Range (ATR) using Wilder's smoothing (EWM with com=window-1).

    True Range = max(High-Low, |High-prev_Close|, |Low-prev_Close|)
    First row is NaN (no previous close); ATR NaN for first (window) rows.
    """
    df = df.copy()
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    return df


def calculate_kama(
    df: pd.DataFrame,
    er_window: int = 10,
    fast: int = 2,
    slow: int = 30,
) -> pd.DataFrame:
    """
    Kaufman's Adaptive Moving Average (KAMA).

    Efficiency Ratio (ER) = |price_change| / sum(|daily_changes|) over er_window.
    Smoothing constant (SC) = (ER * (fast_sc - slow_sc) + slow_sc) ** 2
      where fast_sc = 2/(fast+1), slow_sc = 2/(slow+1).

    Also adds KAMA_Slope: "rising" / "falling" / "flat" based on KAMA.diff()
    relative to KAMA.std() * 0.05 as the flat-zone threshold.
    """
    df = df.copy()
    close = df["Close"].values
    n = len(close)

    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)

    kama = np.full(n, np.nan)
    # Seed KAMA at the first row that has a full ER window
    seed_idx = er_window  # index of first valid ER
    kama[seed_idx - 1] = close[seed_idx - 1]

    for i in range(seed_idx, n):
        direction = abs(close[i] - close[i - er_window])
        volatility = np.sum(np.abs(np.diff(close[i - er_window: i + 1])))
        er = direction / volatility if volatility != 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])

    df["KAMA"] = kama
    kama_series = df["KAMA"]
    kama_diff   = kama_series.diff()
    threshold   = kama_series.std() * 0.05

    df["KAMA_Slope"] = np.where(
        kama_diff > threshold,  "rising",
        np.where(kama_diff < -threshold, "falling", "flat"),
    )
    # Keep KAMA_Slope NaN where KAMA itself is NaN
    df.loc[kama_series.isna(), "KAMA_Slope"] = np.nan
    return df


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3,
) -> pd.DataFrame:
    """
    Supertrend indicator based on ATR.

    Upper band = (High+Low)/2 + multiplier * ATR(period)
    Lower band = (High+Low)/2 - multiplier * ATR(period)

    Direction flips when Close crosses a band.
    Adds columns: Supertrend (line value), Supertrend_Direction ("bullish"/"bearish").
    First `period` rows are NaN (ATR warm-up).
    """
    df = calculate_atr(df.copy(), window=period)
    atr  = df["ATR"]
    hl2  = (df["High"] + df["Low"]) / 2
    close = df["Close"]

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction   = np.full(n, np.nan)
    supertrend  = np.full(n, np.nan)

    # Find first non-NaN ATR index
    first_valid = atr.first_valid_index()
    if first_valid is None:
        df["Supertrend"] = np.nan
        df["Supertrend_Direction"] = np.nan
        return df

    start = df.index.get_loc(first_valid)
    final_upper[start] = basic_upper.iloc[start]
    final_lower[start] = basic_lower.iloc[start]
    direction[start]   = -1  # seed as bearish; will be corrected immediately
    supertrend[start]  = final_upper[start]

    for i in range(start + 1, n):
        # Upper band: only tighten (lower) if previous close was below it
        if basic_upper.iloc[i] < final_upper[i - 1] or close.iloc[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper.iloc[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Lower band: only tighten (raise) if previous close was above it
        if basic_lower.iloc[i] > final_lower[i - 1] or close.iloc[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower.iloc[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction: bullish when close > upper band, bearish when < lower band
        if direction[i - 1] == -1:
            direction[i] = 1 if close.iloc[i] > final_upper[i - 1] else -1
        else:
            direction[i] = -1 if close.iloc[i] < final_lower[i - 1] else 1

        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    df["Supertrend"] = supertrend
    dir_labels = pd.array(
        [np.nan if np.isnan(d) else ("bullish" if d == 1 else "bearish") for d in direction],
        dtype=object,
    )
    df["Supertrend_Direction"] = dir_labels
    return df


def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ichimoku Kinko Hyo indicator.

    Tenkan-sen  (Conversion): (max9  + min9)  / 2  — 9-period midpoint
    Kijun-sen   (Base):       (max26 + min26) / 2  — 26-period midpoint
    Senkou A    (Leading A):  (Tenkan + Kijun) / 2, shifted +26 periods forward
    Senkou B    (Leading B):  (max52 + min52) / 2,  shifted +26 periods forward
    Chikou Span (Lagging):    Close shifted -26 periods back

    Ichimoku_State: "above_cloud" / "below_cloud" / "in_cloud"
      compared against the cloud at today's date (Senkou A & B shifted back 26).
    """
    df = df.copy()
    high = df["High"]
    low  = df["Low"]

    tenkan = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou  = df["Close"].shift(-26)

    df["Tenkan"]   = tenkan
    df["Kijun"]    = kijun
    df["Senkou_A"] = senkou_a
    df["Senkou_B"] = senkou_b
    df["Chikou"]   = chikou

    # Cloud boundaries at current date = Senkou values shifted back 26
    cloud_top    = senkou_a.shift(-26)
    cloud_bottom = senkou_b.shift(-26)
    cloud_high   = pd.concat([cloud_top, cloud_bottom], axis=1).max(axis=1)
    cloud_low    = pd.concat([cloud_top, cloud_bottom], axis=1).min(axis=1)

    conditions = [
        df["Close"] > cloud_high,
        df["Close"] < cloud_low,
    ]
    choices = ["above_cloud", "below_cloud"]
    df["Ichimoku_State"] = np.select(conditions, choices, default="in_cloud")
    # Mark as NaN where cloud data is not yet available
    cloud_missing = cloud_high.isna() | cloud_low.isna()
    df.loc[cloud_missing, "Ichimoku_State"] = np.nan

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
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
