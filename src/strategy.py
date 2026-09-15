"""
Strategy module for BIST100 MCP Trader.
Combines technical and fundamental scores into a unified trading signal.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_collection import fetch_ohlcv
from indicators import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_macd, calculate_technical_score,
)
from support_resistance import find_support_resistance
from fundamental_analysis import (
    load_fundamentals, calculate_derived_ratios,
    calculate_market_multiples, calculate_yoy_growth,
    calculate_fundamental_score,
)


# Quarter end dates → used to map each period string to a calendar date
# for forward-filling fundamental scores onto the daily price series.
_QUARTER_END = {
    "Q1": "03-31",
    "Q2": "06-30",
    "Q3": "09-30",
    "Q4": "12-31",
}


def _period_to_date(period: str) -> pd.Timestamp:
    """Convert a period string like '2025Q3' to its quarter-end date."""
    year, q = period[:4], period[4:]
    return pd.Timestamp(f"{year}-{_QUARTER_END[q]}")


def combine_scores(
    technical_df: pd.DataFrame,
    fundamental_df: pd.DataFrame,
    macro_weight: float = 0.10,
    tech_weight: float = 0.50,
    fund_weight: float = 0.40,
) -> pd.DataFrame:
    """
    Merge daily technical scores with quarterly fundamental scores into one
    daily DataFrame and compute a weighted Total Score.

    Fundamental score forward-fill assumption
    -----------------------------------------
    Financial results are typically published 4-6 weeks after the quarter end.
    For simplicity this function maps each period's score to its quarter-end
    date and forward-fills from that date onward.  This slightly overstates
    information availability (real traders see the data ~4-6 weeks later), but
    it is a common backtesting simplification.  To be conservative in live use,
    shift each quarter-end date forward by 35 days before forward-filling.

    Macro score and weight normalisation
    -------------------------------------
    macro_score is fixed at 0 for all rows because we are not collecting
    macroeconomic data (TCMB rate decisions, CPI, PMI, etc.) in this version.
    The macro_weight slot is reserved in the formula so it can be activated
    later without restructuring downstream code.

    IMPORTANT — artificial score suppression when macro_weight > 0:
    Allocating a non-zero weight (e.g. 0.10) to a component that is always 0
    silently deflates every Total_Score by exactly macro_weight × max_score
    (up to 0.5 points at the maximum).  In the original configuration
    (0.50T + 0.40F + 0.10M) this pushed many days below the BUY threshold
    despite strong technical and fundamental signals, causing the strategy to
    sit in cash unnecessarily.

    For this reason the final production configuration sets macro_weight=0 and
    re-normalises the remaining weights to sum to 1.0:
        technical_weight  = 0.50
        fundamental_weight = 0.50
        macro_weight       = 0.00
    This is equivalent to removing macro from the formula until real macro data
    is available.  When macro data is added, reintroduce its weight and reduce
    the other two proportionally.

    Total Score formula
    -------------------
        Total_Score = tech_weight  * technical_score
                    + fund_weight  * fundamental_score
                    + macro_weight * macro_score          # macro_score = 0

    Args:
        technical_df:   Daily DataFrame with a 'technical_score' column and
                        a DatetimeIndex.
        fundamental_df: Quarterly DataFrame with a 'fundamental_score' column
                        and a string period index (e.g. '2025Q4').
        macro_weight:   Weight for macro score (default 0.10, currently unused).
        tech_weight:    Weight for technical score (default 0.50).
        fund_weight:    Weight for fundamental score (default 0.40).

    Returns:
        Daily DataFrame with columns:
            technical_score, fundamental_score, macro_score, Total_Score
    """
    # Map quarterly fundamental scores to quarter-end dates
    fund_series = fundamental_df["fundamental_score"].copy()
    fund_series.index = pd.Index(
        [_period_to_date(p) for p in fund_series.index], name="Date"
    )

    # Reindex to the daily price calendar then forward-fill
    daily_fund = (
        fund_series
        .reindex(technical_df.index.union(fund_series.index))
        .sort_index()
        .ffill()
        .reindex(technical_df.index)
    )

    df = technical_df.copy()
    df["fundamental_score"] = daily_fund
    df["macro_score"] = 0  # placeholder — no macro data collected

    df["Total_Score"] = (
        tech_weight  * df["technical_score"]
        + fund_weight  * df["fundamental_score"]
        + macro_weight * df["macro_score"]
    )

    return df


def generate_signal(
    df: pd.DataFrame,
    buy_threshold: float = 3.5,
    sell_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Assign a trading signal to each row based on Total_Score thresholds.

    Rules
    -----
        Total_Score >= buy_threshold  → 'BUY'
        Total_Score <= sell_threshold → 'SELL'
        Otherwise                     → 'HOLD'

    Args:
        df:              DataFrame with a 'Total_Score' column.
        buy_threshold:   Minimum Total_Score to generate a BUY signal.
        sell_threshold:  Maximum Total_Score to generate a SELL signal.

    Returns:
        df with an added 'Signal' column (in-place copy).
    """
    df = df.copy()

    conditions = [
        df["Total_Score"] >= buy_threshold,
        df["Total_Score"] <= sell_threshold,
    ]
    choices = ["BUY", "SELL"]
    df["Signal"] = np.select(conditions, choices, default="HOLD")

    return df


def build_full_pipeline(
    ticker: str = "ASELS.IS",
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    current_price: float = 372.50,
    shares_outstanding: float = 4_560_000_000,
    sector_avg_pe: float = 65.6,
    buy_threshold: float = 3.5,
    sell_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Run the full data → indicators → scores → signal pipeline and return the
    combined daily DataFrame.
    """
    # --- Price data & technical indicators ---
    price_df = fetch_ohlcv(ticker, start, end)

    price_df["SMA20"] = calculate_sma(price_df, 20)
    price_df["SMA50"] = calculate_sma(price_df, 50)
    price_df["EMA20"] = calculate_ema(price_df, 20)
    price_df["RSI14"] = calculate_rsi(price_df, 14)
    price_df[["MACD", "MACD_Signal", "MACD_Hist"]] = calculate_macd(price_df)

    _, pivot_lows = find_support_resistance(price_df, order=5)
    tech_df = calculate_technical_score(price_df, pivot_lows=pivot_lows)

    # --- Fundamental data ---
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "fintables_asels.csv")
    fund_df = load_fundamentals(csv_path)
    fund_df = calculate_derived_ratios(fund_df)
    fund_df = calculate_market_multiples(fund_df, current_price, shares_outstanding)
    fund_df = calculate_yoy_growth(fund_df)
    fund_df = calculate_fundamental_score(fund_df, sector_avg_pe=sector_avg_pe)

    # --- Combine & signal ---
    combined = combine_scores(tech_df, fund_df)
    combined = generate_signal(combined, buy_threshold, sell_threshold)

    return combined


if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)

    df = build_full_pipeline()

    # Signal distribution
    print("=== Signal distribution (2024-01-01 to 2025-12-31) ===")
    dist = df["Signal"].value_counts().reindex(["BUY", "HOLD", "SELL"]).fillna(0).astype(int)
    total = len(df)
    for sig, cnt in dist.items():
        print(f"  {sig:4s}: {cnt:4d} days  ({cnt/total*100:.1f}%)")

    # Last 15 rows detail
    display_cols = [
        "Close", "technical_score", "fundamental_score",
        "Total_Score", "Signal",
    ]
    print("\n=== Last 15 days — scores & signals ===")
    print(df[display_cols].tail(15).to_string())
