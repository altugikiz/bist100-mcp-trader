"""
Macro context module for BIST100 MCP Trader.

Provides daily macro series (FX rates, TCMB policy rate, TÜİK CPI YoY,
Fed funds rate) for use as conditioning variables in strategy and analysis.

Design decision: TCMB EVDS requires free registration for API access, which
was outside this project's time budget. FX series were retrieved directly from
Yahoo Finance (no key required). TCMB policy rate and Fed funds rate are exact,
sourced from official meeting-by-meeting decisions (infrequent, discrete
changes — fully accurate). TÜİK CPI YoY uses a small set of confirmed anchor
points from public financial news sources with linear interpolation between
them, since TÜİK's official bulk API was not accessible without additional
registration; this introduces interpolation error for months between anchors
and is documented as a limitation.
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

_SRC       = os.path.dirname(__file__)
_MACRO_DIR = os.path.join(_SRC, "..", "data", "macro")


# ---------------------------------------------------------------------------
# Constant rate tables (discrete step functions)
# ---------------------------------------------------------------------------

# TCMB policy rate — exact from official MPC meeting decisions
TCMB_POLICY_RATE_CHANGES: pd.DataFrame = pd.DataFrame([
    ("2024-01-01", 45.00),
    ("2024-03-21", 50.00),
    ("2024-12-26", 47.50),
    ("2025-01-23", 45.00),
    ("2025-03-06", 42.50),
    ("2025-04-17", 46.00),
    ("2025-07-24", 43.00),
    ("2025-09-11", 40.50),
    ("2025-10-23", 39.50),
    ("2025-12-11", 38.00),
    ("2026-01-22", 37.00),   # held at 37.00 through the 2026-09-10 MPC meeting
], columns=["date", "rate_pct"])
TCMB_POLICY_RATE_CHANGES["date"] = pd.to_datetime(TCMB_POLICY_RATE_CHANGES["date"])

# TÜİK CPI YoY anchor points — confirmed from public financial news
# Linear interpolation is applied between anchors; extrapolation outside
# the anchor range is left as NaN to avoid ungrounded estimates.
TUFE_YOY_ANCHORS: pd.DataFrame = pd.DataFrame([
    ("2024-05-31", 75.45),
    ("2024-12-31", 44.38),
    ("2025-01-31", 42.12),
    ("2025-02-28", 39.05),
    ("2025-03-31", 38.10),
    ("2025-04-30", 37.86),
    ("2025-05-31", 35.41),
    ("2025-06-30", 35.05),
    ("2025-07-31", 33.52),
    ("2025-08-31", 32.95),
    ("2025-09-30", 33.29),
    ("2025-10-31", 32.87),
    ("2025-11-30", 31.07),
    ("2025-12-31", 30.89),
    ("2026-01-31", 30.65),
    ("2026-02-28", 31.53),
    ("2026-03-31", 30.87),
    ("2026-04-30", 32.37),
    ("2026-05-31", 32.61),
    ("2026-06-30", 32.11),
    ("2026-07-31", 31.75),
    ("2026-08-31", 31.51),   # last print published on/before the 2026-09-10 analysis date
], columns=["date", "yoy_pct"])
TUFE_YOY_ANCHORS["date"] = pd.to_datetime(TUFE_YOY_ANCHORS["date"])

# Fed funds rate — exact upper-bound target from FOMC decisions
# (no FOMC change between 2025-12-10 and 2026-09-10; next move was 2026-09-16, after the analysis date)
FED_FUNDS_RATE_CHANGES: pd.DataFrame = pd.DataFrame([
    ("2024-01-01", 5.50),
    ("2024-09-18", 5.00),
    ("2024-11-07", 4.75),
    ("2024-12-18", 4.50),
    ("2025-09-17", 4.25),
    ("2025-10-29", 4.00),
    ("2025-12-10", 3.75),
], columns=["date", "rate_pct"])
FED_FUNDS_RATE_CHANGES["date"] = pd.to_datetime(FED_FUNDS_RATE_CHANGES["date"])


# ---------------------------------------------------------------------------
# 1. FX series from Yahoo Finance
# ---------------------------------------------------------------------------

def fetch_fx_series(
    start: str = "2023-01-01",
    end:   str = "2026-09-11",
) -> pd.DataFrame:
    """
    Download daily USD/TRY and EUR/TRY closing rates from Yahoo Finance.

    Args:
        start: Start date (YYYY-MM-DD, inclusive).
        end:   End date   (YYYY-MM-DD, exclusive).

    Returns:
        DataFrame with DatetimeIndex and columns [usdtry, eurtry].
        Missing dates (weekends/holidays) are forward-filled from the
        last available trading day.
    """
    raw = yf.download(
        ["USDTRY=X", "EURTRY=X"],
        start=start, end=end,
        auto_adjust=True, progress=False,
    )
    close = raw["Close"].copy()
    close.columns = ["eurtry", "usdtry"]   # yfinance alphabetical order
    close = close[["usdtry", "eurtry"]]
    close.index.name = "Date"

    # Forward-fill to cover weekends and holidays
    close = close.ffill()
    return close


# ---------------------------------------------------------------------------
# Helpers: step-function and linear interpolation builders
# ---------------------------------------------------------------------------

def _step_series(
    changes_df: pd.DataFrame,
    date_col:   str,
    value_col:  str,
    date_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Build a daily step-function series from a table of change dates and values.

    The value in effect on any given day is the value from the most recent
    change date on or before that day.  Days before the first change date
    receive NaN.
    """
    result = pd.Series(np.nan, index=date_index, dtype=float)
    sorted_changes = changes_df.sort_values(date_col)
    for _, row in sorted_changes.iterrows():
        mask = date_index >= row[date_col]
        result.loc[mask] = row[value_col]
    return result


def _interpolated_series(
    anchors_df: pd.DataFrame,
    date_col:   str,
    value_col:  str,
    date_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Build a daily series by linearly interpolating between anchor points.

    Days before the first anchor are NaN; days after the last anchor carry the
    last published value forward (no extrapolation of a trend).
    """
    anchors = anchors_df.sort_values(date_col).copy()
    anchor_timestamps = anchors[date_col].values.astype("int64")
    anchor_values     = anchors[value_col].values.astype(float)

    index_timestamps  = date_index.values.astype("int64")
    t_min, t_max      = anchor_timestamps[0], anchor_timestamps[-1]

    interp_values = np.interp(
        index_timestamps,
        anchor_timestamps,
        anchor_values,
    )

    result = pd.Series(interp_values, index=date_index, dtype=float)
    # Mask extrapolation regions
    result.loc[date_index < pd.Timestamp(anchors[date_col].iloc[0])]  = np.nan
    # After the last anchor carry the last published value forward (point-in-time:
    # the latest known print stays valid until the next release).
    result.loc[date_index > pd.Timestamp(anchors[date_col].iloc[-1])] = anchor_values[-1]

    return result


# ---------------------------------------------------------------------------
# 5. Unified daily macro DataFrame
# ---------------------------------------------------------------------------

def build_daily_macro_series(
    start: str = "2024-01-01",
    end:   str = "2026-09-10",
) -> pd.DataFrame:
    """
    Construct a single daily macro DataFrame combining all four sources.

    Columns:
        usdtry    : daily USD/TRY close (Yahoo Finance, ffill on non-trading days)
        eurtry    : daily EUR/TRY close (same source)
        tcmb_rate : TCMB one-week repo rate (%, step function, forward-filled)
        tufe_yoy  : TÜİK CPI YoY (%, linearly interpolated between anchor points;
                    NaN outside the anchor range 2024-05-31 … 2025-12-31)
        fed_rate  : Fed funds upper-bound target (%, step function, forward-filled)

    Args:
        start: First date of the output series (YYYY-MM-DD).
        end:   Last date of the output series (YYYY-MM-DD, inclusive).

    Returns:
        DataFrame with a complete calendar DatetimeIndex (all days, not just
        trading days) from start to end.
    """
    date_index = pd.date_range(start=start, end=end, freq="D")

    # FX — fetch a slightly wider window to ensure ffill covers the start date
    fx = fetch_fx_series(
        start=str((pd.Timestamp(start) - pd.Timedelta(days=7)).date()),
        end=str((pd.Timestamp(end) + pd.Timedelta(days=2)).date()),
    )
    fx_reindexed = fx.reindex(date_index).ffill()

    macro = pd.DataFrame(index=date_index)
    macro.index.name = "Date"
    macro["usdtry"]    = fx_reindexed["usdtry"]
    macro["eurtry"]    = fx_reindexed["eurtry"]
    macro["tcmb_rate"] = _step_series(
        TCMB_POLICY_RATE_CHANGES, "date", "rate_pct", date_index
    )
    macro["tufe_yoy"]  = _interpolated_series(
        TUFE_YOY_ANCHORS, "date", "yoy_pct", date_index
    )
    macro["fed_rate"]  = _step_series(
        FED_FUNDS_RATE_CHANGES, "date", "rate_pct", date_index
    )

    return macro


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 120)

    print("Building daily macro series 2024-01-01 … 2026-09-10 …")
    macro = build_daily_macro_series("2024-01-01", "2026-09-10")

    os.makedirs(_MACRO_DIR, exist_ok=True)
    out_path = os.path.join(_MACRO_DIR, "macro_daily.csv")
    macro.to_csv(out_path)
    print(f"Saved → {os.path.abspath(out_path)}  ({len(macro):,} rows)")

    print("\n=== Last 10 rows ===")
    print(macro.tail(10).to_string())

    print("\n=== NaN counts and rates ===")
    nan_summary = pd.DataFrame({
        "nan_count": macro.isna().sum(),
        "nan_pct_%": (macro.isna().mean() * 100).round(2),
        "note": [
            "weekends/holidays should be 0 (ffilled)",
            "weekends/holidays should be 0 (ffilled)",
            "NaN only before 2024-01-01 first entry",
            "NaN before 2024-05-31 anchor (no extrapolation)",
            "NaN only before 2024-01-01 first entry",
        ],
    })
    print(nan_summary.to_string())

    print("\n=== TCMB rate step-function check (unique values) ===")
    print(macro["tcmb_rate"].value_counts().sort_index().to_string())

    print("\n=== TÜİK YoY range check ===")
    valid_tufe = macro["tufe_yoy"].dropna()
    print(f"  Coverage: {valid_tufe.index[0].date()} … {valid_tufe.index[-1].date()}")
    print(f"  Min: {valid_tufe.min():.2f}%  Max: {valid_tufe.max():.2f}%")
    print(f"  First valid: {valid_tufe.iloc[0]:.2f}%  Last valid: {valid_tufe.iloc[-1]:.2f}%")
