"""
Point-in-time fundamental alignment for BIST100 MCP Trader.

Prevents look-ahead bias when merging fundamental data with price series by
anchoring each quarter's data to its estimated KAP public disclosure date
rather than the quarter-end date.

Design decision: KAP (Public Disclosure Platform) does not offer a free bulk
API for historical disclosure dates. A sector-based fixed-lag proxy is used
instead (30 days for Financials, 45 days for all other sectors — reflecting
BDDK/SPK's faster reporting requirement for financial institutions). This proxy
is validated against real KAP disclosure dates for ASELS (see
validate_proxy_accuracy) as a methodological limitation check.
"""

import os
import sys
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from typing import Dict, Optional

_SRC = os.path.dirname(__file__)
sys.path.insert(0, _SRC)

from universe import UNIVERSE

# Reporting-lag constants (calendar days after quarter-end)
_LAG_FINANCIALS = 30   # BDDK/SPK requirement for banks, insurers, brokers
_LAG_DEFAULT    = 45   # CMB general requirement for all other issuers


# ---------------------------------------------------------------------------
# Quarter-period helpers
# ---------------------------------------------------------------------------

_QUARTER_END: Dict[str, str] = {
    "Q1": "03-31",
    "Q2": "06-30",
    "Q3": "09-30",
    "Q4": "12-31",
}


def _parse_quarter_end(period: str) -> pd.Timestamp:
    """
    Convert a period string like '2024Q1' to its quarter-end date (2024-03-31).

    Args:
        period: String in the format YYYYQn (e.g. '2024Q3').

    Returns:
        pd.Timestamp of the last calendar day of that quarter.
    """
    year, q = period[:4], period[4:]           # '2024', 'Q1'
    mmdd = _QUARTER_END[q]
    return pd.Timestamp(f"{year}-{mmdd}")


# ---------------------------------------------------------------------------
# 1. Disclosure date estimation
# ---------------------------------------------------------------------------

def estimate_disclosure_date(
    quarter_end_date: pd.Timestamp,
    sector_group: str,
) -> pd.Timestamp:
    """
    Estimate the KAP public disclosure date for a quarter's results.

    Adds a fixed lag to the quarter-end date based on sector regulation:
    - Financials sector_group : +30 calendar days (BDDK/SPK fast-track)
    - All other sector_groups  : +45 calendar days (CMB standard)

    If the resulting date falls on a weekend, it is rolled forward to the
    next business day using pandas.tseries.offsets.BDay.

    Args:
        quarter_end_date: Last day of the reporting quarter.
        sector_group:     sector_group label from universe.py.

    Returns:
        pd.Timestamp of the estimated disclosure date (always a business day).
    """
    lag = _LAG_FINANCIALS if sector_group == "Financials" else _LAG_DEFAULT
    raw_date = quarter_end_date + pd.Timedelta(days=lag)

    # Roll forward to next business day if on a weekend (BDay offset = 0 on weekdays)
    if raw_date.weekday() >= 5:          # 5 = Saturday, 6 = Sunday
        raw_date = raw_date + BDay(1)

    return raw_date


# ---------------------------------------------------------------------------
# 2. Build point-in-time fundamentals DataFrame
# ---------------------------------------------------------------------------

def build_point_in_time_fundamentals(
    ticker: str,
    fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add quarter_end and disclosure_date columns to a fundamentals DataFrame.

    The 'period' column (e.g. '2024Q1') is parsed into:
    - quarter_end     : last calendar day of the quarter
    - disclosure_date : estimated KAP disclosure date (quarter_end + lag, adjusted
                        to next business day)

    The ticker's sector_group is looked up from universe.py to select the
    appropriate reporting lag.

    Args:
        ticker:          Short ticker code (e.g. 'ASELS').
        fundamentals_df: DataFrame with a 'period' column in YYYYQn format.

    Returns:
        Copy of fundamentals_df with 'quarter_end' and 'disclosure_date' prepended.

    Raises:
        ValueError: If the ticker is not found in the universe.
    """
    row = UNIVERSE[UNIVERSE["ticker"] == ticker.upper()]
    if row.empty:
        raise ValueError(f"Ticker '{ticker}' not found in universe.")
    sector_group = row.iloc[0]["sector_group"]

    df = fundamentals_df.copy()
    df["quarter_end"]     = df["period"].apply(_parse_quarter_end)
    df["disclosure_date"] = df["quarter_end"].apply(
        lambda qe: estimate_disclosure_date(qe, sector_group)
    )
    df["disclosure_lag_days"] = (df["disclosure_date"] - df["quarter_end"]).dt.days
    df["sector_group"]        = sector_group

    # Reorder: put new columns first
    lead_cols = ["period", "quarter_end", "disclosure_date", "disclosure_lag_days", "sector_group"]
    rest_cols = [c for c in df.columns if c not in lead_cols]
    return df[lead_cols + rest_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Align price series to disclosure dates (point-in-time forward-fill)
# ---------------------------------------------------------------------------

def align_price_to_disclosure(
    price_df: pd.DataFrame,
    pit_fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge fundamental data into the price series using disclosure_date as the
    anchor, not quarter_end.

    For each trading day t, only fundamentals whose disclosure_date ≤ t are
    visible. The most recently disclosed quarter's figures are forward-filled
    until the next disclosure arrives.

    This eliminates the look-ahead bias present in quarter_end-based merges,
    where data from (e.g.) 2024-Q1 would appear on 2024-03-31 even though it
    was not published until 2024-05-15.

    Args:
        price_df:            OHLCV DataFrame with DatetimeIndex.
        pit_fundamentals_df: Output of build_point_in_time_fundamentals().

    Returns:
        price_df extended with fundamental columns; a 'fundamental_vintage'
        column records which quarter's data is active on each day.
    """
    fund = pit_fundamentals_df.copy()
    fund = fund.sort_values("disclosure_date").reset_index(drop=True)

    # Exclude metadata columns that are not fundamental metrics
    skip_cols = {"period", "quarter_end", "disclosure_date", "disclosure_lag_days",
                 "sector_group"}
    fundamental_cols = [c for c in fund.columns if c not in skip_cols
                        and not c.startswith("crit_")]

    price = price_df.copy()
    price["fundamental_vintage"] = pd.NaT

    # Forward-fill: for each disclosure, mark all subsequent price rows
    for _, frow in fund.iterrows():
        disc_date = frow["disclosure_date"]
        mask = price.index >= disc_date
        price.loc[mask, "fundamental_vintage"] = frow["period"]
        for col in fundamental_cols:
            price.loc[mask, col] = frow[col]

    return price


# ---------------------------------------------------------------------------
# 4. Proxy accuracy validation
# ---------------------------------------------------------------------------

def validate_proxy_accuracy(
    ticker: str,
    real_disclosure_dates: Dict[str, Optional[str]],
    fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare estimated disclosure dates against real KAP disclosure dates.

    Args:
        ticker:                Short ticker code (e.g. 'ASELS').
        real_disclosure_dates: Dict mapping period → real disclosure date string
                               (YYYY-MM-DD) or None if not yet collected.
                               Example: {'2024Q1': '2024-05-10', '2024Q2': None}
        fundamentals_df:       Raw fundamentals DataFrame with 'period' column.

    Returns:
        DataFrame with columns: period, quarter_end, estimated_date, real_date,
        error_days (proxy − real; positive = proxy disclosed later than actual).
        Also prints MAE over non-None entries.
    """
    pit = build_point_in_time_fundamentals(ticker, fundamentals_df)

    rows = []
    for _, frow in pit.iterrows():
        period    = frow["period"]
        estimated = frow["disclosure_date"]
        real_str  = real_disclosure_dates.get(period)
        real_date = pd.Timestamp(real_str) if real_str else pd.NaT
        error     = (estimated - real_date).days if not pd.isna(real_date) else np.nan
        rows.append({
            "period":         period,
            "quarter_end":    frow["quarter_end"].date(),
            "estimated_date": estimated.date(),
            "real_date":      real_date.date() if not pd.isna(real_date) else None,
            "error_days":     error,
        })

    result = pd.DataFrame(rows)
    valid  = result["error_days"].dropna()
    if len(valid):
        mae = valid.abs().mean()
        bias = valid.mean()
        print(f"\nProxy validation for {ticker}:")
        print(f"  Quarters with real dates : {len(valid)} / {len(result)}")
        print(f"  MAE (|proxy − real|)     : {mae:.1f} days")
        print(f"  Mean bias (proxy − real) : {bias:+.1f} days  "
              f"({'late' if bias > 0 else 'early'} on average)")
    else:
        print(f"\nProxy validation for {ticker}: no real disclosure dates provided yet.")
        print("  Populate real_disclosure_dates dict to measure proxy accuracy.")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    # ── Load ASELS fundamentals ───────────────────────────────────────
    fund_path = os.path.join(_SRC, "..", "data", "fintables_asels.csv")
    raw_fund  = pd.read_csv(fund_path, comment="#")

    print("=== build_point_in_time_fundamentals (ASELS) ===\n")
    pit = build_point_in_time_fundamentals("ASELS", raw_fund)
    display_cols = ["period", "quarter_end", "disclosure_date", "disclosure_lag_days",
                    "sector_group", "revenue", "net_profit", "fundamental_score"]
    print(pit[display_cols].to_string(index=False))

    print("\n  quarter_end vs. disclosure_date delta (confirming Technology-Defense = 45-day lag):")
    for _, row in pit.iterrows():
        qe = row["quarter_end"]
        dd = row["disclosure_date"]
        wd = dd.strftime("%A")
        print(f"  {row['period']}  quarter_end={qe.date()}  "
              f"disclosure={dd.date()} ({wd})  lag={row['disclosure_lag_days']}d")

    # ── Load ASELS price and align ────────────────────────────────────
    price_path = os.path.join(_SRC, "..", "data", "universe_cache", "ASELS.csv")
    price_df   = pd.read_csv(price_path, index_col="Date", parse_dates=True)

    print("\n=== align_price_to_disclosure (ASELS) — first visible rows per quarter ===\n")
    aligned = align_price_to_disclosure(price_df, pit)

    # Show the first trading day each quarter's fundamentals become available
    for period in pit["period"]:
        disc_date = pit.loc[pit["period"] == period, "disclosure_date"].iloc[0]
        first_row = aligned[aligned["fundamental_vintage"] == period]
        if first_row.empty:
            continue
        first_date = first_row.index[0]
        revenue    = first_row.iloc[0].get("revenue", float("nan"))
        print(f"  {period}  disc={disc_date.date()}  "
              f"first_price_date={first_date.date()}  "
              f"revenue={revenue:.1f}M TRY")

    print("\n  Rows where fundamental_vintage is NaN (pre-first-disclosure period):")
    n_nan = aligned["fundamental_vintage"].isna().sum()
    print(f"  {n_nan} rows — price data before the first KAP disclosure in the dataset")

    # ── Quarter-end vs. disclosure-date misalignment illustration ─────
    print("\n=== Bias illustration: quarter_end vs. disclosure_date ===\n")
    print("  With quarter_end-based forward-fill, data appears on the wrong day:")
    for _, row in pit.iterrows():
        qe  = row["quarter_end"]
        dd  = row["disclosure_date"]
        diff = (dd - qe).days
        print(f"  {row['period']}  quarter_end={qe.date()}  "
              f"disclosure={dd.date()}  → look-ahead by {diff} days if using quarter_end")

    # ── validate_proxy_accuracy (placeholder — real dates TBD) ───────
    real_dates = {
        "2024Q1": None,
        "2024Q2": None,
        "2024Q3": None,
        "2024Q4": None,
        "2025Q1": None,
        "2025Q2": None,
        "2025Q3": None,
        "2025Q4": None,
    }
    print("\n=== validate_proxy_accuracy (ASELS — placeholder run) ===")
    validation = validate_proxy_accuracy("ASELS", real_dates, raw_fund)
    print(validation[["period", "quarter_end", "estimated_date", "real_date", "error_days"]]
          .to_string(index=False))
