"""
Fundamental analysis module for BIST100 MCP Trader.
Loads and processes quarterly financial data sourced from fintables.com.
"""

import os
import pandas as pd


_FUND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fundamentals")

# Fundamentals files that hold cumulative year-to-date flows instead of single quarters.
YTD_TICKERS = {"ASELS"}
_FLOW_COLS = ["revenue", "gross_profit", "ebitda", "operating_profit", "net_profit",
              "operating_cash_flow", "investing_cash_flow", "financing_cash_flow"]


def load_fundamentals(ticker: str) -> pd.DataFrame:
    """
    Load quarterly fundamental data for a ticker.

    Reads from data/fundamentals/<TICKER>_fundamentals.csv.
    Lines starting with '#' are treated as comments and skipped.
    The 'period' column is kept as a string (e.g. '2024Q1').
    All numeric columns are coerced to float so empty cells become NaN.

    Args:
        ticker: Short ticker code (e.g. 'ASELS').  Case-insensitive.

    Returns:
        DataFrame indexed by 'period' with float columns for all metrics.

    Raises:
        FileNotFoundError: If no fundamentals file exists for the ticker.
    """
    path = os.path.join(_FUND_DIR, f"{ticker.upper()}_fundamentals.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No fundamentals file for '{ticker}'. Expected: {os.path.abspath(path)}"
        )
    df = pd.read_csv(path, comment="#")

    numeric_cols = [c for c in df.columns if c != "period"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    if ticker.upper() in YTD_TICKERS:
        # This file stores year-to-date flows (Q2 = H1, Q4 = full year); convert to single quarters
        # so TTM sums, QoQ and YoY changes are computed on comparable quarterly values.
        year = df["period"].str[:4]
        for c in [x for x in _FLOW_COLS if x in df.columns]:
            df[c] = df[c] - df[c].groupby(year).shift(1).fillna(0)

    df = df.set_index("period")
    return df


def calculate_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fundamental ratios from raw quarterly financial data and write
    the results back into the DataFrame (in-place, also returned).

    Calculations
    ------------
    Margin metrics (quarterly, no TTM needed):
        ebitda_margin     = ebitda / revenue
        net_profit_margin = net_profit / revenue

    TTM (trailing twelve months = rolling sum of last 4 quarters) metrics:
        roe               = TTM_net_profit / shareholders_equity (period-end)
        roa               = TTM_net_profit / total_assets (period-end)
        net_debt_ebitda   = net_debt (period-end) / TTM_ebitda
        First 3 rows are NaN because a full trailing year is not yet available.

    Balance-sheet ratios (no TTM, period-end balances):
        debt_equity       = financial_debt / shareholders_equity

    Liquidity ratios (require current_assets, current_liabilities, inventory columns):
        current_ratio     = current_assets / current_liabilities
        quick_ratio       = (current_assets - inventory) / current_liabilities

    Market multiples (pe_ratio, pb_ratio, ev_ebitda) depend on a live share
    price and are computed separately in calculate_market_multiples().

    Args:
        df: DataFrame returned by load_fundamentals(), modified in-place.

    Returns:
        The same DataFrame with ratio columns filled where calculable.
    """
    # --- Margin metrics (pure quarterly, no rolling) ---
    df["ebitda_margin"]     = df["ebitda"]     / df["revenue"]
    df["net_profit_margin"] = df["net_profit"] / df["revenue"]

    # --- TTM helpers (rolling 4-quarter sum, min_periods=4 → NaN for first 3) ---
    ttm_net_profit = df["net_profit"].rolling(4, min_periods=4).sum()
    ttm_ebitda     = df["ebitda"].rolling(4, min_periods=4).sum()

    # --- ROE / ROA (TTM earnings, period-end balance-sheet denominator) ---
    df["roe"] = ttm_net_profit / df["shareholders_equity"]
    df["roa"] = ttm_net_profit / df["total_assets"]

    # --- Net Debt / EBITDA (period-end net debt, TTM EBITDA) ---
    df["net_debt_ebitda"] = df["net_debt"] / ttm_ebitda

    # --- Debt / Equity (period-end balances; requires financial_debt column) ---
    if "financial_debt" in df.columns:
        df["debt_equity"] = df["financial_debt"] / df["shareholders_equity"]
    else:
        df["debt_equity"] = float("nan")

    if all(c in df.columns for c in ("current_assets", "current_liabilities", "inventory")):
        df["current_ratio"] = df["current_assets"] / df["current_liabilities"]
        df["quick_ratio"]   = (df["current_assets"] - df["inventory"]) / df["current_liabilities"]

    return df


def calculate_market_multiples(
    fundamentals_df: pd.DataFrame,
    current_price: float,
    shares_outstanding: float,
) -> pd.DataFrame:
    """
    Compute market-price-dependent multiples for every period and write the
    results (pe_ratio, pb_ratio, ev_ebitda) back into fundamentals_df in-place.

    A single share price is applied uniformly across all periods — this is a
    snapshot valuation, not a historical price series. Periods where a full
    TTM is not yet available (first 3 rows) receive NaN for PE and EV/EBITDA.

    Formulas
    --------
        market_cap (million TRY) = current_price * shares_outstanding / 1_000_000
            shares_outstanding is passed as an absolute count (e.g. 4_560_000_000).
        pe_ratio   = market_cap / TTM_net_profit   (rolling 4-quarter sum)
        pb_ratio   = market_cap / shareholders_equity  (period-end)
        ev_ebitda  = (market_cap + net_debt) / TTM_ebitda  (period-end net_debt)

    Args:
        fundamentals_df:    DataFrame from load_fundamentals() (after
                            calculate_derived_ratios() has been called).
        current_price:      Share price in TRY (point-in-time snapshot).
        shares_outstanding: Total shares as an absolute integer count.

    Returns:
        The same DataFrame with pe_ratio, pb_ratio, ev_ebitda columns filled.
    """
    df = fundamentals_df

    # shares_outstanding is an absolute count; convert to millions to stay
    # consistent with the million-TRY monetary unit used throughout the CSV.
    market_cap = current_price * (shares_outstanding / 1_000_000)  # million TRY

    ttm_net_profit = df["net_profit"].rolling(4, min_periods=4).sum()
    ttm_ebitda     = df["ebitda"].rolling(4, min_periods=4).sum()

    df["pe_ratio"] = market_cap / ttm_net_profit
    df["pb_ratio"] = market_cap / df["shareholders_equity"]
    df["ev_ebitda"] = (market_cap + df["net_debt"]) / ttm_ebitda

    return df


def calculate_yoy_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add year-over-year (YoY) growth columns for revenue and net_profit.

    YoY growth compares each quarter against the same quarter one year earlier
    (4 periods back), which eliminates seasonality distortion that would affect
    QoQ comparisons for a company like ASELS where Q4 is structurally the
    strongest quarter.

    Formula: growth = (current - prior_year) / abs(prior_year)
    Using abs() in the denominator handles the edge case where the prior-year
    value is negative.

    First 4 rows will be NaN (no prior-year quarter available).
    Results are written into df in-place and also returned.
    """
    df["revenue_growth_yoy"]    = df["revenue"].pct_change(periods=4)
    df["net_profit_growth_yoy"] = (
        df["net_profit"].diff(4) / df["net_profit"].shift(4).abs()
    )
    return df


def calculate_fundamental_score(
    df: pd.DataFrame,
    sector_avg_pe: float | None = None,
) -> pd.DataFrame:
    """
    Score each quarter on 0-5 fundamental quality criteria.

    Criteria (each worth +1 point)
    --------------------------------
    1. revenue_growth_yoy  > 0.15  (YoY revenue growth above 15%)
    2. net_profit_growth_yoy > 0.15 (YoY net profit growth above 15%)
    3. roe                 > 0.20  (return on equity above 20%, TTM)
    4. net_debt_ebitda     < 2.00  (leverage below 2x, TTM)
    5. pe_ratio            < sector_avg_pe  (skipped / NaN if sector_avg_pe is None)

    Requires calculate_yoy_growth() and calculate_derived_ratios() to have
    been called first so the necessary columns exist.

    Args:
        df:             fundamentals DataFrame with all ratio columns present.
        sector_avg_pe:  Sector average P/E for comparison. Pass None to skip
                        this criterion (score remains out of 4 for those rows).

    Returns:
        df with added columns:
            crit_revenue_growth, crit_net_profit_growth, crit_roe,
            crit_net_debt_ebitda, crit_pe  (bool/NaN each)
            fundamental_score  (int 0-5, NaN where all criteria are NaN)
    """
    df["crit_revenue_growth"]    = df["revenue_growth_yoy"]    > 0.15
    df["crit_net_profit_growth"] = df["net_profit_growth_yoy"] > 0.15
    df["crit_roe"]               = df["roe"]                   > 0.20
    df["crit_net_debt_ebitda"]   = df["net_debt_ebitda"]       < 2.00

    if sector_avg_pe is not None:
        df["crit_pe"] = df["pe_ratio"] < sector_avg_pe
    else:
        df["crit_pe"] = float("nan")

    crit_cols = [
        "crit_revenue_growth", "crit_net_profit_growth",
        "crit_roe", "crit_net_debt_ebitda", "crit_pe",
    ]
    # Sum booleans; NaN criteria contribute 0 so score is conservative
    df["fundamental_score"] = (
        df[crit_cols]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1, skipna=True)
        .astype(int)
    )

    return df


if __name__ == "__main__":
    df = load_fundamentals("ASELS")
    df = calculate_derived_ratios(df)
    df = calculate_market_multiples(df, current_price=372.50, shares_outstanding=4_560_000_000)
    df = calculate_yoy_growth(df)
    df = calculate_fundamental_score(df, sector_avg_pe=None)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 160)

    ratio_cols = [
        "ebitda_margin", "net_profit_margin",
        "roe", "roa",
        "net_debt_ebitda", "debt_equity",
        "current_ratio", "quick_ratio",
    ]

    print("=== Derived ratios ===")
    print(df[ratio_cols].to_string())

    print("\n=== NaN counts for ratio columns ===")
    print(df[ratio_cols].isnull().sum().to_string())

    print("\n=== Market multiples — last 5 periods (price=372.50 TRY, shares=4.56B) ===")
    print(df[["pe_ratio", "pb_ratio", "ev_ebitda"]].tail(5).to_string())

    print("\n=== Fundamental score — last 5 periods ===")
    score_cols = [
        "revenue_growth_yoy", "net_profit_growth_yoy",
        "crit_revenue_growth", "crit_net_profit_growth",
        "crit_roe", "crit_net_debt_ebitda", "crit_pe",
        "fundamental_score",
    ]
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df[score_cols].tail(5).to_string())

    csv_out = os.path.join(_FUND_DIR, "ASELS_fundamentals.csv")
    comment_lines = []
    with open(csv_out) as f:
        for line in f:
            if line.startswith("#"):
                comment_lines.append(line.rstrip("\n"))
            else:
                break
    with open(csv_out, "w") as f:
        for line in comment_lines:
            f.write(line + "\n")
        df.to_csv(f)
    print(f"\nCSV updated → {os.path.abspath(csv_out)}")
    print("\n=== Notes ===")
    print("  current_ratio : NaN — needs 'current_liabilities' column (add from Fintables balance sheet)")
    print("  quick_ratio   : NaN — needs 'current_liabilities' AND 'inventory' columns")
    print("  pe_ratio, pb_ratio, ev_ebitda : not in CSV — use calculate_market_multiples(df, price, shares)")
