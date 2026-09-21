"""
Fundamental reaction analysis for BIST100 MCP Trader.

Measures how stock prices respond to quarterly earnings disclosures by
combining point-in-time fundamental changes with post-disclosure abnormal
returns.  Covers the 8 tickers that have populated fundamentals files.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)

_SRC       = os.path.dirname(__file__)
_CACHE_DIR = os.path.join(_SRC, "..", "data", "universe_cache")
_FUND_DIR  = os.path.join(_SRC, "..", "data", "fundamentals")

sys.path.insert(0, _SRC)

from data.universe import UNIVERSE, get_relative_benchmark
from analysis.point_in_time import build_point_in_time_fundamentals
from analysis.fundamental_analysis import load_fundamentals, calculate_derived_ratios

# Tickers with populated fundamentals files (ARCLK and TUPRS are template-only)
COVERED_TICKERS = ["AKBNK", "ASELS", "BIMAS", "ENJSA", "KCHOL", "KRDMD", "PGSUS", "TOASO"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_price(ticker: str) -> pd.Series:
    path = os.path.join(_CACHE_DIR, f"{ticker}.csv")
    return pd.read_csv(path, index_col="Date", parse_dates=True)["Close"]


def _load_xu100() -> pd.Series:
    path = os.path.join(_CACHE_DIR, "XU100.csv")
    return pd.read_csv(path, index_col="Date", parse_dates=True)["Close"]


def _forward_return(close: pd.Series, date: pd.Timestamp, horizon: int) -> float:
    """(close[t+horizon] - close[t]) / close[t], or NaN if out of range."""
    idx = close.index
    pos = idx.get_indexer([date], method="ffill")[0]
    if pos < 0 or pos + horizon >= len(idx):
        return np.nan
    return (close.iloc[pos + horizon] - close.iloc[pos]) / close.iloc[pos]


def _build_pit_fundamentals(ticker: str) -> pd.DataFrame:
    """Load, compute ratios, and attach disclosure dates for one ticker."""
    raw = load_fundamentals(ticker)
    raw_reset = raw.reset_index()                          # restore 'period' column
    df  = calculate_derived_ratios(raw_reset.set_index("period").copy())
    df  = df.reset_index()                                 # period back as column
    pit = build_point_in_time_fundamentals(ticker, df)
    return pit


# ---------------------------------------------------------------------------
# 1. Quarterly fundamental changes (QoQ and YoY)
# ---------------------------------------------------------------------------

def calculate_quarterly_changes(ticker: str) -> pd.DataFrame:
    """
    Compute quarter-over-quarter and year-over-year changes in key fundamentals.

    Metrics calculated where data allows:
        revenue_growth_qoq       : (rev_t - rev_{t-1}) / |rev_{t-1}|
        net_profit_growth_qoq    : (np_t  - np_{t-1})  / |np_{t-1}|
        revenue_growth_yoy       : (rev_t - rev_{t-4}) / |rev_{t-4}|
        net_profit_growth_yoy    : (np_t  - np_{t-4})  / |np_{t-4}|
        roe_change_qoq           : roe_t - roe_{t-1}
        net_debt_ebitda_change_qoq: nd_ebitda_t - nd_ebitda_{t-1}

    Returns a DataFrame indexed by period with the change columns and
    the disclosure_date, quarter_end columns retained for downstream joins.
    """
    pit = _build_pit_fundamentals(ticker)
    df  = pit.copy()

    def _pct_change(series: pd.Series, lag: int) -> pd.Series:
        prev = series.shift(lag)
        return (series - prev) / prev.abs()

    df["revenue_growth_qoq"]          = _pct_change(df["revenue"],    1)
    df["net_profit_growth_qoq"]       = _pct_change(df["net_profit"],  1)
    df["revenue_growth_yoy"]          = _pct_change(df["revenue"],    4)
    df["net_profit_growth_yoy"]       = _pct_change(df["net_profit"],  4)
    df["roe_change_qoq"]              = df["roe"].diff(1)
    df["net_debt_ebitda_change_qoq"]  = df["net_debt_ebitda"].diff(1)
    df["ticker"] = ticker

    keep = [
        "ticker", "period", "quarter_end", "disclosure_date",
        "revenue", "net_profit", "roe", "net_debt_ebitda",
        "revenue_growth_qoq", "net_profit_growth_qoq",
        "revenue_growth_yoy", "net_profit_growth_yoy",
        "roe_change_qoq", "net_debt_ebitda_change_qoq",
    ]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------------
# 2. Post-disclosure abnormal returns
# ---------------------------------------------------------------------------

def calculate_post_disclosure_returns(
    ticker: str,
    horizons: List[int] = [1, 5, 20],
) -> pd.DataFrame:
    """
    Measure raw and abnormal returns following each KAP disclosure date.

    Abnormal return = ticker_return - benchmark_return over the same window.

    Two benchmarks are computed:
        xu100_abnormal_return_Nd  : excess over XU100 index
        sector_abnormal_return_Nd : excess over sector-peer median (or XU100
                                    fallback when no sector peers exist)

    Args:
        ticker:   Short ticker code.
        horizons: Forward windows in trading days.

    Returns:
        DataFrame with one row per disclosure date; columns include raw returns
        and both abnormal return series for each horizon.
    """
    pit    = _build_pit_fundamentals(ticker)
    close  = _load_price(ticker)
    xu100  = _load_xu100()

    # Sector benchmark
    bm_info = get_relative_benchmark(ticker)
    if bm_info["type"] == "sector":
        peer_closes = [_load_price(p) for p in bm_info["peers"]]
        bm_close = pd.concat(peer_closes, axis=1, sort=True).ffill().median(axis=1)
    else:
        bm_close = xu100.copy()

    rows = []
    for _, row in pit.iterrows():
        disc = row["disclosure_date"]
        rec  = {
            "ticker":         ticker,
            "period":         row["period"],
            "disclosure_date": disc,
            "benchmark_type": bm_info["type"],
        }
        for h in horizons:
            raw      = _forward_return(close,  disc, h)
            xu100_r  = _forward_return(xu100,  disc, h)
            bm_r     = _forward_return(bm_close, disc, h)

            rec[f"raw_return_{h}d"]             = raw
            rec[f"xu100_abnormal_return_{h}d"]  = (raw - xu100_r) if not np.isnan(raw) else np.nan
            rec[f"sector_abnormal_return_{h}d"] = (raw - bm_r)    if not np.isnan(raw) else np.nan

        rows.append(rec)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Combined reaction dataset
# ---------------------------------------------------------------------------

def build_fundamental_reaction_dataset(
    tickers: List[str] = COVERED_TICKERS,
) -> pd.DataFrame:
    """
    Join quarterly fundamental changes with post-disclosure return reactions
    for all covered tickers.

    Each row = one (ticker, quarter) observation.
    Columns combine fundamental change metrics with abnormal return series.
    """
    change_parts  = []
    return_parts  = []

    for ticker in tickers:
        try:
            chg = calculate_quarterly_changes(ticker)
            ret = calculate_post_disclosure_returns(ticker)
            change_parts.append(chg)
            return_parts.append(ret)
        except Exception as e:
            print(f"  [WARN] {ticker}: {e}")

    if not change_parts:
        return pd.DataFrame()

    changes = pd.concat(change_parts, ignore_index=True)
    returns = pd.concat(return_parts, ignore_index=True)

    # Merge on (ticker, period, disclosure_date)
    merge_keys = ["ticker", "period", "disclosure_date"]
    df = changes.merge(
        returns.drop(columns=["benchmark_type"], errors="ignore"),
        on=merge_keys,
        how="left",
    )

    return df.sort_values(["ticker", "period"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Reaction pattern summary
# ---------------------------------------------------------------------------

def summarize_reaction_patterns(df: pd.DataFrame) -> None:
    """
    Print a multi-part summary of earnings reaction patterns.

    (a) Positive vs. negative fundamental surprise: compare mean abnormal
        returns for net_profit_growth_qoq above vs. below the cross-ticker
        median at each horizon.

    (b) Weak-fundamental / strong-technical overlap: identify quarters where
        net_profit_growth_qoq is below median (weak fundamental surprise)
        and report how many observations have technical_score available.

    (c) Pearson correlation: net_profit_growth_qoq vs. xu100_abnormal_return_20d.
    """
    fwd_cols = [c for c in df.columns if c.startswith("xu100_abnormal_return_")]
    horizons = sorted({int(c.split("_")[-1].replace("d", "")) for c in fwd_cols})

    # ── (a) Surprise split ────────────────────────────────────────────
    valid = df.dropna(subset=["net_profit_growth_qoq"])
    median_surprise = valid["net_profit_growth_qoq"].median()
    pos_mask = valid["net_profit_growth_qoq"] >= median_surprise
    neg_mask = ~pos_mask

    print("=" * 65)
    print("(a) FUNDAMENTAL SURPRISE vs. ABNORMAL RETURN")
    print(f"    Median net_profit_growth_qoq across all obs: {median_surprise*100:.1f}%")
    print(f"    Positive surprise (≥ median): n={pos_mask.sum()}  |  "
          f"Negative surprise (< median): n={neg_mask.sum()}")
    print()
    print(f"  {'Horizon':>8}  {'Pos surprise AR':>16}  {'Neg surprise AR':>16}  {'Difference':>12}")
    print("  " + "-" * 57)
    for h in horizons:
        col = f"xu100_abnormal_return_{h}d"
        if col not in valid.columns:
            continue
        pos_mean = valid.loc[pos_mask, col].mean()
        neg_mean = valid.loc[neg_mask, col].mean()
        diff     = pos_mean - neg_mean
        print(f"  {h:>6}d  {pos_mean*100:>15.2f}%  {neg_mean*100:>15.2f}%  {diff*100:>+11.2f}%")

    # ── (b) Sector abnormal return split (same grouping) ─────────────
    sect_cols = [c for c in df.columns if c.startswith("sector_abnormal_return_")]
    if sect_cols:
        print()
        print("  Sector-relative abnormal return (same positive/negative split):")
        print(f"  {'Horizon':>8}  {'Pos surprise AR':>16}  {'Neg surprise AR':>16}  {'Difference':>12}")
        print("  " + "-" * 57)
        for h in horizons:
            col = f"sector_abnormal_return_{h}d"
            if col not in valid.columns:
                continue
            pos_mean = valid.loc[pos_mask, col].mean()
            neg_mean = valid.loc[neg_mask, col].mean()
            diff     = pos_mean - neg_mean
            print(f"  {h:>6}d  {pos_mean*100:>15.2f}%  {neg_mean*100:>15.2f}%  {diff*100:>+11.2f}%")

    # ── (c) Weak fundamental / strong technical overlap ───────────────
    print()
    print("=" * 65)
    print("(b) WEAK FUNDAMENTAL SURPRISE SUBSET")
    weak_fund = valid[neg_mask].copy()
    print(f"    n={len(weak_fund)} observations below median surprise")
    if "technical_score" in weak_fund.columns:
        strong_tech = weak_fund[weak_fund["technical_score"] >= 3]
        print(f"    With technical_score ≥ 3 (strong tech): n={len(strong_tech)}")
        print(f"    Mean xu100_abnormal_return_20d (strong tech + weak fund): "
              f"{strong_tech['xu100_abnormal_return_20d'].mean()*100:.2f}%")
    else:
        print("    technical_score not available in dataset — skipping overlap analysis.")
        print("    (Merge with technical score via indicators.py to enable this filter.)")

    # ── (d) Pearson correlation ────────────────────────────────────────
    print()
    print("=" * 65)
    print("(c) PEARSON CORRELATION: net_profit_growth_qoq vs. xu100_abnormal_return_20d")
    col20 = "xu100_abnormal_return_20d"
    if col20 in df.columns:
        both = df[["net_profit_growth_qoq", col20]].dropna()
        if len(both) >= 5:
            r, p = stats.pearsonr(both["net_profit_growth_qoq"], both[col20])
            print(f"    n={len(both)}  r={r:.4f}  p={p:.4f}  "
                  f"({'statistically significant at 5%' if p < 0.05 else 'not significant at 5%'})")
            print(f"    Interpretation: {'positive' if r > 0 else 'negative'} relationship — "
                  f"{'larger earnings surprises tend to produce larger abnormal returns' if r > 0 else 'larger earnings surprises do not predict higher abnormal returns in this sample'}")
        else:
            print(f"    Insufficient observations (n={len(both)}) for reliable correlation.")
    else:
        print("    xu100_abnormal_return_20d column not found.")

    # ── Per-ticker reaction summary ───────────────────────────────────
    print()
    print("=" * 65)
    print("PER-TICKER MEAN xu100_abnormal_return_20d (all quarters)")
    if col20 in df.columns:
        per_ticker = (
            df.groupby("ticker")[col20]
            .agg(["count", "mean"])
            .rename(columns={"count": "n", "mean": "mean_ar_20d"})
        )
        per_ticker["mean_ar_20d_%"] = (per_ticker["mean_ar_20d"] * 100).round(2)
        per_ticker = per_ticker.sort_values("mean_ar_20d", ascending=False)
        print(per_ticker[["n", "mean_ar_20d_%"]].to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 25)

    print(f"Building fundamental reaction dataset for: {COVERED_TICKERS}\n")
    df = build_fundamental_reaction_dataset()

    print(f"\nDataset shape : {df.shape}")
    print(f"Tickers       : {sorted(df['ticker'].unique())}")
    print(f"Quarters      : {sorted(df['period'].unique())}")

    print("\n=== Sample rows (first 5) ===")
    sample_cols = [
        "ticker", "period", "disclosure_date",
        "net_profit_growth_qoq", "net_profit_growth_yoy",
        "xu100_abnormal_return_1d", "xu100_abnormal_return_5d", "xu100_abnormal_return_20d",
    ]
    print(df[sample_cols].head(8).to_string(index=False))

    print("\n\n" + "=" * 65)
    print("REACTION PATTERN SUMMARY")
    print("=" * 65)
    summarize_reaction_patterns(df)

    # Save
    out_dir  = os.path.join(_SRC, "..", "data", "events")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fundamental_reaction.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {os.path.abspath(out_path)}")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
