"""
News and corporate events module for BIST100 MCP Trader.

Design decision: full-scale news/speech intelligence per the project brief
(licensed feeds, YouTube speech-to-text) was out of scope for this project's
time budget.  Instead, a small, manually curated set of real, dated, publicly
reported events (10 events across 3 tickers: ASELS, TOASO, BIMAS) was compiled
from public news sources, summarized in original wording (not quoted verbatim,
respecting copyright), and tagged with event categories per the brief's schema
(Section 16).  This demonstrates the intended architecture — timestamped event
metadata feeding into price-reaction analysis — at a scope appropriate for a
single-student project.  The speech-to-text pipeline (Section 16) was not
attempted.
"""

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

_SRC       = os.path.dirname(__file__)
_CACHE_DIR = os.path.join(_SRC, "..", "data", "universe_cache")

sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# 1. Curated news / corporate event list
# ---------------------------------------------------------------------------

NEWS_EVENTS: list[dict] = [
    # ── ASELS ─────────────────────────────────────────────────────────
    {
        "ticker":      "ASELS",
        "date":        "2024-12-25",
        "category":    "export_contract",
        "summary":     (
            "Signed a $58.6M export contract with a Turkic States Organization "
            "member country for electro-optic and communication systems "
            "(KAP disclosure)."
        ),
        "source_type": "company_disclosure",
    },
    {
        "ticker":      "ASELS",
        "date":        "2025-02-25",
        "category":    "earnings",
        "summary":     (
            "FY2024 results: record ~$1B in new export contracts signed, "
            "first-ever contracts with 7 new countries, first exports of "
            "31 distinct products."
        ),
        "source_type": "company_disclosure",
    },
    {
        "ticker":      "ASELS",
        "date":        "2025-08-06",
        "category":    "earnings",
        "summary":     (
            "H1 2025 results: $1.3B in export contracts signed; "
            "Net Debt/EBITDA improved sharply from 1.21x (H1 2024) "
            "to 0.57x (H1 2025)."
        ),
        "source_type": "company_disclosure",
    },
    {
        "ticker":      "ASELS",
        "date":        "2025-12-24",
        "category":    "export_contract",
        "summary":     (
            "Full-year 2025 new export contracts surpassed $2.01B (record), "
            "including a $410M electronic warfare systems deal with a "
            "European NATO member."
        ),
        "source_type": "news_aggregator",
    },
    # ── TOASO ─────────────────────────────────────────────────────────
    {
        "ticker":      "TOASO",
        "date":        "2025-02-13",
        "category":    "earnings",
        "summary":     (
            "FY2024 results: sales volume fell 33% YoY to 174K units; "
            "consolidated net profit collapsed 76% to 5.22B TRY from "
            "21.78B TRY in 2023, driven by weak domestic/export volumes "
            "and the end of Fiorino production."
        ),
        "source_type": "company_disclosure",
    },
    {
        "ticker":      "TOASO",
        "date":        "2025-04-18",
        "category":    "corporate_action",
        "summary":     (
            "KAP disclosure: negotiations opened with Stellantis for a new "
            "CKD light-vehicle export project; investment up to EUR 256M, "
            "targeting ~150K units/year capacity by 2026."
        ),
        "source_type": "company_disclosure",
    },
    {
        "ticker":      "TOASO",
        "date":        "2025-04-29",
        "category":    "financial_result",
        "summary":     (
            "2025Q1 results disclosed a net loss of approximately 197M TRY "
            "amid weak domestic and export demand."
        ),
        "source_type": "company_disclosure",
    },
    # ── BIMAS ─────────────────────────────────────────────────────────
    {
        "ticker":      "BIMAS",
        "date":        "2025-02-19",
        "category":    "regulatory",
        "summary":     (
            "Turkish Competition Authority fined BIM approximately 1.3B TRY "
            "for obstructing an on-site inspection (an employee deleted "
            "WhatsApp messages during a raid); the stock fell about 4.2% "
            "on the news."
        ),
        "source_type": "news_aggregator",
    },
    {
        "ticker":      "BIMAS",
        "date":        "2025-03-10",
        "category":    "earnings",
        "summary":     (
            "FY2024 results: 18.6B TRY net profit, 206 new stores opened in "
            "Q4 2024 (total 13,583); disclosed early discounted fine payment "
            "of 971.9M TRY while appealing, cash impact expected in Q1 2025; "
            "a new Competition Authority investigation was opened the same week."
        ),
        "source_type": "company_disclosure",
    },
    {
        "ticker":      "BIMAS",
        "date":        "2025-08-15",
        "category":    "corporate_action",
        "summary":     (
            "Q2 2025 results: 266 new stores opened (total 14,075, +7% YoY); "
            "the FILE brand was formally spun off into a separate "
            "subsidiary structure."
        ),
        "source_type": "company_disclosure",
    },
]


# ---------------------------------------------------------------------------
# 2. get_news_events
# ---------------------------------------------------------------------------

def get_news_events(
    ticker:     str,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> list[dict]:
    """
    Return curated news events for ``ticker``, optionally filtered by date.

    Args:
        ticker:     Short BIST ticker code (e.g. 'ASELS').
        start_date: Inclusive lower bound in 'YYYY-MM-DD' format (or None).
        end_date:   Inclusive upper bound in 'YYYY-MM-DD' format (or None).

    Returns:
        List of event dicts from NEWS_EVENTS matching the criteria.
    """
    ticker = ticker.upper()
    events = [e for e in NEWS_EVENTS if e["ticker"] == ticker]

    if start_date:
        events = [e for e in events if e["date"] >= start_date]
    if end_date:
        events = [e for e in events if e["date"] <= end_date]

    return sorted(events, key=lambda e: e["date"])


# ---------------------------------------------------------------------------
# 3. calculate_post_news_returns
# ---------------------------------------------------------------------------

def _forward_return(close: pd.Series, date: pd.Timestamp, horizon: int) -> float:
    """(close[t+horizon] - close[t]) / close[t], or NaN if out of range."""
    idx = close.index
    pos = idx.get_indexer([date], method="ffill")[0]
    if pos < 0 or pos + horizon >= len(idx):
        return np.nan
    return float((close.iloc[pos + horizon] - close.iloc[pos]) / close.iloc[pos])


def calculate_post_news_returns(
    price_df: pd.DataFrame,
    ticker:   str,
    horizons: list[int] = [1, 5, 20],
) -> pd.DataFrame:
    """
    Measure raw forward returns following each curated news event for ``ticker``.

    For each event the function locates the closest trading day on or after
    the event date (forward-fill look-up) and computes the price return over
    each horizon window.  This reuses the same forward-return logic as
    fundamental_reaction.py.

    Args:
        price_df: OHLCV DataFrame for the ticker with a DatetimeIndex.
        ticker:   Short BIST ticker code.
        horizons: Forward windows in trading days (default [1, 5, 20]).

    Returns:
        DataFrame with one row per event and columns:
            ticker, date, category, summary, source_type,
            fwd_return_1d, fwd_return_5d, fwd_return_20d
        Monetary forward returns are expressed as decimals (e.g. 0.042 = +4.2%).
    """
    events = get_news_events(ticker)
    close  = price_df["Close"]

    rows = []
    for ev in events:
        ev_date = pd.Timestamp(ev["date"])
        rec = {
            "ticker":      ev["ticker"],
            "date":        ev["date"],
            "category":    ev["category"],
            "summary":     ev["summary"],
            "source_type": ev["source_type"],
        }
        for h in horizons:
            rec[f"fwd_return_{h}d"] = _forward_return(close, ev_date, h)
        rows.append(rec)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 70)

    SEP = "=" * 80

    TICKERS  = ["ASELS", "TOASO", "BIMAS"]
    HORIZONS = [1, 5, 20]

    all_rows = []

    for ticker in TICKERS:
        csv_path = os.path.join(_CACHE_DIR, f"{ticker}.csv")
        price_df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)

        result = calculate_post_news_returns(price_df, ticker, horizons=HORIZONS)
        all_rows.append(result)

        print(SEP)
        print(f"POST-NEWS FORWARD RETURNS — {ticker}  ({len(result)} events)")
        print(SEP)

        display_cols = ["date", "category", "fwd_return_1d", "fwd_return_5d", "fwd_return_20d"]
        fmt = result[display_cols].copy()
        for col in [f"fwd_return_{h}d" for h in HORIZONS]:
            fmt[col] = fmt[col].apply(
                lambda x: f"{x*100:+.2f}%" if not np.isnan(x) else "  n/a  "
            )
        print(fmt.to_string(index=False))

        # Summary stats per category
        print(f"\n  Mean returns by category:")
        for cat, grp in result.groupby("category"):
            means = {f"{h}d": grp[f"fwd_return_{h}d"].mean() for h in HORIZONS}
            parts = "  ".join(
                f"{h}: {v*100:+.2f}%" if not np.isnan(v) else f"{h}: n/a"
                for h, v in means.items()
            )
            print(f"    {cat:<20} {parts}")
        print()

    # ── BIMAS regulatory event spotlight ─────────────────────────────
    print(SEP)
    print("SPOTLIGHT — BIMAS competition fine (2025-02-19)")
    print(SEP)
    bimas_csv  = os.path.join(_CACHE_DIR, "BIMAS.csv")
    bimas_df   = pd.read_csv(bimas_csv, index_col="Date", parse_dates=True)
    bimas_news = calculate_post_news_returns(bimas_df, "BIMAS")
    fine_row   = bimas_news[bimas_news["date"] == "2025-02-19"].iloc[0]
    print(f"  Event    : {fine_row['summary'][:80]}…")
    for h in HORIZONS:
        val = fine_row[f"fwd_return_{h}d"]
        tag = "NEGATIVE ✓" if val < 0 else "positive"
        print(f"  fwd_{h:>2}d  : {val*100:+.2f}%  ({tag})")
    print(
        "\n  Note: a negative short-term return is consistent with the "
        "~4.2% day-of decline\n  mentioned in the event summary and with "
        "BIM's weak 2025Q1 performance observed\n  in the fundamental "
        "reaction analysis (Phase 4)."
    )

    # ── Combined output ───────────────────────────────────────────────
    combined = pd.concat(all_rows, ignore_index=True)
    print(f"\n{SEP}")
    print(f"COMBINED: {len(combined)} events across {len(TICKERS)} tickers")
    print(SEP)
    print(combined[["ticker", "date", "category",
                     "fwd_return_1d", "fwd_return_5d", "fwd_return_20d"]].to_string(index=False))
