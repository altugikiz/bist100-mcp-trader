"""
MCP server for BIST100 MCP Trader.

Exposes six tools that wrap existing analysis modules so that an LLM agent
(Claude, GPT-4o, etc.) can query live price data, technical indicators,
support/resistance levels, fundamental metrics, and backtest results without
writing any code itself.

Run with stdio transport (default MCP mode):
    python src/mcp_server.py

The server communicates over stdin/stdout using the MCP JSON-RPC protocol.
No network port is opened; Claude Desktop or another MCP host manages the
process lifecycle.
"""

import os
import sys
import json
import math

# Ensure sibling modules are importable when launched from any working directory
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from data_collection import fetch_ohlcv
from indicators import (
    calculate_sma, calculate_rsi, calculate_macd, calculate_technical_score,
)
from support_resistance import find_support_resistance
from fundamental_analysis import (
    load_fundamentals, calculate_derived_ratios,
    calculate_market_multiples, calculate_yoy_growth,
    calculate_fundamental_score as _calc_fund_score,
)
from strategy import build_full_pipeline, generate_signal
from backtest import run_backtest, calculate_performance


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TICKER   = "ASELS.IS"
DEFAULT_START    = "2023-01-01"
DEFAULT_END      = "2026-01-01"
CURRENT_PRICE    = 372.50
SHARES_OUTSTANDING = 4_560_000_000
SECTOR_AVG_PE    = 65.6

_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fintables_asels.csv")


# ---------------------------------------------------------------------------
# Helper: make any dict JSON-safe (convert numpy/nan/inf to Python natives)
# ---------------------------------------------------------------------------

def _sanitise(obj):
    """Recursively convert numpy types, NaN, and Inf to JSON-serialisable forms."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="bist100-trader",
    instructions=(
        "Use these tools to analyse ASELS.IS (Aselsan) stock. "
        "Start with get_stock_history to see recent prices, then "
        "calculate_indicators for trend signals, find_support_resistance for "
        "key price levels, get_fundamental_metrics / calculate_fundamental_score "
        "for valuation, and run_backtest_summary for overall strategy performance."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1 — Historical OHLCV data
# ---------------------------------------------------------------------------

@mcp.tool()
def get_stock_history(
    symbol: str = DEFAULT_TICKER,
    start: str = "2025-01-01",
    end: str = "2026-01-01",
    last_n: int = 30,
) -> dict:
    """
    Fetch daily OHLCV (Open, High, Low, Close, Volume) price history for a
    stock from Yahoo Finance and return the most recent `last_n` rows.

    Use this tool when the user asks about recent price action, historical
    prices, or wants to know the current price of a stock.

    Args:
        symbol: Yahoo Finance ticker symbol (e.g. 'ASELS.IS', 'GARAN.IS').
        start:  Start date in YYYY-MM-DD format (inclusive).
        end:    End date in YYYY-MM-DD format (exclusive).
        last_n: Number of most recent rows to return (default 30).

    Returns:
        dict with keys 'symbol', 'rows' (list of OHLCV dicts), 'total_days'.
    """
    df = fetch_ohlcv(symbol, start, end)
    subset = df.tail(last_n)
    rows = []
    for date, row in subset.iterrows():
        rows.append({
            "date":   date.strftime("%Y-%m-%d"),
            "open":   round(float(row["Open"]), 4),
            "high":   round(float(row["High"]), 4),
            "low":    round(float(row["Low"]),  4),
            "close":  round(float(row["Close"]), 4),
            "volume": int(row["Volume"]),
        })
    return _sanitise({"symbol": symbol, "total_days": len(df), "rows": rows})


# ---------------------------------------------------------------------------
# Tool 2 — Technical indicators (latest values)
# ---------------------------------------------------------------------------

@mcp.tool()
def calculate_indicators(
    symbol: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> dict:
    """
    Compute SMA20, SMA50, RSI14, MACD line, MACD Signal, and MACD Histogram
    for a stock and return the most recent values.

    Use this tool when the user asks about trend direction, momentum, moving
    averages, RSI overbought/oversold status, or MACD crossovers.

    Returns:
        dict with the latest values of each indicator and a brief
        interpretation (e.g. 'SMA20 > SMA50 → uptrend').
    """
    df = fetch_ohlcv(symbol, start, end)
    df["SMA20"] = calculate_sma(df, 20)
    df["SMA50"] = calculate_sma(df, 50)
    df["RSI14"] = calculate_rsi(df, 14)
    df[["MACD", "MACD_Signal", "MACD_Hist"]] = calculate_macd(df)

    last = df.iloc[-1]
    date = df.index[-1].strftime("%Y-%m-%d")

    return _sanitise({
        "symbol":       symbol,
        "as_of":        date,
        "close":        round(float(last["Close"]), 4),
        "SMA20":        round(float(last["SMA20"]), 4),
        "SMA50":        round(float(last["SMA50"]), 4),
        "RSI14":        round(float(last["RSI14"]), 2),
        "MACD":         round(float(last["MACD"]), 4),
        "MACD_Signal":  round(float(last["MACD_Signal"]), 4),
        "MACD_Hist":    round(float(last["MACD_Hist"]), 4),
        "signals": {
            "trend":   "uptrend"   if last["SMA20"] > last["SMA50"] else "downtrend",
            "price_vs_sma20": "above" if last["Close"] > last["SMA20"] else "below",
            "rsi_zone": (
                "overbought" if last["RSI14"] > 70
                else "oversold" if last["RSI14"] < 30
                else "neutral"
            ),
            "macd_crossover": "bullish" if last["MACD"] > last["MACD_Signal"] else "bearish",
        },
    })


# ---------------------------------------------------------------------------
# Tool 3 — Support & resistance levels
# ---------------------------------------------------------------------------

@mcp.tool()
def find_support_resistance_levels(
    symbol: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    order: int = 5,
    nearest_n: int = 5,
) -> dict:
    """
    Detect pivot high (resistance) and pivot low (support) price levels using
    scipy.signal.argrelextrema and return the `nearest_n` levels closest to
    the current price.

    Use this tool when the user asks about support levels, resistance levels,
    key price zones, or where the stock might find buying/selling pressure.

    Args:
        order:    Number of candles on each side to confirm a pivot (higher →
                  fewer but more significant levels).
        nearest_n: How many support/resistance levels to return on each side.

    Returns:
        dict with 'current_price', 'resistance' (list), 'support' (list),
        each level including price and date of the pivot.
    """
    df = fetch_ohlcv(symbol, start, end)
    pivot_highs, pivot_lows = find_support_resistance(df, order=order)

    current = float(df["Close"].iloc[-1])

    def _fmt(pivot_df: pd.DataFrame, n: int) -> list:
        return [
            {"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 4)}
            for d, p in zip(pivot_df.index, pivot_df["Close"])
        ][-n:]

    # Nearest resistance: pivots above current price, sorted ascending (closest first)
    res = pivot_highs[pivot_highs["Close"] > current].sort_values("Close")
    # Nearest support: pivots below current price, sorted descending (closest first)
    sup = pivot_lows[pivot_lows["Close"] < current].sort_values("Close", ascending=False)

    return _sanitise({
        "symbol":        symbol,
        "current_price": round(current, 4),
        "resistance":    _fmt(res, nearest_n),
        "support":       _fmt(sup, nearest_n),
        "total_pivots":  {"highs": len(pivot_highs), "lows": len(pivot_lows)},
    })


# ---------------------------------------------------------------------------
# Tool 4 — Fundamental metrics for a quarter
# ---------------------------------------------------------------------------

@mcp.tool()
def get_fundamental_metrics(period: str = "2025Q4") -> dict:
    """
    Return all fundamental financial metrics for a given quarter from the
    local Fintables CSV (data/fintables_asels.csv).

    Use this tool when the user asks about revenue, profit, margins, ROE, ROA,
    debt ratios, cash flow, or any balance-sheet metric for a specific quarter.

    Args:
        period: Quarter string in 'YYYYQn' format, e.g. '2025Q4', '2024Q2'.
                Available periods: 2024Q1 through 2025Q4.

    Returns:
        dict with all metric columns for the requested period.
        Monetary values are in million TRY; margins/returns are decimals;
        multiples are raw ratio values.
    """
    df = load_fundamentals(_CSV_PATH)
    df = calculate_derived_ratios(df)
    df = calculate_market_multiples(df, CURRENT_PRICE, SHARES_OUTSTANDING)

    if period not in df.index:
        available = list(df.index)
        return {"error": f"Period '{period}' not found. Available: {available}"}

    row = df.loc[period].to_dict()
    row["period"] = period
    return _sanitise(row)


# ---------------------------------------------------------------------------
# Tool 5 — Fundamental score for a quarter
# ---------------------------------------------------------------------------

@mcp.tool()
def get_fundamental_score(period: str = "2025Q4") -> dict:
    """
    Compute the fundamental quality score (0-5) for a given quarter and return
    the score together with a True/False breakdown of each criterion.

    Criteria scored:
      +1 if YoY revenue growth > 15%
      +1 if YoY net profit growth > 15%
      +1 if TTM ROE > 20%
      +1 if Net Debt / TTM EBITDA < 2x
      +1 if PE ratio < sector average PE (65.6x for BIST defence sector)

    Use this tool when the user wants to know whether ASELS passes fundamental
    quality thresholds in a specific quarter, or to compare quarters.

    Args:
        period: Quarter string, e.g. '2025Q4'.

    Returns:
        dict with 'fundamental_score' (int) and one bool per criterion.
    """
    df = load_fundamentals(_CSV_PATH)
    df = calculate_derived_ratios(df)
    df = calculate_market_multiples(df, CURRENT_PRICE, SHARES_OUTSTANDING)
    df = calculate_yoy_growth(df)
    df = _calc_fund_score(df, sector_avg_pe=SECTOR_AVG_PE)

    if period not in df.index:
        return {"error": f"Period '{period}' not found. Available: {list(df.index)}"}

    row = df.loc[period]
    return _sanitise({
        "period":                period,
        "fundamental_score":     int(row["fundamental_score"]),
        "crit_revenue_growth":   bool(row["crit_revenue_growth"]),
        "crit_net_profit_growth":bool(row["crit_net_profit_growth"]),
        "crit_roe":              bool(row["crit_roe"]),
        "crit_net_debt_ebitda":  bool(row["crit_net_debt_ebitda"]),
        "crit_pe":               bool(row["crit_pe"]),
        "revenue_growth_yoy":    row["revenue_growth_yoy"],
        "net_profit_growth_yoy": row["net_profit_growth_yoy"],
        "roe":                   row["roe"],
        "net_debt_ebitda":       row["net_debt_ebitda"],
        "pe_ratio":              row["pe_ratio"],
        "sector_avg_pe":         SECTOR_AVG_PE,
    })


# ---------------------------------------------------------------------------
# Tool 6 — Backtest performance summary
# ---------------------------------------------------------------------------

@mcp.tool()
def run_backtest_summary(
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    buy_threshold: float = 3.0,
    sell_threshold: float = 1.5,
    tech_weight: float = 0.50,
    fund_weight: float = 0.50,
) -> dict:
    """
    Run the full strategy pipeline (technical + fundamental scoring → signal
    generation → backtest) and return key performance metrics.

    Use this tool when the user asks how the trading strategy performed, what
    the Sharpe ratio or drawdown is, how it compares to buy-and-hold, or wants
    to evaluate a specific parameter combination.

    Args:
        start / end:     Backtest date range.
        buy_threshold:   Total_Score at or above which a BUY signal is issued.
        sell_threshold:  Total_Score at or below which a SELL signal is issued.
        tech_weight:     Weight for technical_score in Total_Score formula.
        fund_weight:     Weight for fundamental_score in Total_Score formula.

    Returns:
        dict with cumulative_return, sharpe_ratio, max_drawdown, win_rate,
        days_invested, signal_distribution, and parameter echo.
    """
    df = build_full_pipeline(
        start=start, end=end,
        current_price=CURRENT_PRICE,
        shares_outstanding=SHARES_OUTSTANDING,
        sector_avg_pe=SECTOR_AVG_PE,
    )

    df["Total_Score"] = tech_weight * df["technical_score"] + fund_weight * df["fundamental_score"]
    df = generate_signal(df, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    bt = run_backtest(df)
    m  = calculate_performance(bt)

    dist = bt["Signal"].value_counts().to_dict()

    return _sanitise({
        "parameters": {
            "start": start, "end": end,
            "buy_threshold": buy_threshold, "sell_threshold": sell_threshold,
            "tech_weight": tech_weight, "fund_weight": fund_weight,
        },
        "cumulative_return_pct": round(m["cumulative_return"] * 100, 2),
        "sharpe_ratio":          round(m["sharpe_ratio"], 4),
        "max_drawdown_pct":      round(m["max_drawdown"] * 100, 2),
        "win_rate_pct":          round(m["win_rate"] * 100, 2),
        "days_invested":         m["days_invested"],
        "trading_days":          m["trading_days"],
        "signal_distribution":   dist,
    })


# ---------------------------------------------------------------------------
# Entry point — stdio transport (MCP standard)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
