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

import io
import os
import sys
import json
import math
import contextlib

# Ensure sibling modules are importable when launched from any working directory
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from data_collection import fetch_ohlcv
from indicators import (
    calculate_sma, calculate_rsi, calculate_macd, calculate_technical_score,
    calculate_bollinger_bands, calculate_kama, calculate_supertrend, calculate_ichimoku,
)
from support_resistance import find_support_resistance
from fundamental_analysis import (
    load_fundamentals, calculate_derived_ratios,
    calculate_market_multiples, calculate_yoy_growth,
    calculate_fundamental_score as _calc_fund_score,
)
from strategy import build_full_pipeline, generate_signal
from backtest import run_backtest, calculate_performance
from config import (
    SYMBOL as DEFAULT_TICKER,
    START_DATE as DEFAULT_START,
    END_DATE as DEFAULT_END,
    BACKTEST_START, BACKTEST_END,
    CURRENT_PRICE, SHARES_OUTSTANDING, SECTOR_AVG_PE,
    BUY_THRESHOLD, SELL_THRESHOLD, TECH_WEIGHT, FUND_WEIGHT,
    FUNDAMENTAL_CSV_PATH as _CSV_PATH,
)
from universe import UNIVERSE
from sector_analysis import rank_universe_by_sector
from reversal_events import build_event_log
from fundamental_reaction import calculate_quarterly_changes
from point_in_time import build_point_in_time_fundamentals
from evidence import build_evidence_bundle, RiskGate
from action_space import determine_action
from decision_log import log_decision, add_human_review, load_decision_log, replay_analysis
from harness import AnalysisSession, advance_state, record_tool_call

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "universe_cache")
_LOG_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "decision_log")

# In-memory session store: session_id -> {"session": AnalysisSession, ...}
_SESSIONS: dict = {}


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
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
    tech_weight: float = TECH_WEIGHT,
    fund_weight: float = FUND_WEIGHT,
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
# Private helpers for the harness tools
# ---------------------------------------------------------------------------

def _load_price_df(ticker: str) -> pd.DataFrame:
    """Read cached OHLCV CSV for ticker; fall back to live fetch on miss."""
    path = os.path.join(_CACHE_DIR, f"{ticker}.csv")
    if os.path.isfile(path):
        return pd.read_csv(path, index_col="Date", parse_dates=True)
    return fetch_ohlcv(f"{ticker}.IS", "2023-01-02", "2026-01-01")


def _build_price_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator columns to a price DataFrame in place."""
    df["SMA20"] = calculate_sma(df, 20)
    df["SMA50"] = calculate_sma(df, 50)
    df["RSI14"] = calculate_rsi(df, 14)
    df[["MACD", "MACD_Signal", "MACD_Hist"]] = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_kama(df)
    df = calculate_supertrend(df)
    df = calculate_ichimoku(df)
    _, pivot_lows = find_support_resistance(df, order=5)
    df = calculate_technical_score(df, pivot_lows)
    return df, pivot_lows


def _find_safe_quarter(ticker: str, price_end: pd.Timestamp) -> str | None:
    """
    Return the most recent disclosed quarter whose KAP disclosure_date
    is on or before price_end.  Returns None when no safe quarter exists.
    """
    try:
        raw  = load_fundamentals(ticker).reset_index()
        pit  = build_point_in_time_fundamentals(ticker, raw)
        safe = pit[pd.to_datetime(pit["disclosure_date"]) <= price_end]
        return str(safe["period"].iloc[-1]) if not safe.empty else None
    except Exception:
        return None


def _build_fundamental_result(ticker: str, safe_quarter: str) -> dict:
    """Compute fundamental score and QoQ changes for safe_quarter."""
    raw       = load_fundamentals(ticker)
    fund_full = (
        raw
        .pipe(lambda d: calculate_derived_ratios(d.copy()))
        .pipe(calculate_yoy_growth)
        .pipe(_calc_fund_score, sector_avg_pe=SECTOR_AVG_PE)
    )
    chg_df   = calculate_quarterly_changes(ticker).set_index("period")
    q_row    = fund_full.loc[safe_quarter]

    # Previous quarter for _turned_positive logic in action_space
    periods = list(fund_full.index)
    idx     = periods.index(safe_quarter)
    prev_q  = periods[idx - 1] if idx > 0 else None

    prev_np  = float(chg_df.loc[prev_q, "net_profit_growth_qoq"]) if prev_q and prev_q in chg_df.index else None
    prev_rev = float(chg_df.loc[prev_q, "revenue_growth_qoq"])    if prev_q and prev_q in chg_df.index else None

    return {
        "net_profit_growth_qoq":      float(chg_df.loc[safe_quarter, "net_profit_growth_qoq"]) if safe_quarter in chg_df.index else None,
        "revenue_growth_qoq":         float(chg_df.loc[safe_quarter, "revenue_growth_qoq"])    if safe_quarter in chg_df.index else None,
        "prev_net_profit_growth_qoq": prev_np,
        "prev_revenue_growth_qoq":    prev_rev,
        "roe":                        float(q_row["roe"])            if not pd.isna(q_row["roe"])            else None,
        "net_debt_ebitda":            float(q_row["net_debt_ebitda"]) if not pd.isna(q_row["net_debt_ebitda"]) else None,
        "fundamental_score":          int(q_row["fundamental_score"]),
    }


def _build_macro_snapshot(snap_date: pd.Timestamp) -> dict:
    """Read daily macro CSV and return a snapshot dict for snap_date."""
    macro_path = os.path.join(os.path.dirname(__file__), "..", "data", "macro", "macro_daily.csv")
    macro  = pd.read_csv(macro_path, index_col="Date", parse_dates=True)
    snap   = macro.loc[snap_date]
    snap30 = macro.loc[snap_date - pd.Timedelta(days=30)]
    return {
        "usdtry":         float(snap["usdtry"]),
        "eurtry":         float(snap["eurtry"]),
        "tcmb_rate":      float(snap["tcmb_rate"]),
        "tufe_yoy":       float(snap["tufe_yoy"]),
        "fed_rate":       float(snap["fed_rate"]),
        "usdtry_30d_ago": float(snap30["usdtry"]),
        "as_of":          str(snap_date.date()),
    }


def _observed_context_summary(bundle) -> dict:
    """Return a compact, token-friendly summary of observed_context."""
    tech  = bundle.observed_context.get("technical", {})
    sect  = bundle.observed_context.get("sector", {})
    macro = bundle.observed_context.get("macro", {})
    fund  = bundle.observed_context.get("fundamental", {})
    evts  = bundle.observed_context.get("events", {})
    top_events = {
        k: {"bounce_rate_5d": round(v.get("bounce_rate_5d", 0) * 100, 1), "n": v.get("n")}
        for k, v in evts.items()
        if isinstance(v, dict) and v.get("n", 0) >= 5
    }
    return _sanitise({
        "technical_score":      tech.get("technical_score"),
        "KAMA_Slope":           tech.get("KAMA_Slope"),
        "Supertrend_Direction": tech.get("Supertrend_Direction"),
        "RSI14":                round(tech.get("RSI14", 0), 1),
        "sector_group":         sect.get("sector_group"),
        "benchmark_type":       sect.get("benchmark_type"),
        "lag_score_20d_pct":    round(float(sect.get("lag_score_20d", 0)) * 100, 1) if sect.get("lag_score_20d") is not None else None,
        "fundamental_score":    fund.get("fundamental_score"),
        "tcmb_rate":            macro.get("tcmb_rate"),
        "usdtry":               macro.get("usdtry"),
        "top_reversal_events":  top_events,
    })


# ---------------------------------------------------------------------------
# Tool 7 — Start analysis session
# ---------------------------------------------------------------------------

@mcp.tool()
def start_analysis_session(ticker: str) -> dict:
    """
    Create a new analysis session for the given ticker and advance the harness
    through its initial states: START → SELECT_UNIVERSE → LOAD_MARKET_DATA →
    CHECK_DATA_QUALITY.

    CALL THIS TOOL FIRST before any other analysis tool in the harness sequence.
    Do NOT skip this step — subsequent tools (run_full_analysis, get_recommended_action,
    submit_human_review, get_session_replay) all require the session_id returned here.

    The harness enforces a strict ordering: tools called in the wrong state will
    be rejected.  Always call tools in this sequence:
        1. start_analysis_session
        2. run_full_analysis
        3. get_recommended_action
        4. submit_human_review
        5. get_session_replay

    Args:
        ticker: Short BIST ticker code, e.g. 'ASELS', 'TOASO', 'AKBNK'.

    Returns:
        {"session_id": str, "current_state": str, "ticker": str,
         "price_rows": int, "price_nan_pct": float}
    """
    ticker = ticker.upper()

    session = AnalysisSession(ticker)

    # SELECT_UNIVERSE
    advance_state(session, "SELECT_UNIVERSE")
    in_universe = (UNIVERSE["ticker"] == ticker).any()
    record_tool_call(session, "has_sector_peers",
                     f"ticker={ticker}",
                     f"in_universe={in_universe}")

    # LOAD_MARKET_DATA
    advance_state(session, "LOAD_MARKET_DATA")
    df = _load_price_df(ticker)
    record_tool_call(session, "get_stock_history",
                     f"ticker={ticker}.IS  rows={len(df)}",
                     f"start={df.index[0].date()}  end={df.index[-1].date()}")

    # CHECK_DATA_QUALITY
    advance_state(session, "CHECK_DATA_QUALITY")
    nan_pct = float(df["Close"].isna().mean() * 100)
    record_tool_call(session, "check_data_quality",
                     f"ticker={ticker}",
                     f"Close_NaN={nan_pct:.1f}%  rows={len(df)}")

    _SESSIONS[session.session_id] = {
        "session":   session,
        "price_df":  df,
        "ticker":    ticker,
    }

    return _sanitise({
        "session_id":    session.session_id,
        "current_state": session.current_state,
        "ticker":        ticker,
        "price_rows":    len(df),
        "price_nan_pct": round(nan_pct, 2),
    })


# ---------------------------------------------------------------------------
# Tool 8 — Run full analysis
# ---------------------------------------------------------------------------

@mcp.tool()
def run_full_analysis(session_id: str) -> dict:
    """
    Advance the session from CHECK_DATA_QUALITY to BUILD_EVIDENCE, running
    every analysis phase in the correct order:

        CALCULATE_TECHNICALS → RUN_SECTOR_RELATIVE_ANALYSIS →
        RUN_PATTERN_AND_REVERSAL_ANALYSIS → LOAD_FUNDAMENTALS →
        LOAD_MACRO_CONTEXT → LOAD_NEWS_AND_SPEECH_CONTEXT → BUILD_EVIDENCE

    CALL THIS TOOL SECOND — only after start_analysis_session has returned a
    session_id.  Do not call get_recommended_action before this tool completes.

    The safe fundamental quarter is chosen automatically: the most recent
    quarter whose KAP disclosure date falls on or before the price data end
    date, preventing look-ahead bias.

    Args:
        session_id: The session_id returned by start_analysis_session.

    Returns:
        {"committed_evidence": [...], "gate_result": {...},
         "observed_context_summary": {...}, "safe_quarter": str}
    """
    if session_id not in _SESSIONS:
        return {"error": f"session_id '{session_id}' not found. Call start_analysis_session first."}

    store   = _SESSIONS[session_id]
    session = store["session"]
    df_raw  = store["price_df"]
    ticker  = store["ticker"]
    snap    = df_raw.index[-1]   # most recent trading day in cache

    # ── CALCULATE_TECHNICALS ───────────────────────────────────────────
    advance_state(session, "CALCULATE_TECHNICALS")
    df, _ = _build_price_with_indicators(df_raw.copy())
    last = df.iloc[-1]
    technical_result = {
        "technical_score":      int(last["technical_score"]),
        "KAMA_Slope":           str(last["KAMA_Slope"]),
        "Supertrend_Direction": str(last["Supertrend_Direction"]),
        "Ichimoku_State":       None if (isinstance(last["Ichimoku_State"], float) and math.isnan(last["Ichimoku_State"])) else str(last["Ichimoku_State"]),
        "RSI14":                float(last["RSI14"]),
    }
    record_tool_call(session, "calculate_technical_score",
                     f"ticker={ticker} as_of={snap.date()}",
                     f"score={technical_result['technical_score']}/5  KAMA={technical_result['KAMA_Slope']}")

    # ── RUN_SECTOR_RELATIVE_ANALYSIS ──────────────────────────────────
    advance_state(session, "RUN_SECTOR_RELATIVE_ANALYSIS")
    rank_df = rank_universe_by_sector(snap, windows=[5, 10, 20])
    s_row   = rank_df[rank_df["ticker"] == ticker]
    if s_row.empty:
        sector_result = {"sector_group": "unknown", "benchmark_type": "xu100_fallback",
                         "lag_score_5d": None, "lag_score_10d": None, "lag_score_20d": None,
                         "intra_group_rank": None, "potential_catchup_flag": False}
    else:
        s_row = s_row.iloc[0]
        sector_result = {k: s_row[k] for k in ["sector_group", "benchmark_type",
                                                 "lag_score_5d", "lag_score_10d",
                                                 "lag_score_20d", "intra_group_rank"]}
        lag20 = float(s_row["lag_score_20d"]) if s_row["lag_score_20d"] is not None else 0
        sector_result["potential_catchup_flag"] = bool(lag20 < -0.05)
    record_tool_call(session, "rank_universe_by_sector",
                     f"date={snap.date()}",
                     f"lag_20d={sector_result.get('lag_score_20d', 0) or 0:.3f}  catchup={sector_result['potential_catchup_flag']}")

    # ── RUN_PATTERN_AND_REVERSAL_ANALYSIS ─────────────────────────────
    advance_state(session, "RUN_PATTERN_AND_REVERSAL_ANALYSIS")
    log_ev = build_event_log(df, ticker)
    event_result = {}
    for etype, grp in log_ev.groupby("event_type"):
        fwd5       = grp["forward_return_5d"].dropna()
        dates      = pd.to_datetime(grp["event_date"])
        days_since = int((snap - dates.max()).days) if len(dates) else None
        event_result[etype] = {
            "n":               len(grp),
            "bounce_rate_5d":  float((fwd5 > 0).mean()) if len(fwd5) else None,
            "avg_fwd_5d":      float(fwd5.mean())        if len(fwd5) else None,
            "days_since_last": days_since,
        }
    record_tool_call(session, "build_event_log",
                     f"ticker={ticker}",
                     f"event_types={len(event_result)}")

    # ── LOAD_FUNDAMENTALS ─────────────────────────────────────────────
    advance_state(session, "LOAD_FUNDAMENTALS")
    safe_q = _find_safe_quarter(ticker, snap)
    if safe_q:
        fundamental_result = _build_fundamental_result(ticker, safe_q)
    else:
        fundamental_result = {"fundamental_score": 0, "net_profit_growth_qoq": None,
                              "revenue_growth_qoq": None, "prev_net_profit_growth_qoq": None,
                              "prev_revenue_growth_qoq": None, "roe": None, "net_debt_ebitda": None}
    record_tool_call(session, "calculate_fundamental_score",
                     f"ticker={ticker} safe_quarter={safe_q}",
                     f"score={fundamental_result.get('fundamental_score', 0)}/5")

    # ── LOAD_MACRO_CONTEXT ────────────────────────────────────────────
    advance_state(session, "LOAD_MACRO_CONTEXT")
    macro_snapshot = _build_macro_snapshot(snap)
    record_tool_call(session, "build_daily_macro_series",
                     f"end={snap.date()}",
                     f"tcmb={macro_snapshot['tcmb_rate']}%  usdtry={macro_snapshot['usdtry']:.2f}")

    # ── LOAD_NEWS_AND_SPEECH_CONTEXT (no-op Phase 6) ──────────────────
    advance_state(session, "LOAD_NEWS_AND_SPEECH_CONTEXT")

    # ── BUILD_EVIDENCE ────────────────────────────────────────────────
    advance_state(session, "BUILD_EVIDENCE")
    bundle     = build_evidence_bundle(ticker, technical_result, fundamental_result,
                                       sector_result, event_result, macro_snapshot)
    raw_fund   = load_fundamentals(ticker).reset_index()
    pit        = build_point_in_time_fundamentals(ticker, raw_fund)
    gate_result = RiskGate.check_data_quality(bundle, df, pit,
                                               evaluated_quarter=safe_q)
    record_tool_call(session, "build_evidence_bundle",
                     f"ticker={ticker}",
                     f"committed={len(bundle.committed_evidence)}  gate={gate_result['status']}")

    # Persist analysis artifacts in session store for downstream tools
    store.update({
        "price_df_indicators": df,
        "technical_result":    technical_result,
        "sector_result":       sector_result,
        "event_result":        event_result,
        "fundamental_result":  fundamental_result,
        "macro_snapshot":      macro_snapshot,
        "bundle":              bundle,
        "gate_result":         gate_result,
        "safe_quarter":        safe_q,
        "snap_date":           snap,
    })

    return _sanitise({
        "committed_evidence":      bundle.committed_evidence,
        "gate_result":             gate_result,
        "observed_context_summary": _observed_context_summary(bundle),
        "safe_quarter":            safe_q,
    })


# ---------------------------------------------------------------------------
# Tool 9 — Get recommended action
# ---------------------------------------------------------------------------

@mcp.tool()
def get_recommended_action(session_id: str) -> dict:
    """
    Determine the analytical action for this session by running the decision
    tree from action_space.py.  Advances the harness through PREVIEW_STRATEGIES,
    RUN_BACKTEST, RISK_GATE, and GENERATE_EXPLANATION states.

    CALL THIS TOOL THIRD — only after run_full_analysis has completed for this
    session_id.  Do not call submit_human_review before this tool completes.

    If the real RiskGate returned HARD_BLOCK the gate is still honoured and
    action will be ANALYSIS_UNSAFE.  To test downstream flow with pinned-safe
    fundamentals the gate is passed through unchanged.

    Args:
        session_id: The session_id returned by start_analysis_session.

    Returns:
        {"action": str, "reasoning": [str, ...], "gate_status": str,
         "allowed_actions_note": str}
    """
    if session_id not in _SESSIONS:
        return {"error": f"session_id '{session_id}' not found."}

    store   = _SESSIONS[session_id]
    session = store["session"]

    advance_state(session, "PREVIEW_STRATEGIES")
    advance_state(session, "RUN_BACKTEST")
    advance_state(session, "RISK_GATE")
    advance_state(session, "GENERATE_EXPLANATION")

    action_result = determine_action(
        store["bundle"],
        store["gate_result"],
        store["sector_result"],
        store["event_result"],
        store["fundamental_result"],
    )
    record_tool_call(session, "generate_explanation",
                     f"ticker={store['ticker']}",
                     f"action={action_result['action']}")

    store["action_result"] = action_result

    return _sanitise({
        "action":      action_result["action"],
        "reasoning":   action_result["reasoning"],
        "gate_status": store["gate_result"]["status"],
        "allowed_actions_note": (
            "This is an educational analytical label, not investment advice. "
            "Allowed actions: HOLD, WATCH, INVESTIGATE, POTENTIAL_CATCH_UP_CANDIDATE, "
            "TECHNICAL_REVERSAL_CANDIDATE, FUNDAMENTAL_IMPROVEMENT_CANDIDATE, "
            "REJECT_SIGNAL, ANALYSIS_UNSAFE."
        ),
    })


# ---------------------------------------------------------------------------
# Tool 10 — Submit human review
# ---------------------------------------------------------------------------

@mcp.tool()
def submit_human_review(
    session_id: str,
    result:     str,
    reason:     str,
) -> dict:
    """
    Record the human analyst's verdict on the agent's recommended action,
    persist the full decision log to disk, and close the session.

    CALL THIS TOOL FOURTH — only after get_recommended_action has returned an
    action for this session_id.  Always provide a reason even for ACCEPT.

    The session is advanced to HUMAN_REVIEW, the log is written to
    data/decision_log/<TICKER>_<session_id>.json, then advanced to LOG_AND_CLOSE.

    Args:
        session_id: The session_id returned by start_analysis_session.
        result:     One of "ACCEPT", "MODIFY", "REJECT".
        reason:     Free-text justification for the review decision.

    Returns:
        {"logged": True, "log_path": str, "result": str, "action": str}
    """
    allowed = {"ACCEPT", "MODIFY", "REJECT"}
    if result not in allowed:
        return {"error": f"result must be one of {sorted(allowed)}, got {result!r}"}

    if session_id not in _SESSIONS:
        return {"error": f"session_id '{session_id}' not found."}

    store   = _SESSIONS[session_id]
    session = store["session"]
    ticker  = store["ticker"]

    advance_state(session, "HUMAN_REVIEW")

    entry = log_decision(
        session,
        store["bundle"],
        store["gate_result"],
        store["action_result"],
    )

    log_path = os.path.join(_LOG_DIR, f"{ticker}_{session_id}.json")
    add_human_review(log_path, result=result, reason=reason)
    store["log_path"] = log_path

    advance_state(session, "LOG_AND_CLOSE")
    record_tool_call(session, "log_decision",
                     f"result={result}",
                     f"path={log_path}")

    return _sanitise({
        "logged":   True,
        "log_path": log_path,
        "result":   result,
        "action":   store["action_result"]["action"],
    })


# ---------------------------------------------------------------------------
# Tool 11 — Get session replay
# ---------------------------------------------------------------------------

@mcp.tool()
def get_session_replay(session_id: str) -> dict:
    """
    Return a human-readable replay of the completed decision log for this
    session.  Includes state history, tool call trace, committed evidence,
    gate result, agent conclusion, and human review verdict.

    CALL THIS TOOL FIFTH — only after submit_human_review has completed and
    the log has been written to disk.

    The replay text mirrors the Section 27 format from the project brief:
        Agent conclusion: <action>
        Human review:     <ACCEPT|MODIFY|REJECT>
        Reason:           <reason>

    Args:
        session_id: The session_id returned by start_analysis_session.

    Returns:
        {"replay_text": str, "log_path": str}
    """
    if session_id not in _SESSIONS:
        return {"error": f"session_id '{session_id}' not found."}

    store    = _SESSIONS[session_id]
    log_path = store.get("log_path")

    if not log_path or not os.path.isfile(log_path):
        return {"error": "Log file not found. Call submit_human_review first."}

    log_entry = load_decision_log(log_path)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        replay_analysis(log_entry)

    return {"replay_text": buf.getvalue(), "log_path": log_path}


# ---------------------------------------------------------------------------
# Entry point — stdio transport (MCP standard)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # ── Mode switch ────────────────────────────────────────────────────
    # python src/mcp_server.py          → stdio MCP server (production)
    # python src/mcp_server.py --test   → direct Python function test
    # ──────────────────────────────────────────────────────────────────
    if "--test" not in sys.argv:
        mcp.run(transport="stdio")
        sys.exit(0)

    # ── Direct Python function call test (no MCP protocol) ────────────
    # Tests tools 7-11 for ASELS in the required order:
    #   start_analysis_session → run_full_analysis → get_recommended_action
    #   → submit_human_review → get_session_replay
    # ──────────────────────────────────────────────────────────────────

    SEP  = "=" * 65
    SEP2 = "-" * 65
    TICKER = "ASELS"

    print(SEP)
    print("STEP 1 — start_analysis_session")
    print(SEP)
    r1 = start_analysis_session(TICKER)
    sid = r1["session_id"]
    for k, v in r1.items():
        print(f"  {k:<20} {v}")

    print()
    print(SEP)
    print("STEP 2 — run_full_analysis")
    print(SEP)
    r2 = run_full_analysis(sid)
    print(f"  safe_quarter         : {r2.get('safe_quarter')}")
    print(f"  gate_result.status   : {r2['gate_result']['status']}")
    print(f"  gate_result.blocks   : {r2['gate_result']['blocks'] or 'none'}")
    print(f"  gate_result.warnings : {r2['gate_result']['warnings']}")
    print(f"\n  observed_context_summary:")
    for k, v in (r2.get("observed_context_summary") or {}).items():
        print(f"    {k:<30} {v}")
    print(f"\n  committed_evidence ({len(r2['committed_evidence'])} items):")
    for item in r2["committed_evidence"]:
        print(f"    [{item['id']}] {item['statement']}")

    print()
    print(SEP)
    print("STEP 3 — get_recommended_action")
    print(SEP)
    r3 = get_recommended_action(sid)
    print(f"  action      : {r3['action']}")
    print(f"  gate_status : {r3['gate_status']}")
    print(f"  reasoning   :")
    for r in r3["reasoning"]:
        print(f"    • {r}")
    print(f"\n  note: {r3['allowed_actions_note']}")

    print()
    print(SEP)
    print("STEP 4 — submit_human_review")
    print(SEP)
    r4 = submit_human_review(
        sid,
        result = "ACCEPT",
        reason = (
            "Evidence is well-supported. WATCH is an appropriate conservative "
            "label given mixed signals: strong sector leadership and robust "
            "fundamentals (4/5) are offset by borderline technical score (3/5) "
            "and no sign-flip in QoQ growth metrics."
        ),
    )
    for k, v in r4.items():
        print(f"  {k:<15} {v}")

    print()
    print(SEP)
    print("STEP 5 — get_session_replay")
    print(SEP)
    r5 = get_session_replay(sid)
    print(r5["replay_text"])

    # ── Bonus: demonstrate evaluated_quarter scoping ──────────────────
    print()
    print(SEP)
    print("BONUS — check_data_quality: scoped vs. full-scan comparison")
    print(SEP)

    from evidence import RiskGate, build_evidence_bundle
    from point_in_time import build_point_in_time_fundamentals
    from fundamental_analysis import load_fundamentals

    store2  = _SESSIONS[sid]
    bundle2 = store2["bundle"]
    df2     = store2["price_df_indicators"]
    safe_q2 = store2["safe_quarter"]
    raw2    = load_fundamentals(TICKER).reset_index()
    pit2    = build_point_in_time_fundamentals(TICKER, raw2)

    gate_scoped = RiskGate.check_data_quality(bundle2, df2, pit2,
                                               evaluated_quarter=safe_q2)
    gate_full   = RiskGate.check_data_quality(bundle2, df2, pit2,
                                               evaluated_quarter=None)

    print(f"  Scoped (evaluated_quarter={safe_q2!r}):")
    print(f"    status   : {gate_scoped['status']}")
    print(f"    blocks   : {gate_scoped['blocks'] or 'none'}")
    print(f"    warnings : {gate_scoped['warnings']}")
    print()
    print(f"  Full-scan (evaluated_quarter=None — legacy mode):")
    print(f"    status   : {gate_full['status']}")
    print(f"    blocks   : {gate_full['blocks']}")
    print(f"    warnings : {gate_full['warnings']}")

    assert gate_scoped["status"] != "HARD_BLOCK", "Scoped mode should not HARD_BLOCK on Q3"
    assert gate_full["status"] == "HARD_BLOCK",   "Full-scan should still catch Q4 future disclosure"
    print()
    print("  ✓ scoped gate: no HARD_BLOCK for safe quarter")
    print("  ✓ full-scan gate: HARD_BLOCK still raised for future Q4 disclosure")

