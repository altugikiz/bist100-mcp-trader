"""
MCP server for BIST100 MCP Trader.

Exposes 18 tools (6 analytics, 5 harness-control, 7 research) that wrap the
analysis modules for the 30-stock BIST 100 universe so that an LLM agent
(local Qwen3 via Ollama, Claude, etc.) can query live price data, technical indicators,
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
_SRC      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PROJ_ROOT = os.path.abspath(os.path.join(_SRC, '..'))
sys.path.insert(0, _SRC)

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from data.data_collection import fetch_ohlcv
from analysis.indicators import (
    calculate_sma, calculate_rsi, calculate_macd, calculate_technical_score,
    calculate_bollinger_bands, calculate_kama, calculate_supertrend, calculate_ichimoku,
)
from analysis.support_resistance import find_support_resistance
from analysis.fundamental_analysis import (
    load_fundamentals, calculate_derived_ratios,
    calculate_market_multiples, calculate_yoy_growth,
    calculate_fundamental_score as _calc_fund_score,
)
from strategy.strategy import build_full_pipeline, generate_signal
from strategy.backtest import run_backtest, calculate_performance
from config import (
    SYMBOL as DEFAULT_TICKER,
    START_DATE as DEFAULT_START,
    END_DATE as DEFAULT_END,
    BACKTEST_START, BACKTEST_END, ANALYSIS_DATE,
    CURRENT_PRICE, SHARES_OUTSTANDING, SECTOR_AVG_PE,
    BUY_THRESHOLD, SELL_THRESHOLD, TECH_WEIGHT, FUND_WEIGHT,
    FUNDAMENTAL_CSV_PATH as _CSV_PATH,
)
from data.universe import UNIVERSE
from analysis import research_extra as _rx
from analysis.sector_analysis import rank_universe_by_sector
from analysis.reversal_events import build_event_log
from analysis.fundamental_reaction import calculate_quarterly_changes
from analysis.point_in_time import build_point_in_time_fundamentals
from harness.evidence import build_evidence_bundle, RiskGate, EvidenceBundle
from analysis.calendar_patterns import (
    calculate_weekday_returns,
    test_weekday_pattern as _wd_test,
    apply_multiple_testing_correction,
)
from analysis.news_events import get_news_events as _get_news
from harness.action_space import determine_action
from harness.decision_log import log_decision, add_human_review, load_decision_log, replay_analysis
from harness.harness import AnalysisSession, advance_state, record_tool_call

# Default end (exclusive) for price tools: the day after the analysis date, so "latest" means 2026-09-10.
_TOOL_END = str((pd.Timestamp(ANALYSIS_DATE) + pd.Timedelta(days=1)).date())

_CACHE_DIR = os.path.join(_PROJ_ROOT, "data", "universe_cache")
_LOG_DIR   = os.path.join(_PROJ_ROOT, "data", "decision_log")

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
        "Analyse BIST 100 stocks (30-ticker study universe). For a single-stock "
        "analysis call start_analysis_session, run_full_analysis, "
        "get_recommended_action and get_session_replay in that order. Use "
        "rank_sector_relative_strength, test_weekday_pattern, "
        "detect_reversal_events, get_macro_context, get_news_events and "
        "check_data_quality for research questions. All numbers must come from "
        "tool outputs; the human reviewer submits the final verdict via "
        "submit_human_review."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1 — Historical OHLCV data
# ---------------------------------------------------------------------------

@mcp.tool()
def get_stock_history(
    symbol: str = DEFAULT_TICKER,
    start: str = "2026-01-01",
    end: str = _TOOL_END,
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
    end: str = _TOOL_END,
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
    end: str = _TOOL_END,
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
def get_fundamental_metrics(period: str = "2026Q2", ticker: str = "ASELS") -> dict:
    """
    Return all fundamental financial metrics for a given quarter from the
    local Fintables CSV (data/fintables_asels.csv).

    Use this tool when the user asks about revenue, profit, margins, ROE, ROA,
    debt ratios, cash flow, or any balance-sheet metric for a specific quarter.

    Args:
        period: Quarter string in 'YYYYQn' format, e.g. '2026Q2', '2025Q4'.
                Available periods: 2024Q1 through 2026Q2 (later quarters were
                disclosed after the 2026-09-10 analysis date and are excluded).
        ticker: Short ticker (default 'ASELS'). Other tickers return the basic
                statement columns only (revenue, ebitda, net_profit, total_assets,
                shareholders_equity, net_debt), as single quarters in million TRY.

    Returns:
        dict with all metric columns for the requested period.
        Monetary values are in million TRY; margins/returns are decimals;
        multiples are raw ratio values.
    """
    ticker = ticker.upper()
    if ticker != "ASELS":
        try:
            df = _rx._quarterly_frame(ticker)
        except ValueError as exc:
            return {"error": str(exc)}
    else:
        df = load_fundamentals("ASELS")
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
    df = load_fundamentals("ASELS")
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

def _load_price_df(ticker: str, as_of: str = "") -> pd.DataFrame:
    """Read cached OHLCV for ticker, never beyond the analysis date (or beyond `as_of` when given)."""
    cap, _ = _rx.resolve_as_of(as_of)
    path = os.path.join(_CACHE_DIR, f"{ticker}.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        return df.loc[df.index <= cap]
    return fetch_ohlcv(f"{ticker}.IS", "2023-01-02", str((cap + pd.Timedelta(days=1)).date()))


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
    # Basic-column CSVs (no market multiples) have no pe_ratio; keep the
    # P/E criterion NaN instead of crashing.
    if "pe_ratio" not in raw.columns:
        raw["pe_ratio"] = float("nan")
    # A CSV whose core statement lines are all empty carries no information.
    if raw[["revenue", "net_profit"]].isna().all().all():
        raise ValueError(f"Fundamentals for {ticker} contain no data.")
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
    macro_path = os.path.join(_PROJ_ROOT, "data", "macro", "macro_daily.csv")
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
    try:
        df = _load_price_df(ticker)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        import difflib
        close = difflib.get_close_matches(ticker, list(UNIVERSE["ticker"]), n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(f"No price data for ticker '{ticker}'.{hint} "
                         "Use the exact ticker code from the user's request.")
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
        "price_end":     str(df.index[-1].date()),
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
    fundamental_result = None
    if safe_q:
        try:
            fundamental_result = _build_fundamental_result(ticker, safe_q)
        except Exception:
            safe_q = None  # fundamentals unusable (missing/empty data)
    if fundamental_result is None:
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
    try:
        raw_fund = load_fundamentals(ticker).reset_index()
        pit      = build_point_in_time_fundamentals(ticker, raw_fund)
    except Exception:
        pit = None  # no usable fundamentals: gate still runs on price data
    if safe_q is None:
        pit = None  # nothing usable to scan; avoid a legacy full-scan block
    gate_result = RiskGate.check_data_quality(bundle, df, pit,
                                               evaluated_quarter=safe_q)
    if safe_q is None:
        gate_result["warnings"].append(
            f"No usable point-in-time fundamentals for {ticker}; "
            "fundamental evidence is missing (score defaults to 0)."
        )
        if gate_result["status"] == "PASS":
            gate_result["status"] = "PASS_WITH_WARNINGS"
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
# Stateless research tools (brief Section 22)
# These wrap existing analysis modules so an agent can call each research
# scenario directly, without opening a harness session.
# ---------------------------------------------------------------------------

def _default_snap_date() -> pd.Timestamp:
    """Most recent trading day available in the cached XU100 series."""
    return _load_price_df("XU100").index[-1]


def _summarise_group(grp: pd.DataFrame, snap: pd.Timestamp, baseline: dict | None) -> dict:
    """Compact statistics for one set of events (kept small: the caller may be a small local LLM)."""
    dates = pd.to_datetime(grp["event_date"])
    rec = {"n": int(len(grp)), "days_since_last": int((snap - dates.max()).days) if len(dates) else None,
           "low_sample_warning": bool(len(grp) < 10)}
    for h in (1, 3, 5, 10):
        f = grp[f"forward_return_{h}d"].dropna()
        rec[f"n_complete_{h}d"] = int(len(f))
        rec[f"avg_fwd_{h}d"] = float(f.mean()) if len(f) else None
        rec[f"positive_rate_{h}d"] = float((f > 0).mean()) if len(f) else None
        if h == 5:
            rec["median_fwd_5d"] = float(f.median()) if len(f) else None
            rec["n_excluded_incomplete_5d"] = int(len(grp) - len(f))
            if baseline and len(f):
                rec["baseline_avg_fwd_5d_all_days"] = baseline[5]["mean"]
                rec["baseline_positive_rate_5d_all_days"] = baseline[5]["pos"]
                rec["excess_avg_fwd_5d"] = float(f.mean() - baseline[5]["mean"])
    rec["bounce_rate_5d"] = rec["positive_rate_5d"]   # name used by earlier answers
    return rec


def _summarise_events(log: pd.DataFrame, snap: pd.Timestamp, baseline: dict | None = None) -> dict:
    """Per-event-type statistics (events with incomplete forward windows are excluded per horizon)."""
    if log is None or log.empty:
        return {}
    return {etype: _summarise_group(grp, snap, baseline) for etype, grp in log.groupby("event_type")}


def _thin_events(log: pd.DataFrame, min_gap_days: int) -> pd.DataFrame:
    """Keep an event only if it is at least `min_gap_days` calendar days after the previous kept event."""
    if log.empty or min_gap_days <= 0:
        return log
    keep, last = [], None
    for i, d in zip(log.index, pd.to_datetime(log["event_date"])):
        if last is None or (d - last).days >= min_gap_days:
            keep.append(i)
            last = d
    return log.loc[keep]


@mcp.tool()
def rank_sector_relative_strength(
    sector_group: str = "",
    date: str = "",
) -> dict:
    """
    Rank stocks by sector-relative lag score (stock return minus sector-median
    return) over 5, 10 and 20 trading days.  A strongly negative lag score
    means the stock is lagging its peers; it is a research flag, NOT a BUY.

    Args:
        sector_group: Optional sector group filter, e.g. 'Financials',
                      'Retail', 'Automotive'.  Empty string returns all 30
                      stocks.
        date:         Snapshot date 'YYYY-MM-DD'.  Empty string uses the most
                      recent trading day in the data cache.

    Returns:
        {"as_of": str, "rows": [{ticker, sector_group, benchmark_type,
         lag_score_5d, lag_score_10d, lag_score_20d, intra_group_rank}, ...]}
        sorted most-laggard first.  Lag scores are fractions (-0.05 = -5 pp).
    """
    snap = pd.Timestamp(date) if date else _default_snap_date()
    rank_df = rank_universe_by_sector(snap, windows=[5, 10, 20])
    if sector_group.strip().lower() in ("all", "any", "*", "all sectors", "tümü", "hepsi"):
        sector_group = ""   # "all" means no group filter
    if sector_group:
        rank_df = rank_df[rank_df["sector_group"].str.lower() == sector_group.lower()]
        if rank_df.empty:
            groups = sorted(UNIVERSE["sector_group"].unique().tolist())
            return {"error": f"Unknown sector_group '{sector_group}'.", "valid_sector_groups": groups}
    return _sanitise({
        "as_of": str(snap.date()),
        "rows":  rank_df.to_dict("records"),
    })


@mcp.tool()
def test_weekday_pattern(
    ticker: str,
    start_date: str = "",
    end_date: str = "",
    train_end: str = "",
    cost_bps: float = 0.0,
) -> dict:
    """
    Test whether each weekday's mean return differs from the stock's overall
    mean (one-sample t-test), with Bonferroni correction and an out-of-sample
    check (first 70% of days = train, last 30% = test).

    Weekday patterns are frequently spurious.  Treat a day as stable only if
    it is significant after correction AND keeps its sign out of sample.

    Args:
        ticker:     Short BIST ticker, e.g. 'THYAO'.
        start_date: Optional 'YYYY-MM-DD' lower bound.
        end_date:   Optional 'YYYY-MM-DD' upper bound (never later than the analysis date).
        train_end:  Optional 'YYYY-MM-DD' last in-sample day; later days are the out-of-sample
                    test. Empty = chronological 70/30 split.
        cost_bps:   Optional round-trip transaction cost in basis points for a one-day hold;
                    adds Mean_Return_net_of_cost.

    Returns:
        {"ticker", "period", "n_days", "weekdays": [{Weekday, N, Mean_Return,
         Median_Return, Positive_Rate, P_Value, Significant_Raw,
         Significant_Bonferroni, train_mean, test_mean, same_sign_oos}, ...],
         "note": str}
    """
    ticker = ticker.upper()
    df = _load_price_df(ticker, end_date)
    if start_date:
        df = df.loc[df.index >= pd.Timestamp(start_date)]
    df = calculate_weekday_returns(df)

    res = _wd_test(df, ticker)
    if res.empty:
        return {"error": f"Not enough data for {ticker} in the requested period."}
    res = apply_multiple_testing_correction(res)

    # Out-of-sample stability: chronological 70/30 split
    clean = df.dropna(subset=["Daily_Return", "Weekday"])
    if train_end:
        train, test = clean.loc[clean.index <= pd.Timestamp(train_end)], clean.loc[clean.index > pd.Timestamp(train_end)]
    else:
        split = int(len(clean) * 0.7)
        train, test = clean.iloc[:split], clean.iloc[split:]
    train_m = train.groupby("Weekday")["Daily_Return"].mean()
    test_m  = test.groupby("Weekday")["Daily_Return"].mean()
    res["train_mean"] = res["Weekday"].map(train_m)
    res["test_mean"]  = res["Weekday"].map(test_m)
    res["same_sign_oos"] = (res["train_mean"] * res["test_mean"]) > 0
    res["Mean_Return_net_of_cost"] = res["Mean_Return"] - cost_bps / 1e4
    res["n_train"] = len(train)
    res["n_test"] = len(test)

    cols = ["Weekday", "N", "Mean_Return", "Mean_Return_net_of_cost", "Median_Return", "Positive_Rate",
            "P_Value", "Significant_Raw", "Significant_Bonferroni",
            "train_mean", "test_mean", "same_sign_oos"]
    return _sanitise({
        "ticker": ticker,
        "period": f"{clean.index[0].date()} to {clean.index[-1].date()}",
        "n_days": int(len(clean)),
        "train_period": f"{train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)" if len(train) else None,
        "test_period": f"{test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)" if len(test) else None,
        "weekdays": res[cols].to_dict("records"),
        "note": ("Bonferroni corrects over 5 weekday tests for this one ticker only; "
                 "testing many tickers raises the multiple-testing risk further. "
                 + ("Costs applied: %.0f bps round trip." % cost_bps if cost_bps else "No transaction costs are applied (cost_bps=0).")),
    })


@mcp.tool()
def detect_reversal_events(
    ticker: str,
    event_type: str = "",
    last_n: int = 5,
    as_of: str = "",
    rsi_state: str = "",
    volume_regime: str = "",
    supertrend_state: str = "",
    ichimoku_state: str = "",
    min_gap_days: int = 0,
) -> dict:
    """
    Detect timestamped technical reversal events (Bollinger touches, RSI
    extrema, Supertrend flips, KAMA crosses, Ichimoku events) and report
    historical statistics: count, forward returns over 1/3/5/10 trading days,
    positive rate, and the unconditional all-days baseline for comparison.
    Events whose forward window is not complete by the analysis date are
    excluded per horizon (counted in n_excluded_incomplete_*).

    Combined conditions: optional filters select events whose state at the
    event date matches, e.g. RSI trough + high volume
    (event_type='rsi_trough', volume_regime='high') or a bearish Supertrend
    below the Ichimoku cloud (supertrend_state='bearish',
    ichimoku_state='below_cloud').

    Args:
        ticker:           Short BIST ticker, e.g. 'ASELS'.
        event_type:       Optional exact event type (see "event_types" in the result).
        last_n:           How many of the most recent events to list.
        as_of:            Optional analysis date 'YYYY-MM-DD' (never later than 2026-09-10).
        rsi_state:        Optional 'trough_zone', 'peak_zone' or 'neutral'.
        volume_regime:    Optional 'high' or 'normal'.
        supertrend_state: Optional 'bullish' or 'bearish'.
        ichimoku_state:   Optional 'above_cloud', 'in_cloud' or 'below_cloud'.
        min_gap_days:     If > 0, keep only events at least this many calendar days apart
                          (reduces clustered, overlapping observations).

    Returns:
        {"ticker", "as_of", "event_types", "filters", "statistics": {type: {...}},
         "recent_events": [...], "warnings": [...]}
    """
    ticker = ticker.upper()
    df_raw = _load_price_df(ticker, as_of)
    df, _  = _build_price_with_indicators(df_raw.copy())
    snap   = df.index[-1]
    log    = build_event_log(df, ticker)
    if log.empty:
        return {"ticker": ticker, "as_of": str(snap.date()), "event_types": [], "statistics": {}, "recent_events": []}

    all_types = sorted(log["event_type"].unique().tolist())
    if event_type:
        if event_type not in all_types:
            return {"error": f"Unknown event_type '{event_type}'.", "event_types": all_types}
        log = log[log["event_type"] == event_type]
    filters = {k: v for k, v in {"rsi_state": rsi_state, "volume_regime": volume_regime,
                                 "supertrend_state": supertrend_state, "ichimoku_state": ichimoku_state}.items() if v}
    for col, val in filters.items():
        log = log[log[col] == val]
    log = log.sort_values("event_date")
    n_before_thin = len(log)
    log = _thin_events(log, int(min_gap_days))

    close = df["Close"]
    baseline = {}
    for h in (1, 3, 5, 10):
        fr = (close.shift(-h) / close - 1).dropna()
        baseline[h] = {"mean": float(fr.mean()), "pos": float((fr > 0).mean())}

    warnings = ["Forward returns use only closes up to the analysis date; incomplete windows are excluded, not estimated.",
                "Compare with the all-days baseline: a positive average alone is not evidence of a reversal edge.",
                "Events cluster in time, so observations overlap; read small samples with caution."]
    if filters and len(log) < 10:
        warnings.append(f"Only {len(log)} events match the combined condition: low-sample warning.")
    if min_gap_days and n_before_thin != len(log):
        warnings.append(f"min_gap_days={min_gap_days} removed {n_before_thin - len(log)} clustered events.")

    if filters and not event_type:
        # combined condition: one pooled statistic across the matching events (types listed by count only)
        stats_out = {"pooled_matching_events": _summarise_group(log, snap, baseline)} if len(log) else {}
        stats_out["matching_events_by_type"] = {k: int(v) for k, v in log["event_type"].value_counts().items()}
    else:
        stats_out = _summarise_events(log, snap, baseline)

    recent = log.tail(max(1, int(last_n)))
    keep   = ["event_date", "event_type", "close_price", "rsi_state", "supertrend_state",
              "kama_slope", "ichimoku_state", "volume_regime",
              "forward_return_1d", "forward_return_3d", "forward_return_5d", "forward_return_10d"]
    return _sanitise({
        "ticker":        ticker,
        "as_of":         str(snap.date()),
        "event_types":   all_types if not filters else None,
        "filters":       filters,
        "n_matching_events": int(len(log)),
        "statistics":    stats_out,
        "recent_events": recent[[c for c in keep if c in recent.columns]].to_dict("records"),
        "warnings":      warnings,
    })


@mcp.tool()
def get_macro_context(start_date: str = "", end_date: str = "") -> dict:
    """
    Return the macro context (USD/TRY, EUR/TRY, TCMB policy rate, TUIK CPI
    year-over-year, Fed funds rate) for a date range, with start/end values
    and changes.

    Args:
        start_date: 'YYYY-MM-DD' (empty = 30 days before end_date).
        end_date:   'YYYY-MM-DD' (empty = last available date).

    Returns:
        {"start": {...}, "end": {...}, "change": {...}, "notes": [...]}
    """
    path  = os.path.join(_PROJ_ROOT, "data", "macro", "macro_daily.csv")
    macro = pd.read_csv(path, index_col="Date", parse_dates=True)
    end   = pd.Timestamp(end_date) if end_date else macro.index[-1]
    start = pd.Timestamp(start_date) if start_date else end - pd.Timedelta(days=30)
    if end > macro.index[-1] or start < macro.index[0]:
        return {"error": f"Macro data covers {macro.index[0].date()} to {macro.index[-1].date()} only."}
    sl = macro.loc[start:end]
    if sl.empty:
        return {"error": "No macro rows in the requested range."}
    first, last = sl.iloc[0], sl.iloc[-1]
    change = {
        "usdtry_pct":      float((last["usdtry"] / first["usdtry"] - 1) * 100),
        "eurtry_pct":      float((last["eurtry"] / first["eurtry"] - 1) * 100),
        "tcmb_rate_pp":    float(last["tcmb_rate"] - first["tcmb_rate"]),
        "fed_rate_pp":     float(last["fed_rate"] - first["fed_rate"]),
    }
    return _sanitise({
        "start":  {"date": sl.index[0], **first.to_dict()},
        "end":    {"date": sl.index[-1], **last.to_dict()},
        "change": change,
        "notes": [
            "tufe_yoy is linearly interpolated between published anchor points "
            "and is NaN before 2024-05-31.",
            "tcmb_rate and fed_rate are step series forward-filled from decision dates.",
        ],
    })


@mcp.tool()
def get_news_events(ticker: str, start_date: str = "", end_date: str = "") -> dict:
    """
    Return curated, timestamped news/corporate events for a ticker with source
    metadata.  The set is a small hand-curated list (no licensed news feed and
    no speech-to-text), so treat absence of events as "not covered", not as
    "nothing happened".

    Args:
        ticker:     Short BIST ticker, e.g. 'ASELS'.
        start_date: Optional 'YYYY-MM-DD' lower bound.
        end_date:   Optional 'YYYY-MM-DD' upper bound.

    Returns:
        {"ticker", "n_events", "events": [...], "coverage_note": str}
    """
    events = _get_news(ticker, start_date or None, end_date or None)
    return _sanitise({
        "ticker":   ticker.upper(),
        "n_events": len(events),
        "events":   events,
        "coverage_note": ("Curated list only; contextual evidence, not ground truth. "
                          "Speech-to-text commentary is not implemented."),
    })


@mcp.tool()
def check_data_quality(ticker: str) -> dict:
    """
    Run the Risk & Data Quality gate for a ticker without opening a session:
    universe membership, price-history NaN rate, look-ahead check on fundamental
    disclosure dates, sector-peer availability, number of reversal events,
    macro completeness and data freshness.

    Args:
        ticker: Short BIST ticker, e.g. 'ASELS'.

    Returns:
        {"ticker", "status": PASS | PASS_WITH_WARNINGS | HARD_BLOCK,
         "blocks": [...], "warnings": [...], "price_end": str,
         "fundamentals_available": bool}
    """
    ticker = ticker.upper()
    if not (UNIVERSE["ticker"] == ticker).any():
        return {"ticker": ticker, "status": "HARD_BLOCK",
                "blocks": [f"Ticker '{ticker}' is not in the 30-stock coverage universe."],
                "warnings": []}

    df_raw = _load_price_df(ticker)
    df, _  = _build_price_with_indicators(df_raw.copy())
    snap   = df.index[-1]

    bundle = EvidenceBundle(ticker)
    bundle.observed_context["events"] = _summarise_events(build_event_log(df, ticker), snap)
    try:
        bundle.observed_context["macro"] = _build_macro_snapshot(snap)
    except Exception:
        bundle.observed_context["macro"] = {}

    pit, safe_q, extra_warnings = None, None, []
    try:
        raw = load_fundamentals(ticker).reset_index()
        pit = build_point_in_time_fundamentals(ticker, raw)
        safe_q = _find_safe_quarter(ticker, snap)
    except Exception:
        extra_warnings.append(f"No fundamental data available for {ticker}; "
                              "fundamental evidence will be missing.")

    gate = RiskGate.check_data_quality(bundle, df, pit, evaluated_quarter=safe_q)
    warnings = list(gate["warnings"]) + extra_warnings
    if not bundle.observed_context["macro"]:
        warnings.append("Macro snapshot unavailable for the price end date.")
    stale_days = (pd.Timestamp.today().normalize() - snap).days
    if stale_days > 30:
        warnings.append(f"Price data ends {snap.date()} ({stale_days} days old); results are historical only.")

    status = gate["status"]
    if status != "HARD_BLOCK" and warnings:
        status = "PASS_WITH_WARNINGS"
    return _sanitise({
        "ticker": ticker, "status": status,
        "blocks": gate["blocks"], "warnings": warnings,
        "price_end": str(snap.date()),
        "fundamentals_available": pit is not None,
    })


@mcp.tool(name="build_evidence_bundle")
def build_evidence_bundle_tool(session_id: str) -> dict:
    """
    Return the full evidence bundle for a session whose run_full_analysis has
    completed: ALL observed features and the smaller COMMITTED evidence set
    that may be cited in an explanation.  Cite only committed evidence.

    Args:
        session_id: The session_id returned by start_analysis_session.

    Returns:
        {"ticker", "observed_context": {...}, "committed_evidence": [...],
         "n_observed": int, "n_committed": int}
    """
    store = _SESSIONS.get(session_id)
    if store is None:
        return {"error": f"session_id '{session_id}' not found. Call start_analysis_session first."}
    bundle = store.get("bundle")
    if bundle is None:
        return {"error": "No evidence bundle yet. Call run_full_analysis first."}
    return _sanitise({
        "ticker":              bundle.ticker,
        "observed_context":    bundle.observed_context,
        "committed_evidence":  bundle.committed_evidence,
        "n_observed":          len(bundle.observed_context),
        "n_committed":         len(bundle.committed_evidence),
    })


# ---------------------------------------------------------------------------
# Analysis-date aware research tools (sector catch-up, weekday sequences,
# quarterly fundamentals + price reaction, macro relationship)
# ---------------------------------------------------------------------------

@mcp.tool()
def backtest_sector_catchup(
    sector_group: str,
    as_of: str = "",
    start: str = "2024-01-01",
    lag_threshold: float = -0.05,
    horizons: str = "5,10,20",
    cost_bps: float = 0.0,
) -> dict:
    """
    Historical test of the "sector laggard catches up" idea for one sector group.
    Every Friday, a stock whose 20-day return is below its sector benchmark by more
    than the threshold is a laggard. The tool then measures the next 5/10/20 trading
    days: catch-up rate (stock beats the sector benchmark), hit rate (return > 0),
    mean/median return, excess return, and the same statistics for non-laggard peers.
    Signals whose forward window is incomplete at the analysis date are excluded.

    Args:
        sector_group:  Sector group, e.g. 'Financials', 'Metals & Mining', 'Retail'.
        as_of:         Analysis date 'YYYY-MM-DD' (never later than 2026-09-10).
        start:         First snapshot date.
        lag_threshold: Lag score cut-off as a fraction (-0.05 = 5 percentage points behind).
        horizons:      Comma-separated forward windows in trading days.
        cost_bps:      Optional round-trip transaction cost in basis points.

    Returns:
        {"sector_group", "members", "results": {"5d": {n_signals, catchup_rate, hit_rate_abs,
         mean_return, baseline_catchup_rate_non_laggards, binomial_p_vs_baseline, ...}}, "warnings"}
    """
    try:
        hs = tuple(int(x) for x in str(horizons).split(",") if x.strip())
    except ValueError:
        return {"error": "horizons must be comma-separated integers, e.g. '5,10,20'."}
    return _sanitise(_rx.sector_catchup_backtest(sector_group, as_of, start, lag_threshold, 20, hs or (5, 10, 20), cost_bps))


@mcp.tool()
def test_weekday_sequence(
    ticker: str,
    first_day: str = "Monday",
    first_direction: str = "negative",
    second_day: str = "Friday",
    second_direction: str = "positive",
    as_of: str = "",
    start: str = "",
    train_end: str = "",
    cost_bps: float = 0.0,
    n_hypotheses: int = 1,
) -> dict:
    """
    Test a multi-day weekday sequence within each calendar week, e.g. "a negative
    Monday is followed by a positive Friday". Reports P(second-day move | first-day
    move) against the unconditional probability, a binomial test, an in-sample /
    out-of-sample split at train_end, the cost-adjusted hold return from the first
    day's close to the second day's close, and whether the pattern is stable out of sample.

    Args:
        ticker:           Short BIST ticker, e.g. 'THYAO'.
        first_day:        Weekday name of the condition day.
        first_direction:  'negative' or 'positive' return on the first day.
        second_day:       Later weekday name.
        second_direction: 'positive' or 'negative' return on the second day.
        as_of:            Analysis date (never later than 2026-09-10).
        start:            Optional first date.
        train_end:        Last in-sample date; later weeks are out-of-sample (empty = 70/30 split).
        cost_bps:         Optional round-trip transaction cost in basis points.
        n_hypotheses:     Number of patterns scanned (for Bonferroni alpha = 0.05 / n).
    """
    return _sanitise(_rx.weekday_sequence_test(ticker, first_day, first_direction, second_day, second_direction,
                                               as_of, start, train_end, cost_bps, n_hypotheses))


@mcp.tool()
def calculate_fundamental_change(ticker: str, as_of: str = "") -> dict:
    """
    For the latest quarterly result disclosed by the analysis date, compute the
    change in revenue, net profit, EBITDA margin and net debt / TTM EBITDA
    (YoY and QoQ), then the stock's 1-, 5- and 20-trading-day returns after the
    disclosure date, raw and abnormal versus XU100 and versus the sector peer
    median, plus the historical association between profit growth and the
    reaction. Quarters disclosed after the analysis date are excluded and listed;
    return windows that are not complete by the analysis date are reported as null.
    Disclosure dates are proxies; the result is an association, not a causal claim.

    Args:
        ticker: Short BIST ticker, e.g. 'EREGL' (26 tickers have fundamentals).
        as_of:  Analysis date 'YYYY-MM-DD' (never later than 2026-09-10).
    """
    try:
        return _sanitise(_rx.fundamental_change(ticker, as_of))
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def analyze_macro_stock_relationship(
    ticker: str,
    macro_variable: str = "usdtry",
    start_date: str = "",
    end_date: str = "",
    frequency: str = "weekly",
) -> dict:
    """
    Relationship between a stock and a macro series over a period: stock total
    return, macro start/end values, correlation of weekly (or monthly/daily)
    stock returns with the macro series' change, first-half vs second-half
    consistency, and the stock's return in FX terms for usdtry/eurtry.
    Association only, never causation.

    Args:
        ticker:         Short BIST ticker, e.g. 'EREGL'.
        macro_variable: 'usdtry', 'eurtry', 'tcmb_rate', 'tufe_yoy' or 'fed_rate'.
        start_date:     Optional 'YYYY-MM-DD'.
        end_date:       Optional 'YYYY-MM-DD' (never later than 2026-09-10).
        frequency:      'weekly' (default), 'monthly' or 'daily'.
    """
    try:
        return _sanitise(_rx.macro_stock_relationship(ticker, macro_variable, start_date, end_date, frequency))
    except ValueError as exc:
        return {"error": str(exc)}


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

    from harness.evidence import RiskGate, build_evidence_bundle
    from analysis.point_in_time import build_point_in_time_fundamentals
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

