"""
Stock universe definition for BIST100 MCP Trader.
Defines the 30-stock coverage universe with ticker, Yahoo Finance symbol, sector,
and a coarser sector_group used for peer-comparison and relative-value analysis.

Design decision: the assigned 30-stock universe contains many single-member
simplified sectors (e.g., only one bank, one defense company). To enable
meaningful peer comparison, closely related simplified sectors were merged into
broader sector_group categories (e.g., Chemicals, Financials, Metals & Mining).
Tickers with no sector peers even after this merge fall back to XU100-relative
performance instead of sector-median performance. This is documented as a
methodological limitation.
"""

import pandas as pd
from typing import List

# ---------------------------------------------------------------------------
# Sector-group mapping: ticker -> broader group label.
# Tickers absent from this map retain their original fine-grained sector value.
# ---------------------------------------------------------------------------
_SECTOR_GROUP_MAP: dict = {
    # Chemicals
    "SASA":  "Chemicals",
    "GUBRF": "Chemicals",
    "PETKM": "Chemicals",
    # Metals & Mining
    "EREGL": "Metals & Mining",
    "KRDMD": "Metals & Mining",
    "TRALT": "Metals & Mining",
    # Financials
    "AKBNK": "Financials",
    "KCHOL": "Financials",
    "ISMEN": "Financials",
    "TURSG": "Financials",
    # Healthcare
    "MPARK": "Healthcare",
    "ECILC": "Healthcare",
    # Retail
    "BIMAS": "Retail",
    "MGROS": "Retail",
    "MAVI":  "Retail",
    # Real Estate & Construction
    "EKGYO": "Real Estate & Construction",
    "ENKAI": "Real Estate & Construction",
    # Automotive
    "FROTO": "Automotive",
    "TOASO": "Automotive",
    # Transportation-Airlines
    "THYAO": "Transportation-Airlines",
    "PGSUS": "Transportation-Airlines",
    # Telecommunications
    "TCELL": "Telecommunications",
    "TTKOM": "Telecommunications",
    # Electric Utilities
    "ENJSA": "Electric Utilities",
    "AKSEN": "Electric Utilities",
    # Standalone — keep original sector as sector_group
    # ASELS  -> Technology-Defense
    # TUPRS  -> Petroleum-Refining
    # CCOLA  -> Food & Beverage
    # OYAKC  -> Cement
    # ARCLK  -> Durables-Machinery
}

UNIVERSE_RAW: List[dict] = [
    {"ticker": "ASELS",  "yahoo_symbol": "ASELS.IS",  "sector": "Technology-Defense"},
    {"ticker": "TUPRS",  "yahoo_symbol": "TUPRS.IS",  "sector": "Petroleum-Refining"},
    {"ticker": "BIMAS",  "yahoo_symbol": "BIMAS.IS",  "sector": "Retail Trade"},
    {"ticker": "THYAO",  "yahoo_symbol": "THYAO.IS",  "sector": "Transportation-Airlines"},
    {"ticker": "AKBNK",  "yahoo_symbol": "AKBNK.IS",  "sector": "Banking"},
    {"ticker": "KCHOL",  "yahoo_symbol": "KCHOL.IS",  "sector": "Holding-Investment"},
    {"ticker": "EREGL",  "yahoo_symbol": "EREGL.IS",  "sector": "Basic Metals-Steel"},
    {"ticker": "TCELL",  "yahoo_symbol": "TCELL.IS",  "sector": "Telecommunications"},
    {"ticker": "CCOLA",  "yahoo_symbol": "CCOLA.IS",  "sector": "Food & Beverage"},
    {"ticker": "MGROS",  "yahoo_symbol": "MGROS.IS",  "sector": "Retail Trade"},
    {"ticker": "SASA",   "yahoo_symbol": "SASA.IS",   "sector": "Chemicals-Plastics"},
    {"ticker": "TRALT",  "yahoo_symbol": "TRALT.IS",  "sector": "Mining"},
    {"ticker": "FROTO",  "yahoo_symbol": "FROTO.IS",  "sector": "Automotive"},
    {"ticker": "ENKAI",  "yahoo_symbol": "ENKAI.IS",  "sector": "Construction"},
    {"ticker": "PGSUS",  "yahoo_symbol": "PGSUS.IS",  "sector": "Transportation-Airlines"},
    {"ticker": "MPARK",  "yahoo_symbol": "MPARK.IS",  "sector": "Healthcare"},
    {"ticker": "EKGYO",  "yahoo_symbol": "EKGYO.IS",  "sector": "Real Estate Investment Trust"},
    {"ticker": "GUBRF",  "yahoo_symbol": "GUBRF.IS",  "sector": "Chemicals-Fertilizer"},
    {"ticker": "TURSG",  "yahoo_symbol": "TURSG.IS",  "sector": "Insurance"},
    {"ticker": "ENJSA",  "yahoo_symbol": "ENJSA.IS",  "sector": "Electric Utilities"},
    {"ticker": "TTKOM",  "yahoo_symbol": "TTKOM.IS",  "sector": "Telecommunications"},
    {"ticker": "OYAKC",  "yahoo_symbol": "OYAKC.IS",  "sector": "Cement"},
    {"ticker": "AKSEN",  "yahoo_symbol": "AKSEN.IS",  "sector": "Electric Utilities"},
    {"ticker": "ISMEN",  "yahoo_symbol": "ISMEN.IS",  "sector": "Brokerage-Capital Markets"},
    {"ticker": "ARCLK",  "yahoo_symbol": "ARCLK.IS",  "sector": "Durables-Machinery"},
    {"ticker": "ECILC",  "yahoo_symbol": "ECILC.IS",  "sector": "Healthcare-Pharmaceuticals"},
    {"ticker": "MAVI",   "yahoo_symbol": "MAVI.IS",   "sector": "Retail-Apparel"},
    {"ticker": "PETKM",  "yahoo_symbol": "PETKM.IS",  "sector": "Petrochemicals"},
    {"ticker": "TOASO",  "yahoo_symbol": "TOASO.IS",  "sector": "Automotive"},
    {"ticker": "KRDMD",  "yahoo_symbol": "KRDMD.IS",  "sector": "Basic Metals-Steel"},
]

UNIVERSE: pd.DataFrame = pd.DataFrame(UNIVERSE_RAW)
UNIVERSE["sector_group"] = UNIVERSE["ticker"].map(_SECTOR_GROUP_MAP).fillna(UNIVERSE["sector"])


# ---------------------------------------------------------------------------
# Helper: normalise ticker input
# ---------------------------------------------------------------------------

def _normalise(ticker: str) -> str:
    return ticker.upper().replace(".IS", "")


def _lookup(ticker: str) -> pd.Series:
    normalised = _normalise(ticker)
    row = UNIVERSE[UNIVERSE["ticker"] == normalised]
    if row.empty:
        raise ValueError(f"Ticker '{ticker}' not found in universe.")
    return row.iloc[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sector_peers(ticker: str) -> List[str]:
    """
    Return all tickers in the same sector_group as the given ticker, excluding itself.

    Args:
        ticker: Short ticker code (e.g. 'ASELS') or Yahoo symbol (e.g. 'ASELS.IS').

    Returns:
        List of short ticker codes sharing the same sector_group.

    Raises:
        ValueError: If the ticker is not found in the universe.
    """
    normalised = _normalise(ticker)
    row = _lookup(ticker)
    group = row["sector_group"]
    peers = UNIVERSE[
        (UNIVERSE["sector_group"] == group) & (UNIVERSE["ticker"] != normalised)
    ]
    return peers["ticker"].tolist()


def has_sector_peers(ticker: str) -> bool:
    """
    Return True if the ticker has at least one other member in its sector_group.

    Args:
        ticker: Short ticker code or Yahoo symbol.

    Returns:
        True when sector peers exist, False for standalone tickers.

    Raises:
        ValueError: If the ticker is not found in the universe.
    """
    return len(get_sector_peers(ticker)) > 0


def get_relative_benchmark(ticker: str) -> dict:
    """
    Determine the appropriate peer benchmark for relative-value analysis.

    Returns a sector-peer list when the ticker has group members; otherwise
    signals an XU100 fallback for tickers that remain isolated even after
    sector-group merging.

    Args:
        ticker: Short ticker code or Yahoo symbol.

    Returns:
        {"type": "sector",        "peers": [<ticker>, ...]}  — sector comparison
        {"type": "xu100_fallback","peers": []}               — index comparison

    Raises:
        ValueError: If the ticker is not found in the universe.
    """
    peers = get_sector_peers(ticker)
    if peers:
        return {"type": "sector", "peers": peers}
    return {"type": "xu100_fallback", "peers": []}
