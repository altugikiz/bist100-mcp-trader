"""
Central configuration for BIST100 MCP Trader.
All project-wide constants are defined here; import from this module instead
of hardcoding values in individual analysis scripts.
"""

import os

# ---------------------------------------------------------------------------
# Ticker universe
# ---------------------------------------------------------------------------

# Import the full 30-stock universe; use UNIVERSE instead of a single SYMBOL.
from universe import UNIVERSE  # noqa: E402  (circular-safe: universe has no config import)

BENCHMARK_SYMBOL = "XU100.IS"       # BIST 100 index benchmark

# Legacy alias kept for backward compatibility with scripts that still use SYMBOL.
SYMBOL = "ASELS.IS"

# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------

# Full history window used for indicator warm-up and ML feature engineering
START_DATE = "2023-01-01"
END_DATE   = "2026-01-01"

# Narrower window used for backtesting (requires complete fundamental data)
BACKTEST_START = "2024-01-01"
BACKTEST_END   = "2025-12-31"

# ---------------------------------------------------------------------------
# File paths (resolved relative to this file so scripts work from any cwd)
# ---------------------------------------------------------------------------

_SRC_DIR = os.path.dirname(__file__)

FUNDAMENTAL_CSV_PATH = os.path.join(_SRC_DIR, "..", "data", "fintables_asels.csv")
RESULTS_DIR          = os.path.join(_SRC_DIR, "..", "results")

# ---------------------------------------------------------------------------
# Fundamental valuation inputs
# ---------------------------------------------------------------------------

CURRENT_PRICE      = 372.50          # snapshot share price in TRY
SHARES_OUTSTANDING = 4_560_000_000   # total shares (absolute count)
SECTOR_AVG_PE      = 65.6            # BIST defence sector median P/E

# ---------------------------------------------------------------------------
# Strategy parameters (final optimised configuration)
# ---------------------------------------------------------------------------

BUY_THRESHOLD  = 3.0   # Total Score >= BUY_THRESHOLD  → BUY signal
SELL_THRESHOLD = 1.5   # Total Score <= SELL_THRESHOLD → SELL signal
TECH_WEIGHT    = 0.50  # weight for technical_score in Total Score
FUND_WEIGHT    = 0.50  # weight for fundamental_score in Total Score
# MACRO_WEIGHT = 0.00  # reserved; set to 0 until macro data is collected
