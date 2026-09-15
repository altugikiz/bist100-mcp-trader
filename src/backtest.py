"""
Backtesting module for BIST100 MCP Trader.
Simulates the strategy defined in strategy.py against buy-and-hold benchmarks.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from config import SYMBOL, BENCHMARK_SYMBOL, BACKTEST_START, BACKTEST_END, RESULTS_DIR
from strategy import build_full_pipeline


# ---------------------------------------------------------------------------
# Signal → position mapping
# ---------------------------------------------------------------------------
# BUY  = 1  (long, fully invested)
# HOLD = maintain previous position (position carries forward)
# SELL = 0  (exit to cash — no short selling)
#
# This avoids look-ahead bias: Signal generated at close of day T is
# translated to a position that is entered at the open of day T+1.
# We implement this by shifting the raw position by one day before
# computing returns.
# ---------------------------------------------------------------------------

_SIGNAL_MAP = {"BUY": 1, "SELL": 0, "HOLD": np.nan}


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate the strategy on historical price data.

    Position logic
    --------------
    - BUY  → position = 1 (fully long)
    - SELL → position = 0 (exit to cash; no shorting)
    - HOLD → carry forward the previous position (ffill)

    Look-ahead bias prevention
    --------------------------
    Signal at day T is known only after that day's close.  The position is
    therefore entered at day T+1.  We implement this with Signal.shift(1):
    the position column already reflects the T+1 entry, so no additional
    shift is needed when computing daily returns.

    Return calculation
    ------------------
        daily_return     = Close.pct_change()          (price return, day T)
        strategy_return  = position[T] * daily_return[T]
            where position[T] = signal[T-1] (after shift)
        equity           = (1 + strategy_return).cumprod()

    Args:
        df: DataFrame with 'Close' and 'Signal' columns and a DatetimeIndex.

    Returns:
        df copy with added columns:
            raw_position, position, daily_return,
            strategy_return, equity, bah_equity
    """
    df = df.copy()

    # Convert signal strings to numeric, NaN for HOLD (will be forward-filled)
    df["raw_position"] = df["Signal"].map(_SIGNAL_MAP)
    df["raw_position"] = df["raw_position"].ffill().fillna(0)

    # Shift by 1: signal known at close of T → position entered at open of T+1
    df["position"] = df["raw_position"].shift(1).fillna(0)

    # Daily price return
    df["daily_return"] = df["Close"].pct_change()

    # Strategy return: only earned when in position
    df["strategy_return"] = df["position"] * df["daily_return"]

    # Equity curves (start at 1.0)
    df["equity"]     = (1 + df["strategy_return"].fillna(0)).cumprod()
    df["bah_equity"] = (1 + df["daily_return"].fillna(0)).cumprod()

    return df


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough decline of an equity curve."""
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


def _sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio assuming a risk-free rate of `rf` (daily)."""
    excess = returns - rf
    if excess.std() == 0:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def _win_rate(returns: pd.Series, position: pd.Series) -> float:
    """Fraction of days-in-position where the strategy return was positive."""
    active = returns[position > 0].dropna()
    if len(active) == 0:
        return float("nan")
    return float((active > 0).sum() / len(active))


def calculate_performance(
    df: pd.DataFrame,
    label: str = "Strategy",
) -> dict:
    """
    Compute summary performance metrics for one equity series.

    Args:
        df:    DataFrame containing 'equity', 'strategy_return',
               'bah_equity', 'daily_return', 'position' columns.
        label: Name used only as a key in the returned dict.

    Returns:
        dict with keys:
            label, cumulative_return, bah_return, benchmark_diff,
            max_drawdown, sharpe_ratio, win_rate, trading_days, days_invested
    """
    strat_ret  = df["strategy_return"].dropna()
    bah_ret    = df["daily_return"].dropna()

    cum_strat  = float(df["equity"].iloc[-1] - 1)
    cum_bah    = float(df["bah_equity"].iloc[-1] - 1)

    return {
        "label":            label,
        "cumulative_return": cum_strat,
        "bah_return":       cum_bah,
        "benchmark_diff":   cum_strat - cum_bah,
        "max_drawdown":     _max_drawdown(df["equity"]),
        "sharpe_ratio":     _sharpe_ratio(strat_ret),
        "win_rate":         _win_rate(strat_ret, df["position"]),
        "trading_days":     len(df),
        "days_invested":    int((df["position"] > 0).sum()),
    }


def fetch_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    """Download a benchmark close series and return as a normalised equity curve."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw["Close"].dropna()
    daily_ret = close.pct_change().fillna(0)
    return (1 + daily_ret).cumprod()


def plot_equity_curves(
    strategy_equity: pd.Series,
    bah_equity: pd.Series,
    bist100_equity: pd.Series,
    output_path: str,
) -> None:
    """Plot three equity curves on a single chart and save to PNG."""
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(strategy_equity.index, strategy_equity,
            color="#2196F3", linewidth=1.8, label="Strategy (Signal-based)")
    ax.plot(bah_equity.index, bah_equity,
            color="#4CAF50", linewidth=1.4, linestyle="--", label="ASELS Buy & Hold")
    ax.plot(bist100_equity.index, bist100_equity,
            color="#FF9800", linewidth=1.4, linestyle=":", label="BIST100 (XU100.IS)")

    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="-", alpha=0.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    ax.set_title(
        f"{SYMBOL} Strategy vs Benchmarks — Equity Curves "
        f"({BACKTEST_START} → {BACKTEST_END})",
        fontsize=13,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalised Equity (start = 1.0)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved → {os.path.abspath(output_path)}")


def print_performance_table(metrics_list: list[dict]) -> None:
    """Pretty-print a side-by-side performance comparison table."""
    rows = {
        "Cumulative Return":  "cumulative_return",
        "B&H Return":         "bah_return",
        "vs B&H (alpha)":     "benchmark_diff",
        "Max Drawdown":       "max_drawdown",
        "Sharpe Ratio":       "sharpe_ratio",
        "Win Rate":           "win_rate",
        "Trading Days":       "trading_days",
        "Days Invested":      "days_invested",
    }
    pct_keys = {"cumulative_return", "bah_return", "benchmark_diff",
                "max_drawdown", "win_rate"}

    labels = [m["label"] for m in metrics_list]
    col_w = max(16, max(len(l) for l in labels) + 2)

    header = f"{'Metric':<26}" + "".join(f"{l:>{col_w}}" for l in labels)
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for display, key in rows.items():
        row = f"{display:<26}"
        for m in metrics_list:
            val = m[key]
            if key in pct_keys and isinstance(val, float):
                row += f"{val*100:>{col_w-1}.2f}%"
            elif isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{val:>{col_w}}"
        print(row)

    print("=" * len(header))


if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 140)

    print("Building strategy pipeline …")
    signal_df = build_full_pipeline()

    print("Running backtest …")
    bt = run_backtest(signal_df)

    # BIST100 benchmark
    print("Fetching BIST100 benchmark …")
    bist100_equity = fetch_benchmark(BENCHMARK_SYMBOL, BACKTEST_START, BACKTEST_END).reindex(bt.index).ffill()

    # Performance metrics
    strat_metrics = calculate_performance(bt, label="Strategy")

    # Buy-and-hold metrics reuse bah_equity already in bt
    bah_metrics = {
        "label":             "ASELS B&H",
        "cumulative_return":  float(bt["bah_equity"].iloc[-1] - 1),
        "bah_return":         float(bt["bah_equity"].iloc[-1] - 1),
        "benchmark_diff":     0.0,
        "max_drawdown":       _max_drawdown(bt["bah_equity"]),
        "sharpe_ratio":       _sharpe_ratio(bt["daily_return"].dropna()),
        "win_rate":           float((bt["daily_return"] > 0).mean()),
        "trading_days":       len(bt),
        "days_invested":      len(bt),
    }

    bist_daily = bist100_equity.pct_change().fillna(0)
    bist_metrics = {
        "label":             "BIST100",
        "cumulative_return":  float(bist100_equity.iloc[-1] - 1),
        "bah_return":         float(bist100_equity.iloc[-1] - 1),
        "benchmark_diff":     float(bist100_equity.iloc[-1] - 1) - float(bt["bah_equity"].iloc[-1] - 1),
        "max_drawdown":       _max_drawdown(bist100_equity),
        "sharpe_ratio":       _sharpe_ratio(bist_daily),
        "win_rate":           float((bist_daily > 0).mean()),
        "trading_days":       len(bist100_equity),
        "days_invested":      len(bist100_equity),
    }

    print("\n")
    print_performance_table([strat_metrics, bah_metrics, bist_metrics])

    # Last 15 days detail
    display_cols = ["Close", "technical_score", "fundamental_score",
                    "Total_Score", "Signal", "position", "strategy_return", "equity"]
    print("\n=== Last 15 days — backtest detail ===")
    print(bt[display_cols].tail(15).to_string())

    # Equity curve chart
    output = os.path.join(RESULTS_DIR, "backtest_equity_curve.png")
    plot_equity_curves(bt["equity"], bt["bah_equity"], bist100_equity, output)
