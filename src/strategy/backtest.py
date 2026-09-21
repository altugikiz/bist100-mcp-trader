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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import SYMBOL, BENCHMARK_SYMBOL, BACKTEST_START, BACKTEST_END, RESULTS_DIR
from strategy.strategy import build_full_pipeline


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


# ---------------------------------------------------------------------------
# 4 new functions: transaction costs, walk-forward split, walk-forward
# backtest, and low-signal flag  (Section 18 requirements)
# ---------------------------------------------------------------------------

def add_transaction_costs(
    returns_series: pd.Series,
    position_series: pd.Series,
    cost_bps: float = 10,
) -> tuple[pd.Series, int]:
    """
    Deduct a round-trip transaction cost each time the position changes.

    A cost of ``cost_bps`` basis points is subtracted from the strategy
    return on every day where ``position_series.diff() != 0`` (i.e. any
    entry or exit).  The cost is expressed as a fraction:
        cost_fraction = cost_bps / 10_000

    Args:
        returns_series:  Daily strategy returns (e.g. bt["strategy_return"]).
        position_series: Corresponding position series (0 or 1).
        cost_bps:        One-way transaction cost in basis points (default 10 →
                         0.10% per trade leg, 0.20% round-trip).

    Returns:
        (cost_adjusted_returns, turnover_count)
            cost_adjusted_returns : pd.Series with the same index as inputs.
            turnover_count        : int, number of days a position change
                                    was detected (= number of trade legs).
    """
    cost_fraction = cost_bps / 10_000
    trades        = position_series.diff().abs().fillna(0)   # 1 on change days
    turnover_count = int((trades > 0).sum())

    adjusted = returns_series.copy()
    adjusted[trades > 0] -= cost_fraction
    return adjusted, turnover_count


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 4,
    train_ratio: float = 0.6,
) -> list[dict]:
    """
    Partition ``df`` into ``n_splits`` consecutive, non-overlapping
    train/test windows (rolling walk-forward).

    Each window spans  total_days / n_splits  calendar rows.  Within each
    window the first ``train_ratio`` fraction is the training period
    (used only for parameter selection / fitting) and the remaining
    fraction is the out-of-sample test period (used for evaluation).

    This satisfies the "rolling walk-forward periods for data-mined rules"
    requirement from Section 18 of the project brief.

    Args:
        df:          DataFrame with a DatetimeIndex (e.g. the backtest df).
        n_splits:    Number of consecutive windows (default 4).
        train_ratio: Fraction of each window allocated to training
                     (default 0.6 → 60 % train / 40 % test).

    Returns:
        List of dicts, one per split:
            {
                "split":       int   (1-based index),
                "train_start": pd.Timestamp,
                "train_end":   pd.Timestamp,
                "test_start":  pd.Timestamp,
                "test_end":    pd.Timestamp,
                "train_rows":  int,
                "test_rows":   int,
            }
    """
    n = len(df)
    window_size = n // n_splits
    splits = []

    for i in range(n_splits):
        w_start = i * window_size
        # Last split absorbs any leftover rows
        w_end   = (i + 1) * window_size if i < n_splits - 1 else n

        train_end_pos = w_start + int((w_end - w_start) * train_ratio)

        splits.append({
            "split":       i + 1,
            "train_start": df.index[w_start],
            "train_end":   df.index[train_end_pos - 1],
            "test_start":  df.index[train_end_pos],
            "test_end":    df.index[w_end - 1],
            "train_rows":  train_end_pos - w_start,
            "test_rows":   w_end - train_end_pos,
        })

    return splits


def run_walk_forward_backtest(
    price_df: pd.DataFrame,
    strategy_config: dict,
    n_splits: int = 4,
    cost_bps: float = 10,
) -> dict:
    """
    Run a walk-forward backtest: evaluate the strategy on each split's TEST
    period only (train rows are reserved for parameter selection and never
    used for performance measurement).

    For each test window the function:
        1. Slices ``price_df`` to the test period.
        2. Applies ``generate_signal`` with the provided ``strategy_config``.
        3. Runs ``run_backtest`` to obtain equity + returns.
        4. Deducts transaction costs via ``add_transaction_costs``.
        5. Computes Sharpe, Cumulative Return, Max Drawdown, Win Rate.

    Cross-split consistency is then summarised: a high standard deviation
    of Sharpe ratios across splits is a signal of overfitting.

    Args:
        price_df:        Full price DataFrame with 'Signal', 'Close', and
                         score columns already attached (output of
                         ``build_full_pipeline``).
        strategy_config: Dict with keys buy_threshold, sell_threshold,
                         tech_weight, fund_weight.  Passed to
                         ``generate_signal``.
        n_splits:        Number of walk-forward windows (default 4).
        cost_bps:        Transaction cost per trade leg in basis points.

    Returns:
        {
            "splits":      list[dict]  per-split metrics,
            "summary":     dict        mean/std of Sharpe, CumReturn across splits,
            "consistency": str         qualitative overfitting flag,
        }
    """
    from strategy import generate_signal   # avoid circular import at module level

    splits      = walk_forward_split(price_df, n_splits=n_splits)
    split_rows  = []

    for sp in splits:
        test_df = price_df.loc[sp["test_start"]:sp["test_end"]].copy()

        if len(test_df) < 10:
            split_rows.append({
                "split":             sp["split"],
                "test_start":        str(sp["test_start"].date()),
                "test_end":          str(sp["test_end"].date()),
                "test_rows":         sp["test_rows"],
                "cumulative_return": float("nan"),
                "sharpe_ratio":      float("nan"),
                "max_drawdown":      float("nan"),
                "win_rate":          float("nan"),
                "turnover":          0,
                "signal_count":      0,
            })
            continue

        # Re-apply signal to the test slice (scores already computed)
        test_df["Total_Score"] = (
            strategy_config.get("tech_weight", 0.5) * test_df["technical_score"]
            + strategy_config.get("fund_weight", 0.5) * test_df["fundamental_score"]
        )
        test_df = generate_signal(
            test_df,
            buy_threshold  = strategy_config.get("buy_threshold",  3.0),
            sell_threshold = strategy_config.get("sell_threshold", 1.5),
        )

        bt = run_backtest(test_df)

        # Transaction-cost adjustment
        adj_returns, turnover = add_transaction_costs(
            bt["strategy_return"], bt["position"], cost_bps=cost_bps
        )

        # Rebuild equity from cost-adjusted returns
        adj_equity = (1 + adj_returns.fillna(0)).cumprod()

        cum_ret  = float(adj_equity.iloc[-1] - 1)
        sharpe   = _sharpe_ratio(adj_returns)
        mdd      = _max_drawdown(adj_equity)
        wr       = _win_rate(adj_returns, bt["position"])
        n_sigs   = int((test_df["Signal"] == "BUY").sum())

        split_rows.append({
            "split":             sp["split"],
            "test_start":        str(sp["test_start"].date()),
            "test_end":          str(sp["test_end"].date()),
            "test_rows":         sp["test_rows"],
            "cumulative_return": cum_ret,
            "sharpe_ratio":      sharpe,
            "max_drawdown":      mdd,
            "win_rate":          wr,
            "turnover":          turnover,
            "signal_count":      n_sigs,
        })

    # Cross-split consistency
    sharpes  = [r["sharpe_ratio"] for r in split_rows if not np.isnan(r["sharpe_ratio"])]
    cum_rets = [r["cumulative_return"] for r in split_rows if not np.isnan(r["cumulative_return"])]

    sharpe_std = float(np.std(sharpes)) if len(sharpes) >= 2 else float("nan")
    # Heuristic: std > 0.5 across splits suggests overfitting
    if np.isnan(sharpe_std):
        consistency = "insufficient data"
    elif sharpe_std > 0.5:
        consistency = "HIGH VARIANCE — possible overfit (Sharpe std={:.2f})".format(sharpe_std)
    else:
        consistency = "stable (Sharpe std={:.2f})".format(sharpe_std)

    return {
        "splits": split_rows,
        "summary": {
            "mean_sharpe":       float(np.mean(sharpes))  if sharpes  else float("nan"),
            "std_sharpe":        sharpe_std,
            "mean_cum_return":   float(np.mean(cum_rets)) if cum_rets else float("nan"),
            "std_cum_return":    float(np.std(cum_rets))  if len(cum_rets) >= 2 else float("nan"),
        },
        "consistency": consistency,
    }


def flag_low_signal_strategies(
    signal_count: int,
    min_threshold: int = 10,
) -> str | None:
    """
    Warn when a strategy's backtest rests on very few signals.

    A high return derived from fewer than ``min_threshold`` trade signals
    is statistically unreliable: the result may be driven by one or two
    lucky trades rather than a repeatable edge.  This check implements the
    "a high return from very few events should be flagged" safeguard from
    Section 18 of the project brief.

    Args:
        signal_count:  Total number of BUY (or actionable) signals observed
                       in the backtest window.
        min_threshold: Minimum number of signals below which a warning is
                       emitted (default 10).

    Returns:
        A warning string when signal_count < min_threshold, else None.
    """
    if signal_count < min_threshold:
        return (
            f"LOW-SIGNAL WARNING: only {signal_count} BUY signal(s) generated "
            f"(minimum recommended: {min_threshold}). "
            f"Performance metrics may not be statistically reliable."
        )
    return None


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

    # ── NEW: transaction cost comparison ──────────────────────────────
    SEP = "=" * 65
    print(f"\n\n{SEP}")
    print("TRANSACTION COST ANALYSIS (10 bps per trade leg)")
    print(SEP)

    cost_bps = 10
    adj_returns, turnover = add_transaction_costs(
        bt["strategy_return"], bt["position"], cost_bps=cost_bps
    )
    adj_equity = (1 + adj_returns.fillna(0)).cumprod()

    cum_before = float(bt["equity"].iloc[-1] - 1)
    cum_after  = float(adj_equity.iloc[-1] - 1)
    sharpe_before = _sharpe_ratio(bt["strategy_return"])
    sharpe_after  = _sharpe_ratio(adj_returns)

    print(f"  Turnover (trade legs)        : {turnover}")
    print(f"  Cost per leg                 : {cost_bps} bps ({cost_bps/100:.2f}%)")
    print(f"  Total cost drag              : {turnover * cost_bps / 100:.2f}%  "
          f"(= {turnover} × {cost_bps/100:.2f}%)")
    print(f"  Cumulative return  pre-cost  : {cum_before*100:+.2f}%")
    print(f"  Cumulative return  post-cost : {cum_after*100:+.2f}%")
    print(f"  Difference                   : {(cum_after - cum_before)*100:+.2f}%")
    print(f"  Sharpe ratio       pre-cost  : {sharpe_before:.4f}")
    print(f"  Sharpe ratio       post-cost : {sharpe_after:.4f}")

    # Low-signal check on full window
    n_buy_signals = int((signal_df["Signal"] == "BUY").sum())
    low_sig_warn  = flag_low_signal_strategies(n_buy_signals, min_threshold=10)
    print(f"\n  BUY signals in full window   : {n_buy_signals}")
    if low_sig_warn:
        print(f"  ⚠  {low_sig_warn}")
    else:
        print(f"  Signal count OK (≥ 10).")

    # ── NEW: walk-forward backtest ─────────────────────────────────────
    print(f"\n\n{SEP}")
    print("WALK-FORWARD BACKTEST  (4 splits, 60/40 train-test, 10 bps cost)")
    print(SEP)

    from config import BUY_THRESHOLD, SELL_THRESHOLD, TECH_WEIGHT, FUND_WEIGHT
    strategy_cfg = {
        "buy_threshold":  BUY_THRESHOLD,
        "sell_threshold": SELL_THRESHOLD,
        "tech_weight":    TECH_WEIGHT,
        "fund_weight":    FUND_WEIGHT,
    }

    wf = run_walk_forward_backtest(signal_df, strategy_cfg,
                                   n_splits=4, cost_bps=cost_bps)

    # Per-split table
    hdr = f"{'Split':<7} {'Test window':<26} {'Rows':<6} {'CumRet':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7} {'BUY#':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in wf["splits"]:
        window = f"{r['test_start']} → {r['test_end']}"
        cum    = f"{r['cumulative_return']*100:+.2f}%" if not np.isnan(r["cumulative_return"]) else "   n/a"
        sh     = f"{r['sharpe_ratio']:+.3f}"           if not np.isnan(r["sharpe_ratio"])      else "   n/a"
        mdd    = f"{r['max_drawdown']*100:.2f}%"       if not np.isnan(r["max_drawdown"])      else "   n/a"
        wr     = f"{r['win_rate']*100:.1f}%"           if not np.isnan(r["win_rate"])           else "   n/a"
        print(f"{r['split']:<7} {window:<26} {r['test_rows']:<6} {cum:>8} {sh:>8} {mdd:>8} {wr:>8} {r['turnover']:>7} {r['signal_count']:>6}")

        # Per-split low-signal flag
        w = flag_low_signal_strategies(r["signal_count"], min_threshold=10)
        if w:
            print(f"        ⚠  {w}")

    # Summary
    s = wf["summary"]
    print(f"\n  Mean Sharpe across splits    : {s['mean_sharpe']:+.4f}")
    print(f"  Std  Sharpe across splits    : {s['std_sharpe']:.4f}")
    print(f"  Mean CumReturn across splits : {s['mean_cum_return']*100:+.2f}%")
    print(f"  Std  CumReturn across splits : {s['std_cum_return']*100:.2f}%")
    print(f"\n  Consistency verdict          : {wf['consistency']}")
