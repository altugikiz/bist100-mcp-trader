"""
Sensitivity analysis module for BIST100 MCP Trader.
Tests how changes in signal thresholds and score weights affect backtest performance.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from strategy import build_full_pipeline, combine_scores, generate_signal
from backtest import run_backtest, calculate_performance


# ---------------------------------------------------------------------------
# Shared pipeline data (fetched once, reused across all parameter sweeps)
# ---------------------------------------------------------------------------

def _load_base_data(start: str = "2024-01-01", end: str = "2025-12-31") -> tuple:
    """
    Build the full pipeline once and return the raw scored DataFrame plus the
    component DataFrames needed to reconstruct Total_Score with different weights.

    Returns:
        (signal_df, tech_df, fund_df)
        signal_df : full combined DataFrame from build_full_pipeline (default weights)
        tech_df   : same df — technical_score column is the only piece needed
        fund_df   : same df — fundamental_score column is the only piece needed
    """
    df = build_full_pipeline(start=start, end=end)
    return df


# ---------------------------------------------------------------------------
# Helper: recompute Total_Score + signal for a given weight / threshold combo
# ---------------------------------------------------------------------------

def _run_scenario(
    base_df: pd.DataFrame,
    tech_w: float,
    fund_w: float,
    macro_w: float,
    buy_thr: float,
    sell_thr: float,
) -> dict:
    """
    Given a base DataFrame with technical_score, fundamental_score, macro_score,
    recompute Total_Score, apply signal thresholds, run backtest, return metrics.
    """
    import numpy as np

    df = base_df.copy()
    df["Total_Score"] = (
        tech_w  * df["technical_score"]
        + fund_w  * df["fundamental_score"]
        + macro_w * df["macro_score"]
    )

    df = generate_signal(df, buy_threshold=buy_thr, sell_threshold=sell_thr)
    bt  = run_backtest(df)
    m   = calculate_performance(bt)

    return {
        "tech_w":      tech_w,
        "fund_w":      fund_w,
        "buy_thr":     buy_thr,
        "sell_thr":    sell_thr,
        "days_invested":     m["days_invested"],
        "cumulative_return": m["cumulative_return"],
        "sharpe_ratio":      m["sharpe_ratio"],
        "max_drawdown":      m["max_drawdown"],
    }


# ---------------------------------------------------------------------------
# Test 1 — BUY threshold sweep (fixed weights, variable buy threshold)
# ---------------------------------------------------------------------------

def threshold_sensitivity(
    base_df: pd.DataFrame,
    buy_thresholds: list[float] = (3.0, 3.2, 3.5, 3.8),
    sell_threshold: float = 1.5,
    tech_w: float = 0.50,
    fund_w: float = 0.40,
    macro_w: float = 0.10,
) -> pd.DataFrame:
    """
    Run the backtest for each BUY threshold while keeping all other parameters
    constant.  Returns a summary DataFrame sorted by buy threshold.

    Args:
        base_df:        Output of _load_base_data() with pre-computed indicator scores.
        buy_thresholds: List of Total_Score values that trigger a BUY signal.
        sell_threshold: Fixed Total_Score value at or below which SELL is triggered.
        tech_w / fund_w / macro_w: Score weights (must sum to 1.0).

    Returns:
        DataFrame with one row per threshold and columns:
            buy_thr, sell_thr, days_invested,
            cumulative_return, sharpe_ratio, max_drawdown
    """
    records = []
    for thr in buy_thresholds:
        r = _run_scenario(base_df, tech_w, fund_w, macro_w, thr, sell_threshold)
        records.append(r)
    return pd.DataFrame(records).set_index("buy_thr")


# ---------------------------------------------------------------------------
# Test 2 — Weight sensitivity (fixed thresholds, variable weights)
# ---------------------------------------------------------------------------

WEIGHT_CONFIGS = {
    "Current (0.50T+0.40F+0.10M)": (0.50, 0.40, 0.10),
    "Alt A — Tech-heavy (0.70T+0.30F)": (0.70, 0.30, 0.00),
    "Alt B — Fund-heavy (0.30T+0.70F)": (0.30, 0.70, 0.00),
    "Alt C — Equal (0.50T+0.50F)":      (0.50, 0.50, 0.00),
}


def weight_sensitivity(
    base_df: pd.DataFrame,
    buy_threshold: float = 3.5,
    sell_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Run the backtest for each weight configuration while keeping signal thresholds
    constant.  Returns a summary DataFrame indexed by configuration name.

    Args:
        base_df:       Output of _load_base_data().
        buy_threshold: Fixed BUY threshold.
        sell_threshold: Fixed SELL threshold.

    Returns:
        DataFrame with one row per weight config and columns:
            tech_w, fund_w, days_invested,
            cumulative_return, sharpe_ratio, max_drawdown
    """
    records = []
    for name, (tw, fw, mw) in WEIGHT_CONFIGS.items():
        r = _run_scenario(base_df, tw, fw, mw, buy_threshold, sell_threshold)
        r["config"] = name
        records.append(r)
    df = pd.DataFrame(records).set_index("config")
    return df


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _fmt_row(label: str, row: pd.Series, col_w: int = 22) -> str:
    days    = f"{int(row['days_invested']):>{col_w}}"
    cr      = f"{row['cumulative_return']*100:>{col_w-1}.2f}%"
    sharpe  = f"{row['sharpe_ratio']:>{col_w}.4f}"
    mdd     = f"{row['max_drawdown']*100:>{col_w-1}.2f}%"
    return f"{label:<38}{days}{cr}{sharpe}{mdd}"


def print_sensitivity_table(results_df: pd.DataFrame, title: str) -> None:
    """Pretty-print a sensitivity result DataFrame."""
    col_w = 22
    header = (f"{'Config / Threshold':<38}"
              f"{'Days Invested':>{col_w}}"
              f"{'Cum. Return':>{col_w}}"
              f"{'Sharpe Ratio':>{col_w}}"
              f"{'Max Drawdown':>{col_w}}")
    sep = "=" * len(header)

    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)

    best_sharpe = results_df["sharpe_ratio"].idxmax()
    best_return = results_df["cumulative_return"].idxmax()

    for idx, row in results_df.iterrows():
        label = str(idx)
        line  = _fmt_row(label, row, col_w)
        tags  = []
        if idx == best_sharpe:
            tags.append("← best Sharpe")
        if idx == best_return:
            tags.append("← best Return")
        print(line + ("  " + " / ".join(tags) if tags else ""))

    print(sep)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_threshold_sensitivity(
    results_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plot cumulative return and Sharpe ratio against BUY threshold on a dual-axis
    chart and save to PNG.
    """
    x      = results_df.index.astype(float)
    cr     = results_df["cumulative_return"] * 100
    sharpe = results_df["sharpe_ratio"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_cr     = "#2196F3"
    color_sharpe = "#E91E63"

    ax1.plot(x, cr, marker="o", color=color_cr, linewidth=2, label="Cumulative Return (%)")
    ax1.set_xlabel("BUY Signal Threshold (Total Score)", fontsize=12)
    ax1.set_ylabel("Cumulative Return (%)", color=color_cr, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_cr)
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    ax2.plot(x, sharpe, marker="s", color=color_sharpe, linewidth=2,
             linestyle="--", label="Sharpe Ratio")
    ax2.set_ylabel("Sharpe Ratio", color=color_sharpe, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_sharpe)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title(
        "Sensitivity: BUY Threshold vs Cumulative Return & Sharpe Ratio\n"
        "(ASELS.IS, 2024-01-01 → 2025-12-31, SELL threshold fixed at 1.5)",
        fontsize=12,
    )
    ax1.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved → {os.path.abspath(output_path)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading base pipeline data (fetched once) …")
    base_df = _load_base_data()

    # --- Test 1: threshold sweep ---
    print("\nRunning threshold sensitivity …")
    thr_results = threshold_sensitivity(base_df)
    print_sensitivity_table(thr_results, "TEST 1 — BUY Threshold Sensitivity (weights: 0.50T / 0.40F / 0.10M, SELL=1.5)")

    # --- Test 2: weight sweep ---
    print("\nRunning weight sensitivity …")
    wgt_results = weight_sensitivity(base_df)
    print_sensitivity_table(wgt_results, "TEST 2 — Weight Sensitivity (BUY=3.5, SELL=1.5)")

    # --- Chart ---
    output = os.path.join(
        os.path.dirname(__file__), "..", "results", "sensitivity_threshold.png"
    )
    plot_threshold_sensitivity(thr_results, output)
