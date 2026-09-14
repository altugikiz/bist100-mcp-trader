"""
Support and resistance detection via pivot highs/lows.
Uses scipy.signal.argrelextrema to locate local extrema in the Close price series.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import argrelextrema


def find_support_resistance(
    df: pd.DataFrame, order: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify pivot high and pivot low points in the Close price series.

    A pivot high is a local maximum: Close[i] is greater than the `order`
    candles on each side.  A pivot low is the symmetric local minimum.

    Args:
        df:    OHLCV DataFrame with a 'Close' column and a DatetimeIndex.
        order: Number of candles on each side required to confirm a pivot.
               Higher values → fewer, more significant pivots.

    Returns:
        pivot_highs: DataFrame subset at resistance pivot indices.
        pivot_lows:  DataFrame subset at support pivot indices.
    """
    close = df["Close"].values

    high_idx = argrelextrema(close, np.greater, order=order)[0]
    low_idx  = argrelextrema(close, np.less,    order=order)[0]

    pivot_highs = df.iloc[high_idx].copy()
    pivot_lows  = df.iloc[low_idx].copy()

    return pivot_highs, pivot_lows


def plot_support_resistance(
    df: pd.DataFrame,
    pivot_highs: pd.DataFrame,
    pivot_lows: pd.DataFrame,
    output_path: str = "../results/support_resistance.png",
) -> None:
    """
    Plot the Close price with pivot highs (red triangles) and pivot lows
    (green triangles) overlaid.  Saves the figure to `output_path`.
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.2, label="Close")

    ax.scatter(
        pivot_highs.index,
        pivot_highs["Close"],
        marker="v",
        color="red",
        s=80,
        zorder=5,
        label=f"Resistance pivots ({len(pivot_highs)})",
    )
    ax.scatter(
        pivot_lows.index,
        pivot_lows["Close"],
        marker="^",
        color="green",
        s=80,
        zorder=5,
        label=f"Support pivots ({len(pivot_lows)})",
    )

    # Horizontal dashed lines at each pivot level for visual guidance
    for price in pivot_highs["Close"]:
        ax.axhline(price, color="red", linewidth=0.4, linestyle="--", alpha=0.35)
    for price in pivot_lows["Close"]:
        ax.axhline(price, color="green", linewidth=0.4, linestyle="--", alpha=0.35)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    ax.set_title("ASELS.IS — Support & Resistance Pivots (order=5)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (TRY)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved → {os.path.abspath(output_path)}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_collection import fetch_ohlcv

    df = fetch_ohlcv("ASELS.IS", "2023-01-01", "2026-01-01")

    pivot_highs, pivot_lows = find_support_resistance(df, order=5)

    print(f"Pivot highs (resistance): {len(pivot_highs)}")
    print(pivot_highs[["Close"]].round(2).to_string())

    print(f"\nPivot lows (support): {len(pivot_lows)}")
    print(pivot_lows[["Close"]].round(2).to_string())

    output = os.path.join(os.path.dirname(__file__), "..", "results", "support_resistance.png")
    plot_support_resistance(df, pivot_highs, pivot_lows, output_path=output)
