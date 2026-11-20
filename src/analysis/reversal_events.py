"""
Reversal event detection for BIST100 MCP Trader.

Each detect_* function operates on a fully-enriched DataFrame that already
contains the indicator columns produced by src/indicators.py.  The caller is
responsible for computing those columns before calling these functions.

Required indicator columns (add via indicators.py before calling):
    BB_Upper, BB_Middle, BB_Lower    — calculate_bollinger_bands()
    RSI14                            — calculate_rsi(window=14)
    Supertrend, Supertrend_Direction — calculate_supertrend()
    KAMA, KAMA_Slope                 — calculate_kama()
    Ichimoku_State                   — calculate_ichimoku()
    Volume                           — raw OHLCV
"""

import os
import numpy as np
import pandas as pd
from typing import List


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _context_columns(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Extract shared context columns for a set of event row indices.
    Returns a DataFrame aligned to `idx`.
    """
    sub = df.loc[idx].copy()

    vol_avg = df["Volume"].rolling(20).mean()
    sub["volume_regime"] = np.where(
        df.loc[idx, "Volume"] > vol_avg.loc[idx], "high", "normal"
    )
    sub["rsi_state"] = np.where(
        df.loc[idx, "RSI14"] < 35, "trough_zone",
        np.where(df.loc[idx, "RSI14"] > 65, "peak_zone", "neutral"),
    )
    sub["supertrend_state"] = df.loc[idx, "Supertrend_Direction"]
    sub["kama_slope"]       = df.loc[idx, "KAMA_Slope"]
    sub["ichimoku_state"]   = df.loc[idx, "Ichimoku_State"]
    return sub


def _base_record(df: pd.DataFrame, mask: pd.Series, event_type: str) -> pd.DataFrame:
    """
    Build a minimal event record DataFrame for rows where mask is True.
    Returns empty DataFrame if no events match.
    """
    idx = df.index[mask]
    if idx.empty:
        return pd.DataFrame()

    ctx = _context_columns(df, idx)
    records = pd.DataFrame({
        "event_date":       idx,
        "event_type":       event_type,
        "close_price":      df.loc[idx, "Close"].values,
        "rsi_state":        ctx["rsi_state"].values,
        "supertrend_state": ctx["supertrend_state"].values,
        "kama_slope":       ctx["kama_slope"].values,
        "ichimoku_state":   ctx["ichimoku_state"].values,
        "volume_regime":    ctx["volume_regime"].values,
    })
    return records


# ---------------------------------------------------------------------------
# 1. Bollinger Band events
# ---------------------------------------------------------------------------

def detect_bollinger_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect three Bollinger Band event types:

    - lower_band_touch : Close <= BB_Lower
    - middle_band_test : |Close - BB_Middle| / BB_Middle <= 0.01
    - upper_band_touch : Close >= BB_Upper
    """
    parts = []

    lower_mask  = df["Close"] <= df["BB_Lower"]
    middle_mask = ((df["Close"] - df["BB_Middle"]).abs() / df["BB_Middle"]) <= 0.01
    upper_mask  = df["Close"] >= df["BB_Upper"]

    for mask, etype in [
        (lower_mask,  "lower_band_touch"),
        (middle_mask, "middle_band_test"),
        (upper_mask,  "upper_band_touch"),
    ]:
        rec = _base_record(df, mask & df["BB_Upper"].notna(), etype)
        if not rec.empty:
            parts.append(rec)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. RSI local extrema events
# ---------------------------------------------------------------------------

def detect_rsi_events(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Detect RSI local troughs and peaks.

    A local trough : RSI value is lower than the 3 preceding and 3 following days.
    A local peak   : RSI value is higher than the 3 preceding and 3 following days.
    """
    rsi = df["RSI14"]
    n   = 3
    parts = []

    for etype, cmp in [("rsi_trough", "min"), ("rsi_peak", "max")]:
        if cmp == "min":
            mask = (
                (rsi < rsi.shift(1)) & (rsi < rsi.shift(2)) & (rsi < rsi.shift(3)) &
                (rsi < rsi.shift(-1)) & (rsi < rsi.shift(-2)) & (rsi < rsi.shift(-3))
            )
        else:
            mask = (
                (rsi > rsi.shift(1)) & (rsi > rsi.shift(2)) & (rsi > rsi.shift(3)) &
                (rsi > rsi.shift(-1)) & (rsi > rsi.shift(-2)) & (rsi > rsi.shift(-3))
            )

        rec = _base_record(df, mask & rsi.notna(), etype)
        if not rec.empty:
            # Replace generic rsi_state with the actual RSI value for context
            rec["rsi_value"] = rsi[df.index[mask & rsi.notna()]].values
            parts.append(rec)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. Supertrend direction flips
# ---------------------------------------------------------------------------

def detect_supertrend_flips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect days where Supertrend_Direction changes.

    Event types:
    - supertrend_flip_bullish : bearish → bullish transition
    - supertrend_flip_bearish : bullish → bearish transition
    """
    sd = df["Supertrend_Direction"]
    prev = sd.shift(1)

    flip_bull = (prev == "bearish") & (sd == "bullish")
    flip_bear = (prev == "bullish") & (sd == "bearish")

    parts = []
    for mask, etype in [
        (flip_bull, "supertrend_flip_bullish"),
        (flip_bear, "supertrend_flip_bearish"),
    ]:
        rec = _base_record(df, mask & sd.notna(), etype)
        if not rec.empty:
            parts.append(rec)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. KAMA crossover events
# ---------------------------------------------------------------------------

def detect_kama_cross_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect days when Close crosses the KAMA line.

    - bullish_kama_cross : Close was below KAMA yesterday, above today
    - bearish_kama_cross : Close was above KAMA yesterday, below today
    """
    close = df["Close"]
    kama  = df["KAMA"]

    bullish = (close.shift(1) < kama.shift(1)) & (close > kama)
    bearish = (close.shift(1) > kama.shift(1)) & (close < kama)

    parts = []
    for mask, etype in [
        (bullish, "bullish_kama_cross"),
        (bearish, "bearish_kama_cross"),
    ]:
        rec = _base_record(df, mask & kama.notna(), etype)
        if not rec.empty:
            parts.append(rec)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# 5. Ichimoku cloud transition events
# ---------------------------------------------------------------------------

def detect_ichimoku_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Ichimoku_State transition events.

    - cloud_breakout  : any non-above_cloud state → above_cloud
    - cloud_rejection : above_cloud → any non-above_cloud state
    - cloud_entry     : below_cloud → in_cloud (price re-enters cloud)
    - cloud_breakdown : in_cloud or above_cloud → below_cloud
    """
    state = df["Ichimoku_State"]
    prev  = state.shift(1)

    transitions = {
        "cloud_breakout":  (prev != "above_cloud") & (state == "above_cloud"),
        "cloud_rejection": (prev == "above_cloud") & (state != "above_cloud"),
        "cloud_entry":     (prev == "below_cloud") & (state == "in_cloud"),
        "cloud_breakdown": (state == "below_cloud") & (prev != "below_cloud"),
    }

    parts = []
    for etype, mask in transitions.items():
        rec = _base_record(df, mask & state.notna() & prev.notna(), etype)
        if not rec.empty:
            parts.append(rec)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# 6. Forward returns
# ---------------------------------------------------------------------------

def calculate_forward_returns(
    df: pd.DataFrame,
    event_dates: pd.DatetimeIndex,
    horizons: List[int] = [1, 3, 5, 10],
) -> pd.DataFrame:
    """
    For each event date, compute forward returns at each horizon.

    forward_return_Nd = (Close[t+N] - Close[t]) / Close[t]

    Rows where t+N falls outside the available data get NaN.
    No look-ahead bias: uses only t+N close, not intraday data.

    Args:
        df:          OHLCV DataFrame with DatetimeIndex.
        event_dates: DatetimeIndex of event dates to compute returns for.
        horizons:    List of forward horizon lengths in trading days.

    Returns:
        DataFrame indexed like event_dates with columns forward_return_Nd.
    """
    close = df["Close"]
    all_dates = df.index
    records = []

    for ed in event_dates:
        if ed not in all_dates:
            records.append({f"forward_return_{h}d": np.nan for h in horizons})
            continue

        pos = all_dates.get_loc(ed)
        base = close.iloc[pos]
        row = {}
        for h in horizons:
            target_pos = pos + h
            if target_pos < len(all_dates):
                row[f"forward_return_{h}d"] = (close.iloc[target_pos] - base) / base
            else:
                row[f"forward_return_{h}d"] = np.nan
        records.append(row)

    return pd.DataFrame(records, index=event_dates)


# ---------------------------------------------------------------------------
# 6b. Support / resistance touch events and KAMA slope-change events
# ---------------------------------------------------------------------------

def detect_support_resistance_events(df: pd.DataFrame, order: int = 5, tol: float = 0.01) -> pd.DataFrame:
    """
    Detect first touches of confirmed support and resistance levels.

    Levels are pivot lows/highs of the Close series (a pivot needs `order` bars on each side).  A pivot at
    bar i is only known from bar i + order, so a level is used strictly after its confirmation: no look-ahead.

    - support_touch    : Low comes within `tol` of a confirmed support level (pivot low) after the previous
                         close was still more than `tol` above it
    - resistance_touch : High comes within `tol` of a confirmed resistance level (pivot high) after the
                         previous close was still more than `tol` below it

    Extra columns: level_price, level_age_days (calendar days since the pivot), prior_touches (earlier
    touches of the same level).  Reversal strength is read from the forward returns: after a support touch a
    rise is the reversal, after a resistance touch a fall is the reversal.
    """
    close = df["Close"].values.astype(float)
    low = df["Low"].values.astype(float)
    high = df["High"].values.astype(float)
    n = len(df)
    if n < 2 * order + 3:
        return pd.DataFrame()

    pivot_low, pivot_high = [], []                 # (pivot index, price)
    touches = {}                                   # (kind, pivot index) -> number of touches so far
    recs = []
    for t in range(1, n):
        i = t - 1 - order                          # newest pivot candidate confirmed by bar t-1
        if i >= order:
            win = close[i - order:i + order + 1]
            if close[i] == win.min() and (win == close[i]).sum() == 1:
                pivot_low.append((i, close[i]))
            if close[i] == win.max() and (win == close[i]).sum() == 1:
                pivot_high.append((i, close[i]))
        for kind, pivots in (("support", pivot_low), ("resistance", pivot_high)):
            best = None
            for pi, lvl in pivots:
                if kind == "support":
                    near = abs(low[t] - lvl) / lvl <= tol or (low[t] < lvl and close[t] >= lvl * (1 - tol))
                    fresh = close[t - 1] > lvl * (1 + tol)
                else:
                    near = abs(high[t] - lvl) / lvl <= tol or (high[t] > lvl and close[t] <= lvl * (1 + tol))
                    fresh = close[t - 1] < lvl * (1 - tol)
                if near and fresh and t - pi > order:
                    d = abs((low[t] if kind == "support" else high[t]) - lvl) / lvl
                    if best is None or d < best[0]:
                        best = (d, pi, lvl)
            if best is not None:
                _, pi, lvl = best
                key = (kind, pi)
                recs.append({"date": df.index[t], "type": kind + "_touch", "level_price": lvl,
                             "level_age_days": (df.index[t] - df.index[pi]).days,
                             "prior_touches": touches.get(key, 0)})
                touches[key] = touches.get(key, 0) + 1
    if not recs:
        return pd.DataFrame()
    parts = []
    for etype in ("support_touch", "resistance_touch"):
        sub = [r for r in recs if r["type"] == etype]
        if not sub:
            continue
        mask = pd.Series(False, index=df.index)
        mask.loc[[r["date"] for r in sub]] = True
        rec = _base_record(df, mask, etype)
        extra = pd.DataFrame(sub).set_index("date").loc[rec["event_date"]]
        rec["level_price"] = extra["level_price"].values
        rec["level_age_days"] = extra["level_age_days"].values
        rec["prior_touches"] = extra["prior_touches"].values
        parts.append(rec)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def detect_kama_slope_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect KAMA slope-state changes (adaptive trend persistence starts here).

    - kama_slope_turn_up   : slope state becomes 'rising' from another state
    - kama_slope_turn_down : slope state becomes 'falling' from another state
    """
    slope = df["KAMA_Slope"]
    prev = slope.shift(1)
    parts = []
    for mask, etype in [((slope == "rising") & (prev != "rising") & prev.notna(), "kama_slope_turn_up"),
                        ((slope == "falling") & (prev != "falling") & prev.notna(), "kama_slope_turn_down")]:
        rec = _base_record(df, mask, etype)
        if not rec.empty:
            parts.append(rec)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# 7. Master event log builder
# ---------------------------------------------------------------------------

def build_event_log(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Run all detect_* functions (Bollinger, RSI, Supertrend, KAMA cross and slope, Ichimoku, support/resistance), attach forward returns, and return a
    unified, sorted event log DataFrame.

    Args:
        df:     Fully-enriched OHLCV DataFrame (all indicator columns present).
        ticker: Short ticker code (e.g. 'ASELS') added as a label column.

    Returns:
        DataFrame sorted by (event_type, event_date) with columns:
            ticker, event_date, event_type, close_price,
            rsi_state, supertrend_state, kama_slope, ichimoku_state, volume_regime,
            forward_return_1d, forward_return_3d, forward_return_5d, forward_return_10d
    """
    detectors = [
        detect_bollinger_events,
        detect_rsi_events,
        detect_supertrend_flips,
        detect_kama_cross_events,
        detect_ichimoku_events,
        detect_support_resistance_events,
        detect_kama_slope_events,
    ]

    parts = []
    for fn in detectors:
        result = fn(df)
        if not result.empty:
            parts.append(result)

    if not parts:
        return pd.DataFrame()

    log = pd.concat(parts, ignore_index=True)
    log["ticker"] = ticker

    # Forward returns
    event_dates_idx = pd.DatetimeIndex(log["event_date"])
    fwd = calculate_forward_returns(df, event_dates_idx)
    fwd = fwd.reset_index(drop=True)
    log = pd.concat([log.reset_index(drop=True), fwd], axis=1)

    # Sort
    log = log.sort_values(["event_type", "event_date"]).reset_index(drop=True)

    # Column order
    base_cols = [
        "ticker", "event_date", "event_type", "close_price",
        "rsi_state", "supertrend_state", "kama_slope", "ichimoku_state", "volume_regime",
    ]
    fwd_cols  = [c for c in log.columns if c.startswith("forward_return_")]
    extra     = [c for c in log.columns if c not in base_cols + fwd_cols]
    log = log[base_cols + extra + fwd_cols]

    return log


# ---------------------------------------------------------------------------
# 8. Market regime classifier (XU100-based)
# ---------------------------------------------------------------------------

def classify_market_regime(xu100_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce two independent daily regime labels from XU100 price data.

    trend_regime : "rising"   — 20-day return of XU100 Close is positive
                   "falling"  — 20-day return is negative or zero

    vol_regime   : "high_vol" — 20-day rolling std of daily returns exceeds
                                the full-series median of that rolling std
                   "low_vol"  — at or below the median

    Args:
        xu100_df: OHLCV DataFrame for XU100.IS with DatetimeIndex.

    Returns:
        DataFrame indexed by Date with columns [trend_regime, vol_regime].
    """
    close = xu100_df["Close"]
    daily_ret = close.pct_change()

    ret_20d   = close.pct_change(20)
    std_20d   = daily_ret.rolling(20).std()
    median_std = std_20d.median()

    regime = pd.DataFrame(index=xu100_df.index)
    regime["trend_regime"] = np.where(ret_20d > 0, "rising", "falling")
    regime["vol_regime"]   = np.where(std_20d > median_std, "high_vol", "low_vol")

    # Leave NaN for warm-up rows where rolling window is not yet full
    regime.loc[ret_20d.isna(), "trend_regime"] = np.nan
    regime.loc[std_20d.isna(), "vol_regime"]   = np.nan

    return regime


# ---------------------------------------------------------------------------
# 9. Universe-wide event log builder
# ---------------------------------------------------------------------------

def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all required indicators to a raw OHLCV DataFrame."""
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from indicators import (
        calculate_bollinger_bands, calculate_rsi, calculate_supertrend,
        calculate_kama, calculate_ichimoku,
    )
    df = calculate_bollinger_bands(df)
    df["RSI14"] = calculate_rsi(df, window=14)
    df = calculate_supertrend(df)
    df = calculate_kama(df)
    df = calculate_ichimoku(df)
    return df


def build_universe_event_log() -> pd.DataFrame:
    """
    Build a combined event log for all 30 universe stocks.

    Steps:
    1. For each ticker in UNIVERSE, load data/universe_cache/<TICKER>.csv,
       enrich with indicators, and run build_event_log.
    2. Fetch or load XU100.IS data, compute market regime labels via
       classify_market_regime.
    3. Merge regime labels onto every event row by event_date.
    4. Return the unified DataFrame (all tickers, all event types).

    XU100 data is cached at data/universe_cache/XU100.csv to avoid
    redundant downloads.
    """
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from universe import UNIVERSE

    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "universe_cache")
    xu100_path = os.path.join(cache_dir, "XU100.csv")

    # ── Load or fetch XU100 ────────────────────────────────────────────
    if os.path.exists(xu100_path):
        xu100_df = pd.read_csv(xu100_path, index_col="Date", parse_dates=True)
    else:
        import yfinance as yf
        raw = yf.download("XU100.IS", start="2023-01-01", end="2026-01-01",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        xu100_df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        xu100_df.index.name = "Date"
        xu100_df.to_csv(xu100_path)

    regime = classify_market_regime(xu100_df)

    # ── Per-ticker event logs ─────────────────────────────────────────
    parts = []
    failed = []
    for _, row in UNIVERSE.iterrows():
        ticker = row["ticker"]
        csv_path = os.path.join(cache_dir, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            failed.append(ticker)
            continue
        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            df = _enrich(df)
            log = build_event_log(df, ticker)
            if not log.empty:
                parts.append(log)
        except Exception as exc:
            failed.append(ticker)
            print(f"  [WARN] {ticker}: {exc}")

    if failed:
        print(f"Skipped tickers: {failed}")

    if not parts:
        return pd.DataFrame()

    universe_log = pd.concat(parts, ignore_index=True)

    # ── Merge regime labels by event_date ────────────────────────────
    universe_log["event_date"] = pd.to_datetime(universe_log["event_date"])
    universe_log = universe_log.merge(
        regime.rename_axis("event_date").reset_index(),
        on="event_date",
        how="left",
    )

    return universe_log.sort_values(["event_type", "ticker", "event_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 10. Event statistics summary
# ---------------------------------------------------------------------------

def summarize_event_statistics(event_log_df: pd.DataFrame) -> dict:
    """
    Produce grouped summary statistics for the event log.

    Returns a dict with three keys:
    - "overall"      : event_type × [count, avg_fwd5d, bounce_rate_5d]
    - "by_trend"     : event_type × trend_regime × [count, avg_fwd5d, bounce_rate_5d]
    - "by_vol"       : event_type × vol_regime   × [count, avg_fwd5d, bounce_rate_5d]
    """
    fwd_col = "forward_return_5d"

    def _agg(grp: pd.Series) -> pd.Series:
        valid = grp.dropna()
        return pd.Series({
            "count":         len(grp),
            "avg_fwd5d":     valid.mean() if len(valid) else np.nan,
            "bounce_rate_5d": (valid > 0).mean() if len(valid) else np.nan,
        })

    overall = (
        event_log_df.groupby("event_type")[fwd_col]
        .apply(_agg)
        .unstack()
        .sort_values("bounce_rate_5d", ascending=False)
    )

    by_trend = (
        event_log_df.dropna(subset=["trend_regime"])
        .groupby(["event_type", "trend_regime"])[fwd_col]
        .apply(_agg)
        .unstack()
    )

    by_vol = (
        event_log_df.dropna(subset=["vol_regime"])
        .groupby(["event_type", "vol_regime"])[fwd_col]
        .apply(_agg)
        .unstack()
    )

    return {"overall": overall, "by_trend": by_trend, "by_vol": by_vol}


# ---------------------------------------------------------------------------
# Entry point — ASELS test run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    # ── Build universe-wide event log ─────────────────────────────────
    print("Building universe event log for all 30 tickers …")
    universe_log = build_universe_event_log()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "events")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "universe_events.csv")
    universe_log.to_csv(out_path, index=False)

    n_tickers = universe_log["ticker"].nunique()
    print(f"\nTotal events    : {len(universe_log):,}")
    print(f"Unique tickers  : {n_tickers}")
    print(f"Saved           : {os.path.abspath(out_path)}")

    # ── Summary statistics ────────────────────────────────────────────
    stats = summarize_event_statistics(universe_log)

    print("\n" + "="*70)
    print("OVERALL — bounce_rate_5d ranking (all tickers, all regimes)")
    print("="*70)
    overall = stats["overall"].copy()
    overall["count"]         = overall["count"].astype(int)
    overall["avg_fwd5d_%"]   = (overall["avg_fwd5d"] * 100).round(2)
    overall["bounce_rate_%"] = (overall["bounce_rate_5d"] * 100).round(1)
    print(overall[["count", "avg_fwd5d_%", "bounce_rate_%"]].to_string())

    # ── lower_band_touch & rsi_trough: rising vs falling regime ───────
    print("\n" + "="*70)
    print("TREND REGIME SPLIT — lower_band_touch & rsi_trough")
    print("="*70)
    by_trend = stats["by_trend"]
    for etype in ["lower_band_touch", "rsi_trough"]:
        if etype not in by_trend.index.get_level_values(0):
            continue
        sub = by_trend.loc[etype].copy()
        sub = sub.loc[sub.index.isin(["rising", "falling"])]
        sub["count"]         = sub["count"].astype(int)
        sub["avg_fwd5d_%"]   = (sub["avg_fwd5d"] * 100).round(2)
        sub["bounce_rate_%"] = (sub["bounce_rate_5d"] * 100).round(1)
        print(f"\n  {etype}:")
        print(sub[["count", "avg_fwd5d_%", "bounce_rate_%"]].to_string())
        rising  = sub.loc["rising",  "bounce_rate_5d"] if "rising"  in sub.index else np.nan
        falling = sub.loc["falling", "bounce_rate_5d"] if "falling" in sub.index else np.nan
        if not (np.isnan(rising) or np.isnan(falling)):
            print(f"  → bounce_rate difference (rising − falling): "
                  f"{(rising - falling)*100:+.1f} pp")
