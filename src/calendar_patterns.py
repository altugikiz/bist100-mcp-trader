"""
Calendar and sequential pattern analysis for BIST100 MCP Trader.

Tests weekday return patterns and multi-day price sequences across the
30-stock universe, with Bonferroni correction for multiple testing.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)

_SRC       = os.path.dirname(__file__)
_CACHE_DIR = os.path.join(_SRC, "..", "data", "universe_cache")
_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

sys.path.insert(0, _SRC)
from universe import UNIVERSE


# ---------------------------------------------------------------------------
# 1. Weekday return enrichment
# ---------------------------------------------------------------------------

def calculate_weekday_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Weekday and Daily_Return columns to an OHLCV DataFrame.

    Weekday: Monday … Friday (mapped from Date.dayofweek 0-4).
    Daily_Return: Close.pct_change() — first row is NaN.
    """
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Weekday"]      = df.index.dayofweek.map(
        {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    )
    return df


# ---------------------------------------------------------------------------
# 2. Weekday pattern t-test
# ---------------------------------------------------------------------------

def test_weekday_pattern(
    ticker_df: pd.DataFrame,
    ticker_name: str,
) -> pd.DataFrame:
    """
    Test whether each weekday's mean return differs from the overall mean.

    Uses a one-sample t-test (scipy.stats.ttest_1samp) with H0: day mean =
    grand mean of all daily returns.

    Args:
        ticker_df:   OHLCV DataFrame already enriched via calculate_weekday_returns().
        ticker_name: Label added to the output (Ticker column).

    Returns:
        DataFrame with columns:
            Ticker, Weekday, N, Mean_Return, Median_Return,
            Positive_Rate, T_Stat, P_Value
        One row per weekday, sorted Monday → Friday.
    """
    df = ticker_df.dropna(subset=["Daily_Return", "Weekday"]).copy()
    grand_mean = df["Daily_Return"].mean()

    rows = []
    for day in _DAY_NAMES:
        sub = df.loc[df["Weekday"] == day, "Daily_Return"]
        if len(sub) < 5:
            continue
        t_stat, p_val = stats.ttest_1samp(sub, popmean=grand_mean)
        rows.append({
            "Ticker":        ticker_name,
            "Weekday":       day,
            "N":             len(sub),
            "Mean_Return":   sub.mean(),
            "Median_Return": sub.median(),
            "Positive_Rate": (sub > 0).mean(),
            "T_Stat":        t_stat,
            "P_Value":       p_val,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Multi-day sequence detection
# ---------------------------------------------------------------------------

def detect_multiday_sequences(
    df: pd.DataFrame,
    xu100_returns: pd.Series = None,
) -> Dict[str, dict]:
    """
    Detect three multi-day price sequence patterns and measure continuation
    statistics.

    Patterns:
    (a) three_down_then_rebound   : 3 consecutive negative days; measure avg
                                    return on days +1 and +2 after the streak.
    (b) post_large_down_reversal  : a single day with return < -3%; measure
                                    forward returns at +1, +3, +5 days.
    (c) monday_weakness_friday_strength : fraction of weeks where Monday < 0
                                    AND Friday > 0, vs. the independent random
                                    expectation (p_mon_neg × p_fri_pos).

    Args:
        df:             OHLCV DataFrame enriched via calculate_weekday_returns().
        xu100_returns:  Optional XU100 daily returns aligned to df.index; used
                        only for the large-down trigger in pattern (b). When
                        None, the ticker's own return is used instead.

    Returns:
        dict with keys "three_down_then_rebound", "post_large_down_reversal",
        "monday_weakness_friday_strength"; each value is a dict of statistics.
    """
    ret  = df["Daily_Return"].dropna()
    results: Dict[str, dict] = {}

    # ── (a) three_down_then_rebound ─────────────────────────────────
    rebounds = []
    ret_vals = ret.values
    ret_idx  = ret.index
    for i in range(2, len(ret_vals) - 2):
        if ret_vals[i] < 0 and ret_vals[i-1] < 0 and ret_vals[i-2] < 0:
            fwd = (ret_vals[i+1] + ret_vals[i+2]) / 2
            rebounds.append(fwd)

    if rebounds:
        arr = np.array(rebounds)
        results["three_down_then_rebound"] = {
            "occurrence_count":          len(arr),
            "avg_continuation_return":   float(arr.mean()),
            "hit_rate":                  float((arr > 0).mean()),
            "description": "Avg return over days +1/+2 after 3 consecutive down days",
        }

    # ── (b) post_large_down_reversal ─────────────────────────────────
    trigger_rets = xu100_returns.reindex(ret.index).fillna(ret) \
        if xu100_returns is not None else ret
    large_down_dates = trigger_rets.index[trigger_rets < -0.03]

    fwd_by_horizon = {1: [], 3: [], 5: []}
    close = df["Close"]
    for d in large_down_dates:
        if d not in close.index:
            continue
        pos = close.index.get_loc(d)
        base = close.iloc[pos]
        for h in [1, 3, 5]:
            if pos + h < len(close):
                fwd_by_horizon[h].append((close.iloc[pos + h] - base) / base)

    combined = fwd_by_horizon[1] + fwd_by_horizon[3] + fwd_by_horizon[5]
    if combined:
        all_arr = np.array(combined)
        results["post_large_down_reversal"] = {
            "occurrence_count":          len(large_down_dates),
            "avg_continuation_return":   float(all_arr.mean()),
            "hit_rate":                  float((all_arr > 0).mean()),
            "avg_fwd_1d":  float(np.mean(fwd_by_horizon[1])) if fwd_by_horizon[1] else np.nan,
            "avg_fwd_3d":  float(np.mean(fwd_by_horizon[3])) if fwd_by_horizon[3] else np.nan,
            "avg_fwd_5d":  float(np.mean(fwd_by_horizon[5])) if fwd_by_horizon[5] else np.nan,
            "hit_rate_1d": float((np.array(fwd_by_horizon[1]) > 0).mean()) if fwd_by_horizon[1] else np.nan,
            "hit_rate_3d": float((np.array(fwd_by_horizon[3]) > 0).mean()) if fwd_by_horizon[3] else np.nan,
            "hit_rate_5d": float((np.array(fwd_by_horizon[5]) > 0).mean()) if fwd_by_horizon[5] else np.nan,
            "description": "Forward returns after a ≥-3% single-day drop",
        }

    # ── (c) monday_weakness_friday_strength ──────────────────────────
    df_wk = df.dropna(subset=["Weekday", "Daily_Return"])
    mon = df_wk[df_wk["Weekday"] == "Monday"]["Daily_Return"]
    fri = df_wk[df_wk["Weekday"] == "Friday"]["Daily_Return"]

    p_mon_neg = (mon < 0).mean()
    p_fri_pos = (fri > 0).mean()
    expected  = p_mon_neg * p_fri_pos

    # Count calendar weeks where BOTH conditions hold simultaneously
    df_wk = df_wk.copy()
    df_wk["week"] = df_wk.index.to_period("W")
    mon_neg_weeks = set(df_wk[(df_wk["Weekday"] == "Monday") & (df_wk["Daily_Return"] < 0)]["week"])
    fri_pos_weeks = set(df_wk[(df_wk["Weekday"] == "Friday") & (df_wk["Daily_Return"] > 0)]["week"])
    both_weeks    = mon_neg_weeks & fri_pos_weeks
    total_weeks   = len(set(df_wk["week"]))

    observed = len(both_weeks) / total_weeks if total_weeks else np.nan
    results["monday_weakness_friday_strength"] = {
        "occurrence_count":          len(both_weeks),
        "avg_continuation_return":   float(fri[fri > 0].mean()) if (fri > 0).any() else np.nan,
        "hit_rate":                  float(observed),
        "p_monday_negative":         float(p_mon_neg),
        "p_friday_positive":         float(p_fri_pos),
        "expected_independent":      float(expected),
        "observed_cooccurrence":     float(observed),
        "lift":                      float(observed / expected) if expected else np.nan,
        "description": "Fraction of weeks: Monday<0 AND Friday>0 (lift vs. independent expectation)",
    }

    return results


# ---------------------------------------------------------------------------
# 4. Universe-wide calendar analysis
# ---------------------------------------------------------------------------

def run_universe_calendar_analysis() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run weekday pattern tests and multi-day sequence detection for all 30 tickers.

    Returns:
        (weekday_df, sequence_df)
        weekday_df:  Combined weekday t-test results (one row per ticker × weekday).
        sequence_df: Combined multi-day sequence statistics (one row per ticker × pattern).
    """
    xu100_path = os.path.join(_CACHE_DIR, "XU100.csv")
    xu100_df   = pd.read_csv(xu100_path, index_col="Date", parse_dates=True)
    xu100_rets = xu100_df["Close"].pct_change()

    weekday_parts:  List[pd.DataFrame] = []
    sequence_rows:  List[dict]         = []

    for _, row in UNIVERSE.iterrows():
        ticker = row["ticker"]
        path   = os.path.join(_CACHE_DIR, f"{ticker}.csv")
        df_raw = pd.read_csv(path, index_col="Date", parse_dates=True)
        df_enr = calculate_weekday_returns(df_raw)

        # Weekday patterns
        wd = test_weekday_pattern(df_enr, ticker)
        weekday_parts.append(wd)

        # Multi-day sequences
        seq = detect_multiday_sequences(df_enr, xu100_rets)
        for pattern_name, stats_dict in seq.items():
            sequence_rows.append({
                "Ticker":  ticker,
                "Pattern": pattern_name,
                **{k: v for k, v in stats_dict.items() if k != "description"},
                "Description": stats_dict.get("description", ""),
            })

    weekday_df  = pd.concat(weekday_parts, ignore_index=True)
    sequence_df = pd.DataFrame(sequence_rows)
    return weekday_df, sequence_df


# ---------------------------------------------------------------------------
# 5. Bonferroni multiple-testing correction
# ---------------------------------------------------------------------------

def apply_multiple_testing_correction(
    results_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply Bonferroni correction to a DataFrame that contains a P_Value column.

    corrected_alpha = alpha / n_tests

    Adds columns:
        Significant_Raw        : P_Value < alpha (uncorrected)
        Significant_Bonferroni : P_Value < corrected_alpha

    Args:
        results_df: DataFrame with a P_Value column (from test_weekday_pattern).
        alpha:      Family-wise error rate (default 0.05).

    Returns:
        results_df with the two new boolean columns appended.
    """
    df = results_df.copy()
    n_tests = len(df)
    corrected_alpha = alpha / n_tests

    df["Significant_Raw"]        = df["P_Value"] < alpha
    df["Significant_Bonferroni"] = df["P_Value"] < corrected_alpha
    df["Corrected_Alpha"]        = corrected_alpha

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.5f}".format)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print("Running universe calendar analysis …")
    weekday_df, sequence_df = run_universe_calendar_analysis()

    # Apply Bonferroni correction
    weekday_corrected = apply_multiple_testing_correction(weekday_df, alpha=0.05)
    n_tests     = len(weekday_corrected)
    n_raw       = weekday_corrected["Significant_Raw"].sum()
    n_corrected = weekday_corrected["Significant_Bonferroni"].sum()
    corr_alpha  = weekday_corrected["Corrected_Alpha"].iloc[0]

    print("\n" + "=" * 65)
    print("WEEKDAY PATTERN — MULTIPLE TESTING SUMMARY")
    print("=" * 65)
    print(f"Total ticker × weekday tests : {n_tests}  (30 tickers × 5 days)")
    print(f"Bonferroni corrected alpha   : {corr_alpha:.5f}  (0.05 / {n_tests})")
    print(f"Significant before correction (p < 0.05) : {n_raw}")
    print(f"Significant after  correction            : {n_corrected}")

    if n_raw > 0:
        print(f"\nRaw significant (p < 0.05) — top results by p-value:")
        raw_sig = (
            weekday_corrected[weekday_corrected["Significant_Raw"]]
            .sort_values("P_Value")
            [["Ticker", "Weekday", "N", "Mean_Return", "Positive_Rate",
              "T_Stat", "P_Value", "Significant_Bonferroni"]]
        )
        # Format mean return as %
        display = raw_sig.copy()
        display["Mean_Return_%"] = (display["Mean_Return"] * 100).round(3)
        display["Positive_Rate_%"] = (display["Positive_Rate"] * 100).round(1)
        print(display[["Ticker", "Weekday", "N", "Mean_Return_%",
                        "Positive_Rate_%", "T_Stat", "P_Value",
                        "Significant_Bonferroni"]].to_string(index=False))

    if n_corrected > 0:
        print(f"\nBonferroni survivors — effect sizes:")
        survivors = weekday_corrected[weekday_corrected["Significant_Bonferroni"]] \
            .sort_values("P_Value")
        for _, r in survivors.iterrows():
            print(f"  {r['Ticker']:6s} {r['Weekday']:10s}  "
                  f"mean={r['Mean_Return']*100:+.3f}%  "
                  f"pos_rate={r['Positive_Rate']*100:.1f}%  "
                  f"p={r['P_Value']:.5f}")
    else:
        print("\nNo weekday effect survives Bonferroni correction.")
        print("This is consistent with weak-form market efficiency — apparent")
        print("weekday patterns in this universe are likely false discoveries")
        print("arising from 150 simultaneous hypothesis tests.")

    # ── Multi-day sequence summary ────────────────────────────────────
    print("\n" + "=" * 65)
    print("MULTI-DAY SEQUENCE PATTERNS — UNIVERSE SUMMARY")
    print("=" * 65)

    MIN_OCC = 10   # minimum occurrences for a pattern to be considered reliable

    for pattern in sequence_df["Pattern"].unique():
        sub = sequence_df[sequence_df["Pattern"] == pattern].copy()
        valid = sub[sub["occurrence_count"] >= MIN_OCC]

        print(f"\n── {pattern} ──")
        print(f"   Tickers with ≥{MIN_OCC} occurrences : {len(valid)} / {len(sub)}")

        if valid.empty:
            print("   Insufficient occurrences for reliable statistics.")
            continue

        avg_hit  = valid["hit_rate"].mean()
        med_hit  = valid["hit_rate"].median()
        avg_ret  = valid["avg_continuation_return"].mean()
        best_row = valid.loc[valid["hit_rate"].idxmax()]

        print(f"   Cross-ticker avg hit_rate        : {avg_hit*100:.1f}%")
        print(f"   Cross-ticker median hit_rate     : {med_hit*100:.1f}%")
        print(f"   Cross-ticker avg continuation    : {avg_ret*100:.3f}%")
        print(f"   Best ticker  : {best_row['Ticker']}  "
              f"hit={best_row['hit_rate']*100:.1f}%  "
              f"n={int(best_row['occurrence_count'])}")

        # Pattern-specific extra stats
        if pattern == "post_large_down_reversal" and "hit_rate_1d" in valid.columns:
            print(f"   Avg hit_rate by horizon  "
                  f"1d={valid['hit_rate_1d'].mean()*100:.1f}%  "
                  f"3d={valid['hit_rate_3d'].mean()*100:.1f}%  "
                  f"5d={valid['hit_rate_5d'].mean()*100:.1f}%")

        if pattern == "monday_weakness_friday_strength" and "lift" in valid.columns:
            avg_lift = valid["lift"].mean()
            print(f"   Avg lift vs. independent expectation : {avg_lift:.3f}x")

    # ── Best pattern overall ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("MOST CONSISTENT PATTERN (highest median hit_rate, n ≥ 10)")
    print("=" * 65)
    summary_rows = []
    for pattern in sequence_df["Pattern"].unique():
        sub = sequence_df[(sequence_df["Pattern"] == pattern) &
                          (sequence_df["occurrence_count"] >= MIN_OCC)]
        if sub.empty:
            continue
        summary_rows.append({
            "Pattern":           pattern,
            "Tickers_eligible":  len(sub),
            "Median_hit_rate_%": round(sub["hit_rate"].median() * 100, 1),
            "Avg_hit_rate_%":    round(sub["hit_rate"].mean() * 100, 1),
            "Avg_cont_ret_%":    round(sub["avg_continuation_return"].mean() * 100, 3),
        })
    if summary_rows:
        pat_summary = pd.DataFrame(summary_rows).sort_values(
            "Median_hit_rate_%", ascending=False
        )
        pd.set_option("display.float_format", "{:.3f}".format)
        print(pat_summary.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = os.path.join(_SRC, "..", "data", "events")
    os.makedirs(out_dir, exist_ok=True)

    wd_path  = os.path.join(out_dir, "calendar_patterns.csv")
    seq_path = os.path.join(out_dir, "sequence_patterns.csv")
    weekday_corrected.to_csv(wd_path,  index=False)
    sequence_df.to_csv(seq_path, index=False)
    print(f"\nWeekday results  → {os.path.abspath(wd_path)}")
    print(f"Sequence results → {os.path.abspath(seq_path)}")
