"""
Evidence bundle and risk gate for BIST100 MCP Trader.

Collects, filters, and commits analysis findings into a structured evidence
bundle that feeds the final human-readable explanation.  Also runs pre-analysis
data-quality and safety checks via the RiskGate.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore", category=FutureWarning)

_SRC = os.path.dirname(__file__)
sys.path.insert(0, _SRC)

from universe import UNIVERSE, has_sector_peers


# ---------------------------------------------------------------------------
# 1. EvidenceBundle
# ---------------------------------------------------------------------------

class EvidenceBundle:
    """
    Container for all evidence collected during one analysis session.

    Attributes:
        ticker:            The stock being analysed.
        observed_context:  All computed features / raw results keyed by name.
        committed_evidence: Filtered list of evidence items chosen for the
                            final explanation, each {"id", "statement",
                            "source_tool", "value"}.
    """

    def __init__(self, ticker: str) -> None:
        self.ticker:             str        = ticker.upper()
        self.observed_context:   Dict[str, Any] = {}
        self.committed_evidence: List[dict] = []

    def __repr__(self) -> str:
        return (
            f"EvidenceBundle(ticker={self.ticker!r}, "
            f"observed={len(self.observed_context)}, "
            f"committed={len(self.committed_evidence)})"
        )


# ---------------------------------------------------------------------------
# 2. add_observed
# ---------------------------------------------------------------------------

def add_observed(bundle: EvidenceBundle, key: str, value: Any) -> None:
    """Add a raw observation to the bundle's observed_context."""
    bundle.observed_context[key] = value


# ---------------------------------------------------------------------------
# 3. commit_evidence
# ---------------------------------------------------------------------------

def commit_evidence(
    bundle:      EvidenceBundle,
    statement:   str,
    source_tool: str,
    value:       Any,
) -> None:
    """
    Add a filtered evidence item to committed_evidence.

    Auto-assigns an incremental id (E1, E2, …).
    """
    eid = f"E{len(bundle.committed_evidence) + 1}"
    bundle.committed_evidence.append({
        "id":          eid,
        "statement":   statement,
        "source_tool": source_tool,
        "value":       value,
    })


# ---------------------------------------------------------------------------
# 4. build_evidence_bundle
# ---------------------------------------------------------------------------

def build_evidence_bundle(
    ticker:             str,
    technical_result:   Dict[str, Any],
    fundamental_result: Dict[str, Any],
    sector_result:      Dict[str, Any],
    event_result:       Dict[str, Any],
    macro_snapshot:     Dict[str, Any],
) -> EvidenceBundle:
    """
    Construct an EvidenceBundle from the outputs of Phases 0-5.

    All inputs are added verbatim to observed_context.  A subset is promoted
    to committed_evidence according to the rules below.

    Committed-evidence inclusion rules
    ------------------------------------
    ALWAYS committed (independent of threshold):
      - technical_score  : always include (0-5 scale anchor)
      - macro regime     : always include (TCMB rate + TRY 30-day change)

    Committed when threshold is met:
      - sector_lag_20d < -5%   : stock is lagging its benchmark by > 5 pp
      - sector_lag_20d > +5%   : stock is leading its benchmark by > 5 pp
      - KAMA_Slope == "rising" : early uptrend signal
      - Supertrend_Direction == "bullish" : trend confirmation
      - |net_profit_growth_qoq| > 10%  : meaningful earnings change
      - |revenue_growth_qoq|   > 10%   : meaningful revenue change
      - net_debt_ebitda is not NaN and value is reported
      - lower_band_touch bounce_rate_5d > 55%  AND n >= 5 : pattern is reliable
      - rsi_trough bounce_rate_5d > 75%         AND n >= 5 : strong reversal signal

    NOT committed (kept in observed_context only):
      - Raw OHLCV data, all rolling indicator values
      - Individual criterion flags (crit_trend, crit_rsi, etc.)
      - Event dates with NaN forward returns
      - TÜFE YoY when NaN (interpolation gap)

    Args:
        ticker:             Short ticker code.
        technical_result:   Dict with keys: technical_score, KAMA_Slope,
                            Supertrend_Direction, Ichimoku_State, RSI14.
        fundamental_result: Dict with keys: net_profit_growth_qoq,
                            revenue_growth_qoq, roe, net_debt_ebitda,
                            fundamental_score.
        sector_result:      Dict with keys: sector_group, benchmark_type,
                            lag_score_5d, lag_score_10d, lag_score_20d,
                            intra_group_rank.
        event_result:       Dict mapping event_type -> {"n", "bounce_rate_5d",
                            "avg_fwd_5d"}.
        macro_snapshot:     Dict with keys: usdtry, eurtry, tcmb_rate,
                            tufe_yoy, fed_rate, usdtry_30d_ago.

    Returns:
        Populated EvidenceBundle.
    """
    bundle = EvidenceBundle(ticker)

    # ── Store everything in observed_context ─────────────────────────
    add_observed(bundle, "technical",   technical_result)
    add_observed(bundle, "fundamental", fundamental_result)
    add_observed(bundle, "sector",      sector_result)
    add_observed(bundle, "events",      event_result)
    add_observed(bundle, "macro",       macro_snapshot)

    # ── Committed evidence — technical ───────────────────────────────
    ts = technical_result.get("technical_score", None)
    commit_evidence(
        bundle,
        statement   = f"Technical score: {ts}/5",
        source_tool = "calculate_technical_score",
        value       = ts,
    )

    kama_slope = technical_result.get("KAMA_Slope")
    if kama_slope == "rising":
        commit_evidence(
            bundle,
            statement   = "KAMA slope turned positive (rising) — early uptrend signal",
            source_tool = "calculate_kama",
            value       = kama_slope,
        )

    st_dir = technical_result.get("Supertrend_Direction")
    if st_dir == "bullish":
        commit_evidence(
            bundle,
            statement   = "Supertrend direction is bullish — trend confirmation",
            source_tool = "calculate_supertrend",
            value       = st_dir,
        )

    ichimoku = technical_result.get("Ichimoku_State")
    if ichimoku and not (isinstance(ichimoku, float) and np.isnan(ichimoku)):
        commit_evidence(
            bundle,
            statement   = f"Ichimoku state: {ichimoku}",
            source_tool = "calculate_ichimoku",
            value       = ichimoku,
        )

    # ── Committed evidence — sector relative ─────────────────────────
    lag20 = sector_result.get("lag_score_20d")
    if lag20 is not None and not np.isnan(float(lag20)):
        bm_type = sector_result.get("benchmark_type", "xu100_fallback")
        bm_label = "sector median" if bm_type == "sector" else "XU100"
        lag_pct  = float(lag20) * 100
        if lag_pct < -5:
            commit_evidence(
                bundle,
                statement   = (
                    f"Stock is {abs(lag_pct):.1f} pp behind {bm_label} "
                    f"over the last 20 trading days (laggard)"
                ),
                source_tool = "rank_universe_by_sector",
                value       = lag20,
            )
        elif lag_pct > 5:
            commit_evidence(
                bundle,
                statement   = (
                    f"Stock is {lag_pct:.1f} pp ahead of {bm_label} "
                    f"over the last 20 trading days (leader)"
                ),
                source_tool = "rank_universe_by_sector",
                value       = lag20,
            )

    # ── Committed evidence — fundamental ─────────────────────────────
    np_growth = fundamental_result.get("net_profit_growth_qoq")
    if np_growth is not None and not np.isnan(float(np_growth)):
        pct = float(np_growth) * 100
        if abs(pct) > 10:
            direction = "improved" if pct > 0 else "declined"
            commit_evidence(
                bundle,
                statement   = f"Net profit {direction} by {abs(pct):.1f}% QoQ",
                source_tool = "calculate_quarterly_changes",
                value       = np_growth,
            )

    rev_growth = fundamental_result.get("revenue_growth_qoq")
    if rev_growth is not None and not np.isnan(float(rev_growth)):
        pct = float(rev_growth) * 100
        if abs(pct) > 10:
            direction = "grew" if pct > 0 else "contracted"
            commit_evidence(
                bundle,
                statement   = f"Revenue {direction} by {abs(pct):.1f}% QoQ",
                source_tool = "calculate_quarterly_changes",
                value       = rev_growth,
            )

    nd_ebitda = fundamental_result.get("net_debt_ebitda")
    if nd_ebitda is not None and not np.isnan(float(nd_ebitda)):
        commit_evidence(
            bundle,
            statement   = f"Net Debt/EBITDA (TTM): {float(nd_ebitda):.2f}x",
            source_tool = "calculate_derived_ratios",
            value       = nd_ebitda,
        )

    # ── Committed evidence — event patterns ──────────────────────────
    for etype, thresholds in [
        ("lower_band_touch", {"bounce_rate_5d": 0.55, "n_min": 5}),
        ("rsi_trough",       {"bounce_rate_5d": 0.75, "n_min": 5}),
    ]:
        info = event_result.get(etype, {})
        br   = info.get("bounce_rate_5d")
        n    = info.get("n", 0)
        if br is not None and not np.isnan(float(br)):
            if float(br) >= thresholds["bounce_rate_5d"] and n >= thresholds["n_min"]:
                commit_evidence(
                    bundle,
                    statement   = (
                        f"{etype.replace('_', ' ').title()}: "
                        f"{float(br)*100:.1f}% bounce rate within 5 days "
                        f"(n={n})"
                    ),
                    source_tool = "build_event_log",
                    value       = {"bounce_rate_5d": br, "n": n},
                )

    # ── Committed evidence — macro (always) ──────────────────────────
    tcmb  = macro_snapshot.get("tcmb_rate")
    usd   = macro_snapshot.get("usdtry")
    usd30 = macro_snapshot.get("usdtry_30d_ago")
    tufe  = macro_snapshot.get("tufe_yoy")

    try_change = None
    if usd is not None and usd30 is not None:
        try_change = (float(usd) - float(usd30)) / float(usd30) * 100

    macro_parts = []
    if tcmb is not None:
        macro_parts.append(f"TCMB rate {tcmb:.2f}%")
    if try_change is not None:
        macro_parts.append(
            f"TRY {'weakened' if try_change > 0 else 'strengthened'} "
            f"{abs(try_change):.1f}% vs. USD over last 30 days"
        )
    if tufe is not None and not np.isnan(float(tufe)):
        macro_parts.append(f"TÜİK CPI YoY ≈ {float(tufe):.1f}%")

    if macro_parts:
        commit_evidence(
            bundle,
            statement   = "Macro regime: " + "; ".join(macro_parts),
            source_tool = "build_daily_macro_series",
            value       = {
                "tcmb_rate":     tcmb,
                "usdtry":        usd,
                "try_30d_chg_%": try_change,
                "tufe_yoy":      tufe,
            },
        )

    return bundle


# ---------------------------------------------------------------------------
# 5. RiskGate
# ---------------------------------------------------------------------------

class RiskGate:
    """
    Pre-analysis safety and data-quality checker.

    check_data_quality evaluates HARD_BLOCK and SOFT_WARNING conditions
    and returns a structured gate_result dict.

    HARD_BLOCK conditions (any one causes status = "HARD_BLOCK"):
        - Ticker not in the 30-stock universe
        - price_df is empty or has more than 20% NaN in Close
        - Fundamental disclosure_date is in the future relative to price data end
          (would imply look-ahead bias)

    SOFT_WARNING conditions (accumulate; status = "PASS_WITH_WARNINGS"):
        - Ticker has no sector peers (xu100_fallback only)
        - Total reversal events < 5 in the event log
        - macro_context NaN rate > 20% (e.g. tufe_yoy interpolation gap)
    """

    @staticmethod
    def check_data_quality(
        bundle:             EvidenceBundle,
        price_df:           pd.DataFrame,
        fundamental_df:     Optional[pd.DataFrame] = None,
        evaluated_quarter:  Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run hard-block and soft-warning checks for the bundle's ticker.

        Args:
            bundle:            EvidenceBundle (must contain 'macro' in
                               observed_context).
            price_df:          OHLCV DataFrame for the ticker.
            fundamental_df:    Output of build_point_in_time_fundamentals()
                               or None.
            evaluated_quarter: When provided (e.g. "2025Q3"), the look-ahead
                               check (HARD BLOCK 3) is scoped to only that
                               quarter's disclosure_date.  Quarters beyond
                               evaluated_quarter that exist in fundamental_df
                               but are not used in this analysis will NOT
                               trigger a block.  Pass None (default) to retain
                               the original full-DataFrame scan behaviour —
                               useful for bulk universe-wide leakage audits.

        Returns:
            {
                "status":   "PASS" | "PASS_WITH_WARNINGS" | "HARD_BLOCK",
                "warnings": [str, ...],
                "blocks":   [str, ...],
            }
        """
        ticker   = bundle.ticker
        blocks:   List[str] = []
        warnings: List[str] = []

        # ── HARD BLOCK 1: ticker not in universe ──────────────────────
        if not (UNIVERSE["ticker"] == ticker).any():
            blocks.append(
                f"Ticker '{ticker}' is not in the 30-stock coverage universe."
            )

        # ── HARD BLOCK 2: price data quality ─────────────────────────
        if price_df is None or price_df.empty:
            blocks.append("price_df is empty — no market data available.")
        else:
            close_nan_rate = price_df["Close"].isna().mean()
            if close_nan_rate > 0.20:
                blocks.append(
                    f"Close price has {close_nan_rate*100:.1f}% NaN values "
                    f"(threshold 20%)."
                )

        # ── HARD BLOCK 3: future disclosure date (look-ahead check) ───
        # When evaluated_quarter is given, only check that single row.
        # When None, scan the entire DataFrame (backward-compatible behaviour
        # useful for full-universe leakage audits).
        if fundamental_df is not None and not fundamental_df.empty:
            if "disclosure_date" in fundamental_df.columns and not price_df.empty:
                price_end = price_df.index.max()

                if evaluated_quarter is not None:
                    # Scope check to the single quarter being used
                    rows_to_check = fundamental_df[
                        fundamental_df["period"] == evaluated_quarter
                    ]
                else:
                    # Legacy full-scan
                    rows_to_check = fundamental_df

                future_discs = rows_to_check[
                    pd.to_datetime(rows_to_check["disclosure_date"]) > price_end
                ]
                if not future_discs.empty:
                    periods = future_discs["period"].tolist()
                    scope   = f"(evaluated_quarter={evaluated_quarter!r})" if evaluated_quarter else "(full scan)"
                    blocks.append(
                        f"Disclosure dates for {periods} are beyond the price data "
                        f"end date ({price_end.date()}) — potential look-ahead bias. "
                        f"{scope}"
                    )

        # ── SOFT WARNING 1: no sector peers ───────────────────────────
        try:
            if not has_sector_peers(ticker):
                warnings.append(
                    f"'{ticker}' has no sector peers — benchmark falls back to XU100. "
                    f"Sector-relative analysis will be less informative."
                )
        except ValueError:
            pass  # already caught by HARD BLOCK 1 if ticker unknown

        # ── SOFT WARNING 2: insufficient reversal events ──────────────
        event_obs = bundle.observed_context.get("events", {})
        total_events = sum(
            v.get("n", 0) if isinstance(v, dict) else 0
            for v in event_obs.values()
        )
        if total_events > 0 and total_events < 5:
            warnings.append(
                f"Only {total_events} reversal event(s) detected — "
                f"pattern statistics will be unreliable."
            )

        # ── SOFT WARNING 3: macro NaN rate ────────────────────────────
        macro_obs = bundle.observed_context.get("macro", {})
        macro_vals = [v for v in macro_obs.values() if isinstance(v, (int, float))]
        if macro_vals:
            nan_rate = sum(1 for v in macro_vals if v is None or np.isnan(float(v))) / len(macro_vals)
            if nan_rate > 0.20:
                warnings.append(
                    f"Macro context has {nan_rate*100:.0f}% NaN fields "
                    f"(likely tufe_yoy interpolation gap)."
                )

        # ── Determine status ─────────────────────────────────────────
        if blocks:
            status = "HARD_BLOCK"
        elif warnings:
            status = "PASS_WITH_WARNINGS"
        else:
            status = "PASS"

        return {"status": status, "warnings": warnings, "blocks": blocks}


# ---------------------------------------------------------------------------
# Entry point — ASELS demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings as _w
    _w.filterwarnings("ignore")
    sys.path.insert(0, _SRC)

    import pandas as pd
    from indicators import (
        calculate_sma, calculate_rsi, calculate_macd,
        calculate_bollinger_bands, calculate_kama, calculate_supertrend,
        calculate_ichimoku, calculate_technical_score,
    )
    from support_resistance import find_support_resistance
    from sector_analysis import rank_universe_by_sector
    from reversal_events import build_event_log
    from fundamental_analysis import load_fundamentals, calculate_derived_ratios
    from fundamental_reaction import calculate_quarterly_changes
    from macro_context import build_daily_macro_series
    from point_in_time import build_point_in_time_fundamentals

    pd.set_option("display.width", 140)

    TICKER     = "ASELS"
    SNAP_DATE  = pd.Timestamp("2025-12-31")

    # ── Technical ──────────────────────────────────────────────────────
    df = pd.read_csv(
        os.path.join(_SRC, "..", "data", "universe_cache", "ASELS.csv"),
        index_col="Date", parse_dates=True,
    )
    df["SMA20"] = calculate_sma(df, 20); df["SMA50"] = calculate_sma(df, 50)
    df["RSI14"] = calculate_rsi(df, 14)
    df[["MACD", "MACD_Signal", "MACD_Hist"]] = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_kama(df)
    df = calculate_supertrend(df)
    df = calculate_ichimoku(df)
    _, pivot_lows = find_support_resistance(df, order=5)
    df = calculate_technical_score(df, pivot_lows)
    last = df.iloc[-1]
    technical_result = {
        "technical_score":       int(last["technical_score"]),
        "KAMA_Slope":            last["KAMA_Slope"],
        "Supertrend_Direction":  last["Supertrend_Direction"],
        "Ichimoku_State":        last["Ichimoku_State"],
        "RSI14":                 float(last["RSI14"]),
    }

    # ── Sector ─────────────────────────────────────────────────────────
    rank_df = rank_universe_by_sector(SNAP_DATE, windows=[5, 10, 20])
    s_row   = rank_df[rank_df["ticker"] == TICKER].iloc[0]
    sector_result = {k: s_row[k] for k in ["sector_group", "benchmark_type",
                                             "lag_score_5d", "lag_score_10d",
                                             "lag_score_20d", "intra_group_rank"]}

    # ── Events ─────────────────────────────────────────────────────────
    df_ev = calculate_bollinger_bands(df.copy())
    log   = build_event_log(df_ev, TICKER)
    event_result: Dict[str, Any] = {}
    for etype, grp in log.groupby("event_type"):
        fwd5 = grp["forward_return_5d"].dropna()
        event_result[etype] = {
            "n":               len(grp),
            "bounce_rate_5d":  float((fwd5 > 0).mean()) if len(fwd5) else float("nan"),
            "avg_fwd_5d":      float(fwd5.mean()) if len(fwd5) else float("nan"),
        }

    # ── Fundamental ────────────────────────────────────────────────────
    raw_fund = load_fundamentals(TICKER)
    raw_fund = raw_fund.reset_index()
    fund_df  = calculate_derived_ratios(raw_fund.set_index("period").copy()).reset_index()
    chg      = calculate_quarterly_changes(TICKER)
    last_chg = chg.dropna(subset=["net_profit_growth_qoq"]).iloc[-1]
    last_fund = fund_df.iloc[-1]
    fundamental_result = {
        "net_profit_growth_qoq": float(last_chg.get("net_profit_growth_qoq", float("nan"))),
        "revenue_growth_qoq":    float(last_chg.get("revenue_growth_qoq",    float("nan"))),
        "roe":                   float(last_fund.get("roe",           float("nan"))),
        "net_debt_ebitda":       float(last_fund.get("net_debt_ebitda", float("nan"))),
        "fundamental_score":     int(raw_fund.set_index("period").get("fundamental_score", pd.Series([0])).iloc[-1])
            if "fundamental_score" in raw_fund.columns else 0,
    }

    # ── Macro snapshot ─────────────────────────────────────────────────
    macro = pd.read_csv(
        os.path.join(_SRC, "..", "data", "macro", "macro_daily.csv"),
        index_col="Date", parse_dates=True,
    )
    snap    = macro.loc[SNAP_DATE]
    snap30  = macro.loc["2025-12-01"]
    macro_snapshot = {
        "usdtry":       float(snap["usdtry"]),
        "eurtry":       float(snap["eurtry"]),
        "tcmb_rate":    float(snap["tcmb_rate"]),
        "tufe_yoy":     float(snap["tufe_yoy"]),
        "fed_rate":     float(snap["fed_rate"]),
        "usdtry_30d_ago": float(snap30["usdtry"]),
    }

    # ── Build bundle ───────────────────────────────────────────────────
    bundle = build_evidence_bundle(
        TICKER, technical_result, fundamental_result,
        sector_result, event_result, macro_snapshot,
    )

    print("=" * 60)
    print(f"EVIDENCE BUNDLE — {TICKER}")
    print("=" * 60)
    print(f"\nobserved_context keys : {list(bundle.observed_context.keys())}")
    print(f"\ncommitted_evidence ({len(bundle.committed_evidence)} items):")
    for item in bundle.committed_evidence:
        print(f"  [{item['id']}] {item['statement']}")
        print(f"       source={item['source_tool']}  value={item['value']}")

    # ── Risk gate — ASELS (valid) ──────────────────────────────────────
    pit = build_point_in_time_fundamentals(TICKER, raw_fund)
    gate = RiskGate.check_data_quality(bundle, df, pit)
    print(f"\n{'='*60}")
    print(f"RISK GATE — {TICKER}")
    print(f"  status   : {gate['status']}")
    print(f"  blocks   : {gate['blocks']  or 'none'}")
    print(f"  warnings : {gate['warnings'] or 'none'}")

    # ── Risk gate — XXXX (should HARD_BLOCK) ──────────────────────────
    bad_bundle = EvidenceBundle("XXXX")
    bad_df     = pd.DataFrame()
    gate_bad   = RiskGate.check_data_quality(bad_bundle, bad_df)
    print(f"\n{'='*60}")
    print(f"RISK GATE — XXXX (invalid ticker)")
    print(f"  status   : {gate_bad['status']}")
    print(f"  blocks   : {gate_bad['blocks']}")
    assert gate_bad["status"] == "HARD_BLOCK", "Expected HARD_BLOCK for unknown ticker"
    print("  assertion passed: HARD_BLOCK confirmed.")
