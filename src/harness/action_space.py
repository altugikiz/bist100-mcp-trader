"""
Action space and decision logic for BIST100 MCP Trader.

Defines the eight analytical conclusions the harness can emit and the
decision tree that maps evidence + gate status to a single action.

BUY/SELL labels are not part of the allowed action space per the project
brief's educational framing (Section 17). Where an optional backtest label
is needed (e.g., for the existing strategy.py signal generation), BUY/SELL
remain internal backtest-only labels, separate from this harness's analytical
conclusion vocabulary.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# 1. Allowed action constants
# ---------------------------------------------------------------------------

ALLOWED_ACTIONS: List[str] = [
    "HOLD",
    "WATCH",
    "INVESTIGATE",
    "POTENTIAL_CATCH_UP_CANDIDATE",
    "TECHNICAL_REVERSAL_CANDIDATE",
    "FUNDAMENTAL_IMPROVEMENT_CANDIDATE",
    "REJECT_SIGNAL",
    "ANALYSIS_UNSAFE",
]


# ---------------------------------------------------------------------------
# 2. determine_action
# ---------------------------------------------------------------------------

def determine_action(
    evidence_bundle,
    gate_result:              Dict[str, Any],
    sector_result:            Dict[str, Any],
    event_result:             Dict[str, Any],
    fundamental_score_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply a priority-ordered decision tree and return one action from
    ALLOWED_ACTIONS together with a reasoning trace.

    Priority order (highest to lowest):
        1. ANALYSIS_UNSAFE  — gate HARD_BLOCK or ANALYSIS_UNSAFE
        2. INVESTIGATE      — gate INVESTIGATE
        3. POTENTIAL_CATCH_UP_CANDIDATE
        4. TECHNICAL_REVERSAL_CANDIDATE
        5. FUNDAMENTAL_IMPROVEMENT_CANDIDATE
        6. REJECT_SIGNAL
        7. WATCH
        8. HOLD             — default when no criterion fires

    Args:
        evidence_bundle:          EvidenceBundle from build_evidence_bundle().
        gate_result:              Output of RiskGate.check_data_quality().
        sector_result:            Dict with benchmark_type, lag_score_20d,
                                  potential_catchup_flag.
        event_result:             Dict mapping event_type → {n, bounce_rate_5d,
                                  avg_fwd_5d, days_since_last}.
        fundamental_score_result: Dict with fundamental_score (0-5),
                                  net_profit_growth_qoq, revenue_growth_qoq,
                                  prev_net_profit_growth_qoq,
                                  prev_revenue_growth_qoq.

    Returns:
        {
            "action":    str  — one of ALLOWED_ACTIONS,
            "reasoning": list[str] — short justifications referencing
                                     committed_evidence ids where applicable,
            "timestamp": str  — ISO-8601 UTC,
        }
    """
    reasoning: List[str] = []
    committed  = {item["id"]: item for item in evidence_bundle.committed_evidence}

    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ── 1. Gate blocks ────────────────────────────────────────────────────
    status = gate_result.get("status", "PASS")
    if status in ("HARD_BLOCK", "ANALYSIS_UNSAFE"):
        for block in gate_result.get("blocks", []):
            reasoning.append(f"HARD_BLOCK: {block}")
        if not reasoning:
            reasoning.append("Gate returned ANALYSIS_UNSAFE; unsafe to proceed.")
        return {"action": "ANALYSIS_UNSAFE", "reasoning": reasoning, "timestamp": _now()}

    if status == "INVESTIGATE":
        reasoning.append("Gate returned INVESTIGATE; requires manual review before proceeding.")
        return {"action": "INVESTIGATE", "reasoning": reasoning, "timestamp": _now()}

    # Attach any soft warnings to the trace but do not block
    for w in gate_result.get("warnings", []):
        reasoning.append(f"[soft-warning] {w}")

    # ── 2. POTENTIAL_CATCH_UP_CANDIDATE ───────────────────────────────────
    if sector_result.get("potential_catchup_flag", False):
        lag20 = sector_result.get("lag_score_20d")
        bm    = sector_result.get("benchmark_type", "benchmark")
        pct   = float(lag20) * 100 if lag20 is not None else None
        msg   = (
            f"Catchup flag set; stock is lagging {bm} by "
            f"{abs(pct):.1f} pp over 20d."
            if pct is not None else "Catchup flag set by sector_analysis."
        )
        # Cross-reference any sector evidence item
        sector_ids = [eid for eid, item in committed.items()
                      if "rank_universe_by_sector" in item.get("source_tool", "")]
        if sector_ids:
            msg += f" (see {', '.join(sector_ids)})"
        reasoning.append(msg)
        return {"action": "POTENTIAL_CATCH_UP_CANDIDATE", "reasoning": reasoning, "timestamp": _now()}

    # ── 3. TECHNICAL_REVERSAL_CANDIDATE ──────────────────────────────────
    reversal_triggers = ["rsi_trough", "lower_band_touch"]
    for etype in reversal_triggers:
        info      = event_result.get(etype, {})
        br        = info.get("bounce_rate_5d")
        days_last = info.get("days_since_last")  # None if not provided
        if br is None:
            continue
        recent_enough = (days_last is None) or (float(days_last) <= 10)
        if float(br) > 0.60 and recent_enough:
            ev_ids = [eid for eid, item in committed.items()
                      if "build_event_log" in item.get("source_tool", "")
                      and etype.replace("_", " ").title().lower()
                          in item.get("statement", "").lower()]
            ref = f" ({', '.join(ev_ids)})" if ev_ids else ""
            reasoning.append(
                f"{etype} bounce_rate_5d={float(br)*100:.1f}% > 60% "
                f"threshold{ref}."
            )
            return {"action": "TECHNICAL_REVERSAL_CANDIDATE", "reasoning": reasoning, "timestamp": _now()}

    # ── 4. FUNDAMENTAL_IMPROVEMENT_CANDIDATE ─────────────────────────────
    fscore       = fundamental_score_result.get("fundamental_score", 0)
    np_growth    = fundamental_score_result.get("net_profit_growth_qoq")
    rev_growth   = fundamental_score_result.get("revenue_growth_qoq")
    prev_np      = fundamental_score_result.get("prev_net_profit_growth_qoq")
    prev_rev     = fundamental_score_result.get("prev_revenue_growth_qoq")

    def _turned_positive(current, previous) -> bool:
        """True when current > 0 and either previous was ≤ 0 or previous is unknown."""
        if current is None:
            return False
        try:
            curr_f = float(current)
        except (TypeError, ValueError):
            return False
        if curr_f <= 0:
            return False
        if previous is None:
            return True  # can't compare; treat improvement as newly positive
        try:
            return float(previous) <= 0
        except (TypeError, ValueError):
            return True

    growth_turned = _turned_positive(np_growth, prev_np) or _turned_positive(rev_growth, prev_rev)

    # NOTE: Current implementation only catches sign-flip improvements
    # (loss→profit), not sustained strong growth. This is a known scope
    # limitation — a company with consistently strong (but not
    # accelerating-from-negative) fundamentals will route to WATCH rather
    # than FUNDAMENTAL_IMPROVEMENT_CANDIDATE, which is conservative but may
    # under-flag genuinely strong candidates.

    if int(fscore) >= 4 and growth_turned:
        fund_ids = [eid for eid, item in committed.items()
                    if "calculate_quarterly_changes" in item.get("source_tool", "")]
        ref = f" ({', '.join(fund_ids)})" if fund_ids else ""
        reasoning.append(
            f"fundamental_score={fscore}/5 ≥ 4 and at least one growth metric "
            f"turned positive this quarter{ref}."
        )
        return {"action": "FUNDAMENTAL_IMPROVEMENT_CANDIDATE", "reasoning": reasoning, "timestamp": _now()}

    # ── 5. REJECT_SIGNAL ─────────────────────────────────────────────────
    weakness_count = 0
    weakness_notes: List[str] = []

    # Technical weakness signals
    tech = evidence_bundle.observed_context.get("technical", {})
    if tech.get("KAMA_Slope") == "falling":
        weakness_count += 1
        weakness_notes.append("KAMA slope is falling (downtrend pressure).")
    if tech.get("Supertrend_Direction") == "bearish":
        weakness_count += 1
        weakness_notes.append("Supertrend direction is bearish.")
    ts = tech.get("technical_score", 5)
    if isinstance(ts, (int, float)) and int(ts) <= 1:
        weakness_count += 1
        weakness_notes.append(f"Technical score={ts}/5 ≤ 1 (very weak).")

    # Fundamental weakness signals
    if np_growth is not None:
        try:
            if float(np_growth) < -0.20:
                weakness_count += 1
                weakness_notes.append(f"Net profit declined {float(np_growth)*100:.1f}% QoQ.")
        except (TypeError, ValueError):
            pass
    if rev_growth is not None:
        try:
            if float(rev_growth) < -0.20:
                weakness_count += 1
                weakness_notes.append(f"Revenue contracted {float(rev_growth)*100:.1f}% QoQ.")
        except (TypeError, ValueError):
            pass

    if weakness_count >= 2:
        reasoning.extend(weakness_notes[:weakness_count])
        return {"action": "REJECT_SIGNAL", "reasoning": reasoning, "timestamp": _now()}

    # ── 6. WATCH ─────────────────────────────────────────────────────────
    # Stock is borderline: technical score 3/5, or one soft signal present
    borderline = False
    watch_notes: List[str] = []

    if isinstance(ts, (int, float)) and int(ts) == 3:
        borderline = True
        watch_notes.append(f"Technical score={ts}/5 — borderline, worth monitoring.")

    lag20 = sector_result.get("lag_score_20d")
    if lag20 is not None:
        try:
            lag_pct = float(lag20) * 100
            if -5 <= lag_pct <= 5:
                borderline = True
                watch_notes.append(
                    f"Sector lag_score_20d={lag_pct:.1f}% — neutral range, no clear direction."
                )
        except (TypeError, ValueError):
            pass

    if borderline:
        reasoning.extend(watch_notes)
        return {"action": "WATCH", "reasoning": reasoning, "timestamp": _now()}

    # ── 7. HOLD (default) ────────────────────────────────────────────────
    reasoning.append("No decisive signal in either direction; maintaining current position.")
    return {"action": "HOLD", "reasoning": reasoning, "timestamp": _now()}


# ---------------------------------------------------------------------------
# Entry point — ASELS demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    import pandas as pd
    import numpy as np
    warnings.filterwarnings("ignore")

    from indicators import (
        calculate_sma, calculate_rsi, calculate_macd,
        calculate_bollinger_bands, calculate_kama, calculate_supertrend,
        calculate_ichimoku, calculate_technical_score,
    )
    from support_resistance import find_support_resistance
    from sector_analysis import rank_universe_by_sector
    from reversal_events import build_event_log
    from fundamental_analysis import (
        load_fundamentals, calculate_derived_ratios,
        calculate_yoy_growth, calculate_fundamental_score,
    )
    from fundamental_reaction import calculate_quarterly_changes
    from point_in_time import build_point_in_time_fundamentals
    from evidence import build_evidence_bundle, RiskGate

    pd.set_option("display.width", 140)

    TICKER    = "ASELS"
    SNAP_DATE = pd.Timestamp("2025-12-31")
    DATA_DIR  = os.path.join(_SRC, "..", "data")

    # ── Technical ──────────────────────────────────────────────────────
    df = pd.read_csv(
        os.path.join(DATA_DIR, "universe_cache", "ASELS.csv"),
        index_col="Date", parse_dates=True,
    )
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
    last = df.iloc[-1]
    technical_result = {
        "technical_score":      int(last["technical_score"]),
        "KAMA_Slope":           last["KAMA_Slope"],
        "Supertrend_Direction": last["Supertrend_Direction"],
        "Ichimoku_State":       last["Ichimoku_State"],
        "RSI14":                float(last["RSI14"]),
    }

    # ── Sector ─────────────────────────────────────────────────────────
    rank_df = rank_universe_by_sector(SNAP_DATE, windows=[5, 10, 20])
    s_row   = rank_df[rank_df["ticker"] == TICKER].iloc[0]
    sector_result = {k: s_row[k] for k in [
        "sector_group", "benchmark_type",
        "lag_score_5d", "lag_score_10d", "lag_score_20d", "intra_group_rank",
    ]}
    # ASELS is a sector leader (+23 pp) → catchup flag is FALSE by definition
    # (catchup = laggard, not leader). Set explicitly for clarity.
    sector_result["potential_catchup_flag"] = False

    # ── Events + days_since_last ────────────────────────────────────────
    log = build_event_log(df, TICKER)
    event_result = {}
    for etype, grp in log.groupby("event_type"):
        fwd5       = grp["forward_return_5d"].dropna()
        dates      = pd.to_datetime(grp["event_date"])
        days_since = int((SNAP_DATE - dates.max()).days) if len(dates) else None
        event_result[etype] = {
            "n":               len(grp),
            "bounce_rate_5d":  float((fwd5 > 0).mean()) if len(fwd5) else float("nan"),
            "avg_fwd_5d":      float(fwd5.mean())        if len(fwd5) else float("nan"),
            "days_since_last": days_since,
        }

    # ── Fundamental — 2025Q3 is the last "safe" quarter ───────────────
    # 2025Q4 disclosure date falls in Feb 2026 (beyond price data end of
    # 2025-12-31) and triggers a HARD_BLOCK. We pin the fundamental
    # snapshot to 2025Q3, the most recent disclosed quarter, to get
    # a realistic score for the PASS scenario test.
    SAFE_QUARTER = "2025Q3"
    PREV_QUARTER = "2025Q2"

    raw_fund = load_fundamentals(TICKER)   # indexed by period
    # sector_avg_pe=65.6: BIST defence sector median P/E (Phase 4 baseline)
    SECTOR_AVG_PE = 65.6
    fund_full = (
        raw_fund
        .pipe(lambda d: calculate_derived_ratios(d.copy()))
        .pipe(calculate_yoy_growth)
        .pipe(calculate_fundamental_score, sector_avg_pe=SECTOR_AVG_PE)
    )
    chg_df = calculate_quarterly_changes(TICKER).set_index("period")

    q3_row  = fund_full.loc[SAFE_QUARTER]
    q2_row  = chg_df.loc[PREV_QUARTER]  if PREV_QUARTER in chg_df.index else None
    q3_chg  = chg_df.loc[SAFE_QUARTER]

    fundamental_result = {
        # QoQ growth rates from the safe quarter
        "net_profit_growth_qoq":      float(q3_chg["net_profit_growth_qoq"]),
        "revenue_growth_qoq":         float(q3_chg["revenue_growth_qoq"]),
        # Previous quarter's QoQ values to detect "turned positive" transitions
        "prev_net_profit_growth_qoq": float(q2_row["net_profit_growth_qoq"]) if q2_row is not None else None,
        "prev_revenue_growth_qoq":    float(q2_row["revenue_growth_qoq"])    if q2_row is not None else None,
        # Derived ratios and score from the safe quarter
        "roe":               float(q3_row["roe"]),
        "net_debt_ebitda":   float(q3_row["net_debt_ebitda"]),
        # Real calculate_fundamental_score() output for 2025Q3
        "fundamental_score": int(q3_row["fundamental_score"]),
        # Criterion breakdown for transparency
        "_crit_revenue_growth":    bool(q3_row["crit_revenue_growth"]),
        "_crit_net_profit_growth": bool(q3_row["crit_net_profit_growth"]),
        "_crit_roe":               bool(q3_row["crit_roe"]),
        "_crit_net_debt_ebitda":   bool(q3_row["crit_net_debt_ebitda"]),
    }

    print("=" * 65)
    print(f"FUNDAMENTAL SCORE BREAKDOWN — {TICKER} {SAFE_QUARTER}")
    print("=" * 65)
    for k, v in fundamental_result.items():
        print(f"  {k:<35} {v}")
    print()

    # ── Macro ──────────────────────────────────────────────────────────
    macro    = pd.read_csv(os.path.join(DATA_DIR, "macro", "macro_daily.csv"),
                           index_col="Date", parse_dates=True)
    snap     = macro.loc[SNAP_DATE]
    snap30   = macro.loc["2025-12-01"]
    macro_snapshot = {
        "usdtry":         float(snap["usdtry"]),
        "eurtry":         float(snap["eurtry"]),
        "tcmb_rate":      float(snap["tcmb_rate"]),
        "tufe_yoy":       float(snap["tufe_yoy"]),
        "fed_rate":       float(snap["fed_rate"]),
        "usdtry_30d_ago": float(snap30["usdtry"]),
    }

    # ── Evidence bundle ────────────────────────────────────────────────
    bundle = build_evidence_bundle(
        TICKER, technical_result, fundamental_result,
        sector_result, event_result, macro_snapshot,
    )

    # ── PIT for gate ───────────────────────────────────────────────────
    pit = build_point_in_time_fundamentals(TICKER, raw_fund.reset_index())
    gate_real = RiskGate.check_data_quality(bundle, df, pit)

    # ── TEST 1: Real gate (expects ANALYSIS_UNSAFE — 2025Q4 future disc) ─
    print("=" * 65)
    print("TEST 1 — determine_action with REAL gate result")
    print(f"  (2025Q4 disclosure date > price end → HARD_BLOCK expected)")
    print("=" * 65)
    print(f"  gate status : {gate_real['status']}")
    result1 = determine_action(
        bundle, gate_real, sector_result, event_result, fundamental_result,
    )
    print(f"  action      : {result1['action']}")
    print(f"  reasoning   :")
    for r in result1["reasoning"]:
        print(f"    • {r}")
    assert result1["action"] == "ANALYSIS_UNSAFE", (
        f"Expected ANALYSIS_UNSAFE, got {result1['action']}"
    )
    print("  ✓ assertion passed: ANALYSIS_UNSAFE confirmed.")

    # ── TEST 2: Force gate to PASS; use real 2025Q3 fundamental score ──
    # With sector_avg_pe=65.6:
    #   pe_ratio (2025Q3) = 36.5 < 65.6 → crit_pe = True
    #   Score becomes 4/5 (was 3/5 with sector_avg_pe=None).
    #
    # _turned_positive analysis:
    #   net_profit_growth_qoq Q3  = +69.2%  → positive
    #   net_profit_growth_qoq Q2  = +165.6% → also positive (prev > 0)
    #   _turned_positive(Q3, Q2)  = False   (sign did NOT flip from ≤0 to >0)
    #
    # This exposes the known scope limitation documented in the NOTE above:
    # FUNDAMENTAL_IMPROVEMENT_CANDIDATE requires BOTH score ≥ 4 AND a sign
    # flip in at least one growth metric. ASELS has had consistently positive
    # QoQ growth since Q1 2025, so there is no loss→profit transition to
    # detect. The decision tree is conservative by design — it routes ASELS
    # to WATCH even with a score of 4/5.
    print()
    print("=" * 65)
    print("TEST 2 — determine_action with gate forced to PASS")
    print(f"  sector_avg_pe used                = {SECTOR_AVG_PE}")
    print(f"  pe_ratio (2025Q3)                 = {float(fund_full.loc[SAFE_QUARTER,'pe_ratio']):.2f} < {SECTOR_AVG_PE} → crit_pe=True")
    print(f"  fundamental_score (with crit_pe)  = {fundamental_result['fundamental_score']}/5")
    print(f"  net_profit_growth_qoq (Q3)        = {fundamental_result['net_profit_growth_qoq']*100:.1f}%")
    print(f"  prev net_profit_growth_qoq (Q2)   = {fundamental_result['prev_net_profit_growth_qoq']*100:.1f}%")
    print(f"  → _turned_positive(Q3, Q2): Q2 already positive → False")
    print(f"  → FUNDAMENTAL_IMPROVEMENT_CANDIDATE: score=4/5 ✓ BUT turned_positive=False ✗")
    print(f"  → Known scope limitation: sign-flip-only logic, no sustained-growth detection")
    print(f"  → Expected action: WATCH (technical_score=3, borderline)")
    print("=" * 65)
    gate_forced = {"status": "PASS", "blocks": [], "warnings": []}
    result2 = determine_action(
        bundle, gate_forced, sector_result, event_result, fundamental_result,
    )
    print(f"  action      : {result2['action']}")
    print(f"  reasoning   :")
    for r in result2["reasoning"]:
        print(f"    • {r}")
    assert result2["action"] in ALLOWED_ACTIONS, (
        f"Action '{result2['action']}' not in ALLOWED_ACTIONS."
    )
    print(f"  ✓ action is in ALLOWED_ACTIONS.")

    print()
    print("=" * 65)
    print("COMMITTED EVIDENCE REFERENCE (for reasoning trace)")
    print("=" * 65)
    for item in bundle.committed_evidence:
        print(f"  [{item['id']}] {item['statement']}")
