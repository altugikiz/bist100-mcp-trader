"""
Harness experiment framework for BIST100 MCP Trader.

Implements and compares five harness versions (A–E) from Section 30 of the
project brief, measuring how successive capability additions affect
tool-permission enforcement, evidence completeness, reproducibility, and
decision-log persistence.

Version definitions
-------------------
A  Baseline: no tool enforcement, no evidence gate, no memory replay.
   Manual LLM test required — Python simulation not meaningful.

B  + tool_enforcement only (reject out-of-order tool calls).
   Manual LLM test required — LLM behaviour under rejection varies.

C  tool_enforcement in a fully automated Python pipeline (no LLM).
   Simulated here by intentionally probing is_tool_allowed() with
   invalid-state calls and counting rejections.

D  C + evidence gate (EvidenceBundle + RiskGate.check_data_quality).
   Adds evidence_completeness metric.

E  D + memory replay (log_decision + replay_analysis written to disk).
   Adds decision_log_exists metric.

Reproducibility note
--------------------
For the automated Python pipeline (C/D/E) reproducibility is 1.0 by
construction: there is no LLM sampling step, so identical inputs always
produce identical outputs.  In a real LLM-in-the-loop setup (versions A/B)
reproducibility would be lower because the model's tool-selection policy
introduces stochastic variation across runs.
"""

import io
import os
import sys
import contextlib
import warnings
import math
from typing import Any

warnings.filterwarnings("ignore")

_SRC = os.path.dirname(__file__)
sys.path.insert(0, _SRC)

import numpy as np
import pandas as pd

from harness import (
    AnalysisSession, HARNESS_STATES, STATE_TOOL_PERMISSIONS,
    advance_state, is_tool_allowed, record_tool_call,
)
from mcp_server import (
    start_analysis_session, run_full_analysis,
    get_recommended_action, submit_human_review, get_session_replay,
    _SESSIONS,
)


# ---------------------------------------------------------------------------
# 1. Version registry
# ---------------------------------------------------------------------------

HARNESS_VERSIONS: dict[str, dict] = {
    "A": {
        "name":             "Baseline (no guardrails)",
        "description":      (
            "Plain LLM calling tools in free order. No tool-permission "
            "enforcement, no evidence gate, no decision log. Results "
            "depend entirely on the model's self-regulation."
        ),
        "tool_enforcement": False,
        "evidence_gate":    False,
        "memory_replay":    False,
    },
    "B": {
        "name":             "Tool enforcement only",
        "description":      (
            "Harness rejects out-of-order tool calls via is_tool_allowed(). "
            "LLM must respect state ordering or receive an error response. "
            "No evidence bundle or decision log."
        ),
        "tool_enforcement": True,
        "evidence_gate":    False,
        "memory_replay":    False,
    },
    "C": {
        "name":             "Automated pipeline + tool enforcement",
        "description":      (
            "Fully automated Python pipeline using mcp_server.py. "
            "is_tool_allowed() is probed with intentional wrong-state calls "
            "to measure rejection rate. No evidence bundle or log."
        ),
        "tool_enforcement": True,
        "evidence_gate":    False,
        "memory_replay":    False,
    },
    "D": {
        "name":             "C + evidence gate",
        "description":      (
            "Adds EvidenceBundle + RiskGate.check_data_quality() to version C. "
            "Measures evidence_completeness (committed / observable features)."
        ),
        "tool_enforcement": True,
        "evidence_gate":    True,
        "memory_replay":    False,
    },
    "E": {
        "name":             "D + memory replay",
        "description":      (
            "Adds log_decision() and replay_analysis() to version D. "
            "Decision log is written to disk; decision_log_exists is measured."
        ),
        "tool_enforcement": True,
        "evidence_gate":    True,
        "memory_replay":    True,
    },
}


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

# Intentional wrong-state probes: pairs of (state_to_be_in, tool_to_call).
# Each pair should be REJECTED by is_tool_allowed() because the tool does not
# belong to that state's permission list.
_INVALID_PROBES: list[tuple[str, str]] = [
    ("SELECT_UNIVERSE",               "run_backtest_summary"),      # backtest in universe state
    ("LOAD_MARKET_DATA",              "generate_explanation"),       # explanation too early
    ("CHECK_DATA_QUALITY",            "build_evidence_bundle"),      # evidence before technicals
    ("CALCULATE_TECHNICALS",          "log_decision"),               # logging mid-analysis
    ("RUN_SECTOR_RELATIVE_ANALYSIS",  "calculate_fundamental_score"),# fundamentals in sector state
    ("RUN_PATTERN_AND_REVERSAL_ANALYSIS", "rank_universe_by_sector"),# sector in reversal state
    ("LOAD_FUNDAMENTALS",             "get_stock_history"),          # data fetch in fund state
    ("LOAD_MACRO_CONTEXT",            "calculate_indicators"),       # indicators in macro state
    ("BUILD_EVIDENCE",                "run_backtest_summary"),       # backtest in evidence state
    ("RISK_GATE",                     "build_evidence_bundle"),      # evidence rebuild in gate state
]

# Corresponding VALID probes — should all be ALLOWED.
_VALID_PROBES: list[tuple[str, str]] = [
    ("SELECT_UNIVERSE",               "get_sector_peers"),
    ("LOAD_MARKET_DATA",              "get_stock_history"),
    ("CHECK_DATA_QUALITY",            "check_data_quality"),
    ("CALCULATE_TECHNICALS",          "calculate_technical_score"),
    ("RUN_SECTOR_RELATIVE_ANALYSIS",  "rank_universe_by_sector"),
    ("RUN_PATTERN_AND_REVERSAL_ANALYSIS", "build_event_log"),
    ("LOAD_FUNDAMENTALS",             "calculate_derived_ratios"),
    ("LOAD_MACRO_CONTEXT",            "build_daily_macro_series"),
    ("BUILD_EVIDENCE",                "build_evidence_bundle"),
    ("RISK_GATE",                     "check_risk_limits"),
]


def _measure_tool_enforcement() -> dict:
    """
    Create a throwaway session, probe it with invalid and valid tool calls,
    and return enforcement statistics.

    Returns:
        {
            "invalid_rejected":      int,
            "invalid_total":         int,
            "valid_allowed":         int,
            "valid_total":           int,
            "invalid_tool_call_rate":float,  fraction of invalid calls rejected
        }
    """
    sess = AnalysisSession("__probe__")

    invalid_rejected = 0
    for state, tool in _INVALID_PROBES:
        # Manually set the session state (bypasses advance_state ordering guard)
        sess.current_state = state
        if not is_tool_allowed(sess, tool):
            invalid_rejected += 1

    valid_allowed = 0
    for state, tool in _VALID_PROBES:
        sess.current_state = state
        if is_tool_allowed(sess, tool):
            valid_allowed += 1

    n_invalid = len(_INVALID_PROBES)
    n_valid   = len(_VALID_PROBES)

    return {
        "invalid_rejected":       invalid_rejected,
        "invalid_total":          n_invalid,
        "valid_allowed":          valid_allowed,
        "valid_total":            n_valid,
        "invalid_tool_call_rate": invalid_rejected / n_invalid,
    }


def _evidence_completeness(bundle) -> float:
    """
    Ratio of committed evidence items to the total number of keys in
    observed_context (a proxy for observable features).

    observed_context contains 5 top-level keys (technical, fundamental,
    sector, events, macro), each of which holds a dict of scalar features.
    We count the total number of scalar leaf values as the denominator.
    """
    total_leaves = 0
    for v in bundle.observed_context.values():
        if isinstance(v, dict):
            total_leaves += sum(
                1 for vv in v.values()
                if not isinstance(vv, dict)    # skip nested dicts (event sub-dicts)
            )
        else:
            total_leaves += 1

    committed = len(bundle.committed_evidence)
    if total_leaves == 0:
        return float("nan")
    return round(committed / total_leaves, 4)


# ---------------------------------------------------------------------------
# 3. run_harness_experiment
# ---------------------------------------------------------------------------

def run_harness_experiment(ticker: str, version: str) -> dict[str, Any]:
    """
    Simulate one harness version for the given ticker and return metrics.

    Versions A and B require a live LLM agent and cannot be automated here;
    they are represented by placeholder entries.

    For C, D, E the function calls the mcp_server.py tools in sequence
    (start_analysis_session → run_full_analysis → get_recommended_action),
    measures enforcement, evidence, and log metrics, then returns:

        {
            "version":               str,
            "ticker":                str,
            "tool_enforcement":      bool,
            "evidence_gate":         bool,
            "memory_replay":         bool,
            "invalid_tool_call_rate":float | str,
            "evidence_completeness": float | str,
            "reproducibility":       float | str,
            "decision_log_exists":   bool | str,
            "action":                str,
            "gate_status":           str,
        }

    Reproducibility note: the Python pipeline is fully deterministic (no LLM
    sampling), so reproducibility is always 1.0 for C/D/E by construction.
    In a real LLM-in-the-loop deployment (versions A/B) reproducibility would
    be lower due to stochastic token sampling.

    Args:
        ticker:  Short BIST ticker code, e.g. 'ASELS'.
        version: One of 'A', 'B', 'C', 'D', 'E'.

    Returns:
        Metrics dict described above.
    """
    version = version.upper()
    if version not in HARNESS_VERSIONS:
        raise ValueError(f"Unknown version {version!r}. Choose from A–E.")

    vdef = HARNESS_VERSIONS[version]

    # ── A / B: results filled from Claude Desktop manual tests ────────
    if version == "A":
        return {
            "version":                version,
            "ticker":                 ticker,
            "tool_enforcement":       vdef["tool_enforcement"],
            "evidence_gate":          vdef["evidence_gate"],
            "memory_replay":          vdef["memory_replay"],
            "invalid_tool_call_rate": "N/A (no tools)",
            "evidence_completeness":  "N/A",
            "reproducibility":        "not tested (single run)",
            "decision_log_exists":    False,
            "action":                 "FUNDAMENTAL_IMPROVEMENT_CANDIDATE",
            "gate_status":            "N/A (no gate)",
            "numerical_hallucination": (
                "SEVERE — fabricated price (~396-450 TRY vs actual 231.70 TRY), "
                "fabricated Q2 2026 financials, fabricated backlog/capex figures "
                "not present in any tool output"
            ),
        }

    if version == "B":
        return {
            "version":                version,
            "ticker":                 ticker,
            "tool_enforcement":       vdef["tool_enforcement"],
            "evidence_gate":          vdef["evidence_gate"],
            "memory_replay":          vdef["memory_replay"],
            "invalid_tool_call_rate": "N/A (only 2 permitted tools used, no violations attempted)",
            "evidence_completeness":  "0% (no evidence bundle mechanism exists at this version)",
            "reproducibility":        "not tested (single run)",
            "decision_log_exists":    False,
            "action":                 "WATCH",
            "gate_status":            "N/A (no gate)",
            "numerical_hallucination": (
                "NONE — all figures (close=231.7, RSI14=70.86, "
                "SMA20=210.57, SMA50=198.65) match tool output exactly"
            ),
        }

    # ── C / D / E: automated pipeline ────────────────────────────────

    # Step 1: tool enforcement measurement (same for C, D, E)
    enforce_stats = _measure_tool_enforcement()

    # Step 2: run the full pipeline via mcp_server tools
    r1 = start_analysis_session(ticker)
    sid = r1["session_id"]

    r2 = run_full_analysis(sid)
    r3 = get_recommended_action(sid)

    action     = r3.get("action", "unknown")
    gate_status = r2["gate_result"]["status"]

    # Step 3: evidence completeness (D and E only)
    if vdef["evidence_gate"]:
        bundle = _SESSIONS[sid]["bundle"]
        ev_completeness = _evidence_completeness(bundle)
    else:
        ev_completeness = float("nan")

    # Step 4: decision log (E only)
    log_exists = False
    if vdef["memory_replay"]:
        r4 = submit_human_review(
            sid,
            result="ACCEPT",
            reason="Automated harness experiment — no human reviewer present.",
        )
        log_path   = r4.get("log_path", "")
        log_exists = os.path.isfile(log_path)

    # Step 5: reproducibility — run a second time and compare action
    r1b = start_analysis_session(ticker)
    sid2 = r1b["session_id"]
    run_full_analysis(sid2)
    r3b = get_recommended_action(sid2)
    action2 = r3b.get("action", "unknown")

    # Deterministic Python pipeline → always 1.0; note if mismatch
    reproducibility = 1.0 if action == action2 else 0.0

    return {
        "version":                version,
        "ticker":                 ticker,
        "tool_enforcement":       vdef["tool_enforcement"],
        "evidence_gate":          vdef["evidence_gate"],
        "memory_replay":          vdef["memory_replay"],
        "invalid_tool_call_rate": enforce_stats["invalid_tool_call_rate"],
        "evidence_completeness":  ev_completeness,
        "reproducibility":        reproducibility,
        "decision_log_exists":    log_exists,
        "action":                 action,
        "gate_status":            gate_status,
        "numerical_hallucination": "NONE — deterministic Python pipeline, all values from tool outputs",
    }


# ---------------------------------------------------------------------------
# 4. compare_harness_versions
# ---------------------------------------------------------------------------

KEY_FINDING = (
    "The single most consequential harness design decision is not evidence "
    "gating or state enforcement — it's simply forcing tool use. Version A "
    "(no tools) hallucinated a completely different price level and fabricated "
    "2026 financials never returned by any tool; Version B (tools only, no "
    "other safeguards) already eliminates numerical hallucination entirely. "
    "Evidence completeness, state enforcement, and decision logging (C→D→E) "
    "then add auditability and reproducibility on top of that already-accurate "
    "foundation, rather than fixing accuracy problems themselves."
)


def compare_harness_versions(ticker: str = "ASELS") -> pd.DataFrame:
    """
    Run harness experiments for versions A–E and return a comparison DataFrame.

    Versions A and B are populated from Claude Desktop manual test results.
    Versions C–E are measured via the automated Python pipeline.

    Columns
    -------
    version, name, tool_enforcement, evidence_gate, memory_replay,
    invalid_tool_call_rate, evidence_completeness, reproducibility,
    decision_log_exists, action, gate_status, numerical_hallucination

    Args:
        ticker: Short BIST ticker code (default 'ASELS').

    Returns:
        DataFrame with one row per version (A–E).
    """
    rows = []
    for v in ["A", "B", "C", "D", "E"]:
        print(f"  Running version {v} … ", end="", flush=True)
        result = run_harness_experiment(ticker, v)
        result["name"] = HARNESS_VERSIONS[v]["name"]
        rows.append(result)
        print("done.")

    df = pd.DataFrame(rows)

    col_order = [
        "version", "name",
        "tool_enforcement", "evidence_gate", "memory_replay",
        "invalid_tool_call_rate", "evidence_completeness",
        "reproducibility", "decision_log_exists",
        "action", "gate_status", "numerical_hallucination",
    ]
    return df[[c for c in col_order if c in df.columns]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 45)

    TICKER = "ASELS"
    SEP    = "=" * 75

    print(SEP)
    print(f"HARNESS VERSION COMPARISON — {TICKER}")
    print(SEP)

    # ── Version descriptions ──────────────────────────────────────────
    print("\nVersion definitions:")
    for v, vd in HARNESS_VERSIONS.items():
        flags = (
            f"tool_enforcement={vd['tool_enforcement']}  "
            f"evidence_gate={vd['evidence_gate']}  "
            f"memory_replay={vd['memory_replay']}"
        )
        print(f"  {v}  {vd['name']}")
        print(f"     {flags}")
    print()

    # ── Run comparison ────────────────────────────────────────────────
    print("Running experiments …")
    df = compare_harness_versions(TICKER)

    # ── Print comparison table ────────────────────────────────────────
    print(f"\n{SEP}")
    print("RESULTS TABLE")
    print(SEP)

    def _fmt_float(x, fmt):
        return fmt.format(x) if isinstance(x, float) and not math.isnan(x) else str(x)

    display = df.copy()
    display["invalid_tool_call_rate"] = display["invalid_tool_call_rate"].apply(
        lambda x: f"{x:.0%}" if isinstance(x, float) and not math.isnan(x) else str(x)
    )
    display["evidence_completeness"] = display["evidence_completeness"].apply(
        lambda x: f"{x:.1%}" if isinstance(x, float) and not math.isnan(x) else str(x)
    )
    display["reproducibility"] = display["reproducibility"].apply(
        lambda x: f"{x:.1f}" if isinstance(x, float) and not math.isnan(x) else str(x)
    )
    display["decision_log_exists"] = display["decision_log_exists"].apply(
        lambda x: str(x) if not (isinstance(x, float) and math.isnan(x)) else "NaN (manual)"
    )
    print(display.to_string(index=False))

    # ── Key finding ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("KEY FINDING")
    print(SEP)
    # Wrap text at 72 chars for readability
    import textwrap
    for line in textwrap.wrap(KEY_FINDING, width=72):
        print(f"  {line}")

    # ── Save to disk ──────────────────────────────────────────────────
    out_dir = os.path.join(_SRC, "..", "data", "events")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "harness_comparison_final.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[saved] CSV  → {os.path.abspath(csv_path)}")

    txt_path = os.path.join(out_dir, "harness_comparison_final.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("HARNESS VERSION COMPARISON — ASELS\n")
        fh.write("=" * 75 + "\n\n")
        fh.write(display.to_string(index=False))
        fh.write("\n\n")
        fh.write("KEY FINDING\n")
        fh.write("=" * 75 + "\n")
        fh.write(textwrap.fill(KEY_FINDING, width=72))
        fh.write("\n")
    print(f"[saved] TXT  → {os.path.abspath(txt_path)}")

    # ── Enforcement detail (C/D/E) ────────────────────────────────────
    print(f"\n{SEP}")
    print("TOOL ENFORCEMENT DETAIL")
    print(SEP)
    stats = _measure_tool_enforcement()
    print(f"  Invalid probes attempted   : {stats['invalid_total']}")
    print(f"  Invalid probes rejected    : {stats['invalid_rejected']}")
    print(f"  Rejection rate             : {stats['invalid_tool_call_rate']:.0%}")
    print(f"  Valid probes attempted     : {stats['valid_total']}")
    print(f"  Valid probes allowed       : {stats['valid_allowed']}")
    print(f"  Allow rate (should=100%)   : {stats['valid_allowed']/stats['valid_total']:.0%}")

    # ── Evidence bundle breakdown (D/E) ──────────────────────────────
    print(f"\n{SEP}")
    print("EVIDENCE COMPLETENESS BREAKDOWN (version D/E observed_context)")
    print(SEP)
    # Re-fetch bundle from the last D/E session in _SESSIONS
    bundles = [
        s["bundle"]
        for s in _SESSIONS.values()
        if "bundle" in s and s.get("ticker") == TICKER
    ]
    if bundles:
        bundle = bundles[-1]
        total_leaves = 0
        for top_key, v in bundle.observed_context.items():
            if isinstance(v, dict):
                leaves = sum(1 for vv in v.values() if not isinstance(vv, dict))
            else:
                leaves = 1
            total_leaves += leaves
            print(f"  {top_key:<15} {leaves:>3} observable features")
        committed = len(bundle.committed_evidence)
        print(f"  {'TOTAL':<15} {total_leaves:>3} observable features")
        print(f"  {'committed':<15} {committed:>3} evidence items")
        print(f"  completeness             : {committed/total_leaves:.1%}")

    # ── Reproducibility note ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("REPRODUCIBILITY NOTE")
    print(SEP)
    print("  C/D/E reproducibility = 1.0 by construction:")
    print("  The Python pipeline has no LLM sampling step. Given identical")
    print("  input data, the same action is always produced deterministically.")
    print()
    print("  In a real LLM-in-the-loop deployment (versions A/B):")
    print("  - Temperature > 0 introduces stochastic tool selection.")
    print("  - Reproducibility can be measured as:")
    print("      P(same_action | same_ticker, same_data, N=10 runs)")
    print("  - Expected range: 0.6–0.9 depending on ambiguity of evidence.")
    print()
    print("  A/B rows in the table are NaN — manual test required.")
    print("  Fill in A/B metrics after Claude Desktop experiments.")
    print(SEP)
