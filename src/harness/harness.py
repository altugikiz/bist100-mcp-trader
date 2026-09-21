"""
Analysis harness for BIST100 MCP Trader.

Defines the ordered state machine that governs a full analysis session,
tool-permission guards per state, and an AnalysisSession class that tracks
progress, state history, and tool call logs.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# 1. Ordered state list
# ---------------------------------------------------------------------------

HARNESS_STATES: List[str] = [
    "START",
    "SELECT_UNIVERSE",
    "LOAD_MARKET_DATA",
    "CHECK_DATA_QUALITY",
    "CALCULATE_TECHNICALS",
    "RUN_SECTOR_RELATIVE_ANALYSIS",
    "RUN_PATTERN_AND_REVERSAL_ANALYSIS",
    "LOAD_FUNDAMENTALS",
    "LOAD_MACRO_CONTEXT",
    "LOAD_NEWS_AND_SPEECH_CONTEXT",   # Phase 6 pending — tool list empty
    "BUILD_EVIDENCE",
    "PREVIEW_STRATEGIES",
    "RUN_BACKTEST",
    "RISK_GATE",
    "GENERATE_EXPLANATION",
    "HUMAN_REVIEW",
    "LOG_AND_CLOSE",
]

# States that may be jumped to from any position (bypass normal ordering)
_JUMP_ALLOWED: set = {"HUMAN_REVIEW", "LOG_AND_CLOSE"}


# ---------------------------------------------------------------------------
# 2. Tool permissions per state
# ---------------------------------------------------------------------------

STATE_TOOL_PERMISSIONS: Dict[str, List[str]] = {
    "START": [
        "log_decision",
    ],
    "SELECT_UNIVERSE": [
        "get_sector_peers",
        "get_relative_benchmark",
        "has_sector_peers",
        "log_decision",
    ],
    "LOAD_MARKET_DATA": [
        "get_stock_history",
        "fetch_universe_history",
        "check_data_quality",
        "log_decision",
    ],
    "CHECK_DATA_QUALITY": [
        "check_data_quality",
        "get_stock_history",
        "log_decision",
    ],
    "CALCULATE_TECHNICALS": [
        "calculate_indicators",
        "calculate_bollinger_bands",
        "calculate_atr",
        "calculate_kama",
        "calculate_supertrend",
        "calculate_ichimoku",
        "calculate_technical_score",
        "log_decision",
    ],
    "RUN_SECTOR_RELATIVE_ANALYSIS": [
        "rank_universe_by_sector",
        "detect_catchup_candidates",
        "calculate_relative_return",
        "log_decision",
    ],
    "RUN_PATTERN_AND_REVERSAL_ANALYSIS": [
        "build_event_log",
        "detect_bollinger_events",
        "detect_rsi_events",
        "detect_supertrend_flips",
        "detect_kama_cross_events",
        "detect_ichimoku_events",
        "test_weekday_pattern",
        "detect_multiday_sequences",
        "log_decision",
    ],
    "LOAD_FUNDAMENTALS": [
        "get_fundamental_metrics",
        "get_fundamental_score",
        "calculate_derived_ratios",
        "calculate_quarterly_changes",
        "calculate_post_disclosure_returns",
        "log_decision",
    ],
    "LOAD_MACRO_CONTEXT": [
        "build_daily_macro_series",
        "fetch_fx_series",
        "log_decision",
    ],
    # Phase 6 pending — news/speech NLP layer not yet implemented
    "LOAD_NEWS_AND_SPEECH_CONTEXT": [],
    "BUILD_EVIDENCE": [
        "build_evidence_bundle",
        "build_fundamental_reaction_dataset",
        "summarize_reaction_patterns",
        "classify_market_regime",
        "log_decision",
    ],
    "PREVIEW_STRATEGIES": [
        "build_evidence_bundle",
        "preview_strategy",
        "log_decision",
    ],
    "RUN_BACKTEST": [
        "run_backtest_summary",
        "backtest_catchup_signal",
        "log_decision",
    ],
    "RISK_GATE": [
        "check_risk_limits",
        "log_decision",
    ],
    "GENERATE_EXPLANATION": [
        "build_evidence_bundle",
        "generate_explanation",
        "log_decision",
    ],
    "HUMAN_REVIEW": [
        # All read-only tools permitted during human review
        "get_stock_history",
        "get_fundamental_metrics",
        "get_fundamental_score",
        "build_evidence_bundle",
        "run_backtest_summary",
        "log_decision",
    ],
    "LOG_AND_CLOSE": [
        "log_decision",
    ],
}


# ---------------------------------------------------------------------------
# 3. AnalysisSession class
# ---------------------------------------------------------------------------

class AnalysisSession:
    """
    Represents a single end-to-end analysis session for one ticker.

    Attributes:
        ticker        : The stock under analysis (e.g. 'ASELS').
        session_id    : UUID4 string, unique per session.
        current_state : The active state name from HARNESS_STATES.
        state_history : Ordered list of (timestamp, state) tuples.
        tool_call_log : List of tool-call records added via record_tool_call().
    """

    def __init__(self, ticker: str) -> None:
        self.ticker:        str        = ticker.upper()
        self.session_id:    str        = str(uuid.uuid4())
        self.current_state: str        = "START"
        self.state_history: List[dict] = [
            {"timestamp": _now(), "state": "START"}
        ]
        self.tool_call_log: List[dict] = []

    def __repr__(self) -> str:
        return (
            f"AnalysisSession(ticker={self.ticker!r}, "
            f"state={self.current_state!r}, "
            f"steps={len(self.state_history)}, "
            f"tool_calls={len(self.tool_call_log)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 4. advance_state
# ---------------------------------------------------------------------------

def advance_state(session: AnalysisSession, next_state: str) -> None:
    """
    Transition the session to next_state, enforcing ordering rules.

    Allowed transitions:
    - Forward by exactly one step in HARNESS_STATES.
    - Jump to HUMAN_REVIEW or LOG_AND_CLOSE from any state.

    Args:
        session:    The active AnalysisSession.
        next_state: Target state name.

    Raises:
        ValueError: If next_state is unknown, or the transition violates
                    the ordering rules (backward move or skipping states).
    """
    if next_state not in HARNESS_STATES:
        raise ValueError(
            f"Unknown state '{next_state}'. "
            f"Valid states: {HARNESS_STATES}"
        )

    current_idx = HARNESS_STATES.index(session.current_state)
    next_idx    = HARNESS_STATES.index(next_state)

    # Allow jump targets regardless of position
    if next_state in _JUMP_ALLOWED:
        pass
    elif next_idx != current_idx + 1:
        direction = "backward" if next_idx < current_idx else "skips states"
        raise ValueError(
            f"Invalid transition from '{session.current_state}' to '{next_state}' "
            f"({direction}). "
            f"Expected next state: '{HARNESS_STATES[current_idx + 1]}'."
        )

    session.current_state = next_state
    session.state_history.append({"timestamp": _now(), "state": next_state})


# ---------------------------------------------------------------------------
# 5. is_tool_allowed
# ---------------------------------------------------------------------------

def is_tool_allowed(session: AnalysisSession, tool_name: str) -> bool:
    """
    Return True if tool_name is permitted in the session's current state.

    Args:
        session:   The active AnalysisSession.
        tool_name: Name of the MCP tool function to check.

    Returns:
        bool — True when the tool appears in STATE_TOOL_PERMISSIONS for the
        current state, False otherwise.
    """
    allowed = STATE_TOOL_PERMISSIONS.get(session.current_state, [])
    return tool_name in allowed


# ---------------------------------------------------------------------------
# 6. record_tool_call
# ---------------------------------------------------------------------------

def record_tool_call(
    session:        AnalysisSession,
    tool_name:      str,
    input_summary:  str,
    output_summary: str,
) -> None:
    """
    Append a tool-call record to session.tool_call_log.

    Args:
        session:        The active AnalysisSession.
        tool_name:      Name of the tool that was called.
        input_summary:  Brief description of the arguments passed.
        output_summary: Brief description of the result returned.
    """
    session.tool_call_log.append({
        "timestamp":      _now(),
        "state":          session.current_state,
        "tool_name":      tool_name,
        "input_summary":  input_summary,
        "output_summary": output_summary,
    })


# ---------------------------------------------------------------------------
# Entry point — self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("HARNESS SELF-TEST")
    print("=" * 60)

    # ── Create session ────────────────────────────────────────────────
    sess = AnalysisSession("ASELS")
    print(f"\nCreated  : {sess}")

    # ── Valid forward transitions ─────────────────────────────────────
    advance_state(sess, "SELECT_UNIVERSE")
    print(f"→ {sess.current_state}")

    # is_tool_allowed: 'get_stock_history' should be DENIED in SELECT_UNIVERSE
    tool = "get_stock_history"
    allowed = is_tool_allowed(sess, tool)
    print(f"\n  is_tool_allowed('{tool}', state='SELECT_UNIVERSE') → {allowed}")
    assert not allowed, "Expected False in SELECT_UNIVERSE"

    advance_state(sess, "LOAD_MARKET_DATA")
    print(f"→ {sess.current_state}")

    # is_tool_allowed: 'get_stock_history' should be ALLOWED in LOAD_MARKET_DATA
    allowed = is_tool_allowed(sess, tool)
    print(f"  is_tool_allowed('{tool}', state='LOAD_MARKET_DATA')  → {allowed}")
    assert allowed, "Expected True in LOAD_MARKET_DATA"

    # ── Record a tool call ────────────────────────────────────────────
    record_tool_call(
        sess,
        tool_name      = "get_stock_history",
        input_summary  = "ticker=ASELS, start=2024-01-01, end=2025-12-31",
        output_summary = "749 rows of OHLCV data returned",
    )
    print(f"\n  tool_call_log entries: {len(sess.tool_call_log)}")
    print(f"  Last entry: {sess.tool_call_log[-1]}")

    # ── Jump to HUMAN_REVIEW from any state ───────────────────────────
    advance_state(sess, "HUMAN_REVIEW")
    print(f"\n→ jumped to {sess.current_state} (allowed from any state)")

    # ── Invalid transition: START → RUN_BACKTEST (skips states) ───────
    sess2 = AnalysisSession("ASELS")
    print(f"\nInvalid transition test on fresh session (state=START):")
    try:
        advance_state(sess2, "RUN_BACKTEST")
        print("  ERROR: should have raised ValueError")
    except ValueError as e:
        print(f"  OK — ValueError raised: {e}")

    # ── Invalid transition: backward move ─────────────────────────────
    sess3 = AnalysisSession("ASELS")
    advance_state(sess3, "SELECT_UNIVERSE")
    advance_state(sess3, "LOAD_MARKET_DATA")
    print(f"\nBackward transition test (LOAD_MARKET_DATA → SELECT_UNIVERSE):")
    try:
        advance_state(sess3, "SELECT_UNIVERSE")
        print("  ERROR: should have raised ValueError")
    except ValueError as e:
        print(f"  OK — ValueError raised: {e}")

    # ── State history ─────────────────────────────────────────────────
    print(f"\nState history for sess:")
    for entry in sess.state_history:
        print(f"  {entry['timestamp']}  {entry['state']}")

    print("\nAll assertions passed.")
