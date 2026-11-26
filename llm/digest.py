"""Model-facing digest of MCP tool outputs (used by the v2 model path and by its training data).

The full tool output is always kept in the execution trace; only the copy the model reads is shortened so that
prompt + tool results + answer fit the training window (6144 tokens). Numbers, evidence summaries and warnings are
kept verbatim; bulky per-event tables and duplicated nested dictionaries are dropped. Same function for training
and inference, so the model sees the same format in both.
"""
import json

_STAT = ("n", "low_sample_warning", "n_complete_5d", "n_positive_5d", "positive_rate_5d", "avg_fwd_1d", "avg_fwd_3d",
         "avg_fwd_5d", "avg_fwd_10d", "bounce_rate_5d", "n_excluded_incomplete_5d")
_STAT_LIST = ("n", "bounce_rate_5d", "avg_fwd_5d", "low_sample_warning")
_CATCHUP = ("n_signals", "n_baseline", "catchup_rate", "n_false_positives", "hit_rate_abs", "mean_return",
            "mean_excess_vs_sector", "baseline_catchup_rate_non_laggards", "binomial_p_vs_baseline")


def _round(v):
    return round(v, 4) if isinstance(v, float) else v


def _pick(d, keys):
    return {k: _round(d[k]) for k in keys if k in d}


def _trim(name, d):
    if not isinstance(d, dict):
        return d
    if name == "detect_reversal_events":
        d = {k: v for k, v in d.items() if k not in ("event_types", "recent_events")}
        if isinstance(d.get("standard_comparisons"), dict):
            d.pop("statistics", None)   # a filtered call: the comparisons carry the numbers
        elif isinstance(d.get("statistics"), dict):
            d["statistics"] = {k: _pick(v, _STAT_LIST) for k, v in d["statistics"].items() if isinstance(v, dict)}
        if isinstance(d.get("standard_comparisons"), dict):
            d["standard_comparisons"] = {k: _pick(v, _STAT) for k, v in d["standard_comparisons"].items() if isinstance(v, dict)}
    elif name == "run_full_analysis":
        d = {k: v for k, v in d.items() if k != "observed_context_summary"}
    elif name == "backtest_sector_catchup" and isinstance(d.get("results"), dict):
        d = dict(d)
        d["results"] = {h: _pick(v, _CATCHUP) for h, v in d["results"].items() if isinstance(v, dict)}
    elif name == "calculate_fundamental_change":
        d = {k: v for k, v in d.items() if k not in ("changes", "reaction_to_latest")}
    elif name in ("test_weekday_pattern", "test_weekday_sequence") and "evidence_summary" in d:
        d = {k: v for k, v in d.items() if k not in ("full_sample", "in_sample", "out_of_sample")}
    return d


def digest(name: str, out: str) -> str:
    """out = json.dumps(list of JSON strings) as produced by the agent loop; returns the same shape, shortened."""
    try:
        v = json.loads(out)
        items = v if isinstance(v, list) else [v]
        res = []
        for x in items:
            d = json.loads(x) if isinstance(x, str) else x
            res.append(json.dumps(_trim(name, d), ensure_ascii=False, separators=(",", ":"), default=str))
        return json.dumps(res, ensure_ascii=False)
    except Exception:
        return out
