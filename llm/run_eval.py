"""
Evaluate a local Ollama model against the MCP tools (brief Section 30 metrics).

Metrics per model:
  - tool_use_rate          : share of questions where at least one tool was called
  - invalid_tool_call_rate : invalid calls / all calls (unknown tool or bad args)
  - numeric_hallucination  : share of numbers in the answer that cannot be found
                             in any tool output (within 0.6% or as a percentage)
  - avg_tool_calls, avg_latency_s

Usage:  python llm/run_eval.py --model bist-analyst [--limit 5]
Output: data/events/llm_eval_<model>.csv and a printed summary.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import agent  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for tok in _NUM.findall(text):
        try:
            out.append(float(tok.replace(",", ".")))
        except ValueError:
            pass
    return out


def hallucinated_numbers(answer: str, tool_outputs: str) -> tuple[int, int]:
    """Return (unsupported, total) numbers in the answer; small ints (<=10) and years are ignored."""
    pool = _numbers(tool_outputs)
    total = bad = 0
    for x in _numbers(answer):
        if abs(x) <= 10 or 1990 <= abs(x) <= 2100:
            continue
        total += 1
        ok = any(
            abs(x - p) <= 0.006 * max(abs(p), 1e-9) or abs(x - p * 100) <= 0.006 * max(abs(p * 100), 1e-9)
            for p in pool
        )
        bad += 0 if ok else 1
    return bad, total


# Expected routing for the 20 questions in eval_questions.json: the model must call at least one tool of
# "must" and may only call tools of "allowed" (so a weekday question answered by a full analysis counts as wrong).
_FLOW = ["start_analysis_session", "run_full_analysis", "get_recommended_action", "build_evidence_bundle", "check_data_quality"]
EXPECTED_MAIN = (
    [{"must_any": ["run_full_analysis"], "allowed": _FLOW}] * 4
    + [{"must_any": ["run_full_analysis", "check_data_quality"], "allowed": _FLOW}]
    + [{"must_any": ["rank_sector_relative_strength"], "allowed": ["rank_sector_relative_strength"]}] * 3
    + [{"must_any": ["test_weekday_pattern"], "allowed": ["test_weekday_pattern"]}] * 3
    + [{"must_any": ["detect_reversal_events"], "allowed": ["detect_reversal_events"]}] * 3
    + [{"must_any": ["get_macro_context"], "allowed": ["get_macro_context"]}] * 2
    + [{"must_any": ["get_news_events"], "allowed": ["get_news_events"]}]
    + [{"must_any": ["check_data_quality"], "allowed": ["check_data_quality"]}] * 2
    + [{"must_any": ["run_full_analysis", "get_recommended_action"], "allowed": _FLOW}]
)


def routing_ok(called: list, spec: dict) -> bool:
    """True when the called tools satisfy the expected routing (see EXPECTED_MAIN)."""
    must = spec.get("must_any", [])
    if not must:                      # no-tool question: any tool call is over-calling
        return not called
    return any(t in must for t in called) and all(t in spec["allowed"] for t in called)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bist-analyst")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--routing", action="store_true", help="run the unseen-question routing set (eval_routing.json)")
    a = ap.parse_args()

    here = os.path.dirname(__file__)
    if a.routing:
        specs = json.load(open(os.path.join(here, "eval_routing.json")))
        qs = [x["q"] for x in specs]
    else:
        qs = json.load(open(os.path.join(here, "eval_questions.json")))
        specs = EXPECTED_MAIN
    if a.limit:
        qs, specs = qs[: a.limit], specs[: a.limit]
    server = agent.load_server()
    rows = []
    for q, spec in zip(qs, specs):
        t0 = time.time()
        r = agent.run_agent(q, a.model, think=a.think, server=server, verbose=False)
        lat = time.time() - t0
        outs = " ".join(t["output"] for t in r["trace"])
        bad, tot = hallucinated_numbers(r["answer"], outs)
        called = [t["tool"] for t in r["trace"]]
        content_ok = all(k.lower() in r["answer"].lower() for k in spec.get("contains", []))
        rows.append({
            "question": q, "tool_calls": r["stats"]["tool_calls"], "tools": " > ".join(called),
            "routing_ok": routing_ok(called, spec), "content_ok": content_ok,
            "invalid_calls": r["stats"]["invalid_calls"],
            "numbers_total": tot, "numbers_unsupported": bad,
            "latency_s": round(lat, 1), "answer": r["answer"].replace("\n", " ")[:600],
        })
        print(f"{len(rows):2}/{len(qs)} calls={rows[-1]['tool_calls']} invalid={rows[-1]['invalid_calls']} "
              f"unsupported={bad}/{tot} route={'ok' if rows[-1]['routing_ok'] else 'WRONG'} {lat:.0f}s")

    n = len(rows)
    calls = sum(r["tool_calls"] for r in rows)
    nums = sum(r["numbers_total"] for r in rows)
    summary = {
        "model": a.model, "n_questions": n,
        "tool_use_rate": round(sum(r["tool_calls"] > 0 for r in rows) / n, 3),
        "invalid_tool_call_rate": round(sum(r["invalid_calls"] for r in rows) / calls, 3) if calls else None,
        "numeric_hallucination_rate": round(sum(r["numbers_unsupported"] for r in rows) / nums, 3) if nums else 0.0,
        "tool_selection_accuracy": round(sum(r["routing_ok"] for r in rows) / n, 3),
        "required_phrase_rate": round(sum(r["content_ok"] for r in rows) / n, 3),
        "avg_tool_calls": round(calls / n, 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 1),
    }
    tag = "_routing" if a.routing else ""
    out = os.path.join(_ROOT, "data", "events", f"llm_eval{tag}_{a.model.replace(':', '_')}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2)); print("saved", out)
