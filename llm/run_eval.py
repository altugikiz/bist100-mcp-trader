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
import guard  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_NUM = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:[.,]\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for tok in _NUM.findall(text):
        try:
            out.append(float(tok.replace(",", "") if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", tok) else tok.replace(",", ".")))
        except ValueError:
            pass
    return out


def unsupported_conclusion(answer: str, trace: list) -> bool:
    """True when the answer states a conclusion label the harness did not produce.

    A conclusion label is any allowed action except INVESTIGATE / ANALYSIS_UNSAFE (which are the
    safe fallbacks).  It is supported only if get_recommended_action returned exactly that action."""
    harness = set()
    for t in trace:
        if t["tool"] == "get_recommended_action":
            harness |= set(re.findall(r'"action\\*"\s*:\s*\\*"([A-Z_]+)', t["output"]))
    shown = guard.strip_negated_labels(re.sub(r"(DÜZELTME|CORRECTION):[^\n]*", "", answer.upper()))
    said = {a for a in guard.ALLOWED_ACTIONS if a not in ("INVESTIGATE", "ANALYSIS_UNSAFE")
            and re.search(rf"\b{a}\b", shown)}
    return bool(said - harness)


def hallucinated_numbers(answer: str, tool_outputs: str) -> tuple[int, int]:
    """Return (unsupported, total) numbers in the answer; small ints (<=10) and years are ignored."""
    pool = _numbers(tool_outputs)
    # dates such as 10.09.2026, 31.12 or 2026-09-10 are not measured values
    answer = re.sub(r"\b\d{1,2}\.\d{1,2}(?:\.\d{2,4})?\b|\b\d{4}-\d{2}-\d{2}\b", " ", answer)
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
    + [{"must_any": ["test_weekday_pattern", "test_weekday_sequence", "test_weekday_multiday_pattern"], "allowed": ["test_weekday_pattern", "test_weekday_sequence", "test_weekday_multiday_pattern"]}] * 3
    + [{"must_any": ["detect_reversal_events"], "allowed": ["detect_reversal_events"]}] * 3
    + [{"must_any": ["get_macro_context"], "allowed": ["get_macro_context"]}] * 2
    + [{"must_any": ["get_news_events"], "allowed": ["get_news_events"]}]
    + [{"must_any": ["check_data_quality"], "allowed": ["check_data_quality"]}] * 2
    + [{"must_any": ["run_full_analysis", "get_recommended_action"], "allowed": _FLOW}]
)


def routing_ok(called: list, spec: dict) -> bool:
    """True when the called tools satisfy the expected routing (see EXPECTED_MAIN)."""
    must = spec.get("must_any", [])
    must_all = spec.get("must_all", [])
    if spec.get("any_ok"):             # either answering directly or using allowed tools is acceptable
        return all(t in spec["allowed"] for t in called)
    if not must and not must_all:     # no-tool question: any tool call is over-calling
        return not called
    return (any(t in must for t in called) if must else True) and all(t in called for t in must_all) \
        and all(t in spec["allowed"] for t in called)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bist-analyst")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--routing", action="store_true", help="run the unseen-question routing set (eval_routing.json)")
    ap.add_argument("--teacher", action="store_true",
                    help="run the analysis-date (2026-09-10) question set mirroring the 5 sample test question types")
    ap.add_argument("--repeat", type=int, default=1, help="ask every question N times and report reproducibility")
    a = ap.parse_args()

    here = os.path.dirname(__file__)
    if a.teacher:
        specs = json.load(open(os.path.join(here, "eval_teacher.json")))
        qs = [x["q"] for x in specs]
    elif a.routing:
        specs = json.load(open(os.path.join(here, "eval_routing.json")))
        qs = [x["q"] for x in specs]
    else:
        qs = json.load(open(os.path.join(here, "eval_questions.json")))
        specs = EXPECTED_MAIN
    if a.limit:
        qs, specs = qs[: a.limit], specs[: a.limit]
    server = agent.load_server()
    rows = []
    for q, spec in [(q, sp) for q, sp in zip(qs, specs) for _ in range(max(1, a.repeat))]:
        t0 = time.time()
        try:
            r = agent.run_agent(q, a.model, think=a.think, server=server, verbose=False)
        except Exception as exc:          # one failed question must not lose the whole run
            r = {"answer": f"(run failed: {exc})", "trace": [], "stats": {"tool_calls": 0, "invalid_calls": 0}}
        lat = time.time() - t0
        outs = " ".join(t["output"] for t in r["trace"])
        raw = r.get("raw_answer", r["answer"])
        bad, tot = hallucinated_numbers(raw, outs)  # measured on the raw model text, before the guard
        unsup = unsupported_conclusion(raw, r["trace"])
        _, viol_raw = guard.check_and_repair(q, raw, r["trace"])
        _, viol_fin = guard.check_and_repair(q, r["answer"], r["trace"])
        viol_raw = [v for v in viol_raw if v != "numbers"]
        viol_fin = [v for v in viol_fin if v != "numbers"]
        called = [t["tool"] for t in r["trace"]]
        content_ok = all(any(alt.lower() in r["answer"].lower() for alt in k.split("|")) for k in spec.get("contains", []))
        # look-ahead: a date after the analysis date in any tool output or in the answer
        late = re.compile(r"2026-(09-(1[1-9]|2\d|30)|1[0-2]-\d\d)|2027-\d\d-\d\d|2026Q[34]")
        lookahead = bool(late.search(outs)) or bool(late.search(raw))
        rows.append({
            "question": q, "tool_calls": r["stats"]["tool_calls"], "tools": " > ".join(called),
            "routing_ok": routing_ok(called, spec), "content_ok": content_ok,
            "invalid_calls": r["stats"]["invalid_calls"],
            "tool_args": " || ".join(f'{t["tool"]}({json.dumps(t["args"], ensure_ascii=False)[:160]})' for t in r["trace"]),
            "tool_errors": " || ".join(f'{t["tool"]}({json.dumps(t["args"], ensure_ascii=False)[:150]}): {t["output"][:150]}'
                                       for t in r["trace"] if '"error"' in t["output"][:60]),
            "numbers_total": tot, "numbers_unsupported": bad, "unsupported_conclusion": unsup, "lookahead": lookahead, "lookahead_hit": (late.search(outs) or late.search(raw) or [None])[0] if lookahead else "",
            "violations_raw": "|".join(viol_raw), "violations_guarded": "|".join(viol_fin),
            "latency_s": round(lat, 1), "answer": r["answer"].replace("\n", " ")[:1800],
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
        "unsupported_conclusion_rate": round(sum(r["unsupported_conclusion"] for r in rows) / n, 3),
        "rule_violation_rate_raw": round(sum(bool(r["violations_raw"]) for r in rows) / n, 3),
        "rule_violation_rate_guarded": round(sum(bool(r["violations_guarded"]) for r in rows) / n, 3),
        "tool_selection_accuracy": round(sum(r["routing_ok"] for r in rows) / n, 3),
        "required_phrase_rate": round(sum(r["content_ok"] for r in rows) / n, 3),
        "lookahead_rate": round(sum(r["lookahead"] for r in rows) / n, 3),
        "avg_tool_calls": round(calls / n, 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 1),
    }
    if a.repeat > 1:   # reproducibility: identical tool sequence on every repeat of a question
        k = a.repeat
        same = sum(len({rows[i + j]["tools"] for j in range(k)}) == 1 for i in range(0, n, k))
        summary["repeat"] = k
        summary["tool_sequence_reproducible_rate"] = round(same / (n // k), 3)
    tag = "_teacher" if a.teacher else ("_routing" if a.routing else "")
    out = os.path.join(_ROOT, "data", "events", f"llm_eval{tag}_{a.model.replace(':', '_')}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2)); print("saved", out)
