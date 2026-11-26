"""Run the five sample scenarios with the submitted model and write all execution traces to one JSON file.

Usage (on the Mac, offline is fine once the base model is cached):
    HF_HUB_OFFLINE=1 python llm/make_traces.py --model mlx:bist8b-v2 --out data/events/execution_traces.json
The questions are read from the first five entries of llm/eval_teacher.json (the instructor's wording).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import agent  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx:bist8b-v2")
    ap.add_argument("--out", default="data/events/execution_traces.json")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    qs = [x["q"] for x in json.load(open(os.path.join(os.path.dirname(__file__), "eval_teacher.json")))[: a.n]]
    server = agent.load_server()
    out = {"schema": "bist100-execution-traces/1", "scenarios": []}
    for i, q in enumerate(qs, 1):
        print(f"[{i}/{len(qs)}] running...", flush=True)
        try:
            r = agent.run_agent(q, a.model, server=server, verbose=False)
            out["scenarios"].append({"scenario": i, **r["execution_trace"]})
        except Exception as exc:
            out["scenarios"].append({"scenario": i, "question": q, "error": repr(exc)})
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=2, default=str)
    print("written", a.out)
