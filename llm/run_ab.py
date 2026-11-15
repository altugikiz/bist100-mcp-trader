"""
Real LLM runs for harness versions A and B (brief Section 30), replacing the
hard-coded ASELS manual results.

  A  baseline : plain base model (no system prompt, NO tools) answers from memory.
  B  tools only: bist-analyst with just start_analysis_session + run_full_analysis
                 (no recommendation tool, no evidence bundle, no replay).

Ground truth for both = the real tool outputs of the Python pipeline for that
ticker; any number in the answer that cannot be found there counts as unsupported.

Usage: python llm/run_ab.py [--tickers ASELS TUPRS THYAO EREGL] [--runs 2]
Output: data/events/llm_ab_results.csv + printed summary.
"""
import argparse, asyncio, csv, json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import agent  # noqa: E402
from run_eval import hallucinated_numbers  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROMPT = "Analyze {t} stock: current price, technical indicators (RSI, moving averages) and fundamentals. Give a short conclusion."
B_TOOLS = {"start_analysis_session", "run_full_analysis"}


def ground_truth(server, ticker: str) -> str:
    """Run the deterministic pipeline and return its raw tool outputs as one string."""
    def call(name, args):
        r = asyncio.run(server.mcp.call_tool(name, args))
        r = r[1] if isinstance(r, tuple) and len(r) > 1 else r
        if isinstance(r, list):
            r = [getattr(c, "text", str(c)) for c in r]
        return json.dumps(r, default=str)
    o1 = call("start_analysis_session", {"ticker": ticker})
    sid = json.loads(json.loads(o1)[0])["session_id"]
    return o1 + call("run_full_analysis", {"session_id": sid})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["ASELS", "TUPRS", "THYAO", "EREGL"])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", default="llm_ab_results.csv")
    ap.add_argument("--base-model", default="qwen3:8b")
    ap.add_argument("--agent-model", default="bist-analyst")
    a = ap.parse_args()
    server = agent.load_server()
    no_tools = lambda m, msgs, tools, think: agent.ollama_chat(m, msgs, [], think)
    rows = []
    for t in a.tickers:
        truth = ground_truth(server, t)
        for run in range(1, a.runs + 1):
            for ver in ("A", "B"):
                t0 = time.time()
                if ver == "A":
                    r = agent.run_agent(PROMPT.format(t=t), a.base_model, chat=no_tools,
                                        server=server, verbose=False)
                    pool = truth
                else:
                    r = agent.run_agent(PROMPT.format(t=t), a.agent_model, server=server,
                                        verbose=False, only_tools=B_TOOLS)
                    pool = truth + " ".join(x["output"] for x in r["trace"])
                bad, tot = hallucinated_numbers(r["answer"], pool)
                rows.append({"ticker": t, "version": ver, "run": run,
                             "tool_calls": r["stats"]["tool_calls"],
                             "invalid_calls": r["stats"]["invalid_calls"],
                             "numbers_total": tot, "numbers_unsupported": bad,
                             "latency_s": round(time.time() - t0, 1),
                             "answer": r["answer"].replace("\n", " ")[:500]})
                print("   >", rows[-1]["answer"][:220], flush=True)
                print(f"{t} {ver} run{run}: calls={rows[-1]['tool_calls']} unsupported={bad}/{tot} "
                      f"{rows[-1]['latency_s']}s", flush=True)
    out = os.path.join(_ROOT, "data", "events", a.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for ver in ("A", "B"):
        rs = [r for r in rows if r["version"] == ver]
        n = sum(r["numbers_total"] for r in rs)
        print(ver, {"numeric_hallucination_rate": round(sum(r["numbers_unsupported"] for r in rs) / n, 3) if n else 0,
                    "tool_use_rate": round(sum(r["tool_calls"] > 0 for r in rs) / len(rs), 2),
                    "avg_latency_s": round(sum(r["latency_s"] for r in rs) / len(rs), 1)})
    print("saved", out)
