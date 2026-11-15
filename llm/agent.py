"""
Local LLM agent loop: Ollama model <-> MCP tools of this project.

The LLM only decides which tool to call and explains the results.  All numbers
come from the deterministic MCP tools in src/mcptools/mcp_server.py.

Usage:
    ollama create bist-analyst -f llm/Modelfile
    python llm/agent.py "Analyse ASELS and tell me the recommended action"
    python llm/agent.py --model qwen3.5:9b "Which Financials stock lags its sector?"
"""

import argparse
import asyncio
import json
import os
import sys
import urllib.request

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "mcptools"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MAX_STEPS = 12


def load_server():
    """Import the MCP server module (tools are plain Python functions)."""
    import warnings
    warnings.filterwarnings("ignore")
    import mcp_server
    return mcp_server


# The human review step must come from a human, never from the model.
HUMAN_ONLY_TOOLS = {"submit_human_review"}


def tool_specs(server, allow_human_review: bool = False, only: set | None = None) -> list[dict]:
    """Convert the MCP tool list into Ollama/OpenAI-style function schemas."""
    tools = asyncio.run(server.mcp.list_tools())
    if not allow_human_review:
        tools = [t for t in tools if t.name not in HUMAN_ONLY_TOOLS]
    if only is not None:
        tools = [t for t in tools if t.name in only]
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": (t.description or "").strip(),
                "parameters": t.inputSchema,
            },
        }
        for t in tools
    ]


def _validate_args(spec: dict, args: dict) -> str | None:
    """Return an error string when args do not match the tool schema."""
    schema = spec["function"]["parameters"]
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            return f"missing required argument '{req}'"
    for k in args:
        if k not in props:
            return f"unknown argument '{k}'"
    return None


def ollama_chat(model: str, messages: list, tools: list, think: bool) -> dict:
    body = {"model": model, "messages": messages, "tools": tools,
            "stream": False, "think": think,
            "options": {"temperature": 0.2, "num_ctx": 8192}}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]



def _ground_ticker(arg: str, question: str) -> str:
    """Return the ticker named in the question when the model's argument differs from it."""
    import re
    known = set(load_server().UNIVERSE["ticker"]) if hasattr(load_server(), "UNIVERSE") else set()
    named = {w for w in re.findall(r"[A-Z]{4,6}", question) if w in known}
    if len(named) == 1 and arg.upper() not in named and arg.upper() not in known:
        return next(iter(named))
    return arg

def run_agent(question: str, model: str, think: bool = False,
              chat=ollama_chat, server=None, verbose: bool = True,
              allow_human_review: bool = False, only_tools: set | None = None) -> dict:
    """Run one question; return answer, tool trace and call statistics.

    submit_human_review is hidden from the model unless allow_human_review is
    set, which should only be done when a human supplied the verdict.
    """
    server = server or load_server()
    specs = tool_specs(server, allow_human_review, only_tools)
    by_name = {s["function"]["name"]: s for s in specs}
    messages = [{"role": "user", "content": question}]
    trace, stats = [], {"tool_calls": 0, "invalid_calls": 0, "no_tool_text_replies": 0}

    nudged = False
    for _ in range(MAX_STEPS):
        msg = chat(model, messages, specs, think)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            if not (msg.get("content") or "").strip() and trace and not nudged:
                # Small models sometimes return an empty reply right after tool output
                nudged = True
                messages.append({"role": "user", "content":
                                 "Now write the final answer for the user, using only the tool results above."})
                continue
            return {"answer": msg.get("content", ""), "trace": trace, "stats": stats}
        for call in calls:
            fn = call["function"]
            name, args = fn["name"], fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            # Argument grounding: a small model may garble a ticker (EREGL -> ERELG).
            # If the user's question names exactly one known ticker, use it verbatim.
            if isinstance(args.get("ticker"), str):
                fixed = _ground_ticker(args["ticker"], question)
                if fixed != args["ticker"]:
                    stats["ticker_corrected"] = stats.get("ticker_corrected", 0) + 1
                    args["ticker"] = fixed
            stats["tool_calls"] += 1
            err = None if name in by_name else f"unknown tool '{name}'"
            err = err or _validate_args(by_name[name], args)
            if err:
                stats["invalid_calls"] += 1
                result = {"error": err}
            else:
                try:
                    result = asyncio.run(server.mcp.call_tool(name, args))
                except Exception as exc:  # tool crashed (e.g. a mistyped ticker): feed the error back
                    stats["invalid_calls"] += 1
                    result = {"error": f"tool failed: {str(exc)[:200]}"}
                # call_tool returns (content_list, structured) or content_list
                result = result[1] if isinstance(result, tuple) and len(result) > 1 else result
                if isinstance(result, list):
                    result = [getattr(c, "text", str(c)) for c in result]
            out = json.dumps(result, default=str)
            trace.append({"tool": name, "args": args, "output": out})
            if verbose:
                print(f"[tool] {name}({args}) -> {out[:120]}...")
            messages.append({"role": "tool", "tool_name": name, "content": out})
    return {"answer": "(stopped: step limit reached)", "trace": trace, "stats": stats}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="bist-analyst")
    ap.add_argument("--think", action="store_true", help="enable the model's thinking mode")
    ap.add_argument("--allow-human-review", action="store_true",
                    help="expose submit_human_review (only when YOU give the verdict in the question)")
    a = ap.parse_args()
    res = run_agent(a.question, a.model, think=a.think,
                    allow_human_review=a.allow_human_review)
    print("\n=== ANSWER ===\n" + res["answer"])
    print("\n=== STATS ===", res["stats"])
