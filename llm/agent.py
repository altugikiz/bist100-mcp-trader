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
import re
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


def _short(text: str, limit: int = 220) -> str:
    """First paragraph of a docstring, cut to `limit` characters (keeps prompts small)."""
    first = (text or "").strip().split("\n\n")[0].replace("\n", " ")
    first = " ".join(first.split())
    return first if len(first) <= limit else first[: limit - 1].rsplit(" ", 1)[0] + "."


def _slim_schema(schema: dict) -> dict:
    """Drop verbose titles/descriptions from a JSON schema, keep names, types, required."""
    props = {}
    for k, v in (schema.get("properties") or {}).items():
        props[k] = {kk: vv for kk, vv in v.items() if kk in ("type", "default", "enum")}
    return {"type": "object", "properties": props, "required": schema.get("required", [])}


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
                "description": _short(t.description),
                "parameters": _slim_schema(t.inputSchema),
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


_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)
_TOKENIZER = None
_MLX = None


def parse_raw_reply(text: str) -> dict:
    """Turn raw Qwen3 output (optional empty <think> block + <tool_call> JSON blocks) into a chat message."""
    text = _THINK.sub("", text, count=1)
    calls = []
    for blob in _TOOL_CALL.findall(text):
        try:
            d = json.loads(blob)
            calls.append({"type": "function", "function": {"name": d["name"], "arguments": d.get("arguments") or {}}})
        except (json.JSONDecodeError, KeyError):
            continue  # malformed call: dropped, counted as no tool call
    content = _TOOL_CALL.sub("", text).strip()
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return msg


def _system_prompt() -> str:
    txt = open(os.path.join(os.path.dirname(__file__), "Modelfile")).read()
    sp = txt.split('SYSTEM """', 1)[1].rsplit('"""', 1)[0].strip()
    return sp


def _raw_options() -> dict:
    """Sampling for fine-tuned raw mode. BIST_SAMPLING=qwen uses Qwen3's recommended non-thinking
    settings; low temperature (0.2) can push a small model into repetition loops."""
    opts = {"temperature": 0.2, "num_ctx": 12288, "num_predict": 1024}
    if os.environ.get("BIST_SAMPLING") == "qwen":
        opts.update({"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.1})
    return opts


_DISCLAIMER_END = ("belongs to a human reviewer.", "insan incelemesine aittir.")


def _dedupe_paragraphs(text: str) -> str:
    """Drop paragraphs that repeat an earlier paragraph verbatim (loop guard for small fine-tuned models)."""
    seen, out = set(), []
    for para in text.split("\n\n"):
        key = para.strip()
        if key and key in seen:
            continue
        seen.add(key)
        out.append(para)
    return "\n\n".join(out)


def raw_chat(model: str, messages: list, tools: list, think: bool) -> dict:
    """Fine-tuned models: render the exact training-time chat format ourselves and parse the reply.

    Ollama's template path drops tool calls that follow Qwen3's empty <think> block,
    so the prompt is built with the model's own chat template and sent with raw=True.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(os.environ.get("BIST_TOKENIZER", "mlx-community/Qwen3-4B-4bit"))
    msgs = [{"role": "system", "content": _system_prompt()}] + messages
    prompt = _TOKENIZER.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    if model.startswith("mlx:"):
        # In-process MLX: base 4-bit model + LoRA adapter, no export and no server (mlx_lm.server would parse
        # and swallow the <tool_call> block). Needs no extra disk and no internet.
        global _MLX
        if _MLX is None:
            from mlx_lm import load
            _MLX = load(os.environ.get("MLX_BASE", "mlx-community/Qwen3-8B-4bit"),
                        adapter_path=os.environ.get("MLX_ADAPTER", "llm/adapters8b"))
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        o = _raw_options()
        # Repetition penalty: a small LoRA-tuned model can loop on long final answers at low temperature
        procs = make_logits_processors(repetition_penalty=float(os.environ.get("MLX_REP", "1.15")),
                                       repetition_context_size=256)
        from mlx_lm import stream_generate
        text = ""
        for chunk in stream_generate(_MLX[0], _MLX[1], prompt=prompt, max_tokens=o["num_predict"],
                                     sampler=make_sampler(temp=o["temperature"]), logits_processors=procs):
            text += chunk.text
            # Stop criterion: every training answer ends with the disclaimer; a second one means the model is looping
            if sum(text.count(m) for m in _DISCLAIMER_END) >= 2:
                break
        return parse_raw_reply(_dedupe_paragraphs(text))
    body = {"model": model, "prompt": prompt, "raw": True, "stream": False,
            "options": _raw_options()}
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return parse_raw_reply(json.loads(r.read())["response"])


def ollama_chat(model: str, messages: list, tools: list, think: bool) -> dict:
    if model.endswith("-ft"):
        return raw_chat(model, messages, tools, think)
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

def _run_agent(question: str, model: str, think: bool = False,
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
    final_only = False   # set once the analysis is complete or the model starts looping
    last_sid = None      # most recent real session id returned by start_analysis_session
    for _ in range(MAX_STEPS):
        msg = chat(model, messages, [] if final_only else specs, think)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if final_only and calls:
            # Asked for the final answer but the model called tools again: do not execute them
            stats["ignored_calls_after_final"] = stats.get("ignored_calls_after_final", 0) + len(calls)
            return {"answer": (msg.get("content") or "").strip() or "(no final answer produced)",
                    "trace": trace, "stats": stats}
        if not calls:
            if not (msg.get("content") or "").strip() and trace and not nudged:
                # Small models sometimes return an empty reply right after tool output
                nudged = True
                messages.append({"role": "user", "content":
                                 "Now write the final answer for the user, using only the tool results above."})
                continue
            return {"answer": msg.get("content", ""), "trace": trace, "stats": stats}
        round_failed = False
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
            # Session grounding: a small model may garble a UUID; use the real one from this run.
            if last_sid and isinstance(args.get("session_id"), str) and args["session_id"] != last_sid:
                stats["session_corrected"] = stats.get("session_corrected", 0) + 1
                args["session_id"] = last_sid
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
            if name == "start_analysis_session" and not err:
                try:
                    last_sid = json.loads(result[0])["session_id"]
                except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                    pass
            round_failed = round_failed or bool(err) or '\\"error\\"' in out or '"error"' in out[:40]
            trace.append({"tool": name, "args": args, "output": out})
            if verbose:
                print(f"[tool] {name}({args}) -> {out[:120]}...")
            messages.append({"role": "tool", "tool_name": name, "content": out})
        analysis_done = any(t["tool"] == "get_recommended_action" and "action" in t["output"] and "error" not in t["output"][:40]
                            for t in trace)
        if not final_only and ((analysis_done and round_failed) or len(trace) >= 8):
            # The harness has already produced its answer (or the model is looping): ask for the write-up
            final_only = True
            stats["forced_final"] = 1
            messages.append({"role": "user", "content":
                             "Stop calling tools. Write the final answer for the user now, using only the tool results above."})
    return {"answer": "(stopped: step limit reached)", "trace": trace, "stats": stats}


def run_agent(question: str, model: str, *args, **kwargs) -> dict:
    """Run the agent loop, then apply the post-answer guard (disable with BIST_GUARD=0)."""
    res = _run_agent(question, model, *args, **kwargs)
    res["raw_answer"] = res["answer"]
    if os.environ.get("BIST_GUARD", "1") != "0" and res["trace"] is not None:
        import guard
        res["answer"], viol = guard.check_and_repair(question, res["answer"], res["trace"])
        res["stats"]["guard_violations"] = viol
    return res

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
