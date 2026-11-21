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
# Narrower weekday tools stay on the MCP server, but the local model sees only test_weekday_pattern
# (which accepts a multi-day `pattern`), so one question type always routes to one tool.
MODEL_HIDDEN_TOOLS = {"test_weekday_sequence", "test_weekday_multiday_pattern"}
_STATE_FILTERS = ("rsi_state", "volume_regime", "supertrend_state", "ichimoku_state")


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
        tools = [t for t in tools if t.name not in HUMAN_ONLY_TOOLS | MODEL_HIDDEN_TOOLS]
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


_DESCRIBES_DATA = re.compile(r"(JSON|data|output|results?)\s+(you|that you)('ve| have)?\s+(provided|shared|given)|provided (appears|represents|is)|appears to be a structured", re.I)
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


def _system_prompt(model: str = "") -> str:
    # The fine-tuned adapters were trained with the earlier prompt (Modelfile.ft); keep it for them.
    fn = "Modelfile.ft" if (model.startswith("mlx:") or model.endswith("-ft")) else "Modelfile"
    txt = open(os.path.join(os.path.dirname(__file__), fn)).read()
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
    msgs = [{"role": "system", "content": _system_prompt(model)}] + messages
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
            "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 2000}}   # cap runaway generations
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["message"]
        except (TimeoutError, OSError):          # includes socket timeouts and URLError
            continue
    return {"role": "assistant", "content": "(model timeout: no answer was produced)"}



_PLAN_RE = re.compile(r"(let's begin|let me (start|begin)|i will (start|begin|follow|call|run|first)|first step|"
                      r"following steps|could you (please )?provide|please provide|please specify|need to know|"
                      r"the user'?s (question|request)|according to the rules|the rules (say|require|state))", re.I)


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
    seen_calls = set()
    plan_nudged = False
    desc_nudged = False
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
            if not trace and not plan_nudged and _PLAN_RE.search(msg.get("content") or ""):
                # The model described a plan or asked for details instead of calling a tool: tool arguments have defaults
                plan_nudged = True
                messages.append({"role": "user", "content":
                 "Do not describe a plan, discuss these instructions or ask for more details. Every tool parameter has a default: "
                                 "if the question names a stock, call the best-matching tool now; otherwise answer directly."})
                continue
            if trace and not desc_nudged and _DESCRIBES_DATA.search((msg.get("content") or "")[:300]):
                # The model describes the JSON structure instead of answering the question: ask again once
                desc_nudged = True
                stats["describe_nudge"] = 1
                messages.append({"role": "user", "content":
                                 "Do not describe the data or its structure. Answer the original question directly in the "
                                 "question's language, quoting the exact numbers from the tool results above."})
                continue
            used = {t["tool"] for t in trace}
            last_by_tool = {t["tool"]: t["output"] for t in trace}
            failed = [n for n, o in last_by_tool.items() if re.search(r'\\?"error\\?":', o[:60])]
            if failed and not stats.get("error_nudge") and not (failed == ["start_analysis_session"]):
                # A research tool returned an error and the model is about to answer anyway (it would have to invent numbers)
                stats["error_nudge"] = 1
                messages.append({"role": "user", "content":
                                 f"The tool call {failed[0]} returned an error. Do not answer from it and do not invent numbers: "
                                 "read the error message, fix the arguments and call the tool again."})
                continue
            if (not trace and not stats.get("route_nudge") and not stats.get("macro_nudge")
                    and re.search(r"\b(monday|tuesday|wednesday|thursday|friday|pazartesi|salı|çarşamba|perşembe|cuma)\b", question, re.I)
                    and re.search(r"pattern|örüntü|down|up|düşüş|yüksel", question, re.I)):
                # Weekday-pattern question answered without any tool call: ask once for the right tool
                stats["route_nudge"] = 1
                messages.append({"role": "user", "content":
                                 "Call test_weekday_pattern now: ticker from the question, start_date 2024-01-01, end_date 2026-09-10, "
                                 "train_end = the last day of the in-sample period named in the question (for example 2025-12-31), and "
                                 "pattern = exactly the pattern asked, written as 'condition then outcome' (for example "
                                 "'Monday down, Tuesday down, Wednesday down, then Thursday up, Friday up'). Do not answer without the tool."})
                continue
            named = {w for w in re.findall(r"[A-Z]{4,6}", question) if w in set(getattr(load_server(), "UNIVERSE", {}).get("ticker", []))} \
                if hasattr(load_server(), "UNIVERSE") else set()
            if (re.search(r"hedef fiyat|price target|should i buy|al[ae]y[ıi]m m[ıi]|sat[ae]y[ıi]m m[ıi]", question, re.I)
                    and len(named) == 1 and "get_macro_context" not in used and not stats.get("macro_nudge")):
                # Buy/target-price request naming a stock: the safe answer is built on macro tool evidence, not a bare refusal
                stats["macro_nudge"] = 1
                messages.append({"role": "user", "content":
                                 "Do not answer yet. Call get_macro_context (start_date 2024-01-01, end_date 2026-09-10) and "
                                 f"analyze_macro_stock_relationship for {next(iter(named))} first, then answer with the historical "
                                 "relationship (include the latest CBRT policy rate and the USD/TRY move from get_macro_context), no recommendation, no price target, and end with INVESTIGATE."})
                continue
            if ("calculate_fundamental_change" in used and "get_fundamental_metrics" not in used
                    and "get_fundamental_metrics" in by_name and not stats.get("metrics_nudge")):
                stats["metrics_nudge"] = 1
                messages.append({"role": "user", "content":
                                 "Before answering, also call get_fundamental_metrics for the same ticker (period 2026Q2) "
                                 "to confirm the latest quarter, then write the answer: list EVERY quarter from quarterly_history of the earlier "
                                 "calculate_fundamental_change result, ONE short line per quarter, for example '2024Q1: EBITDA margin 3.7%, "
                                 "Net Debt/EBITDA n/a, abnormal 1d -2.0%, 5d +0.1%, 20d -9.7% vs XU100 / -9.7% vs sector', then one sentence "
                                 "on improvement_vs_reaction (did improving quarters react better?), "
                                 "flag the Q2 2026 20-day return as incomplete, make no causal claim, and use only numbers printed by the tools."})
                continue
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
            for key in ("ticker", "symbol"):   # legacy tools name the argument "symbol"
                if isinstance(args.get(key), str):
                    fixed = _ground_ticker(args[key], question)
                    if fixed != args[key]:
                        stats["ticker_corrected"] = stats.get("ticker_corrected", 0) + 1
                        args[key] = fixed
            # Date grounding: when the question names no year, a date range invented by the model is dropped
            # so the tool's own default window (the full local data) applies.
            if not re.search(r"\b(19|20)\d{2}\b", question):
                for key in ("start", "end", "start_date", "end_date", "date", "as_of", "train_end"):
                    if key in args and key in by_name.get(name, {}).get("function", {}).get("parameters", {}).get("properties", {}):
                        args.pop(key)
                        stats["dates_dropped"] = stats.get("dates_dropped", 0) + 1
            # Session grounding: a small model may garble a UUID; use the real one from this run.
            if last_sid and isinstance(args.get("session_id"), str) and args["session_id"] != last_sid:
                stats["session_corrected"] = stats.get("session_corrected", 0) + 1
                args["session_id"] = last_sid
            # Ticker repair: a missing 'ticker' is filled from the one ticker named in the question
            if name in by_name and "ticker" in by_name[name]["function"]["parameters"].get("properties", {}) and "ticker" not in args:
                known = set(load_server().UNIVERSE["ticker"]) if hasattr(load_server(), "UNIVERSE") else set()
                named = {w for w in re.findall(r"[A-Z]{4,6}", question) if w in known}
                if len(named) == 1:
                    args["ticker"] = next(iter(named))
                    stats["ticker_filled"] = stats.get("ticker_filled", 0) + 1
            # Argument repair: map common synonyms onto the tool's real parameter names; unknown extras are dropped.
            if name in by_name:
                props = by_name[name]["function"]["parameters"].get("properties", {})
                syn = {"start_date": "start", "from_date": "start", "end_date": "as_of", "end": "as_of", "date": "as_of",
                       "as_of_date": "as_of", "analysis_date": "as_of", "symbol": "ticker", "stock": "ticker"}
                for k in list(args):
                    if k in props:
                        continue
                    tgt = syn.get(k)
                    if tgt and tgt in props and tgt not in args:
                        args[tgt] = args.pop(k)
                    else:
                        args.pop(k)
                    stats["args_repaired"] = stats.get("args_repaired", 0) + 1
            if "cost_bps" in args and not re.search(r"cost|fee|slippage|commission|maliyet|komisyon|baz puan|bps|bp\b", question, re.I):
                if args.pop("cost_bps", None) not in (None, 0, 0.0):
                    stats["cost_dropped"] = stats.get("cost_dropped", 0) + 1   # the model invented a cost the user never gave
            if name == "detect_reversal_events":
                # The tool already returns the single-event and combined-condition comparisons (standard_comparisons);
                # state filters would shrink the sample, so they are dropped and the full event log is used.
                for k in _STATE_FILTERS:
                    if args.pop(k, None) is not None:
                        stats["filters_dropped"] = stats.get("filters_dropped", 0) + 1
            for k, v in list(args.items()):
                m_ = re.fullmatch(r"\s*(\d{1,2})\.(\d{1,2})\.(\d{4})\s*", v) if isinstance(v, str) else None
                if m_:   # dd.mm.yyyy -> ISO, so no tool can read it month-first
                    args[k] = f"{m_.group(3)}-{int(m_.group(2)):02d}-{int(m_.group(1)):02d}"
                    stats["dates_normalised"] = stats.get("dates_normalised", 0) + 1
            sig = (name, json.dumps(args, sort_keys=True, default=str))
            if sig in seen_calls:
                # Exact repeat of an earlier call in this run: do not execute or count it again
                stats["duplicate_calls_skipped"] = stats.get("duplicate_calls_skipped", 0) + 1
                messages.append({"role": "tool", "tool_name": name,
                                 "content": "Duplicate call skipped: the same call was already made, use its earlier result."})
                continue
            seen_calls.add(sig)
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
        # A research tool that already returned its plain-language summary lines is complete: do not drift into the
        # session workflow (start_analysis_session / run_full_analysis) for a pure event-statistics question.
        events_done = any(t["tool"] == "detect_reversal_events" and re.search(r'summary_lines\\?": \[', t["output"]) for t in trace)
        used_now = {t["tool"] for t in trace}
        weekday_done = any(t["tool"] == "test_weekday_pattern" and "evidence_summary" in t["output"] for t in trace)
        catchup_done = "backtest_sector_catchup" in used_now and "rank_sector_relative_strength" in used_now
        no_session = not re.search(r"recommended action|önerilen aksiyon|run_full_analysis", question, re.I)
        final_msg = None
        if events_done and no_session:
            final_msg = ("The tool results above already answer the question. Write the final answer now, in the question's language, "
                         "quoting the summary_lines numbers exactly: total events, how many rose within 5 days and how many did not, "
                         "the average forward returns at 1, 3, 5 and 10 days, the two comparisons (RSI trough + high volume; bearish "
                         "Supertrend + below the Ichimoku cloud), any small-sample warning, and the statement about events excluded "
                         "because their forward window is incomplete.")
        elif weekday_done and no_session:
            final_msg = ("The tool result above already answers the question. Write the final answer now, in the question's language, "
                         "quoting evidence_summary exactly: how often the full pattern occurred, the in-sample and out-of-sample "
                         "probability against the unconditional probability with their sample sizes, the mean and median hold return "
                         "and the return after costs, the multiple-testing risk, and say clearly whether the pattern is stable.")
        elif catchup_done and no_session:
            final_msg = ("The tool results above already answer the question. Write the final answer now, in the question's language: "
                         "name the most lagging stock with its 5/10/20-day lag scores, then quote the backtest evidence_summary for "
                         "5, 10 and 20 days (catch-up rate, baseline, p-value, false positives), the label verdict, and the "
                         "laggard_diagnosis (structural or temporary, with its EBITDA margin and macro numbers). Say the label "
                         "POTENTIAL_CATCH_UP_CANDIDATE only if the label verdict supports it.")
        if not final_only and final_msg:
            final_only = True
            stats["forced_final"] = 1
            messages.append({"role": "user", "content": final_msg})
            continue
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
