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
import datetime as _dt
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
    # v2 adapters (model name contains "v2") are trained with the current Modelfile prompt
    fn = "Modelfile.ft" if ((model.startswith("mlx:") or model.endswith("-ft")) and "v2" not in model) else "Modelfile"
    txt = open(os.path.join(os.path.dirname(__file__), fn)).read()
    sp = txt.split('SYSTEM """', 1)[1].rsplit('"""', 1)[0].strip()
    if "v2" in model:
        # these two tools are hidden from the model (MODEL_HIDDEN_TOOLS): point it at the visible one instead
        old = ("test_weekday_pattern, test_weekday_sequence (two weekdays) and test_weekday_multiday_pattern (three or more weekdays, "
               "e.g. Mon, Tue, Wed down then Thu, Fri up) for weekday effects")
        assert old in sp
        sp = sp.replace(old, "test_weekday_pattern for weekday effects (for any weekday sequence pass its pattern argument, e.g. "
                             "'Monday, Tuesday down, then Wednesday up', and train_end as the last in-sample day)")
    return sp


# Fixed sampling seed (override with BIST_SEED) so a recorded run can be replayed with the same settings.
SEED = int(os.environ.get("BIST_SEED", "42"))
_OLLAMA_OPTIONS = {"temperature": 0.2, "num_ctx": 8192, "num_predict": 2000, "seed": SEED}   # cap runaway generations


def _raw_options() -> dict:
    """Sampling for fine-tuned raw mode. BIST_SAMPLING=qwen uses Qwen3's recommended non-thinking
    settings; low temperature (0.2) can push a small model into repetition loops."""
    opts = {"temperature": 0.2, "num_ctx": 12288, "num_predict": 1024, "seed": SEED}
    if os.environ.get("BIST_SAMPLING") == "qwen":
        opts.update({"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.1})
    return opts


_DISCLAIMER_END = ("belongs to a human reviewer.", "insan incelemesine aittir.")


def _dedupe_paragraphs(text: str) -> str:
    """Drop paragraphs that repeat an earlier paragraph verbatim (loop guard for small fine-tuned models)."""
    seen, seen_s, out = set(), set(), []
    for para in text.split("\n\n"):
        key = para.strip()
        if key and key in seen:
            continue
        seen.add(key)
        # inside a paragraph, also drop a sentence that repeats an earlier sentence of the same text verbatim
        sents, kept = re.split(r"(?<=[.!?])\s+", para), []
        for sent in sents:
            k = sent.strip()
            if len(k) > 25 and k in seen_s:
                continue
            if len(k) > 25:
                seen_s.add(k)
            kept.append(sent)
        out.append(" ".join(kept) if len(kept) != len(sents) else para)
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
                        adapter_path=os.environ.get("MLX_ADAPTER", "llm/adapters8b_v2" if "v2" in model else "llm/adapters8b"))
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        o = _raw_options()
        # Repetition penalty: a small LoRA-tuned model can loop on long final answers at low temperature
        procs = make_logits_processors(repetition_penalty=float(os.environ.get("MLX_REP", "1.0" if "v2" in model else "1.15")),
                                       repetition_context_size=256)
        from mlx_lm import stream_generate
        import mlx.core as mx
        mx.random.seed(SEED)   # same seed before every generation
        text = ""
        for chunk in stream_generate(_MLX[0], _MLX[1], prompt=prompt, max_tokens=o["num_predict"],
                                     sampler=make_sampler(temp=o["temperature"]), logits_processors=procs):
            text += chunk.text
            # Stop criterion: every training answer ends with the disclaimer; a second one means the model is looping
            if sum(text.count(m) for m in _DISCLAIMER_END) >= 2:
                break
            # Degenerate loop (the same 8-100 character unit repeated 4+ times at the end): stop and keep one copy
            if len(text) > 200 and (m_ := re.search(r"(.{8,100}?)\1{3,}$", text[-600:], re.S)):
                text = text[: len(text) - len(m_.group(0))] + m_.group(1)
                break
        return parse_raw_reply(_dedupe_paragraphs(text))
    body = {"model": model, "prompt": prompt, "raw": True, "stream": False,
            "options": _raw_options()}
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return parse_raw_reply(json.loads(r.read())["response"])


def ollama_chat(model: str, messages: list, tools: list, think: bool) -> dict:
    if model.endswith("-ft") or model.startswith("mlx:"):
        return raw_chat(model, messages, tools, think)
    body = {"model": model, "messages": messages, "tools": tools,
            "stream": False, "think": think,
            "options": _OLLAMA_OPTIONS}
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
            if ("rank_sector_relative_strength" in used and "backtest_sector_catchup" not in used and not stats.get("backtest_nudge")
                    and re.search(r"catch|yeti[sş]|kapan|closed|history|histor|geçmiş|past", question, re.I)):
                stats["backtest_nudge"] = 1
                messages.append({"role": "user", "content":
                                 "The question also asks whether similar lags closed historically. Call backtest_sector_catchup now for the "
                                 "same sector_group you used, with as_of 2026-09-10 and start 2024-01-01, then answer with its evidence."})
                continue
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
            # Window repair (steering only): a weekday-pattern call must cover the whole period up to the analysis date,
            # so the out-of-sample half named in the question is not cut off by the model's own end_date.
            if name == "test_weekday_pattern":
                m_ad = re.search(r"Analysis date:\s*(\d{2})\.(\d{2})\.(\d{4})", question)
                if m_ad:
                    ad = f"{m_ad.group(3)}-{m_ad.group(2)}-{m_ad.group(1)}"
                    key = "end_date" if "end_date" in args else ("as_of" if "as_of" in args else "end_date")
                    if args.get(key) != ad and str(args.get("train_end", "")) < ad:
                        args[key] = ad
                        stats["end_date_repaired"] = stats.get("end_date_repaired", 0) + 1
            # Variable repair: a question about the dollar/lira rate is analysed with the usdtry series.
            if name == "analyze_macro_stock_relationship" and re.search(r"USD\s*/?\s*TRY|dolar|dollar", question, re.I) \
                    and not re.search(r"policy rate|faiz|interest|TCMB|inflation|enflasyon|CPI", question, re.I) \
                    and args.get("macro_variable") != "usdtry":
                args["macro_variable"] = "usdtry"
                stats["macro_variable_repaired"] = stats.get("macro_variable_repaired", 0) + 1
            if name == "backtest_sector_catchup" and "lag_threshold" in args and not re.search(
                    r"threshold|e[şs]ik|percentage point|puan|\d\s*%|\bpp\b|lag(ged)? by", question, re.I):
                args.pop("lag_threshold", None)   # the model invented a threshold the user never gave: use the tool default
                stats["threshold_dropped"] = stats.get("threshold_dropped", 0) + 1
            if "cost_bps" in args and not re.search(r"cost|fee|slippage|commission|maliyet|komisyon|baz puan|bps|bp\b", question, re.I):
                if args.pop("cost_bps", None) not in (None, 0, 0.0):
                    stats["cost_dropped"] = stats.get("cost_dropped", 0) + 1   # the model invented a cost the user never gave
            if name == "detect_reversal_events":
                touch_q = bool(re.search(r"touch|dokun|band", question, re.I))
                state_q = bool(re.search(r"supertrend", question, re.I) and re.search(r"bearish|düşüş|negatif", question, re.I)
                               and re.search(r"below.{0,30}cloud|cloud.{0,30}below|bulutun altında|bulut altı", question, re.I))
                if touch_q and args.get("event_type"):
                    # The tool already returns the single-event and combined-condition comparisons (standard_comparisons);
                    # state filters would shrink the sample, so they are dropped and the full event log is used.
                    for k in _STATE_FILTERS:
                        if args.pop(k, None) is not None:
                            stats["filters_dropped"] = stats.get("filters_dropped", 0) + 1
                elif state_q and not touch_q:
                    # "when Supertrend was bearish and price below the cloud ..." is a state question: filter on those states
                    args.pop("event_type", None)
                    args["supertrend_state"], args["ichimoku_state"] = "bearish", "below_cloud"
                    stats["state_filters_set"] = 1
            if name == "backtest_sector_catchup":
                # Horizons: always include 5/10/20 (the model sometimes sends '20d' or '5-10' and would then repeat one value)
                got = {int(x) for x in re.findall(r"\d+", str(args.get("horizons", ""))) if 0 < int(x) <= 60}
                args["horizons"] = ",".join(str(h) for h in sorted({5, 10, 20} | got))
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
            trace.append({"tool": name, "args": args, "output": out,
                          "t": _dt.datetime.now().astimezone().isoformat(timespec="milliseconds")})
            if verbose:
                print(f"[tool] {name}({args}) -> {out[:120]}...")
            if "v2" in model:   # v2 adapters read a shortened copy of the tool output; the trace keeps the full output
                import digest
                messages.append({"role": "tool", "tool_name": name, "content": digest.digest(name, out)})
            else:
                messages.append({"role": "tool", "tool_name": name, "content": out})
        analysis_done = any(t["tool"] == "get_recommended_action" and "action" in t["output"] and "error" not in t["output"][:40]
                            for t in trace)
        # A research tool that already returned its plain-language summary lines is complete: do not drift into the
        # session workflow (start_analysis_session / run_full_analysis) for a pure event-statistics question.
        events_done = any(t["tool"] == "detect_reversal_events" and re.search(r'summary_lines\\?": \[', t["output"]) for t in trace)
        used_now = {t["tool"] for t in trace}
        state_done = any(t["tool"] == "detect_reversal_events" and re.search(r'evidence_summary\\?": \\?"', t["output"])
                         and not re.search(r'summary_lines\\?": \[', t["output"]) for t in trace)
        weekday_done = any(t["tool"] == "test_weekday_pattern" and "evidence_summary" in t["output"] for t in trace)
        if (not weekday_done and not final_only and not stats.get("pattern_nudge")
                and any(t["tool"] == "test_weekday_pattern" and not t["args"].get("pattern") for t in trace)
                and re.search(r"\b(monday|tuesday|wednesday|thursday|friday|pazartesi|salı|çarşamba|perşembe|cuma)\b", question, re.I)
                and re.search(r"then|ardından|sonra|pattern|örüntü|desen", question, re.I)):
            # The model called the weekday tool without the pattern asked: one request to repeat the call with it
            stats["pattern_nudge"] = 1
            messages.append({"role": "user", "content":
                             "The weekday call did not include the pattern from the question, so it did not test it. Call test_weekday_pattern "
                             "again with ticker, start_date, end_date (the analysis date), train_end = the last day of the in-sample period "
                             "named in the question (for example 2025-12-31) and pattern = exactly the pattern asked, written as "
                             "'condition then outcome' (for example 'Monday down, Tuesday down, Wednesday down, then Thursday up, Friday up')."})
            continue
        catchup_done = "backtest_sector_catchup" in used_now and "rank_sector_relative_strength" in used_now
        fund_done = "calculate_fundamental_change" in used_now and "get_fundamental_metrics" in used_now
        no_session = not re.search(r"recommended action|önerilen aksiyon|run_full_analysis", question, re.I)
        final_msg = None
        advice_q = bool(re.search(r"should i buy|buy or sell|buy/sell|price target|guarantee|definitely go up|sure winner|certain to rise|"
                                  r"hedef fiyat|al[ae]y[ıi]m m[ıi]|almal[ıi] m[ıi]yım|al[- ]sat|garanti", question, re.I))
        macro_done = "analyze_macro_stock_relationship" in used_now
        asked = [w for w in re.findall(r"\b[A-Z]{4,6}\b", question) if w in set(getattr(load_server(), "UNIVERSE", {}).get("ticker", []))]
        kind = None
        if advice_q and trace and no_session and (macro_done or not asked):   # a named stock waits for its macro relationship
            kind = "advice"
        elif macro_done and no_session:
            kind = "macro"
        elif events_done and no_session:
            kind = "events"
        elif state_done and no_session:
            kind = "state"
        elif weekday_done and no_session:
            kind = "weekday"
        elif fund_done and no_session:
            kind = "fund"
        elif catchup_done and no_session:
            kind = "catchup"
        if kind:
            import nudges
            final_msg = nudges.final_instruction(kind, asked[0] if (kind == "advice" and asked and macro_done) else None)
            # Language steering: an English question gets an explicit English instruction (the model sometimes opens in Turkish).
            if kind != "advice" and not re.search(r"[çğıöşüÇĞİÖŞÜ]", question) and not re.search(r"\b(ve|mı|mi|mu|mü|hangi|nedir|kaç|için|sonra|arasında)\b", question, re.I):
                final_msg += " Write the answer in English."
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
    """Run the agent loop, then apply the post-answer guard (disable with BIST_GUARD=0).

    Every call also records a full execution trace (res["execution_trace"]): prompts, sampling parameters, every model
    call and tool call with timestamps, the raw model answer, the guarded answer and what the guard changed.
    """
    import tracing
    rec = tracing.Recorder(model, question)
    kwargs["chat"] = rec.wrap(kwargs.get("chat", ollama_chat))
    res = _run_agent(question, model, *args, **kwargs)
    res["raw_answer"] = res["answer"]
    viol = []
    if os.environ.get("BIST_GUARD", "1") != "0" and res["trace"] is not None:
        import guard
        res["answer"], viol = guard.check_and_repair(question, res["answer"], res["trace"])
        res["stats"]["guard_violations"] = viol
    res["execution_trace"] = rec.finish(res, viol)
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
