"""Execution-trace recorder: one JSON-serialisable record per question.

The record holds everything needed to audit an answer: model identity, sampling parameters, system and user prompt,
the full message list (including the harness's own nudges), every model call and every tool call with timestamps,
the model's raw answer, the guarded final answer and what the guard changed.
"""

import copy
import datetime as dt
import hashlib
import json
import os
import time

SCHEMA = "bist100-execution-trace/1"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATE_KEYS = ("date", "start", "end", "from", "to", "period", "as_of", "analysis")


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def _sha256(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _version(pkg: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None


def _unwrap(out):
    """Tool outputs are json.dumps(list of pretty-printed JSON strings): decode both layers when possible."""
    try:
        v = json.loads(out) if isinstance(out, str) else out
    except Exception:
        return out
    if isinstance(v, list):
        res = []
        for x in v:
            try:
                res.append(json.loads(x) if isinstance(x, str) else x)
            except Exception:
                res.append(x)
        return res[0] if len(res) == 1 else res
    return v


def model_identity(model: str) -> dict:
    if model.startswith("mlx:") or model.endswith("-ft"):
        base = os.environ.get("MLX_BASE", "mlx-community/Qwen3-8B-4bit")
        adapter = os.environ.get("MLX_ADAPTER", "llm/adapters8b_v2" if "v2" in model else "llm/adapters8b")
        ap = adapter if os.path.isabs(adapter) else os.path.join(_ROOT, adapter)
        return {
            "name": model, "kind": "base model + LoRA adapter (in-process MLX)",
            "base_model": base, "base_model_origin": "Qwen/Qwen3-8B (Alibaba Cloud, Apache-2.0), 4-bit MLX conversion",
            "base_model_source": "https://huggingface.co/" + base,
            "adapter_path": adapter,
            "adapter_sha256": _sha256(os.path.join(ap, "adapters.safetensors")),
            "adapter_config_sha256": _sha256(os.path.join(ap, "adapter_config.json")),
            "tokenizer": os.environ.get("BIST_TOKENIZER", "mlx-community/Qwen3-4B-4bit"),
            "mlx_lm": _version("mlx-lm"), "mlx": _version("mlx"), "transformers": _version("transformers"),
        }
    return {"name": model, "kind": "Ollama model (base weights + Modelfile system prompt, no adapter)",
            "base_model": "qwen3:8b", "base_model_source": "https://ollama.com/library/qwen3",
            "modelfile_sha256": _sha256(os.path.join(_ROOT, "llm", "Modelfile"))}


def parameters(model: str) -> dict:
    import agent
    if model.startswith("mlx:") or model.endswith("-ft"):
        o = agent._raw_options()
        return {"temperature": o["temperature"], "seed": agent.SEED, "num_ctx": o["num_ctx"],
                "max_new_tokens": o["num_predict"],
                "repetition_penalty": float(os.environ.get("MLX_REP", "1.0" if "v2" in model else "1.15")), "repetition_context_size": 256,
                "thinking_mode": "off (Qwen3 empty think block)", "max_agent_steps": agent.MAX_STEPS}
    o = agent._OLLAMA_OPTIONS
    return {"temperature": o["temperature"], "seed": o["seed"], "num_ctx": o["num_ctx"],
            "max_new_tokens": o["num_predict"], "max_agent_steps": agent.MAX_STEPS}


class Recorder:
    def __init__(self, model: str, question: str):
        self.model, self.question = model, question
        self.started = _now()
        self.t0 = time.time()
        self.calls = []
        self._messages = None

    def wrap(self, chat):
        def inner(model, messages, tools, think):
            self._messages = messages          # same list the agent keeps appending to
            t = time.time()
            ts = _now()
            msg = chat(model, messages, tools, think)
            self.calls.append({"index": len(self.calls) + 1, "started": ts, "seconds": round(time.time() - t, 2),
                               "n_messages_in": len(messages), "tools_offered": len(tools or []),
                               "reply_content": msg.get("content", ""),
                               "reply_tool_calls": msg.get("tool_calls") or []})
            return msg
        return inner

    def finish(self, res: dict, violations: list) -> dict:
        import agent
        tool_calls = []
        for i, t in enumerate(res.get("trace") or [], 1):
            args = t.get("args") or {}
            tool_calls.append({
                "index": i, "tool": t["tool"], "timestamp": t.get("t"), "arguments": args,
                "date_range": {k: v for k, v in args.items() if any(d in k.lower() for d in _DATE_KEYS)},
                "output": _unwrap(t.get("output")),
            })
        raw, final = res.get("raw_answer", res["answer"]), res["answer"]
        return {
            "schema": SCHEMA, "question": self.question, "started": self.started, "finished": _now(),
            "total_seconds": round(time.time() - self.t0, 1),
            "model": model_identity(self.model), "parameters": parameters(self.model),
            "system_prompt": agent._system_prompt(self.model), "user_prompt": self.question,
            "messages": copy.deepcopy(self._messages) if self._messages is not None else [],
            "model_calls": self.calls, "tool_calls": tool_calls,
            "raw_answer": raw, "final_answer": final,
            "guard": {"violations_repaired": violations, "answer_changed": raw != final,
                      "note": "The guard only appends fixed notes or, for the weekday question, appends the tool's own "
                              "evidence_summary; the model's raw answer is kept in raw_answer."},
            "stats": res.get("stats"),
        }
