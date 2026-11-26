"""Assemble the course submission: submission_package/ (submission.json manifest + all attachments) and a .zip.

Prerequisites (run on the Mac, repo root):
  python llm/build_rag.py                      # RAG knowledge base JSON
  python llm/make_traces.py --model mlx:bist8b-v2   # execution traces of the 5 scenarios
  python llm/build_submission.py
The package keeps the repository's relative paths, so the code runs from the package root.
"""

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG = os.path.join(ROOT, "submission_package")
TRACES = os.path.join(ROOT, "data", "events", "execution_traces.json")
ADAPTER_DIR = os.path.join(ROOT, os.environ.get("ADAPTER", "llm/adapters8b_v2"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def py_files():
    out = []
    for base in ("llm", "src", "scripts"):
        for d, dirs, fs in os.walk(os.path.join(ROOT, base)):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", "adapters", "adapters8b", "adapters8b_v2", "fused", "fused8b")]
            out += [os.path.relpath(os.path.join(d, f), ROOT) for f in fs if f.endswith(".py")]
    return sorted(out)


def main():
    need = {"adapter": os.path.join(ADAPTER_DIR, "adapters.safetensors"),
            "rag": os.path.join(ROOT, "data", "rag", "rag_knowledge_base.json"), "traces": TRACES}
    miss = [k for k, p in need.items() if not os.path.isfile(p)]
    if miss:
        sys.exit("missing: " + ", ".join(miss) + " (see the prerequisites in this file's docstring)")
    tr = json.load(open(TRACES, encoding="utf-8"))
    bad = [s["scenario"] for s in tr["scenarios"] if "error" in s]
    ids = {s["model"]["adapter_sha256"] for s in tr["scenarios"] if "model" in s}
    if bad:
        sys.exit(f"traces contain failed scenarios {bad}; rerun make_traces.py")
    if ids != {sha(need["adapter"])}:
        sys.exit("traces were not produced with this adapter file (SHA-256 mismatch); rerun make_traces.py")

    groups = {
        "python_source": py_files() + ["requirements.txt", "llm/Modelfile", "llm/Modelfile.ft",
                                       "llm/eval_teacher.json", "llm/eval_routing.json", "llm/eval_questions.json",
                                       "docs/DATA_DICTIONARY.md"],
        "fine_tuning": ["data/llm_train_v6/train.jsonl", "data/llm_train_v6/valid.jsonl", os.path.join(os.environ.get("ADAPTER", "llm/adapters8b_v2"), "adapter_config.json"),
                        "llm/finetune.sh"],
        "model_weights": [os.path.join(os.environ.get("ADAPTER", "llm/adapters8b_v2"), "adapters.safetensors")],
        "rag": ["data/rag/rag_knowledge_base.json"],
        "execution_traces": ["data/events/execution_traces.json"],
    }
    data_dirs = ["data/universe_cache", "data/fundamentals", "data/macro", "data/fintables_asels.csv"]
    shutil.rmtree(PKG, ignore_errors=True)
    entries = {}
    for g, files in groups.items():
        entries[g] = []
        for rel in files:
            src = os.path.join(ROOT, rel)
            if not os.path.isfile(src):
                print("skipped (not found):", rel)
                continue
            dst = os.path.join(PKG, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            entries[g].append({"path": rel, "bytes": os.path.getsize(src), "sha256": sha(src)})
    support = []
    for rel in data_dirs:
        src = os.path.join(ROOT, rel)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(PKG, rel))
            for d, _, fs in os.walk(src):
                support += [os.path.relpath(os.path.join(d, f), ROOT) for f in fs]
        elif os.path.isfile(src):
            os.makedirs(os.path.dirname(os.path.join(PKG, rel)), exist_ok=True)
            shutil.copy2(src, os.path.join(PKG, rel))
            support.append(rel)

    cfg = json.load(open(os.path.join(ADAPTER_DIR, "adapter_config.json")))
    sample = tr["scenarios"][0]
    manifest = {
        "schema": "bist100-submission/1",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "project": "Integrated BIST 100 Agentic Financial Analytics Harness (CSE-481 Engineering Economics, Akdeniz University)",
        "authors": ["Kemal Altuğ İkiz", "Hakan Berkiten"],
        "data_period": "2024-01-01 to 2026-09-10", "analysis_date": "2026-09-10",
        "model": {
            "base_model_name": "Qwen3-8B (4-bit MLX conversion: mlx-community/Qwen3-8B-4bit)",
            "base_model_version": "Qwen3-8B, original release by Qwen/Alibaba Cloud (Apache-2.0)",
            "base_model_download_source": "https://huggingface.co/mlx-community/Qwen3-8B-4bit (original: https://huggingface.co/Qwen/Qwen3-8B)",
            "base_weights_included": False,
            "adapter": {"type": "LoRA", "format": "safetensors", "path": os.path.join(os.environ.get("ADAPTER", "llm/adapters8b_v2"), "adapters.safetensors"),
                        "sha256": sha(need["adapter"]), "bytes": os.path.getsize(need["adapter"])},
            "inference": "in-process MLX (mlx_lm) on Apple silicon: base 4-bit model + this adapter, "
                         "run with `python llm/agent.py \"<question>\" --model mlx:bist8b-v2`",
            "runtime_versions": {k: sample["model"].get(k) for k in ("mlx_lm", "mlx", "transformers")},
            "inference_parameters": sample["parameters"],
        },
        "fine_tuning": {
            "method": "LoRA (mlx_lm lora)", "hyperparameters": {k: cfg[k] for k in (
                "model", "iters", "batch_size", "grad_accumulation_steps", "learning_rate", "lora_parameters", "num_layers",
                "max_seq_length", "mask_prompt", "grad_checkpoint", "optimizer", "seed", "fine_tune_type")},
            "training_data": "data/llm_train_v6/train.jsonl (agent transcripts: question, tool calls, tool results, final answer), "
                             "validation: data/llm_train_v6/valid.jsonl",
            "data_generation": "llm/make_dataset.py then llm/filter_dataset.py; the five sample questions are not in the "
                               "training data (checked by pattern search)",
            "tools_covered_in_training": "18 of the 23 MCP tools (the 18 tools the model can call; submit_human_review, test_weekday_sequence and test_weekday_multiday_pattern are hidden from the model, build_evidence_bundle and get_session_replay are not trained)",
        },
        "rag": {"file": "data/rag/rag_knowledge_base.json",
                "description": "10 curated news/corporate events (ASELS, TOASO, BIMAS) with source metadata, the data dictionary "
                               "and the MCP tool descriptions. Retrieval is a direct lookup by ticker and date range; there is no "
                               "embedding or vector search, no live news feed and no speech-to-text."},
        "execution_traces": {"file": "data/events/execution_traces.json", "scenarios": len(tr["scenarios"]),
                             "produced_with": "the submitted model (base + adapter above); each record holds model name/version, "
                                              "system and user prompt, parameters (temperature, seed, context, max tokens), the full "
                                              "message list, every model call and tool call with arguments, date range, output and "
                                              "timestamp, the raw model answer, the final answer and the guard changes"},
        "files": entries,
        "supporting_data_files": {"count": len(support), "note": "price, fundamentals and macro caches the tools read; "
                                  "no model weights"},
        "academic_honesty_disclosures": [
            "No question-to-answer mapping exists in the code. llm/eval_teacher.json holds sample questions with expected "
            "tool names and must-contain checks; it is used only by llm/run_eval.py to grade answers and is never read by "
            "llm/agent.py or llm/guard.py.",
            "All numbers come from deterministic Python tools. llm/run_eval.py checks every number in an answer against the "
            "tool outputs.",
            "llm/guard.py runs after the model and may APPEND fixed notes (disclaimer, scope date, latest-quarter and "
            "research-label lines, KAP-proxy and benchmark notes) or correct a label the evidence does not support. For the "
            "weekday-pattern question it appends the tool's own evidence_summary text verbatim. The model's unmodified answer "
            "is kept as raw_answer in every trace next to final_answer, and guard.violations_repaired lists each change.",
            "llm/agent.py contains regular-expression based nudges that steer the model to call a missing tool or to finish "
            "(for example for advice-type questions). They inject instructions into the conversation; they never return an "
            "answer. Every nudge is visible in the trace's message list.",
            "The submitted adapter was trained on 18 of 23 tools; the tool-use behaviour for the other tools relies on the "
            "base model and the tool schemas.",
            "Tool outputs are shortened by llm/digest.py before the v2 model reads them (bulky per-event tables and duplicated "
            "nested dictionaries are dropped; evidence summaries, numbers and warnings are kept verbatim). The full tool output "
            "is always stored in the execution trace (tool_calls[].output); the shortened copy the model saw is in messages[].",
            "Argument repairs in llm/agent.py (disclosed, visible in the stats of each run): a missing ticker is filled from the one "
            "ticker named in the question, date formats dd.mm.yyyy are converted to ISO, synonyms of parameter names are mapped, "
            "an invented cost or lag threshold the question never gave is dropped, one nudge asks for the weekday call to be repeated "
            "when the pattern argument was left out, a weekday-pattern call whose end_date is before the analysis date gets end_date set to the "
            "analysis date given in the question, and macro_variable is set to usdtry when the question is about the dollar/lira rate and not about "
            "interest or inflation (stats keys end_date_repaired, macro_variable_repaired). For research questions written in English, the "
            "final-answer instruction ends with 'Write the answer in English.' (the model sometimes opened in Turkish). No nudge returns an answer.",
            "llm/guard.py also removes a stray <think> tag and replaces a ticker that is neither in the question nor in any tool "
            "output (for example 'ASELS.IS' in an answer about EREGL) with the single ticker named in the question; both repairs are "
            "listed in guard.violations_repaired.",
            "Training data: llm/make_dataset.py and llm/dataset_research.py build transcripts by calling the real tools on tickers "
            "that are not the five evaluation tickers (ASELS, TUPRS, GUBRF, THYAO, EREGL) and on sector groups other than Retail; "
            "answers are assembled from the tool outputs (the tool's own evidence_summary lines are quoted). The wording of the "
            "five sample questions does not appear in the training data (checked by pattern search).",
            "No speech-to-text pipeline was built; the news layer is a small curated sample (see rag).",
        ],
        "how_to_run": "see README_SUBMISSION.md",
    }
    json.dump(manifest, open(os.path.join(PKG, "submission.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    shutil.copy2(os.path.join(ROOT, "README_SUBMISSION.md"), PKG) if os.path.isfile(os.path.join(ROOT, "README_SUBMISSION.md")) else None
    z = shutil.make_archive(PKG, "zip", PKG)
    print("package:", PKG, "\nzip:", z, round(os.path.getsize(z) / 1e6, 1), "MB\nadapter sha256:", manifest["model"]["adapter"]["sha256"])


if __name__ == "__main__":
    main()
