"""
Drop training samples that would break LoRA training.

mlx-lm truncates samples to --max-seq-length. With --mask-prompt only the last
assistant message is trained, so a truncated sample can end up with zero
trainable tokens: loss = 0/0 = NaN, and one NaN gradient poisons all weights.
This script measures every sample with the real tokenizer + chat template and
keeps only samples that fit with a safety margin.

Usage: python llm/filter_dataset.py [--model mlx-community/Qwen3-4B-4bit] [--limit 5900]
"""
import argparse, json, os, shutil

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
ap.add_argument("--limit", type=int, default=5900)
ap.add_argument("--dir", default="data/llm_train")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)


def n_tokens(messages, tools):
    """Token count of the rendered chat (render to text first, then tokenize it)."""
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])


for name in ("train", "valid"):
    path = os.path.join(a.dir, f"{name}.jsonl")
    backup = path + ".unfiltered"
    if not os.path.exists(backup):
        shutil.copy(path, backup)
    kept, dropped, longest = [], 0, 0
    for line in open(backup):
        r = json.loads(line)
        last = r["messages"][-1]
        if last["role"] != "assistant" or not (last.get("content") or last.get("tool_calls")):
            dropped += 1
            continue
        n = n_tokens(r["messages"], r.get("tools"))
        if n > a.limit:
            dropped += 1
            continue
        longest = max(longest, n)
        kept.append(line)
    with open(path, "w") as f:
        f.writelines(kept)
    print(f"{name}: kept {len(kept)}, dropped {dropped}, longest kept {longest} tokens")
