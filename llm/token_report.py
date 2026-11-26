"""Token length report: baseline (system + tool schemas) and the dropped final-answer samples per tool."""
import collections, json, statistics, sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("mlx-community/Qwen3-8B-4bit")
path = sys.argv[1] if len(sys.argv) > 1 else "data/llm_train_v3/train.jsonl.unfiltered"


def n(messages, tools):
    t = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    return len(tok(t, add_special_tokens=False)["input_ids"])


rows = [json.loads(l) for l in open(path)]
r0 = rows[0]
print("baseline (system + user stub + tools):", n(r0["messages"][:2], r0["tools"]), "tokens;",
      "tools only:", n(r0["messages"][:2], r0["tools"]) - n(r0["messages"][:2], None))
by = collections.defaultdict(list)
for r in rows:
    m = r["messages"]
    if m[-1]["role"] == "assistant" and not m[-1].get("tool_calls"):
        tl = [x.get("name") for x in m if x["role"] == "tool"]
        by[tl[-1] if tl else "(none)"].append(n(m, r["tools"]))
for k, v in sorted(by.items()):
    print(f"{k:34s} n={len(v):4d} median={int(statistics.median(v)):5d} p90={int(sorted(v)[int(len(v)*.9)]):5d} max={max(v):5d} over5900={sum(x>5900 for x in v)}")
