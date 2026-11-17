"""Send one raw prompt to the local MLX server and print exactly what comes back (diagnostics)."""
import json, os, sys, urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import agent  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

q = sys.argv[1] if len(sys.argv) > 1 else "Analyse THYAO with the harness and tell me the recommended action."
server = agent.load_server()
tools = agent.tool_specs(server)
tok = AutoTokenizer.from_pretrained("mlx-community/Qwen3-4B-4bit")
msgs = [{"role": "system", "content": agent._system_prompt()}, {"role": "user", "content": q}]
for label, kw in (("default", {}), ("enable_thinking=False", {"enable_thinking": False})):
    prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, **kw)
    print(f"\n=== {label}: prompt tail = {prompt[-90:]!r}")
    body = {"prompt": prompt, "max_tokens": 300, "temperature": 0.2, "stream": False}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    print(json.dumps(out, ensure_ascii=False)[:900])
