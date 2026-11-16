"""Compare Ollama's template path with the exact training-format prompt (raw mode)."""
import json, sys, os, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
import agent
from transformers import AutoTokenizer

model = sys.argv[1] if len(sys.argv) > 1 else "bist-analyst-ft"
q = sys.argv[2] if len(sys.argv) > 2 else "THYAO için tam analiz yap"
url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
tok = AutoTokenizer.from_pretrained("mlx-community/Qwen3-4B-4bit")
system = open(os.path.join(os.path.dirname(__file__), "Modelfile")).read().split('SYSTEM """', 1)[1].rsplit('"""', 1)[0].strip()
specs = agent.tool_specs(agent.load_server())
msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
prompt = tok.apply_chat_template(msgs, tools=specs, tokenize=False, add_generation_prompt=True)
print("prompt tail:", repr(prompt[-120:]))
body = {"model": model, "prompt": prompt, "raw": True, "stream": False,
        "options": {"temperature": 0.2, "num_predict": 200, "num_ctx": 8192}}
req = urllib.request.Request(url + "/api/generate", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=300))
print("RAW RESPONSE:", repr(r["response"]))
