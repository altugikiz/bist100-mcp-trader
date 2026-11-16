"""Print the raw reply of a model for one question (diagnostics for template / tool-call problems)."""
import json, sys, os, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
import agent

model = sys.argv[1] if len(sys.argv) > 1 else "bist-analyst-ft"
q = sys.argv[2] if len(sys.argv) > 2 else "THYAO için tam analiz yap"
specs = agent.tool_specs(agent.load_server())
for think in (False, True):
    body = {"model": model, "stream": False, "think": think,
            "options": {"temperature": 0.2, "num_ctx": 8192},
            "messages": [{"role": "user", "content": q}], "tools": specs}
    req = urllib.request.Request(os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    m = r["message"]
    print(f"--- think={think} done_reason={r.get('done_reason')} eval_count={r.get('eval_count')}")
    print("content :", repr(m.get("content"))[:600])
    print("thinking:", repr(m.get("thinking"))[:300])
    print("tool_calls:", json.dumps(m.get("tool_calls"))[:300])
