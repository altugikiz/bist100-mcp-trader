"""Export the RAG knowledge base (documents + source metadata) to data/rag/rag_knowledge_base.json.

Contents:
  - the curated news/corporate events used by get_news_events (10 events, 3 tickers)
  - the project's data dictionary, split by heading
  - the MCP tool descriptions the model sees (docstrings), when the MCP server can be imported
Retrieval is a direct lookup by ticker/date (get_news_events reads this file); there is no embedding or vector search.
Run:  python llm/build_rag.py
"""

import datetime as dt
import hashlib
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(ROOT, "data", "rag", "rag_knowledge_base.json")


def events() -> list[dict]:
    import importlib
    ne = importlib.import_module("analysis.news_events")
    docs = []
    for i, e in enumerate(ne.NEWS_EVENTS, 1):
        docs.append({
            "id": f"event-{i:02d}", "type": "news_event", "ticker": e["ticker"], "date": e["date"],
            "category": e["category"], "text": e["summary"],
            "source": {"source_type": e["source_type"],
                       "origin": "manually curated from public reporting; summarised in original wording",
                       "url": None, "url_note": "per-event URLs were not recorded when the list was compiled"},
        })
    return docs


def data_dictionary() -> list[dict]:
    p = os.path.join(ROOT, "docs", "DATA_DICTIONARY.md")
    if not os.path.isfile(p):
        return []
    parts = re.split(r"\n(?=#{1,3} )", open(p, encoding="utf-8").read())
    return [{"id": f"datadict-{i:02d}", "type": "reference", "title": (s.splitlines() or [""])[0].lstrip("# ").strip(),
             "text": s.strip(), "source": {"source_type": "project_document", "path": "docs/DATA_DICTIONARY.md"}}
            for i, s in enumerate(parts, 1) if s.strip()]


def tool_docs() -> list[dict]:
    try:
        import agent
        specs = agent.tool_specs(agent.load_server())
    except Exception as exc:
        print("tool descriptions skipped:", repr(exc))
        return []
    return [{"id": f"tool-{s['function']['name']}", "type": "tool_description", "title": s["function"]["name"],
             "text": s["function"].get("description", ""),
             "source": {"source_type": "mcp_server_docstring", "path": "src/mcptools/mcp_server.py"}} for s in specs]


if __name__ == "__main__":
    docs = events() + data_dictionary() + tool_docs()
    kb = {"schema": "bist100-rag-knowledge-base/1", "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
          "retrieval": "direct lookup by ticker and date range (get_news_events); no embeddings, no vector index",
          "limitations": "10 curated events over ASELS, TOASO and BIMAS only; no speech-to-text or live news feed",
          "documents": docs}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(kb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(len(docs), "documents ->", OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:12])
