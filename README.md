# BIST 100 MCP Trader

A CSE-481 Engineering Economics term project covering a **30-stock BIST 100 universe** across four mandatory research scenarios: technical/fundamental scoring, sector-relative and macro/news context layers, walk-forward backtesting with transaction costs, and a stateful MCP agent harness. The project builds a point-in-time fundamental pipeline (anti-look-ahead), curated news-event reaction analysis, and a five-version harness experiment (A–E) documenting how forcing tool use cuts numerical hallucination (60.2% → 3.7% unsupported numbers with the local Qwen3 8B model). The entire analysis stack is exposed as an **MCP (Model Context Protocol) AI agent** with 23 tool endpoints. The project runs its **own local LLM** (Ollama + Qwen3 8B, custom `bist-analyst` model) that drives full evidence-bundle analysis sessions — with human review and JSON audit logging — through structured tool calls; Claude Desktop can use the same server too.

---

## Features

- **Yahoo Finance data collection** — daily OHLCV history via `yfinance`
- **Technical indicators** — SMA, EMA, RSI, MACD with a composite technical score (0–5)
- **Support & resistance detection** — pivot high/low detection via `scipy.signal.argrelextrema`
- **Fundamental data integration** — quarterly income statement, balance sheet, cash flow, and ratio data sourced from Fintables and stored locally in CSV
- **Combined scoring strategy** — weighted Total Score from technical + fundamental signals; configurable thresholds generate BUY / HOLD / SELL signals
- **Backtesting engine** — chronological backtest with look-ahead-bias prevention, equity curves, Sharpe ratio, max drawdown, and win rate
- **Sensitivity analysis** — sweep over BUY thresholds and score weight configurations to find the optimal parameter set
- **Optional ML prediction** — Logistic Regression and Random Forest classifiers with chronological train/test split
- **MCP AI agent** — 23 tool endpoints exposed via FastMCP over stdio; supports full evidence-bundle sessions with human review and JSON audit logging
- **Own local LLM (submitted model)** — Qwen3-8B-4bit + LoRA adapter (`llm/adapters8b_v2`), run in-process with MLX (`--model mlx:bist8b-v2`); the Ollama `bist-analyst` base model is kept as the comparison baseline. Agent loop, evaluation and trace scripts in `llm/`

---

## Project Structure

```
bist100-mcp-trader/
├── data/
│   ├── fintables_asels.csv           # Quarterly fundamental data (2024Q1–2026Q2)
│   ├── fundamentals/                 # Per-ticker fundamental CSVs
│   ├── universe_cache/               # Cached daily OHLCV CSVs (one per ticker)
│   ├── macro/                        # Macro series (TCMB rate, USD/TRY, CPI)
│   ├── rag/                          # rag_knowledge_base.json (44 documents with source metadata)
│   ├── events/                       # Event logs, harness comparison and LLM eval outputs (derived CSVs = 2025-12-31 snapshot used in the report)
│   └── decision_log/                 # Per-session JSON audit logs
├── llm/
│   ├── adapters8b_v2/                # Submitted LoRA adapter (adapters.safetensors, SHA-256 c854af57...) + adapter_config.json
│   ├── agent.py                      # Agent loop (Ollama or in-process MLX) over the MCP tools, with traces and nudges
│   ├── guard.py / nudges.py / digest.py / tracing.py   # Post-answer guard, harness steering texts, model-facing tool-output digest, execution-trace recorder
│   ├── make_dataset.py / dataset_research.py / filter_dataset.py / finetune.sh   # Training-data generation and LoRA fine-tuning (report 13.2, 13.5)
│   ├── make_traces.py                # Execution traces of the 5 scenarios -> data/events/execution_traces.json
│   ├── build_rag.py / build_submission.py   # RAG knowledge base JSON and the submission package (submission.json + zip)
│   ├── run_eval.py / eval_teacher.json / eval_questions.json   # Evaluation metrics and question sets
│   ├── Modelfile                     # bist-analyst baseline: Qwen3 8B (Ollama) + system prompt + parameters
│   └── run_ab.py                     # Real A/B runs (no tools vs. tools) for harness versions A and B
├── results/
│   ├── support_resistance.png        # Pivot high/low chart
│   ├── final_equity_curve.png        # Equity curves — final config (BUY=3.0), report Figure 11.2
│   └── sensitivity_threshold.png     # BUY threshold sensitivity chart
├── src/
│   ├── config.py                     # Central constants (tickers, paths, thresholds)
│   ├── data/
│   │   ├── universe.py               # 30-stock universe definition
│   │   └── data_collection.py        # Fetches daily OHLCV data from Yahoo Finance
│   ├── analysis/
│   │   ├── indicators.py             # SMA, EMA, RSI, MACD, Bollinger, KAMA, Supertrend, Ichimoku
│   │   ├── support_resistance.py     # Pivot-based support/resistance detection
│   │   ├── fundamental_analysis.py   # Loads CSV, computes derived ratios and fundamental score
│   │   ├── reversal_events.py        # Technical reversal event detection and statistics
│   │   ├── sector_analysis.py        # Sector-relative ranking and benchmark comparison
│   │   ├── calendar_patterns.py      # Day-of-week and seasonal return patterns
│   │   ├── point_in_time.py          # Point-in-time fundamental alignment (anti-look-ahead)
│   │   ├── fundamental_reaction.py   # Post-earnings price reactions and QoQ changes
│   │   ├── macro_context.py          # Macro regime data (TCMB rate, USD/TRY, CPI)
│   │   └── news_events.py            # Curated news events and post-news return analysis
│   ├── strategy/
│   │   ├── strategy.py               # Combines scores into Total Score and generates signals
│   │   ├── backtest.py               # Walk-forward backtest with transaction costs
│   │   ├── sensitivity_analysis.py   # Threshold and weight sensitivity sweep
│   │   └── ml_prediction.py          # ML feature engineering, model training, and evaluation
│   ├── harness/
│   │   ├── harness.py                # 17-state analysis state machine and tool enforcement
│   │   ├── evidence.py               # EvidenceBundle and RiskGate data-quality checks
│   │   ├── action_space.py           # 8-label action space and decision tree
│   │   ├── decision_log.py           # JSON audit log: record, load, replay, human review
│   │   └── harness_experiments.py    # Harness versions A–E comparison experiment
│   └── mcptools/
│       └── mcp_server.py             # MCP server exposing 23 tools via FastMCP (stdio)
├── README_SUBMISSION.md              # Submission run instructions
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/altugikiz/bist100-mcp-trader.git
cd bist100-mcp-trader
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes `mcp<2` and `anthropic`, which are required only for the MCP server. They are safe to install alongside the data/analysis packages.

---

## Usage

Run each module independently from the project root:

```bash
# Fetch and preview OHLCV data
python src/data/data_collection.py

# Compute and display technical indicators
python src/analysis/indicators.py

# Detect support/resistance levels and save chart
python src/analysis/support_resistance.py

# Load fundamental data and compute derived ratios
python src/analysis/fundamental_analysis.py

# Build combined score and generate signals
python src/strategy/strategy.py

# Run full backtest and plot equity curves
python src/strategy/backtest.py

# Run threshold and weight sensitivity sweep
python src/strategy/sensitivity_analysis.py

# Train and evaluate ML models
python src/strategy/ml_prediction.py

# Start MCP server (stdio transport — used by Claude Desktop)
python src/mcptools/mcp_server.py

# Run 5-step end-to-end test (ASELS, no server needed)
python src/mcptools/mcp_server.py --test
```

---

## Submitted Model (Qwen3-8B-4bit + LoRA adapter, MLX)

The model for the course submission is the base `mlx-community/Qwen3-8B-4bit` (original: `Qwen/Qwen3-8B`, Apache-2.0) with the LoRA adapter in `llm/adapters8b_v2/` (SHA-256 `c854af57b30bf68ef88bf5395a948242f8d5e4445a1a8ea2b280a951ec678b4f`). It runs in-process with MLX on Apple silicon; temperature 0.2, seed 42.

```bash
source venv/bin/activate
HF_HUB_OFFLINE=1 python llm/agent.py "<question>" --model mlx:bist8b-v2          # one question
HF_HUB_OFFLINE=1 python llm/make_traces.py --model mlx:bist8b-v2                 # 5 scenario execution traces
HF_HUB_OFFLINE=1 python llm/run_eval.py --teacher --model mlx:bist8b-v2          # 25-question evaluation
python llm/build_rag.py && python llm/build_submission.py                        # RAG file and submission package
```

Training: `llm/make_dataset.py` (transcripts built by calling the real tools) -> `llm/filter_dataset.py` -> `llm/finetune.sh` (`mlx_lm lora`, rank 8, scale 20, 8 layers, lr 1e-4, 250 iterations, batch 1 x 4 accumulation, max length 8192, seed 0). Files: `data/llm_train_v6/train.jsonl` (2,204 samples) and `valid.jsonl` (239). The adapter covers 18 of the 23 MCP tools; the held-out evaluation tickers (ASELS, TUPRS, GUBRF, THYAO, EREGL) and their sector groups are not in the training data, and the five sample questions are not in it either.

Evaluation (`llm/eval_teacher.json`, 25 questions, evaluation only, not used for training; numbers compared with tool output):

| Metric | Value |
|---|---|
| Tool-selection accuracy | 0.92 |
| Tool-use rate | 0.96 (the last question is a "guarantee me a winner" request, refused without tools) |
| Invalid tool-call rate | 0.0 |
| Numeric hallucination rate | 0.001 |
| Unsupported conclusion rate | 0.0 |
| Look-ahead rate | 0.0 |
| Rule violations, raw model text | 1.0 |
| Rule violations after guard | 0.0 |
| Average latency | 115 s per question |

Known weaknesses (not hidden): the raw answers violate the formatting/disclaimer rules in every question and are repaired by `llm/guard.py`; for the weekday-pattern scenario the guard replaces the model's text with the tool's own evidence summary (the raw draft stays in the trace); two questions (3 and 8, the ASELS Bollinger touches) were counted as a routing miss because the model skipped `calculate_indicators` and called `detect_reversal_events` directly, although the answer uses the tool's numbers; the adapter sometimes emits a stray `<think>` tag (removed by the guard); Q5-type answers can drift into listing the tool output. The adapter was trained over several rounds; an earlier round (25-question tool selection 0.84, unsupported conclusions 0.04) was replaced by the submitted one.

Disclosed harness steering (`llm/agent.py`, `llm/guard.py`, `llm/nudges.py`, `llm/digest.py`; none of them returns an answer): regex nudges that ask for a missing tool call or a final answer, argument repairs (missing ticker, date formats, invented thresholds or costs dropped, weekday `end_date` set to the analysis date, `usdtry` chosen for dollar/lira questions), an English-language instruction for English research questions, a shortened copy of tool outputs for the model (the full output stays in the trace), and the post-answer guard. The execution traces (`data/events/execution_traces.json`) keep the raw model answer, the messages, every model and tool call and the guard changes. The knowledge base is `data/rag/rag_knowledge_base.json` (44 documents; direct lookup, no vector search); there is no speech-to-text layer and the news layer is 10 curated events.

The sections below describe the Ollama base model (`bist-analyst`) and the earlier fine-tuning experiments, which remain as the comparison baseline.

---

## Local LLM Setup (Ollama + Qwen3 8B)

Requires macOS/Linux with about 8 GB of free memory (tested on a 16 GB MacBook Air M4).

```bash
brew install ollama && brew services start ollama   # or install from ollama.com
ollama pull qwen3:8b
ollama create bist-analyst -f llm/Modelfile

# Ask the agent a question (it calls the MCP tools itself)
python llm/agent.py "Run a full analysis for THYAO" --model bist-analyst

# Evaluate tool-use quality on the 20 questions -> data/events/llm_eval_<model>.csv
python llm/run_eval.py --model bist-analyst

# Teacher-style sample test (25 questions, analysis date 2026-09-10) with reproducibility check (each asked twice)
python llm/run_eval.py --model bist-analyst --teacher --repeat 2

# Real A/B runs (A: base model without tools, B: bist-analyst with tools)
python llm/run_ab.py
```

Latest teacher-style run (base `bist-analyst` + guard, 25 questions × 2 = 50 answers; the first five are the instructor's sample questions in English): tool selection 100%, invented-number rate 0.2% under a strict check (3 of 1,423 numbers, all model-computed complement percentages) (every decimal and percentage compared with tool output), look-ahead 0%, tool-sequence reproducibility 100%, guarded rule violations 0%. The tool-use rate is 96% because the last question (a 'guarantee me a winner' request) is correctly refused without tools. On the 14-question routing set: tool selection 100%, look-ahead 0%, guarded violations 0%; the only unsupported numbers there are generic definitions answered without tools (for example RSI 70/30). Answers are not word-for-word identical between runs; the numbers are compared with tool output.

Routing notes: the local model sees `test_weekday_pattern` (which takes a multi-day `pattern`) rather than the narrower `test_weekday_sequence` / `test_weekday_multiday_pattern`, which stay on the MCP server; `detect_reversal_events` state filters are dropped by the agent because the tool already returns the single-event and combined-condition comparisons; dotted `dd.mm.yyyy` dates are read day-first.

### Fine-tuning experiment (LoRA on Qwen3 4B and 8B)

`llm/make_dataset.py` builds the training set from real MCP tool transcripts (question → tool call → tool output → grounded answer; evaluation tickers held out; refusal cases end in `ANALYSIS_UNSAFE` / `INVESTIGATE`), `llm/filter_dataset.py` drops samples that would truncate to zero trainable tokens (the cause of the NaN loss), and `llm/finetune.sh` trains a LoRA adapter with MLX (final 8B round: 1,535 train / 156 valid samples, rank 8, 8 layers, lr 5e-5, 150 iterations, validation loss 1.685 → 0.251). The fine-tuned 8B runs in-process (`--model mlx:bist8b-ft`, base 4-bit model + adapter, fully local, no export).

Base 8B (`bist-analyst`) vs fine-tuned 8B on the unseen 14-question set (`llm/eval_routing.json`) and the 20-question set, raw model text:

| Metric | Base (14 / 20) | Fine-tuned (14 / 20) |
|---|---|---|
| Tool-selection accuracy | 1.00 / 1.00 | 0.86 / 0.80 |
| Unsupported conclusion rate | 7.1% / 0% | 14.3% / 0% |
| Rule violations (raw) | 100% / 85% | 36% / 20% |
| Numeric hallucination | 9.0% / 0.0% | 6.5% / 1.2% |
| Invalid tool-call rate | 7.7% / 0% | 5.9% / 0% |
| Latency per question | 18 s / 31 s | 71 s / 104 s |

`llm/guard.py` (post-answer guard, on by default, `BIST_GUARD=0` disables it) brings rule violations to 0% for both. Course submission: the later fine-tuned adapter (`llm/adapters8b_v2`, section "Submitted Model" above); the base 8B + guard and this first fine-tuning round are the measured comparison. The 4B fine-tune was a negative result. Price data is read offline from `data/universe_cache`; Yahoo Finance is only used with `BIST_ALLOW_NETWORK=1`. Details: report Sections 13.2–13.3.

### Backtest Performance

| Metric | Strategy | ASELS Buy & Hold | BIST100 |
|---|---|---|---|
| Cumulative Return | **216.3%** | 389.7% | 47.2% |
| Sharpe Ratio | **2.06** | 2.24 | 0.94 |
| Max Drawdown | **−19.71%** | −21.13% | −22.86% |
| Win Rate | 54.8% | 53.3% | 50.9% |
| Days Invested | 250 / 499 | 499 / 499 | 499 / 499 |

> The strategy significantly outperforms BIST100 (+169pp) and improves drawdown and Sharpe ratio vs. buy-and-hold, while spending only 50% of the period in position.

### ML Prediction (test set: Nov–Dec 2025, 38 days)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 63.16% | 84.21% | 59.26% | 69.57% |
| Random Forest | 31.58% | 100.00% | 3.70% | 7.14% |
| Majority Baseline | 71.05% | — | — | — |

> The dataset (188 usable rows after NaN removal) is too small for tree-based models. Logistic Regression shows meaningful precision (84%) but does not beat the majority-class baseline on accuracy. Expanding the historical window is the recommended next step.

---

## Deliverables

| Folder | Contents |
|---|---|
| [`docs/report/`](docs/report/) | Final term project report (`ASELS_BIST100_Harness_Final_Report.docx`) |
| [`docs/presentation/`](docs/presentation/) | Project presentation slides (`BIST100_Harness_Presentation.pptx`) |
| [`llm/adapters8b_v2/`](llm/adapters8b_v2/) | Submitted LoRA adapter (safetensors) and its config; base model: `mlx-community/Qwen3-8B-4bit` |
| [`data/events/execution_traces.json`](data/events/execution_traces.json) | Execution traces of the five scenarios (prompts, parameters, tool calls, raw and final answers) |
| [`data/rag/rag_knowledge_base.json`](data/rag/rag_knowledge_base.json) | RAG knowledge base (44 documents with source metadata) |
| `submission.json` (built by `llm/build_submission.py`) | Course submission manifest and package; see `README_SUBMISSION.md` |
| [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | Field-level documentation for all data sources, CSV schemas, and module outputs |

---

## Disclaimer

This project is for **educational purposes only** and does not constitute investment advice. All analysis is performed on historical data; past performance is not indicative of future results. Use at your own risk.

---

## Author

**Altuğ** — CSE-481 Engineering Economics
