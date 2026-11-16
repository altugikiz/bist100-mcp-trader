# BIST 100 MCP Trader

A CSE-481 Engineering Economics term project covering a **30-stock BIST 100 universe** across four mandatory research scenarios: technical/fundamental scoring, sector-relative and macro/news context layers, walk-forward backtesting with transaction costs, and a stateful MCP agent harness. The project builds a point-in-time fundamental pipeline (anti-look-ahead), curated news-event reaction analysis, and a five-version harness experiment (A–E) documenting how forcing tool use cuts numerical hallucination (60.2% → 3.7% unsupported numbers with the local Qwen3 8B model). The entire analysis stack is exposed as an **MCP (Model Context Protocol) AI agent** with 18 tool endpoints. The project runs its **own local LLM** (Ollama + Qwen3 8B, custom `bist-analyst` model) that drives full evidence-bundle analysis sessions — with human review and JSON audit logging — through structured tool calls; Claude Desktop can use the same server too.

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
- **MCP AI agent** — 18 tool endpoints exposed via FastMCP over stdio; supports full evidence-bundle sessions with human review and JSON audit logging
- **Own local LLM** — Ollama + Qwen3 8B (`bist-analyst`), agent loop and evaluation scripts in `llm/`

---

## Project Structure

```
bist100-mcp-trader/
├── data/
│   ├── fintables_asels.csv           # Quarterly fundamental data (8 quarters, 2024Q1–2025Q4)
│   ├── fundamentals/                 # Per-ticker fundamental CSVs
│   ├── universe_cache/               # Cached daily OHLCV CSVs (one per ticker)
│   ├── macro/                        # Macro series (TCMB rate, USD/TRY, CPI)
│   ├── events/                       # Event logs and harness comparison outputs
│   └── decision_log/                 # Per-session JSON audit logs
├── llm/
│   ├── Modelfile                     # bist-analyst: Qwen3 8B + system prompt + parameters
│   ├── agent.py                      # Ollama tool-calling agent loop over the MCP tools
│   ├── eval_questions.json           # 20 evaluation questions
│   ├── run_eval.py                   # Tool-use / invalid-call / unsupported-number metrics
│   └── run_ab.py                     # Real A/B runs (no tools vs. tools) for harness versions A and B
├── results/
│   ├── support_resistance.png        # Pivot high/low chart
│   ├── backtest_equity_curve.png     # Equity curves — original config (BUY=3.5)
│   ├── final_equity_curve.png        # Equity curves — final config (BUY=3.0)
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
│       └── mcp_server.py             # MCP server exposing 18 tools via FastMCP (stdio)
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

# Real A/B runs (A: base model without tools, B: bist-analyst with tools)
python llm/run_ab.py
```

The model never sees `submit_human_review`; the final verdict always belongs to a human (`--allow-human-review` exposes it only when you supply the verdict yourself). Price and macro data end on 2025-12-31 (fundamentals extend to 2026 Q3); every analysis states its data end date.

---

## MCP Agent Setup (Claude Desktop)

Add the following entry to your `claude_desktop_config.json`:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "bist-analytics": {
      "command": "<your-path>/venv/bin/python",
      "args": ["<your-path>/src/mcptools/mcp_server.py"],
      "cwd": "<your-path>"
    }
  }
}
```

Replace `<your-path>` with the absolute path to the cloned repository. Restart Claude Desktop after editing the file. Once connected, Claude can call 18 tools:

| Tool | Description |
|---|---|
| `get_stock_history` | Recent OHLCV price data |
| `calculate_indicators` | Latest SMA, RSI, MACD values and signal interpretation |
| `find_support_resistance_levels` | Nearest pivot-based support and resistance prices |
| `get_fundamental_metrics` | All financial metrics for a given quarter |
| `get_fundamental_score` | Fundamental quality score (0–5) with criterion breakdown |
| `run_backtest_summary` | Strategy performance metrics for a given parameter set |
| `start_analysis_session` | Open a new harness session for a ticker (START → CHECK_DATA_QUALITY) |
| `run_full_analysis` | Run the full evidence-bundle pipeline and risk gate for an open session |
| `get_recommended_action` | Return the 8-label analytical action for an open session |
| `submit_human_review` | Record ACCEPT / MODIFY / REJECT verdict and close the session |
| `get_session_replay` | Print the full Section-27 audit replay for a completed session |
| `rank_sector_relative_strength` | Rank sector-group laggards/leaders versus the benchmark |
| `test_weekday_pattern` | Weekday return test with 70/30 train/test split and Bonferroni correction |
| `detect_reversal_events` | Recent reversal events and forward-return statistics |
| `get_macro_context` | USD/TRY, policy rate, CPI and regime context |
| `get_news_events` | Curated news events and post-news returns |
| `check_data_quality` | Risk-gate and data-freshness check for a ticker |
| `build_evidence_bundle` | Observed vs. committed evidence for a session |

---

## Key Results

All results are based on ASELS.IS data from 2024-01-01 to 2025-12-31 using the final configuration: weights 0.50 technical + 0.50 fundamental, BUY threshold 3.0, SELL threshold 1.5.

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
| [`docs/mcp_demo/`](docs/mcp_demo/) | Five screenshots (original Claude Desktop test) showing the 11 first-phase MCP tools to analyse ASELS.IS end-to-end (read part1 → part5 in order) |
| [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | Field-level documentation for all data sources, CSV schemas, and module outputs |

---

## Disclaimer

This project is for **educational purposes only** and does not constitute investment advice. All analysis is performed on historical data; past performance is not indicative of future results. Use at your own risk.

---

## Author

**Altuğ** — CSE-481 Engineering Economics
