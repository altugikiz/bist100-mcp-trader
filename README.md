# BIST 100 MCP Trader

A CSE-481 Engineering Economics term project that performs end-to-end quantitative analysis of an ASELS.IS (Aselsan) stock listed on Borsa Istanbul. The project combines technical indicator scoring, Fintables-sourced fundamental data, a rule-based backtesting engine, and a sensitivity analysis suite into a unified pipeline. It also exposes the entire analysis stack as an **MCP (Model Context Protocol) AI agent**, allowing a large language model such as Claude to query live price data, fundamental metrics, and backtest results through structured tool calls.

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
- **MCP AI agent** — six tool endpoints exposed via FastMCP; connects to Claude Desktop over stdio transport

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
│       └── mcp_server.py             # MCP server exposing 11 tools via FastMCP (stdio)
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

Replace `<your-path>` with the absolute path to the cloned repository. Restart Claude Desktop after editing the file. Once connected, Claude can call six tools:

| Tool | Description |
|---|---|
| `get_stock_history` | Recent OHLCV price data |
| `calculate_indicators` | Latest SMA, RSI, MACD values and signal interpretation |
| `find_support_resistance_levels` | Nearest pivot-based support and resistance prices |
| `get_fundamental_metrics` | All financial metrics for a given quarter |
| `get_fundamental_score` | Fundamental quality score (0–5) with criterion breakdown |
| `run_backtest_summary` | Strategy performance metrics for a given parameter set |

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
| [`docs/report/`](docs/report/) | Final term project report (`ASELS_Term_Project_Final_Report.docx`) |
| [`docs/presentation/`](docs/presentation/) | Project presentation slides (`ASELS_Term_Project_Presentation_EN.pptx`) |
| [`docs/mcp_demo/`](docs/mcp_demo/) | Five screenshots showing Claude Desktop calling all six MCP tools to analyse ASELS.IS end-to-end (read part1 → part5 in order) |

---

## Disclaimer

This project is for **educational purposes only** and does not constitute investment advice. All analysis is performed on historical data; past performance is not indicative of future results. Use at your own risk.

---

## Author

**Altuğ** — CSE-481 Engineering Economics
