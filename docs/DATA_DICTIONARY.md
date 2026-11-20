# Data Dictionary — Integrated BIST 100 Agentic Financial Analytics Harness

This document describes every structured and unstructured feature produced or consumed by the
system, per the project brief's Section 36 requirement. It is organized by pipeline stage.

---

## 1. Market Data (`data/universe_cache/<TICKER>.csv`)

| Field | Type | Description |
|---|---|---|
| `Date` | date | Trading date (index) |
| `Open`, `High`, `Low`, `Close` | float | Daily OHLC price (TRY) |
| `Volume` | int | Shares traded |

Source: Yahoo Finance (`yfinance`). Coverage: 2023-01-02 to 2026-09-10 (analysis date 2026-09-10), 30 tickers + XU100 benchmark.

## 2. Technical Indicators (`src/analysis/indicators.py`)

| Field | Description |
|---|---|
| `SMA20`, `SMA50` | Simple moving averages (20/50-day) |
| `RSI14` | 14-day Relative Strength Index (0–100) |
| `MACD`, `MACD_Signal`, `MACD_Hist` | Moving Average Convergence Divergence, its signal line, and histogram |
| `BB_Upper`, `BB_Middle`, `BB_Lower` | Bollinger Bands (20-day, 2σ) |
| `ATR` | 14-day Average True Range |
| `KAMA` | Kaufman's Adaptive Moving Average |
| `KAMA_Slope` | Categorical: `rising` / `falling` / `flat` |
| `Supertrend` | Supertrend line value |
| `Supertrend_Direction` | Categorical: `bullish` / `bearish` |
| `Tenkan_sen`, `Kijun_sen`, `Senkou_Span_A`, `Senkou_Span_B`, `Chikou_Span` | Ichimoku Cloud components |
| `Ichimoku_State` | Categorical: `above_cloud` / `below_cloud` / `in_cloud` |
| `technical_score` | 0–5 composite score (see `strategy.py`: SMA20>SMA50, Close>SMA20, RSI 30–70, MACD bullish, near support) |

## 3. Support/Resistance (`src/analysis/support_resistance.py`)

| Field | Description |
|---|---|
| `date`, `price` | Pivot high/low date and price (`scipy.signal.argrelextrema`, order=5) |
| `total_pivots.highs`, `.lows` | Count of detected resistance/support pivots |

## 4. Reversal Events (`data/events/<...>_events.csv`, `src/analysis/reversal_events.py`)

| Field | Description |
|---|---|
| `ticker` | Stock symbol |
| `event_type` | One of: `lower_band_touch`, `middle_band_test`, `upper_band_touch`, `rsi_trough`, `rsi_peak`, `supertrend_flip_bullish`, `supertrend_flip_bearish`, `bullish_kama_cross`, `bearish_kama_cross`, `cloud_breakout`, `cloud_breakdown`, `cloud_rejection`, `cloud_entry`, `kama_slope_turn_up`, `kama_slope_turn_down`, `support_touch`, `resistance_touch` |
| `level_price`, `level_age_days`, `prior_touches` | Only for `support_touch` / `resistance_touch`: price of the confirmed pivot level, calendar days since the pivot, number of earlier touches of the same level (a pivot is used only after its `order`-bar confirmation, so there is no look-ahead) |
| `event_date` | Date the event occurred |
| `close_price` | Close price on event date |
| `rsi_state`, `supertrend_state`, `kama_slope`, `ichimoku_state` | Market-state context at event time |
| `volume_regime` | `high` / `normal` (vs. 20-day average volume) |
| `forward_return_1d/3d/5d/10d` | Forward return after the event (strictly post-event, no look-ahead) |
| `trend_regime` | `rising` / `falling` (XU100, trailing 20-day) |
| `vol_regime` | `high_vol` / `low_vol` (relative to universe median) |

## 5. Sector Analysis (`src/analysis/sector_analysis.py`)

| Field | Description |
|---|---|
| `sector_group` | Consolidated sector (15 groups; see `universe.py`) |
| `benchmark_type` | `sector` or `xu100_fallback` |
| `lag_score_5/10/20` | Ticker return minus benchmark return, over N days |
| `potential_catchup_flag` | Boolean; True if laggard (20d) + technical turn (KAMA rising or Supertrend flip) |
| `reason` | Text explanation for the flag |

## 6. Calendar Patterns (`data/events/calendar_patterns.csv`, `sequence_patterns.csv`)

| Field | Description |
|---|---|
| `Weekday` | Monday–Friday |
| `N`, `Mean_Return`, `Median_Return`, `Positive_Rate` | Per-weekday statistics |
| `T_Stat`, `P_Value` | One-sample t-test vs. ticker's overall mean |
| `significant_after_correction` | Boolean, Bonferroni-corrected (α = 0.05/150) |
| `pattern` (sequence file) | One of `three_down_then_rebound`, `post_large_down_reversal`, `monday_weakness_friday_strength` |
| `occurrence_count`, `avg_continuation_return`, `hit_rate` | Per-pattern statistics |

## 7. Fundamental Data (`data/fundamentals/<TICKER>_fundamentals.csv`)

| Field | Description |
|---|---|
| `period` | Quarter label (e.g., `2025Q3`) |
| `revenue`, `ebitda`, `net_profit` | Million TRY, quarterly (derived via consecutive-difference for several tickers — see README methodology note) |
| `total_assets`, `shareholders_equity`, `net_debt` | Million TRY, balance-sheet snapshot at quarter-end |
| `financial_debt`, `cash` | Used to derive `net_debt` where applicable |
| *(ASELS only, 24-field schema)* | `gross_profit`, `operating_profit`, `operating_cash_flow`, `investing_cash_flow`, `financing_cash_flow`, `current_liabilities`, `inventory`, `current_assets` |
| `ebitda_margin`, `net_profit_margin`, `roe`, `roa`, `net_debt_ebitda`, `debt_equity`, `current_ratio`, `quick_ratio` | Derived ratios (`calculate_derived_ratios`) |
| `revenue_growth_yoy`, `net_profit_growth_yoy` | Year-over-year growth |
| `crit_revenue_growth`, `crit_net_profit_growth`, `crit_roe`, `crit_net_debt_ebitda`, `crit_pe` | Boolean pass/fail per fundamental-score criterion |
| `fundamental_score` | 0–5 composite score |

Note: `ebitda` and `net_debt` are intentionally `NaN` for AKBNK (bank; not economically meaningful line items).

## 8. Point-in-Time Alignment (`src/analysis/point_in_time.py`)

| Field | Description |
|---|---|
| `disclosure_date` | Estimated KAP disclosure date (proxy: quarter-end + 30 days for Financials, +45 days otherwise, business-day adjusted) |
| `quarter` | Fiscal quarter the disclosure covers |

## 9. Macro Context (`data/macro/macro_daily.csv`)

| Field | Description |
|---|---|
| `usdtry`, `eurtry` | Daily FX rate (Yahoo Finance, no key required) |
| `tcmb_rate` | TCMB policy rate (%), step function from 10 official PPK decisions |
| `fed_rate` | Fed funds rate upper bound (%), step function from 7 official FOMC decisions |
| `tufe_yoy` | TÜİK CPI year-over-year (%), interpolated between 8 confirmed anchor points; `NaN` before 2024-05-31 (first anchor) |

## 10. News Events (`src/analysis/news_events.py`)

| Field | Description |
|---|---|
| `ticker` | Stock symbol (ASELS, TOASO, or BIMAS) |
| `date` | Event date |
| `category` | One of: `earnings`, `export_contract`, `regulatory`, `corporate_action`, `financial_result` |
| `summary` | Original-wording summary of the publicly reported event (not a verbatim quote) |
| `source_type` | `company_disclosure` or `news_aggregator` |
| `forward_return_1d/5d/20d` | Post-event forward return |

## 11. Fundamental Reaction Dataset (`data/events/fundamental_reaction.csv`)

| Field | Description |
|---|---|
| `ticker`, `quarter` | Identifiers |
| `revenue_growth`, `net_profit_growth`, `roe_change`, `net_debt_ebitda_change` | QoQ/YoY fundamental change |
| `abnormal_return_1d/5d/20d` (vs. XU100) | Ticker return minus XU100 return, post-disclosure |
| `sector_abnormal_return_1d/5d/20d` | Ticker return minus sector-group median return |

## 12. Strategy & Backtest (`src/strategy/`)

| Field | Description |
|---|---|
| `Total_Score` | 0.50 × `technical_score` + 0.50 × `fundamental_score` |
| `Signal` | `BUY` (≥3.0) / `SELL` (≤1.5) / `HOLD` — internal backtest label only, not part of the harness action space |
| `Position` | Shifted signal (t+1) to avoid look-ahead bias |
| `cumulative_return_pct`, `sharpe_ratio`, `max_drawdown_pct`, `win_rate_pct` | Backtest performance metrics |
| `days_invested`, `trading_days` | Exposure statistics |

## 13. Agent Harness (`src/harness/`)

| Field | Description |
|---|---|
| `session_id` | UUID for an analysis session |
| `current_state`, `state_history` | Position in the 17-state machine |
| `tool_call_log` | Timestamped list of `{state, tool_name, input_summary, output_summary}` |
| `observed_context` | All raw computed features for a session (unfiltered) |
| `committed_evidence` | Selected list of `{id, statement, source_tool, value}` — the subset used in the final explanation |
| `gate_result.status` | `PASS` / `PASS_WITH_WARNINGS` / `INVESTIGATE` / `ANALYSIS_UNSAFE` / `HARD_BLOCK` |
| `gate_result.warnings`, `.blocks` | Lists of triggered soft-warning / hard-block reasons |
| `action` | One of the 8 finite educational actions (see README) |
| `reasoning` | List of evidence-id references supporting the action |
| `human_review.result` | `ACCEPT` / `MODIFY` / `REJECT` |
| `human_review.reason`, `.reviewer_note` | Free-text justification |

---

*This dictionary covers the fields produced by the pipeline as of the final submission. For exact
column order and dtypes, refer to each module's docstring in `src/`.*
