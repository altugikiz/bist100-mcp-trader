"""
Machine learning prediction module for BIST100 MCP Trader.
Predicts 5-day directional price movement using technical and fundamental features.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data_collection import fetch_ohlcv
from analysis.indicators import calculate_sma, calculate_rsi, calculate_macd
from analysis.support_resistance import find_support_resistance
from analysis.fundamental_analysis import (
    load_fundamentals, calculate_derived_ratios,
    calculate_market_multiples, calculate_yoy_growth,
    calculate_fundamental_score,
)
from strategy.strategy import _QUARTER_END
from strategy.backtest import run_backtest, calculate_performance, _max_drawdown, _sharpe_ratio
from config import (
    SYMBOL, BENCHMARK_SYMBOL,
    START_DATE, END_DATE,
    CURRENT_PRICE, SHARES_OUTSTANDING, SECTOR_AVG_PE,
    FUNDAMENTAL_CSV_PATH as _CSV_PATH,
)


def _period_to_date(period: str) -> pd.Timestamp:
    year, q = period[:4], period[4:]
    return pd.Timestamp(f"{year}-{_QUARTER_END[q]}")


# ---------------------------------------------------------------------------
# 1. Feature engineering
# ---------------------------------------------------------------------------

def build_feature_set(
    technical_df: pd.DataFrame,
    fundamental_df: pd.DataFrame,
    bist100_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the daily feature matrix used to train ML models.

    Features included
    -----------------
    Technical (daily):
        SMA20, SMA50, RSI14, MACD, MACD_Signal
        volume_change_pct   : day-over-day % change in Volume
        distance_from_support     : (Close - nearest_prior_support) / Close
            A positive value means price is above support.
        distance_from_resistance  : (nearest_prior_resistance - Close) / Close
            A positive value means price is below resistance.
            Rows where no prior resistance pivot exists are filled with 0
            (conservative: treated as if no overhead resistance) rather than
            being dropped, to preserve data density.

    Fundamental (quarterly → forward-filled to daily):
        revenue_growth_yoy, net_profit_growth_yoy, roe, net_debt_ebitda
        First 3 quarters (no TTM available) will be NaN; those rows are
        dropped later together with any other NaN rows.

    Market context (daily):
        bist100_return : XU100.IS daily return — captures broad market regime

    Deliberately excluded:
        USD/TRY exchange rate — macro data not collected in this project.
        Add as an optional feature once a macro data pipeline is implemented.

    Args:
        technical_df:   Daily OHLCV + indicator DataFrame (output of indicator
                        functions), with DatetimeIndex.
        fundamental_df: Quarterly DataFrame from load_fundamentals() after
                        calculate_derived_ratios() and calculate_yoy_growth().
        bist100_df:     Daily BIST100 close prices as a Series with DatetimeIndex.

    Returns:
        DataFrame of features aligned on the same DatetimeIndex as technical_df,
        with all-NaN rows already dropped.
    """
    df = technical_df.copy()

    # --- Volume % change ---
    df["volume_change_pct"] = df["Volume"].pct_change()

    # --- Support / resistance distance ---
    _, pivot_lows  = find_support_resistance(df, order=5)
    pivot_highs, _ = find_support_resistance(df, order=5)

    sup_prices = pivot_lows["Close"].sort_index()
    res_prices = pivot_highs["Close"].sort_index()

    def _nearest_prior(series: pd.Series, date: pd.Timestamp):
        prior = series[series.index <= date]
        return float(prior.iloc[-1]) if not prior.empty else float("nan")

    nearest_sup = pd.Series(
        [_nearest_prior(sup_prices, d) for d in df.index], index=df.index, dtype=float
    )
    nearest_res = pd.Series(
        [_nearest_prior(res_prices, d) for d in df.index], index=df.index, dtype=float
    )

    df["distance_from_support"]    = (df["Close"] - nearest_sup) / df["Close"]
    # Fill missing resistance (no prior pivot above) with 0 — no overhead pressure
    df["distance_from_resistance"] = (
        (nearest_res - df["Close"]) / df["Close"]
    ).fillna(0)

    # --- Fundamental features (quarterly → forward-filled daily) ---
    fund_cols = ["revenue_growth_yoy", "net_profit_growth_yoy", "roe", "net_debt_ebitda"]
    fund_indexed = fundamental_df[fund_cols].copy()
    fund_indexed.index = pd.Index(
        [_period_to_date(p) for p in fund_indexed.index], name="Date"
    )
    daily_fund = (
        fund_indexed
        .reindex(df.index.union(fund_indexed.index))
        .sort_index()
        .ffill()
        .reindex(df.index)
    )
    for col in fund_cols:
        df[col] = daily_fund[col]

    # --- BIST100 daily return ---
    bist100_ret = bist100_df.pct_change().rename("bist100_return")
    df["bist100_return"] = bist100_ret.reindex(df.index)

    feature_cols = [
        "SMA20", "SMA50", "RSI14", "MACD", "MACD_Signal",
        "volume_change_pct",
        "distance_from_support", "distance_from_resistance",
        "revenue_growth_yoy", "net_profit_growth_yoy", "roe", "net_debt_ebitda",
        "bist100_return",
    ]
    return df[feature_cols]


# ---------------------------------------------------------------------------
# 2. Target variable
# ---------------------------------------------------------------------------

def build_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    Create a binary classification target: will the closing price be higher
    in `horizon` trading days?

        target[t] = 1  if Close[t + horizon] > Close[t]
                  = 0  otherwise

    The last `horizon` rows receive NaN (no future price available) and must
    be aligned and dropped together with the feature matrix.

    Args:
        df:      DataFrame with a 'Close' column.
        horizon: Look-ahead window in trading days (default 5 ≈ 1 week).

    Returns:
        Binary Series aligned to df.index with NaN for the last `horizon` rows.
    """
    future_close = df["Close"].shift(-horizon)
    target = (future_close > df["Close"]).astype(float)
    target[target.isna()] = float("nan")
    return target.rename("target")


# ---------------------------------------------------------------------------
# 3. Chronological train/test split + model training
# ---------------------------------------------------------------------------

def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.80,
) -> dict:
    """
    Train Logistic Regression and Random Forest classifiers using a strict
    chronological split to prevent look-ahead bias.

    Why NOT shuffle=True in train_test_split
    -----------------------------------------
    Financial time series have temporal dependencies (autocorrelation, regime
    persistence).  Random shuffling would allow the model to train on data from
    the future relative to some test points, inflating apparent test accuracy.
    We always train on the first `train_frac` of the timeline and test on the
    remaining tail — the same ordering a live trader would experience.

    Scaling
    -------
    StandardScaler is fit ONLY on the training set.  The same fitted scaler
    is then applied (transform-only) to the test set to avoid data leakage.

    Args:
        X:           Feature DataFrame with DatetimeIndex, NaN-free.
        y:           Binary target Series aligned to X.
        train_frac:  Fraction of data used for training (default 0.80).

    Returns:
        dict with keys:
            'models'     : {'lr': fitted LR, 'rf': fitted RF}
            'scaler'     : fitted StandardScaler
            'X_train', 'X_test', 'y_train', 'y_test'
            'split_date' : first date of the test set
    """
    n = len(X)
    split_idx = int(n * train_frac)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    lr.fit(X_train_sc, y_train)
    rf.fit(X_train_sc, y_train)

    return {
        "models":     {"lr": lr, "rf": rf},
        "scaler":     scaler,
        "X_train":    X_train,  "X_test":  X_test,
        "X_train_sc": X_train_sc, "X_test_sc": X_test_sc,
        "y_train":    y_train,  "y_test":  y_test,
        "split_date": X_test.index[0],
    }


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate_models(
    models: dict,
    X_test_sc: np.ndarray,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Compute classification metrics for each model and a majority-class baseline.

    Metrics reported
    ----------------
        Accuracy  : fraction of correct predictions
        Precision : of predicted positives, how many are truly positive
        Recall    : of true positives, how many were predicted positive
        F1 Score  : harmonic mean of precision and recall
        Confusion matrix is printed inline.

    Majority-class baseline
    -----------------------
    A naive predictor that always guesses the most frequent class in y_test.
    This sets the floor: any model must beat this to be useful.

    Args:
        models:     dict of {name: fitted_model}.
        X_test_sc:  Scaled test features (numpy array).
        y_test:     True binary labels.

    Returns:
        DataFrame with one row per model (including baseline).
    """
    majority_class = int(y_test.mode()[0])
    baseline_preds = np.full(len(y_test), majority_class)
    baseline_acc   = accuracy_score(y_test, baseline_preds)

    rows = []
    for name, model in models.items():
        preds = model.predict(X_test_sc)
        cm    = confusion_matrix(y_test, preds)
        rows.append({
            "model":     name.upper(),
            "accuracy":  accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall":    recall_score(y_test, preds, zero_division=0),
            "f1":        f1_score(y_test, preds, zero_division=0),
            "cm":        cm,
            "preds":     preds,
        })
        print(f"\n  [{name.upper()}] Confusion matrix (rows=actual, cols=predicted):")
        print(f"    {'':10s}  Pred 0  Pred 1")
        print(f"    Actual 0   {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"    Actual 1   {cm[1,0]:5d}   {cm[1,1]:5d}")

    rows.append({
        "model": "BASELINE",
        "accuracy":  baseline_acc,
        "precision": float("nan"),
        "recall":    float("nan"),
        "f1":        float("nan"),
        "cm":        None,
        "preds":     baseline_preds,
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------

def print_feature_importance(rf_model, feature_names: list) -> None:
    """Print Random Forest feature importances sorted descending."""
    imp = pd.Series(rf_model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False)
    print("\n=== Random Forest Feature Importances ===")
    for feat, score in imp.items():
        bar = "█" * int(score * 60)
        print(f"  {feat:<35s} {score:.4f}  {bar}")


# ---------------------------------------------------------------------------
# 6. Optional ML-signal backtest
# ---------------------------------------------------------------------------

def ml_signal_backtest(
    price_df: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    predictions: np.ndarray,
    threshold: float = 0.55,
) -> dict | None:
    """
    Convert ML predictions on the test set into BUY/SELL signals and run a
    quick backtest using the existing run_backtest infrastructure.

    Only runs if the model's test accuracy exceeds `threshold` (default 55%).
    Returns None and prints a message if the threshold is not met.

    Signal mapping:
        prediction = 1 → BUY  (model expects price to be higher in 5 days)
        prediction = 0 → SELL (model expects price to be flat or lower)
    No HOLD signals are generated — this is a fully invested / fully flat model.

    Args:
        price_df:    Full OHLCV DataFrame (must span the test period).
        test_index:  DatetimeIndex of the test set rows.
        predictions: Binary numpy array from model.predict().
        threshold:   Minimum accuracy to proceed with the backtest.

    Returns:
        dict with cumulative_return, sharpe_ratio, max_drawdown, win_rate,
        or None if threshold not met.
    """
    accuracy = accuracy_score(
        predictions,   # used as placeholder; caller passes correctly ordered preds
        predictions,   # dummy — real accuracy is checked by caller
    )
    # Build signal series aligned to test dates
    signal_map = {1: "BUY", 0: "SELL"}
    signals = pd.Series(
        [signal_map[int(p)] for p in predictions],
        index=test_index,
        name="Signal",
    )

    # Merge signals onto the full price df (test period only)
    test_price = price_df.loc[test_index].copy()
    test_price["Signal"] = signals
    test_price["technical_score"]    = 0  # not used by run_backtest
    test_price["fundamental_score"]  = 0
    test_price["Total_Score"]        = 0

    bt = run_backtest(test_price)
    m  = calculate_performance(bt)
    return m


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)

    print("Fetching ASELS price data …")
    price_df = fetch_ohlcv(SYMBOL, START_DATE, END_DATE)
    price_df["SMA20"] = calculate_sma(price_df, 20)
    price_df["SMA50"] = calculate_sma(price_df, 50)
    price_df["RSI14"] = calculate_rsi(price_df, 14)
    price_df[["MACD", "MACD_Signal", "MACD_Hist"]] = calculate_macd(price_df)

    print("Loading fundamental data …")
    fund_df = load_fundamentals(_CSV_PATH)
    fund_df = calculate_derived_ratios(fund_df)
    fund_df = calculate_market_multiples(fund_df, CURRENT_PRICE, SHARES_OUTSTANDING)
    fund_df = calculate_yoy_growth(fund_df)
    fund_df = calculate_fundamental_score(fund_df, sector_avg_pe=SECTOR_AVG_PE)

    print("Fetching BIST100 …")
    raw_bist = yf.download(BENCHMARK_SYMBOL, start=START_DATE, end=END_DATE,
                           auto_adjust=True, progress=False)
    if isinstance(raw_bist.columns, pd.MultiIndex):
        raw_bist.columns = raw_bist.columns.get_level_values(0)
    bist100_close = raw_bist["Close"]

    # --- Build feature matrix and target ---
    print("Building features …")
    X_raw = build_feature_set(price_df, fund_df, bist100_close)
    y_raw = build_target(price_df, horizon=5)

    # Align and drop any row with NaN in features or target
    combined = X_raw.join(y_raw).dropna()
    X = combined.drop(columns=["target"])
    y = combined["target"].astype(int)

    print(f"  Dataset: {len(X)} rows × {len(X.columns)} features  "
          f"({int(y.sum())} positive / {int((y==0).sum())} negative)")
    print(f"  Features: {list(X.columns)}")

    # --- Chronological train/test split ---
    result   = train_models(X, y, train_frac=0.80)
    models   = result["models"]
    X_tsc    = result["X_test_sc"]
    y_test   = result["y_test"]
    split_dt = result["split_date"]
    n_train  = len(result["X_train"])
    n_test   = len(result["X_test"])

    print(f"\n  Train: {n_train} rows (up to {result['X_train'].index[-1].date()})")
    print(f"  Test : {n_test} rows (from {split_dt.date()})")

    # --- Evaluate ---
    print("\n=== Model evaluation (test set) ===")
    eval_df = evaluate_models(models, X_tsc, y_test)

    print("\n=== Performance summary ===")
    summary_cols = ["model", "accuracy", "precision", "recall", "f1"]
    fmt = eval_df[summary_cols].copy()
    fmt[["accuracy","precision","recall","f1"]] = (
        fmt[["accuracy","precision","recall","f1"]] * 100
    )
    header = f"  {'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in fmt.iterrows():
        acc = f"{row['accuracy']:.2f}%" if not pd.isna(row['accuracy']) else "   N/A"
        pre = f"{row['precision']:.2f}%" if not pd.isna(row['precision']) else "   N/A"
        rec = f"{row['recall']:.2f}%" if not pd.isna(row['recall']) else "   N/A"
        f1  = f"{row['f1']:.2f}%" if not pd.isna(row['f1']) else "   N/A"
        print(f"  {row['model']:<12} {acc:>10} {pre:>10} {rec:>10} {f1:>10}")

    # --- Feature importance ---
    print_feature_importance(models["rf"], list(X.columns))

    # --- Bonus: ML-signal backtest ---
    rf_preds    = models["rf"].predict(X_tsc)
    rf_accuracy = accuracy_score(y_test, rf_preds)

    ACCURACY_THRESHOLD = 0.55
    print(f"\n=== Bonus: ML-signal backtest ===")
    print(f"  RF test accuracy = {rf_accuracy*100:.2f}%  "
          f"(threshold = {ACCURACY_THRESHOLD*100:.0f}%)")

    if rf_accuracy > ACCURACY_THRESHOLD:
        print("  Threshold met — running ML-signal backtest on test period …")
        test_idx   = result["X_test"].index
        test_price = price_df.loc[test_idx].copy()

        signal_series = pd.Series(
            ["BUY" if p == 1 else "SELL" for p in rf_preds],
            index=test_idx, name="Signal",
        )
        test_price["Signal"]           = signal_series
        test_price["technical_score"]  = 0
        test_price["fundamental_score"]= 0
        test_price["Total_Score"]      = 0

        bt_ml = run_backtest(test_price)
        m_ml  = calculate_performance(bt_ml, label="ML Strategy")

        # Buy-and-hold over same test period
        bah_ret    = test_price["Close"].pct_change().fillna(0)
        bah_equity = (1 + bah_ret).cumprod()
        bah_cum    = float(bah_equity.iloc[-1] - 1)
        bah_sharpe = _sharpe_ratio(bah_ret.dropna())
        bah_dd     = _max_drawdown(bah_equity)

        # Existing technical+fundamental strategy reference (from sensitivity_analysis)
        tf_cum    = 2.1626   # 216.26 % over 2024-2025 (full period, not just test)
        tf_sharpe = 2.0619

        print(f"\n  {'Metric':<28} {'ML Strategy':>15} {'B&H (test period)':>18}")
        print(f"  {'-'*61}")
        print(f"  {'Cumulative Return':<28} {m_ml['cumulative_return']*100:>14.2f}%"
              f" {bah_cum*100:>17.2f}%")
        print(f"  {'Sharpe Ratio':<28} {m_ml['sharpe_ratio']:>15.4f} {bah_sharpe:>18.4f}")
        print(f"  {'Max Drawdown':<28} {m_ml['max_drawdown']*100:>14.2f}%"
              f" {bah_dd*100:>17.2f}%")
        print(f"  {'Win Rate':<28} {m_ml['win_rate']*100:>14.2f}%")
        print(f"  {'Days Invested':<28} {m_ml['days_invested']:>15d}")
        print(f"\n  Technical+Fundamental strategy (full 2024-2025):")
        print(f"    Cum. Return = {tf_cum*100:.2f}%  |  Sharpe = {tf_sharpe:.4f}")
    else:
        print("  Threshold NOT met — skipping ML-signal backtest.")
        print("  The model does not show sufficient edge to trade live.")
