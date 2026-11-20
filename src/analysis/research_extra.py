"""
Research helpers for the stateless MCP tools (analysis date aware).

Every function takes an `as_of` date and uses only data on or before it.
Results are plain dicts of numbers + explicit warnings (small samples,
incomplete forward windows, overlapping signals, proxy disclosure dates).
"""
import os
import sys

import re
import numpy as np
import pandas as pd
from scipy import stats

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SRC)

from config import ANALYSIS_DATE                                   # noqa: E402
from data.universe import UNIVERSE, get_relative_benchmark          # noqa: E402

_ROOT = os.path.abspath(os.path.join(_SRC, ".."))
_CACHE = os.path.join(_ROOT, "data", "universe_cache")
_FUND = os.path.join(_ROOT, "data", "fundamentals")
_MACRO = os.path.join(_ROOT, "data", "macro", "macro_daily.csv")

from analysis.fundamental_analysis import YTD_TICKERS               # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def resolve_as_of(as_of: str = "") -> tuple:
    """Return (timestamp, warning|None). The date is capped at ANALYSIS_DATE and at the cache end."""
    cap = pd.Timestamp(ANALYSIS_DATE)
    warn = None
    ts = pd.Timestamp(as_of) if as_of else cap
    if ts > cap:
        warn = f"The requested date is after the analysis date; it was capped at {cap.date()}."
        ts = cap
    return ts, warn


def load_ohlcv(ticker: str, as_of: str = "") -> pd.DataFrame:
    path = os.path.join(_CACHE, f"{ticker.upper()}.csv")
    if not os.path.isfile(path):
        raise ValueError(f"No cached price data for {ticker}; it is not in the local study universe.")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    ts, _ = resolve_as_of(as_of)
    return df.loc[df.index <= ts]


def _close(ticker: str, as_of: str = "") -> pd.Series:
    return load_ohlcv(ticker, as_of)["Close"]


def _fwd(close: pd.Series, date, h: int) -> float:
    """Forward h-trading-day return from the last close on/before `date`; NaN if the window is incomplete."""
    pos = close.index.get_indexer([pd.Timestamp(date)], method="ffill")[0]
    if pos < 0 or pos + h >= len(close):
        return np.nan
    return float(close.iloc[pos + h] / close.iloc[pos] - 1)


def _benchmark_close(ticker: str, as_of: str = "") -> tuple:
    """Sector-peer median Close (or XU100 when the ticker has no peers) and its type."""
    info = get_relative_benchmark(ticker)
    if info["type"] == "sector":
        cl = pd.concat([_close(p, as_of) for p in info["peers"]], axis=1, sort=True).ffill()
        return cl.median(axis=1), "sector_peer_median"
    return _close("XU100", as_of), "xu100_fallback"


_SECTOR_ALIASES = {
    "finans": "Financials", "finansal": "Financials", "bankacılık": "Financials", "banka": "Financials",
    "perakende": "Retail", "market": "Retail", "metal": "Metals & Mining", "metaller": "Metals & Mining",
    "madencilik": "Metals & Mining", "demir çelik": "Metals & Mining", "otomotiv": "Automotive",
    "kimya": "Chemicals", "enerji": "Electric Utilities", "elektrik": "Electric Utilities",
    "telekom": "Telecommunications", "telekomünikasyon": "Telecommunications",
    "havacılık": "Transportation-Airlines", "ulaştırma": "Transportation-Airlines", "ulaşım": "Transportation-Airlines",
    "sağlık": "Healthcare", "çimento": "Cement", "gıda": "Food & Beverage", "yiyecek içecek": "Food & Beverage",
    "gayrimenkul": "Real Estate & Construction", "inşaat": "Real Estate & Construction",
    "savunma": "Technology-Defense", "teknoloji": "Technology-Defense", "petrol": "Petroleum-Refining",
    "rafineri": "Petroleum-Refining", "dayanıklı": "Durables-Machinery", "makine": "Durables-Machinery",
    "metals": "Metals & Mining", "mining": "Metals & Mining", "banking": "Financials", "banks": "Financials",
    "airlines": "Transportation-Airlines", "telecom": "Telecommunications", "energy": "Electric Utilities",
    "defense": "Technology-Defense", "defence": "Technology-Defense", "real estate": "Real Estate & Construction",
}


def normalize_sector_group(name: str) -> str:
    """Map a Turkish/short sector name (e.g. 'Finans', 'perakende', 'banking') to the universe sector_group."""
    key = name.strip().lower().replace("sektörü", "").replace("sektör", "").strip()
    return _SECTOR_ALIASES.get(key, name.strip())


def _group_members(sector_group: str) -> list:
    sector_group = normalize_sector_group(sector_group)
    m = UNIVERSE[UNIVERSE["sector_group"].str.lower() == sector_group.lower()]
    return m["ticker"].tolist()


# ---------------------------------------------------------------------------
# 1. Sector laggard / catch-up backtest
# ---------------------------------------------------------------------------

def sector_catchup_backtest(sector_group: str, as_of: str = "", start: str = "2024-01-01",
                            lag_threshold: float = -0.05, lag_window: int = 20,
                            horizons=(5, 10, 20), cost_bps: float = 0.0) -> dict:
    """
    Weekly snapshots: a stock is a "laggard" when its `lag_window`-day return is below its
    sector benchmark by more than |lag_threshold|. Measure the next h trading days:
    catch-up rate = P(stock return > benchmark return), hit rate = P(stock return > 0),
    compared with the same statistics for non-laggard peers on the same snapshots.
    """
    sector_group = normalize_sector_group(sector_group)
    if abs(lag_threshold) >= 1:          # a model may pass percent points (-5 or -2.5) instead of a fraction
        lag_threshold = lag_threshold / 100.0
    lag_threshold = -abs(lag_threshold) if lag_threshold else -0.05
    members = _group_members(sector_group)
    if not members:
        return {"error": f"Unknown sector_group '{sector_group}'.",
                "valid_sector_groups": sorted(UNIVERSE["sector_group"].unique().tolist())}
    ts, warn = resolve_as_of(as_of)
    closes = {t: _close(t, as_of) for t in members}
    bench = {t: _benchmark_close(t, as_of) for t in members}
    xu = _close("XU100", as_of)
    snaps = pd.bdate_range(start=start, end=ts, freq="W-FRI")
    rows = []
    for snap in snaps:
        for t in members:
            c, (b, _) = closes[t], bench[t]
            pos = c.index.get_indexer([snap], method="ffill")[0]
            posb = b.index.get_indexer([snap], method="ffill")[0]
            if pos < lag_window or posb < lag_window:
                continue
            lag = (c.iloc[pos] / c.iloc[pos - lag_window] - 1) - (b.iloc[posb] / b.iloc[posb - lag_window] - 1)
            rec = {"ticker": t, "date": snap, "lag": float(lag), "laggard": bool(lag < lag_threshold)}
            for h in horizons:
                r, rb, rx = _fwd(c, snap, h), _fwd(b, snap, h), _fwd(xu, snap, h)
                rec[f"ret_{h}"], rec[f"rel_{h}"], rec[f"xu_{h}"] = r, r - rb, r - rx
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return {"error": "No snapshots with enough history."}
    cost = cost_bps / 1e4
    out = {}
    for h in horizons:
        sub = df.dropna(subset=[f"ret_{h}"])
        sig, base = sub[sub["laggard"]], sub[~sub["laggard"]]
        n = len(sig)
        d = {"n_signals": int(n), "n_baseline": int(len(base))}
        if n:
            k = int((sig[f"rel_{h}"] > 0).sum())
            base_rate = float((base[f"rel_{h}"] > 0).mean()) if len(base) else None
            d.update({
                "catchup_rate": k / n,
                "n_false_positives": int(n - k),
                "hit_rate_abs": float((sig[f"ret_{h}"] > 0).mean()),
                "mean_return": float(sig[f"ret_{h}"].mean()),
                "median_return": float(sig[f"ret_{h}"].median()),
                "mean_return_net_of_cost": float(sig[f"ret_{h}"].mean() - cost),
                "mean_excess_vs_sector": float(sig[f"rel_{h}"].mean()),
                "mean_excess_vs_xu100": float(sig[f"xu_{h}"].mean()),
                "baseline_catchup_rate_non_laggards": base_rate,
                "baseline_mean_return": float(base[f"ret_{h}"].mean()) if len(base) else None,
                "binomial_p_vs_baseline": float(stats.binomtest(k, n, base_rate).pvalue)
                                          if base_rate and 0 < base_rate < 1 else None,
                "approx_independent_signals": int(max(1, n * 5 // max(h, 5))),
                "low_sample_warning": bool(n < 30),
            })
        out[f"{h}d"] = d
    summary, edges = [], []
    for h, d in out.items():
        if not d.get("n_signals"):
            summary.append(f"{h}: no laggard signals at this threshold, so nothing can be concluded.")
            continue
        base = d.get("baseline_catchup_rate_non_laggards")
        p = d.get("binomial_p_vs_baseline")
        edge = bool(base is not None and d["catchup_rate"] > base and p is not None and p < 0.05 and d["n_signals"] >= 30)
        edges.append(edge)
        summary.append(
            f"{h}: catch-up rate {d['catchup_rate']*100:.1f}% vs {base*100:.1f}% for non-laggards "
            f"(n={d['n_signals']}, p={p:.3f}; false positives, i.e. laggards that kept underperforming the sector: "
            f"{d['n_signals'] - int(round(d['catchup_rate']*d['n_signals']))} of {d['n_signals']}) -> " + ("evidence of a catch-up edge." if edge else
                                                       "NO evidence that laggards catch up more often than other stocks.")
            if base is not None and p is not None else f"{h}: baseline not available.")
    verdict = ("POTENTIAL_CATCH_UP_CANDIDATE is supported by the history (catch-up rate beats non-laggards with p<0.05 at a horizon)."
               if any(edges) else
               "Label verdict: INVESTIGATE. The historical catch-up rates do not beat the non-laggard baseline, so "
               "POTENTIAL_CATCH_UP_CANDIDATE is NOT supported by this evidence; a current lag alone is not enough.")
    summary.append(verdict)
    incomplete = int(df["laggard"].sum() - len(df[df["laggard"]].dropna(subset=[f"ret_{horizons[-1]}"])))
    return {
        "sector_group": sector_group, "members": members, "as_of": str(ts.date()),
        "period": f"{pd.Timestamp(start).date()} to {ts.date()}", "lag_window_days": lag_window,
        "lag_threshold": lag_threshold, "n_snapshots": int(len(snaps)), "evidence_summary": summary, "results": out,
        "laggard_signals_excluded_incomplete_window": incomplete,
        "warnings": ([warn] if warn else []) + [
            "Weekly snapshots with 5-20 day horizons overlap, so signals are not independent "
            "(see approx_independent_signals).",
            "Signals whose forward window ends after the analysis date are excluded, not estimated.",
            "Single-member sector groups are benchmarked against XU100.",
            "Historical statistics only; not a trading recommendation.",
        ] + ([] if cost_bps else ["No transaction costs applied (cost_bps=0)."]),
    }


# ---------------------------------------------------------------------------
# 2. Weekday sequence (e.g. weak Monday -> strong Friday) with date-based OOS split
# ---------------------------------------------------------------------------

DEFAULT_START = "2024-01-01"   # start of the instructor-defined data period (01.01.2024 - 10.09.2026)

_DAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
         "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4,
         "pazartesi": 0, "salı": 1, "çarşamba": 2, "perşembe": 3, "cuma": 4}
_NEG = {"negative", "down", "fall", "falling", "decline", "drop", "loss", "red", "bearish", "lower", "düşüş", "negatif", "düşük"}
_POS = {"positive", "up", "rise", "rising", "gain", "green", "bullish", "higher", "yükseliş", "pozitif", "artış"}


def _direction(x: str):
    x = str(x).strip().lower()
    return "negative" if x in _NEG else "positive" if x in _POS else None


def weekday_sequence_test(ticker: str, first_day: str = "Monday", first_direction: str = "negative",
                          second_day: str = "Friday", second_direction: str = "positive",
                          as_of: str = "", start: str = "", train_end: str = "",
                          cost_bps: float = 0.0, n_hypotheses: int = 1) -> dict:
    """
    Within each calendar week: P(second-day move | first-day move) versus the unconditional
    P(second-day move), with a binomial test, an in-sample / out-of-sample split at `train_end`
    (default: 70% of the weeks) and a cost-adjusted hold return from first-day close to
    second-day close.
    """
    fd, sd = _DAYS.get(first_day.strip().lower()), _DAYS.get(second_day.strip().lower())
    if fd is None or sd is None or fd >= sd:
        return {"error": "first_day and second_day must be weekday names (Monday..Friday) with first_day earlier in the week."}
    fdir, sdir = _direction(first_direction), _direction(second_direction)
    if fdir is None or sdir is None:
        return {"error": "first_direction and second_direction must be 'positive' or 'negative'."}
    first_direction, second_direction = fdir, sdir
    first_day, second_day = [k for k, v in _DAYS.items() if v == fd and len(k) > 3 and k.isascii()][0].capitalize(), \
                            [k for k, v in _DAYS.items() if v == sd and len(k) > 3 and k.isascii()][0].capitalize()
    df = load_ohlcv(ticker, as_of)
    df = df.loc[df.index >= pd.Timestamp(start or DEFAULT_START)]
    df = df.copy()
    df["ret"] = df["Close"].pct_change()
    df["wk"] = df.index.to_period("W-SUN")
    rows = []
    for wk, g in df.groupby("wk"):
        a, b = g[g.index.dayofweek == fd], g[g.index.dayofweek == sd]
        if len(a) == 1 and len(b) == 1 and not np.isnan(a["ret"].iloc[0]) and not np.isnan(b["ret"].iloc[0]):
            rows.append({"week": a.index[0], "r1": a["ret"].iloc[0], "r2": b["ret"].iloc[0],
                         "hold": b["Close"].iloc[0] / a["Close"].iloc[0] - 1})
    w = pd.DataFrame(rows)
    if len(w) < 20:
        return {"error": f"Only {len(w)} complete weeks available for {ticker}."}
    cond = (w["r1"] < 0) if first_direction == "negative" else (w["r1"] > 0)
    outc = (w["r2"] > 0) if second_direction == "positive" else (w["r2"] < 0)
    cut = pd.Timestamp(train_end) if train_end else w["week"].iloc[int(len(w) * 0.7)]
    cost = cost_bps / 1e4

    def seg(mask):
        c, o, h = cond[mask], outc[mask], w.loc[mask, "hold"]
        n, nc = int(mask.sum()), int(c.sum())
        res = {"weeks": n, "weeks_with_condition": nc}
        if n and nc:
            k = int(o[c].sum())
            p_unc = float(o.mean())
            res.update({"p_outcome_given_condition": k / nc, "p_outcome_unconditional": p_unc,
                        "lift": (k / nc) / p_unc if p_unc else None,
                        "binomial_p": float(stats.binomtest(k, nc, p_unc).pvalue) if 0 < p_unc < 1 else None,
                        "mean_hold_return": float(h[c].mean()),
                        "mean_hold_return_net_of_cost": float(h[c].mean() - cost),
                        "low_sample_warning": bool(nc < 15)})
        return res
    allm = pd.Series(True, index=w.index)
    ins, oos = seg(w["week"] <= cut), seg(w["week"] > cut)
    full = seg(allm)
    stable = bool(ins.get("p_outcome_given_condition") and oos.get("p_outcome_given_condition")
                  and ins["p_outcome_given_condition"] > ins["p_outcome_unconditional"]
                  and oos["p_outcome_given_condition"] > oos["p_outcome_unconditional"]
                  and not oos.get("low_sample_warning", True))
    alpha = 0.05 / max(1, n_hypotheses)
    return {
        "ticker": ticker.upper(), "as_of": str(resolve_as_of(as_of)[0].date()),
        "pattern": f"{first_day} {first_direction} -> {second_day} {second_direction}",
        "train_end": str(cut.date()), "full_sample": full, "in_sample": ins, "out_of_sample": oos,
        "evidence_summary": (
            f"{first_day} {first_direction} -> {second_day} {second_direction}: "
            + (("same direction in and out of sample and significant after multiple-testing correction."
                if full.get("binomial_p") is not None and full["binomial_p"] < alpha else
                "same direction out of sample, but NOT statistically significant after correction: weak evidence only.")
               if stable else "NOT stable out of sample (no reliable pattern).")
            + (f" Full-sample P(outcome|condition)={full['p_outcome_given_condition']*100:.1f}% vs "
               f"{full['p_outcome_unconditional']*100:.1f}% unconditional (p={full['binomial_p']:.3f}, "
               f"n={full['weeks_with_condition']} weeks)." if full.get("binomial_p") is not None else "")),
        "bonferroni_alpha": alpha,
        "significant_full_sample_after_correction": bool(full.get("binomial_p") is not None and full["binomial_p"] < alpha),
        "stable_out_of_sample": stable,
        "warnings": ["Weekday patterns are often spurious; require significance after correction AND the same "
                     "direction out of sample.",
                     f"Multiple-testing alpha uses n_hypotheses={n_hypotheses}.",
                     "Hold return = first-day close to second-day close." +
                     ("" if cost_bps else " No transaction costs applied (cost_bps=0).")],
    }


def _parse_legs(text: str):
    """
    Parse weekday moves such as 'Mon down, Tue down', 'Monday-, Tuesday-', 'Mon–Tue–Wed down' or
    'Pazartesi–Salı–Çarşamba düşüş' into [(weekday_index, 'negative'|'positive')].
    A direction word applies to all weekdays listed since the previous direction word.
    """
    t = str(text or "").lower().replace("–", " ").replace("—", " ")
    days = "|".join(sorted((re.escape(k) for k in _DAYS), key=len, reverse=True))
    dirs = "|".join(sorted((re.escape(k) for k in (_NEG | _POS)), key=len, reverse=True))
    legs, pending = [], []
    for m in re.finditer(rf"(?P<day>{days})(?P<sign>[+-])?|(?P<dir>{dirs})|(?P<sg>[+-])", t):
        if m.group("day"):
            d = _DAYS[m.group("day")]
            if m.group("sign"):
                legs.append((d, "negative" if m.group("sign") == "-" else "positive"))
            else:
                pending.append(d)
        else:
            word = m.group("dir") or m.group("sg")
            direction = "negative" if (word in _NEG or word == "-") else "positive"
            legs += [(d, direction) for d in pending]
            pending = []
    if pending:
        return None
    legs = sorted(set(legs))
    return legs or None


def weekday_multiday_pattern_test(ticker: str, condition: str = "Mon down, Tue down, Wed down",
                                  outcome: str = "Thu up, Fri up", as_of: str = "", start: str = "",
                                  train_end: str = "", cost_bps: float = 0.0, n_hypotheses: int = 1) -> dict:
    """
    Multi-day weekly pattern: in each calendar week, how often do all `outcome` days move as stated
    given that all `condition` days moved as stated, versus the unconditional frequency of the outcome.
    In-sample / out-of-sample split at `train_end` (default 70% of weeks), cost-adjusted hold return
    from the last condition day's close to the last outcome day's close.
    """
    cl, ol = _parse_legs(condition), _parse_legs(outcome)
    if not cl or not ol:
        return {"error": "condition and outcome must list weekday + direction, e.g. condition='Mon down, Tue down, Wed down', "
                         "outcome='Thu up, Fri up'."}
    if max(d for d, _ in cl) >= min(d for d, _ in ol):
        return {"error": "every outcome day must be later in the week than every condition day."}
    names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    fmt = lambda legs: ", ".join(f"{names[d]} {'down' if x == 'negative' else 'up'}" for d, x in legs)
    df = load_ohlcv(ticker, as_of)
    df = df.loc[df.index >= pd.Timestamp(start or DEFAULT_START)].copy()
    df["ret"] = df["Close"].pct_change()
    df["wk"] = df.index.to_period("W-SUN")
    need = sorted({d for d, _ in cl + ol})
    last_c, last_o = max(d for d, _ in cl), max(d for d, _ in ol)
    rows = []
    for wk, g in df.groupby("wk"):
        sel = {d: g[g.index.dayofweek == d] for d in need}
        if any(len(v) != 1 or np.isnan(v["ret"].iloc[0]) for v in sel.values()):
            continue
        ok_c = all((sel[d]["ret"].iloc[0] < 0) if x == "negative" else (sel[d]["ret"].iloc[0] > 0) for d, x in cl)
        ok_o = all((sel[d]["ret"].iloc[0] < 0) if x == "negative" else (sel[d]["ret"].iloc[0] > 0) for d, x in ol)
        rows.append({"week": sel[need[0]].index[0], "cond": ok_c, "outc": ok_o,
                     "hold": sel[last_o]["Close"].iloc[0] / sel[last_c]["Close"].iloc[0] - 1})
    w = pd.DataFrame(rows)
    if len(w) < 20:
        return {"error": f"Only {len(w)} complete weeks available for {ticker}."}
    cut = pd.Timestamp(train_end) if train_end else w["week"].iloc[int(len(w) * 0.7)]
    cost = cost_bps / 1e4

    def seg(mask):
        sub = w[mask]
        n, nc = len(sub), int(sub["cond"].sum())
        res = {"weeks": int(n), "weeks_with_condition": nc, "weeks_with_full_pattern": int((sub["cond"] & sub["outc"]).sum())}
        if n and nc:
            k = int(sub.loc[sub["cond"], "outc"].sum())
            p_unc = float(sub["outc"].mean())
            h = sub.loc[sub["cond"], "hold"]
            res.update({"p_outcome_given_condition": k / nc, "p_outcome_unconditional": p_unc,
                        "lift": (k / nc) / p_unc if p_unc else None,
                        "binomial_p": float(stats.binomtest(k, nc, p_unc).pvalue) if 0 < p_unc < 1 else None,
                        "mean_hold_return": float(h.mean()), "median_hold_return": float(h.median()),
                        "mean_hold_return_net_of_cost": float(h.mean() - cost),
                        "low_sample_warning": bool(nc < 15 or k < 5)})
        return res
    full, ins, oos = seg(pd.Series(True, index=w.index)), seg(w["week"] <= cut), seg(w["week"] > cut)
    stable = bool(ins.get("p_outcome_given_condition") is not None and oos.get("p_outcome_given_condition") is not None
                  and ins["p_outcome_given_condition"] > ins["p_outcome_unconditional"]
                  and oos["p_outcome_given_condition"] > oos["p_outcome_unconditional"]
                  and not oos.get("low_sample_warning", True))
    alpha = 0.05 / max(1, n_hypotheses)
    def pct(x): return "n/a" if x is None else f"{x*100:.1f}%"
    summ = (f"Pattern [{fmt(cl)}] then [{fmt(ol)}]: full pattern occurred in {full['weeks_with_full_pattern']} of {full['weeks']} weeks "
            f"(condition met in {full['weeks_with_condition']}). "
            f"In-sample (to {cut.date()}): P(outcome|condition)={pct(ins.get('p_outcome_given_condition'))} vs {pct(ins.get('p_outcome_unconditional'))} unconditional, "
            f"n={ins['weeks_with_condition']}. Out-of-sample: P={pct(oos.get('p_outcome_given_condition'))} vs {pct(oos.get('p_outcome_unconditional'))}, "
            f"n={oos['weeks_with_condition']}. "
            + ("Same direction out of sample, but small samples: weak evidence only." if stable else
               "NOT stable out of sample (no reliable pattern), or too few occurrences to say."))
    def hold_txt(seg_res, label):
        if seg_res.get("mean_hold_return") is None:
            return f"{label}: no condition weeks."
        return (f"{label}: trading the pattern (buy at the last condition day's close, sell at the last outcome day's close) gave a mean "
                f"{seg_res['mean_hold_return']*100:.2f}% and median {seg_res['median_hold_return']*100:.2f}% per condition week"
                f" ({seg_res['mean_hold_return_net_of_cost']*100:.2f}% mean after {cost_bps:.0f} bps cost).")
    summ += (" " + hold_txt(ins, "In-sample") + " " + hold_txt(oos, "Out-of-sample")
             + f" Multiple-testing risk: this is {n_hypotheses} pre-specified pattern (Bonferroni alpha {alpha:.4f}); scanning many weekday sequences "
               "across many stocks would produce chance patterns, so a pattern found by search needs a much stricter threshold and out-of-sample confirmation.")
    return {"ticker": ticker.upper(), "as_of": str(resolve_as_of(as_of)[0].date()), "condition": fmt(cl), "outcome": fmt(ol),
            "train_end": str(cut.date()), "full_sample": full, "in_sample": ins, "out_of_sample": oos,
            "evidence_summary": summ, "bonferroni_alpha": alpha,
            "significant_full_sample_after_correction": bool(full.get("binomial_p") is not None and full["binomial_p"] < alpha),
            "stable_out_of_sample": stable,
            "warnings": ["Long weekday patterns occur rarely, so samples are small; weekday patterns are often spurious.",
                         f"Multiple-testing alpha uses n_hypotheses={n_hypotheses}.",
                         "Hold return = last condition day's close to last outcome day's close." +
                         ("" if cost_bps else " No transaction costs applied (cost_bps=0).")]}


# ---------------------------------------------------------------------------
# 3. Quarterly fundamental change + post-disclosure reaction
# ---------------------------------------------------------------------------

def _quarterly_frame(ticker: str) -> pd.DataFrame:
    """Single-quarter fundamentals indexed by period (the shared loader converts YTD files)."""
    from analysis.fundamental_analysis import load_fundamentals
    try:
        return load_fundamentals(ticker)
    except FileNotFoundError:
        raise ValueError(f"No fundamentals file for {ticker}.")


def _pct(cur, prev):
    if prev is None or pd.isna(prev) or pd.isna(cur) or prev == 0:
        return None
    return float((cur - prev) / abs(prev))


def fundamental_change(ticker: str, as_of: str = "", horizons=(1, 5, 20)) -> dict:
    """Latest quarter disclosed by `as_of`: YoY/QoQ changes and 1/5/20-day abnormal returns after disclosure."""
    from analysis.point_in_time import build_point_in_time_fundamentals
    ticker = ticker.upper()
    ts, warn = resolve_as_of(as_of)
    q = _quarterly_frame(ticker)
    pit = build_point_in_time_fundamentals(ticker, q.reset_index()).set_index("period")
    avail = pit[pit["disclosure_date"] <= ts]
    if avail.empty:
        return {"error": f"No quarter of {ticker} was disclosed by {ts.date()}."}
    excluded = [{"period": p, "estimated_disclosure": str(r["disclosure_date"].date())}
                for p, r in pit[pit["disclosure_date"] > ts].iterrows()]
    per = list(avail.index)
    cur = avail.iloc[-1]
    cp = per[-1]
    prevq = avail.iloc[-2] if len(avail) > 1 else None
    yoy_p = f"{int(cp[:4]) - 1}{cp[4:]}"
    yoy = avail.loc[yoy_p] if yoy_p in avail.index else None

    def ebitda_margin(r):
        return None if r is None or pd.isna(r.get("ebitda")) or not r.get("revenue") else float(r["ebitda"] / r["revenue"])

    ttm = avail["ebitda"].rolling(4).sum() if "ebitda" in avail.columns else None
    nd_ebitda = (float(cur["net_debt"] / ttm.iloc[-1]) if ttm is not None and not pd.isna(ttm.iloc[-1])
                 and ttm.iloc[-1] and not pd.isna(cur.get("net_debt")) else None)
    changes = {
        "revenue_yoy": _pct(cur["revenue"], yoy["revenue"]) if yoy is not None else None,
        "revenue_qoq": _pct(cur["revenue"], prevq["revenue"]) if prevq is not None else None,
        "net_profit_yoy": _pct(cur["net_profit"], yoy["net_profit"]) if yoy is not None else None,
        "net_profit_qoq": _pct(cur["net_profit"], prevq["net_profit"]) if prevq is not None else None,
        "ebitda_margin": ebitda_margin(cur),
        "ebitda_margin_yoy_change_pp": (None if yoy is None or ebitda_margin(cur) is None or ebitda_margin(yoy) is None
                                        else float((ebitda_margin(cur) - ebitda_margin(yoy)) * 100)),
        "net_debt_to_ttm_ebitda": nd_ebitda,
        "net_profit_sign_change": (None if prevq is None else
                                   ("turned_positive" if prevq["net_profit"] < 0 <= cur["net_profit"] else
                                    "turned_negative" if prevq["net_profit"] >= 0 > cur["net_profit"] else "unchanged_sign")),
    }

    close, xu = _close(ticker, as_of), _close("XU100", as_of)
    bench, btype = _benchmark_close(ticker, as_of)

    def reaction(disc):
        rec = {"disclosure_date_estimate": str(pd.Timestamp(disc).date())}
        for h in horizons:
            r, rx, rb = _fwd(close, disc, h), _fwd(xu, disc, h), _fwd(bench, disc, h)
            ok = not np.isnan(r)
            rec[f"raw_{h}d"] = r if ok else None
            rec[f"abnormal_vs_xu100_{h}d"] = (r - rx) if ok else None
            rec[f"abnormal_vs_sector_{h}d"] = (r - rb) if ok and not np.isnan(rb) else None
            rec[f"window_complete_{h}d"] = bool(ok)
        return rec

    latest_reaction = reaction(cur["disclosure_date"])
    history = []
    for p, r in avail.iterrows():
        rr = reaction(r["disclosure_date"])
        rr["period"] = p
        yp = f"{int(p[:4]) - 1}{p[4:]}"
        rr["net_profit_yoy"] = _pct(r["net_profit"], avail.loc[yp, "net_profit"]) if yp in avail.index else None
        history.append(rr)
    hd = pd.DataFrame(history)
    assoc = {}
    for h in horizons:
        col = f"abnormal_vs_xu100_{h}d"
        s = hd.dropna(subset=["net_profit_yoy", col])
        pos, neg = s[s["net_profit_yoy"] > 0], s[s["net_profit_yoy"] <= 0]
        assoc[f"{h}d"] = {"n_quarters": int(len(s)),
                          "mean_abnormal_after_profit_growth": float(pos[col].mean()) if len(pos) else None,
                          "n_growth": int(len(pos)),
                          "mean_abnormal_after_profit_decline": float(neg[col].mean()) if len(neg) else None,
                          "n_decline": int(len(neg))}
    def pc(x):
        return "n/a" if x is None else f"{x*100:+.1f}%"

    # per-quarter table: margin, leverage, improvement flag and the three abnormal returns
    table, prev_margin = [], None
    for i, (p, r) in enumerate(avail.iterrows()):
        m = ebitda_margin(r)
        t = ttm.iloc[i] if ttm is not None else np.nan
        nd = float(r["net_debt"] / t) if ttm is not None and not pd.isna(t) and t and not pd.isna(r.get("net_debt")) else None
        h_rec = history[i]
        yp = f"{int(p[:4]) - 1}{p[4:]}"
        ym = ebitda_margin(avail.loc[yp]) if yp in avail.index else None
        row = {"period": p, "disclosure_date_estimate": h_rec["disclosure_date_estimate"],
               "ebitda_margin": m, "ebitda_margin_change_vs_prev_quarter_pp": (None if m is None or prev_margin is None else float((m - prev_margin) * 100)),
               "ebitda_margin_yoy_change_pp": (None if m is None or ym is None else float((m - ym) * 100)),
               "net_debt_to_ttm_ebitda": nd, "net_profit_yoy": h_rec.get("net_profit_yoy"),
               "improved": (None if m is None or prev_margin is None else bool(m > prev_margin))}
        for h in horizons:
            row[f"abnormal_vs_xu100_{h}d"] = h_rec.get(f"abnormal_vs_xu100_{h}d")
            row[f"abnormal_vs_sector_{h}d"] = h_rec.get(f"abnormal_vs_sector_{h}d")
        table.append(row)
        prev_margin = m if m is not None else prev_margin
    qlines = []
    for row in table:
        mm = "n/a" if row["ebitda_margin"] is None else f"{row['ebitda_margin']*100:.1f}%"
        nd_s = "n/a" if row["net_debt_to_ttm_ebitda"] is None else f"{row['net_debt_to_ttm_ebitda']:.2f}x"
        rets = ", ".join(f"{h}d {pc(row[f'abnormal_vs_xu100_{h}d'])} vs XU100 / {pc(row[f'abnormal_vs_sector_{h}d'])} vs sector"
                         for h in horizons)
        flag = {True: "margin improved", False: "margin did not improve", None: "no prior quarter"}[row["improved"]]
        qlines.append(f"{row['period']} (disclosed ~{row['disclosure_date_estimate']}): EBITDA margin {mm}, net debt/TTM EBITDA {nd_s}, {flag}; abnormal returns {rets}.")
    imp_assoc = {}
    td = pd.DataFrame(table)
    for h in horizons:
        col = f"abnormal_vs_xu100_{h}d"
        sub = td.dropna(subset=["improved", col])
        a_, b_ = sub[sub["improved"] == True], sub[sub["improved"] == False]
        imp_assoc[f"{h}d"] = {"n_improved": int(len(a_)), "mean_abnormal_improved": float(a_[col].mean()) if len(a_) else None,
                              "n_not_improved": int(len(b_)), "mean_abnormal_not_improved": float(b_[col].mean()) if len(b_) else None}
    imp_line = ("Improved-margin quarters vs others (mean abnormal return vs XU100; only quarters with a complete window): "
                + "; ".join(f"{k}: {pc(v['mean_abnormal_improved'])} (n={v['n_improved']}) vs {pc(v['mean_abnormal_not_improved'])} (n={v['n_not_improved']})"
                            for k, v in imp_assoc.items())
                + f". Samples are tiny; this is association, not causation. Benchmark for 'vs sector' is {btype} (identical to XU100 when the stock has no sector peers).")

    lines = [f"Latest quarter disclosed by {ts.date()}: {cp} (estimated disclosure {latest_reaction['disclosure_date_estimate']}).",
             f"Revenue vs same quarter last year: {pc(changes['revenue_yoy'])}; vs previous quarter: {pc(changes['revenue_qoq'])}.",
             f"Net profit vs same quarter last year: {pc(changes['net_profit_yoy'])}; vs previous quarter: {pc(changes['net_profit_qoq'])}."]
    if changes["ebitda_margin"] is not None:
        d = changes["ebitda_margin_yoy_change_pp"]
        lines.append(f"EBITDA margin {changes['ebitda_margin']*100:.1f}%"
                     + ("" if d is None else f", {abs(d):.1f} percentage points {'higher' if d > 0 else 'lower'} than a year earlier") + ".")
    if nd_ebitda is not None:
        lines.append(f"Net debt / TTM EBITDA: {nd_ebitda:.2f}x.")
    for h in horizons:
        if latest_reaction[f"window_complete_{h}d"]:
            lines.append(f"{h}-day return after disclosure: {pc(latest_reaction[f'raw_{h}d'])}; abnormal vs XU100 "
                         f"{pc(latest_reaction[f'abnormal_vs_xu100_{h}d'])}, vs sector {pc(latest_reaction[f'abnormal_vs_sector_{h}d'])}.")
        else:
            lines.append(f"{h}-day window after disclosure is NOT complete by the analysis date (no value).")
    return {
        "ticker": ticker, "as_of": str(ts.date()), "latest_disclosed_quarter": cp, "summary": lines,
        "benchmark_type": btype, "changes": changes, "reaction_to_latest": latest_reaction,
        "quarterly_history": qlines, "improvement_vs_reaction": imp_line, "quarters_excluded_not_yet_disclosed": excluded,
        "warnings": ([warn] if warn else []) + [
            "Disclosure dates are proxies (quarter end + 30 days for financials, 45 for others), not real KAP timestamps.",
            "Abnormal return = stock return minus benchmark return over the same window; association only, "
            "not a causal effect of the results.",
            "A window that ends after the analysis date is incomplete and reported as None.",
            "Statements disclosed after the analysis date are excluded (listed above).",
            "Amounts are Fintables TRY figures restated to the 2026/9 inflation index.",
        ] + (["Fundamentals stored as year-to-date were converted to single quarters."] if ticker in YTD_TICKERS else []),
    }


# ---------------------------------------------------------------------------
# 4. Macro variable vs stock return
# ---------------------------------------------------------------------------

_MACRO_ALIASES = {
    "usdtry": "usdtry", "usd/try": "usdtry", "usd-try": "usdtry", "usd_try": "usdtry", "usd": "usdtry",
    "dolar": "usdtry", "dollar": "usdtry", "dolar/tl": "usdtry", "usdtl": "usdtry", "usd/tl": "usdtry",
    "eurtry": "eurtry", "eur/try": "eurtry", "eur-try": "eurtry", "eur_try": "eurtry", "eur": "eurtry",
    "euro": "eurtry", "eur/tl": "eurtry",
    "tcmb_rate": "tcmb_rate", "policy rate": "tcmb_rate", "policy_rate": "tcmb_rate", "faiz": "tcmb_rate",
    "interest rate": "tcmb_rate", "tcmb": "tcmb_rate", "politika faizi": "tcmb_rate", "rate": "tcmb_rate",
    "tufe_yoy": "tufe_yoy", "tufe": "tufe_yoy", "cpi": "tufe_yoy", "inflation": "tufe_yoy", "enflasyon": "tufe_yoy",
    "fed_rate": "fed_rate", "fed": "fed_rate", "fed funds": "fed_rate",
}


def _normalize_macro_variable(v: str) -> str:
    """Map free-form macro names (USD/TRY, inflation, ...) to canonical keys."""
    key = (v or "").strip().lower()
    return _MACRO_ALIASES.get(key, key)


def macro_stock_relationship(ticker: str, macro_variable: str = "usdtry", start_date: str = "",
                             end_date: str = "", frequency: str = "weekly") -> dict:
    """Correlation between stock returns and changes in a macro series, with a first-half / second-half check."""
    var = _normalize_macro_variable(macro_variable)
    if var not in ("usdtry", "eurtry", "tcmb_rate", "tufe_yoy", "fed_rate"):
        return {"error": "macro_variable must be one of usdtry, eurtry, tcmb_rate, tufe_yoy, fed_rate."}
    ts, warn = resolve_as_of(end_date)
    close = _close(ticker, str(ts.date()))
    macro = pd.read_csv(_MACRO, index_col="Date", parse_dates=True)
    macro = macro.loc[macro.index <= ts]
    st = pd.Timestamp(start_date) if start_date else max(close.index[0], macro.index[0])
    if st < macro.index[0]:
        st = macro.index[0]
    m = macro[var].reindex(close.index, method="ffill")
    d = pd.DataFrame({"px": close, "m": m}).loc[st:ts].dropna()
    if len(d) < 30:
        return {"error": "Not enough overlapping data."}
    rule = {"weekly": "W-FRI", "monthly": "ME", "daily": None}.get(frequency, "W-FRI")
    x = d if rule is None else d.resample(rule).last().dropna()
    fx_like = var in ("usdtry", "eurtry")
    sr = x["px"].pct_change()
    mc = x["m"].pct_change() if fx_like else x["m"].diff()
    z = pd.DataFrame({"s": sr, "m": mc}).dropna()
    if len(z) < 15 or z["m"].std() == 0:
        return {"error": "Macro series has too little variation in this window (e.g. a step-function rate).",
                "n_obs": int(len(z))}
    pr, pp = stats.pearsonr(z["s"], z["m"])
    sr_, sp = stats.spearmanr(z["s"], z["m"])
    half = len(z) // 2
    halves = [float(np.corrcoef(z["s"].iloc[:half], z["m"].iloc[:half])[0, 1]),
              float(np.corrcoef(z["s"].iloc[half:], z["m"].iloc[half:])[0, 1])]
    tot = float(d["px"].iloc[-1] / d["px"].iloc[0] - 1)
    out = {
        "ticker": ticker.upper(), "macro_variable": var, "frequency": frequency,
        "period": f"{d.index[0].date()} to {d.index[-1].date()}", "n_obs": int(len(z)),
        "stock_total_return": tot, "macro_start": float(d["m"].iloc[0]), "macro_end": float(d["m"].iloc[-1]),
        "pearson_r": float(pr), "pearson_p": float(pp), "spearman_r": float(sr_), "spearman_p": float(sp),
        "first_half_r": halves[0], "second_half_r": halves[1],
        "same_sign_in_both_halves": bool(halves[0] * halves[1] > 0),
        "warnings": ([warn] if warn else []) + [
            "Correlation of returns/changes is association, not causation.",
            "Nominal TRY returns embed inflation and currency depreciation; check the USD-terms return too.",
        ],
    }
    if fx_like:
        fxr = float(d["m"].iloc[-1] / d["m"].iloc[0] - 1)
        out["fx_change"] = fxr
        out["stock_return_in_fx_terms"] = float((1 + tot) / (1 + fxr) - 1)
    return out
