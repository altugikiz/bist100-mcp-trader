"""
Post-answer guard for the local LLM agent.

The harness already decides the action; this module checks the model's written
answer against the tool trace and repairs it deterministically (append-only, it
never rewrites the model's analysis).  Checks:
  advice        : buy/sell wording that gives investment advice
  disclaimer    : missing "not investment advice / human review" line
  bad_action    : an action-like label that is not in ALLOWED_ACTIONS
  action_missing: harness returned an action but the answer does not state it
  unsafe_missing: a tool failed (unknown ticker, no data) but the answer has no ANALYSIS_UNSAFE
  invest_missing: a tool rejected the input (e.g. unknown sector group) but the answer has no INVESTIGATE
  numbers       : numbers in the answer that are not in any tool output
"""
import re

ALLOWED_ACTIONS = ["HOLD", "WATCH", "INVESTIGATE", "POTENTIAL_CATCH_UP_CANDIDATE",
                   "TECHNICAL_REVERSAL_CANDIDATE", "FUNDAMENTAL_IMPROVEMENT_CANDIDATE",
                   "REJECT_SIGNAL", "ANALYSIS_UNSAFE"]
_FORBIDDEN_LABELS = ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL", "ACCUMULATE", "REDUCE"]
_ADVICE = re.compile(
    r"(you should (buy|sell)|i (recommend|suggest) (buying|selling|to buy|to sell)|"
    r"is a (good|strong) buy|worth buying|time to (buy|sell)|"
    r"(almanızı|satmanızı|alın|satın)\b.{0,20}(öneririm|tavsiye)|(alım|satım) tavsiyesi veriyorum)", re.I)
_DISCLAIMER = re.compile(r"(not investment advice|yatırım tavsiyesi değildir|not a recommendation)", re.I)
_TAIL = {"tr": "Bu, eğitim amaçlı tarihsel bir analizdir; yatırım tavsiyesi değildir. Nihai karar insan incelemesine aittir.",
         "en": "This is an educational, historical analysis and not investment advice; the final decision belongs to a human reviewer."}
_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def _numbers(text):
    out = []
    for tok in _NUM.findall(text):
        try:
            out.append(float(tok.replace(",", ".")))
        except ValueError:
            pass
    return out


_INDEX_NAMES = re.compile(r"\b(BIST[\s-]?(100|30|50)|XU(100|030)|top[\s-]?100)\b", re.I)


def unsupported_numbers(answer, tool_text):
    answer = _INDEX_NAMES.sub(" ", answer)   # index names such as "BIST 100" are labels, not figures
    pool = _numbers(tool_text)
    bad, total = 0, 0
    for x in _numbers(answer):
        if abs(x) <= 10 or 1990 <= abs(x) <= 2100:
            continue
        total += 1
        if not any(abs(x - p) <= 0.006 * max(abs(p), 1e-9) or abs(x - p * 100) <= 0.006 * max(abs(p * 100), 1e-9) for p in pool):
            bad += 1
    return bad, total


def _is_tr(question):
    return bool(re.search(r"[çğıöşüÇĞİÖŞÜ]|\b(için|nedir|hissesi|analiz|var mı|hangi)\b", question))


def check_and_repair(question, answer, trace):
    """Return (answer, violations). Repairs only by appending short, tool-grounded lines."""
    viol, add = [], []
    tr = _is_tr(question)
    tool_text = " ".join(t["output"] for t in trace)
    up = answer.upper()

    if _ADVICE.search(answer):
        viol.append("advice")
    for lab in _FORBIDDEN_LABELS:
        if re.search(rf"\*\*{lab}\*\*", up):
            viol.append("bad_action")
            break

    trace = [dict(t, output=t["output"].replace('\\"', '"')) for t in trace]  # undo JSON double-encoding
    failed = [t for t in trace if '"error"' in t["output"][:80] or "tool failed" in t["output"][:120]]
    rejected = [t for t in trace if "valid_sector_groups" in t["output"] and "ANALYSIS" not in t["output"][:20]
                and ('"error"' in t["output"] or "not a valid" in t["output"].lower() or "unknown" in t["output"].lower())]
    final_action = None
    for t in trace:
        if t["tool"] == "get_recommended_action":
            m = re.search(r'"action"\s*:\s*"([A-Z_]+)', t["output"])
            if m and m.group(1) in ALLOWED_ACTIONS:
                final_action = m.group(1)

    analysed = any(t["tool"] == "get_recommended_action" and t not in failed for t in trace)
    succeeded = [t for t in trace if t not in failed]
    if failed and not analysed and not rejected and not succeeded:
        # a tool failed and the harness never produced an action: no evidence for a conclusion
        if "ANALYSIS_UNSAFE" not in up:
            viol.append("unsafe_missing")
            add.append("Harness label: **ANALYSIS_UNSAFE** — the tools returned no usable evidence." if not tr else
                       "Harness etiketi: **ANALYSIS_UNSAFE** — araçlar kullanılabilir kanıt döndürmedi.")
    elif rejected and "INVESTIGATE" not in up and "ANALYSIS_UNSAFE" not in up:
        viol.append("invest_missing")
        add.append("Evidence is insufficient for the requested input: **INVESTIGATE**." if not tr else
                   "İstenen girdi için kanıt yetersiz: **INVESTIGATE**.")
    elif final_action and final_action not in up:
        viol.append("action_missing")
        add.append(f"Harness action: **{final_action}**." if not tr else f"Harness aksiyonu: **{final_action}**.")

    research = {"backtest_sector_catchup", "test_weekday_sequence", "test_weekday_pattern", "detect_reversal_events",
                "calculate_fundamental_change", "analyze_macro_stock_relationship", "rank_sector_relative_strength"}
    if (not final_action and succeeded and any(t["tool"] in research for t in succeeded)
            and not any(a in up for a in ALLOWED_ACTIONS) and "ANALYSIS_UNSAFE" not in up):
        viol.append("label_missing")
        add.append("Research label: **INVESTIGATE** (historical evidence only; no harness action was produced)." if not tr else
                   "Araştırma etiketi: **INVESTIGATE** (yalnızca tarihsel kanıt; harness aksiyonu üretilmedi).")

    # scope line: restate analysis date / latest disclosed quarter taken from the question and tool output
    scope = []
    m = re.search(r"Analysis date:\s*(\d{2}\.\d{2}\.\d{4})", question)
    if m:
        iso = "-".join(reversed(m.group(1).split(".")))
        if m.group(1) not in answer and iso not in answer:
            scope.append(f"Analiz tarihi: {m.group(1)}" if tr else f"Analysis date: {m.group(1)}")
    q = re.search(r'(?:latest_disclosed_quarter|Latest quarter disclosed[^:]*:)\W{0,10}(\d{4}Q\d)', tool_text)
    if q and q.group(1) not in answer:
        scope.append(f"Son açıklanan çeyrek: {q.group(1)}" if tr else f"Latest disclosed quarter: {q.group(1)}")
    if scope:
        viol.append("scope")
        add.append(" | ".join(scope) + ".")

    # ignore the label where the answer itself negates it or where an earlier correction already mentions it
    up_lbl = re.sub(r"(CORRECTION|DÜZELTME):[^\n]*", "", up)
    up_lbl = re.sub(r"(NOT MEET|NOT QUALIFY|NOT LABELED|NOT BE LABELED|NOT SUPPORT|NO EVIDENCE)[^.\n]{0,120}", "", up_lbl)
    cand = [l for l in ("POTENTIAL_CATCH_UP_CANDIDATE", "TECHNICAL_REVERSAL_CANDIDATE", "FUNDAMENTAL_IMPROVEMENT_CANDIDATE")
            if l in up_lbl and l != final_action]
    if cand and not final_action:
        viol.append("unsupported_label")
        add.append(f"Correction: the label {cand[0]} above is not supported by tool output (the harness produced no such action); "
                   "the research label is **INVESTIGATE**." if not tr else
                   f"Düzeltme: yukarıdaki {cand[0]} etiketi araç çıktısıyla desteklenmiyor (harness böyle bir aksiyon üretmedi); "
                   "araştırma etiketi **INVESTIGATE**.")

    if (re.search(r"alay[ıi]m m[ıi]|hedef fiyat|target price|should i (buy|sell)|buy or sell|\bBUY\b.*\bSELL\b", question, re.I)
            and not re.search(r"no buy/sell recommendation|Al/sat önerisi ve hedef fiyat verilemez", answer)):
        viol.append("advice_request")
        add.append("No buy/sell recommendation and no target price can be given; the evidence above is historical and educational. "
                   "Claims such as 'will certainly rise' are not verified by the data." if not tr else
                   "Al/sat önerisi ve hedef fiyat verilemez; yukarıdaki kanıt tarihsel ve eğitim amaçlıdır. "
                   "'Kesin yükselir' gibi iddialar veriyle doğrulanmış değildir.")

    if not trace:
        add.append("Bu soru araç kanıtı olmadan yanıtlanamaz; al/sat kararı verilmez." if tr else
                   "No tool evidence was gathered, so no buy/sell call can be made; this is not an investment recommendation.")

    if not _DISCLAIMER.search(answer):
        viol.append("disclaimer")
        add.append(_TAIL["tr" if tr else "en"])

    bad, tot = unsupported_numbers(answer, tool_text)
    if bad:
        viol.append("numbers")
        add.append(f"Note: {bad} figure(s) above could not be matched to tool output and must not be relied on." if not tr else
                   f"Not: yukarıdaki {bad} sayı araç çıktısıyla eşleşmedi, güvenilmemelidir.")

    if add:
        answer = answer.rstrip() + "\n\n" + "\n".join(add)
    return answer, viol
