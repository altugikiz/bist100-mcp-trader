"""Training examples for the research tools (sector catch-up, weekday patterns, reversal events, fundamentals, macro).

Every answer is assembled from the real tool output: the tool's own evidence_summary / summary lines are quoted
verbatim, so the model learns to copy the evidence instead of re-deriving numbers. The five evaluation tickers
(ASELS, TUPRS, GUBRF, THYAO, EREGL) and the sector groups that contain them (plus Retail, used by the instructor's
sample question) are never used here; question wording is varied and is not copied from the sample questions.
"""
import json

import nudges

PRE = "Data period: 01.01.2024-10.09.2026. Analysis date: 10.09.2026. Do not use anything published after that date. "
DATES = ["2026-09-10", "2026-09-10", "2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"]
LAST_Q = {"2026-09-10": "2026Q2", "2026-06-30": "2026Q1", "2026-03-31": "2025Q4", "2025-12-31": "2025Q3", "2025-09-30": "2025Q2"}
TRAIN_END = {"2026-09-10": ["2025-12-31", "2025-06-30"], "2026-06-30": ["2025-12-31"], "2026-03-31": ["2025-09-30"],
             "2025-12-31": ["2025-06-30"], "2025-09-30": ["2025-03-31"]}
# groups that contain an evaluation ticker (or Retail) are skipped
SKIP_GROUPS = {"Retail", "Petroleum-Refining", "Technology-Defense", "Transportation-Airlines", "Chemicals", "Metals & Mining"}


def dmy(d):
    y, m, dd = d.split("-")
    return f"{dd}.{m}.{y}"


def bullets(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return "\n".join("- " + (e if isinstance(e, str) else json.dumps(e, ensure_ascii=False)) for e in x)


def pc(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def build(Chat, rnd, DISC, TICKERS, groups, call_raw):
    out = []

    def pre():
        return PRE if rnd.random() < 0.55 else ""

    def safe(fn, *a):
        try:
            r = fn(*a)
            if r:
                out.append(r)
        except Exception as exc:
            print("skip", fn.__name__, a[:2], str(exc)[:90])

    def tail(lang, d, label="INVESTIGATE"):
        if lang == "tr":
            return f"\n\nAnaliz tarihi: {d}. Etiket: {label}. Bu bir araştırma bulgusudur, al/sat önerisi ve hedef fiyat içermez.\n\n{DISC['tr']}"
        return f"\n\nAnalysis date: {d}. Label: {label}. This is research evidence, not a buy/sell recommendation or a target price.\n\n{DISC['en']}"

    # ---------------- 1) sector laggards: rank + catch-up backtest ----------------
    Q_LAG = [
        ("Within {g} ({m}), who is trailing its peers the hardest as of {dd}, judging by the past month of trading? "
         "Looking back to 2024, have such gaps tended to be made up over the following 5, 10 and 20 days, and is a catch-up flag defensible?", "en"),
        ("As of {dd}, which {g} name ({m}) has fallen furthest behind the group median recently? Did comparable gaps since January 2024 "
         "narrow within a week, two weeks or a month? Would you call it POTENTIAL_CATCH_UP_CANDIDATE?", "en"),
        ("{g} grubunda ({m}) {dd} tarihinde son bir ayda gruptan en çok kopan hisse hangisi? 2024'ten beri benzer farklar "
         "1, 2 veya 4 hafta içinde kapandı mı? Toparlanma adayı denebilir mi?", "tr"),
        ("{dd} itibarıyla {g} ({m}) içinde sektör ortalamasının en altında kalan hisseyi bul; geçmişte bu tür geri kalmalar "
         "5, 10 ve 20 günde telafi edildi mi, kanıt var mı?", "tr"),
    ]
    def ex_lag(g, d, q, lang):
        pf = pre()
        r0 = call_raw("backtest_sector_catchup", {"sector_group": g, "as_of": d})
        members = ", ".join(r0["members"])
        c = Chat(pf + q.format(g=g, m=members, dd=dmy(d)))
        rk = c.tool("rank_sector_relative_strength", {"sector_group": g, "date": d})
        bt = c.tool("backtest_sector_catchup", {"sector_group": g, "as_of": d})
        row = rk["rows"][0]
        diag = rk.get("laggard_diagnosis")
        diag = diag if isinstance(diag, str) else json.dumps(diag, ensure_ascii=False)
        if lang == "tr":
            txt = (f"{rk['as_of']} itibarıyla {g} grubunda en geride kalan hisse {row['ticker']}: lag_score_5d = {row['lag_score_5d']:.3f}, "
                   f"lag_score_10d = {row['lag_score_10d']:.3f}, lag_score_20d = {row['lag_score_20d']:.3f} (negatif skor, grup medyanının gerisinde demektir). "
                   f"Gerilik teşhisi: {diag}\n\nGeçmiş test (backtest_sector_catchup, araç özeti aynen):\n{bullets(bt['evidence_summary'])}")
        else:
            txt = (f"As of {rk['as_of']}, the biggest laggard in the {g} group is {row['ticker']}: lag_score_5d = {row['lag_score_5d']:.3f}, "
                   f"lag_score_10d = {row['lag_score_10d']:.3f}, lag_score_20d = {row['lag_score_20d']:.3f} (a negative score means behind the group median). "
                   f"Laggard diagnosis: {diag}\n\nHistorical test (backtest_sector_catchup, tool summary verbatim):\n{bullets(bt['evidence_summary'])}")
        c.msgs.append({"role": "user", "content": nudges.final_instruction("catchup")})
        return c.answer(txt + tail(lang, d))

    for g in groups:
        if g in SKIP_GROUPS:
            continue
        for d in DATES:
            q, lang = rnd.choice(Q_LAG)
            safe(ex_lag, g, d, q, lang)

    # ---------------- 2) weekday patterns ----------------
    PATS = [("Monday, Tuesday, Wednesday down, then Thursday, Friday up", "three straight losing days from Monday to Wednesday followed by gains on Thursday and Friday", "Pazartesi–Çarşamba arası üç düşüş günü, ardından Perşembe ve Cuma kazançları"),
            ("Monday down, then Friday up", "a down Monday followed by an up Friday", "düşen bir Pazartesi ve ardından yükselen bir Cuma"),
            ("Thursday up, then Friday down", "a rising Thursday followed by a falling Friday", "yükselen bir Perşembe ve ardından düşen bir Cuma"),
            ("Monday, Tuesday down, then Wednesday up", "Monday and Tuesday down, then Wednesday up", "Pazartesi ve Salı düşüş, ardından Çarşamba yükseliş"),
            ("Tuesday, Wednesday, Thursday up, then Friday down", "Tuesday–Wednesday–Thursday up, then Friday down", "Salı–Çarşamba–Perşembe yükseliş, ardından Cuma düşüş"),
            ("Monday, Tuesday, Wednesday, Thursday down, then Friday up", "four down days Monday to Thursday, then a Friday rise", "Pazartesi'den Perşembe'ye dört düşüş günü, ardından Cuma yükselişi")]
    Q_PAT = [("Was there a recurring weekly shape in {t} where {p}, between 01.01.2024 and {e1}? Check whether it carried on from {s2} to {dd}.", "en"),
             ("For {t}, look for the weekly sequence: {p}. Fit it on data up to {e1} and see if it survives {s2} through {dd}.", "en"),
             ("{t} için 01.01.2024 – {e1} döneminde şu haftalık dizilim görülüyor mu: {p}? {s2} – {dd} döneminde de sürdü mü?", "tr"),
             ("{t} hissesinde {p} dizilimini {e1} tarihine kadar test et, sonra {s2} – {dd} aralığında sınadığında ne oluyor?", "tr")]

    def ex_pat(t, d, te, pi, q, lang):
        pf = pre()
        pat, pen, ptr = PATS[pi]
        nxt = __import__("datetime").date.fromisoformat(te) + __import__("datetime").timedelta(days=1)
        c = Chat(pf + q.format(t=t, p=pen if lang == "en" else ptr, e1=dmy(te), s2=dmy(nxt.isoformat()), dd=dmy(d)))
        r = c.tool("test_weekday_pattern", {"ticker": t, "start_date": "2024-01-01", "end_date": d, "train_end": te, "pattern": pat})
        if lang == "tr":
            txt = f"{t}, desen: {ptr}.\nAraç özeti (aynen, İngilizce): {bullets(r['evidence_summary'])}"
        else:
            txt = f"{t}, pattern: {pen}.\nTool evidence summary (verbatim): {bullets(r['evidence_summary'])}"
        c.msgs.append({"role": "user", "content": nudges.final_instruction("weekday")})
        return c.answer(txt + tail(lang, d))

    for t in TICKERS:
        for _ in range(3):
            d = rnd.choice(DATES)
            te = rnd.choice(TRAIN_END[d])
            q, lang = rnd.choice(Q_PAT)
            safe(ex_pat, t, d, te, rnd.randrange(len(PATS)), q, lang)

    # ---------------- 3) reversal events: indicators then detect_reversal_events ----------------
    EV = [("lower_band_touch", "touched the lower Bollinger Band", "alt Bollinger bandına dokundu", "bounce within 5 days"),
          ("rsi_trough", "hit an RSI trough", "RSI dip bölgesine girdi", "rise within 5 days"),
          ("upper_band_touch", "touched the upper Bollinger Band", "üst Bollinger bandına dokundu", "pullback within 5 days")]

    def ex_rev(t, d, k, lang):
        et, en, tr, eff = EV[k]
        pf = pre()
        q = rnd.choice([
            f"Count the sessions where {t} {en} from the start of 2024 until {dmy(d)}. How often was the price higher five days later? "
            f"Split the touches into the ones with oversold RSI and heavy volume versus the ones during a bearish Supertrend under the cloud.",
            f"Between January 2024 and {dmy(d)}, what did {t} do after it {en}? Give the number of events, the share that rose within 5 days, "
            f"and compare the RSI-trough-with-high-volume cases to the bearish-Supertrend-below-Ichimoku cases."]) if lang == "en" else \
            rnd.choice([
            f"{t} 2024 başından {dmy(d)} tarihine kadar kaç kez {tr}? Beş gün sonra fiyat kaç kez daha yüksekti? "
            f"Aşırı satım RSI + yüksek hacimli olayları, bulut altındaki düşüş Supertrend'li olaylarla ayrı ayrı değerlendir.",
            f"{dmy(d)} tarihine kadar {t} için {tr} olaylarının sayısını ve 5 günde yükselen payını ver; RSI dibi/yüksek hacim ile "
            f"düşüş Supertrend/Ichimoku altı gruplarını karşılaştır."])
        c = Chat(pf + q)
        c.tool("calculate_indicators", {"symbol": f"{t}.IS"})
        r = c.tool("detect_reversal_events", {"ticker": t, "event_type": et, "as_of": d})
        lines = []
        for name, s in r["standard_comparisons"].items():
            n5 = s.get("n_complete_5d")
            lines.append(f"- {name}: n = {s['n']}; 5-day positive in {s.get('n_positive_5d')} of {n5} complete windows ({pc(s.get('positive_rate_5d'))}); "
                         f"average forward return 1d {pc(s.get('avg_fwd_1d'), 2)}, 3d {pc(s.get('avg_fwd_3d'), 2)}, 5d {pc(s.get('avg_fwd_5d'), 2)}, 10d {pc(s.get('avg_fwd_10d'), 2)}"
                         + ("; LOW SAMPLE WARNING" if s.get("low_sample_warning") else "")
                         + (f"; {s.get('n_excluded_incomplete_5d')} excluded (window incomplete)" if s.get("n_excluded_incomplete_5d") else ""))
        body = (f"{t}: {r['n_matching_events']} events of type {et} up to {r['as_of']}.\n" + "\n".join(lines) +
                f"\n\nTool evidence summary (verbatim): {bullets(r['evidence_summary'])}\nWarnings: {bullets(r['warnings'])}")
        if lang == "tr":
            body = body.replace("events of type", "olay, tür:").replace("Tool evidence summary (verbatim)", "Araç özeti (aynen)").replace("Warnings", "Uyarılar")
        c.msgs.append({"role": "user", "content": nudges.final_instruction("state")})
        return c.answer(body + tail(lang, d))

    for t in TICKERS:
        for k in (0, 0, rnd.randrange(len(EV))):
            safe(ex_rev, t, rnd.choice(DATES), k, rnd.choice(["en", "tr"]))

    # ---------------- 4) quarterly fundamentals ----------------
    def ex_fund(t, d, lang):
        pf = pre()
        q = rnd.choice([
            f"Walk through every quarterly report {t} published from 2024 to {dmy(d)}: the trend in EBITDA margin and net debt to EBITDA, and the "
            f"abnormal price move over the next 1, 5 and 20 trading days against the index and against sector peers. Were the better quarters rewarded more?",
            f"Using only filings public by {dmy(d)}, how have {t}'s profitability and leverage evolved quarter by quarter, and how did the stock react "
            f"after each release compared with XU100 and its sector?"]) if lang == "en" else \
            rnd.choice([
            f"{t} için 2024'ten {dmy(d)} tarihine kadar yayımlanan her çeyrek raporunda FAVÖK marjı ve net borç/FAVÖK eğilimi nasıl, "
            f"rapor sonrası 1, 5, 20 günde endeks ve sektöre göre anormal getiri ne oldu? İyileşen dönemler daha çok ödüllendirildi mi?",
            f"{dmy(d)} tarihine kadar kamuya açıklanmış bilançolarla {t} çeyrek çeyrek nasıl bir görünüm sergiledi ve açıklama sonrası fiyat tepkisi nasıldı?"])
        c = Chat(pf + q)
        c.tool("get_fundamental_metrics", {"period": LAST_Q[d], "ticker": t})
        r = c.tool("calculate_fundamental_change", {"ticker": t, "as_of": d})
        if lang == "tr":
            txt = (f"{t}, analiz tarihi {r['as_of']}, açıklanan son çeyrek {r['latest_disclosed_quarter']}.\nÖzet (araç çıktısı aynen):\n{bullets(r['summary'])}\n\n"
                   f"Çeyrek çeyrek:\n{bullets(r['quarterly_history'])}\n\nİyileşme ve fiyat tepkisi: {r['improvement_vs_reaction']}\nUyarılar:\n{bullets(r['warnings'])}")
        else:
            txt = (f"{t}, analysis date {r['as_of']}, latest disclosed quarter {r['latest_disclosed_quarter']}.\nSummary (tool output verbatim):\n{bullets(r['summary'])}\n\n"
                   f"Quarter by quarter:\n{bullets(r['quarterly_history'])}\n\nImprovement versus reaction: {r['improvement_vs_reaction']}\nWarnings:\n{bullets(r['warnings'])}")
        c.msgs.append({"role": "user", "content": nudges.final_instruction("fund")})
        return c.answer(txt + tail(lang, d))

    for t in TICKERS:
        safe(ex_fund, t, rnd.choice(DATES[:5]), rnd.choice(["en", "tr"]))

    # ---------------- 5) buy/sell or "certain to rise" requests that name a stock ----------------
    MV = [("usdtry", "USD/TRY is rising", "USD/TRY yükseliyor", "exporter", "ihracatçı"),
          ("tcmb_rate", "the policy rate is being cut", "politika faizi indiriliyor", "bank", "banka"),
          ("tufe_yoy", "inflation is high", "enflasyon yüksek", "real-asset holder", "reel varlık sahibi")]

    def ex_adv(t, d, k, lang):
        mv, claim_en, claim_tr, role_en, role_tr = MV[k]
        pf = pre()
        q = rnd.choice([
            f"It is {dmy(d)}. Since {claim_en} and {t} is a {role_en}, I'm sure it can only rise. Tell me whether to buy {t} next session and where it will be in a month.",
            f"Given that {claim_en}, is {t} (an {role_en}) a sure winner? I want a clear buy or sell and a target price. Date: {dmy(d)}.",
            f"With today's interest rate, inflation and dollar-lira level in mind, is {t} a buy for the next session? Reply with one word, BUY or SELL. Date: {dmy(d)}."]) if lang == "en" else \
            rnd.choice([
            f"Tarih {dmy(d)}. {claim_tr.capitalize()}, {t} de {role_tr}; bence sadece yükselebilir. {t} hissesini bir sonraki seans almalı mıyım, bir ay sonraki fiyatı ne olur?",
            f"{claim_tr.capitalize()} diye {t} kesin kazandırır mı? Net bir al/sat ve hedef fiyat istiyorum. Tarih: {dmy(d)}.",
            f"Son faiz, enflasyon ve USD/TRY'ye bakarak {t} yarın alınır mı? Sadece AL veya SAT de. Tarih: {dmy(d)}."])
        c = Chat(pf + q)
        c.tool("get_macro_context", {"end_date": d})
        r = c.tool("analyze_macro_stock_relationship", {"ticker": t, "macro_variable": mv, "end_date": d})
        rel = (f"{r['period']}, n = {r['n_obs']} {r['frequency']} observations: correlation r = {r['pearson_r']:.2f} (p = {r['pearson_p']:.3f}); "
               f"first half r = {r['first_half_r']}, second half r = {r['second_half_r']}.")
        extra = ""
        if r.get("fx_change_pct") is not None:
            extra = (f" Over the period the currency moved {r['fx_change_pct']}%, the stock returned {r['stock_total_return_pct']}% in TRY and "
                     f"{r['stock_return_in_fx_terms_pct']}% in FX terms.")
        if lang == "tr":
            txt = (f"Al/sat önerisi ve hedef fiyat veremem. \"Kesin yükselir\" iddiası veriyle doğrulanmıyor: {t} için {mv} ilişkisi (araç çıktısı): {rel}{extra} "
                   f"Uyarılar: {bullets(r['warnings'])} Korelasyon nedensellik değildir ve geçmiş ilişki gelecek için garanti vermez.")
        else:
            txt = (f"I cannot give a buy/sell recommendation or a price target. The claim that {t} will 'definitely go up' is not verified by the data. "
                   f"Historical relationship between {t} and {mv} (tool output): {rel}{extra} Warnings: {bullets(r['warnings'])} "
                   f"Correlation is not causation and a past relationship is not a forecast.")
        c.msgs.append({"role": "user", "content": nudges.final_instruction("advice", t)})
        return c.answer(txt + tail(lang, d))

    for t in TICKERS:
        for _ in range(2):
            safe(ex_adv, t, rnd.choice(DATES), rnd.randrange(len(MV)), rnd.choice(["en", "tr"]))

    # ---------------- 6) plain macro-relationship questions ----------------
    MVN = {"usdtry": ("USD/TRY", "USD/TRY kuru"), "tcmb_rate": ("the policy rate", "politika faizi"), "tufe_yoy": ("inflation", "enflasyon"),
           "eurtry": ("EUR/TRY", "EUR/TRY kuru"), "fed_rate": ("the Fed rate", "Fed faizi")}

    def ex_macrorel(t, d, mv, st, lang):
        pf = pre()
        en, tr = MVN[mv]
        q = (rnd.choice([f"Describe the relationship between {en} and {t} since {st}, and say whether {en} caused the moves.",
                         f"How has {t} moved with {en} since {st}? Is it a causal link?"]) if lang == "en" else
             rnd.choice([f"{st} tarihinden beri {tr} ile {t} arasındaki ilişkiyi anlat; {tr} hareketlerin nedeni mi?",
                         f"{t} hissesi {st} sonrasında {tr} ile birlikte nasıl hareket etti? Nedensel bir bağ var mı?"]))
        c = Chat(pf + q + (f" Analysis date: {dmy(d)}." if not pf else ""))
        c.tool("get_macro_context", {"end_date": d})
        r = c.tool("analyze_macro_stock_relationship", {"ticker": t, "macro_variable": mv, "start_date": st, "end_date": d})
        rel = (f"{r['period']}, n = {r['n_obs']} {r['frequency']} observations: correlation r = {r['pearson_r']:.2f} (p = {r['pearson_p']:.3f}); "
               f"first half r = {r['first_half_r']}, second half r = {r['second_half_r']}"
               + (" (same sign)." if r.get("same_sign_in_both_halves") else " (the sign differs)."))
        extra = ""
        if r.get("fx_change_pct") is not None:
            extra = (f" Over the period the currency moved {r['fx_change_pct']}%, the stock returned {r['stock_total_return_pct']}% in TRY and "
                     f"{r['stock_return_in_fx_terms_pct']}% in FX terms.")
        c.msgs.append({"role": "user", "content": nudges.final_instruction("macro")})
        if lang == "tr":
            txt = (f"{t} ve {mv} ilişkisi (araç çıktısı): {rel}{extra} Uyarılar: {bullets(r['warnings'])} Bu bir ilişkidir, nedensellik kanıtı değildir.")
        else:
            txt = (f"Relationship between {t} and {mv} (tool output): {rel}{extra} Warnings: {bullets(r['warnings'])} "
                   f"This is an association, not proof that {en} caused the moves.")
        return c.answer(txt + tail(lang, d))

    for t in TICKERS:
        mv = rnd.choice(list(MVN))
        safe(ex_macrorel, t, rnd.choice(DATES[:2]), mv, rnd.choice(["2024-06-01", "2025-01-01", "2025-06-30"]), rnd.choice(["en", "tr"]))
        mv = rnd.choice(["usdtry", "tcmb_rate", "tufe_yoy"])
        safe(ex_macrorel, t, rnd.choice(DATES[:2]), mv, rnd.choice(["2024-01-01", "2025-03-01"]), rnd.choice(["en", "tr"]))

    # advice requests that name no stock need no tool
    NS = [("Pick me one stock that cannot lose next week, and promise it.", "en",
           "No one can guarantee which stock will rise, and I cannot give a buy/sell call. If you name a stock, I can show its historical relationship with a macro variable or test a specific pattern. "),
          ("Gelecek hafta asla kaybettirmeyecek bir hisse seç ve söz ver.", "tr",
           "Hangi hissenin yükseleceğine dair garanti verilemez ve al/sat önerisi de veremem. Bir hisse adı verirsen, makro değişkenlerle tarihsel ilişkisini gösterebilir veya belirli bir örüntüyü test edebilirim. ")]
    for q, lang, a in NS * 2:
        out.append({"messages": [{"role": "system", "content": Chat("").msgs[0]["content"]}, {"role": "user", "content": pre() + q},
                                 {"role": "assistant", "content": a + DISC[lang]}], "tools": Chat("").answer("")["tools"]})
    return out
