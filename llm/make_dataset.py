"""
Build a supervised fine-tuning dataset from the deterministic MCP harness.

Every example is a chat transcript: user question -> assistant tool calls ->
real tool outputs -> grounded final answer. All numbers in the answers are
copied from the tool outputs, so the data teaches tool use and grounding, not
financial facts. The five evaluation tickers (ASELS, TUPRS, GUBRF, THYAO, EREGL)
are held out and never appear in training.

Usage: python llm/make_dataset.py   ->  data/llm_train/{train,valid}.jsonl
"""
import asyncio, json, os, random, sys

sys.path.insert(0, os.path.dirname(__file__))
import agent  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HELD_OUT = {"ASELS", "TUPRS", "GUBRF", "THYAO", "EREGL"}
random.seed(7)
server = agent.load_server()
TICKERS = [t for t in server.UNIVERSE["ticker"] if t not in HELD_OUT]
TOOLS = agent.tool_specs(server)


def _system_prompt() -> str:
    """Use the same system prompt as the deployed Modelfile so training and inference match."""
    txt = open(os.path.join(os.path.dirname(__file__), "Modelfile")).read()
    return txt.split('SYSTEM """', 1)[1].rsplit('"""', 1)[0].strip()


SYSTEM = _system_prompt()


def call(name, args):
    r = asyncio.run(server.mcp.call_tool(name, args))
    r = r[1] if isinstance(r, tuple) and len(r) > 1 else r
    if isinstance(r, list):
        r = [getattr(c, "text", str(c)) for c in r]
        return r[0] if len(r) == 1 else r
    return r


def parsed(raw):
    return json.loads(raw) if isinstance(raw, str) else raw


class Chat:
    """Collects one transcript while executing real tool calls."""
    def __init__(self, question):
        self.msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]

    def tool(self, name, args):
        # Same wire format the agent loop uses: list with one JSON string
        raw = call(name, args)
        self.msgs.append({"role": "assistant", "content": "",
                          "tool_calls": [{"type": "function",
                                          "function": {"name": name, "arguments": args}}]})
        out = json.dumps([raw] if isinstance(raw, str) else raw, default=str)
        self.msgs.append({"role": "tool", "name": name, "content": out})
        return parsed(raw)

    def answer(self, text):
        self.msgs.append({"role": "assistant", "content": text})
        return {"messages": self.msgs, "tools": TOOLS}


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ---------------- question phrasings (tr, en) ----------------
Q_FULL = [("{t} hissesi için tam analiz yap.", "tr"), ("{t} için tam analiz yapar mısın?", "tr"),
          ("{t} hissesini harness ile analiz et ve sonucu açıkla.", "tr"),
          ("{t} hissesinin durumu nedir? Tüm araçları kullanarak analiz et.", "tr"),
          ("Run a full analysis for {t}.", "en"), ("Analyze {t} stock with the harness and explain the result.", "en")]
Q_ADV = [("{t} hissesini alayım mı?", "tr"), ("{t} için al/sat tavsiyesi verir misin?", "tr"),
         ("Should I buy {t} today?", "en"), ("Is {t} a good buy right now?", "en")]
Q_SECT = [("Sektör içinde en geride kalan hisseler hangileri?", "tr"), ("Hangi hisseler sektörlerine göre geri kalmış?", "tr"),
          ("Which stocks lag their sector the most?", "en")]
Q_WD = [("{t} için haftanın günü etkisi var mı?", "tr"), ("{t} hissesinde gün bazlı getiri örüntülerini test et.", "tr"),
        ("Is there a weekday effect in {t}?", "en")]
Q_REV = [("{t} için teknik dönüş olaylarının istatistikleri nedir?", "tr"), ("{t} hissesinde RSI ve Bollinger dönüş olayları ne kadar güvenilir?", "tr"),
         ("What do reversal events look like for {t}?", "en")]
Q_MAC = [("Makro ortam nasıl? Kur ve faiz durumu nedir?", "tr"), ("Güncel makro bağlamı özetle.", "tr"), ("Summarize the macro context.", "en")]
Q_NEWS = [("{t} için haber olayları neler?", "tr"), ("{t} hakkında kayıtlı haber olaylarını göster.", "tr"), ("What news events are recorded for {t}?", "en")]
Q_DQ = [("{t} için veri kalitesi nasıl, analiz yapmak güvenli mi?", "tr"), ("Is the data for {t} good enough to analyze?", "en")]

DISC = {"tr": "Bu, eğitim amaçlı tarihsel bir analizdir; yatırım tavsiyesi değildir. Nihai karar insan incelemesine aittir.",
        "en": "This is an educational, historical analysis and not investment advice; the final decision belongs to a human reviewer."}

NO_TOOL = [
    ("RSI nedir?", "RSI (Göreceli Güç Endeksi), fiyatın son dönemdeki yükseliş ve düşüşlerinin gücünü 0-100 arasında ölçen bir momentum göstergesidir. Geleneksel olarak 70 üzeri aşırı alım, 30 altı aşırı satım olarak yorumlanır; ancak tek başına al/sat sinyali değildir."),
    ("Bollinger Bantları nedir?", "Bollinger Bantları, hareketli ortalamanın üstüne ve altına standart sapma kadar çizilen iki banttan oluşur. Fiyatın bantlara değmesi oynaklık ve olası dönüş bölgeleri hakkında bilgi verir, ancak tek başına sinyal sayılmamalıdır."),
    ("Bonferroni düzeltmesi neden gerekli?", "Çok sayıda hipotez test edildiğinde bazıları sadece şans eseri anlamlı çıkar. Bonferroni düzeltmesi anlamlılık eşiğini test sayısına bölerek bu yanlış pozitif riskini azaltır."),
    ("Look-ahead bias nedir?", "Look-ahead bias, bir analizde o tarihte henüz bilinmeyen bilginin kullanılmasıdır. Örneğin bilanço açıklanmadan önce o bilançonun verisini kullanmak. Bu yüzden temel veriler açıklanma tarihine göre hizalanır."),
    ("Backtest başarısı gelecekteki kazancı garanti eder mi?", "Hayır. Backtest geçmiş veriye dayanır; aşırı uyum, işlem maliyetleri ve rejim değişimleri nedeniyle gelecekte aynı sonucun gelmesi beklenemez. Bu yüzden walk-forward doğrulama ve benchmark karşılaştırması gerekir."),
    ("Tahmin doğruluğu ile kârlı strateji aynı şey mi?", "Hayır. Bir model yönü yüksek doğrulukla tahmin edebilir ama işlem maliyetleri, küçük getiriler veya nadir büyük kayıplar yüzünden kârlı olmayabilir. Doğruluk ile finansal performans ayrı ölçülür."),
    ("Sharpe oranı ne anlatır?", "Sharpe oranı, birim risk (getiri oynaklığı) başına elde edilen fazla getiriyi ölçer. Yüksek değer daha iyi risk-ayarlı performansa işaret eder, ancak geçmiş veriye dayandığı için garanti sayılmaz."),
    ("Maksimum drawdown nedir?", "Maksimum drawdown, bir portföy değerinin tepe noktasından sonraki en büyük yüzdesel düşüşüdür. Stratejinin yaşayabileceği en kötü tarihsel kaybı gösterir."),
    ("Point-in-time veri ne demek?", "Point-in-time veri, bir bilginin o tarihte gerçekten kamuya açık olduğu haliyle kullanılmasıdır. Finansal tablolar dönem sonuna değil, açıklandıkları tarihe göre hizalanır; böylece geleceğe bakma hatası önlenir."),
    ("What is RSI?", "RSI (Relative Strength Index) is a 0-100 momentum indicator comparing recent gains and losses. Readings above 70 are conventionally seen as overbought and below 30 as oversold, but it is not a standalone buy or sell signal."),
    ("What is look-ahead bias?", "Look-ahead bias means using information in an analysis that was not yet known at that date, such as using a quarterly report before it was published. Point-in-time alignment of disclosures prevents it."),
    ("Why does backtest performance not guarantee future returns?", "A backtest only describes the past. Overfitting, transaction costs and regime changes mean future results can differ, which is why walk-forward validation and benchmark comparison are needed."),
    ("Sen kimsin?", "Ben, bu projede BIST 100 analizleri için hazırlanan yerel bir dil modeliyim. Sayısal verileri kendim hesaplamam; fiyat, gösterge ve istatistikleri MCP araçlarından alırım. Yatırım tavsiyesi vermem."),
    ("Which tools do you use?", "I call MCP tools for stock history, indicators, fundamentals, sector ranking, weekday tests, reversal events, macro context, news and data quality. I never invent numbers; they come from tool outputs."),
]


def full_flow(t, q, lang):
    c = Chat(q)
    s = c.tool("start_analysis_session", {"ticker": t})
    sid = s["session_id"]
    f = c.tool("run_full_analysis", {"session_id": sid})
    a = c.tool("get_recommended_action", {"session_id": sid})
    ev = f["committed_evidence"][:5]
    g = f["gate_result"]
    ev_txt = "\n".join(f"- {e['id']}: {e['statement']}" for e in ev)
    warn = g["warnings"] or []
    warn_txt = "; ".join(warn) if warn else ("yok" if lang == "tr" else "none")
    why = a["reasoning"][0] if a.get("reasoning") else ""
    return c, s, g, a, ev_txt, warn_txt, why


def ex_full(t, q, lang, advice=False):
    c, s, g, a, ev, warn, why = full_flow(t, q, lang)
    act = a["action"]
    if lang == "tr":
        head = (f"{t} için araçların önerdiği analitik etiket: **{act}**. "
                + (f"Gerekçe: {why} " if why else ""))
        if advice:
            head = ("Al/sat tavsiyesi veremem; bu bir eğitim projesi ve nihai karar insana ait. "
                    f"Yine de {t} için harness'in analitik etiketi: **{act}**. " + (f"Gerekçe: {why} " if why else ""))
        txt = (f"{head}\n\nKanıtlar:\n{ev}\n\nVeri kalite kapısı: {g['status']}. Uyarılar: {warn}.\n"
               f"Fiyat verisi bitiş tarihi: {s['price_end']}.\n\n{DISC['tr']}")
    else:
        head = f"The harness's analytical label for {t} is **{act}**. " + (f"Reason: {why} " if why else "")
        if advice:
            head = ("I can't give buy or sell advice; this is an educational project and the final decision belongs to a human. "
                    f"The harness's analytical label for {t} is **{act}**. " + (f"Reason: {why} " if why else ""))
        txt = (f"{head}\n\nEvidence:\n{ev}\n\nData-quality gate: {g['status']}. Warnings: {warn}.\n"
               f"Price data end date: {s['price_end']}.\n\n{DISC['en']}")
    return c.answer(txt)


def ex_sector(q, lang):
    c = Chat(q)
    r = c.tool("rank_sector_relative_strength", {})
    rows = sorted(r["rows"], key=lambda x: x["lag_score_20d"])[:3]
    lines = "\n".join(f"- {x['ticker']} ({x['sector_group']}): lag_score_20d = {x['lag_score_20d']:.3f}, "
                      f"lag_score_5d = {x['lag_score_5d']:.3f}" for x in rows)
    if lang == "tr":
        txt = (f"{r['as_of']} tarihi itibarıyla 20 günlük göreli gerilik skoruna göre en geride kalan hisseler:\n{lines}\n\n"
               f"Negatif skor, hissenin kıyas grubunun gerisinde kaldığını gösterir; bu tek başına bir toparlanma sinyali değildir. {DISC['tr']}")
    else:
        txt = (f"As of {r['as_of']}, the biggest laggards by the 20-day relative lag score:\n{lines}\n\n"
               f"A negative score means the stock trails its benchmark; on its own it is not a catch-up signal. {DISC['en']}")
    return c.answer(txt)


def ex_weekday(t, q, lang):
    c = Chat(q)
    r = c.tool("test_weekday_pattern", {"ticker": t})
    days = r["weekdays"]
    best = max(days, key=lambda d: d["Mean_Return"])
    sig = [d["Weekday"] for d in days if d["Significant_Bonferroni"]]
    oos = sum(1 for d in days if d["same_sign_oos"])
    if lang == "tr":
        sg = ("Bonferroni sonrası anlamlı gün: " + ", ".join(sig)) if sig else "Bonferroni düzeltmesi sonrası anlamlı gün yok"
        txt = (f"{t}, {r['period']} ({r['n_days']} gün). En yüksek ortalama getirili gün {best['Weekday']}: ortalama {pct(best['Mean_Return'], 2)}, "
               f"ham p = {best['P_Value']:.3f}. {sg}. Eğitim/test ayrımında {len(days)} günün {oos} tanesinde işaret test döneminde de aynı kaldı. "
               f"Not: {r['note']}\n\n{DISC['tr']}")
    else:
        sg = ("Bonferroni-significant: " + ", ".join(sig)) if sig else "no weekday is significant after Bonferroni correction"
        txt = (f"{t}, {r['period']} ({r['n_days']} days). The highest mean return is on {best['Weekday']}: {pct(best['Mean_Return'], 2)}, "
               f"raw p = {best['P_Value']:.3f}. {sg}. The sign held out-of-sample for {oos} of {len(days)} days. Note: {r['note']}\n\n{DISC['en']}")
    return c.answer(txt)


def ex_reversal(t, q, lang):
    c = Chat(q)
    r = c.tool("detect_reversal_events", {"ticker": t})
    st = r["statistics"]
    picks = [k for k in ("lower_band_touch", "rsi_trough", "upper_band_touch", "rsi_peak") if k in st]
    lines = "\n".join(f"- {k}: 5 günlük bounce oranı {pct(st[k]['bounce_rate_5d'])} (n={st[k]['n']})"
                      + (" — düşük örneklem uyarısı" if st[k].get("low_sample_warning") else "") for k in picks)
    if lang == "en":
        lines = lines.replace("5 günlük bounce oranı", "5-day bounce rate").replace("düşük örneklem uyarısı", "low-sample warning")
        txt = f"Reversal-event statistics for {t} (as of {r['as_of']}):\n{lines}\n\nBounce rates are historical frequencies, not forecasts. {DISC['en']}"
    else:
        txt = f"{t} için teknik dönüş olayı istatistikleri ({r['as_of']} itibarıyla):\n{lines}\n\nBounce oranları tarihsel sıklıktır, tahmin değildir. {DISC['tr']}"
    return c.answer(txt)


def ex_macro(q, lang):
    c = Chat(q)
    r = c.tool("get_macro_context", {})
    e, ch = r["end"], r["change"]
    if lang == "tr":
        txt = (f"{e['date']} itibarıyla: USD/TRY {e['usdtry']:.2f}, EUR/TRY {e['eurtry']:.2f}, TCMB politika faizi %{e['tcmb_rate']}, "
               f"TÜFE yıllık %{_n(e['tufe_yoy'], 1)}, Fed faizi %{e['fed_rate']}. Dönem içinde USD/TRY %{ch['usdtry_pct']:.1f} değişti, "
               f"TCMB faizi {ch['tcmb_rate_pp']:+} puan değişti. Not: TÜFE bazı noktalarda doğrusal interpolasyonla üretilmiştir.\n\n{DISC['tr']}")
    else:
        txt = (f"As of {e['date']}: USD/TRY {e['usdtry']:.2f}, EUR/TRY {e['eurtry']:.2f}, CBRT policy rate {e['tcmb_rate']}%, "
               f"CPI YoY {_n(e['tufe_yoy'], 1)}%, Fed rate {e['fed_rate']}%. Over the window USD/TRY moved {ch['usdtry_pct']:.1f}% and the CBRT rate changed {ch['tcmb_rate_pp']:+} pp. "
               f"Note: CPI is linearly interpolated between anchor points.\n\n{DISC['en']}")
    return c.answer(txt)


def ex_news(t, q, lang):
    c = Chat(q)
    r = c.tool("get_news_events", {"ticker": t})
    ev = r["events"][:3]
    lines = "\n".join(f"- {x['date']} [{x['category']}]: {x['summary'].split('; ')[0][:170]}" for x in ev)
    if not ev:
        txt = (f"{t} için kayıtlı haber olayı bulunmuyor (araç sonucu: {r['n_events']} olay)." if lang == "tr"
               else f"No news events are recorded for {t} (tool result: {r['n_events']} events).")
    elif lang == "tr":
        txt = f"{t} için {r['n_events']} kayıtlı olay var. İlk olaylar:\n{lines}\n\nHaberler bağlam kanıtıdır, tek başına sinyal değildir. {DISC['tr']}"
    else:
        txt = f"{t} has {r['n_events']} recorded events. First ones:\n{lines}\n\nNews is contextual evidence, not a signal by itself. {DISC['en']}"
    return c.answer(txt)


def ex_dq(t, q, lang):
    c = Chat(q)
    r = c.tool("check_data_quality", {"ticker": t})
    w = "; ".join(r["warnings"]) or ("yok" if lang == "tr" else "none")
    b = "; ".join(r["blocks"]) or ("yok" if lang == "tr" else "none")
    if lang == "tr":
        txt = (f"{t} veri kalitesi durumu: {r['status']}. Engeller: {b}. Uyarılar: {w}. Fiyat verisi bitiş tarihi: {r['price_end']}. "
               f"Temel veri mevcut mu: {'evet' if r['fundamentals_available'] else 'hayır'}.\n\n{DISC['tr']}")
    else:
        txt = (f"Data-quality status for {t}: {r['status']}. Blocks: {b}. Warnings: {w}. Price data end date: {r['price_end']}. "
               f"Fundamentals available: {'yes' if r['fundamentals_available'] else 'no'}.\n\n{DISC['en']}")
    return c.answer(txt)


# ---------------- extra generators: sector groups, ranges, event types, data-quality focus ----------------
import calendar

MONTHS_TR = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
GROUPS = sorted(server.UNIVERSE["sector_group"].unique().tolist())

Q_GRP_LAG = [("{g} grubunda 20 günde sektörüne göre en çok geride kalan hisseler hangileri?", "tr"),
             ("{g} grubunda en geride kalan hisse hangisi?", "tr"),
             ("{g} sektöründeki hisseleri göreli güce göre sırala.", "tr"),
             ("Which stocks in the {g} group lag their sector most over 20 days?", "en"),
             ("Which {g} stock is the biggest laggard right now?", "en"),
             ("Rank the {g} group by sector-relative strength.", "en")]
Q_DQ_FULL = [("{t} hissesini harness ile analiz et ve veri kalitesi uyarılarını açıkla.", "tr"),
             ("{t} için analiz yap, veri kalitesi uyarıları ne diyor?", "tr"),
             ("Analyse {t} with the harness and explain the data-quality warnings.", "en"),
             ("Run the harness on {t} and tell me what the data-quality gate warns about.", "en")]
Q_REV_KEY = [("Supertrend", "supertrend", "Supertrend dönüşlerinden sonra {t} için ne oldu?", "What happened after Supertrend flips for {t}?"),
             ("KAMA", "kama", "{t} için KAMA kesişimlerinin sonrası nasıl?", "How did {t} behave after KAMA crosses?"),
             ("Ichimoku", "ichimoku", "{t} hissesinde Ichimoku olaylarının istatistikleri nedir?", "What are the Ichimoku event statistics for {t}?"),
             ("alt Bollinger", "lower_band", "{t} hissesinde alt Bollinger bandına dokunuşlar kaç kez 5 günlük toparlanmaya yol açtı?",
              "How often did lower Bollinger band touches on {t} lead to a 5-day bounce?"),
             ("RSI", "rsi", "{t} için RSI uç olaylarından sonra ne oluyor?", "What happens after RSI extremes for {t}?")]
Q_WD_DAY = [("{t} hissesinde Pazartesi veya Cuma etkisi örneklem dışında kalıcı mı?", "tr", ["Monday", "Friday"]),
            ("Does {t} show a stable Monday or Friday effect out of sample?", "en", ["Monday", "Friday"]),
            ("{t} için Perşembe etkisi istatistiksel olarak anlamlı mı?", "tr", ["Thursday"]),
            ("Is the Wednesday effect in {t} significant after Bonferroni correction?", "en", ["Wednesday"])]
Q_MAC_RNG = [("{m} {y} döneminde TCMB faizi ve USD/TRY nasıl değişti?", "tr"),
             ("How did the TCMB policy rate and USD/TRY change during {m} {y}?", "en"),
             ("{m} {y} için makro bağlam nedir (USD/TRY, TCMB faizi, TÜFE, Fed)?", "tr"),
             ("What is the macro context (USD/TRY, TCMB rate, CPI, Fed) for {m} {y}?", "en")]


def _n(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def ex_group(g, q, lang):
    c = Chat(q)
    r = c.tool("rank_sector_relative_strength", {"sector_group": g})
    if "error" in r or not r.get("rows"):
        return None
    rows = sorted(r["rows"], key=lambda x: (x["lag_score_20d"] is None, x["lag_score_20d"]))
    top = rows[:5] if "sırala" in q or "Rank" in q else rows[:3]
    if "biggest laggard" in q or "en geride kalan hisse hangisi" in q:
        top = rows[:1]
    lines = "\n".join(f"- {x['ticker']}: lag_score_20d = {_n(x['lag_score_20d'])}, lag_score_5d = {_n(x['lag_score_5d'])}, "
                      f"grup içi sıra {x['intra_group_rank']}" if lang == "tr" else
                      f"- {x['ticker']}: lag_score_20d = {_n(x['lag_score_20d'])}, lag_score_5d = {_n(x['lag_score_5d'])}, "
                      f"intra-group rank {x['intra_group_rank']}" for x in top)
    if lang == "tr":
        txt = (f"{r['as_of']} itibarıyla {g} grubunda 20 günlük göreli gerilik skoruna göre sıralama:\n{lines}\n\n"
               f"Negatif skor, hissenin grup medyanının gerisinde kaldığını gösterir; bu tek başına bir toparlanma sinyali değildir. {DISC['tr']}")
    else:
        txt = (f"As of {r['as_of']}, ranking in the {g} group by the 20-day relative lag score:\n{lines}\n\n"
               f"A negative score means the stock trails its group median; on its own it is not a catch-up signal. {DISC['en']}")
    return c.answer(txt)


def ex_full_dq(t, q, lang):
    c, s, g, a, ev, warn, why = full_flow(t, q, lang)
    if lang == "tr":
        txt = (f"{t} için veri kalite kapısının durumu: {g['status']}. Uyarılar: {warn}.\n"
               f"Bu uyarılar analizi engellemez ama yorumun güvenilirliğini azaltır. Fiyat verisi bitiş tarihi: {s['price_end']}.\n"
               f"Harness'in analitik etiketi: **{a['action']}**. " + (f"Gerekçe: {why}\n" if why else "\n")
               + f"\nKanıtlar:\n{ev}\n\n{DISC['tr']}")
    else:
        txt = (f"Data-quality gate for {t}: {g['status']}. Warnings: {warn}.\n"
               f"These warnings do not block the analysis but reduce how far the result can be trusted. Price data end date: {s['price_end']}.\n"
               f"The harness's analytical label is **{a['action']}**. " + (f"Reason: {why}\n" if why else "\n")
               + f"\nEvidence:\n{ev}\n\n{DISC['en']}")
    return c.answer(txt)


def ex_rev_key(t, kw, qtr, qen, lang):
    q = (qtr if lang == "tr" else qen).format(t=t)
    c = Chat(q)
    r = c.tool("detect_reversal_events", {"ticker": t})
    st = {k: v for k, v in r.get("statistics", {}).items() if kw in k.lower()}
    if not st:
        return None
    lines = "\n".join(f"- {k}: n={v['n']}, 5-day bounce rate {pct(v['bounce_rate_5d'])}, average 5-day forward return {pct(v['avg_fwd_5d'], 2)}"
                      + (", low-sample warning" if v.get("low_sample_warning") else "") for k, v in list(st.items())[:4])
    if lang == "tr":
        lines = (lines.replace("5-day bounce rate", "5 günlük bounce oranı").replace("average 5-day forward return", "ortalama 5 günlük ileri getiri")
                 .replace("low-sample warning", "düşük örneklem uyarısı"))
        txt = f"{t} için ilgili olay istatistikleri ({r['as_of']} itibarıyla):\n{lines}\n\nBunlar tarihsel sıklıklardır, tahmin değildir. {DISC['tr']}"
    else:
        txt = f"Relevant event statistics for {t} (as of {r['as_of']}):\n{lines}\n\nThese are historical frequencies, not forecasts. {DISC['en']}"
    return c.answer(txt)


def ex_wd_day(t, q, lang, days):
    c = Chat(q.format(t=t))
    r = c.tool("test_weekday_pattern", {"ticker": t})
    byday = {d["Weekday"]: d for d in r["weekdays"]}
    lines = []
    for dn in days:
        d = byday[dn]
        lines.append(f"- {dn}: mean {pct(d['Mean_Return'], 2)}, raw p = {d['P_Value']:.3f}, Bonferroni significant: "
                     f"{'yes' if d['Significant_Bonferroni'] else 'no'}, sign held out-of-sample: {'yes' if d['same_sign_oos'] else 'no'}")
    body = "\n".join(lines)
    stable = [dn for dn in days if byday[dn]["Significant_Bonferroni"] and byday[dn]["same_sign_oos"]]
    if lang == "tr":
        body = (body.replace("mean", "ortalama").replace("raw p", "ham p").replace("Bonferroni significant", "Bonferroni sonrası anlamlı")
                .replace("sign held out-of-sample", "işaret örneklem dışında korundu").replace("yes", "evet").replace("no", "hayır"))
        verdict = ("Kalıcı sayılabilecek gün: " + ", ".join(stable)) if stable else "Bu günlerin hiçbiri hem Bonferroni sonrası anlamlı hem de örneklem dışında kalıcı değil."
        txt = f"{t}, {r['period']} ({r['n_days']} gün):\n{body}\n\n{verdict} Not: {r['note']}\n\n{DISC['tr']}"
    else:
        verdict = ("Days that look stable: " + ", ".join(stable)) if stable else "None of these days is both Bonferroni-significant and stable out of sample."
        txt = f"{t}, {r['period']} ({r['n_days']} days):\n{body}\n\n{verdict} Note: {r['note']}\n\n{DISC['en']}"
    return c.answer(txt)


def ex_macro_rng(y, m, q, lang):
    last = calendar.monthrange(y, m)[1]
    start, end = f"{y}-{m:02d}-01", f"{y}-{m:02d}-{last:02d}"
    name = MONTHS_TR[m - 1] if lang == "tr" else MONTHS_EN[m - 1]
    c = Chat(q.format(m=name, y=y))
    r = c.tool("get_macro_context", {"start_date": start, "end_date": end})
    if "error" in r:
        return None
    s, e, ch = r["start"], r["end"], r["change"]
    if lang == "tr":
        txt = (f"{s['date']} - {e['date']} aralığında: USD/TRY {s['usdtry']:.2f} -> {e['usdtry']:.2f} (%{ch['usdtry_pct']:.1f}), "
               f"TCMB politika faizi %{s['tcmb_rate']} -> %{e['tcmb_rate']} ({ch['tcmb_rate_pp']:+} puan), TÜFE yıllık %{_n(e['tufe_yoy'], 1)}, "
               f"Fed faizi %{e['fed_rate']}. Not: TÜFE bazı noktalarda doğrusal interpolasyonla üretilmiştir.\n\n{DISC['tr']}")
    else:
        txt = (f"From {s['date']} to {e['date']}: USD/TRY {s['usdtry']:.2f} -> {e['usdtry']:.2f} ({ch['usdtry_pct']:.1f}%), "
               f"CBRT policy rate {s['tcmb_rate']}% -> {e['tcmb_rate']}% ({ch['tcmb_rate_pp']:+} pp), CPI YoY {_n(e['tufe_yoy'], 1)}%, "
               f"Fed rate {e['fed_rate']}%. Note: CPI is linearly interpolated between anchor points.\n\n{DISC['en']}")
    return c.answer(txt)


# ---------------- extra generators 2: legacy analytics tools, unknown inputs, backtest ----------------
import re

NOT_IN_UNIVERSE = ["GARAN", "SISE", "ASTOR", "SAHOL", "HEKTS", "KOZAL", "VAKBN", "YKBNK", "PGSUS1", "BRSAN", "XYZAB", "QWERT", "LMNOP", "TEST1", "FOOBR", "MNBVC"]
Q_IND = [("{t} için teknik göstergeler (SMA, RSI, MACD) ne diyor?", "tr"), ("{t} hissesinde RSI ve MACD durumu nedir?", "tr"),
         ("What do the SMA, RSI and MACD indicators say for {t}?", "en"), ("Is {t} overbought or oversold by RSI?", "en")]
Q_HIST = [("{t} hissesinin son 5 günlük fiyat hareketi nasıl?", "tr"), ("Show the last 5 trading days of {t}.", "en"),
          ("{t} son kapanış fiyatı nedir?", "tr"), ("What was the latest close for {t}?", "en")]
Q_SR = [("{t} için destek ve direnç seviyeleri nerede?", "tr"), ("Where are the nearest support and resistance levels for {t}?", "en")]
Q_FS = [("ASELS için {p} dönemi temel analiz skoru nedir?", "tr"), ("What is the ASELS fundamental score for {p}?", "en"),
        ("{p} döneminde ASELS temel kalite kriterlerinden hangilerini geçiyor?", "tr")]
Q_BT = [("ASELS stratejisi 2024-2025'te nasıl performans gösterdi?", "tr", {}), ("How did the ASELS strategy perform in the backtest?", "en", {}),
        ("Backtest özeti: Sharpe oranı ve maksimum düşüş nedir?", "tr", {}),
        ("Run the backtest with 60% technical and 40% fundamental weight.", "en", {"tech_weight": 0.6, "fund_weight": 0.4}),
        ("Teknik ağırlığı 0.4, temel ağırlığı 0.6 yaparak backtest çalıştır.", "tr", {"tech_weight": 0.4, "fund_weight": 0.6}),
        ("Backtest with technical weight 0.7 and fundamental weight 0.3: how does it compare?", "en", {"tech_weight": 0.7, "fund_weight": 0.3})]
Q_UNK = [("{t} hissesini analiz et.", "tr"), ("{t} için tam analiz yapar mısın?", "tr"), ("Run a full analysis for {t}.", "en"), ("Analyse {t} with the harness.", "en")]
Q_BADGRP = [("Crypto grubunda en geride kalan hisseler hangileri?", "tr", "Crypto"), ("Which stocks in the Aviation group lag their sector most?", "en", "Aviation"),
            ("Rank the Gaming group by sector-relative strength.", "en", "Gaming"),
            ("Space grubunda sektöre göre zayıf kalan hisseleri sırala.", "tr", "Space"),
            ("Which Retail stocks are weakest versus their sector?", "en", "Retail"),
            ("Cryptocurrency grubunda en geride kalanlar kimler?", "tr", "Cryptocurrency")]


def _find(d, key):
    """Depth-first search for a key in nested dicts (None when absent)."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def ex_ind(t, q, lang):
    c = Chat(q.format(t=t))
    r = c.tool("calculate_indicators", {"symbol": f"{t}.IS"})
    if "error" in r:
        return None
    sg = r["signals"]
    if lang == "tr":
        txt = (f"{t}.IS, {r['as_of']} itibarıyla: kapanış {r['close']}, SMA20 {r['SMA20']}, SMA50 {r['SMA50']}, RSI14 {r['RSI14']}, "
               f"MACD {r['MACD']} (sinyal {r['MACD_Signal']}, histogram {r['MACD_Hist']}). Trend: {sg['trend']}, fiyat SMA20'ye göre {sg['price_vs_sma20']}, "
               f"RSI bölgesi: {sg['rsi_zone']}, MACD kesişimi: {sg['macd_crossover']}.\n\nBunlar tarihsel göstergelerdir, tahmin değildir. {DISC['tr']}")
    else:
        txt = (f"{t}.IS as of {r['as_of']}: close {r['close']}, SMA20 {r['SMA20']}, SMA50 {r['SMA50']}, RSI14 {r['RSI14']}, "
               f"MACD {r['MACD']} (signal {r['MACD_Signal']}, histogram {r['MACD_Hist']}). Trend: {sg['trend']}, price is {sg['price_vs_sma20']} SMA20, "
               f"RSI zone: {sg['rsi_zone']}, MACD crossover: {sg['macd_crossover']}.\n\nThese are historical indicators, not forecasts. {DISC['en']}")
    return c.answer(txt)


def ex_hist(t, q, lang):
    c = Chat(q.format(t=t))
    r = c.tool("get_stock_history", {"symbol": f"{t}.IS", "last_n": 5})
    rows = r.get("rows") or []
    if not rows:
        return None
    lines = "\n".join(f"- {x['date']}: açılış {x['open']}, yüksek {x['high']}, düşük {x['low']}, kapanış {x['close']}, hacim {x['volume']}" if lang == "tr" else
                      f"- {x['date']}: open {x['open']}, high {x['high']}, low {x['low']}, close {x['close']}, volume {x['volume']}" for x in rows)
    last = rows[-1]
    if lang == "tr":
        txt = f"{t}.IS için son {len(rows)} işlem günü:\n{lines}\n\nSon kapanış {last['close']} ({last['date']}). Fiyat verisi bu tarihte bitiyor.\n\n{DISC['tr']}"
    else:
        txt = f"Last {len(rows)} trading days for {t}.IS:\n{lines}\n\nLatest close {last['close']} ({last['date']}). Price data ends on this date.\n\n{DISC['en']}"
    return c.answer(txt)


def ex_sr(t, q, lang):
    c = Chat(q.format(t=t))
    r = c.tool("find_support_resistance_levels", {"symbol": f"{t}.IS"})
    if "error" in r:
        return None
    body = json.dumps({k: v for k, v in r.items() if k not in ("symbol",)}, ensure_ascii=False, default=str)[:700]
    if lang == "tr":
        txt = f"{t}.IS için araç sonucu (en yakın destek/direnç seviyeleri):\n{body}\n\nBu seviyeler geçmiş pivot noktalarından hesaplanır; fiyatın oradan döneceğini garanti etmez. {DISC['tr']}"
    else:
        txt = f"Tool result for {t}.IS (nearest support and resistance levels):\n{body}\n\nThese levels come from past pivot points and do not guarantee a reversal. {DISC['en']}"
    return c.answer(txt)


def ex_fscore(p, q, lang):
    c = Chat(q.format(p=p))
    r = c.tool("get_fundamental_score", {"period": p})
    if "error" in r or "fundamental_score" not in r:
        return None
    crit = {k: v for k, v in r.items() if isinstance(v, bool)}
    passed = [k for k, v in crit.items() if v]
    failed = [k for k, v in crit.items() if not v]
    if lang == "tr":
        txt = (f"ASELS {p} temel analiz skoru: {r['fundamental_score']}/5. Geçilen kriterler: {', '.join(passed) or 'yok'}. "
               f"Geçilemeyen kriterler: {', '.join(failed) or 'yok'}.\n\n{DISC['tr']}")
    else:
        txt = (f"ASELS fundamental score for {p}: {r['fundamental_score']}/5. Criteria passed: {', '.join(passed) or 'none'}. "
               f"Criteria not passed: {', '.join(failed) or 'none'}.\n\n{DISC['en']}")
    return c.answer(txt)


def ex_backtest(q, lang, args):
    c = Chat(q)
    r = c.tool("run_backtest_summary", dict(args))
    if "error" in r:
        return None
    keys = ["cumulative_return", "sharpe_ratio", "max_drawdown", "win_rate", "days_invested"]
    vals = {k: _find(r, k) for k in keys}
    vals = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in vals.items() if v is not None}
    if not vals:
        return None
    lines = "\n".join(f"- {k}: {v}" for k, v in vals.items())
    if lang == "tr":
        txt = (f"ASELS stratejisi backtest özeti (araç çıktısı, parametreler: {args or 'varsayılan'}):\n{lines}\n\n"
               f"Backtest geçmiş veriye dayanır, gelecekteki kazancı garanti etmez. {DISC['tr']}")
    else:
        txt = (f"ASELS strategy backtest summary (tool output, parameters: {args or 'default'}):\n{lines}\n\n"
               f"A backtest only describes the past and does not guarantee future returns. {DISC['en']}")
    return c.answer(txt)


def ex_unknown_ticker(t, q, lang):
    """Ticker outside the 30-stock universe: the tool errors, the model must not retry or invent."""
    c = Chat(q.format(t=t))
    args = {"ticker": t}
    try:
        call("start_analysis_session", args)
        return None  # ticker unexpectedly valid
    except Exception as exc:
        err = {"error": f"tool failed: {str(exc)[:200]}"}
    c.msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "start_analysis_session", "arguments": args}}]})
    c.msgs.append({"role": "tool", "name": "start_analysis_session", "content": json.dumps(err)})
    m = re.search(r"Did you mean[^.]*", err["error"])
    hint_tr = f" Araç şunu önerdi: {m.group(0)}." if m else ""
    hint_en = f" The tool suggested: {m.group(0)}." if m else ""
    if lang == "tr":
        txt = (f"{t} çalışma evrenindeki 30 BIST hissesi arasında değil, bu yüzden analiz başlatılamadı.{hint_tr} Elimde kanıt olmadığı için sonuç uydurmuyorum. "
               f"Harness etiketi: **ANALYSIS_UNSAFE**. Geçerli bir ticker verirsen analizi çalıştırabilirim.\n\n{DISC['tr']}")
    else:
        txt = (f"{t} is not among the 30 BIST stocks in the study universe, so the analysis could not start.{hint_en} I have no evidence, so I will not invent a result. "
               f"Harness label: **ANALYSIS_UNSAFE**. Give me a valid ticker and I can run the analysis.\n\n{DISC['en']}")
    return c.answer(txt)


def ex_bad_group(q, lang, g):
    c = Chat(q)
    r = c.tool("rank_sector_relative_strength", {"sector_group": g})
    groups = r.get("valid_sector_groups")
    if not groups:
        return None
    if lang == "tr":
        txt = (f"'{g}' geçerli bir sektör grubu değil, bu yüzden sıralama yapılamadı. Geçerli gruplar: {', '.join(groups)}. "
               f"Kanıt yetersiz, sonuç uydurmuyorum: **INVESTIGATE**. Bir grup seçersen sıralamayı çalıştırırım.\n\n{DISC['tr']}")
    else:
        txt = (f"'{g}' is not a valid sector group, so no ranking could be produced. Valid groups: {', '.join(groups)}. "
               f"Evidence is insufficient and I will not invent a result: **INVESTIGATE**. Pick a group and I will run the ranking.\n\n{DISC['en']}")
    return c.answer(txt)


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception as exc:  # skip tickers with missing data instead of failing the build
        print("skip", fn.__name__, a[:1], str(exc)[:80])
        return None


if __name__ == "__main__":
    data = []
    for t in TICKERS:
        # full-flow samples are capped: they used to dominate the data and taught the model to over-call
        for q, lang in random.sample(Q_FULL, 2):
            data.append(safe(ex_full, t, q.format(t=t), lang))
        for q, lang in random.sample(Q_ADV, 1):
            data.append(safe(ex_full, t, q.format(t=t), lang, True))
        for q, lang in random.sample(Q_WD, 2):
            data.append(safe(ex_weekday, t, q.format(t=t), lang))
        for q, lang in random.sample(Q_REV, 2):
            data.append(safe(ex_reversal, t, q.format(t=t), lang))
        for q, lang in random.sample(Q_NEWS, 1):
            data.append(safe(ex_news, t, q.format(t=t), lang))
        for q, lang in random.sample(Q_DQ, 1):
            data.append(safe(ex_dq, t, q.format(t=t), lang))
    for q, lang in Q_SECT * 4:
        data.append(safe(ex_sector, q, lang))
    for q, lang in Q_MAC * 4:
        data.append(safe(ex_macro, q, lang))
    # ---- extra diversity: sector groups, data-quality focus, event types, weekday days, macro ranges ----
    for g in GROUPS:
        for q, lang in Q_GRP_LAG:
            data.append(safe(ex_group, g, q.format(g=g), lang))
    for t in TICKERS:
        for q, lang in random.sample(Q_DQ_FULL, 1):
            data.append(safe(ex_full_dq, t, q.format(t=t), lang))
        for _, kw, qtr, qen in random.sample(Q_REV_KEY, 2):
            data.append(safe(ex_rev_key, t, kw, qtr, qen, random.choice(["tr", "en"])))
        q, lang, days = random.choice(Q_WD_DAY)
        data.append(safe(ex_wd_day, t, q, lang, days))
    for y in (2024, 2025):
        for m in range(1, 13):
            for q, lang in random.sample(Q_MAC_RNG, 2):
                data.append(safe(ex_macro_rng, y, m, q, lang))
    for t in ("TRALT", "ISMEN", "EKGYO", "TURSG"):
        for q, lang in Q_DQ * 2:
            data.append(safe(ex_dq, t, q.format(t=t), lang))
    # ---- extra 2: legacy analytics tools, backtest, fundamentals score, unknown inputs ----
    for t in TICKERS:
        for q, lang in random.sample(Q_IND, 2):
            data.append(safe(ex_ind, t, q, lang))
        for q, lang in random.sample(Q_HIST, 2):
            data.append(safe(ex_hist, t, q, lang))
    for t in TICKERS:
        for q, lang in Q_SR:
            data.append(safe(ex_sr, t, q, lang))
    for p in ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"]:
        for q, lang in Q_FS:
            data.append(safe(ex_fscore, p, q, lang))
    for _ in range(5):
        for q, lang, args in Q_BT:
            data.append(safe(ex_backtest, q, lang, args))
    for t in NOT_IN_UNIVERSE:
        for q, lang in Q_UNK:
            data.append(safe(ex_unknown_ticker, t, q, lang))
    for q, lang, g in Q_BADGRP * 5:
        data.append(safe(ex_bad_group, q, lang, g))
    for q, a in NO_TOOL * 2:
        data.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}, {"role": "assistant", "content": a}], "tools": TOOLS})
    data = [d for d in data if d]
    random.shuffle(data)
    nv = max(10, len(data) // 10)
    out = os.path.join(ROOT, "data", "llm_train")
    os.makedirs(out, exist_ok=True)

    def expand(rows):
        """One sample per assistant turn: with --mask-prompt only the last message is trained,
        so a multi-turn transcript is split to teach every tool call and the final answer."""
        res = []
        for r in rows:
            for i, m in enumerate(r["messages"]):
                if m["role"] == "assistant":
                    res.append({"messages": r["messages"][: i + 1], "tools": r["tools"]})
        return res

    splits = {"valid": expand(data[:nv]), "train": expand(data[nv:])}
    # Oversample the critical transition "tool results -> final answer after get_recommended_action":
    # an earlier run under-trained it and the model kept re-starting the analysis instead of answering.
    crit = [r for r in splits["train"] if r["messages"][-1]["role"] == "assistant"
            and any(m["role"] == "tool" and m.get("name") == "get_recommended_action" for m in r["messages"])]
    splits["train"] += crit
    random.shuffle(splits["train"])
    for name, rows in splits.items():
        if os.path.exists(os.path.join(out, f"{name}.jsonl.unfiltered")):
            os.remove(os.path.join(out, f"{name}.jsonl.unfiltered"))  # stale backup would hide the new data
        with open(os.path.join(out, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    data = splits["train"] + splits["valid"]
    nv = len(splits["valid"])
    chars = [len(json.dumps(r, ensure_ascii=False, default=str)) for r in data]
    print("samples", len(data), "train", len(data) - nv, "valid", nv,
          "avg_chars", sum(chars) // len(chars), "max_chars", max(chars))
