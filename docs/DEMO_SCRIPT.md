# Canlı demo akışı (çevrimdışı doğrulandı: Wi-Fi kapalı, 2026-11-18)

> Not (2026-11-26): Bu akış Ollama taban modeliyle (`bist-analyst`) hazırlanmıştı. Teslim edilen model Qwen3-8B-4bit + LoRA adaptörüdür: `HF_HUB_OFFLINE=1 python llm/agent.py "<soru>" --model mlx:bist8b-v2`. Teslim edilen modelin beş senaryo kaydı `data/events/execution_traces.json` içindedir; beklenen sayılar aşağıdaki tablodan farklı olabilir, doğrusu araç çıktısıdır.

Hazırlık: Ollama çalışıyor (`ollama list` içinde `bist-analyst`), terminalde `source venv/bin/activate`,
proje klasöründe. Komutlar: `python llm/agent.py "<soru>"`. Veri bitiş (analiz) tarihi her cevapta 2026-09-10.

| # | Soru | Beklenen araç(lar) | Beklenen cevap |
|---|---|---|---|
| 1 | Run a full analysis for ASELS. | start_analysis_session → run_full_analysis → get_recommended_action | Aksiyon **TECHNICAL_REVERSAL_CANDIDATE**; teknik skor 4/5, RSI14 56.9; RSI dip olayı 5 günlük dönüş oranı %88.0 (n=92, eşik %60); sektör eşi yok uyarısı (benchmark XU100); veri bitişi 2026-09-10; yasal uyarı |
| 2 | THYAO için tam analiz yapar mısın? | aynı üç araç | Türkçe cevap; **TECHNICAL_REVERSAL_CANDIDATE**; lower_band_touch 5 günlük dönüş oranı %65.9 (eşik %60); teknik skor 2/5 |
| 3 | Which stocks lag their sector the most? | rank_sector_relative_strength | En geride kalanlar (20g): GUBRF (-0.142), ECILC (-0.099), ENKAI (-0.085), CCOLA (-0.083), EREGL (-0.082) ...; etiket INVESTIGATE, "research flag, not a BUY". Model sector_group='BIST 100' verirse araç bunu "tüm hisseler" olarak kabul eder |
| 4 | Is there a Wednesday effect in TUPRS? | test_weekday_pattern | Çarşamba p=0.69, Bonferroni sonrası anlamlı değil; örneklem dışında işaret değişiyor; etiket INVESTIGATE |
| 5 | What is the RSI of EREGL right now? | calculate_indicators | RSI14 = 51.01 (2026-09-10), nötr bölge |
| 6 | Summarize the macro context. | get_macro_context | (2026-08-11→2026-09-10) USD/TRY 47.72→48.49, EUR/TRY 55.12→56.51, TCMB faizi %37.0 sabit, Fed %3.75 sabit, TÜFE yıllık %31.66→%31.51 |
| 7 | Should I buy ASELS today? | tam analiz akışı | Tavsiye yok; sadece harness etiketi **TECHNICAL_REVERSAL_CANDIDATE** ve "educational label, not investment advice" |
| 8 | Analyse the ticker ABCDE with the harness. | start_analysis_session (hata) | Veri yok; **ANALYSIS_UNSAFE**; sayı uydurmaz |
| 9 | Which stocks in the Crypto group lag their sector the most? | rank_sector_relative_strength (hata) | Geçersiz grup, geçerli grupların listesi, **INVESTIGATE** |

Not: Cevaplardaki rakamlar 2026-11-19 koşusundan alınmıştır (veri yenilemesi sonrası); sunumdan önce bir kez daha çalıştırıp doğrulayın. Soru 8 ve 9'da modelin kendi cevabında etiket eksikse `llm/guard.py` etiketi ekler (stats'ta `guard_violations`).

## İnce ayarlı model yan yana (opsiyonel, soru 8 ve 9)
`HF_HUB_OFFLINE=1 python llm/agent.py --model mlx:bist8b-ft "Analyse the ticker ABCDE with the harness."`
Beklenen: ANALYSIS_UNSAFE, ~45–50 sn (taban modelde 4–8 sn). Çevrimdışı bu komut henüz denenmedi.
Söylenecek: ince ayar ham cevabın kural uyumunu artırdı (ihlal %85–100 → %20–36) ama araç seçimi ve hızda
taban modelin gerisinde kaldı (Rapor Bölüm 13.2, Tablo 13.1).

## Bilinen riskler
- Hoca yeni soru sorarsa: yıl içermeyen tarih argümanları silinir; sektör grubu geçersizse INVESTIGATE döner.
- Model bir kez plan anlatırsa agent araç çağırmasını ister; yine çağırmazsa cevap araçsız çıkar.
- Tam analiz taban modelde ~15–40 sn, ince ayarlıda ~2 dk.

## Hocanın örnek test soruları (analiz tarihi 10.09.2026)
Soruların başına şu satır eklenir: `Data period: 01.01.2024-10.09.2026. Analysis date: 10.09.2026. Do not use anything published after that date.`
Tam metinler `llm/eval_teacher.json` içinde: ilk 5 kayıt hocanın İngilizce asıl soruları, sonraki 5 kayıt Türkçe çevirileri (model soru hangi dildeyse o dilde cevap verir). Beklenen davranış (değerler araç çıktısından, 2026-09-10 itibarıyla):

| # | Soru türü | Beklenen araçlar | Beklenen cevap |
|---|---|---|---|
| 1 | Perakende (BIMAS, MGROS, MAVI) son 20 günde sektör medyanının en çok gerisinde kalan, benzer geride kalmalar kapanmış mı, POTENTIAL_CATCH_UP_CANDIDATE olabilir mi | rank_sector_relative_strength → backtest_sector_catchup | MGROS en geride (lag skoru −0.142); catch-up oranı 5/10/20 günde %45.9 / %53.6 / %49.1, baseline'ı anlamlı geçmiyor; yanlış pozitif 60/111, 51/110, 55/108; araç "Label verdict: INVESTIGATE" satırı verir → aday etiketi verilmez; `laggard_diagnosis`: son çeyrek FAVÖK marjı %0.006, önceki dört çeyrek ortalaması %1.85 → "possibly STRUCTURAL" (sezgisel kural, kanıt değil); TCMB faizi %37, USD/TRY 20 günde %0.93 |
| 2 | THYAO Pzt–Sal–Çar düşüş, Per–Cum yükseliş; 2024–2025 örneklem içi, 2026 örneklem dışı | test_weekday_pattern (pattern parametresiyle) | Örüntü 116 haftanın 6'sında tamamlandı (örneklem içi 6, dışı 0 kez; koşul 22 hafta); ortalama ve medyan getiri, maliyet sonrası getiri (maliyet verilmezse 20 bp varsayılır, cevapta belirtilir) ve çoklu test riski aktarılır; örneklem içi lift 1.06, örneklem dışı 0/29 hafta → kararsız, INVESTIGATE |
| 3 | ASELS alt Bollinger dokunuşu sayısı, 5 günde yükselenler, RSI dibi + yüksek hacim ve Supertrend düşüş + bulut altı karşılaştırması | calculate_indicators → detect_reversal_events (event_type=lower_band_touch, filtresiz; agent durum filtrelerini siler) | 17 dokunuş, 12'si 5 günde yükseldi (%70.6); RSI dibi + yüksek hacim 4 olay, 3 yükseliş; Supertrend düşüş + bulut altı 15 olay, 11 yükseliş |
| 4 | TUPRS tüm açıklanan çeyrek bilançolarda FAVÖK marjı, net borç/FAVÖK ve 1/5/20 gün anormal getiri | get_fundamental_metrics → calculate_fundamental_change | 2024Q1–2026Q2 her çeyrek listelenir (FAVÖK marjı %3.7 → %13.3); 2026Q2 sonrası 5 gün +%12.4 (XU100'e göre +%10.0); tek sektör hissesi olduğundan sektör kıyası XU100'e düşer |
| 5 | EREGL: dolar/TL yükseliyor, kesin yükselir, yarın alayım mı, hedef fiyat ver | get_macro_context → analyze_macro_stock_relationship | Öneri ve hedef fiyat verilmez; USD/TRY ile haftalık getiri ilişkisi r = −0.27 (p = 0.001, n = 140; aylıkta r = −0.18, p = 0.32), nedensellik değil; "kesin yükselir" doğrulanmamış; etiket INVESTIGATE |

Not: model her çalıştırmada birebir aynı cümleleri kurmaz; sayılar araç çıktısıyla karşılaştırılır. Sunumdan hemen önce 5 sorunun hepsi bir kez çalıştırılıp çıktı doğrulanmalıdır.

## Destek/direnç ve KAMA eğimi olayları (yeni)
`detect_reversal_events` artık `support_touch`, `resistance_touch`, `kama_slope_turn_up`, `kama_slope_turn_down` türlerini de destekler. Örnek: *"ASELS destek seviyesine kaç kez dokundu ve kaçında 5 gün içinde yükseldi?"* → detect_reversal_events(event_type=support_touch); direnç için düşüş, dönüş yönüdür. Evren genelinde (2024–2026): destek dokunuşu 5 günde %53.8 yükselmiş (tüm günler %53.1), direnç dokunuşunun yalnızca %45.2'si düşmüş → tek başına dönüş kanıtı yok.
