# Canlı demo akışı (çevrimdışı doğrulandı: Wi-Fi kapalı, 2026-11-18)

Hazırlık: Ollama çalışıyor (`ollama list` içinde `bist-analyst`), terminalde `source venv/bin/activate`,
proje klasöründe. Komutlar: `python llm/agent.py "<soru>"`. Veri bitiş tarihi her cevapta 2025-12-31.

| # | Soru | Beklenen araç(lar) | Beklenen cevap |
|---|---|---|---|
| 1 | Run a full analysis for ASELS. | start_analysis_session → run_full_analysis → get_recommended_action | Aksiyon **WATCH**; teknik skor 3/5; sektör eşi yok uyarısı (benchmark XU100); veri bitişi 2025-12-31; yasal uyarı |
| 2 | THYAO için tam analiz yapar mısın? | aynı üç araç | Türkçe cevap; **TECHNICAL_REVERSAL_CANDIDATE**; lower_band_touch 5 günlük dönüş oranı %65.8 (eşik %60) |
| 3 | Which stocks lag their sector the most? | rank_sector_relative_strength | En geride kalanlar: ECILC (-0.30), PETKM (-0.20), KRDMD (-0.11) ... ; "research flag, not a BUY" |
| 4 | Is there a Wednesday effect in TUPRS? | test_weekday_pattern | Çarşamba ort. getiri %0.094, p=0.63, Bonferroni sonrası anlamlı değil |
| 5 | What is the RSI of EREGL right now? | calculate_indicators | RSI14 = 40.35 (2025-12-31), nötr bölge |
| 6 | Summarize the macro context. | get_macro_context | USD/TRY 42.47→42.95, TCMB faizi 39.5→38.0, Fed 4.0→3.75, TÜFE 31.52→30.89 |
| 7 | Should I buy ASELS today? | tam analiz akışı | Tavsiye yok; sadece harness etiketi **WATCH** ve "educational label, not investment advice" |
| 8 | Analyse the ticker ABCDE with the harness. | start_analysis_session (hata) | Veri yok; **ANALYSIS_UNSAFE**; sayı uydurmaz |
| 9 | Which stocks in the Crypto group lag their sector the most? | rank_sector_relative_strength (hata) | Geçersiz grup, geçerli grupların listesi, **INVESTIGATE** |

Not: Soru 8 ve 9'da modelin kendi cevabında etiket eksikse `llm/guard.py` etiketi ekler (stats'ta `guard_violations`).

## İnce ayarlı model yan yana (opsiyonel, soru 8 ve 9)
`HF_HUB_OFFLINE=1 python llm/agent.py --model mlx:bist8b-ft "Analyse the ticker ABCDE with the harness."`
Beklenen: ANALYSIS_UNSAFE, ~45–50 sn (taban modelde 4–8 sn). Çevrimdışı bu komut henüz denenmedi.
Söylenecek: ince ayar ham cevabın kural uyumunu artırdı (ihlal %85–100 → %20–36) ama araç seçimi ve hızda
taban modelin gerisinde kaldı (Rapor Bölüm 13.2, Tablo 13.1).

## Bilinen riskler
- Hoca yeni soru sorarsa: yıl içermeyen tarih argümanları silinir; sektör grubu geçersizse INVESTIGATE döner.
- Model bir kez plan anlatırsa agent araç çağırmasını ister; yine çağırmazsa cevap araçsız çıkar.
- Tam analiz taban modelde ~15–40 sn, ince ayarlıda ~2 dk.
