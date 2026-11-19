# Canlı demo akışı (çevrimdışı doğrulandı: Wi-Fi kapalı, 2026-11-18)

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
