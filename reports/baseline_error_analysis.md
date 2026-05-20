# Baseline Error Analysis

## Ozet

- Dataset: `E:\rent-agent\dataset\train_ready_ml.parquet`
- Yuklenen model: `E:\rent-agent\models\baseline_model.joblib`
- Yuklenen preprocessor: `E:\rent-agent\models\baseline_preprocessor.joblib`
- Test ornek sayisi: **975**

| Metric | Value |
| --- | --- |
| MAE | 4,683.35 |
| RMSE | 6,911.91 |
| R2 | 0.7632 |
| MAPE (%) | 14.85 |

## Test Set Error Breakdowns

### District bazli MAE / MAPE

| district | sample_count | mean_actual_price | mae | mape |
| --- | --- | --- | --- | --- |
| Beypazarı | 1 | 10,000.00 | 20,156.40 | 201.56 |
| Çankaya | 343 | 38,966.32 | 6,247.77 | 17.04 |
| Yenimahalle | 94 | 36,671.28 | 5,313.07 | 15.35 |
| Keçiören | 143 | 27,497.90 | 3,975.46 | 14.73 |
| Etimesgut | 104 | 32,560.96 | 3,906.19 | 11.63 |
| Polatlı | 18 | 20,319.44 | 3,551.89 | 19.42 |
| Sincan | 61 | 25,117.18 | 3,384.38 | 13.67 |
| Gölbaşı | 55 | 33,482.73 | 3,296.18 | 10.42 |
| Altındağ | 30 | 28,875.00 | 3,268.56 | 11.26 |
| Mamak | 101 | 25,269.79 | 3,150.82 | 12.55 |
| Akyurt | 1 | 29,000.00 | 3,074.15 | 10.60 |
| Pursaklar | 8 | 27,718.75 | 2,543.54 | 9.88 |
| Çubuk | 16 | 17,734.38 | 2,348.95 | 13.90 |

### Fiyat araligi bazli MAE / MAPE

| price_range | sample_count | mean_actual_price | mae | mape |
| --- | --- | --- | --- | --- |
| 0-20k TRY | 96 | 17,174.47 | 4,525.79 | 27.98 |
| 20k-30k TRY | 407 | 24,495.42 | 3,143.95 | 13.02 |
| 30k-40k TRY | 255 | 33,843.92 | 4,286.48 | 12.63 |
| 40k-50k TRY | 112 | 43,178.12 | 6,095.08 | 14.19 |
| 50k-75k TRY | 80 | 57,375.00 | 9,568.76 | 16.63 |
| 75k+ TRY | 25 | 86,360.00 | 12,440.24 | 14.20 |

### m2 araligi bazli MAE / MAPE

| m2_range | sample_count | mean_actual_price | mae | mape |
| --- | --- | --- | --- | --- |
| 0-75 m2 | 151 | 28,956.62 | 3,918.44 | 14.86 |
| 75-100 m2 | 144 | 29,077.43 | 3,954.86 | 13.77 |
| 100-125 m2 | 302 | 28,048.47 | 3,768.64 | 13.94 |
| 125-150 m2 | 229 | 31,427.72 | 4,407.22 | 14.31 |
| 150-200 m2 | 92 | 45,140.76 | 6,897.04 | 17.30 |
| 200+ m2 | 57 | 60,585.96 | 10,932.90 | 20.65 |

### Rooms bazli MAE / MAPE

| rooms_group | sample_count | mean_actual_price | mae | mape |
| --- | --- | --- | --- | --- |
| 6+ rooms | 29 | 59,186.21 | 12,167.72 | 24.13 |
| 5 rooms | 86 | 50,359.30 | 8,338.55 | 19.27 |
| 1 rooms | 12 | 24,062.50 | 5,863.43 | 29.39 |
| Unknown | 2 | 40,000.00 | 5,492.18 | 13.73 |
| 2 rooms | 147 | 30,446.94 | 4,140.32 | 14.20 |
| 3 rooms | 230 | 29,729.99 | 4,027.99 | 14.08 |
| 4 rooms | 469 | 30,072.25 | 4,008.28 | 13.68 |

### En yuksek 30 hata yapan ilan

| listing_id | district | neighborhood | rooms | m2_gross | actual_price_try | predicted_price_try | residual | abs_error | ape_pct | title_short |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hepsiemlak:124381-850 | Çankaya | Sancak Mah. | 4+1 | 380.00 | 56,000.00 | 110,334.68 | 54,334.68 | 54,334.68 | 97.03 | ÇANKAYA YILDIZ T.GÜNEŞ BULVARI ÜZERİNDE 4+1 KİRALIK ARAKAT DAİRE |
| hepsiemlak:149516-72 | Çankaya | Gaziosmanpaşa Mah. | 2+1 | 100.00 | 99,000.00 | 58,324.64 | -40,675.36 | 40,675.36 | 41.09 | FİLİSTİN CADDESİN'DE ELÇİLİKLERE KOMŞU PRESTİJLİ KİRALIK DUBLEKS |
| hepsiemlak:11098-2007 | Çankaya | Çankaya Mah. | 5+1 | 350.00 | 75,000.00 | 113,663.31 | 38,663.31 | 38,663.31 | 51.55 | Sancakta Kiralık Dubleks |
| hepsiemlak:54420-2303 | Çankaya | Bahçelievler Mah. | 4+1 | 160.00 | 16,500.00 | 54,931.50 | 38,431.50 | 38,431.50 | 232.92 | BAHÇELİEVLER'DE BEŞEVLER METROSU VE BAŞKENT HAST.YAKINI 4+1 KATTA DAİREDE HERŞEY DAHİL KİR |
| hepsiemlak:56844-1166 | Çankaya | Bahçelievler Mah. | 1+1 | 55.00 | 70,000.00 | 40,829.10 | -29,170.90 | 29,170.90 | 41.67 | BAHÇELİEVLER 7.CADDE YAKINI 0 BİNADA 0 SIFIR EŞYALI TAM ARAKAT BALKONLU ULTRA LÜKS 1+1 |
| hepsiemlak:140580-2726 | Yenimahalle | İnönü Mah. | 2+1 | 140.00 | 85,000.00 | 56,457.41 | -28,542.59 | 28,542.59 | 33.58 | Velux Ankara'da Büyük Tip Kiralık 2+1 Geniş Teraslı Daire/Ofis |
| hepsiemlak:136808-317 | Etimesgut | Göksu Mah. | 4+1 | 196.00 | 95,000.00 | 68,104.93 | -26,895.07 | 26,895.07 | 28.31 | KAŞMİR MAVİ ORKİDE GÖL CEPHE JAKUZİ KLİMA AVİZE PERDE ULTRA LÜKS |
| hepsiemlak:58419-4096 | Çankaya | Bağcılar Mah. | 8+1 | 280.00 | 42,000.00 | 68,617.59 | 26,617.59 | 26,617.59 | 63.38 | ÇANKAYA AÇIN CADDESİ YAPILI GİRİŞ KAT TERS DUBLEKS 8+1 DAİRE |
| hepsiemlak:156378-48 | Çankaya | Bağcılar Mah. | 4+1 | 200.00 | 65,000.00 | 91,211.25 | 26,211.25 | 26,211.25 | 40.32 | GOP ANKA PİA TOWERS B BLOK 17.KATTA GÜNEY CEPHE MUHTEŞEM DAİRE |
| hepsiemlak:123623-3557 | Çankaya | Alacaatlı Mah. | 4+1 | 230.00 | 68,000.00 | 93,646.58 | 25,646.58 | 25,646.58 | 37.72 | ÇAYYOLU ALACAATLI İNCEK İCON'DA GENİŞ BALKONLU KİRALIK 4+1 DAİRE |
| hepsiemlak:144844-123 | Yenimahalle | Gazi Mah. | 4+2 | 195.00 | 65,000.00 | 39,832.53 | -25,167.47 | 25,167.47 | 38.72 | GAZİ MH SİLAHTAR CADDESİ ÜZERİ TERASLI KATTA DUBLEX KİRALIK 4+2 |
| hepsiemlak:3966-4018 | Çankaya | Aziziye Mah. | 4+1 | 260.00 | 55,000.00 | 79,737.61 | 24,737.61 | 24,737.61 | 44.98 | ATAKULE YAKINI HAVA SOKAK 4+1+2 BANYO DUBLEKS 250M2 BAKIMLI |
| hepsiemlak:4937-5881 | Çankaya | Akpınar Mah. | 8+1 | 320.00 | 66,000.00 | 90,392.88 | 24,392.88 | 24,392.88 | 36.96 | ETS'DEN AÇIK CEPHELİ 8+1 TERASLI DUBLEX DAİRE |
| hepsiemlak:80912-450 | Çankaya | Aşıkpaşa Mah. | 2+1 | 75.00 | 33,250.00 | 56,483.93 | 23,233.93 | 23,233.93 | 69.88 | ÇANKAYA AŞIKPAŞA MAH. EŞYALI KİRALIK SIFIR DAİRE |
| hepsiemlak:1955-8175 | Mamak | Cengizhan Mah. | 3+1 | 168.00 | 53,000.00 | 75,817.00 | 22,817.00 | 22,817.00 | 43.05 | Akadia Modern Konutları'nda \| Sıfır \| 3+1 Lüks Daire \| 149m² Net |
| hepsiemlak:3420-10089 | Çankaya | Yukarı Dikmen Mah. | 5+2 | 450.00 | 105,000.00 | 127,348.72 | 22,348.72 | 22,348.72 | 21.28 | ÇANKAYA ORAN'DA ÖZEL MİMARİLİ ULTRA LÜKS MOBİLYALI 450m² 5+2 |
| hepsiemlak:64413-5078 | Çankaya | Ayrancı Mah. | 3+1 | 130.00 | 63,000.00 | 41,847.66 | -21,152.34 | 21,152.34 | 33.58 | AND EMLAKTAN YAYLAGÜL SK 3+1 MOBİLYALI MANZARALI ASANSÖRLÜ OTOPARKLI |
| hepsiemlak:118256-398 | Polatlı | Karapınar Mah. | 5+2 | 200.00 | 20,000.00 | 41,146.01 | 21,146.01 | 21,146.01 | 105.73 | Polatlı Karapınar mahallesi Kralik bina |
| hepsiemlak:85433-905 | Yenimahalle | Yakacık Mah. | 3+1 | 135.00 | 23,000.00 | 43,507.04 | 20,507.04 | 20,507.04 | 89.16 | KUZEY EMLAKTAN YAKACIK TOKİDE 3+1 KİRALIK DAİRE |
| hepsiemlak:102968-3947 | Yenimahalle | Beştepe Mah. | 5+1 | 280.00 | 62,000.00 | 41,698.93 | -20,301.07 | 20,301.07 | 32.74 | Beştepe Mah. Emek Metrosuna 250m Kiralık Teraslı 5+1 /ENBATI A.Ş |
| hepsiemlak:158018-41 | Beypazarı | Ayvaşık Mah. | 2+1 | 100.00 | 10,000.00 | 30,156.40 | 20,156.40 | 20,156.40 | 201.56 | BEYPAZARI TOKİ KONUTLARINDA SIFIR KİRALIK DAİRELER |
| hepsiemlak:1990-847 | Çankaya | Akpınar Mah. | 6+2 | 360.00 | 85,000.00 | 104,954.95 | 19,954.95 | 19,954.95 | 23.48 | DİKMEN CAD. ENDER EVLER£DE İÇİ YAPILI, KEYİFLİ KİRALIK DUBLEKS |
| hepsiemlak:158237-122 | Çankaya | Oran Mah. | 3+1 | 120.00 | 37,500.00 | 56,969.20 | 19,469.20 | 19,469.20 | 51.92 | ORAN ATATÜRK SİTESİ 3+1 KİRALIK DAİRE |
| hepsiemlak:37602-2791 | Etimesgut | Aşağıyurtçu Mah. | 4+1 | 180.00 | 42,000.00 | 61,067.11 | 19,067.11 | 19,067.11 | 45.40 | LÜKS SİTEDE KİRALIK 4+1 HAVUZ-HAMAM-SAUNA-FITNESS-GÜVENLİK |
| hepsiemlak:22003-2178 | Çankaya | Murat Mah. | 4+1 | 250.00 | 40,000.00 | 57,828.70 | 17,828.70 | 17,828.70 | 44.57 | Gezegen'de, Asansörlü, Otoparklı, Muhteşem Manzaralı,4+1+2 Kiralık Dubleks |
| hepsiemlak:167435-28 | Çankaya | Cebeci Mah. | 1+0 | 55.00 | 20,000.00 | 37,814.59 | 17,814.59 | 17,814.59 | 89.07 | CEBECİDE KİRALIK 1+0 YENİ BİNADA KİRALIK EŞYALI DAİRE |
| hepsiemlak:128426-61 | Çankaya | Prof. Dr. Ahmet Taner Kışlalı Mah. | 4+1 | 180.00 | 55,000.00 | 72,621.08 | 17,621.08 | 17,621.08 | 32.04 | ALACAATLI CAD.PAMUKKALE SİT.DE BOŞ,BOYALI,TEMİZ,KİRALIK4+1 DAİRE |
| hepsiemlak:144404-339 | Sincan | Alcı Mah. | 3+1 | 145.00 | 18,000.00 | 35,254.88 | 17,254.88 | 17,254.88 | 95.86 | AC GAYRİMENKULDEN 3+1 LÜKS KİRALIK DAİRE |
| hepsiemlak:4581-7769 | Etimesgut | Alsancak Mah. | 4+1 | 180.00 | 32,000.00 | 49,184.22 | 17,184.22 | 17,184.22 | 53.70 | Toki Turkuaz Konutları'nda, Merkezi Konumda, Manzaralı, 4+1 |
| hepsiemlak:69026-484 | Keçiören | Şenyuva Mah. | 4+1 | 165.00 | 23,000.00 | 39,875.42 | 16,875.42 | 16,875.42 | 73.37 | ARIKANDAN KUZEY ANKARA 6.ETAP 4+1 MANZARALI KİRALIK DAİRE |

## Uretilen Plotlar

- Actual vs predicted scatter: `E:\rent-agent\reports\plots\actual_vs_predicted.png`
- Residual histogram: `E:\rent-agent\reports\plots\residual_histogram.png`
- District MAE bar chart: `E:\rent-agent\reports\plots\district_error_bar.png`

## Text Branch Hazirlik Analizi

### Description uzunluk ozeti

| metric | value |
| --- | --- |
| count | 6,498.00 |
| blank_count | 66.00 |
| blank_ratio_pct | 1.02 |
| char_mean | 687.68 |
| char_median | 521.00 |
| char_p90 | 1,368.00 |
| char_p95 | 1,768.00 |
| char_max | 10,434.00 |
| word_mean | 94.75 |
| word_median | 71.00 |
| word_p90 | 189.00 |
| word_p95 | 246.15 |
| word_max | 1,593.00 |

### Description uzunluk dagilimi (character bins)

| length_range | sample_count | share_pct |
| --- | --- | --- |
| 0-99 | 510 | 7.85 |
| 100-299 | 1474 | 22.68 |
| 300-599 | 1624 | 24.99 |
| 600-999 | 1597 | 24.58 |
| 1000-1999 | 1053 | 16.20 |
| 2000+ | 240 | 3.69 |

### Title uzunluk ozeti

| metric | value |
| --- | --- |
| count | 6,498.00 |
| blank_count | 0.00 |
| blank_ratio_pct | 0.00 |
| char_mean | 60.19 |
| char_median | 60.00 |
| char_p90 | 81.30 |
| char_p95 | 92.15 |
| char_max | 100.00 |
| word_mean | 8.77 |
| word_median | 9.00 |
| word_p90 | 12.00 |
| word_p95 | 13.00 |
| word_max | 19.00 |

### Title uzunluk dagilimi (character bins)

| length_range | sample_count | share_pct |
| --- | --- | --- |
| 0-39 | 527 | 8.11 |
| 40-59 | 2534 | 39.00 |
| 60-79 | 2724 | 41.92 |
| 80-99 | 654 | 10.06 |
| 100+ | 59 | 0.91 |

### Bos description var mi?

- Bos description sayisi: **66**
- Bos description orani: **1.02%**

### En sik gecen title kelimeleri

| token | count |
| --- | --- |
| daire | 3706 |
| kiralik | 3356 |
| kat | 1038 |
| esyali | 860 |
| katta | 762 |
| yapili | 740 |
| yakini | 628 |
| kiralk | 577 |
| cephe | 573 |
| dan | 569 |
| full | 548 |
| mah | 547 |
| on | 523 |
| kombili | 523 |
| ara | 515 |
| emlak | 509 |
| tan | 490 |
| manzarali | 408 |
| asansorlu | 393 |
| cadde | 391 |

### En sik gecen description kelimeleri

| token | count |
| --- | --- |
| daire | 5806 |
| telefonu | 3916 |
| goster | 3916 |
| dairemiz | 3601 |
| merkezi | 3450 |
| yurume | 3176 |
| m2 | 2873 |
| emlak | 2775 |
| ankara | 2767 |
| bilgi | 2749 |
| kat | 2717 |
| gayrimenkul | 2704 |
| genis | 2630 |
| yasam | 2618 |
| alan | 2466 |
| yakin | 2064 |
| otopark | 2023 |
| cephe | 1969 |
| mutfak | 1967 |
| ozellikleri | 1927 |

### Text temizleme gerekip gerekmedigi

| signal | count | share_pct |
| --- | --- | --- |
| blank_description | 66 | 1.02 |
| title_upper_heavy | 5490 | 84.49 |
| description_upper_heavy | 4007 | 61.67 |
| html_like_description | 3 | 0.05 |
| url_like_description | 26 | 0.40 |
| phrase:telefonu goster | 2416 | 37.18 |
| phrase:detayli bilgi | 1407 | 21.65 |
| phrase:arayiniz | 798 | 12.28 |
| phrase:gayrimenkul | 1976 | 30.41 |
| phrase:kahve icmeye | 50 | 0.77 |
| phrase:tapu | 581 | 8.94 |
| phrase:kredi islemleri | 230 | 3.54 |

- Description tarafinda bos kayit var (1.02%). Text branch icin bos metinleri ayri ele almak gerekir.
- Metinlerin buyuk bolumu buyuk harf agirlikli. Lowercase normalizasyonu ve unicode-folding faydali olur.
- Iletisim ve emlak-ofisi boilerplate kaliplari yuksek. `telefonu goster`, `detayli bilgi`, ofis tanitimi gibi kaliplari temizlemek mantikli.
- HTML ve URL temizligi dusuk oncelikli; ana kazanc boilerplate ve case normalization tarafinda.

### Fiyatla iliskili olabilecek kelimeler icin basit analiz

| keyword | listing_count | share_pct | mean_price_try | median_price_try | mean_price_delta_try | median_price_delta_try |
| --- | --- | --- | --- | --- | --- | --- |
| ebeveyn | 618 | 9.51 | 45,771.36 | 42,000.00 | 11,795.34 | 12,000.00 |
| guvenlik | 890 | 13.70 | 45,083.54 | 40,000.00 | 11,107.52 | 10,000.00 |
| teras | 219 | 3.37 | 44,896.58 | 38,500.00 | 10,920.56 | 8,500.00 |
| luks | 571 | 8.79 | 43,352.01 | 38,250.00 | 9,376.00 | 8,250.00 |
| site | 932 | 14.34 | 42,538.68 | 37,000.00 | 8,562.66 | 7,000.00 |
| asansor | 722 | 11.11 | 40,603.86 | 35,000.00 | 6,627.85 | 5,000.00 |
| manzarali | 660 | 10.16 | 40,600.48 | 35,000.00 | 6,624.47 | 5,000.00 |
| ankastre | 881 | 13.56 | 40,191.56 | 36,000.00 | 6,215.55 | 6,000.00 |
| otopark | 1714 | 26.38 | 39,340.87 | 35,000.00 | 5,364.86 | 5,000.00 |
| balkon | 1405 | 21.62 | 36,308.58 | 32,000.00 | 2,332.57 | 2,000.00 |
| sifir | 532 | 8.19 | 35,575.00 | 32,000.00 | 1,598.98 | 2,000.00 |
| metro | 970 | 14.93 | 34,784.74 | 32,000.00 | 808.73 | 2,000.00 |
| esyali | 984 | 15.14 | 34,661.33 | 30,750.00 | 685.32 | 750.00 |