# Final Log-Target Fusion Results

## Ozet

- Multimodal source: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- Image embedding source: `E:\rent-agent\dataset\image_embeddings.parquet`
- Kaydedilen model bundle: `E:\rent-agent\models\final_logtarget_model.joblib`
- Join sonucu kalan ornek sayisi: **6,389**
- Original image embedding dimension: **1280**
- Reducer: **PCA**
- Reduced image_dim: **16**
- En iyi validation adayi: **XGBRegressor**

## Validation Leaderboard

| candidate | effective_n_estimators | used_early_stopping | validation_mae | validation_rmse | validation_r2 | validation_mape |
| --- | --- | --- | --- | --- | --- | --- |
| XGBRegressor | 700 | False | 4,693.0067 | 6,881.0951 | 0.8207 | 13.9153 |
| XGBRegressorEarlyStopping | 700 | True | 4,693.0067 | 6,881.0951 | 0.8207 | 13.9153 |

## Final Test Metrics

| Metric | Value |
| --- | --- |
| MAE | 4,626.18 |
| RMSE | 7,021.95 |
| R2 | 0.8033 |
| MAPE (%) | 13.32 |

## Log-Target Etkisi

- Egitim hedefi `log1p(price_try)` olarak olusturuldu; tum validation ve test skorları `expm1` sonrasi gercek TRY fiyatlari uzerinde hesaplandi. Secilen aday: **XGBRegressor**, effective_n_estimators=**700**.

## Baseline ve Onceki Fusion Karsilastirmasi

| row | mae | rmse | r2 | mape |
| --- | --- | --- | --- | --- |
| Matched tabular baseline | 4,700.0787 | 7,057.7587 | 0.8013 | 13.7867 |
| Previous 6-image reduced fusion | 4,555.7300 | 6,842.2000 | 0.8133 | 13.5300 |
| Previous all-image reduced fusion | 4,570.6500 | 6,673.1600 | 0.8224 | 13.5900 |
| Log-target final fusion | 4,626.1784 | 7,021.9462 | 0.8033 | 13.3247 |
| Improvement vs matched baseline | 73.9003 | 35.8124 | 0.0020 | 0.4620 |
| Improvement vs previous 6-image reduced | -70.4484 | -179.7462 | -0.0100 | 0.2053 |
| Improvement vs previous all-image reduced | -55.5284 | -348.7862 | -0.0191 | 0.2653 |

## High-Price Segment

- Tanim: `actual_price_try >= 75,000 TRY`
| segment | sample_count | baseline_mae | fusion_mae | baseline_rmse | fusion_rmse | baseline_r2 | fusion_r2 | baseline_mape | fusion_mape | mae_improvement | rmse_improvement | r2_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_price_75k_plus | 31 | 17,122.9349 | 18,649.0971 | 21,688.4102 | 23,643.6892 | 0.0384 | -0.1428 | 18.5068 | 19.9913 | -1,526.1622 | -1,955.2790 | -0.1812 | -1.4845 |

## Luxury Segment

- Tanim: luxury proxy = `actual_price_try >= 75,000` veya `m2_gross >= 180` veya `rooms_count >= 5`
| segment | sample_count | baseline_mae | fusion_mae | baseline_rmse | fusion_rmse | baseline_r2 | fusion_r2 | baseline_mape | fusion_mape | mae_improvement | rmse_improvement | r2_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| luxury_proxy | 116 | 9,819.0360 | 10,011.7020 | 13,772.2244 | 14,456.1293 | 0.6988 | 0.6681 | 17.1309 | 16.9263 | -192.6660 | -683.9048 | -0.0307 | 0.2047 |

## District Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Akyurt | 5 | 3,156.1552 | 2,159.2691 | 14.7552 | 11.3360 | 996.8862 | 3.4192 |
| Altındağ | 33 | 4,002.8446 | 3,692.6572 | 15.8675 | 15.5157 | 310.1874 | 0.3517 |
| Çankaya | 379 | 6,277.1725 | 6,013.0529 | 15.2434 | 14.2837 | 264.1196 | 0.9596 |
| Keçiören | 142 | 3,827.3778 | 3,657.2500 | 13.6167 | 12.7025 | 170.1278 | 0.9142 |
| Gölbaşı | 56 | 3,835.0616 | 3,709.3236 | 11.8777 | 11.0059 | 125.7380 | 0.8718 |
| Çubuk | 15 | 2,087.9365 | 2,007.6332 | 13.8520 | 13.5480 | 80.3033 | 0.3040 |
| Etimesgut | 82 | 3,632.4278 | 3,679.1814 | 11.7475 | 11.5825 | -46.7537 | 0.1650 |
| Sincan | 58 | 3,421.4291 | 3,497.8980 | 13.3003 | 13.5497 | -76.4689 | -0.2493 |
| Mamak | 75 | 2,936.1545 | 3,097.9971 | 10.8235 | 11.3060 | -161.8426 | -0.4825 |
| Yenimahalle | 90 | 4,086.4141 | 4,559.2117 | 11.4672 | 12.1894 | -472.7977 | -0.7222 |
| Polatlı | 12 | 3,214.2595 | 3,952.9536 | 16.4078 | 19.6193 | -738.6941 | -3.2116 |
| Pursaklar | 10 | 5,075.2632 | 6,113.2320 | 15.7168 | 18.4954 | -1,037.9688 | -2.7786 |

## Price-Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-20k TRY | 84 | 4,130.3212 | 4,065.0840 | 24.3314 | 24.0759 | 65.2372 | 0.2555 |
| 20k-30k TRY | 372 | 2,905.2472 | 2,638.0635 | 11.9290 | 10.7864 | 267.1837 | 1.1425 |
| 30k-40k TRY | 266 | 4,224.3421 | 4,101.1465 | 12.4604 | 12.0907 | 123.1956 | 0.3697 |
| 40k-50k TRY | 122 | 5,594.9134 | 6,005.3667 | 12.8582 | 13.8065 | -410.4533 | -0.9483 |
| 50k-75k TRY | 84 | 8,840.6085 | 8,476.1505 | 15.2758 | 14.5622 | 364.4579 | 0.7136 |
| 75k+ TRY | 31 | 17,122.9349 | 18,649.0971 | 18.5068 | 19.9913 | -1,526.1622 | -1.4845 |

## m2-Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-75 m2 | 152 | 3,504.8093 | 3,643.2014 | 12.3153 | 12.5753 | -138.3922 | -0.2600 |
| 75-100 m2 | 150 | 4,117.2768 | 4,046.9380 | 14.2892 | 13.7236 | 70.3388 | 0.5656 |
| 100-125 m2 | 273 | 3,954.3141 | 3,952.1054 | 13.6953 | 13.5052 | 2.2087 | 0.1901 |
| 125-150 m2 | 235 | 4,205.7464 | 3,958.4399 | 12.9367 | 11.8668 | 247.3065 | 1.0699 |
| 150-200 m2 | 103 | 8,025.6704 | 8,053.4384 | 17.2320 | 16.7964 | -27.7679 | 0.4356 |
| 200+ m2 | 46 | 10,055.0157 | 9,500.7698 | 14.1807 | 13.1041 | 554.2458 | 1.0766 |

## Residual Analysis

| segment | sample_count | mean_residual_try | median_residual_try | std_residual_try | mean_abs_error_try | underprediction_share_pct | overprediction_share_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 959 | -976.0044 | -227.2053 | 6,953.7863 | 4,626.1784 | 50.9906 | 49.0094 |
| high_price_75k_plus | 31 | -15,104.9382 | -10,947.4290 | 18,189.6916 | 18,649.0971 | 80.6452 | 19.3548 |
| luxury_proxy | 116 | -3,521.0149 | -1,542.1133 | 14,020.7749 | 10,011.7020 | 55.1724 | 44.8276 |

## Image Branch Ablation

- Log-target fusion MAE: **4626.18**
- Image branch sifirlandiginda MAE: **4946.30**
- Image branch sifirlandiginda MAPE: **14.22%**
- Ablasyon yorumu: Reduced image block standardized ve PCA ile olusturuldugu icin ablasyon testinde son image block sifira cekildi; bu yaklasim image branch katkisini ayirmak icin kullanildi.
- Yorum: Image branch sifirlandiginda MAE 320.12 kadar kotulesiyor; image embeddingler modele net pozitif katkili.

## En Yuksek Hata Yapan 20 Ilan

| listing_id | district | neighborhood | rooms | m2_gross | actual_price_try | baseline_prediction | fusion_prediction | fusion_residual | fusion_abs_error | fusion_ape_pct | abs_error_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hepsiemlak:149516-72 | Çankaya | Gaziosmanpaşa Mah. | 2+1 | 100.0000 | 99,000.0000 | 56,922.9254 | 45,263.6908 | -53,736.3092 | 53,736.3092 | 54.2791 | -11,659.2346 |
| hepsiemlak:43014-3617 | Çankaya | Oran Mah. | 4+1 | 175.0000 | 180,000.0000 | 128,045.0401 | 131,795.4364 | -48,204.5636 | 48,204.5636 | 26.7803 | 3,750.3963 |
| hepsiemlak:162654-112 | Çankaya | Çiğdem Mah. | 4+1 | 200.0000 | 120,000.0000 | 74,196.3756 | 72,859.0486 | -47,140.9514 | 47,140.9514 | 39.2841 | -1,337.3270 |
| hepsiemlak:130551-1404 | Çankaya | Yukarı Dikmen Mah. | 5+1 | 305.0000 | 149,000.0000 | 130,170.0458 | 108,569.6801 | -40,430.3199 | 40,430.3199 | 27.1344 | -21,600.3657 |
| hepsiemlak:5665-1536 | Çankaya | Namık Kemal Mah. | 3+0 | 139.0000 | 83,000.0000 | 57,041.3502 | 43,176.9082 | -39,823.0918 | 39,823.0918 | 47.9796 | -13,864.4420 |
| hepsiemlak:0-46238403 | Çankaya | Kavaklıdere Mah. | 3+2 | 150.0000 | 19,000.0000 | 55,611.1481 | 51,269.7806 | 32,269.7806 | 32,269.7806 | 169.8410 | 4,341.3676 |
| hepsiemlak:5978-6315 | Çankaya | Cevizlidere Mah. | 4+1 | 190.0000 | 80,000.0000 | 46,478.5771 | 48,934.7062 | -31,065.2938 | 31,065.2938 | 38.8316 | 2,456.1291 |
| hepsiemlak:7677-4917 | Çankaya | Mustafa Kemal Mah. | 3+1 | 145.0000 | 75,000.0000 | 42,669.2486 | 45,026.2025 | -29,973.7975 | 29,973.7975 | 39.9651 | 2,356.9540 |
| hepsiemlak:96373-1507 | Çankaya | Nasuh Akar Mah. | 4+1 | 175.0000 | 75,000.0000 | 66,813.3266 | 45,739.4217 | -29,260.5783 | 29,260.5783 | 39.0141 | -21,073.9049 |
| hepsiemlak:57012-7524 | Etimesgut | Yapracık Mah. | 4+1 | 210.0000 | 70,000.0000 | 45,262.2050 | 43,421.3694 | -26,578.6306 | 26,578.6306 | 37.9695 | -1,840.8356 |
| hepsiemlak:165177-42 | Çankaya | Prof. Dr. Ahmet Taner Kışlalı Mah. | 4+1 | 200.0000 | 93,000.0000 | 62,194.5871 | 67,051.9802 | -25,948.0198 | 25,948.0198 | 27.9011 | 4,857.3930 |
| hepsiemlak:8315-16683 | Çankaya | Kızılırmak Mah. | 4+1 | 185.0000 | 85,000.0000 | 59,649.0568 | 59,330.9307 | -25,669.0693 | 25,669.0693 | 30.1989 | -318.1261 |
| hepsiemlak:9800-2550 | Çankaya | Emek Mah. | 3+1 | 145.0000 | 65,000.0000 | 39,001.4751 | 42,902.6339 | -22,097.3661 | 22,097.3661 | 33.9959 | 3,901.1588 |
| hepsiemlak:56844-1171 | Çankaya | Remzi Oğuz Arık Mah. | 1+1 | 85.0000 | 52,000.0000 | 25,329.4443 | 30,295.7631 | -21,704.2369 | 21,704.2369 | 41.7389 | 4,966.3188 |
| hepsiemlak:40409-5765 | Çankaya | Yaşamkent Mah. | 4+1 | 185.0000 | 83,000.0000 | 60,659.2518 | 61,814.4733 | -21,185.5267 | 21,185.5267 | 25.5247 | 1,155.2215 |
| hepsiemlak:16602-1909 | Çankaya | Ayrancı Mah. | 1+1 | 50.0000 | 52,000.0000 | 34,228.3998 | 31,312.5044 | -20,687.4956 | 20,687.4956 | 39.7836 | -2,915.8954 |
| hepsiemlak:145733-213 | Çankaya | Cevizlidere Mah. | 5+1 | 375.0000 | 78,000.0000 | 96,425.8913 | 98,269.4480 | 20,269.4480 | 20,269.4480 | 25.9865 | -1,843.5567 |
| hepsiemlak:72233-408 | Çankaya | Ayrancı Mah. | 2+1 | 90.0000 | 60,000.0000 | 45,524.4441 | 41,144.7934 | -18,855.2066 | 18,855.2066 | 31.4253 | -4,379.6506 |
| hepsiemlak:31332-1974 | Keçiören | Basınevleri Mah. | 3+1 | 169.0000 | 56,000.0000 | 36,525.1977 | 37,366.4719 | -18,633.5281 | 18,633.5281 | 33.2742 | 841.2742 |
| hepsiemlak:61156-6048 | Çankaya | Büyükesat Mah. | 2+1 | 170.0000 | 40,000.0000 | 62,619.8139 | 58,281.3677 | 18,281.3677 | 18,281.3677 | 45.7034 | 4,338.4462 |

## Final Yorum

- Log-target model, matched tabular baseline'e gore MAE farkini 73.90 olarak degistirdi.
- Onceki 6-image reduced fusion referansina gore MAE degisimi: -70.45.
- Onceki all-image reduced fusion referansina gore MAE degisimi: -55.53.
- Bu kosuda temel hedef, long-tail kira dagiliminda ozellikle yuksek fiyat segmentini daha dengeli ogrenmekti; high-price ve luxury tablolari bu etkiyi ozetliyor.