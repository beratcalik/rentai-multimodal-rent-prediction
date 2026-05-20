# Final Multimodal Text+CLIP Results

## Ozet

- Multimodal source: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- CLIP caps embedding source: `E:\rent-agent\dataset\clip_image_embeddings_caps.parquet`
- Sabit image_cap: **16**
- Sabit image representation: **clip_meanmax_embedding**
- Ham CLIP embedding dimension: **1024**
- Join sonucu kalan ornek sayisi: **6,389**
- Denenen TF-IDF max_features: **2000, 5000**
- Denenen text SVD dim: **16, 32, 64**
- Denenen image PCA dim: **16, 32**
- En iyi kombinasyon: **tfidf=5000 + text_svd=32 + image_pca=32 + XGBRegressor**
- Kaydedilen model bundle: `E:\rent-agent\models\final_multimodal_text_clip_model.joblib`

## Validation Leaderboard

| text_max_features | text_svd_dim | image_pca_dim | model | validation_mae | validation_rmse | validation_r2 | validation_mape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | 32 | 32 | XGBRegressor | 4,249.5445 | 6,270.4630 | 0.8511 | 12.9750 |
| 5000 | 64 | 32 | XGBRegressor | 4,304.5729 | 6,342.0604 | 0.8477 | 13.0749 |
| 5000 | 64 | 16 | XGBRegressor | 4,310.5005 | 6,339.2436 | 0.8478 | 13.1550 |
| 2000 | 32 | 16 | XGBRegressor | 4,312.7056 | 6,412.2647 | 0.8443 | 13.1979 |
| 2000 | 64 | 16 | XGBRegressor | 4,322.0535 | 6,499.9975 | 0.8400 | 13.2104 |
| 5000 | 32 | 16 | XGBRegressor | 4,334.5878 | 6,309.4378 | 0.8493 | 13.2969 |
| 2000 | 64 | 32 | XGBRegressor | 4,335.9896 | 6,567.1087 | 0.8367 | 13.1392 |
| 2000 | 16 | 32 | XGBRegressor | 4,342.3489 | 6,575.7283 | 0.8363 | 13.1559 |
| 5000 | 16 | 16 | XGBRegressor | 4,343.6266 | 6,373.8568 | 0.8462 | 13.3176 |
| 2000 | 64 | 32 | LightGBMRegressor | 4,354.2032 | 6,511.9135 | 0.8394 | 13.2022 |
| 5000 | 16 | 32 | XGBRegressor | 4,356.2922 | 6,414.2787 | 0.8442 | 13.2622 |
| 5000 | 64 | 32 | LightGBMRegressor | 4,368.2027 | 6,432.4736 | 0.8433 | 13.2321 |
| 5000 | 32 | 32 | LightGBMRegressor | 4,375.0828 | 6,445.9024 | 0.8427 | 13.2607 |
| 2000 | 64 | 16 | LightGBMRegressor | 4,383.0252 | 6,546.9817 | 0.8377 | 13.2210 |
| 2000 | 32 | 32 | XGBRegressor | 4,385.8053 | 6,620.5440 | 0.8340 | 13.2469 |
| 2000 | 32 | 32 | LightGBMRegressor | 4,385.9714 | 6,640.8395 | 0.8330 | 13.2320 |
| 5000 | 16 | 32 | LightGBMRegressor | 4,386.8587 | 6,533.4579 | 0.8384 | 13.2092 |
| 2000 | 16 | 16 | XGBRegressor | 4,403.0024 | 6,603.3246 | 0.8349 | 13.4183 |
| 5000 | 64 | 16 | LightGBMRegressor | 4,436.2291 | 6,523.0586 | 0.8389 | 13.4811 |
| 5000 | 32 | 16 | LightGBMRegressor | 4,438.8079 | 6,557.2453 | 0.8372 | 13.4547 |
| 2000 | 16 | 16 | LightGBMRegressor | 4,454.2451 | 6,664.7635 | 0.8318 | 13.4671 |
| 2000 | 16 | 32 | LightGBMRegressor | 4,460.2655 | 6,657.3074 | 0.8322 | 13.4625 |
| 5000 | 16 | 16 | LightGBMRegressor | 4,478.5468 | 6,554.2835 | 0.8373 | 13.5003 |
| 5000 | 16 | 32 | HistGradientBoostingRegressor | 4,487.8482 | 6,654.5168 | 0.8323 | 13.6031 |
| 5000 | 64 | 32 | HistGradientBoostingRegressor | 4,489.0324 | 6,712.8949 | 0.8294 | 13.4777 |
| 5000 | 16 | 16 | HistGradientBoostingRegressor | 4,503.2205 | 6,645.5290 | 0.8328 | 13.6163 |
| 2000 | 32 | 32 | HistGradientBoostingRegressor | 4,507.0563 | 6,748.3609 | 0.8275 | 13.4477 |
| 2000 | 32 | 16 | LightGBMRegressor | 4,508.3591 | 6,751.8886 | 0.8274 | 13.5773 |
| 5000 | 32 | 32 | HistGradientBoostingRegressor | 4,511.2246 | 6,685.1571 | 0.8308 | 13.6124 |
| 5000 | 32 | 16 | HistGradientBoostingRegressor | 4,517.3407 | 6,683.1350 | 0.8309 | 13.7028 |
| 2000 | 64 | 32 | HistGradientBoostingRegressor | 4,526.8731 | 6,741.5893 | 0.8279 | 13.6764 |
| 2000 | 16 | 32 | HistGradientBoostingRegressor | 4,534.9272 | 6,772.1498 | 0.8263 | 13.7119 |
| 5000 | 64 | 16 | HistGradientBoostingRegressor | 4,554.7447 | 6,733.9417 | 0.8283 | 13.8203 |
| 2000 | 64 | 16 | HistGradientBoostingRegressor | 4,585.6396 | 6,837.3844 | 0.8230 | 13.8055 |
| 2000 | 16 | 16 | HistGradientBoostingRegressor | 4,595.0591 | 6,947.2627 | 0.8172 | 13.8694 |
| 2000 | 32 | 16 | HistGradientBoostingRegressor | 4,598.1785 | 6,908.2121 | 0.8193 | 13.8035 |

## Text Feature Config Comparison

| text_max_features | text_svd_dim | best_image_pca_dim | best_model | best_validation_mae | best_validation_rmse | best_validation_r2 | best_validation_mape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2000 | 16 | 32 | XGBRegressor | 4,342.3489 | 6,575.7283 | 0.8363 | 13.1559 |
| 2000 | 32 | 16 | XGBRegressor | 4,312.7056 | 6,412.2647 | 0.8443 | 13.1979 |
| 2000 | 64 | 16 | XGBRegressor | 4,322.0535 | 6,499.9975 | 0.8400 | 13.2104 |
| 5000 | 16 | 16 | XGBRegressor | 4,343.6266 | 6,373.8568 | 0.8462 | 13.3176 |
| 5000 | 32 | 32 | XGBRegressor | 4,249.5445 | 6,270.4630 | 0.8511 | 12.9750 |
| 5000 | 64 | 32 | XGBRegressor | 4,304.5729 | 6,342.0604 | 0.8477 | 13.0749 |

## Image PCA Comparison

| image_pca_dim | best_text_max_features | best_text_svd_dim | best_model | best_validation_mae | best_validation_rmse | best_validation_r2 | best_validation_mape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 5000 | 64 | XGBRegressor | 4,310.5005 | 6,339.2436 | 0.8478 | 13.1550 |
| 32 | 5000 | 32 | XGBRegressor | 4,249.5445 | 6,270.4630 | 0.8511 | 12.9750 |

## Final Test Sonuclari

| Metric | Value |
| --- | --- |
| MAE | 4,172.32 |
| RMSE | 6,179.27 |
| R2 | 0.8477 |
| MAPE (%) | 12.50 |

## Baseline vs CLIP vs Multimodal Karsilastirmasi

| row | mae | rmse | r2 | mape |
| --- | --- | --- | --- | --- |
| Matched tabular baseline | 4,700.0787 | 7,057.7587 | 0.8013 | 13.7867 |
| Historical CLIP best | 4,346.6200 | 6,441.2800 | 0.8345 | 12.9500 |
| Rebuilt CLIP best | 4,346.6193 | 6,441.2836 | 0.8345 | 12.9463 |
| Historical SigLIP best | 4,441.8600 | 6,484.5000 | 0.8323 | 12.9800 |
| Final multimodal text+clip | 4,172.3186 | 6,179.2749 | 0.8477 | 12.5003 |
| Improvement vs matched baseline | 527.7601 | 878.4837 | 0.0464 | 1.2864 |
| Improvement vs CLIP best | 174.3007 | 262.0087 | 0.0132 | 0.4460 |

## District Bazli Improvement (CLIP vs Multimodal)

| group | sample_count | reference_mae | target_mae | reference_mape | target_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Polatlı | 12 | 4,371.9486 | 3,311.7438 | 18.2191 | 14.4090 | 1,060.2048 | 3.8102 |
| Pursaklar | 10 | 4,341.7141 | 3,919.5713 | 12.9942 | 12.5676 | 422.1428 | 0.4266 |
| Çankaya | 379 | 5,561.6106 | 5,167.4669 | 13.6780 | 12.7358 | 394.1437 | 0.9422 |
| Keçiören | 142 | 3,603.4710 | 3,522.0590 | 12.9012 | 12.4500 | 81.4120 | 0.4512 |
| Sincan | 58 | 3,443.9251 | 3,383.1629 | 13.6709 | 12.8661 | 60.7622 | 0.8048 |
| Yenimahalle | 90 | 4,337.7071 | 4,287.3323 | 12.2459 | 12.1751 | 50.3748 | 0.0709 |
| Gölbaşı | 56 | 3,422.2551 | 3,380.0661 | 9.9762 | 9.7445 | 42.1890 | 0.2317 |
| Mamak | 75 | 3,012.4807 | 3,040.8324 | 11.1966 | 11.2529 | -28.3517 | -0.0563 |
| Etimesgut | 82 | 3,251.6871 | 3,290.4068 | 11.0785 | 11.3535 | -38.7198 | -0.2750 |
| Çubuk | 15 | 2,519.0405 | 2,580.3443 | 17.0009 | 17.9161 | -61.3038 | -0.9152 |
| Altındağ | 33 | 3,538.2641 | 3,881.8659 | 13.6985 | 15.4634 | -343.6018 | -1.7649 |
| Akyurt | 5 | 2,007.9992 | 2,889.4855 | 10.8640 | 13.8950 | -881.4863 | -3.0310 |

## Price Range Bazli Improvement (CLIP vs Multimodal)

| group | sample_count | reference_mae | target_mae | reference_mape | target_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-20k TRY | 84 | 3,929.6850 | 3,829.9739 | 23.0467 | 22.6155 | 99.7111 | 0.4311 |
| 20k-30k TRY | 372 | 2,882.7568 | 2,776.2419 | 11.7813 | 11.3462 | 106.5149 | 0.4352 |
| 30k-40k TRY | 266 | 3,872.8646 | 3,811.5805 | 11.4050 | 11.2352 | 61.2840 | 0.1698 |
| 40k-50k TRY | 122 | 5,306.8410 | 4,939.0090 | 12.1833 | 11.3698 | 367.8320 | 0.8135 |
| 50k-75k TRY | 84 | 7,048.1305 | 6,829.8999 | 12.2640 | 11.8631 | 218.2306 | 0.4008 |
| 75k+ TRY | 31 | 16,008.6870 | 14,729.7607 | 17.6356 | 15.9727 | 1,278.9263 | 1.6629 |

## m2 Range Bazli Improvement (CLIP vs Multimodal)

| group | sample_count | reference_mae | target_mae | reference_mape | target_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-75 m2 | 152 | 3,284.7246 | 3,258.6507 | 11.7161 | 11.8284 | 26.0739 | -0.1122 |
| 75-100 m2 | 150 | 3,939.8535 | 3,914.4968 | 13.9045 | 13.7259 | 25.3567 | 0.1787 |
| 100-125 m2 | 273 | 3,774.4189 | 3,605.9439 | 13.1244 | 12.4944 | 168.4750 | 0.6299 |
| 125-150 m2 | 235 | 3,704.1705 | 3,526.5089 | 11.4512 | 10.9638 | 177.6616 | 0.4874 |
| 150-200 m2 | 103 | 7,283.0401 | 6,642.5377 | 15.9679 | 14.7009 | 640.5024 | 1.2670 |
| 200+ m2 | 46 | 9,284.8301 | 9,161.5304 | 13.7027 | 13.6814 | 123.2997 | 0.0213 |

## Text Branch Ablation

- Final multimodal MAE: **4172.32**
- Text branch sifirlaninca MAE: **4482.72**
- Text branch sifirlaninca RMSE: **6565.93**
- Text branch sifirlaninca R2: **0.8280**
- Text branch sifirlaninca MAPE: **13.50%**

## Image Branch Ablation

- Image branch sifirlaninca MAE: **5494.53**
- Image branch sifirlaninca RMSE: **8248.88**
- Image branch sifirlaninca R2: **0.7286**
- Image branch sifirlaninca MAPE: **16.46%**

## Text+Image Birlikte Ablation

- Text+image birlikte sifirlaninca MAE: **5965.51**
- Text+image birlikte sifirlaninca RMSE: **8745.75**
- Text+image birlikte sifirlaninca R2: **0.6949**
- Text+image birlikte sifirlaninca MAPE: **18.29%**
- Ablation notu: Ablasyonlar reduced text ve reduced image bloklarini sifira cekerek olculdu; boylece tabular blok sabit tutulurken text ve image katkisi ayrica gozlemlendi.

## En Yuksek Hata Yapan 20 Ilan

| listing_id | district | neighborhood | rooms | m2_gross | actual_price_try | clip_prediction | multimodal_prediction | multimodal_residual | multimodal_abs_error | multimodal_ape_pct | abs_error_gain_vs_clip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hepsiemlak:149516-72 | Çankaya | Gaziosmanpaşa Mah. | 2+1 | 100.0000 | 99,000.0000 | 47,343.8516 | 47,833.4766 | -51,166.5234 | 51,166.5234 | 51.6834 | 489.6250 |
| hepsiemlak:130551-1404 | Çankaya | Yukarı Dikmen Mah. | 5+1 | 305.0000 | 149,000.0000 | 111,101.3672 | 108,768.1172 | -40,231.8828 | 40,231.8828 | 27.0013 | -2,333.2500 |
| hepsiemlak:162654-112 | Çankaya | Çiğdem Mah. | 4+1 | 200.0000 | 120,000.0000 | 84,473.9062 | 83,999.3047 | -36,000.6953 | 36,000.6953 | 30.0006 | -474.6016 |
| hepsiemlak:5978-6315 | Çankaya | Cevizlidere Mah. | 4+1 | 190.0000 | 80,000.0000 | 47,879.2344 | 49,528.3555 | -30,471.6445 | 30,471.6445 | 38.0896 | 1,649.1211 |
| hepsiemlak:5665-1536 | Çankaya | Namık Kemal Mah. | 3+0 | 139.0000 | 83,000.0000 | 48,879.7070 | 52,919.8477 | -30,080.1523 | 30,080.1523 | 36.2411 | 4,040.1406 |
| hepsiemlak:0-46238403 | Çankaya | Kavaklıdere Mah. | 3+2 | 150.0000 | 19,000.0000 | 47,173.6484 | 48,684.9922 | 29,684.9922 | 29,684.9922 | 156.2368 | -1,511.3438 |
| hepsiemlak:8315-16683 | Çankaya | Kızılırmak Mah. | 4+1 | 185.0000 | 85,000.0000 | 58,653.1406 | 57,765.2227 | -27,234.7773 | 27,234.7773 | 32.0409 | -887.9180 |
| hepsiemlak:7677-4917 | Çankaya | Mustafa Kemal Mah. | 3+1 | 145.0000 | 75,000.0000 | 48,382.3867 | 48,119.5508 | -26,880.4492 | 26,880.4492 | 35.8406 | -262.8359 |
| hepsiemlak:156606-360 | Keçiören | Ovacık Mah. | 4+1 | 230.0000 | 50,000.0000 | 77,646.5859 | 73,928.6953 | 23,928.6953 | 23,928.6953 | 47.8574 | 3,717.8906 |
| hepsiemlak:56844-1171 | Çankaya | Remzi Oğuz Arık Mah. | 1+1 | 85.0000 | 52,000.0000 | 30,927.4902 | 31,330.6309 | -20,669.3691 | 20,669.3691 | 39.7488 | 403.1406 |
| hepsiemlak:138926-205 | Etimesgut | Eryaman Mah. | 3+1 | 167.0000 | 60,000.0000 | 45,941.3945 | 39,562.3867 | -20,437.6133 | 20,437.6133 | 34.0627 | -6,379.0078 |
| hepsiemlak:38897-1578 | Çankaya | Namık Kemal Mah. | 3+1 | 155.0000 | 85,000.0000 | 63,033.4570 | 64,605.3086 | -20,394.6914 | 20,394.6914 | 23.9938 | 1,571.8516 |
| hepsiemlak:167779-54 | Sincan | Mustafa Kemal Mah. | 4+1 | 180.0000 | 34,000.0000 | 50,690.8125 | 53,454.7656 | 19,454.7656 | 19,454.7656 | 57.2199 | -2,763.9531 |
| hepsiemlak:16602-1909 | Çankaya | Ayrancı Mah. | 1+1 | 50.0000 | 52,000.0000 | 28,986.7500 | 32,698.5000 | -19,301.5000 | 19,301.5000 | 37.1183 | 3,711.7500 |
| hepsiemlak:145733-213 | Çankaya | Cevizlidere Mah. | 5+1 | 375.0000 | 78,000.0000 | 113,562.3047 | 96,546.6406 | 18,546.6406 | 18,546.6406 | 23.7777 | 17,015.6641 |
| hepsiemlak:37602-2791 | Etimesgut | Aşağıyurtçu Mah. | 4+1 | 180.0000 | 42,000.0000 | 61,492.4180 | 60,289.3438 | 18,289.3438 | 18,289.3438 | 43.5461 | 1,203.0742 |
| hepsiemlak:9800-2550 | Çankaya | Emek Mah. | 3+1 | 145.0000 | 65,000.0000 | 47,472.9883 | 46,747.6562 | -18,252.3438 | 18,252.3438 | 28.0805 | -725.3320 |
| hepsiemlak:61156-6048 | Çankaya | Büyükesat Mah. | 2+1 | 170.0000 | 40,000.0000 | 68,035.0781 | 57,849.3281 | 17,849.3281 | 17,849.3281 | 44.6233 | 10,185.7500 |
| hepsiemlak:165177-42 | Çankaya | Prof. Dr. Ahmet Taner Kışlalı Mah. | 4+1 | 200.0000 | 93,000.0000 | 68,810.7969 | 75,278.3203 | -17,721.6797 | 17,721.6797 | 19.0556 | 6,467.5234 |
| hepsiemlak:155679-108 | Gölbaşı | Taşpınar Mah. | 3+1 | 230.0000 | 52,000.0000 | 69,879.7969 | 69,587.6172 | 17,587.6172 | 17,587.6172 | 33.8223 | 292.1797 |

## Model Durumu

- `XGBRegressor`: calisti; denenen kombinasyonlar arasinda tfidf=2000|text=16|image=16, tfidf=2000|text=16|image=32, tfidf=2000|text=32|image=16, tfidf=2000|text=32|image=32, tfidf=2000|text=64|image=16, tfidf=2000|text=64|image=32, tfidf=5000|text=16|image=16, tfidf=5000|text=16|image=32 ...
- `LightGBMRegressor`: calisti; denenen kombinasyonlar arasinda tfidf=2000|text=16|image=16, tfidf=2000|text=16|image=32, tfidf=2000|text=32|image=16, tfidf=2000|text=32|image=32, tfidf=2000|text=64|image=16, tfidf=2000|text=64|image=32, tfidf=5000|text=16|image=16, tfidf=5000|text=16|image=32 ...
- `HistGradientBoostingRegressor`: calisti; denenen kombinasyonlar arasinda tfidf=2000|text=16|image=16, tfidf=2000|text=16|image=32, tfidf=2000|text=32|image=16, tfidf=2000|text=32|image=32, tfidf=2000|text=64|image=16, tfidf=2000|text=64|image=32, tfidf=5000|text=16|image=16, tfidf=5000|text=16|image=32 ...

## Sonuc Yorumu

- matched tabular baseline'a gore MAE tarafinda 527.76 TRY iyilesme var; CLIP referansina gore MAE 174.30 TRY daha iyi; en iyi validation kombinasyonu TF-IDF max_features=5000, text_svd=32, image_pca=32, model=XGBRegressor; text branch ablasyonda MAE 310.40 kadar kotulestigi icin text sinyali ek fayda uretiyor; image branch ablasyonda MAE 1322.21 kadar kotulestigi icin CLIP image sinyali kritik katkida bulunuyor; text+image birlikte kaldirildiginda MAE 1793.19 kadar kotulesti.