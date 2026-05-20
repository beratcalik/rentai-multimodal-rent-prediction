# CLIP Fusion Reduced Results

## Ozet

- Multimodal source: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- CLIP embedding source: `E:\rent-agent\dataset\clip_image_embeddings.parquet`
- Kaydedilen model bundle: `E:\rent-agent\models\clip_fusion_reduced_model.joblib`
- Join sonucu kalan ornek sayisi: **6,389**
- Deneen representationlar: **clip_mean_embedding, clip_max_embedding, clip_meanmax_embedding**
- Representation dimensionlari: **clip_mean_embedding=512, clip_max_embedding=512, clip_meanmax_embedding=1024**
- Reducer: **PCA**
- Deneen image_dim degerleri: **16, 32, 64, 128**
- En iyi kombinasyon: **clip_mean_embedding + XGBRegressor + image_dim=16**

## Validation Leaderboard

| representation | image_dim | model | validation_mae | validation_rmse | validation_r2 | validation_mape |
| --- | --- | --- | --- | --- | --- | --- |
| clip_mean_embedding | 16 | XGBRegressor | 4,346.7265 | 6,538.0503 | 0.8381 | 13.2896 |
| clip_meanmax_embedding | 16 | XGBRegressor | 4,352.0361 | 6,518.6539 | 0.8391 | 13.2839 |
| clip_max_embedding | 16 | XGBRegressor | 4,356.6684 | 6,401.4574 | 0.8448 | 13.2426 |
| clip_max_embedding | 32 | XGBRegressor | 4,372.8677 | 6,433.9758 | 0.8432 | 13.2249 |
| clip_meanmax_embedding | 32 | XGBRegressor | 4,390.2760 | 6,600.2898 | 0.8350 | 13.2709 |
| clip_mean_embedding | 32 | XGBRegressor | 4,395.6641 | 6,584.6020 | 0.8358 | 13.3562 |
| clip_meanmax_embedding | 64 | XGBRegressor | 4,402.6724 | 6,534.6393 | 0.8383 | 13.3298 |
| clip_meanmax_embedding | 128 | XGBRegressor | 4,413.9327 | 6,609.3295 | 0.8346 | 13.3046 |
| clip_meanmax_embedding | 64 | LightGBMRegressor | 4,430.5047 | 6,684.4117 | 0.8308 | 13.3279 |
| clip_mean_embedding | 64 | XGBRegressor | 4,433.0612 | 6,658.8446 | 0.8321 | 13.4314 |
| clip_meanmax_embedding | 16 | LightGBMRegressor | 4,448.3955 | 6,765.9433 | 0.8266 | 13.4178 |
| clip_mean_embedding | 32 | LightGBMRegressor | 4,456.7373 | 6,668.2061 | 0.8316 | 13.4217 |
| clip_mean_embedding | 64 | LightGBMRegressor | 4,478.9256 | 6,678.5694 | 0.8311 | 13.4265 |
| clip_max_embedding | 64 | XGBRegressor | 4,481.9367 | 6,560.4342 | 0.8370 | 13.5426 |
| clip_mean_embedding | 16 | LightGBMRegressor | 4,512.3118 | 6,796.1086 | 0.8251 | 13.4878 |
| clip_max_embedding | 16 | LightGBMRegressor | 4,514.2147 | 6,679.6294 | 0.8310 | 13.5180 |
| clip_meanmax_embedding | 32 | LightGBMRegressor | 4,517.0325 | 6,692.7959 | 0.8304 | 13.6177 |
| clip_meanmax_embedding | 16 | HistGradientBoostingRegressor | 4,537.7967 | 6,991.4829 | 0.8149 | 13.5971 |
| clip_meanmax_embedding | 128 | LightGBMRegressor | 4,543.0920 | 6,872.8877 | 0.8211 | 13.5847 |
| clip_max_embedding | 32 | LightGBMRegressor | 4,546.1009 | 6,682.3337 | 0.8309 | 13.6427 |
| clip_mean_embedding | 32 | HistGradientBoostingRegressor | 4,547.4599 | 6,855.0960 | 0.8220 | 13.6416 |
| clip_max_embedding | 128 | XGBRegressor | 4,550.7409 | 6,843.3675 | 0.8227 | 13.6478 |
| clip_mean_embedding | 128 | XGBRegressor | 4,559.2795 | 6,853.8742 | 0.8221 | 13.7615 |
| clip_mean_embedding | 16 | HistGradientBoostingRegressor | 4,559.7654 | 6,911.5619 | 0.8191 | 13.6077 |
| clip_max_embedding | 64 | LightGBMRegressor | 4,567.3506 | 6,858.4675 | 0.8219 | 13.6064 |
| clip_mean_embedding | 128 | LightGBMRegressor | 4,570.3898 | 6,829.1153 | 0.8234 | 13.7503 |
| clip_mean_embedding | 64 | HistGradientBoostingRegressor | 4,572.4362 | 6,865.1711 | 0.8215 | 13.7183 |
| clip_meanmax_embedding | 32 | HistGradientBoostingRegressor | 4,584.6516 | 6,882.2349 | 0.8206 | 13.7338 |
| clip_meanmax_embedding | 64 | HistGradientBoostingRegressor | 4,585.8505 | 6,860.8137 | 0.8218 | 13.6605 |
| clip_max_embedding | 64 | HistGradientBoostingRegressor | 4,600.2201 | 6,823.1020 | 0.8237 | 13.7669 |
| clip_max_embedding | 16 | HistGradientBoostingRegressor | 4,610.6497 | 6,804.3219 | 0.8247 | 13.7716 |
| clip_mean_embedding | 128 | HistGradientBoostingRegressor | 4,613.0988 | 6,963.6535 | 0.8164 | 13.7491 |
| clip_max_embedding | 128 | LightGBMRegressor | 4,615.3219 | 6,907.6347 | 0.8193 | 13.7921 |
| clip_meanmax_embedding | 128 | HistGradientBoostingRegressor | 4,641.2703 | 6,936.8199 | 0.8178 | 13.9259 |
| clip_max_embedding | 32 | HistGradientBoostingRegressor | 4,660.4597 | 6,914.2041 | 0.8190 | 13.9415 |
| clip_max_embedding | 128 | HistGradientBoostingRegressor | 4,767.6111 | 7,230.3794 | 0.8020 | 14.2106 |

## Representation Bazli Skorlar

| representation | best_model | best_image_dim | best_validation_mae | best_validation_rmse | best_validation_r2 | best_validation_mape |
| --- | --- | --- | --- | --- | --- | --- |
| clip_mean_embedding | XGBRegressor | 16 | 4,346.7265 | 6,538.0503 | 0.8381 | 13.2896 |
| clip_max_embedding | XGBRegressor | 16 | 4,356.6684 | 6,401.4574 | 0.8448 | 13.2426 |
| clip_meanmax_embedding | XGBRegressor | 16 | 4,352.0361 | 6,518.6539 | 0.8391 | 13.2839 |

## Image Dim Bazli Skorlar

| image_dim | best_representation | best_model | best_validation_mae | best_validation_rmse | best_validation_r2 | best_validation_mape |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | clip_mean_embedding | XGBRegressor | 4,346.7265 | 6,538.0503 | 0.8381 | 13.2896 |
| 32 | clip_max_embedding | XGBRegressor | 4,372.8677 | 6,433.9758 | 0.8432 | 13.2249 |
| 64 | clip_meanmax_embedding | XGBRegressor | 4,402.6724 | 6,534.6393 | 0.8383 | 13.3298 |
| 128 | clip_meanmax_embedding | XGBRegressor | 4,413.9327 | 6,609.3295 | 0.8346 | 13.3046 |

## Final Test Sonuclari

| Metric | Value |
| --- | --- |
| MAE | 4,381.56 |
| RMSE | 6,359.81 |
| R2 | 0.8387 |
| MAPE (%) | 13.10 |

## Matched Baseline ve EfficientNet Referanslari Karsilastirmasi

| row | mae | rmse | r2 | mape |
| --- | --- | --- | --- | --- |
| Matched tabular baseline | 4,700.0787 | 7,057.7587 | 0.8013 | 13.7867 |
| EfficientNet 6-image reduced fusion | 4,555.7300 | 6,842.2000 | 0.8133 | 13.5300 |
| EfficientNet all-image reduced fusion | 4,570.6500 | 6,673.1600 | 0.8224 | 13.5900 |
| CLIP reduced fusion | 4,381.5586 | 6,359.8115 | 0.8387 | 13.0994 |
| Improvement vs matched baseline | 318.5201 | 697.9471 | 0.0374 | 0.6873 |
| Improvement vs EfficientNet 6-image | 174.1714 | 482.3885 | 0.0254 | 0.4306 |
| Improvement vs EfficientNet all-image | 189.0914 | 313.3485 | 0.0163 | 0.4906 |

## District Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Çankaya | 379 | 6,277.1725 | 5,594.6489 | 15.2434 | 13.9253 | 682.5236 | 1.3181 |
| Akyurt | 5 | 3,156.1552 | 2,486.6086 | 14.7552 | 12.7104 | 669.5466 | 2.0448 |
| Pursaklar | 10 | 5,075.2632 | 4,493.0730 | 15.7168 | 13.4041 | 582.1901 | 2.3127 |
| Altındağ | 33 | 4,002.8446 | 3,535.4833 | 15.8675 | 13.8360 | 467.3613 | 2.0315 |
| Etimesgut | 82 | 3,632.4278 | 3,256.5794 | 11.7475 | 10.9629 | 375.8483 | 0.7846 |
| Gölbaşı | 56 | 3,835.0616 | 3,553.7482 | 11.8777 | 10.4229 | 281.3134 | 1.4548 |
| Keçiören | 142 | 3,827.3778 | 3,596.7587 | 13.6167 | 12.8235 | 230.6191 | 0.7932 |
| Sincan | 58 | 3,421.4291 | 3,471.7566 | 13.3003 | 13.6531 | -50.3276 | -0.3528 |
| Mamak | 75 | 2,936.1545 | 3,094.7009 | 10.8235 | 11.3466 | -158.5464 | -0.5231 |
| Polatlı | 12 | 3,214.2595 | 3,458.9525 | 16.4078 | 15.5198 | -244.6930 | 0.8879 |
| Yenimahalle | 90 | 4,086.4141 | 4,412.6467 | 11.4672 | 12.4545 | -326.2327 | -0.9874 |
| Çubuk | 15 | 2,087.9365 | 2,984.7154 | 13.8520 | 19.4255 | -896.7789 | -5.5735 |

## Price Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-20k TRY | 84 | 4,130.3212 | 4,106.8462 | 24.3314 | 24.0087 | 23.4750 | 0.3227 |
| 20k-30k TRY | 372 | 2,905.2472 | 2,861.0300 | 11.9290 | 11.7100 | 44.2172 | 0.2190 |
| 30k-40k TRY | 266 | 4,224.3421 | 3,960.8518 | 12.4604 | 11.6486 | 263.4903 | 0.8117 |
| 40k-50k TRY | 122 | 5,594.9134 | 5,500.8138 | 12.8582 | 12.5945 | 94.0996 | 0.2637 |
| 50k-75k TRY | 84 | 8,840.6085 | 7,041.1506 | 15.2758 | 12.2270 | 1,799.4579 | 3.0488 |
| 75k+ TRY | 31 | 17,122.9349 | 15,370.7717 | 18.5068 | 17.0117 | 1,752.1632 | 1.4951 |

## m2 Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-75 m2 | 152 | 3,504.8093 | 3,335.9763 | 12.3153 | 12.1112 | 168.8330 | 0.2041 |
| 75-100 m2 | 150 | 4,117.2768 | 3,977.8804 | 14.2892 | 13.9672 | 139.3964 | 0.3221 |
| 100-125 m2 | 273 | 3,954.3141 | 3,803.3918 | 13.6953 | 13.2258 | 150.9223 | 0.4694 |
| 125-150 m2 | 235 | 4,205.7464 | 3,797.0953 | 12.9367 | 11.6888 | 408.6511 | 1.2479 |
| 150-200 m2 | 103 | 8,025.6704 | 7,465.8488 | 17.2320 | 16.4000 | 559.8216 | 0.8321 |
| 200+ m2 | 46 | 10,055.0157 | 8,663.8794 | 14.1807 | 12.6013 | 1,391.1363 | 1.5794 |

## Used Image Count Analizi

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-4 images | 1 | 35.1493 | 17,703.5859 | 0.0469 | 23.6048 | -17,668.4367 | -23.5579 |
| 5-8 images | 9 | 1,999.3300 | 1,841.6433 | 7.9406 | 7.9345 | 157.6867 | 0.0061 |
| 9-12 images | 43 | 5,249.6558 | 4,551.0342 | 19.1167 | 16.7783 | 698.6216 | 2.3384 |
| 13-16 images | 906 | 4,705.9726 | 4,384.0418 | 13.6070 | 12.9645 | 321.9308 | 0.6424 |

## Image Branch Ablation

- Final CLIP fusion MAE: **4381.56**
- Ablation sonrasi MAE: **5802.33**
- Ablation sonrasi RMSE: **8381.41**
- Ablation sonrasi R2: **0.7198**
- Ablation sonrasi MAPE: **17.68%**
- Ablation notu: Reduced image block standardized ve PCA ile olusturuldugu icin ablasyon testinde son image block sifira cekildi; bu yaklasim reduced image branch katkisini ayirmak icin kullanildi.

## En Yuksek Hata Yapan 20 Ilan

| listing_id | district | neighborhood | rooms | m2_gross | actual_price_try | baseline_prediction | fusion_prediction | fusion_residual | fusion_abs_error | fusion_ape_pct | abs_error_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hepsiemlak:149516-72 | Çankaya | Gaziosmanpaşa Mah. | 2+1 | 100.0000 | 99,000.0000 | 56,922.9254 | 50,332.0352 | -48,667.9648 | 48,667.9648 | 49.1596 | -6,590.8902 |
| hepsiemlak:162654-112 | Çankaya | Çiğdem Mah. | 4+1 | 200.0000 | 120,000.0000 | 74,196.3756 | 80,541.1641 | -39,458.8359 | 39,458.8359 | 32.8824 | 6,344.7884 |
| hepsiemlak:5665-1536 | Çankaya | Namık Kemal Mah. | 3+0 | 139.0000 | 83,000.0000 | 57,041.3502 | 49,794.6914 | -33,205.3086 | 33,205.3086 | 40.0064 | -7,246.6588 |
| hepsiemlak:5978-6315 | Çankaya | Cevizlidere Mah. | 4+1 | 190.0000 | 80,000.0000 | 46,478.5771 | 47,091.3945 | -32,908.6055 | 32,908.6055 | 41.1358 | 612.8174 |
| hepsiemlak:130551-1404 | Çankaya | Yukarı Dikmen Mah. | 5+1 | 305.0000 | 149,000.0000 | 130,170.0458 | 117,026.2734 | -31,973.7266 | 31,973.7266 | 21.4589 | -13,143.7724 |
| hepsiemlak:145733-213 | Çankaya | Cevizlidere Mah. | 5+1 | 375.0000 | 78,000.0000 | 96,425.8913 | 108,836.7109 | 30,836.7109 | 30,836.7109 | 39.5342 | -12,410.8196 |
| hepsiemlak:0-46238403 | Çankaya | Kavaklıdere Mah. | 3+2 | 150.0000 | 19,000.0000 | 55,611.1481 | 49,486.2344 | 30,486.2344 | 30,486.2344 | 160.4539 | 6,124.9138 |
| hepsiemlak:156606-360 | Keçiören | Ovacık Mah. | 4+1 | 230.0000 | 50,000.0000 | 63,554.6823 | 76,639.6172 | 26,639.6172 | 26,639.6172 | 53.2792 | -13,084.9348 |
| hepsiemlak:7677-4917 | Çankaya | Mustafa Kemal Mah. | 3+1 | 145.0000 | 75,000.0000 | 42,669.2486 | 48,961.7383 | -26,038.2617 | 26,038.2617 | 34.7177 | 6,292.4897 |
| hepsiemlak:61156-6048 | Çankaya | Büyükesat Mah. | 2+1 | 170.0000 | 40,000.0000 | 62,619.8139 | 65,997.7969 | 25,997.7969 | 25,997.7969 | 64.9945 | -3,377.9830 |
| hepsiemlak:165177-42 | Çankaya | Prof. Dr. Ahmet Taner Kışlalı Mah. | 4+1 | 200.0000 | 93,000.0000 | 62,194.5871 | 70,877.1328 | -22,122.8672 | 22,122.8672 | 23.7880 | 8,682.5457 |
| hepsiemlak:8315-16683 | Çankaya | Kızılırmak Mah. | 4+1 | 185.0000 | 85,000.0000 | 59,649.0568 | 63,288.2695 | -21,711.7305 | 21,711.7305 | 25.5432 | 3,639.2127 |
| hepsiemlak:56844-1171 | Çankaya | Remzi Oğuz Arık Mah. | 1+1 | 85.0000 | 52,000.0000 | 25,329.4443 | 30,851.5000 | -21,148.5000 | 21,148.5000 | 40.6702 | 5,522.0557 |
| hepsiemlak:38897-1578 | Çankaya | Namık Kemal Mah. | 3+1 | 155.0000 | 85,000.0000 | 93,543.4714 | 64,435.7578 | -20,564.2422 | 20,564.2422 | 24.1932 | -12,020.7708 |
| hepsiemlak:31332-1974 | Keçiören | Basınevleri Mah. | 3+1 | 169.0000 | 56,000.0000 | 36,525.1977 | 35,931.7891 | -20,068.2109 | 20,068.2109 | 35.8361 | -593.4086 |
| hepsiemlak:37602-2791 | Etimesgut | Aşağıyurtçu Mah. | 4+1 | 180.0000 | 42,000.0000 | 63,423.8385 | 60,594.3516 | 18,594.3516 | 18,594.3516 | 44.2723 | 2,829.4869 |
| hepsiemlak:155679-108 | Gölbaşı | Taşpınar Mah. | 3+1 | 230.0000 | 52,000.0000 | 67,633.3731 | 70,470.3594 | 18,470.3594 | 18,470.3594 | 35.5199 | -2,836.9863 |
| hepsiemlak:16602-1909 | Çankaya | Ayrancı Mah. | 1+1 | 50.0000 | 52,000.0000 | 34,228.3998 | 34,175.9219 | -17,824.0781 | 17,824.0781 | 34.2771 | -52.4779 |
| hepsiemlak:159356-99 | Çankaya | Yaşamkent Mah. | 4+1 | 195.0000 | 75,000.0000 | 74,964.8507 | 92,703.5859 | 17,703.5859 | 17,703.5859 | 23.6048 | -17,668.4367 |
| hepsiemlak:167779-54 | Sincan | Mustafa Kemal Mah. | 4+1 | 180.0000 | 34,000.0000 | 43,337.1916 | 51,324.8711 | 17,324.8711 | 17,324.8711 | 50.9555 | -7,987.6795 |

## Model Durumu

- `XGBRegressor`: calisti; denenen kombinasyonlar arasinda clip_max_embedding@128, clip_max_embedding@16, clip_max_embedding@32, clip_max_embedding@64, clip_mean_embedding@128, clip_mean_embedding@16, clip_mean_embedding@32, clip_mean_embedding@64 ...
- `LightGBMRegressor`: calisti; denenen kombinasyonlar arasinda clip_max_embedding@128, clip_max_embedding@16, clip_max_embedding@32, clip_max_embedding@64, clip_mean_embedding@128, clip_mean_embedding@16, clip_mean_embedding@32, clip_mean_embedding@64 ...
- `HistGradientBoostingRegressor`: calisti; denenen kombinasyonlar arasinda clip_max_embedding@128, clip_max_embedding@16, clip_max_embedding@32, clip_max_embedding@64, clip_mean_embedding@128, clip_mean_embedding@16, clip_mean_embedding@32, clip_mean_embedding@64 ...

## Sonuc Yorumu

- matched tabular baseline'a gore MAE tarafinda 318.52 TRY iyilesme var; EfficientNet 6-image referansina gore MAE 174.17 TRY daha iyi; all-image EfficientNet referansina gore RMSE 313.35 kadar daha iyi; image branch ablasyonda MAE 1420.77 kadar kotulestigi icin CLIP image sinyali modele net pozitif katki veriyor.