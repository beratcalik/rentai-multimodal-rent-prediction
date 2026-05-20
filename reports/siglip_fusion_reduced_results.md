# SigLIP Fusion Reduced Results

## Ozet

- Multimodal source: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- SigLIP embedding source: `E:\rent-agent\dataset\siglip_image_embeddings.parquet`
- Kaydedilen model bundle: `E:\rent-agent\models\siglip_fusion_reduced_model.joblib`
- Join sonucu kalan ornek sayisi: **6,389**
- Denenen representationlar: **siglip_mean_embedding, siglip_max_embedding, siglip_meanmax_embedding**
- Representation dimensionlari: **siglip_mean_embedding=768, siglip_max_embedding=768, siglip_meanmax_embedding=1536**
- Reducer: **PCA**
- Denenen image_dim degerleri: **16, 32, 64, 128**
- En iyi kombinasyon: **siglip_meanmax_embedding + XGBRegressor + image_dim=16**

## Validation Leaderboard

| representation | image_dim | model | validation_mae | validation_rmse | validation_r2 | validation_mape |
| --- | --- | --- | --- | --- | --- | --- |
| siglip_meanmax_embedding | 16 | XGBRegressor | 4,483.7369 | 6,628.1224 | 0.8336 | 13.6576 |
| siglip_max_embedding | 16 | XGBRegressor | 4,488.7337 | 6,442.6613 | 0.8428 | 13.5995 |
| siglip_meanmax_embedding | 32 | XGBRegressor | 4,514.2765 | 6,516.3479 | 0.8392 | 13.7258 |
| siglip_max_embedding | 32 | XGBRegressor | 4,517.8728 | 6,591.0432 | 0.8355 | 13.6723 |
| siglip_mean_embedding | 32 | XGBRegressor | 4,519.9347 | 6,487.3209 | 0.8406 | 13.8113 |
| siglip_max_embedding | 64 | XGBRegressor | 4,524.6300 | 6,480.7740 | 0.8410 | 13.7541 |
| siglip_meanmax_embedding | 128 | XGBRegressor | 4,537.8138 | 6,551.4924 | 0.8375 | 13.8492 |
| siglip_mean_embedding | 16 | XGBRegressor | 4,538.6928 | 6,569.8548 | 0.8365 | 13.7962 |
| siglip_mean_embedding | 64 | LightGBMRegressor | 4,551.0916 | 6,585.4970 | 0.8358 | 13.7296 |
| siglip_mean_embedding | 64 | XGBRegressor | 4,566.1900 | 6,687.2552 | 0.8307 | 13.8101 |
| siglip_mean_embedding | 128 | XGBRegressor | 4,569.3310 | 6,497.9741 | 0.8401 | 13.9918 |
| siglip_meanmax_embedding | 64 | XGBRegressor | 4,587.7136 | 6,594.7046 | 0.8353 | 13.8783 |
| siglip_meanmax_embedding | 64 | LightGBMRegressor | 4,621.9265 | 6,855.0738 | 0.8220 | 13.9125 |
| siglip_max_embedding | 128 | XGBRegressor | 4,631.9613 | 6,807.4847 | 0.8245 | 13.9552 |
| siglip_meanmax_embedding | 16 | LightGBMRegressor | 4,638.6126 | 6,928.9684 | 0.8182 | 13.8240 |
| siglip_mean_embedding | 16 | LightGBMRegressor | 4,649.8804 | 6,839.4945 | 0.8229 | 13.9479 |
| siglip_max_embedding | 64 | LightGBMRegressor | 4,654.0997 | 6,754.4513 | 0.8272 | 13.9945 |
| siglip_meanmax_embedding | 64 | HistGradientBoostingRegressor | 4,658.4848 | 6,926.9240 | 0.8183 | 14.0637 |
| siglip_mean_embedding | 64 | HistGradientBoostingRegressor | 4,662.7695 | 6,827.8605 | 0.8235 | 13.9300 |
| siglip_mean_embedding | 128 | LightGBMRegressor | 4,685.7318 | 6,747.1000 | 0.8276 | 14.1632 |
| siglip_mean_embedding | 16 | HistGradientBoostingRegressor | 4,688.2984 | 6,909.5100 | 0.8192 | 14.0853 |
| siglip_meanmax_embedding | 32 | LightGBMRegressor | 4,689.4507 | 6,973.5007 | 0.8158 | 14.1019 |
| siglip_meanmax_embedding | 128 | LightGBMRegressor | 4,693.9281 | 6,911.1233 | 0.8191 | 14.1264 |
| siglip_mean_embedding | 32 | LightGBMRegressor | 4,708.0117 | 6,779.2492 | 0.8260 | 14.1626 |
| siglip_max_embedding | 16 | LightGBMRegressor | 4,718.6412 | 6,880.2084 | 0.8207 | 14.1108 |
| siglip_max_embedding | 128 | LightGBMRegressor | 4,725.7279 | 6,846.2072 | 0.8225 | 14.2732 |
| siglip_mean_embedding | 32 | HistGradientBoostingRegressor | 4,728.5044 | 6,965.5314 | 0.8163 | 14.2197 |
| siglip_meanmax_embedding | 16 | HistGradientBoostingRegressor | 4,729.3855 | 7,063.0709 | 0.8111 | 14.0509 |
| siglip_max_embedding | 64 | HistGradientBoostingRegressor | 4,739.3675 | 6,944.4352 | 0.8174 | 14.2142 |
| siglip_max_embedding | 32 | LightGBMRegressor | 4,748.4752 | 7,068.4615 | 0.8108 | 14.1859 |
| siglip_meanmax_embedding | 32 | HistGradientBoostingRegressor | 4,756.5183 | 7,083.6085 | 0.8100 | 14.2193 |
| siglip_max_embedding | 32 | HistGradientBoostingRegressor | 4,764.6414 | 7,049.6520 | 0.8118 | 14.1962 |
| siglip_max_embedding | 16 | HistGradientBoostingRegressor | 4,771.4655 | 6,860.8021 | 0.8218 | 14.2396 |
| siglip_mean_embedding | 128 | HistGradientBoostingRegressor | 4,778.4426 | 6,823.2926 | 0.8237 | 14.5004 |
| siglip_max_embedding | 128 | HistGradientBoostingRegressor | 4,782.0668 | 7,013.5352 | 0.8137 | 14.3852 |
| siglip_meanmax_embedding | 128 | HistGradientBoostingRegressor | 4,798.9983 | 6,994.8036 | 0.8147 | 14.3451 |

## Representation Bazli Skorlar

| representation | best_model | best_image_dim | best_validation_mae | best_validation_rmse | best_validation_r2 | best_validation_mape |
| --- | --- | --- | --- | --- | --- | --- |
| siglip_mean_embedding | XGBRegressor | 32 | 4,519.9347 | 6,487.3209 | 0.8406 | 13.8113 |
| siglip_max_embedding | XGBRegressor | 16 | 4,488.7337 | 6,442.6613 | 0.8428 | 13.5995 |
| siglip_meanmax_embedding | XGBRegressor | 16 | 4,483.7369 | 6,628.1224 | 0.8336 | 13.6576 |

## Image Dim Bazli Skorlar

| image_dim | best_representation | best_model | best_validation_mae | best_validation_rmse | best_validation_r2 | best_validation_mape |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | siglip_meanmax_embedding | XGBRegressor | 4,483.7369 | 6,628.1224 | 0.8336 | 13.6576 |
| 32 | siglip_meanmax_embedding | XGBRegressor | 4,514.2765 | 6,516.3479 | 0.8392 | 13.7258 |
| 64 | siglip_max_embedding | XGBRegressor | 4,524.6300 | 6,480.7740 | 0.8410 | 13.7541 |
| 128 | siglip_meanmax_embedding | XGBRegressor | 4,537.8138 | 6,551.4924 | 0.8375 | 13.8492 |

## Final Test Sonuclari

| Metric | Value |
| --- | --- |
| MAE | 4,441.86 |
| RMSE | 6,484.50 |
| R2 | 0.8323 |
| MAPE (%) | 12.98 |

## EfficientNet vs CLIP vs SigLIP Karsilastirmasi

| row | mae | rmse | r2 | mape |
| --- | --- | --- | --- | --- |
| Matched tabular baseline | 4,700.0787 | 7,057.7587 | 0.8013 | 13.7867 |
| EfficientNet 6-image reduced fusion | 4,555.7300 | 6,842.2000 | 0.8133 | 13.5300 |
| CLIP reduced fusion | 4,381.5600 | 6,359.8100 | 0.8387 | 13.1000 |
| SigLIP reduced fusion | 4,441.8572 | 6,484.5027 | 0.8323 | 12.9846 |
| Improvement vs matched baseline | 258.2214 | 573.2560 | 0.0310 | 0.8022 |
| Improvement vs EfficientNet 6-image | 113.8728 | 357.6973 | 0.0190 | 0.5454 |
| Improvement vs CLIP | -60.2972 | -124.6927 | -0.0064 | 0.1154 |

## District Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Akyurt | 5 | 3,156.1552 | 1,244.5117 | 14.7552 | 7.7779 | 1,911.6435 | 6.9773 |
| Altındağ | 33 | 4,002.8446 | 3,374.8455 | 15.8675 | 13.1034 | 627.9991 | 2.7641 |
| Çankaya | 379 | 6,277.1725 | 5,820.0976 | 15.2434 | 14.1942 | 457.0749 | 1.0491 |
| Pursaklar | 10 | 5,075.2632 | 4,660.9393 | 15.7168 | 14.5298 | 414.3239 | 1.1870 |
| Etimesgut | 82 | 3,632.4278 | 3,316.1223 | 11.7475 | 11.1048 | 316.3055 | 0.6427 |
| Gölbaşı | 56 | 3,835.0616 | 3,577.2426 | 11.8777 | 10.6193 | 257.8189 | 1.2584 |
| Sincan | 58 | 3,421.4291 | 3,166.1896 | 13.3003 | 12.5381 | 255.2395 | 0.7622 |
| Keçiören | 142 | 3,827.3778 | 3,658.7236 | 13.6167 | 12.7771 | 168.6542 | 0.8396 |
| Çubuk | 15 | 2,087.9365 | 1,984.1382 | 13.8520 | 13.3120 | 103.7983 | 0.5400 |
| Mamak | 75 | 2,936.1545 | 3,003.4617 | 10.8235 | 10.7975 | -67.3072 | 0.0261 |
| Yenimahalle | 90 | 4,086.4141 | 4,442.1275 | 11.4672 | 12.3108 | -355.7134 | -0.8437 |
| Polatlı | 12 | 3,214.2595 | 3,578.2415 | 16.4078 | 15.7703 | -363.9820 | 0.6374 |

## Price Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-20k TRY | 84 | 4,130.3212 | 3,739.5123 | 24.3314 | 21.7518 | 390.8089 | 2.5796 |
| 20k-30k TRY | 372 | 2,905.2472 | 2,770.7532 | 11.9290 | 11.3096 | 134.4940 | 0.6193 |
| 30k-40k TRY | 266 | 4,224.3421 | 4,011.5268 | 12.4604 | 11.8091 | 212.8153 | 0.6512 |
| 40k-50k TRY | 122 | 5,594.9134 | 5,677.5157 | 12.8582 | 13.0658 | -82.6023 | -0.2076 |
| 50k-75k TRY | 84 | 8,840.6085 | 7,740.3078 | 15.2758 | 13.3418 | 1,100.3007 | 1.9341 |
| 75k+ TRY | 31 | 17,122.9349 | 16,290.0959 | 18.5068 | 18.1250 | 832.8390 | 0.3818 |

## m2 Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-75 m2 | 152 | 3,504.8093 | 3,310.4607 | 12.3153 | 11.6502 | 194.3486 | 0.6651 |
| 75-100 m2 | 150 | 4,117.2768 | 3,716.4295 | 14.2892 | 12.9935 | 400.8473 | 1.2958 |
| 100-125 m2 | 273 | 3,954.3141 | 3,771.6593 | 13.6953 | 12.8789 | 182.6548 | 0.8163 |
| 125-150 m2 | 235 | 4,205.7464 | 3,882.0844 | 12.9367 | 11.8572 | 323.6620 | 1.0795 |
| 150-200 m2 | 103 | 8,025.6704 | 7,873.2177 | 17.2320 | 17.2621 | 152.4528 | -0.0300 |
| 200+ m2 | 46 | 10,055.0157 | 9,699.8349 | 14.1807 | 14.1733 | 355.1808 | 0.0073 |

## Used Image Count Analizi

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-4 images | 1 | 35.1493 | 7,843.6250 | 0.0469 | 10.4582 | -7,808.4757 | -10.4113 |
| 5-8 images | 9 | 1,999.3300 | 2,049.3785 | 7.9406 | 9.1555 | -50.0485 | -1.2149 |
| 9-12 images | 43 | 5,249.6558 | 4,838.3341 | 19.1167 | 17.2359 | 411.3216 | 1.8808 |
| 13-16 images | 906 | 4,705.9726 | 4,443.0515 | 13.6070 | 12.8236 | 262.9210 | 0.7834 |

## Image Branch Ablation

- Final SigLIP fusion MAE: **4441.86**
- Ablation sonrasi MAE: **5574.62**
- Ablation sonrasi RMSE: **8264.95**
- Ablation sonrasi R2: **0.7275**
- Ablation sonrasi MAPE: **16.88%**
- Ablation notu: Reduced image block standardized ve PCA ile olusturuldugu icin ablasyon testinde son image block sifira cekildi; bu yaklasim reduced image branch katkisini ayirmak icin kullanildi.

## En Yuksek Hata Yapan 20 Ilan

| listing_id | district | neighborhood | rooms | m2_gross | actual_price_try | baseline_prediction | fusion_prediction | fusion_residual | fusion_abs_error | fusion_ape_pct | abs_error_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hepsiemlak:149516-72 | Çankaya | Gaziosmanpaşa Mah. | 2+1 | 100.0000 | 99,000.0000 | 56,922.9254 | 47,262.8359 | -51,737.1641 | 51,737.1641 | 52.2598 | -9,660.0895 |
| hepsiemlak:162654-112 | Çankaya | Çiğdem Mah. | 4+1 | 200.0000 | 120,000.0000 | 74,196.3756 | 78,665.0000 | -41,335.0000 | 41,335.0000 | 34.4458 | 4,468.6244 |
| hepsiemlak:5665-1536 | Çankaya | Namık Kemal Mah. | 3+0 | 139.0000 | 83,000.0000 | 57,041.3502 | 46,732.4922 | -36,267.5078 | 36,267.5078 | 43.6958 | -10,308.8580 |
| hepsiemlak:61156-6048 | Çankaya | Büyükesat Mah. | 2+1 | 170.0000 | 40,000.0000 | 62,619.8139 | 71,805.8906 | 31,805.8906 | 31,805.8906 | 79.5147 | -9,186.0767 |
| hepsiemlak:5978-6315 | Çankaya | Cevizlidere Mah. | 4+1 | 190.0000 | 80,000.0000 | 46,478.5771 | 51,204.3906 | -28,795.6094 | 28,795.6094 | 35.9945 | 4,725.8135 |
| hepsiemlak:0-46238403 | Çankaya | Kavaklıdere Mah. | 3+2 | 150.0000 | 19,000.0000 | 55,611.1481 | 47,062.2305 | 28,062.2305 | 28,062.2305 | 147.6959 | 8,548.9177 |
| hepsiemlak:7677-4917 | Çankaya | Mustafa Kemal Mah. | 3+1 | 145.0000 | 75,000.0000 | 42,669.2486 | 48,360.0859 | -26,639.9141 | 26,639.9141 | 35.5199 | 5,690.8374 |
| hepsiemlak:165177-42 | Çankaya | Prof. Dr. Ahmet Taner Kışlalı Mah. | 4+1 | 200.0000 | 93,000.0000 | 62,194.5871 | 67,866.9922 | -25,133.0078 | 25,133.0078 | 27.0247 | 5,672.4050 |
| hepsiemlak:8315-16683 | Çankaya | Kızılırmak Mah. | 4+1 | 185.0000 | 85,000.0000 | 59,649.0568 | 60,766.5625 | -24,233.4375 | 24,233.4375 | 28.5099 | 1,117.5057 |
| hepsiemlak:156606-360 | Keçiören | Ovacık Mah. | 4+1 | 230.0000 | 50,000.0000 | 63,554.6823 | 72,632.6328 | 22,632.6328 | 22,632.6328 | 45.2653 | -9,077.9505 |
| hepsiemlak:130551-1404 | Çankaya | Yukarı Dikmen Mah. | 5+1 | 305.0000 | 149,000.0000 | 130,170.0458 | 126,504.0156 | -22,495.9844 | 22,495.9844 | 15.0980 | -3,666.0302 |
| hepsiemlak:38897-1578 | Çankaya | Namık Kemal Mah. | 3+1 | 155.0000 | 85,000.0000 | 93,543.4714 | 62,869.0117 | -22,130.9883 | 22,130.9883 | 26.0365 | -13,587.5169 |
| hepsiemlak:131460-334 | Çankaya | Büyükesat Mah. | 3+1 | 160.0000 | 38,000.0000 | 53,603.4707 | 57,667.5039 | 19,667.5039 | 19,667.5039 | 51.7566 | -4,064.0332 |
| hepsiemlak:37602-2791 | Etimesgut | Aşağıyurtçu Mah. | 4+1 | 180.0000 | 42,000.0000 | 63,423.8385 | 61,217.3008 | 19,217.3008 | 19,217.3008 | 45.7555 | 2,206.5377 |
| hepsiemlak:56844-1171 | Çankaya | Remzi Oğuz Arık Mah. | 1+1 | 85.0000 | 52,000.0000 | 25,329.4443 | 33,084.5508 | -18,915.4492 | 18,915.4492 | 36.3759 | 7,755.1064 |
| hepsiemlak:96594-5199 | Çankaya | Alacaatlı Mah. | 4+1 | 210.0000 | 62,000.0000 | 71,033.7614 | 80,645.5703 | 18,645.5703 | 18,645.5703 | 30.0735 | -9,611.8090 |
| hepsiemlak:145733-213 | Çankaya | Cevizlidere Mah. | 5+1 | 375.0000 | 78,000.0000 | 96,425.8913 | 96,298.0312 | 18,298.0312 | 18,298.0312 | 23.4590 | 127.8600 |
| hepsiemlak:2220-9583 | Çankaya | Dodurga Mah. | 4+1 | 270.0000 | 77,000.0000 | 98,159.6172 | 95,117.2969 | 18,117.2969 | 18,117.2969 | 23.5290 | 3,042.3204 |
| hepsiemlak:4666-1389 | Çankaya | Emek Mah. | 3+1 | 150.0000 | 59,000.0000 | 47,116.7151 | 40,903.5469 | -18,096.4531 | 18,096.4531 | 30.6720 | -6,213.1683 |
| hepsiemlak:16602-1909 | Çankaya | Ayrancı Mah. | 1+1 | 50.0000 | 52,000.0000 | 34,228.3998 | 34,147.5742 | -17,852.4258 | 17,852.4258 | 34.3316 | -80.8256 |

## Model Durumu

- `XGBRegressor`: calisti; denenen kombinasyonlar arasinda siglip_max_embedding@128, siglip_max_embedding@16, siglip_max_embedding@32, siglip_max_embedding@64, siglip_mean_embedding@128, siglip_mean_embedding@16, siglip_mean_embedding@32, siglip_mean_embedding@64 ...
- `LightGBMRegressor`: calisti; denenen kombinasyonlar arasinda siglip_max_embedding@128, siglip_max_embedding@16, siglip_max_embedding@32, siglip_max_embedding@64, siglip_mean_embedding@128, siglip_mean_embedding@16, siglip_mean_embedding@32, siglip_mean_embedding@64 ...
- `HistGradientBoostingRegressor`: calisti; denenen kombinasyonlar arasinda siglip_max_embedding@128, siglip_max_embedding@16, siglip_max_embedding@32, siglip_max_embedding@64, siglip_mean_embedding@128, siglip_mean_embedding@16, siglip_mean_embedding@32, siglip_mean_embedding@64 ...

## Sonuc Yorumu

- matched tabular baseline'a gore MAE tarafinda 258.22 TRY iyilesme var; EfficientNet 6-image referansina gore MAE 113.87 TRY daha iyi; CLIP referansina gore MAE 60.30 TRY daha zayif; image branch ablasyonda MAE 1132.76 kadar kotulestigi icin SigLIP image sinyali modele net pozitif katki veriyor.