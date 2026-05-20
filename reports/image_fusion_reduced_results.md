# Reduced Image Fusion Results

## Ozet

- Multimodal source: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- Image embedding source: `E:\rent-agent\dataset\image_embeddings.parquet`
- Kaydedilen model bundle: `E:\rent-agent\models\image_fusion_reduced_model.joblib`
- Join sonucu kalan ornek sayisi: **6,389**
- Original image embedding dimension: **1280**
- Reducer: **PCA**
- Denenen reduced image_dim degerleri: **16, 32, 64, 128, 256**
- En iyi kombinasyon: **XGBRegressor + image_dim=16**

## Validation Leaderboard

| reducer | image_dim | model | validation_mae | validation_rmse | validation_r2 | validation_mape |
| --- | --- | --- | --- | --- | --- | --- |
| PCA | 16 | XGBRegressor | 4,778.9070 | 6,903.9670 | 0.8195 | 14.5308 |
| PCA | 32 | XGBRegressor | 4,802.3863 | 6,855.9902 | 0.8220 | 14.6764 |
| PCA | 64 | XGBRegressor | 4,862.1306 | 7,117.8142 | 0.8081 | 14.8104 |
| PCA | 128 | XGBRegressor | 4,905.8745 | 7,192.0432 | 0.8041 | 14.8842 |
| PCA | 16 | LightGBMRegressor | 4,928.4696 | 7,456.3006 | 0.7895 | 14.6731 |
| PCA | 32 | LightGBMRegressor | 4,945.3508 | 7,316.3928 | 0.7973 | 14.9019 |
| PCA | 64 | LightGBMRegressor | 5,030.4421 | 7,602.0284 | 0.7812 | 15.1584 |
| PCA | 128 | LightGBMRegressor | 5,055.6615 | 7,596.1457 | 0.7815 | 15.1639 |
| PCA | 16 | HistGradientBoostingRegressor | 5,061.6654 | 7,545.3875 | 0.7844 | 15.0327 |
| PCA | 256 | XGBRegressor | 5,077.6046 | 7,389.5541 | 0.7932 | 15.3569 |
| PCA | 32 | HistGradientBoostingRegressor | 5,082.4297 | 7,581.3614 | 0.7823 | 15.2128 |
| PCA | 64 | HistGradientBoostingRegressor | 5,092.0105 | 7,558.7616 | 0.7836 | 15.3948 |
| PCA | 256 | LightGBMRegressor | 5,104.9757 | 7,652.3878 | 0.7782 | 15.3358 |
| PCA | 16 | RandomForestRegressor | 5,179.6960 | 7,715.4604 | 0.7746 | 15.8379 |
| PCA | 256 | HistGradientBoostingRegressor | 5,193.9404 | 7,804.3769 | 0.7694 | 15.5899 |
| PCA | 128 | HistGradientBoostingRegressor | 5,212.0464 | 7,840.8503 | 0.7672 | 15.6381 |
| PCA | 32 | RandomForestRegressor | 5,271.3587 | 7,836.4326 | 0.7675 | 16.2288 |
| PCA | 64 | RandomForestRegressor | 5,409.6120 | 8,135.9550 | 0.7493 | 16.4877 |
| PCA | 128 | RandomForestRegressor | 5,491.8258 | 8,232.6739 | 0.7433 | 16.7362 |
| PCA | 256 | RandomForestRegressor | 5,564.2510 | 8,239.7024 | 0.7429 | 16.9864 |
| PCA | 128 | Ridge | 5,784.2284 | 20,891.0743 | -0.6527 | 17.5947 |
| PCA | 64 | Ridge | 5,806.3186 | 21,079.1220 | -0.6826 | 17.7140 |
| PCA | 32 | Ridge | 5,824.9114 | 21,202.5933 | -0.7024 | 17.6196 |
| PCA | 256 | Ridge | 5,848.6351 | 20,694.7466 | -0.6218 | 17.8800 |
| PCA | 16 | Ridge | 5,860.9771 | 21,113.8541 | -0.6881 | 17.7409 |
| PCA | 128 | MLPRegressor | 5,884.3763 | 9,050.7534 | 0.6898 | 18.1450 |
| PCA | 256 | MLPRegressor | 6,381.6089 | 19,369.6139 | -0.4207 | 19.8173 |
| PCA | 64 | MLPRegressor | 6,690.9806 | 50,748.6412 | -8.7527 | 20.0993 |
| PCA | 32 | MLPRegressor | 6,998.4499 | 31,410.9866 | -2.7363 | 21.5719 |
| PCA | 16 | MLPRegressor | 8,077.7476 | 73,160.0255 | -19.2686 | 24.5177 |

## Image Dim Bazli Skorlar

| image_dim | best_model | best_validation_mae | best_validation_rmse | best_validation_r2 | best_validation_mape |
| --- | --- | --- | --- | --- | --- |
| 16 | XGBRegressor | 4,778.9070 | 6,903.9670 | 0.8195 | 14.5308 |
| 32 | XGBRegressor | 4,802.3863 | 6,855.9902 | 0.8220 | 14.6764 |
| 64 | XGBRegressor | 4,862.1306 | 7,117.8142 | 0.8081 | 14.8104 |
| 128 | XGBRegressor | 4,905.8745 | 7,192.0432 | 0.8041 | 14.8842 |
| 256 | XGBRegressor | 5,077.6046 | 7,389.5541 | 0.7932 | 15.3569 |

## Test Sonuclari

| Metric | Value |
| --- | --- |
| MAE | 4,555.73 |
| RMSE | 6,842.20 |
| R2 | 0.8133 |
| MAPE (%) | 13.53 |

## Matched Baseline vs Reduced Image Fusion

| row | mae | rmse | r2 | mape |
| --- | --- | --- | --- | --- |
| Matched tabular baseline | 4,700.0787 | 7,057.7587 | 0.8013 | 13.7867 |
| Reduced image fusion model | 4,555.7339 | 6,842.1991 | 0.8133 | 13.5254 |
| Improvement vs matched baseline | 144.3448 | 215.5596 | 0.0120 | 0.2613 |

## District Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Akyurt | 5 | 3,156.1552 | 2,091.5535 | 14.7552 | 11.9552 | 1,064.6017 | 2.8000 |
| Çankaya | 379 | 6,277.1725 | 5,871.1835 | 15.2434 | 14.3588 | 405.9890 | 0.8846 |
| Keçiören | 142 | 3,827.3778 | 3,572.0221 | 13.6167 | 12.6746 | 255.3557 | 0.9421 |
| Gölbaşı | 56 | 3,835.0616 | 3,705.6173 | 11.8777 | 11.2858 | 129.4442 | 0.5919 |
| Altındağ | 33 | 4,002.8446 | 3,899.8852 | 15.8675 | 16.5150 | 102.9594 | -0.6475 |
| Etimesgut | 82 | 3,632.4278 | 3,568.9118 | 11.7475 | 11.6481 | 63.5160 | 0.0994 |
| Polatlı | 12 | 3,214.2595 | 3,352.4233 | 16.4078 | 17.1717 | -138.1637 | -0.7639 |
| Çubuk | 15 | 2,087.9365 | 2,277.2395 | 13.8520 | 15.6063 | -189.3030 | -1.7543 |
| Sincan | 58 | 3,421.4291 | 3,629.1515 | 13.3003 | 14.1152 | -207.7224 | -0.8149 |
| Yenimahalle | 90 | 4,086.4141 | 4,361.6933 | 11.4672 | 12.1278 | -275.2792 | -0.6606 |
| Mamak | 75 | 2,936.1545 | 3,246.9007 | 10.8235 | 12.1435 | -310.7462 | -1.3200 |
| Pursaklar | 10 | 5,075.2632 | 6,255.4289 | 15.7168 | 19.1866 | -1,180.1657 | -3.4698 |

## Price Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-20k TRY | 84 | 4,130.3212 | 4,429.8552 | 24.3314 | 26.1956 | -299.5340 | -1.8643 |
| 20k-30k TRY | 372 | 2,905.2472 | 2,857.9583 | 11.9290 | 11.7334 | 47.2889 | 0.1956 |
| 30k-40k TRY | 266 | 4,224.3421 | 3,969.7934 | 12.4604 | 11.7172 | 254.5487 | 0.7431 |
| 40k-50k TRY | 122 | 5,594.9134 | 5,657.6355 | 12.8582 | 13.0058 | -62.7221 | -0.1476 |
| 50k-75k TRY | 84 | 8,840.6085 | 7,680.5396 | 15.2758 | 13.2242 | 1,160.0689 | 2.0516 |
| 75k+ TRY | 31 | 17,122.9349 | 17,494.1482 | 18.5068 | 19.0737 | -371.2133 | -0.5669 |

## m2 Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-75 m2 | 152 | 3,504.8093 | 3,496.2948 | 12.3153 | 12.5759 | 8.5145 | -0.2607 |
| 75-100 m2 | 150 | 4,117.2768 | 4,172.7657 | 14.2892 | 14.6118 | -55.4889 | -0.3226 |
| 100-125 m2 | 273 | 3,954.3141 | 3,911.5228 | 13.6953 | 13.6042 | 42.7913 | 0.0911 |
| 125-150 m2 | 235 | 4,205.7464 | 3,977.5950 | 12.9367 | 12.2506 | 228.1515 | 0.6861 |
| 150-200 m2 | 103 | 8,025.6704 | 7,624.0882 | 17.2320 | 16.4831 | 401.5823 | 0.7490 |
| 200+ m2 | 46 | 10,055.0157 | 9,211.6419 | 14.1807 | 12.5431 | 843.3738 | 1.6376 |

## Image Branch Katkisi

- Matched tabular baseline MAE: **4700.08**
- Reduced image fusion MAE: **4555.73**
- Reduced image block sifirlandiginda MAE: **4794.17**
- Reduced image block sifirlandiginda MAPE: **14.20%**
- Ablasyon yorumu: Reduced image block standardized ve PCA ile olusturuldugu icin ablasyon testinde son image block sifira cekildi; bu yaklasim reduced image branch katkisini ayirmak icin kullanildi.
- Yorum: Reduced image block net pozitif katkili; block sifirlandiginda MAE 238.43 kadar kotulesiyor.

## En Yuksek Hata Yapan 20 Ilan

| listing_id | district | neighborhood | rooms | m2_gross | actual_price_try | baseline_prediction | fusion_prediction | fusion_residual | fusion_abs_error | fusion_ape_pct | abs_error_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hepsiemlak:149516-72 | Çankaya | Gaziosmanpaşa Mah. | 2+1 | 100.0000 | 99,000.0000 | 56,922.9254 | 45,657.3438 | -53,342.6562 | 53,342.6562 | 53.8815 | -11,265.5816 |
| hepsiemlak:162654-112 | Çankaya | Çiğdem Mah. | 4+1 | 200.0000 | 120,000.0000 | 74,196.3756 | 68,322.9297 | -51,677.0703 | 51,677.0703 | 43.0642 | -5,873.4459 |
| hepsiemlak:145733-213 | Çankaya | Cevizlidere Mah. | 5+1 | 375.0000 | 78,000.0000 | 96,425.8913 | 126,656.9609 | 48,656.9609 | 48,656.9609 | 62.3807 | -30,231.0696 |
| hepsiemlak:5665-1536 | Çankaya | Namık Kemal Mah. | 3+0 | 139.0000 | 83,000.0000 | 57,041.3502 | 45,700.5820 | -37,299.4180 | 37,299.4180 | 44.9391 | -11,340.7682 |
| hepsiemlak:43014-3617 | Çankaya | Oran Mah. | 4+1 | 175.0000 | 180,000.0000 | 128,045.0401 | 146,613.3438 | -33,386.6562 | 33,386.6562 | 18.5481 | 18,568.3037 |
| hepsiemlak:0-46238403 | Çankaya | Kavaklıdere Mah. | 3+2 | 150.0000 | 19,000.0000 | 55,611.1481 | 48,616.6445 | 29,616.6445 | 29,616.6445 | 155.8771 | 6,994.5036 |
| hepsiemlak:130551-1404 | Çankaya | Yukarı Dikmen Mah. | 5+1 | 305.0000 | 149,000.0000 | 130,170.0458 | 119,968.9766 | -29,031.0234 | 29,031.0234 | 19.4839 | -10,201.0692 |
| hepsiemlak:5978-6315 | Çankaya | Cevizlidere Mah. | 4+1 | 190.0000 | 80,000.0000 | 46,478.5771 | 51,863.2148 | -28,136.7852 | 28,136.7852 | 35.1710 | 5,384.6377 |
| hepsiemlak:165177-42 | Çankaya | Prof. Dr. Ahmet Taner Kışlalı Mah. | 4+1 | 200.0000 | 93,000.0000 | 62,194.5871 | 64,992.1992 | -28,007.8008 | 28,007.8008 | 30.1159 | 2,797.6121 |
| hepsiemlak:7677-4917 | Çankaya | Mustafa Kemal Mah. | 3+1 | 145.0000 | 75,000.0000 | 42,669.2486 | 47,461.6328 | -27,538.3672 | 27,538.3672 | 36.7178 | 4,792.3842 |
| hepsiemlak:96373-1507 | Çankaya | Nasuh Akar Mah. | 4+1 | 175.0000 | 75,000.0000 | 66,813.3266 | 50,026.2852 | -24,973.7148 | 24,973.7148 | 33.2983 | -16,787.0415 |
| hepsiemlak:57012-7524 | Etimesgut | Yapracık Mah. | 4+1 | 210.0000 | 70,000.0000 | 45,262.2050 | 45,491.4961 | -24,508.5039 | 24,508.5039 | 35.0121 | 229.2911 |
| hepsiemlak:40409-5765 | Çankaya | Yaşamkent Mah. | 4+1 | 185.0000 | 83,000.0000 | 60,659.2518 | 60,376.7773 | -22,623.2227 | 22,623.2227 | 27.2569 | -282.4744 |
| hepsiemlak:61156-6048 | Çankaya | Büyükesat Mah. | 2+1 | 170.0000 | 40,000.0000 | 62,619.8139 | 61,799.0469 | 21,799.0469 | 21,799.0469 | 54.4976 | 820.7670 |
| hepsiemlak:16602-1909 | Çankaya | Ayrancı Mah. | 1+1 | 50.0000 | 52,000.0000 | 34,228.3998 | 30,786.0684 | -21,213.9316 | 21,213.9316 | 40.7960 | -3,442.3314 |
| hepsiemlak:56844-1171 | Çankaya | Remzi Oğuz Arık Mah. | 1+1 | 85.0000 | 52,000.0000 | 25,329.4443 | 32,268.7949 | -19,731.2051 | 19,731.2051 | 37.9446 | 6,939.3506 |
| hepsiemlak:18229-1817 | Çankaya | Kavaklıdere Mah. | 3+1 | 145.0000 | 30,000.0000 | 52,931.5772 | 47,651.0664 | 17,651.0664 | 17,651.0664 | 58.8369 | 5,280.5108 |
| hepsiemlak:72233-408 | Çankaya | Ayrancı Mah. | 2+1 | 90.0000 | 60,000.0000 | 45,524.4441 | 42,520.2500 | -17,479.7500 | 17,479.7500 | 29.1329 | -3,004.1941 |
| hepsiemlak:8315-16683 | Çankaya | Kızılırmak Mah. | 4+1 | 185.0000 | 85,000.0000 | 59,649.0568 | 67,786.7656 | -17,213.2344 | 17,213.2344 | 20.2509 | 8,137.7088 |
| hepsiemlak:37602-2791 | Etimesgut | Aşağıyurtçu Mah. | 4+1 | 180.0000 | 42,000.0000 | 63,423.8385 | 58,772.7031 | 16,772.7031 | 16,772.7031 | 39.9350 | 4,651.1353 |

## LightGBM ve XGBoost Durumu

- `LightGBMRegressor`: calisti; validation denemelerinde image_dim=16, 32, 64, 128, 256 icin egitildi.
- `XGBRegressor`: calisti; validation denemelerinde image_dim=16, 32, 64, 128, 256 icin egitildi.