# All-Image Reduced Fusion Results

## Ozet

- Multimodal source: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- All-image embedding source: `E:\rent-agent\dataset\image_embeddings_all.parquet`
- Kaydedilen model bundle: `E:\rent-agent\models\image_fusion_all_reduced_model.joblib`
- Join sonucu kalan ornek sayisi: **6,389**
- Original image embedding dimension: **1280**
- Reducer: **PCA**
- Denenen reduced image_dim degerleri: **16, 32, 64, 128, 256**
- En iyi kombinasyon: **XGBRegressor + image_dim=16**
- used_image_count ortalamasi: **15.56**
- used_image_count medyani: **16.00**
- used_image_count araligi: **3 - 17**

## Validation Leaderboard

| reducer | image_dim | model | validation_mae | validation_rmse | validation_r2 | validation_mape |
| --- | --- | --- | --- | --- | --- | --- |
| PCA | 16 | XGBRegressor | 4,667.1263 | 6,848.5944 | 0.8224 | 14.1688 |
| PCA | 32 | XGBRegressor | 4,673.3004 | 6,859.7136 | 0.8218 | 14.1732 |
| PCA | 128 | XGBRegressor | 4,758.9967 | 7,080.6124 | 0.8101 | 14.3805 |
| PCA | 64 | XGBRegressor | 4,774.7208 | 7,035.6926 | 0.8125 | 14.3366 |
| PCA | 16 | HistGradientBoostingRegressor | 4,811.6144 | 7,095.8560 | 0.8093 | 14.5261 |
| PCA | 16 | LightGBMRegressor | 4,813.2277 | 7,159.6677 | 0.8059 | 14.3922 |
| PCA | 128 | LightGBMRegressor | 4,830.1733 | 7,267.6863 | 0.8000 | 14.3773 |
| PCA | 64 | HistGradientBoostingRegressor | 4,832.3616 | 7,193.4984 | 0.8040 | 14.5482 |
| PCA | 64 | LightGBMRegressor | 4,836.9950 | 7,192.2585 | 0.8041 | 14.4842 |
| PCA | 32 | LightGBMRegressor | 4,891.6078 | 7,238.9369 | 0.8016 | 14.6001 |
| PCA | 128 | HistGradientBoostingRegressor | 4,905.2706 | 7,333.7369 | 0.7963 | 14.6670 |
| PCA | 256 | XGBRegressor | 4,912.3433 | 7,269.9180 | 0.7999 | 14.7598 |
| PCA | 32 | HistGradientBoostingRegressor | 4,931.1316 | 7,284.0660 | 0.7991 | 14.7538 |
| PCA | 256 | LightGBMRegressor | 4,956.9107 | 7,434.5764 | 0.7907 | 14.8749 |
| PCA | 256 | HistGradientBoostingRegressor | 5,014.5366 | 7,428.9798 | 0.7910 | 15.0909 |
| PCA | 16 | RandomForestRegressor | 5,146.1569 | 7,660.8572 | 0.7778 | 15.6454 |
| PCA | 32 | RandomForestRegressor | 5,238.3568 | 7,813.4064 | 0.7688 | 15.9527 |
| PCA | 64 | RandomForestRegressor | 5,332.2488 | 8,039.3570 | 0.7553 | 16.1930 |
| PCA | 128 | RandomForestRegressor | 5,357.7674 | 8,113.5669 | 0.7507 | 16.3233 |
| PCA | 256 | RandomForestRegressor | 5,481.0433 | 8,284.6021 | 0.7401 | 16.6749 |
| PCA | 64 | Ridge | 5,649.2889 | 20,494.5424 | -0.5906 | 17.2013 |
| PCA | 128 | Ridge | 5,649.5804 | 19,503.4962 | -0.4405 | 17.2175 |
| PCA | 32 | Ridge | 5,692.8414 | 20,360.9874 | -0.5699 | 17.2925 |
| PCA | 16 | Ridge | 5,771.4789 | 21,009.2156 | -0.6715 | 17.5078 |
| PCA | 256 | Ridge | 5,874.9903 | 20,851.7335 | -0.6465 | 17.9637 |
| PCA | 128 | MLPRegressor | 6,229.2585 | 13,475.0632 | 0.3124 | 19.2431 |
| PCA | 256 | MLPRegressor | 6,556.2680 | 19,137.4750 | -0.3869 | 20.3628 |
| PCA | 32 | MLPRegressor | 6,667.4241 | 25,909.0039 | -1.5420 | 20.4134 |
| PCA | 64 | MLPRegressor | 6,677.1973 | 47,053.7654 | -7.3842 | 19.8420 |
| PCA | 16 | MLPRegressor | 7,548.9755 | 55,328.9962 | -10.5926 | 23.4180 |

## Test Sonuclari

| Metric | Value |
| --- | --- |
| MAE | 4,570.65 |
| RMSE | 6,673.16 |
| R2 | 0.8224 |
| MAPE (%) | 13.59 |

## Matched Baseline vs All-Image Fusion

| row | mae | rmse | r2 | mape |
| --- | --- | --- | --- | --- |
| Matched tabular baseline | 4,700.0787 | 7,057.7587 | 0.8013 | 13.7867 |
| All-image reduced fusion model | 4,570.6515 | 6,673.1618 | 0.8224 | 13.5878 |
| Improvement vs matched baseline | 129.4272 | 384.5968 | 0.0211 | 0.1989 |

## 6-Image vs All-Image Fusion

| row | mae | rmse | r2 | mape |
| --- | --- | --- | --- | --- |
| 6-image reduced fusion reference | 4,555.7300 | 6,842.2000 | 0.8133 | 13.5300 |
| All-image reduced fusion model | 4,570.6515 | 6,673.1618 | 0.8224 | 13.5878 |
| Improvement vs 6-image reduced fusion | -14.9215 | 169.0382 | 0.0091 | -0.0578 |

## District Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Çankaya | 379 | 6,277.1725 | 5,891.8698 | 15.2434 | 14.4641 | 385.3027 | 0.7793 |
| Polatlı | 12 | 3,214.2595 | 2,905.5920 | 16.4078 | 14.9246 | 308.6676 | 1.4831 |
| Gölbaşı | 56 | 3,835.0616 | 3,598.3863 | 11.8777 | 11.0751 | 236.6752 | 0.8026 |
| Keçiören | 142 | 3,827.3778 | 3,710.3632 | 13.6167 | 13.0554 | 117.0146 | 0.5613 |
| Akyurt | 5 | 3,156.1552 | 3,134.0633 | 14.7552 | 17.8891 | 22.0919 | -3.1340 |
| Sincan | 58 | 3,421.4291 | 3,429.9151 | 13.3003 | 13.4511 | -8.4860 | -0.1507 |
| Etimesgut | 82 | 3,632.4278 | 3,666.9853 | 11.7475 | 12.0556 | -34.5575 | -0.3081 |
| Çubuk | 15 | 2,087.9365 | 2,148.9790 | 13.8520 | 15.0849 | -61.0425 | -1.2329 |
| Altındağ | 33 | 4,002.8446 | 4,110.6313 | 15.8675 | 17.0063 | -107.7866 | -1.1388 |
| Mamak | 75 | 2,936.1545 | 3,152.9536 | 10.8235 | 11.6076 | -216.7991 | -0.7841 |
| Yenimahalle | 90 | 4,086.4141 | 4,418.8527 | 11.4672 | 12.4376 | -332.4386 | -0.9704 |
| Pursaklar | 10 | 5,075.2632 | 5,438.6951 | 15.7168 | 16.5292 | -363.4319 | -0.8124 |

## Price Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-20k TRY | 84 | 4,130.3212 | 4,309.6505 | 24.3314 | 25.5096 | -179.3293 | -1.1783 |
| 20k-30k TRY | 372 | 2,905.2472 | 2,866.2704 | 11.9290 | 11.7623 | 38.9768 | 0.1667 |
| 30k-40k TRY | 266 | 4,224.3421 | 4,123.6642 | 12.4604 | 12.1812 | 100.6779 | 0.2791 |
| 40k-50k TRY | 122 | 5,594.9134 | 5,738.1905 | 12.8582 | 13.1804 | -143.2771 | -0.3223 |
| 50k-75k TRY | 84 | 8,840.6085 | 7,689.7471 | 15.2758 | 13.1807 | 1,150.8614 | 2.0952 |
| 75k+ TRY | 31 | 17,122.9349 | 16,519.3174 | 18.5068 | 17.9660 | 603.6175 | 0.5408 |

## m2 Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-75 m2 | 152 | 3,504.8093 | 3,564.4979 | 12.3153 | 12.8096 | -59.6886 | -0.4943 |
| 75-100 m2 | 150 | 4,117.2768 | 4,145.5156 | 14.2892 | 14.6404 | -28.2388 | -0.3512 |
| 100-125 m2 | 273 | 3,954.3141 | 3,925.1674 | 13.6953 | 13.5890 | 29.1467 | 0.1063 |
| 125-150 m2 | 235 | 4,205.7464 | 3,964.7521 | 12.9367 | 12.1781 | 240.9943 | 0.7586 |
| 150-200 m2 | 103 | 8,025.6704 | 7,675.4232 | 17.2320 | 16.8379 | 350.2472 | 0.3941 |
| 200+ m2 | 46 | 10,055.0157 | 9,255.8198 | 14.1807 | 12.6441 | 799.1958 | 1.5365 |

## Used Image Count Analizi

| used_image_count | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | avg_actual_price | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0000 | 1.0000 | 35.1493 | 5,602.1719 | 0.0469 | 7.4696 | 75,000.0000 | -5,567.0226 | -7.4227 |
| 7.0000 | 3.0000 | 3,002.3301 | 3,028.2272 | 12.7850 | 13.4107 | 21,666.6667 | -25.8971 | -0.6257 |
| 8.0000 | 6.0000 | 1,497.8300 | 2,317.1943 | 5.5184 | 8.6843 | 25,458.3333 | -819.3643 | -3.1660 |
| 9.0000 | 7.0000 | 6,240.3328 | 5,745.9612 | 23.1610 | 22.6770 | 30,658.7143 | 494.3716 | 0.4840 |
| 10.0000 | 9.0000 | 4,361.8808 | 4,158.0171 | 12.0770 | 11.9657 | 34,777.7778 | 203.8637 | 0.1113 |
| 11.0000 | 12.0000 | 6,864.4122 | 6,657.5396 | 27.7052 | 25.8124 | 31,750.0000 | 206.8726 | 1.8928 |
| 12.0000 | 15.0000 | 4,028.1996 | 4,452.1462 | 14.5824 | 18.0283 | 29,200.0000 | -423.9466 | -3.4459 |
| 13.0000 | 17.0000 | 4,980.9162 | 4,182.9257 | 14.0008 | 12.5060 | 33,294.1176 | 797.9905 | 1.4948 |
| 14.0000 | 24.0000 | 4,772.5377 | 4,434.7463 | 16.7251 | 15.5438 | 31,770.8333 | 337.7914 | 1.1813 |
| 15.0000 | 28.0000 | 5,047.6699 | 4,653.4454 | 18.6387 | 16.9426 | 28,960.7143 | 394.2245 | 1.6961 |
| 16.0000 | 794.0000 | 4,620.6440 | 4,448.6104 | 13.2148 | 12.9766 | 34,331.9887 | 172.0336 | 0.2382 |
| 17.0000 | 43.0000 | 5,913.2228 | 6,751.4287 | 15.6757 | 16.7634 | 36,546.5116 | -838.2059 | -1.0877 |

## Image Branch Ablation

- Matched tabular baseline MAE: **4700.08**
- All-image reduced fusion MAE: **4570.65**
- Reduced image block sifirlandiginda MAE: **4915.60**
- Reduced image block sifirlandiginda MAPE: **14.59%**
- Ablasyon yorumu: Reduced image block standardized ve PCA ile olusturuldugu icin ablasyon testinde son image block sifira cekildi; bu yaklasim reduced image branch katkisini ayirmak icin kullanildi.

## En Yuksek Hata Yapan 20 Ilan

| listing_id | district | neighborhood | rooms | m2_gross | used_image_count | actual_price_try | baseline_prediction | fusion_prediction | fusion_residual | fusion_abs_error | fusion_ape_pct | abs_error_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hepsiemlak:149516-72 | Çankaya | Gaziosmanpaşa Mah. | 2+1 | 100.0000 | 16 | 99,000.0000 | 56,922.9254 | 46,205.2578 | -52,794.7422 | 52,794.7422 | 53.3280 | -10,717.6676 |
| hepsiemlak:162654-112 | Çankaya | Çiğdem Mah. | 4+1 | 200.0000 | 17 | 120,000.0000 | 74,196.3756 | 69,665.7109 | -50,334.2891 | 50,334.2891 | 41.9452 | -4,530.6647 |
| hepsiemlak:5665-1536 | Çankaya | Namık Kemal Mah. | 3+0 | 139.0000 | 17 | 83,000.0000 | 57,041.3502 | 49,302.0156 | -33,697.9844 | 33,697.9844 | 40.6000 | -7,739.3346 |
| hepsiemlak:5978-6315 | Çankaya | Cevizlidere Mah. | 4+1 | 190.0000 | 16 | 80,000.0000 | 46,478.5771 | 46,921.0859 | -33,078.9141 | 33,078.9141 | 41.3486 | 442.5088 |
| hepsiemlak:0-46238403 | Çankaya | Kavaklıdere Mah. | 3+2 | 150.0000 | 11 | 19,000.0000 | 55,611.1481 | 50,994.8438 | 31,994.8438 | 31,994.8438 | 168.3939 | 4,616.3044 |
| hepsiemlak:130551-1404 | Çankaya | Yukarı Dikmen Mah. | 5+1 | 305.0000 | 16 | 149,000.0000 | 130,170.0458 | 117,085.6562 | -31,914.3438 | 31,914.3438 | 21.4190 | -13,084.3896 |
| hepsiemlak:57012-7524 | Etimesgut | Yapracık Mah. | 4+1 | 210.0000 | 16 | 70,000.0000 | 45,262.2050 | 41,835.5469 | -28,164.4531 | 28,164.4531 | 40.2349 | -3,426.6581 |
| hepsiemlak:7677-4917 | Çankaya | Mustafa Kemal Mah. | 3+1 | 145.0000 | 16 | 75,000.0000 | 42,669.2486 | 47,145.8359 | -27,854.1641 | 27,854.1641 | 37.1389 | 4,476.5874 |
| hepsiemlak:43014-3617 | Çankaya | Oran Mah. | 4+1 | 175.0000 | 16 | 180,000.0000 | 128,045.0401 | 154,122.4062 | -25,877.5938 | 25,877.5938 | 14.3764 | 26,077.3662 |
| hepsiemlak:165177-42 | Çankaya | Prof. Dr. Ahmet Taner Kışlalı Mah. | 4+1 | 200.0000 | 16 | 93,000.0000 | 62,194.5871 | 69,315.8594 | -23,684.1406 | 23,684.1406 | 25.4668 | 7,121.2722 |
| hepsiemlak:96373-1507 | Çankaya | Nasuh Akar Mah. | 4+1 | 175.0000 | 16 | 75,000.0000 | 66,813.3266 | 52,342.7891 | -22,657.2109 | 22,657.2109 | 30.2096 | -14,470.5375 |
| hepsiemlak:61156-6048 | Çankaya | Büyükesat Mah. | 2+1 | 170.0000 | 16 | 40,000.0000 | 62,619.8139 | 62,323.6094 | 22,323.6094 | 22,323.6094 | 55.8090 | 296.2045 |
| hepsiemlak:38897-1578 | Çankaya | Namık Kemal Mah. | 3+1 | 155.0000 | 17 | 85,000.0000 | 93,543.4714 | 65,399.5312 | -19,600.4688 | 19,600.4688 | 23.0594 | -11,056.9973 |
| hepsiemlak:9800-2550 | Çankaya | Emek Mah. | 3+1 | 145.0000 | 16 | 65,000.0000 | 39,001.4751 | 45,676.0586 | -19,323.9414 | 19,323.9414 | 29.7291 | 6,674.5835 |
| hepsiemlak:40409-5765 | Çankaya | Yaşamkent Mah. | 4+1 | 185.0000 | 16 | 83,000.0000 | 60,659.2518 | 64,086.9492 | -18,913.0508 | 18,913.0508 | 22.7868 | 3,427.6975 |
| hepsiemlak:16602-1909 | Çankaya | Ayrancı Mah. | 1+1 | 50.0000 | 11 | 52,000.0000 | 34,228.3998 | 33,870.0469 | -18,129.9531 | 18,129.9531 | 34.8653 | -358.3529 |
| hepsiemlak:56844-1171 | Çankaya | Remzi Oğuz Arık Mah. | 1+1 | 85.0000 | 16 | 52,000.0000 | 25,329.4443 | 33,982.5000 | -18,017.5000 | 18,017.5000 | 34.6490 | 8,653.0557 |
| hepsiemlak:8315-16683 | Çankaya | Kızılırmak Mah. | 4+1 | 185.0000 | 14 | 85,000.0000 | 59,649.0568 | 67,170.9219 | -17,829.0781 | 17,829.0781 | 20.9754 | 7,521.8651 |
| hepsiemlak:16602-1913 | Çankaya | Güvenevler Mah. | 3+1 | 145.0000 | 17 | 60,000.0000 | 52,042.3478 | 42,220.2852 | -17,779.7148 | 17,779.7148 | 29.6329 | -9,822.0626 |
| hepsiemlak:130229-310 | Sincan | Törekent Mah. | 4+1 | 193.0000 | 16 | 38,750.0000 | 46,119.6833 | 55,838.3086 | 17,088.3086 | 17,088.3086 | 44.0989 | -9,718.6253 |

## LightGBM ve XGBoost Durumu

- `XGBRegressor`: calisti; validation denemelerinde image_dim=16, 32, 64, 128, 256 icin egitildi.
- `LightGBMRegressor`: calisti; validation denemelerinde image_dim=16, 32, 64, 128, 256 icin egitildi.

## Sonuc Yorumu

- All-image fusion, matched tabular baseline'i MAE tarafinda 129.43 puan geciyor.
- All-image embeddingler, 6-image reduced fusion referansina gore MAE'yi 14.92 kotulestiriyor.
- Image branch ablasyonu sonrasinda MAE 344.95 kotulesti; reduced image block net pozitif katkili.