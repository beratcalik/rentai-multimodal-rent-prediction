# Image Fusion Results

## Ozet

- Multimodal source: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- Image embedding source: `E:\rent-agent\dataset\image_embeddings.parquet`
- Kaydedilen model bundle: `E:\rent-agent\models\image_fusion_model.joblib`
- En iyi model: **HistGradientBoostingRegressor**
- Join sonucu kalan ornek sayisi: **6,389**
- Image embedding dimension: **1280**

## Validation Leaderboard

| model | validation_mae | validation_rmse | validation_r2 | validation_mape |
| --- | --- | --- | --- | --- |
| HistGradientBoostingRegressor | 5,247.0525 | 7,811.8453 | 0.7689 | 15.7942 |
| RandomForestRegressor | 5,790.8379 | 8,618.6311 | 0.7187 | 17.6421 |
| Ridge | 7,352.8095 | 22,171.0639 | -0.8614 | 23.2262 |
| MLPRegressor | 7,367.3713 | 27,838.0800 | -1.9346 | 22.6658 |

## Test Sonuclari

| Metric | Value |
| --- | --- |
| MAE | 4,982.21 |
| RMSE | 7,542.15 |
| R2 | 0.7731 |
| MAPE (%) | 14.57 |

## Baseline vs Image Fusion Karsilastirmasi

| row | mae | rmse | r2 | mape |
| --- | --- | --- | --- | --- |
| Baseline reference (full tabular test) | 4,683.3500 | 6,911.9100 | 0.7632 | 14.8500 |
| Baseline on matched multimodal test subset | 4,700.0787 | 7,057.7587 | 0.8013 | 13.7867 |
| Image fusion model | 4,982.2091 | 7,542.1509 | 0.7731 | 14.5671 |
| Improvement vs matched subset baseline | -282.1304 | -484.3923 | -0.0282 | -0.7804 |
| Improvement vs reference baseline | -298.8591 | -630.2409 | 0.0099 | 0.2829 |

- Not: `Baseline reference`, daha once `train_ready_ml.parquet` uzerinde elde edilen sabit tabular sonucudur.
- Not: `Baseline on matched multimodal test subset`, ayni join ve ayni split uzerinde image embedding kullanmadan yeniden egitilen adil karsilastirma baseline'idir.

## District Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pursaklar | 10 | 5,075.2632 | 4,319.8586 | 15.7168 | 16.0603 | 755.4046 | -0.3435 |
| Akyurt | 5 | 3,156.1552 | 2,817.3088 | 14.7552 | 15.9185 | 338.8464 | -1.1634 |
| Keçiören | 142 | 3,827.3778 | 3,780.3942 | 13.6167 | 12.9033 | 46.9836 | 0.7134 |
| Çubuk | 15 | 2,087.9365 | 2,102.7930 | 13.8520 | 15.4973 | -14.8565 | -1.6453 |
| Altındağ | 33 | 4,002.8446 | 4,139.7847 | 15.8675 | 17.0470 | -136.9401 | -1.1795 |
| Etimesgut | 82 | 3,632.4278 | 3,816.3093 | 11.7475 | 13.0123 | -183.8816 | -1.2648 |
| Gölbaşı | 56 | 3,835.0616 | 4,027.3210 | 11.8777 | 12.2605 | -192.2595 | -0.3828 |
| Çankaya | 379 | 6,277.1725 | 6,495.0990 | 15.2434 | 15.5511 | -217.9265 | -0.3077 |
| Sincan | 58 | 3,421.4291 | 3,909.7592 | 13.3003 | 14.7953 | -488.3301 | -1.4949 |
| Mamak | 75 | 2,936.1545 | 3,534.6510 | 10.8235 | 12.9312 | -598.4965 | -2.1077 |
| Yenimahalle | 90 | 4,086.4141 | 4,906.0413 | 11.4672 | 13.3432 | -819.6272 | -1.8760 |
| Polatlı | 12 | 3,214.2595 | 5,089.7002 | 16.4078 | 24.6266 | -1,875.4407 | -8.2189 |

## Price Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-20k TRY | 84 | 4,130.3212 | 4,634.9086 | 24.3314 | 27.5628 | -504.5874 | -3.2314 |
| 20k-30k TRY | 372 | 2,905.2472 | 3,032.1813 | 11.9290 | 12.5152 | -126.9341 | -0.5862 |
| 30k-40k TRY | 266 | 4,224.3421 | 4,224.0074 | 12.4604 | 12.4887 | 0.3347 | -0.0284 |
| 40k-50k TRY | 122 | 5,594.9134 | 5,987.8912 | 12.8582 | 13.7437 | -392.9778 | -0.8855 |
| 50k-75k TRY | 84 | 8,840.6085 | 9,295.7567 | 15.2758 | 15.9775 | -455.1482 | -0.7017 |
| 75k+ TRY | 31 | 17,122.9349 | 20,183.3057 | 18.5068 | 21.2282 | -3,060.3708 | -2.7214 |

## m2 Range Bazli Improvement

| group | sample_count | baseline_mae | fusion_mae | baseline_mape | fusion_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-75 m2 | 152 | 3,504.8093 | 3,926.1270 | 12.3153 | 14.3112 | -421.3178 | -1.9959 |
| 75-100 m2 | 150 | 4,117.2768 | 4,602.5299 | 14.2892 | 16.1236 | -485.2531 | -1.8343 |
| 100-125 m2 | 273 | 3,954.3141 | 3,919.8809 | 13.6953 | 13.7051 | 34.4332 | -0.0098 |
| 125-150 m2 | 235 | 4,205.7464 | 4,145.1227 | 12.9367 | 12.6738 | 60.6237 | 0.2629 |
| 150-200 m2 | 103 | 8,025.6704 | 8,580.2237 | 17.2320 | 18.2299 | -554.5533 | -0.9978 |
| 200+ m2 | 46 | 10,055.0157 | 12,234.6374 | 14.1807 | 16.9242 | -2,179.6218 | -2.7435 |

## Image Branch Katkisi

- Matched subset baseline MAE: **4700.08**
- Full image fusion MAE: **4982.21**
- Image blok sifirlandiginda MAE: **5779.93**
- Image blok sifirlandiginda MAPE: **17.77%**
- Ablasyon yorumu: Image embedding kolonlari standardized oldugu icin ablasyon testinde bu blok sifira cekildi; bu durum image branch'in ortalama temsilini kaldirip katkisini olcmek icin kullanildi.
- Yorum: Image embeddingler modele net pozitif katkı sagliyor; image block kaldirilinca MAE 797.72 kadar kotulesiyor.

## En Yuksek Hata Yapan 20 Ilan

| listing_id | district | neighborhood | rooms | m2_gross | actual_price_try | baseline_prediction | fusion_prediction | fusion_residual | fusion_abs_error | fusion_ape_pct | abs_error_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hepsiemlak:43014-3617 | Çankaya | Oran Mah. | 4+1 | 175.0000 | 180,000.0000 | 128,045.0401 | 123,060.7737 | -56,939.2263 | 56,939.2263 | 31.6329 | -4,984.2664 |
| hepsiemlak:130551-1404 | Çankaya | Yukarı Dikmen Mah. | 5+1 | 305.0000 | 149,000.0000 | 130,170.0458 | 97,047.3460 | -51,952.6540 | 51,952.6540 | 34.8676 | -33,122.6999 |
| hepsiemlak:162654-112 | Çankaya | Çiğdem Mah. | 4+1 | 200.0000 | 120,000.0000 | 74,196.3756 | 69,423.1031 | -50,576.8969 | 50,576.8969 | 42.1474 | -4,773.2725 |
| hepsiemlak:149516-72 | Çankaya | Gaziosmanpaşa Mah. | 2+1 | 100.0000 | 99,000.0000 | 56,922.9254 | 49,438.9880 | -49,561.0120 | 49,561.0120 | 50.0616 | -7,483.9374 |
| hepsiemlak:5665-1536 | Çankaya | Namık Kemal Mah. | 3+0 | 139.0000 | 83,000.0000 | 57,041.3502 | 48,422.6883 | -34,577.3117 | 34,577.3117 | 41.6594 | -8,618.6619 |
| hepsiemlak:8315-16683 | Çankaya | Kızılırmak Mah. | 4+1 | 185.0000 | 85,000.0000 | 59,649.0568 | 52,668.7682 | -32,331.2318 | 32,331.2318 | 38.0367 | -6,980.2887 |
| hepsiemlak:7677-4917 | Çankaya | Mustafa Kemal Mah. | 3+1 | 145.0000 | 75,000.0000 | 42,669.2486 | 43,348.8567 | -31,651.1433 | 31,651.1433 | 42.2015 | 679.6082 |
| hepsiemlak:145733-213 | Çankaya | Cevizlidere Mah. | 5+1 | 375.0000 | 78,000.0000 | 96,425.8913 | 109,161.0674 | 31,161.0674 | 31,161.0674 | 39.9501 | -12,735.1761 |
| hepsiemlak:5978-6315 | Çankaya | Cevizlidere Mah. | 4+1 | 190.0000 | 80,000.0000 | 46,478.5771 | 48,880.7889 | -31,119.2111 | 31,119.2111 | 38.8990 | 2,402.2118 |
| hepsiemlak:37695-6754 | Çankaya | Beytepe Mah. | 4+1 | 206.0000 | 99,750.0000 | 91,363.0166 | 73,056.1146 | -26,693.8854 | 26,693.8854 | 26.7608 | -18,306.9020 |
| hepsiemlak:9800-2550 | Çankaya | Emek Mah. | 3+1 | 145.0000 | 65,000.0000 | 39,001.4751 | 38,449.7005 | -26,550.2995 | 26,550.2995 | 40.8466 | -551.7746 |
| hepsiemlak:165177-42 | Çankaya | Prof. Dr. Ahmet Taner Kışlalı Mah. | 4+1 | 200.0000 | 93,000.0000 | 62,194.5871 | 66,452.3255 | -26,547.6745 | 26,547.6745 | 28.5459 | 4,257.7384 |
| hepsiemlak:0-46238403 | Çankaya | Kavaklıdere Mah. | 3+2 | 150.0000 | 19,000.0000 | 55,611.1481 | 44,596.7964 | 25,596.7964 | 25,596.7964 | 134.7200 | 11,014.3517 |
| hepsiemlak:18229-1817 | Çankaya | Kavaklıdere Mah. | 3+1 | 145.0000 | 30,000.0000 | 52,931.5772 | 54,181.9140 | 24,181.9140 | 24,181.9140 | 80.6064 | -1,250.3367 |
| hepsiemlak:158676-83 | Keçiören | Ovacık Mah. | 4+1 | 200.0000 | 70,000.0000 | 75,474.4604 | 93,471.7312 | 23,471.7312 | 23,471.7312 | 33.5310 | -17,997.2708 |
| hepsiemlak:156606-360 | Keçiören | Ovacık Mah. | 4+1 | 230.0000 | 50,000.0000 | 63,554.6823 | 73,441.3021 | 23,441.3021 | 23,441.3021 | 46.8826 | -9,886.6198 |
| hepsiemlak:159469-437 | Çankaya | Keklik Pınarı Mah. | 3+1 | 150.0000 | 36,000.0000 | 51,811.9600 | 58,639.6905 | 22,639.6905 | 22,639.6905 | 62.8880 | -6,827.7305 |
| hepsiemlak:56844-1171 | Çankaya | Remzi Oğuz Arık Mah. | 1+1 | 85.0000 | 52,000.0000 | 25,329.4443 | 30,326.6950 | -21,673.3050 | 21,673.3050 | 41.6794 | 4,997.2507 |
| hepsiemlak:16602-1909 | Çankaya | Ayrancı Mah. | 1+1 | 50.0000 | 52,000.0000 | 34,228.3998 | 30,781.2798 | -21,218.7202 | 21,218.7202 | 40.8052 | -3,447.1200 |
| hepsiemlak:3966-4018 | Çankaya | Aziziye Mah. | 4+1 | 260.0000 | 55,000.0000 | 75,853.4477 | 75,393.9162 | 20,393.9162 | 20,393.9162 | 37.0798 | 459.5315 |

## Atlanan Modeller

- `LightGBMRegressor`: lightgbm kurulu degil
- `XGBoostRegressor`: xgboost kurulu degil