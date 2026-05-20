# Text-Enhanced Tabular Model Results

## Ozet

- Dataset: `E:\rent-agent\dataset\train_ready_ml.parquet`
- Kaydedilen model bundle: `E:\rent-agent\models\text_model.joblib`
- Kaydedilen rapor: `E:\rent-agent\reports\text_model_results.md`
- En iyi candidate: **Ridge + TFIDF**
- Secilen text representation: **tfidf**
- Baseline MAE referansi: **4683.35**
- Baseline MAPE referansi: **14.85%**
- Test MAE improvement: **-18.77**
- Test MAPE improvement: **-0.29 puan**

## Validation Leaderboard

| candidate | representation | validation_mae | validation_rmse | validation_r2 | validation_mape |
| --- | --- | --- | --- | --- | --- |
| Ridge + TFIDF | tfidf | 5,002.3341 | 8,151.6108 | 0.7886 | 14.5405 |
| HistGradientBoosting + SVD | svd | 5,007.6220 | 8,118.8396 | 0.7903 | 14.6614 |
| ElasticNet + TFIDF | tfidf | 5,024.7654 | 8,142.4653 | 0.7891 | 14.6496 |
| Ridge + SVD | svd | 5,125.7585 | 8,259.5587 | 0.7830 | 14.9441 |
| ElasticNet + SVD | svd | 5,126.7046 | 8,274.4907 | 0.7822 | 14.9385 |

## Baseline vs Text Karsilastirmasi

| model | mae | rmse | r2 | mape |
| --- | --- | --- | --- | --- |
| Baseline HistGradientBoosting | 4,683.3548 | 6,911.9063 | 0.7632 | 14.8518 |
| Best text-enhanced model | 4,702.1261 | 6,629.6711 | 0.7821 | 15.1436 |
| Improvement vs baseline | -18.7714 | 282.2352 | 0.0189 | -0.2918 |

## District Bazli Improvement

| district | sample_count | baseline_mae | text_mae | baseline_mape | text_mape | mae_improvement | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Yenimahalle | 94 | 5,313.0664 | 4,749.6239 | 15.3470 | 13.5294 | 563.4425 | 1.8176 |
| Çankaya | 343 | 6,247.7723 | 5,685.7281 | 17.0422 | 15.8177 | 562.0443 | 1.2245 |
| Altındağ | 30 | 3,268.5567 | 3,511.7893 | 11.2572 | 12.3502 | -243.2326 | -1.0931 |
| Etimesgut | 104 | 3,906.1851 | 4,274.6545 | 11.6324 | 12.5785 | -368.4694 | -0.9461 |
| Mamak | 101 | 3,150.8197 | 3,526.5006 | 12.5501 | 14.3020 | -375.6810 | -1.7519 |
| Polatlı | 18 | 3,551.8903 | 3,940.6091 | 19.4242 | 20.0553 | -388.7188 | -0.6311 |
| Keçiören | 143 | 3,975.4617 | 4,398.4980 | 14.7347 | 16.1897 | -423.0362 | -1.4551 |
| Sincan | 61 | 3,384.3757 | 3,828.2861 | 13.6683 | 15.6431 | -443.9104 | -1.9747 |
| Çubuk | 16 | 2,348.9465 | 3,190.6926 | 13.9025 | 17.7765 | -841.7461 | -3.8739 |
| Gölbaşı | 55 | 3,296.1817 | 4,515.8352 | 10.4230 | 14.0371 | -1,219.6535 | -3.6141 |
| Pursaklar | 8 | 2,543.5430 | 4,345.0103 | 9.8797 | 16.6283 | -1,801.4673 | -6.7487 |

## Luxury Listing Improvement

- Luxury proxy tanimi: `actual price >= 75k` veya temizlenmis metinde `lux/luks/ultra/rezidans/ebeveyn/guvenlik/teras/manzarali` tokenlarindan biri geciyor.

| segment | sample_count | baseline_mae | text_mae | mae_improvement | baseline_mape | text_mape | mape_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All test listings | 975 | 4,683.3548 | 4,702.1261 | -18.7714 | 14.8518 | 15.1436 | -0.2918 |
| Luxury proxy | 314 | 5,758.5367 | 6,132.3368 | -373.8001 | 14.6420 | 16.1896 | -1.5476 |
| High-price (>=75k) | 25 | 12,440.2355 | 11,903.3547 | 536.8808 | 14.2046 | 13.4253 | 0.7793 |
| Luxury keywords | 308 | 5,569.8249 | 6,027.9779 | -458.1530 | 14.5441 | 16.2255 | -1.6814 |

## Feature Space Ozet

| metric | value |
| --- | --- |
| train_validation_samples | 5523 |
| tabular_feature_count | 411 |
| raw_tfidf_feature_count | 10000 |
| final_feature_count | 10411 |
| representation | tfidf |
| clean_text_blank_count | 0 |
| clean_text_blank_ratio_pct | 0.0000 |

## Overall Feature Importance

| feature | importance | signed_value |
| --- | --- | --- |
| text_tfidf__yasarkent | 40,759.873749 | 40,759.873749 |
| neighborhood_Oran Mah. | 27,104.411529 | 27,104.411529 |
| neighborhood_Beytepe Mah. | 16,193.140531 | 16,193.140531 |
| text_tfidf__one tower | 14,957.598081 | 14,957.598081 |
| text_tfidf__sifir | 14,841.145362 | 14,841.145362 |
| district_Çankaya | 14,529.012834 | 14,529.012834 |
| neighborhood_Namık Kemal Mah. | 13,777.949100 | 13,777.949100 |
| text_tfidf__kapal | 11,559.384579 | 11,559.384579 |
| text_tfidf__luks | 11,007.568113 | 11,007.568113 |
| neighborhood_Alcı Mah. | 10,882.031737 | -10,882.031737 |
| text_tfidf__tower | 10,861.826206 | 10,861.826206 |
| text_tfidf__ten | 10,508.871257 | 10,508.871257 |
| neighborhood_Fatih Sultan Mah. | 10,466.563951 | 10,466.563951 |
| text_tfidf__kiralik esyali | 10,158.034484 | 10,158.034484 |
| home_shape_Bahçe Dubleksi | 10,135.131530 | 10,135.131530 |
| neighborhood_Çayyolu Mah. | 9,653.890515 | 9,653.890515 |
| text_tfidf__teras | 9,472.226177 | -9,472.226177 |
| neighborhood_Çukurambar Mah. | 9,371.509490 | 9,371.509490 |
| neighborhood_Kızılırmak Mah. | 8,903.289191 | 8,903.289191 |
| neighborhood_Gaziosmanpaşa Mah. | 8,782.982887 | 8,782.982887 |

## Text Feature Importance

| feature | importance | signed_value |
| --- | --- | --- |
| text_tfidf__yasarkent | 40,759.873749 | 40,759.873749 |
| text_tfidf__one tower | 14,957.598081 | 14,957.598081 |
| text_tfidf__sifir | 14,841.145362 | 14,841.145362 |
| text_tfidf__kapal | 11,559.384579 | 11,559.384579 |
| text_tfidf__luks | 11,007.568113 | 11,007.568113 |
| text_tfidf__tower | 10,861.826206 | 10,861.826206 |
| text_tfidf__ten | 10,508.871257 | 10,508.871257 |
| text_tfidf__kiralik esyali | 10,158.034484 | 10,158.034484 |
| text_tfidf__teras | 9,472.226177 | -9,472.226177 |
| text_tfidf__genis | 8,691.009101 | -8,691.009101 |
| text_tfidf__loft | 8,586.720399 | 8,586.720399 |
| text_tfidf__kapali | 7,975.789939 | 7,975.789939 |
| text_tfidf__camasir | 7,878.531691 | 7,878.531691 |
| text_tfidf__effect | 7,824.554749 | 7,824.554749 |
| text_tfidf__kuzu effect | 7,723.516499 | 7,723.516499 |
| text_tfidf__bathrooms | 7,586.364108 | 7,586.364108 |
| text_tfidf__oran | 7,493.381374 | -7,493.381374 |
| text_tfidf__terasli | 7,297.976997 | -7,297.976997 |
| text_tfidf__prestijli | 7,216.461532 | 7,216.461532 |
| text_tfidf__remax | 7,166.016410 | 7,166.016410 |

## En Faydali Text Tokenlari

- En iyi yorumlanabilir raw-TFIDF lineer model: **Ridge + TFIDF**

### Pozitif Fiyat Sinyali Veren Tokenlar

| token | coefficient | abs_coefficient |
| --- | --- | --- |
| yasarkent | 40,759.873749 | 40,759.873749 |
| one tower | 14,957.598081 | 14,957.598081 |
| sifir | 14,841.145362 | 14,841.145362 |
| kapal | 11,559.384579 | 11,559.384579 |
| luks | 11,007.568113 | 11,007.568113 |
| tower | 10,861.826206 | 10,861.826206 |
| ten | 10,508.871257 | 10,508.871257 |
| kiralik esyali | 10,158.034484 | 10,158.034484 |
| loft | 8,586.720399 | 8,586.720399 |
| kapali | 7,975.789939 | 7,975.789939 |
| camasir | 7,878.531691 | 7,878.531691 |
| effect | 7,824.554749 | 7,824.554749 |
| kuzu effect | 7,723.516499 | 7,723.516499 |
| bathrooms | 7,586.364108 | 7,586.364108 |
| prestijli | 7,216.461532 | 7,216.461532 |

### Negatif Fiyat Sinyali Veren Tokenlar

| token | coefficient | abs_coefficient |
| --- | --- | --- |
| teras | -9,472.226177 | 9,472.226177 |
| genis | -8,691.009101 | 8,691.009101 |
| oran | -7,493.381374 | 7,493.381374 |
| terasli | -7,297.976997 | 7,297.976997 |
| ataturk | -6,312.701809 | 6,312.701809 |
| toki | -6,312.243995 | 6,312.243995 |
| giris | -5,465.818809 | 5,465.818809 |
| kombili | -5,196.233062 | 5,196.233062 |
| bolgede | -5,124.697663 | 5,124.697663 |
| kurtulus | -5,097.552096 | 5,097.552096 |
| banyo tuvalet | -4,954.369504 | 4,954.369504 |
| de kiralik | -4,873.505344 | 4,873.505344 |
| dairemiz | -4,846.792818 | 4,846.792818 |
| cok merkezi | -4,830.221367 | 4,830.221367 |
| tuvalet | -4,696.427066 | 4,696.427066 |

### En Guclu Mutlak Token Katsayilari

| token | coefficient | abs_coefficient |
| --- | --- | --- |
| yasarkent | 40,759.873749 | 40,759.873749 |
| one tower | 14,957.598081 | 14,957.598081 |
| sifir | 14,841.145362 | 14,841.145362 |
| kapal | 11,559.384579 | 11,559.384579 |
| luks | 11,007.568113 | 11,007.568113 |
| tower | 10,861.826206 | 10,861.826206 |
| ten | 10,508.871257 | 10,508.871257 |
| kiralik esyali | 10,158.034484 | 10,158.034484 |
| teras | -9,472.226177 | 9,472.226177 |
| genis | -8,691.009101 | 8,691.009101 |
| loft | 8,586.720399 | 8,586.720399 |
| kapali | 7,975.789939 | 7,975.789939 |
| camasir | 7,878.531691 | 7,878.531691 |
| effect | 7,824.554749 | 7,824.554749 |
| kuzu effect | 7,723.516499 | 7,723.516499 |

## SVD Text Component Analizi

_Veri yok_

## Kullanilan Text Cleaning Kurallari

- Lowercase uygulandi.
- Unicode normalize + combining mark temizligi yapildi.
- Fazla whitespace ve noktalama sadelestirildi.
- Su boilerplate kaliplari kaldirildi: `telefonu goster`, `detayli bilgi`, `arayiniz`, `gayrimenkul`, `emlak`, `kahve icmeye`.
- 2 token altindaki cok kisa metinler bos kabul edildi.

## Atlanan Modeller

- `LightGBM + SVD`: lightgbm kurulu degil
- `XGBoost + SVD`: xgboost kurulu degil