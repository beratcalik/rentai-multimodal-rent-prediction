# Baseline Model Sonuclari

## Ozet

- Calisma zamani: `2026-05-10T01:02:27`
- Dataset: `E:\rent-agent\dataset\train_ready_ml.parquet`
- Hedef degisken: `price_try`
- Model secim metriği: validation `MAE`
- Kaydedilen model: `E:\rent-agent\models\baseline_model.joblib`
- Kaydedilen preprocessing pipeline: `E:\rent-agent\models\baseline_preprocessor.joblib`
- En iyi model: **HistGradientBoostingRegressor**
- Test icin kaydedilen final model: **HistGradientBoostingRegressor**

## Veri ve Split Bilgileri

- Toplam ornek sayisi: **6,498**
- Train sayisi: **4,548**
- Validation sayisi: **975**
- Test sayisi: **975**
- Ham feature sayisi: **16**
- Donusmus feature sayisi: **411**

## Kullanilan Feature'lar

- Tum feature'lar: `city`, `district`, `neighborhood`, `rooms`, `bathrooms`, `m2_gross`, `building_age`, `floor`, `total_floors`, `heating_type`, `fuel_type`, `is_furnished`, `dues_try`, `home_type`, `home_shape`, `image_count`
- Sayisallastirilan kolonlar: `rooms`, `bathrooms`, `m2_gross`, `building_age`, `floor`, `total_floors`, `dues_try`, `image_count`
- One-hot encode edilen kategorik kolonlar: `city`, `district`, `neighborhood`, `heating_type`, `fuel_type`, `is_furnished`, `home_type`, `home_shape`
- `rooms` kolonu `3+1 -> 4` mantigiyla sayisallastirildi.
- `rooms` icin toplam oda sayisi 12'yi asan acikca anomalik degerler `NaN` kabul edildi.
- `floor` kolonu icin `Giris Kati`, `Zemin`, `Bahce Kati`, `Yuksek Giris` degerleri `0` kabul edildi.
- Belirsiz veya baglama bagli kat degerleri (`Ara Kat`, `En Ust Kat`, `Kot 1`, `Bodrum`, `Cati Kati` vb.) `NaN` birakildi.
- Numerik eksikler median ile, kategorik eksikler most-frequent ile dolduruldu.

## Validation Sonuclari

| Model | Validation MAE | Validation RMSE | Validation R2 | Validation MAPE (%) |
| --- | --- | --- | --- | --- |
| HistGradientBoostingRegressor | 5,008.70 | 7,893.61 | 0.8018 | 14.69 |
| RandomForestRegressor | 5,050.10 | 8,262.03 | 0.7828 | 15.02 |

## Test Sonuclari

| Metric | Value |
| --- | --- |
| MAE | 4,683.35 |
| RMSE | 6,911.91 |
| R2 | 0.7632 |
| MAPE (%) | 14.85 |

## Feature Importance

- Yontem: `permutation_importance`

| Feature | Importance |
| --- | --- |
| m2_gross | 3,301.680215 |
| district_Çankaya | 2,662.362313 |
| dues_try | 764.378473 |
| is_furnished_Evet | 734.782807 |
| building_age | 729.524775 |
| total_floors | 588.328200 |
| rooms | 266.275234 |
| bathrooms | 260.586123 |
| district_Yenimahalle | 231.754670 |
| floor | 209.908440 |
| heating_type_Kombi | 174.139961 |
| district_Polatlı | 135.048453 |
| district_Çubuk | 112.943672 |
| district_Etimesgut | 85.897172 |
| neighborhood_Oran Mah. | 83.713912 |
| neighborhood_Atayurt Mah. | 80.813904 |
| neighborhood_Emek Mah. | 78.520355 |
| neighborhood_Bahçelievler Mah. | 67.373014 |
| district_Gölbaşı | 48.032721 |
| home_shape_Ara Kat | 46.463127 |

## Hata Analizi

| Ozet | Deger |
| --- | --- |
| Ortalama gercek fiyat | 32,649.78 |
| Ortalama tahmin | 33,434.15 |
| Ortalama signed error | 784.37 |
| Median absolute error | 3,364.56 |
| P90 absolute error | 10,285.17 |
| P95 absolute error | 13,281.71 |

### En Yuksek Hata Ureten Test Ornekleri

| listing_id | district | neighborhood | rooms | m2_gross | floor | image_count | actual_price_try | predicted_price_try | abs_error | ape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hepsiemlak:124381-850 | Çankaya | Sancak Mah. | 4+1 | 380.00 | 2. Kat | 20 | 56,000.00 | 110,334.68 | 54,334.68 | 97.03 |
| hepsiemlak:149516-72 | Çankaya | Gaziosmanpaşa Mah. | 2+1 | 100.00 | 3. Kat | 20 | 99,000.00 | 58,324.64 | 40,675.36 | 41.09 |
| hepsiemlak:11098-2007 | Çankaya | Çankaya Mah. | 5+1 | 350.00 | En Üst Kat | 20 | 75,000.00 | 113,663.31 | 38,663.31 | 51.55 |
| hepsiemlak:54420-2303 | Çankaya | Bahçelievler Mah. | 4+1 | 160.00 | 3. Kat | 20 | 16,500.00 | 54,931.50 | 38,431.50 | 232.92 |
| hepsiemlak:56844-1166 | Çankaya | Bahçelievler Mah. | 1+1 | 55.00 | 2. Kat | 20 | 70,000.00 | 40,829.10 | 29,170.90 | 41.67 |
| hepsiemlak:140580-2726 | Yenimahalle | İnönü Mah. | 2+1 | 140.00 | Ara Kat | 20 | 85,000.00 | 56,457.41 | 28,542.59 | 33.58 |
| hepsiemlak:136808-317 | Etimesgut | Göksu Mah. | 4+1 | 196.00 | 19. Kat | 20 | 95,000.00 | 68,104.93 | 26,895.07 | 28.31 |
| hepsiemlak:58419-4096 | Çankaya | Bağcılar Mah. | 8+1 | 280.00 | Yüksek Giriş | 20 | 42,000.00 | 68,617.59 | 26,617.59 | 63.38 |
| hepsiemlak:156378-48 | Çankaya | Bağcılar Mah. | 4+1 | 200.00 | 17. Kat | 20 | 65,000.00 | 91,211.25 | 26,211.25 | 40.32 |
| hepsiemlak:123623-3557 | Çankaya | Alacaatlı Mah. | 4+1 | 230.00 | 2. Kat | 20 | 68,000.00 | 93,646.58 | 25,646.58 | 37.72 |

### Ilce Bazli Test Hata Ozeti

| district | sample_count | mae | mean_signed_error | mape |
| --- | --- | --- | --- | --- |
| Çankaya | 343 | 6,247.77 | 1,677.31 | 17.04 |
| Yenimahalle | 94 | 5,313.07 | -708.52 | 15.35 |
| Keçiören | 143 | 3,975.46 | 651.24 | 14.73 |
| Etimesgut | 104 | 3,906.19 | 570.75 | 11.63 |
| Polatlı | 18 | 3,551.89 | 2,467.72 | 19.42 |
| Sincan | 61 | 3,384.38 | -515.95 | 13.67 |
| Gölbaşı | 55 | 3,296.18 | 457.18 | 10.42 |
| Altındağ | 30 | 3,268.56 | -985.37 | 11.26 |
| Mamak | 101 | 3,150.82 | 734.10 | 12.55 |
| Pursaklar | 8 | 2,543.54 | -44.73 | 9.88 |

## Atlanan Modeller

- `LightGBMRegressor`: lightgbm kurulu degil
- `XGBoostRegressor`: xgboost kurulu degil

## Notlar

- Validation sonuclari model secimi icin train split uzerinde egitilen modellerden hesaplandi.
- Kaydedilen final model, en iyi algoritma secildikten sonra `train + validation` verisi ile yeniden egitildi.
- Test sonuclari sadece final model icin, hic gorulmemis test split uzerinde raporlandi.