# Frontend Metadata Raporu

## Özet

- Çalışma zamanı: `2026-05-18T21:16:57`
- Kaynak dataset: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- Üretilen lokasyon dosyası: `E:\rent-agent\frontend\public\meta\locations.json`
- Üretilen kategorik seçenek dosyası: `E:\rent-agent\frontend\public\meta\categorical-options.json`
- Üretilen sayısal aralık dosyası: `E:\rent-agent\frontend\public\meta\numeric-ranges.json`

## Lokasyon Kapsamı

- Toplam city sayısı: **1**
- Toplam district sayısı: **16**
- Toplam neighborhood sayısı: **361**

### En çok örneği olan ilçeler

| İlçe | Örnek sayısı | Neighborhood sayısı |
| --- | --- | --- |
| Çankaya | 2,452 | 109 |
| Keçiören | 891 | 45 |
| Etimesgut | 634 | 34 |
| Yenimahalle | 632 | 51 |
| Mamak | 561 | 50 |
| Gölbaşı | 359 | 12 |
| Sincan | 344 | 29 |
| Altındağ | 216 | 19 |
| Çubuk | 103 | 9 |
| Polatlı | 97 | 13 |

## Kategorik Alan Option Sayıları

| Alan | Field key | Option sayısı |
| --- | --- | --- |
| Oda tipi | rooms | 25 |
| Bulunduğu kat | floor | 36 |
| Isıtma tipi | heating_type | 10 |
| Yakıt tipi | fuel_type | 4 |
| Konut tipi | home_type | 1 |
| Bulunduğu tip | home_shape | 12 |
| Eşyalı mı? | is_furnished | 3 |
| Banyo sayısı | bathrooms | 10 |
| Bina yaşı | building_age | 58 |
| Toplam kat | total_floors | 42 |

## Numeric Range Özeti

| Alan | Field key | Min | Max | Median | P05 | P95 | Missing count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Brüt m² | m2_gross | 14 | 480 | 115 | 50.4 | 200 | 0 |
| Aidat | dues_try | 1 | 1750000 | 500 | 100 | 4500 | 2,126 |
| Kira fiyatı | price_try | 8000 | 320000 | 30000 | 18000 | 65000 | 0 |

## Frontend Kullanımı

- `locations.json`, şehir -> ilçe -> mahalle seçimlerini bağımlı select veya autocomplete akışında beslemek için kullanılabilir.
- `categorical-options.json`, serbest yazı yerine dataset temelli dropdown, combobox veya suggestion listeleri üretmek için kullanılabilir.
- `numeric-ranges.json`, sayısal input placeholder, slider sınırı, yardımcı metin ve validation önerileri için referans sağlar.
- `price_try` sadece referans aralık olarak tutulur; form input’u olarak kullanılmamalıdır.
- Null veya boş değerler frontend tarafında `Belirtilmemiş` etiketiyle gösterilebilir.