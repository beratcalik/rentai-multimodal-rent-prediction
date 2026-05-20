# Dataset Summary

- Project root: `E:\rent-agent\dataset`
- Report path: `E:\rent-agent\dataset\reports\dataset_summary.md`
- Total parquet files found: **6**
- Main raw listing count: **6,875**
- Union of listing_id values across all parquet files: **6,920**
- Image files physically present under `images/`: **100,522**
- Parquet engines: `pyarrow=yes`, `fastparquet=no`

## Project Structure

- Top-level parquet files are at the dataset root.
- The `images/` directory contains the local image assets referenced by image-related parquet files.
- No existing Python analysis script or requirements file was found in this workspace before this run.

## Quick Summary

- Raw main listing table: `listings.parquet`
- Recommended tabular training file: `train_ready_ml.parquet`
- Recommended multimodal training file: `train_ready_multimodal.parquet`
- Image metadata table: `images.parquet`
- Validation/report table: `validation_report.parquet`
- Process log table: `run_log.parquet`
- Suggested target column: `price_try`

## Dependency Note

- Current run succeeded because at least one parquet backend is available.
- Requirements suggestion: add `pandas`, `pyarrow`, and optionally `fastparquet` if you want both engines reproducibly.

## Per-File Analysis

### `images.parquet`

- Absolute path: `E:\rent-agent\dataset\images.parquet`
- Role guess: Image metadata table
- Shape: **130,525 rows x 12 columns**
- Duplicate rows: no
- Listing-like columns: `listing_id`
- Unique `listing_id` count: **6,561**
- Possible price target: none
- Visual local path columns: `local_path`
- Visual URL columns: `source_url`
- Text columns: none
- Trainability: Kismen. Image branch icin yardimci tablo olarak kullanilabilir, ancak fiyat hedefi yok.

#### Columns

`image_id`, `listing_id`, `order_index`, `source_url`, `local_path`, `is_valid`, `width`, `height`, `file_size`, `format`, `aspect_ratio`, `scraped_at`

#### Dtypes

| Column | Dtype |
| --- | --- |
| image_id | object |
| listing_id | object |
| order_index | int64 |
| source_url | object |
| local_path | object |
| is_valid | bool |
| width | float64 |
| height | float64 |
| file_size | float64 |
| format | object |
| aspect_ratio | float64 |
| scraped_at | object |

#### Missing Values

| Column | Missing |
| --- | --- |
| aspect_ratio | 30,082 |
| file_size | 30,082 |
| format | 30,082 |
| height | 30,082 |
| width | 30,082 |
| image_id | 0 |
| is_valid | 0 |
| listing_id | 0 |
| local_path | 0 |
| order_index | 0 |
| scraped_at | 0 |
| source_url | 0 |

#### First 5 Rows

```text
               image_id            listing_id order_index                                                                               source_url                                   local_path is_valid  width height file_size format aspect_ratio          scraped_at
hepsiemlak:49652-2518:0 hepsiemlak:49652-2518           0                         https://hecdnnw.hemlak.com/img/multilanguage_desktop_flag_tr.png dataset\images\hepsiemlak_49652-2518\000.jpg    False     NA     NA        NA     NA           NA 2026-04-01T19:55:51
hepsiemlak:49652-2518:1 hepsiemlak:49652-2518           1                         https://hecdnnw.hemlak.com/img/multilanguage_desktop_flag_en.png dataset\images\hepsiemlak_49652-2518\001.jpg    False     NA     NA        NA     NA           NA 2026-04-01T19:55:53
hepsiemlak:49652-2518:2 hepsiemlak:49652-2518           2                         https://hecdnnw.hemlak.com/img/multilanguage_desktop_flag_ru.png dataset\images\hepsiemlak_49652-2518\002.jpg    False     NA     NA        NA     NA           NA 2026-04-01T19:56:02
hepsiemlak:49652-2518:3 hepsiemlak:49652-2518           3                                        https://hecdn01.hemlak.com/img/eids-logo-mini.png dataset\images\hepsiemlak_49652-2518\003.jpg    False     NA     NA        NA     NA           NA 2026-04-01T19:56:06
hepsiemlak:49652-2518:4 hepsiemlak:49652-2518           4 https://hecdn01.hemlak.com/ds01/7/1/1/1/2/2/6/4/b52dda02-ad73-4c7d-9ac6-378c1c8e4053.jpg dataset\images\hepsiemlak_49652-2518\004.jpg     True 1280.0  960.0  281574.0   JPEG       1.3333 2026-04-01T19:56:10
```

#### Image Path Audit

- Column `local_path` total paths: **130,525**
- Existing local files: **100,443**
- Broken local files: **30,082**
- Average valid images per listing: **15.43**
- Listings with at least one valid image: **6,509**
- Sample broken paths: `dataset\images\hepsiemlak_49652-2518\000.jpg`, `dataset\images\hepsiemlak_49652-2518\001.jpg`, `dataset\images\hepsiemlak_49652-2518\002.jpg`, `dataset\images\hepsiemlak_49652-2518\003.jpg`, `dataset\images\hepsiemlak_6664-2537\000.jpg`

### `listings.parquet`

- Absolute path: `E:\rent-agent\dataset\listings.parquet`
- Role guess: Raw main listing table
- Shape: **6,875 rows x 26 columns**
- Duplicate rows: no
- Listing-like columns: `listing_id`, `listing_no`
- Unique `listing_id` count: **6,875**
- Possible price target: `price_try`
- Visual local path columns: none
- Visual URL columns: none
- Text columns: `title`, `description`, `raw_specs_json`
- Trainability: Evet. Tabular + text egitimi icin uygun.

#### Columns

`listing_id`, `url`, `listing_no`, `price_try`, `city`, `district`, `neighborhood`, `rooms`, `bathrooms`, `m2_gross`, `m2_net`, `building_age`, `floor`, `total_floors`, `heating_type`, `fuel_type`, `is_furnished`, `dues_try`, `home_type`, `home_shape`, `updated_at`, `title`, `description`, `image_count`, `raw_specs_json`, `scraped_at`

#### Dtypes

| Column | Dtype |
| --- | --- |
| listing_id | object |
| url | object |
| listing_no | object |
| price_try | float64 |
| city | object |
| district | object |
| neighborhood | object |
| rooms | object |
| bathrooms | float64 |
| m2_gross | float64 |
| m2_net | object |
| building_age | float64 |
| floor | object |
| total_floors | float64 |
| heating_type | object |
| fuel_type | object |
| is_furnished | object |
| dues_try | float64 |
| home_type | object |
| home_shape | object |
| updated_at | object |
| title | object |
| description | object |
| image_count | int64 |
| raw_specs_json | object |
| scraped_at | object |

#### Missing Values

| Column | Missing |
| --- | --- |
| m2_net | 6,875 |
| fuel_type | 3,010 |
| dues_try | 2,518 |
| home_shape | 1,397 |
| building_age | 1,129 |
| is_furnished | 624 |
| bathrooms | 416 |
| description | 399 |
| m2_gross | 337 |
| floor | 333 |
| heating_type | 333 |
| home_type | 333 |
| listing_no | 333 |
| rooms | 333 |
| total_floors | 333 |
| updated_at | 333 |
| city | 328 |
| district | 328 |
| neighborhood | 328 |
| price_try | 319 |
| title | 154 |
| image_count | 0 |
| listing_id | 0 |
| raw_specs_json | 0 |
| scraped_at | 0 |
| url | 0 |

#### First 5 Rows

```text
           listing_id                                                                               url listing_no price_try   city district       neighborhood rooms bathrooms m2_gross m2_net building_age   floor total_floors heating_type fuel_type is_furnished dues_try home_type home_shape updated_at                                                                          title                                                                                                              description image_count                                                                                                           raw_specs_json          scraped_at
hepsiemlak:49652-2518 https://www.hepsiemlak.com/ankara-kecioren-asagi-eglence-kiralik/daire/49652-2518 49652-2518   29000.0 Ankara Keçiören Aşağı Eğlence Mah.   4+1       2.0    155.0     NA         21.0  2. Kat          2.0        Kombi        NA        False       NA     Daire   Tripleks 01/04/2026                  ERKA'DAN AŞAĞIEĞLENCE'DE 4+1 BAĞIMSIZ GENİŞ 2 BANYOLU DUBLEKS ERKA'DAN AŞAĞIEĞLENCE SARAR SOKAK'TA CADDEYE 1 BİNA MESAFEDE 4+1 BAĞIMSIZ BAKIMLI 2.KAT 2 BANYOLU GENİŞ M2'Lİ BALKONL...          20 {"İlan no": "49652-2518", "Son Güncelleme": "01/04/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Konut Şek... 2026-04-01T19:55:49
 hepsiemlak:6664-2537     https://www.hepsiemlak.com/ankara-cankaya-mebusevleri-kiralik/daire/6664-2537  6664-2537   40000.0 Ankara  Çankaya   Mebusevleri Mah.   3+1       1.0    115.0     NA         31.0  3. Kat          4.0        Kombi        NA         True   1000.0     Daire    Ara Kat 24/03/2026                                       EŞYALI MEBUSEVLERİ NDE 3+1 ARA KAT DAİRE          BEŞEVLER METROSUNA VE ANADOLU METROSUNA YAKIN KONUMDA BAKIMLI BİR BİNADA BULUNMAKTADIR. OTURUMA HAZIR HALDEDİR.          20 {"İlan no": "6664-2537", "Son Güncelleme": "24/03/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Konut Şekl... 2026-04-01T19:57:29
 hepsiemlak:6664-2480      https://www.hepsiemlak.com/ankara-altindag-hacettepe-kiralik/daire/6664-2480  6664-2480   28000.0 Ankara Altındağ     Hacettepe Mah.   3+1       1.0    120.0     NA         30.0  1. Kat          2.0        Kombi  Doğalgaz        False       NA     Daire      Daire 24/03/2026 CEBECİ DUMLUPINAR CAD. MİLLET BAHÇESİ CEPHELİ 3+1 KATTA ÖN CEPHE KOMBİLİ DAİRE DAİRE DUMLUPUNAR CADDESİ ÜSTÜNDE VE MİLLET BAHÇESİ MANZARALIDIR. TALATPAŞA BULVARI VE KURTULUŞ METRO İSTASYONUNA YAKI...          20 {"İlan no": "6664-2480", "Son Güncelleme": "24/03/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Konut Şekl... 2026-04-01T19:59:15
 hepsiemlak:1920-2929         https://www.hepsiemlak.com/ankara-cankaya-aziziye-kiralik/daire/1920-2929  1920-2929   41000.0 Ankara  Çankaya       Aziziye Mah.   3+1       1.0    110.0     NA         31.0  4. Kat          4.0        Kombi        NA        False    650.0     Daire         NA 31/03/2026               Hoşdere Kiralık Daire Çankaya 3+1 Asansörlü Otoparklı HemenTaşın Hoşdere Kiralık Daire Çankaya 3+1 Asansörlü Otoparklı HemenTaşın Aziziye Mahallesi'nin Kalbinde, Huzurlu ve Konforlu ...          20 {"İlan no": "1920-2929", "Son Güncelleme": "31/03/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Oda Sayısı... 2026-04-01T20:00:33
hepsiemlak:14225-3100        https://www.hepsiemlak.com/ankara-mamak-sahintepe-kiralik/daire/14225-3100 14225-3100   19000.0 Ankara    Mamak     Şahintepe Mah.   3+1       1.0    125.0     NA         15.0 Ara Kat          3.0        Kombi  Doğalgaz        False    150.0     Daire    Ara Kat 31/03/2026                       ANKARA MAMAK ŞAHİN TEPE MAHALLESİ ORTA KAT 19.000 KİRLIK DAİREMİZ 3+1 OLUP 2 BALKONLUDUR.KATTA OLUP BAĞIMSIZ SALONDUR. MEMLUN KALINDIĞI SÜRECE UZUN ZAMAN OTURALABİLİR.KİRAMIZ...          20 {"İlan no": "14225-3100", "Son Güncelleme": "31/03/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Konut Şek... 2026-04-01T20:02:00
```

### `run_log.parquet`

- Absolute path: `E:\rent-agent\dataset\run_log.parquet`
- Role guess: Operational run log
- Shape: **8,389 rows x 21 columns**
- Duplicate rows: no
- Listing-like columns: `listing_id`
- Unique `listing_id` count: **6,920**
- Possible price target: none
- Visual local path columns: none
- Visual URL columns: none
- Text columns: none
- Trainability: Hayir. Bu dosya toplama/boru hatti logu; model egitimi icin ana veri degil.

#### Columns

`run_id`, `ts`, `event`, `search_url`, `limit`, `headful`, `discovered`, `new_added`, `queue_size`, `url`, `listing_id`, `images`, `ml_ready`, `multimodal_ready`, `reject_reason`, `processed`, `queue_left`, `visited_total`, `ml_ready_added`, `multimodal_ready_added`, `error`

#### Dtypes

| Column | Dtype |
| --- | --- |
| run_id | object |
| ts | object |
| event | object |
| search_url | object |
| limit | float64 |
| headful | object |
| discovered | float64 |
| new_added | float64 |
| queue_size | float64 |
| url | object |
| listing_id | object |
| images | float64 |
| ml_ready | object |
| multimodal_ready | object |
| reject_reason | object |
| processed | float64 |
| queue_left | float64 |
| visited_total | float64 |
| ml_ready_added | float64 |
| multimodal_ready_added | float64 |
| error | object |

#### Missing Values

| Column | Missing |
| --- | --- |
| ml_ready_added | 8,361 |
| multimodal_ready_added | 8,361 |
| processed | 8,361 |
| queue_left | 8,361 |
| visited_total | 8,361 |
| discovered | 8,350 |
| new_added | 8,350 |
| queue_size | 8,350 |
| headful | 8,337 |
| limit | 8,337 |
| search_url | 8,337 |
| reject_reason | 7,854 |
| error | 7,253 |
| images | 1,255 |
| listing_id | 1,255 |
| ml_ready | 1,255 |
| multimodal_ready | 1,255 |
| url | 127 |
| event | 0 |
| run_id | 0 |
| ts | 0 |

#### First 5 Rows

```text
      run_id                  ts        event                                search_url limit headful discovered new_added queue_size                                                                               url            listing_id images ml_ready multimodal_ready reject_reason processed queue_left visited_total ml_ready_added multimodal_ready_added error
b63b7de4d23f 2026-04-01T19:54:42    run_start https://www.hepsiemlak.com/ankara-kiralik  20.0    True         NA        NA         NA                                                                                NA                    NA     NA       NA               NA            NA        NA         NA            NA             NA                     NA    NA
b63b7de4d23f 2026-04-01T19:55:42 queue_filled                                        NA    NA      NA       69.0      69.0       69.0                                                                                NA                    NA     NA       NA               NA            NA        NA         NA            NA             NA                     NA    NA
b63b7de4d23f 2026-04-01T19:57:22   listing_ok                                        NA    NA      NA         NA        NA         NA https://www.hepsiemlak.com/ankara-kecioren-asagi-eglence-kiralik/daire/49652-2518 hepsiemlak:49652-2518   20.0     True             True            NA        NA         NA            NA             NA                     NA    NA
b63b7de4d23f 2026-04-01T19:59:09   listing_ok                                        NA    NA      NA         NA        NA         NA     https://www.hepsiemlak.com/ankara-cankaya-mebusevleri-kiralik/daire/6664-2537  hepsiemlak:6664-2537   20.0     True             True            NA        NA         NA            NA             NA                     NA    NA
b63b7de4d23f 2026-04-01T20:00:26   listing_ok                                        NA    NA      NA         NA        NA         NA      https://www.hepsiemlak.com/ankara-altindag-hacettepe-kiralik/daire/6664-2480  hepsiemlak:6664-2480   20.0     True             True            NA        NA         NA            NA             NA                     NA    NA
```

### `train_ready_ml.parquet`

- Absolute path: `E:\rent-agent\dataset\train_ready_ml.parquet`
- Role guess: Train-ready tabular/text table
- Shape: **6,498 rows x 26 columns**
- Duplicate rows: no
- Listing-like columns: `listing_id`, `listing_no`
- Unique `listing_id` count: **6,498**
- Possible price target: `price_try`
- Visual local path columns: none
- Visual URL columns: none
- Text columns: `title`, `description`, `raw_specs_json`
- Trainability: Evet. Tabular + text egitimi icin uygun.

#### Columns

`listing_id`, `url`, `listing_no`, `price_try`, `city`, `district`, `neighborhood`, `rooms`, `bathrooms`, `m2_gross`, `m2_net`, `building_age`, `floor`, `total_floors`, `heating_type`, `fuel_type`, `is_furnished`, `dues_try`, `home_type`, `home_shape`, `updated_at`, `title`, `description`, `image_count`, `raw_specs_json`, `scraped_at`

#### Dtypes

| Column | Dtype |
| --- | --- |
| listing_id | object |
| url | object |
| listing_no | object |
| price_try | float64 |
| city | object |
| district | object |
| neighborhood | object |
| rooms | object |
| bathrooms | float64 |
| m2_gross | float64 |
| m2_net | object |
| building_age | float64 |
| floor | object |
| total_floors | int64 |
| heating_type | object |
| fuel_type | object |
| is_furnished | object |
| dues_try | float64 |
| home_type | object |
| home_shape | object |
| updated_at | object |
| title | object |
| description | object |
| image_count | int64 |
| raw_specs_json | object |
| scraped_at | object |

#### Missing Values

| Column | Missing |
| --- | --- |
| m2_net | 6,498 |
| fuel_type | 2,667 |
| dues_try | 2,163 |
| home_shape | 1,058 |
| building_age | 792 |
| is_furnished | 284 |
| bathrooms | 81 |
| description | 66 |
| city | 0 |
| district | 0 |
| floor | 0 |
| heating_type | 0 |
| home_type | 0 |
| image_count | 0 |
| listing_id | 0 |
| listing_no | 0 |
| m2_gross | 0 |
| neighborhood | 0 |
| price_try | 0 |
| raw_specs_json | 0 |
| rooms | 0 |
| scraped_at | 0 |
| title | 0 |
| total_floors | 0 |
| updated_at | 0 |
| url | 0 |

#### First 5 Rows

```text
           listing_id                                                                               url listing_no price_try   city district       neighborhood rooms bathrooms m2_gross m2_net building_age   floor total_floors heating_type fuel_type is_furnished dues_try home_type home_shape updated_at                                                                          title                                                                                                              description image_count                                                                                                           raw_specs_json          scraped_at
hepsiemlak:49652-2518 https://www.hepsiemlak.com/ankara-kecioren-asagi-eglence-kiralik/daire/49652-2518 49652-2518   29000.0 Ankara Keçiören Aşağı Eğlence Mah.   4+1       2.0    155.0     NA         21.0  2. Kat            2        Kombi        NA        False       NA     Daire   Tripleks 01/04/2026                  ERKA'DAN AŞAĞIEĞLENCE'DE 4+1 BAĞIMSIZ GENİŞ 2 BANYOLU DUBLEKS ERKA'DAN AŞAĞIEĞLENCE SARAR SOKAK'TA CADDEYE 1 BİNA MESAFEDE 4+1 BAĞIMSIZ BAKIMLI 2.KAT 2 BANYOLU GENİŞ M2'Lİ BALKONL...          20 {"İlan no": "49652-2518", "Son Güncelleme": "01/04/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Konut Şek... 2026-04-01T19:55:49
 hepsiemlak:6664-2537     https://www.hepsiemlak.com/ankara-cankaya-mebusevleri-kiralik/daire/6664-2537  6664-2537   40000.0 Ankara  Çankaya   Mebusevleri Mah.   3+1       1.0    115.0     NA         31.0  3. Kat            4        Kombi        NA         True   1000.0     Daire    Ara Kat 24/03/2026                                       EŞYALI MEBUSEVLERİ NDE 3+1 ARA KAT DAİRE          BEŞEVLER METROSUNA VE ANADOLU METROSUNA YAKIN KONUMDA BAKIMLI BİR BİNADA BULUNMAKTADIR. OTURUMA HAZIR HALDEDİR.          20 {"İlan no": "6664-2537", "Son Güncelleme": "24/03/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Konut Şekl... 2026-04-01T19:57:29
 hepsiemlak:6664-2480      https://www.hepsiemlak.com/ankara-altindag-hacettepe-kiralik/daire/6664-2480  6664-2480   28000.0 Ankara Altındağ     Hacettepe Mah.   3+1       1.0    120.0     NA         30.0  1. Kat            2        Kombi  Doğalgaz        False       NA     Daire      Daire 24/03/2026 CEBECİ DUMLUPINAR CAD. MİLLET BAHÇESİ CEPHELİ 3+1 KATTA ÖN CEPHE KOMBİLİ DAİRE DAİRE DUMLUPUNAR CADDESİ ÜSTÜNDE VE MİLLET BAHÇESİ MANZARALIDIR. TALATPAŞA BULVARI VE KURTULUŞ METRO İSTASYONUNA YAKI...          20 {"İlan no": "6664-2480", "Son Güncelleme": "24/03/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Konut Şekl... 2026-04-01T19:59:15
 hepsiemlak:1920-2929         https://www.hepsiemlak.com/ankara-cankaya-aziziye-kiralik/daire/1920-2929  1920-2929   41000.0 Ankara  Çankaya       Aziziye Mah.   3+1       1.0    110.0     NA         31.0  4. Kat            4        Kombi        NA        False    650.0     Daire         NA 31/03/2026               Hoşdere Kiralık Daire Çankaya 3+1 Asansörlü Otoparklı HemenTaşın Hoşdere Kiralık Daire Çankaya 3+1 Asansörlü Otoparklı HemenTaşın Aziziye Mahallesi'nin Kalbinde, Huzurlu ve Konforlu ...          20 {"İlan no": "1920-2929", "Son Güncelleme": "31/03/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Oda Sayısı... 2026-04-01T20:00:33
hepsiemlak:14225-3100        https://www.hepsiemlak.com/ankara-mamak-sahintepe-kiralik/daire/14225-3100 14225-3100   19000.0 Ankara    Mamak     Şahintepe Mah.   3+1       1.0    125.0     NA         15.0 Ara Kat            3        Kombi  Doğalgaz        False    150.0     Daire    Ara Kat 31/03/2026                       ANKARA MAMAK ŞAHİN TEPE MAHALLESİ ORTA KAT 19.000 KİRLIK DAİREMİZ 3+1 OLUP 2 BALKONLUDUR.KATTA OLUP BAĞIMSIZ SALONDUR. MEMLUN KALINDIĞI SÜRECE UZUN ZAMAN OTURALABİLİR.KİRAMIZ...          20 {"İlan no": "14225-3100", "Son Güncelleme": "31/03/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Konut Şek... 2026-04-01T20:02:00
```

### `train_ready_multimodal.parquet`

- Absolute path: `E:\rent-agent\dataset\train_ready_multimodal.parquet`
- Role guess: Train-ready multimodal table
- Shape: **6,389 rows x 28 columns**
- Duplicate rows: no
- Listing-like columns: `listing_id`, `listing_no`
- Unique `listing_id` count: **6,389**
- Possible price target: `price_try`
- Visual local path columns: `valid_image_paths`
- Visual URL columns: none
- Text columns: `title`, `description`, `raw_specs_json`
- Trainability: Evet. Multimodal egitim icin dogrudan aday.

#### Columns

`listing_id`, `url`, `listing_no`, `price_try`, `city`, `district`, `neighborhood`, `rooms`, `bathrooms`, `m2_gross`, `m2_net`, `building_age`, `floor`, `total_floors`, `heating_type`, `fuel_type`, `is_furnished`, `dues_try`, `home_type`, `home_shape`, `updated_at`, `title`, `description`, `image_count`, `raw_specs_json`, `scraped_at`, `valid_image_paths`, `valid_image_count`

#### Dtypes

| Column | Dtype |
| --- | --- |
| listing_id | object |
| url | object |
| listing_no | object |
| price_try | float64 |
| city | object |
| district | object |
| neighborhood | object |
| rooms | object |
| bathrooms | float64 |
| m2_gross | float64 |
| m2_net | object |
| building_age | float64 |
| floor | object |
| total_floors | int64 |
| heating_type | object |
| fuel_type | object |
| is_furnished | object |
| dues_try | float64 |
| home_type | object |
| home_shape | object |
| updated_at | object |
| title | object |
| description | object |
| image_count | int64 |
| raw_specs_json | object |
| scraped_at | object |
| valid_image_paths | object |
| valid_image_count | int64 |

#### Missing Values

| Column | Missing |
| --- | --- |
| m2_net | 6,389 |
| fuel_type | 2,606 |
| dues_try | 2,126 |
| home_shape | 1,034 |
| building_age | 779 |
| is_furnished | 278 |
| bathrooms | 80 |
| city | 0 |
| description | 0 |
| district | 0 |
| floor | 0 |
| heating_type | 0 |
| home_type | 0 |
| image_count | 0 |
| listing_id | 0 |
| listing_no | 0 |
| m2_gross | 0 |
| neighborhood | 0 |
| price_try | 0 |
| raw_specs_json | 0 |
| rooms | 0 |
| scraped_at | 0 |
| title | 0 |
| total_floors | 0 |
| updated_at | 0 |
| url | 0 |
| valid_image_count | 0 |
| valid_image_paths | 0 |

#### First 5 Rows

```text
           listing_id                                                                               url listing_no price_try   city district       neighborhood rooms bathrooms m2_gross m2_net building_age   floor total_floors heating_type fuel_type is_furnished dues_try home_type home_shape updated_at                                                                          title                                                                                                              description image_count                                                                                                           raw_specs_json          scraped_at                                                                                                        valid_image_paths valid_image_count
hepsiemlak:49652-2518 https://www.hepsiemlak.com/ankara-kecioren-asagi-eglence-kiralik/daire/49652-2518 49652-2518   29000.0 Ankara Keçiören Aşağı Eğlence Mah.   4+1       2.0    155.0     NA         21.0  2. Kat            2        Kombi        NA        False       NA     Daire   Tripleks 01/04/2026                  ERKA'DAN AŞAĞIEĞLENCE'DE 4+1 BAĞIMSIZ GENİŞ 2 BANYOLU DUBLEKS ERKA'DAN AŞAĞIEĞLENCE SARAR SOKAK'TA CADDEYE 1 BİNA MESAFEDE 4+1 BAĞIMSIZ BAKIMLI 2.KAT 2 BANYOLU GENİŞ M2'Lİ BALKONL...          20 {"İlan no": "49652-2518", "Son Güncelleme": "01/04/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Konut Şek... 2026-04-01T19:55:49 ["dataset\\images\\hepsiemlak_49652-2518\\004.jpg", "dataset\\images\\hepsiemlak_49652-2518\\005.jpg", "dataset\\imag...                16
 hepsiemlak:6664-2537     https://www.hepsiemlak.com/ankara-cankaya-mebusevleri-kiralik/daire/6664-2537  6664-2537   40000.0 Ankara  Çankaya   Mebusevleri Mah.   3+1       1.0    115.0     NA         31.0  3. Kat            4        Kombi        NA         True   1000.0     Daire    Ara Kat 24/03/2026                                       EŞYALI MEBUSEVLERİ NDE 3+1 ARA KAT DAİRE          BEŞEVLER METROSUNA VE ANADOLU METROSUNA YAKIN KONUMDA BAKIMLI BİR BİNADA BULUNMAKTADIR. OTURUMA HAZIR HALDEDİR.          20 {"İlan no": "6664-2537", "Son Güncelleme": "24/03/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Konut Şekl... 2026-04-01T19:57:29 ["dataset\\images\\hepsiemlak_6664-2537\\004.jpg", "dataset\\images\\hepsiemlak_6664-2537\\005.jpg", "dataset\\images...                11
 hepsiemlak:6664-2480      https://www.hepsiemlak.com/ankara-altindag-hacettepe-kiralik/daire/6664-2480  6664-2480   28000.0 Ankara Altındağ     Hacettepe Mah.   3+1       1.0    120.0     NA         30.0  1. Kat            2        Kombi  Doğalgaz        False       NA     Daire      Daire 24/03/2026 CEBECİ DUMLUPINAR CAD. MİLLET BAHÇESİ CEPHELİ 3+1 KATTA ÖN CEPHE KOMBİLİ DAİRE DAİRE DUMLUPUNAR CADDESİ ÜSTÜNDE VE MİLLET BAHÇESİ MANZARALIDIR. TALATPAŞA BULVARI VE KURTULUŞ METRO İSTASYONUNA YAKI...          20 {"İlan no": "6664-2480", "Son Güncelleme": "24/03/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Konut Şekl... 2026-04-01T19:59:15 ["dataset\\images\\hepsiemlak_6664-2480\\004.jpg", "dataset\\images\\hepsiemlak_6664-2480\\005.jpg", "dataset\\images...                12
 hepsiemlak:1920-2929         https://www.hepsiemlak.com/ankara-cankaya-aziziye-kiralik/daire/1920-2929  1920-2929   41000.0 Ankara  Çankaya       Aziziye Mah.   3+1       1.0    110.0     NA         31.0  4. Kat            4        Kombi        NA        False    650.0     Daire         NA 31/03/2026               Hoşdere Kiralık Daire Çankaya 3+1 Asansörlü Otoparklı HemenTaşın Hoşdere Kiralık Daire Çankaya 3+1 Asansörlü Otoparklı HemenTaşın Aziziye Mahallesi'nin Kalbinde, Huzurlu ve Konforlu ...          20 {"İlan no": "1920-2929", "Son Güncelleme": "31/03/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Oda Sayısı... 2026-04-01T20:00:33 ["dataset\\images\\hepsiemlak_1920-2929\\004.jpg", "dataset\\images\\hepsiemlak_1920-2929\\005.jpg", "dataset\\images...                16
hepsiemlak:14225-3100        https://www.hepsiemlak.com/ankara-mamak-sahintepe-kiralik/daire/14225-3100 14225-3100   19000.0 Ankara    Mamak     Şahintepe Mah.   3+1       1.0    125.0     NA         15.0 Ara Kat            3        Kombi  Doğalgaz        False    150.0     Daire    Ara Kat 31/03/2026                       ANKARA MAMAK ŞAHİN TEPE MAHALLESİ ORTA KAT 19.000 KİRLIK DAİREMİZ 3+1 OLUP 2 BALKONLUDUR.KATTA OLUP BAĞIMSIZ SALONDUR. MEMLUN KALINDIĞI SÜRECE UZUN ZAMAN OTURALABİLİR.KİRAMIZ...          20 {"İlan no": "14225-3100", "Son Güncelleme": "31/03/2026", "İlan Durumu": "Kiralık", "Konut Tipi": "Daire", "Konut Şek... 2026-04-01T20:02:00 ["dataset\\images\\hepsiemlak_14225-3100\\004.jpg", "dataset\\images\\hepsiemlak_14225-3100\\005.jpg", "dataset\\imag...                16
```

#### Image Path Audit

- Column `valid_image_paths` total paths: **99,392**
- Existing local files: **99,392**
- Broken local files: **0**
- Average valid images per listing: **15.56**
- Listings with at least one valid image: **6,389**

### `validation_report.parquet`

- Absolute path: `E:\rent-agent\dataset\validation_report.parquet`
- Role guess: Validation/report table
- Shape: **6,875 rows x 18 columns**
- Duplicate rows: no
- Listing-like columns: `listing_id`
- Unique `listing_id` count: **6,875**
- Possible price target: `price_per_m2` (derived metric)
- Visual local path columns: none
- Visual URL columns: none
- Text columns: none
- Trainability: Kismen. Kalite filtresi ve train-ready kararlarini destekler, ama ana egitim tablosu degil.

#### Columns

`listing_id`, `has_price`, `has_location`, `has_rooms`, `has_m2`, `has_description`, `has_min_images`, `all_images_valid`, `valid_image_count`, `price_per_m2`, `is_outlier_price`, `is_outlier_m2`, `is_outlier_price_per_m2`, `is_train_ready_ml`, `is_train_ready_dl`, `is_train_ready_multimodal`, `reject_reason`, `validated_at`

#### Dtypes

| Column | Dtype |
| --- | --- |
| listing_id | object |
| has_price | bool |
| has_location | bool |
| has_rooms | bool |
| has_m2 | bool |
| has_description | bool |
| has_min_images | bool |
| all_images_valid | bool |
| valid_image_count | int64 |
| price_per_m2 | float64 |
| is_outlier_price | bool |
| is_outlier_m2 | bool |
| is_outlier_price_per_m2 | bool |
| is_train_ready_ml | bool |
| is_train_ready_dl | bool |
| is_train_ready_multimodal | bool |
| reject_reason | object |
| validated_at | object |

#### Missing Values

| Column | Missing |
| --- | --- |
| reject_reason | 6,397 |
| price_per_m2 | 337 |
| all_images_valid | 0 |
| has_description | 0 |
| has_location | 0 |
| has_m2 | 0 |
| has_min_images | 0 |
| has_price | 0 |
| has_rooms | 0 |
| is_outlier_m2 | 0 |
| is_outlier_price | 0 |
| is_outlier_price_per_m2 | 0 |
| is_train_ready_dl | 0 |
| is_train_ready_ml | 0 |
| is_train_ready_multimodal | 0 |
| listing_id | 0 |
| valid_image_count | 0 |
| validated_at | 0 |

#### First 5 Rows

```text
           listing_id has_price has_location has_rooms has_m2 has_description has_min_images all_images_valid valid_image_count       price_per_m2 is_outlier_price is_outlier_m2 is_outlier_price_per_m2 is_train_ready_ml is_train_ready_dl is_train_ready_multimodal reject_reason        validated_at
hepsiemlak:49652-2518      True         True      True   True            True           True            False                16 187.09677419354838            False         False                   False              True              True                      True            NA 2026-04-01T19:57:22
 hepsiemlak:6664-2537      True         True      True   True            True           True            False                11 347.82608695652175            False         False                   False              True              True                      True            NA 2026-04-01T19:59:09
 hepsiemlak:6664-2480      True         True      True   True            True           True            False                12 233.33333333333334            False         False                   False              True              True                      True            NA 2026-04-01T20:00:26
 hepsiemlak:1920-2929      True         True      True   True            True           True            False                16 372.72727272727275            False         False                   False              True              True                      True            NA 2026-04-01T20:01:54
hepsiemlak:14225-3100      True         True      True   True            True           True            False                16              152.0            False         False                   False              True              True                      True            NA 2026-04-01T20:03:12
```

## Cross-File Relationships

### `images.parquet` <-> `listings.parquet`

- Common columns (2): `listing_id`, `scraped_at`
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,561**
- Coverage: left=100.00%, right=95.43%
- Relationship shape guess: many-to-one from images.parquet to listings.parquet

### `images.parquet` <-> `run_log.parquet`

- Common columns (1): `listing_id`
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,561**
- Coverage: left=100.00%, right=94.81%
- Relationship shape guess: many-to-many / event-style

### `images.parquet` <-> `train_ready_ml.parquet`

- Common columns (2): `listing_id`, `scraped_at`
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,429**
- Coverage: left=97.99%, right=98.94%
- Relationship shape guess: many-to-one from images.parquet to train_ready_ml.parquet

### `images.parquet` <-> `train_ready_multimodal.parquet`

- Common columns (2): `listing_id`, `scraped_at`
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,389**
- Coverage: left=97.38%, right=100.00%
- Relationship shape guess: many-to-one from images.parquet to train_ready_multimodal.parquet

### `images.parquet` <-> `validation_report.parquet`

- Common columns (1): `listing_id`
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,561**
- Coverage: left=100.00%, right=95.43%
- Relationship shape guess: many-to-one from images.parquet to validation_report.parquet

### `listings.parquet` <-> `run_log.parquet`

- Common columns (2): `listing_id`, `url`
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,875**
- Coverage: left=100.00%, right=99.35%
- Relationship shape guess: many-to-one from run_log.parquet to listings.parquet

### `listings.parquet` <-> `train_ready_ml.parquet`

- Common columns (26): `bathrooms`, `building_age`, `city`, `description`, `district`, `dues_try`, `floor`, `fuel_type`, `heating_type`, `home_shape`, `home_type`, `image_count`, ...
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,498**
- Coverage: left=94.52%, right=100.00%
- Relationship shape guess: mostly one-to-one

### `listings.parquet` <-> `train_ready_multimodal.parquet`

- Common columns (26): `bathrooms`, `building_age`, `city`, `description`, `district`, `dues_try`, `floor`, `fuel_type`, `heating_type`, `home_shape`, `home_type`, `image_count`, ...
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,389**
- Coverage: left=92.93%, right=100.00%
- Relationship shape guess: mostly one-to-one

### `listings.parquet` <-> `validation_report.parquet`

- Common columns (1): `listing_id`
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,875**
- Coverage: left=100.00%, right=100.00%
- Relationship shape guess: mostly one-to-one

### `run_log.parquet` <-> `train_ready_ml.parquet`

- Common columns (2): `listing_id`, `url`
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,498**
- Coverage: left=93.90%, right=100.00%
- Relationship shape guess: many-to-one from run_log.parquet to train_ready_ml.parquet

### `run_log.parquet` <-> `train_ready_multimodal.parquet`

- Common columns (2): `listing_id`, `url`
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,389**
- Coverage: left=92.33%, right=100.00%
- Relationship shape guess: many-to-one from run_log.parquet to train_ready_multimodal.parquet

### `run_log.parquet` <-> `validation_report.parquet`

- Common columns (2): `listing_id`, `reject_reason`
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,875**
- Coverage: left=99.35%, right=100.00%
- Relationship shape guess: many-to-one from run_log.parquet to validation_report.parquet

### `train_ready_ml.parquet` <-> `train_ready_multimodal.parquet`

- Common columns (26): `bathrooms`, `building_age`, `city`, `description`, `district`, `dues_try`, `floor`, `fuel_type`, `heating_type`, `home_shape`, `home_type`, `image_count`, ...
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,389**
- Coverage: left=98.32%, right=100.00%
- Relationship shape guess: mostly one-to-one

### `train_ready_ml.parquet` <-> `validation_report.parquet`

- Common columns (1): `listing_id`
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,498**
- Coverage: left=100.00%, right=94.52%
- Relationship shape guess: mostly one-to-one

### `train_ready_multimodal.parquet` <-> `validation_report.parquet`

- Common columns (2): `listing_id`, `valid_image_count`
- Joinable on `listing_id`: yes
- Shared listing_id count: **6,389**
- Coverage: left=100.00%, right=92.93%
- Relationship shape guess: mostly one-to-one

## File Roles

- Main raw listing table: `listings.parquet`
- Image table: `images.parquet`
- Validation/report table: `validation_report.parquet`
- Train-ready tabular/text table: `train_ready_ml.parquet`
- Train-ready multimodal table: `train_ready_multimodal.parquet`
- Crawl/process log table: `run_log.parquet`

## Consistency Notes

- `run_log.parquet` contains 45 listing_id values that do not appear in the raw main listing table.
- `validation_report.parquet` marks 6,506 listings as `is_train_ready_ml=True`, but `train_ready_ml.parquet` contains 6,498; 8 flagged IDs are missing from the train-ready file.
- `validation_report.parquet` marks 6,397 listings as `is_train_ready_multimodal=True`, but `train_ready_multimodal.parquet` contains 6,389; 8 flagged IDs are missing from the train-ready multimodal file.

## Dataset Readiness

- Tabular model: Evet. `train_ready_ml.parquet` fiyat hedefi ve temel yapisal kolonlarla hazir.
- Text branch: Evet. `title` ve `description` kolonlari train-ready dosyalarda mevcut.
- Image branch: Evet. `images.parquet` uzerinden yerel goruntuler bulunuyor; ancak bozuk pathler de var.
- Multimodal fusion: Evet. `train_ready_multimodal.parquet` fiyat + metin + goruntu pathlerini bir araya getiriyor.

## Recommendations

- Suggested target: `price_try`
- Baseline model feature columns: `city`, `district`, `neighborhood`, `rooms`, `bathrooms`, `m2_gross`, `building_age`, `floor`, `total_floors`, `heating_type`, `fuel_type`, `is_furnished`, `dues_try`, `home_type`, `home_shape`, `image_count`
- Multimodal model feature columns: `city`, `district`, `neighborhood`, `rooms`, `bathrooms`, `m2_gross`, `building_age`, `floor`, `total_floors`, `heating_type`, `fuel_type`, `is_furnished`, `dues_try`, `home_type`, `home_shape`, `image_count`, `title`, `description`, `valid_image_paths`, `valid_image_count`
- Preprocessing note: `rooms` and `floor` still look string-like; `m2_net` appears fully missing in the main listing tables and should be dropped or reconstructed before training.
- Join note: `listings.parquet` is the raw source of truth, while `train_ready_*` tables are filtered training subsets.

## Errors

- No parquet read errors occurred.
