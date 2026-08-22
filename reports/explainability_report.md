# Explainability Report

## Amaç

Final multimodal kira tahmin sistemine, tek ilan seviyesinde neden-sonuç açıklaması üreten bir XAI katmanı eklendi.

Yeni akışın hedefi:

- Tahmini kira sonucunu korumak
- Tahmini etkileyen ana faktörleri kullanıcı dostu dilde göstermek
- Mevcut `/predict` endpointini bozmadan ayrı bir explainable endpoint sunmak

## Kullanılan yaklaşım

### 1. SHAP tabanlı local explanation

- Model: `models/final_multimodal_text_clip_model.joblib`
- Regressor: `XGBRegressor`
- Açıklama yöntemi: `shap.TreeExplainer`

Not:

- Mevcut XGBoost model bundle’ı `base_score` değerini köşeli parantezli (`[3.394937E4]`) biçimde serialize ettiği için `shap 0.49.x` ile doğrudan uyumsuzluk oluşuyordu.
- Bu yüzden `src/generate_prediction_explanations.py` içinde küçük bir compatibility patch eklendi.
- Patch, `TreeExplainer` akışını değiştirmiyor; yalnızca SHAP loader içindeki `base_score` parse adımını XGBoost 3 serisinin bu serialization biçimine uyumlu hale getiriyor.

### 2. Modalite bazlı explanation gruplama

Ham SHAP feature isimleri son kullanıcıya verilmedi. Bunun yerine katkılar üç üst gruba çevrildi:

- Tabular katkılar
- Text katkıları
- Image katkıları

Tabular tarafta SHAP skorları aşağıdaki kullanıcı dostu gruplara toplandı:

- Lokasyon
- Oda planı
- Brüt m²
- Banyo
- Bina yaşı
- Kat / toplam kat
- Isıtma / yakıt
- Eşyalı durumu
- Aidat
- Konut tipi / konut şekli

### 3. Text explanation

Model text branch’i `TF-IDF + TruncatedSVD` kullandığı için kullanıcıya doğrudan `text_svd_012` gibi teknik feature isimleri gösterilmiyor.

Bunun yerine:

- ilgili local SHAP skorları text bloğundan alındı
- SVD bileşenleri üzerinden yaklaşık token katkısı geri projekte edildi
- en güçlü pozitif ve negatif token/ifade grupları çıkarıldı
- boilerplate ve anlamsız listing kelimeleri filtrelendi

Örnek kullanıcı cümlesi:

- `İlan metnindeki “ankastre”, “lüks” ve “metro” ifadeleri tahmini yukarı çekiyor.`

### 4. Image explanation

Image branch’te doğrudan PCA feature isimleri kullanıcıya gösterilmedi.

Onun yerine:

- image SHAP bloğu ayrı toplandı
- CLIP mean embedding üzerinden küçük bir prompt bank ile semantik yorum üretildi
- sonuç kullanıcı dostu cümleye çevrildi

Örnek kullanıcı cümleleri:

- `Görseller daha modern ve bakımlı bir ev algısı oluşturuyor.`
- `Görseller daha karanlık alanlar sinyali veriyor.`

## Confidence score

Response içine `confidence_score` eklendi.

Bu skor:

- istatistiksel güven aralığı değildir
- calibrated uncertainty değildir
- kullanıcıya tek başına “kesinlik” olarak sunulmamalıdır

Skor, şu iki bileşenin birleşiminden türetilen heuristik bir sinyaldir:

- veri tamlığı
- modalite ablation stabilitesi

Kullanılan ablation karşılaştırmaları:

- full
- no_text
- no_image
- tabular_only

## Eklenen backend endpoint

Yeni endpoint:

- `POST /predict-with-explanations`

Eski endpoint korunmuştur:

- `POST /predict`

Yeni response alanları:

```json
{
  "predicted_rent_try": 46210,
  "predicted_rent_formatted": "46.210 TL",
  "used_image_count": 0,
  "model_name": "XGBRegressor",
  "warnings": [],
  "message": "Tahmin ilan bilgileri, açıklama metni ve fotoğraflar birlikte analiz edilerek üretildi.",
  "confidence_score": 81,
  "top_positive_factors": [
    "Brüt 155 m² büyüklük tahmini yukarı çekiyor.",
    "2 banyo bilgisi tahmini yukarı çekiyor."
  ],
  "top_negative_factors": [
    "Çankaya / Ayrancı Mah. lokasyonu tahmini aşağı çekiyor."
  ]
}
```

## Örnek explanation çıktıları

### Örnek 1

`examples/sample_listing_input.json` ile çalıştırılan sonuç:

- Tahmin: `31.246 TL`
- Güven skoru: `91/100`
- Pozitif faktörler:
  - `Brüt 155 m² büyüklük tahmini yukarı çekiyor.`
  - `2 banyo bilgisi tahmini yukarı çekiyor.`
  - `4+1 oda planı tahmini yukarı çekiyor.`
- Negatif faktörler:
  - `Görseller daha karanlık alanlar sinyali veriyor.`
  - `Keçiören / Aşağı Eğlence Mah. lokasyonu tahmini aşağı çekiyor.`

### Örnek 2

Çankaya / Ayrancı, lüks-ankastre-metro vurgulu örnek form-data ile:

- Tahmin: `46.210 TL`
- Güven skoru: `81/100`
- Pozitif faktörler:
  - `Brüt 155 m² büyüklük tahmini yukarı çekiyor.`
  - `2 banyo bilgisi tahmini yukarı çekiyor.`
  - `4+1 oda planı tahmini yukarı çekiyor.`
  - `İlan metnindeki “ankastre”, “site” ve benzeri ifadeler tahmini yukarı çekiyor.`
- Negatif faktörler:
  - `Kombi tahmini aşağı çekiyor.`
  - `Aidat seviyesi tahmini aşağı çekiyor.`
  - `5. Kat ve 8 katlı bina bilgisi tahmini aşağı çekiyor.`

## Inference latency etkisi

Doğrulama ortamı:

- yerel FastAPI `TestClient`
- warm cache ölçümü
- CPU fallback

Warm cache ortalama süreleri:

- `/predict`: yaklaşık `0.054 sn`
- `/predict-with-explanations`: yaklaşık `0.075 sn`

Yorum:

- explainability katmanı warm durumda yaklaşık `20-25 ms` ek yük getiriyor
- esas cold-start maliyeti CLIP runtime ve model cache yüklemesinden geliyor
- SHAP explainer cache ile tutulduğu için tekrar eden isteklerde maliyet sınırlı kalıyor

## Frontend entegrasyonu

Frontend tarafında explainability akışı şu şekilde eklendi:

- API client yeni proxy route üzerinden `/api/predict-with-explanations` çağırıyor
- `PredictionResultCard` içine `Tahmini etkileyen faktörler` alanı eklendi
- en fazla 5 factor gösteriliyor
- pozitif factorlar yeşil
- negatif factorlar kırmızı
- teknik `model_name` son kullanıcıya ana yüzeyde öne çıkarılmıyor

Güncellenen önemli frontend dosyaları:

- `frontend/app/api/predict-with-explanations/route.ts`
- `frontend/lib/api/prediction.ts`
- `frontend/lib/validation/prediction-schema.ts`
- `frontend/components/predict/PredictionResultCard.tsx`

## Güncellenen backend / inference dosyaları

- `src/predict_single_listing.py`
- `src/generate_prediction_explanations.py`
- `backend/inference_service.py`
- `backend/main.py`
- `backend/schemas.py`

## Doğrulama sonuçları

Tamamlanan kontroller:

- `python -B src/predict_single_listing.py --input examples/sample_listing_input.json`
- `python -B src/generate_prediction_explanations.py --input examples/sample_listing_input.json --json`
- `POST /predict-with-explanations` gerçek form-data ile test edildi
- görselsiz submit test edildi
- görselli submit test edildi
- eski `/predict` endpointinin bozulmadığı doğrulandı
- `npm run build` geçti

Ek notlar:

- Bu oturumdaki doğrulamada CUDA runtime hazır olmadığı için inference CPU fallback ile çalıştı.
- `pandas` tarafında `numexpr` ve `bottleneck` için uyarı mesajları gözlendi; bunlar explainability akışını durdurmadı.

## Sonuç

Sistem artık yalnızca kira tahmini üretmiyor; aynı zamanda bu tahmini hangi lokasyon, metin ve görsel sinyallerinin yukarı veya aşağı çektiğini de kullanıcı dostu biçimde açıklayabiliyor.

Bu sayede ürün:

- daha güven veren
- karar destek açısından daha açıklanabilir
- frontend ürün deneyimine daha uygun

bir yapıya taşındı.
