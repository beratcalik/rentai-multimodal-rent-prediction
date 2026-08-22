# Confidence Score Report

## Amaç

Explainability akışındaki güven skoru katmanı güçlendirildi.

Yeni yaklaşımın hedefi:

- tek bir tahmin için daha sistematik bir `confidence_score` üretmek
- skoru kullanıcıya istatistiksel kesinlik gibi değil, karar destek güven göstergesi olarak sunmak
- güven skorunun neden yüksek veya düşük olduğunu kısa ve anlaşılır maddelerle açıklamak

## Yeni modül

Eklenen modül:

- `src/confidence_estimation.py`

Bu modül, tek ilan için aşağıdaki çıktıları üretir:

- `confidence_score`
- `confidence_label`
- `confidence_reasons`

## Kullanılan sinyaller

Confidence skoru beş sinyalden hesaplanır.

### 1. Veri tamlığı

Kontrol edilen ana alanlar:

- `city`
- `district`
- `neighborhood`
- `rooms`
- `m2_gross`
- `title`
- `description`
- temizlenmiş metin boş mu
- görsel var mı

Eksik temel alanlar, confidence skorunu düşürür.

### 2. Görsel yeterliliği

`used_image_count` üzerinden görsel katkı puanı üretilir.

Temel mantık:

- `0 görsel` → düşük
- `1-3 görsel` → düşük / orta
- `4-8 görsel` → orta
- `9-16 görsel` → yüksek

### 3. Lokasyon yoğunluğu

Eğitim verisi:

- `dataset/train_ready_multimodal.parquet`

Runtime’da bir kez cache’lenir ve şu istatistikler çıkarılır:

- district frequency
- neighborhood frequency
- price `p05`
- price `p95`

Lokasyon confidence mantığı:

- eğitim verisinde çok görülen ilçe / mahalle → daha yüksek güven
- çok az görülen mahalle → daha düşük güven

### 4. Tahminin fiyat dağılımına uygunluğu

Eğitim verisindeki fiyat dağılımında:

- `p05 = 18.000 TL`
- `p95 = 65.000 TL`

Kurgu:

- tahmin bu tipik aralık içindeyse confidence artar
- tahmin `p95` üstüne veya `p05` altına kaydıkça confidence düşer

### 5. Ablation stabilitesi

Explainability akışındaki şu varyantlar kullanılır:

- `full`
- `no_text`
- `no_image`
- `tabular_only`

Yorum:

- modaliteler kapatıldığında tahmin çok oynuyorsa confidence düşer
- daha stabil kalıyorsa confidence artar

## Ağırlıklar

Toplam skor `0-100` bandında normalize edilir.

- Veri tamlığı: `%25`
- Görsel yeterliliği: `%20`
- Lokasyon yoğunluğu: `%20`
- Fiyat dağılımına uygunluk: `%20`
- Ablation stabilitesi: `%15`

## Label eşikleri

- `80-100` → `Yüksek`
- `60-79` → `Orta`
- `0-59` → `Düşük`

## Kullanıcıya nasıl sunuluyor

Frontend yüzeyinde başlık:

- `Tahmin güven seviyesi`

Gösterilen alanlar:

- label: `Yüksek / Orta / Düşük`
- score: örn. `84/100`
- kısa progress bar
- en fazla 3 açıklama maddesi

Kullanıcı dilinde açıklama:

- `Bu skor, girilen bilgilerin veri setindeki benzer örneklerle ne kadar uyumlu olduğunu gösterir.`

Önemli:

- bu skor accuracy değildir
- bu skor kesinlik oranı değildir
- bu skor resmi ekspertiz sonucu değildir

## Endpoint değişikliği

Yeni alanlar yalnızca explainable endpointte döner:

- `POST /predict-with-explanations`

Eklenen response alanları:

```json
{
  "confidence_score": 84,
  "confidence_label": "Yüksek",
  "confidence_reasons": [
    "Konum bilgileri eğitim verisinde güçlü temsil ediliyor.",
    "Yeterli sayıda fotoğraf analiz edildi.",
    "Tahmin fiyatı eğitim verisindeki tipik aralıkta."
  ]
}
```

Mevcut endpoint korunmuştur:

- `POST /predict`

Bu endpoint bozulmadı ve eski sade response’u döndürmeye devam ediyor.

## Eski confidence mantığından farkı

Eski yaklaşım:

- veri tamlığı
- modalite ablation stabilitesi

Yeni yaklaşım:

- veri tamlığı
- görsel yeterliliği
- lokasyon yoğunluğu
- fiyat dağılım uygunluğu
- ablation stabilitesi

Yani yeni skor:

- daha sistematik
- eğitim verisine daha bağlı
- kullanıcıya nedenleriyle açıklanabilir

## Test senaryoları

### 1. Tam veri + 10+ görsel + yoğun lokasyon

Örnek sonuç:

- confidence: `80`
- label: `Yüksek`

Örnek nedenler:

- `İlan bilgileri büyük ölçüde eksiksiz paylaşıldı.`
- `Tahmin fiyatı eğitim verisindeki tipik aralıkta.`
- `Yeterli sayıda fotoğraf analiz edildi.`

### 2. Görselsiz ama tam tabular/text veri

Örnek sonuç:

- confidence: `63`
- label: `Orta`

Örnek nedenler:

- `Tahmin fiyatı eğitim verisindeki tipik aralıkta.`
- `Seçilen konum eğitim verisinde daha sınırlı temsil ediliyor.`
- `Fotoğraf sayısı sınırlı olduğu için görsel sinyal zayıf kalıyor.`

### 3. Eksik description + az görsel + seyrek lokasyon

Örnek sonuç:

- confidence: `67`
- label: `Orta`

Yorum:

- lokasyon seyrekliği ve sınırlı fotoğraf skoru aşağı çekiyor
- yine de fiyat tipik aralıkta kaldığı için skor tamamen düşüğe inmiyor

### 4. Outlier benzeri yüksek fiyat örneği

Örnek sonuç:

- predicted rent: `78.743 TL`
- confidence: `77`
- label: `Orta`

Yorum:

- skor yüksek seviyeye çıkmıyor
- tipik fiyat aralığının üzerine taşındığı için güven seviyesi orta bandında kalıyor

## Canlı / script doğrulamaları

Çalıştırılan doğrulamalar:

```bash
python -B src/generate_prediction_explanations.py --input examples/sample_listing_input.json --json
```

```bash
npm run build
```

Ek backend doğrulamaları:

- `POST /predict-with-explanations` → `200 OK`
- `POST /predict` → `200 OK`

## Örnek script çıktısı

`examples/sample_listing_input.json` için:

```json
{
  "predicted_rent_try": 31246,
  "predicted_rent_formatted": "31.246 TL",
  "confidence_score": 96,
  "confidence_label": "Yüksek",
  "confidence_reasons": [
    "İlan bilgileri büyük ölçüde eksiksiz paylaşıldı.",
    "Tahmin fiyatı eğitim verisindeki tipik aralıkta.",
    "Yeterli sayıda fotoğraf analiz edildi."
  ]
}
```

## Güncellenen dosyalar

- `src/confidence_estimation.py`
- `src/generate_prediction_explanations.py`
- `backend/schemas.py`
- `backend/inference_service.py`
- `backend/main.py`
- `frontend/lib/validation/prediction-schema.ts`
- `frontend/lib/api/prediction.ts`
- `frontend/components/predict/PredictionResultCard.tsx`

## Sonuç

Confidence katmanı artık:

- rastgele veya süs amaçlı bir skor değil
- eğitim verisi ve modalite davranışıyla beslenen
- kullanıcıya sade dille açıklanabilen
- ürün seviyesinde kullanılabilir

bir güven göstergesi haline getirildi.
