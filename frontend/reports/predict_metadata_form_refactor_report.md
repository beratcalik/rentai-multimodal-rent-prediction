# Predict Metadata Form Refactor Report

## Kapsam

Bu revizyonda `/predict` sayfasındaki serbest metin ağırlıklı form, eğitim verisinden türetilmiş metadata JSON dosyalarını kullanan ürün seviyesinde seçilebilir akışa taşındı.

Değişiklikler yalnızca `frontend/` altında yapıldı. Backend sözleşmesi korunarak gerçek `POST /predict` endpoint'ine aynı form alanları gönderilmeye devam ediyor.

## Kullanılan metadata dosyaları

- `public/meta/locations.json`
- `public/meta/categorical-options.json`
- `public/meta/numeric-ranges.json`

Bu dosyalar `lib/meta/load-metadata.ts` üzerinden yüklenir ve normalize edilir.

### Kullanım mantığı

- `locations.json`
  - `city -> district -> neighborhood` bağımlı seçim akışı için kullanılır.
  - Mahalle listeleri metadata içindeki frekans sırasını korur.
- `categorical-options.json`
  - Select / searchable select alanlarının dataset tabanlı seçeneklerini sağlar.
  - `null` değerler frontend tarafında `Belirtilmemiş` etiketiyle gösterilir.
  - Backend gönderiminde `null` değerler boş stringe normalize edilir.
- `numeric-ranges.json`
  - `m2_gross` ve `dues_try` alanları için örnek placeholder ve tipik aralık yardımcı metni üretir.
  - `p05-p95` dışı değerlerde hata yerine nazik uyarı gösterilir.

## Select / combobox alanları

Serbest metin olmaktan çıkarılan alanlar:

- `city`
- `district`
- `neighborhood`
- `rooms`
- `bathrooms`
- `building_age`
- `floor`
- `total_floors`
- `heating_type`
- `fuel_type`
- `is_furnished`
- `home_type`
- `home_shape`

Serbest bırakılan alanlar:

- `m2_gross`
- `dues_try`
- `title`
- `description`

## Backend contract

Frontend form submit'i aşağıdaki alan adlarını korur:

- `city`
- `district`
- `neighborhood`
- `rooms`
- `bathrooms`
- `m2_gross`
- `building_age`
- `floor`
- `total_floors`
- `heating_type`
- `fuel_type`
- `is_furnished`
- `dues_try`
- `home_type`
- `home_shape`
- `title`
- `description`
- `images`

Yani backend tarafında herhangi bir endpoint veya schema değişikliği gerekmez.

## Result range mantığı

Backend şu an tek bir merkez tahmin döndürür:

- `predicted_rent_try`

Frontend bunu ürün diliyle aşağıdaki aralığa dönüştürür:

- `lower = predicted_rent_try * 0.90`
- `upper = predicted_rent_try * 1.10`

Bu aralık:

- `Beklenen kira aralığı` başlığıyla gösterilir
- güven aralığı veya model confidence olarak sunulmaz
- yalnızca karar destek amaçlı UX sunumu olarak kullanılır

## Eklenen / güncellenen dosyalar

Yeni dosyalar:

- `lib/meta/load-metadata.ts`
- `components/predict/LocationSelector.tsx`
- `components/predict/MetadataSelect.tsx`
- `components/predict/NumericField.tsx`
- `reports/predict_metadata_form_refactor_report.md`

Güncellenen dosyalar:

- `components/predict/PredictWorkspace.tsx`
- `components/predict/PredictionResultCard.tsx`
- `lib/api/prediction.ts`
- `lib/validation/prediction-schema.ts`
- `app/predict/page.tsx`
- `lib/utils.ts`

## Test sonuçları

Doğrulanan başlıklar:

- `npm run build` geçti.
- `/predict` route'u build çıktısında başarıyla üretildi.
- `npm run dev` başarılı şekilde ayağa kalktı ve `http://localhost:3000` adresini yayınladı.
- Metadata yükleme akışı public JSON dosyalarına bağlandı.
- `district` değişiminde `neighborhood` reset mantığı forma işlendi.
- Görsel limiti `16` olarak korunuyor.
- Frontend API client, gerçek backend ile canlı olarak test edildi.

Canlı API client doğrulaması:

- `createPrediction(...)` çağrısı gerçek backend'e gönderildi.
- Örnek canlı sonuç:
  - `predicted_rent_try: 45340`
  - `predicted_rent_formatted: 45.340 TL`
  - `used_image_count: 0`
  - `model_name: XGBRegressor`

## Notlar

- Backend kapalı senaryosu için kullanıcı dostu hata mesajı korunmuştur.
- Metadata yüklenene kadar form skeleton gösterilir.
- İlk geçersiz alana scroll davranışı eklendi.
- Sonuç paneli tek fiyat yerine ürün diliyle kira aralığı sunacak şekilde güncellendi.
