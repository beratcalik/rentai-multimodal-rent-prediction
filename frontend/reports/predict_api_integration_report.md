## Predict API Integration Report

### Guncellenen dosyalar

- `app/predict/page.tsx`
- `components/predict/PredictWorkspace.tsx`
- `components/predict/ImageUploader.tsx`
- `components/predict/PredictionResultCard.tsx`
- `components/ui/native-select.tsx`
- `lib/api/prediction.ts`
- `lib/validation/prediction-schema.ts`

### Entegrasyon ozeti

`/predict` sayfasi artik gercek FastAPI backend endpoint'ine baglidir. Form verileri ve secilen gorseller `multipart/form-data` olarak `POST {NEXT_PUBLIC_API_BASE_URL}/predict` istegine donusturulur. Basarili response, sagdaki sonuc panelinde dogrudan backend cevabi ile gosterilir.

Akis:

1. React Hook Form + Zod ile alanlar dogrulanir.
2. `ImageUploader` en fazla 16 JPG/JPEG/PNG gorsel kabul eder.
3. Gorseller tarayici tarafinda 10MB sinirina gore kontrol edilir ve gerekirse sikistirilir.
4. TanStack Query mutation `createPrediction(...)` cagrisi yapar.
5. Backend response `PredictionResultCard` icinde loading, success veya error durumlarina gore sunulur.

### API client davranisi

- `NEXT_PUBLIC_API_BASE_URL` kullanilir. Bos ise varsayilan `http://127.0.0.1:8000` hedeflenir.
- `AbortController` ile timeout destegi vardir.
- Backend `400` detail mesaji kullaniciya aktarilir.
- Baglanti hatasi durumunda su mesaj gosterilir:
  `Tahmin servisine ulasilamadi. Lutfen API'nin calistigindan emin olun.`

### Form validasyonu

- `city`, `district`, `neighborhood`, `rooms`, `title`, `description` zorunludur.
- `m2_gross` pozitif sayi olmalidir.
- `images` en fazla 16 adet olabilir.
- Her gorsel `jpg`, `jpeg` veya `png` olmalidir.
- Her gorsel en fazla `10MB` olabilir.

### Calistirma adimlari

Backend:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Build kontrolu:

```bash
cd frontend
npm run build
```

### Ornek test senaryolari

1. Backend acikken formu doldurup 3 adet gorsel ile submit edin.
   Beklenen: `200 OK`, sonuc kartinda `predicted_rent_formatted`, `used_image_count`, `model_name` ve `message` gorunur.

2. Backend kapaliyken submit edin.
   Beklenen: sonuc kartinda kullanici dostu hata mesaji gorunur.

3. 16'dan fazla gorsel secmeyi deneyin.
   Beklenen: frontend dogrulamasi `En fazla 16 gorsel ekleyebilirsiniz.` hatasini gosterir.

4. Gecersiz dosya turu secin.
   Beklenen: sadece JPG/JPEG/PNG kabul edilir uyarisi gorunur.

### Dogrulanan durumlar

- `npm run build` basariyla gecti.
- Gercek backend HTTP endpoint'i `GET /health` ile `{"status":"ok"}` dondu.
- Gercek backend endpoint'ine frontend API client uzerinden istek atildi ve basarili response alindi:

```json
{
  "predicted_rent_try": 29764,
  "predicted_rent_formatted": "29.764 TL",
  "used_image_count": 3,
  "model_name": "XGBRegressor",
  "warnings": [],
  "message": "Tahmin ilan bilgileri, açıklama metni ve fotoğraflar birlikte analiz edilerek üretildi."
}
```

- Backend kapali senaryosunda kullanici dostu hata mesaji dogrulandi.
- 16 gorsel limiti frontend schema seviyesinde dogrulandi.

### Bilinen sinirlamalar

- Siralama icin su an yalnizca `one al` ve `geriye al` aksiyonlari var; drag-and-drop reorder yok.
- Tarayici tarafli gorsel sikistirma, buyuk gorsellerde submit oncesi kisa bir bekleme ekleyebilir.
- Bu adimda frontend yalnizca tek ilan tahmini akisina baglandi; batch inference arayuzu yok.
