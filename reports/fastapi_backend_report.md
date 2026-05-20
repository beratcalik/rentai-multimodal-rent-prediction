# FastAPI Backend Report

## Ozet

Bu katman, final multimodal kira tahmin modelini HTTP API olarak servis etmek icin olusturuldu.

- Backend package: `E:\rent-agent\backend`
- FastAPI app: `E:\rent-agent\backend\main.py`
- Inference service: `E:\rent-agent\backend\inference_service.py`
- Reusable predictor: `E:\rent-agent\src\predict_single_listing.py`
- Final model bundle: `E:\rent-agent\models\final_multimodal_text_clip_model.joblib`

## Calistirma

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## Endpointler

### `GET /`

Ornek response:

```json
{
  "name": "Rent Agent API",
  "status": "ok",
  "model": "final_multimodal_text_clip_model"
}
```

### `GET /health`

Ornek response:

```json
{
  "status": "ok"
}
```

### `POST /predict`

- `multipart/form-data` kabul eder
- Tabular + text alanlarini form-data ile alir
- `images` alaninda coklu dosya kabul eder
- `jpg`, `jpeg`, `png` allowlist uygular
- En fazla 16 gorsel kullanir

Ornek response:

```json
{
  "predicted_rent_try": 27916,
  "predicted_rent_formatted": "27.916 TL",
  "used_image_count": 3,
  "model_name": "XGBRegressor",
  "warnings": [],
  "message": "Tahmin ilan bilgileri, açıklama metni ve fotoğraflar birlikte analiz edilerek üretildi."
}
```

## Ornek curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "city=Ankara" \
  -F "district=Keçiören" \
  -F "neighborhood=Aşağı Eğlence Mah." \
  -F "rooms=4+1" \
  -F "bathrooms=2" \
  -F "m2_gross=155" \
  -F "building_age=21" \
  -F "floor=2. Kat" \
  -F "total_floors=2" \
  -F "heating_type=Kombi" \
  -F "fuel_type=" \
  -F "is_furnished=false" \
  -F "dues_try=" \
  -F "home_type=Daire" \
  -F "home_shape=Tripleks" \
  -F "title=ERKA'DAN AŞAĞIEĞLENCE'DE 4+1 BAĞIMSIZ GENİŞ 2 BANYOLU DUBLEKS" \
  -F "description=Bakimli, genis ve aile kullanimina uygun ornek ilan aciklamasi." \
  -F "images=@dataset/images/hepsiemlak_49652-2518/004.jpg;type=image/jpeg" \
  -F "images=@dataset/images/hepsiemlak_49652-2518/005.jpg;type=image/jpeg" \
  -F "images=@dataset/images/hepsiemlak_49652-2518/006.jpg;type=image/jpeg"
```

## Frontend Entegrasyon Notlari

- Frontend `.env.example` dosyasinda `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000` zaten hazir.
- `/predict` sayfasi sonraki adimda bu endpoint'e `multipart/form-data` gonderecek.
- Backend, `image_count` ve `valid_image_count` alanlarini upload edilen dosya sayisina gore kendi hesaplar.
- Frontend tarafinda 16'dan fazla gorsel secilirse backend yine koruma uygular ve warnings icinde bilgi doner.

## Fallback Davranislari

- Hic gorsel gelmezse image branch zero-vector fallback ile calisir.
- Title/description bos gelirse reusable predictor text sparse fallback ile devam eder.
- Eksik tabular alanlar fitted imputer tarafina birakilir.
- 16'dan fazla gorsel gelirse ilk 16 dosya kullanilir, kalanlar `warnings` icinde raporlanir.
- Bozuk veya okunamayan gorseller reusable predictor katmaninda warning ureterek skip edilir.

## Guvenlik Notlari

- Kullanici dosya adi dogrudan kullanılmaz; her gorsel benzersiz UUID tabanli dosya adi ile kaydedilir.
- Uzanti allowlist: `.jpg`, `.jpeg`, `.png`
- Content-type allowlist: `image/jpeg`, `image/png`
- Maksimum boyut: `10MB / image`
- Gorseller `backend/uploads/temp/<request-id>/` altina yazilir.
- Request tamamlandiktan sonra temp dosyalari temizlenir.
- Beklenmeyen tahmin hatalarinda detay loglanir, ancak istemciye stack trace donulmez.

## Lazy-Load ve Reuse

- Final model bundle her requestte yeniden yuklenmez; cache ile reuse edilir.
- OpenCLIP encoder da cache ile reuse edilir.
- Bu sayede ilk request disindaki cagrilarda model yukleme maliyeti dusurulur.

## Dogrulama

Asagidaki kontroller calistirildi:

1. `GET /health` -> `200 OK`
2. `GET /` -> `200 OK`
3. `POST /predict` -> gercek form-data ve ornek gorseller ile `200 OK`
4. CLI korunumu:

```bash
python src/predict_single_listing.py --input examples/sample_listing_input.json
```

CLI ornek sonucu:

```text
Model adi: XGBRegressor
Tahmini kira: 31.246 TRY
Kullanilan gorsel sayisi: 16
Uyarilar: yok
```
