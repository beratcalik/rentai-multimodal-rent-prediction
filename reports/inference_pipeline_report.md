# Inference Pipeline Report

## Amac

Bu pipeline, egitilmis final multimodal modeli kullanarak tek bir yeni ilan icin kira tahmini uretir.

- Model bundle: `E:\rent-agent\models\final_multimodal_text_clip_model.joblib`
- Inference script: `E:\rent-agent\src\predict_single_listing.py`
- Ornek input: `E:\rent-agent\examples\sample_listing_input.json`

## Nasil Calisir

Script su akisi uygular:

1. Egitilmis model bundle yuklenir.
2. Tek ilan JSON'u okunur.
3. Tabular alanlar egitimdeki ayni `clean_tabular_features + fitted tabular_preprocessor` akisindan gecirilir.
4. `title + description` egitimdeki ayni text cleaning mantigi ile temizlenir.
5. Fitted `TF-IDF + TruncatedSVD` objeleri ile text feature uretilir.
6. `image_paths` icindeki ilk 16 okunabilir gorsel OpenCLIP `ViT-B-16 / laion2b_s34b_b88k` ile encode edilir.
7. Gorsellerden `mean + max concat` CLIP embedding uretilir.
8. Fitted image scaler/PCA pipeline'i uygulanir.
9. `[tabular | text | image]` feature block'u ile final `XGBRegressor` tahmini uretilir.

## Ornek Input JSON

```json
{
  "city": "Ankara",
  "district": "Keçiören",
  "neighborhood": "Aşağı Eğlence Mah.",
  "rooms": "4+1",
  "bathrooms": 2.0,
  "m2_gross": 155.0,
  "building_age": 21.0,
  "floor": "2. Kat",
  "total_floors": 2,
  "heating_type": "Kombi",
  "fuel_type": null,
  "is_furnished": false,
  "dues_try": null,
  "home_type": "Daire",
  "home_shape": "Tripleks",
  "image_count": 20,
  "valid_image_count": 16,
  "title": "ERKA'DAN AŞAĞIEĞLENCE'DE 4+1 BAĞIMSIZ GENİŞ 2 BANYOLU DUBLEKS",
  "description": "ERKA'DAN AŞAĞIEĞLENCE SARAR SOKAK'TA CADDEYE 1 BİNA MESAFEDE 4+1 BAĞIMSIZ BAKIMLI 2.KAT 2 BANYOLU GENİŞ M2'Lİ BALKONLU KAPALI DUBLEKS DAİRE SAHİBİNİN MEMUR VEYA MEMUR KEFİL ŞARTI VAR DETAYLI BİLGİ VE DİĞER ALTERNATİFLER İÇİN ARAYINIZ. YAŞAM KONFOR ve MUTLULUK İÇİN DOĞRU YERDESİNİZ EMLAK SEKTÖRÜNDE 13 YILLIK TECRÜBEMİZLE ON BİNLERCE MÜŞTERİMİZİ GAYRİMENKUL SAHİBİ YAPMANIN GURURUNU YAŞIYORUZ. DAİRELERİNİZE ÜCRETSİZ EKSPERTİZLİK HİZMETİ VERİLİR. GÜNÜNDE DEĞERİNDEN NAKİTE ÇEVRİLİR. TAPU ve KREDİ İŞLEMLERİNİZ PROFESYONEL EKİBİMİZ TARAFINDAN TAKİP EDİLMEKTEDİR. DİĞER ALTERNATİFLERİMİZİ GÖRMEK İÇİN İLANLARIMIZA BAKINIZ Daha detaylı bilgi için Sizleri ofisimize kahve içmeye bekleriz… ADRES:GENERAL DOKTOR TEVFİK SAĞLAM CADDESİ 101/D KEÇİÖREN HAYVAN HASTANESİ KARŞISI İRTİBAT TELEFONLARIMIZ Telefonu Göster Telefonu Göster",
  "image_paths": [
    "dataset\\images\\hepsiemlak_49652-2518\\004.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\005.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\006.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\007.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\008.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\009.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\010.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\011.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\012.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\013.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\014.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\015.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\016.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\017.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\018.jpg",
    "dataset\\images\\hepsiemlak_49652-2518\\019.jpg"
  ]
}
```

## Ornek Komut

```bash
python src/predict_single_listing.py --input examples/sample_listing_input.json
```

## Beklenen Cikti

Bu komut mevcut workspace icinde calistirilip dogrulandi. Ornek terminal ciktisi:

```text
Model adi: XGBRegressor
Tahmini kira: 31.246 TRY
Kullanilan gorsel sayisi: 16
Uyarilar: yok
```

## Fallback Davranislari

- `image_paths` bos ise model hata vermez; `clip_meanmax_embedding` boyutunda zero-vector uretilir ve egitimdeki fitted image pipeline'ina verilir.
- `image_paths` icindeki bozuk veya bulunamayan gorseller skip edilir; kalan okunabilir gorseller kullanilir.
- 16'dan fazla gorsel gelirse ilk 16 okunabilir gorsel kullanilir.
- `title` veya `description` bos ise text cleaning sonrasi bos string kalabilir; bu durumda TF-IDF sparse-zero fallback ile devam edilir.
- Tabular alanlar eksikse script bunlari `NaN` olarak birakir; fitted imputer egitimdeki mantikla doldurur.
- Beklenmeyen ekstra JSON alanlari ignore edilir ve terminalde uyari olarak gosterilir.

## Kullanici Notlari

- Farkli model bundle ile calistirmak icin `--model` parametresi verilebilir.
- Script egitim dosyalarini veya model artifact'larini degistirmez.
- Gorsel embedding inference asamasinda OpenCLIP backbone tekrar yuklenir; bu nedenle ilk tahmin birkac saniye surebilir.
