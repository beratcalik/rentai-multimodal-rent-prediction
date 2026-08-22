# Benzer Piyasa Örnekleri Sistemi

## Amaç

Tahmin sonucunun altında, girilen ilana yapısal olarak benzeyen geçmiş ilanlardan 3-5 örnek göstermek. Bu katman bir ilan öneri sistemi değildir; karar destek amaçlı sade bir piyasa karşılaştırması sunar.

## Kullanılan Yaklaşım

Yeni modül:

- [src/similar_listing_retrieval.py](E:/rent-agent/src/similar_listing_retrieval.py)

Kurulan akış:

1. `train_ready_multimodal.parquet` içinden retrieval için gerekli kolonlar yüklenir.
2. `rooms`, `floor`, `is_furnished` gibi alanlarda eğitimle uyumlu normalizasyon uygulanır.
3. Tabular ağırlıklı bir özellik matrisi oluşturulur.
4. `NearestNeighbors` ile geniş bir aday havuzu bulunur.
5. Adaylar, kullanıcıya daha mantıklı görünecek ikinci bir kural tabanlı sıralama katmanından geçirilir.

## Similarity İçin Kullanılan Alanlar

Benzerlik hesabında kullanılan alanlar:

- `district`
- `neighborhood`
- `rooms`
- `m2_gross`
- `building_age`
- `floor`
- `total_floors`
- `bathrooms`
- `heating_type`
- `is_furnished`
- `home_shape`

Çıktıda gösterilen ama ana similarity inputunda doğrudan kullanılmayan alan:

- `price_try`

## Neden `price_try` Doğrudan Similarity Inputunda Kullanılmadı?

Kullanıcı yeni ilan için gerçek kira fiyatını bilmediği için `price_try` benzerlik hesabına doğrudan sokulmadı. Aksi durumda retrieval, kullanıcının henüz bilmediği hedef değişkene dayalı yapay bir yakınlık üretmiş olurdu.

Bu nedenle:

- `price_try` komşu seçiminin ana girdisi değildir
- sonuç kartında yalnızca referans amaçlı gösterilir
- yalnızca çok aşırı fiyat sapmalarını azaltmak için son sıralama/filtre aşamasında yumuşak biçimde kullanılabilir

## İkinci Aşama Yeniden Sıralama Kuralları

KNN sonrası adaylar şu mantıkla yeniden puanlanır:

- aynı ilçe güçlü bonus alır
- aynı mahalle daha güçlü bonus alır
- oda tipi aynıysa belirgin avantaj alır
- m² farkı küçükse avantaj alır
- bina yaşı, kat, banyo gibi alanlar yardımcı sinyal olarak kullanılır

Ek koruma kuralları:

- oda tipi çok farklıysa aday elenir
- m² farkı çok yüksekse aday elenir
- predicted rent ile aşırı uzak fiyatlı kayıtlar yumuşak biçimde geri atılır veya filtrelenir
- çok sert filtre sonucu yeterli kayıt kalmazsa sistem fallback ile biraz gevşer

## similarity_reasons Alanı

Yeni sürümde her benzer örnek için `similarity_reasons` alanı üretilmektedir.

Örnek nedenler:

- Aynı ilçe
- Aynı mahalle
- Oda tipi aynı
- Oda tipi yakın
- m² değeri yakın
- m² aralığı benzer
- Kat seviyesi benzer
- Bina yaşı yakın
- Isıtma tipi aynı

Bu alan frontend modalında küçük chip'ler halinde gösterilir.

## Backend Response Örneği

```json
{
  "similar_listings": [
    {
      "district": "Keçiören",
      "neighborhood": "Aşağı Eğlence Mah.",
      "rooms": "4+1",
      "m2_gross": 155.0,
      "building_age": 21.0,
      "floor": "2. Kat",
      "price_try": 29000,
      "price_formatted": "29.000 TL",
      "similarity_score": 99,
      "similarity_reasons": [
        "Aynı ilçe",
        "Aynı mahalle",
        "Oda tipi aynı",
        "m² değeri yakın"
      ]
    }
  ]
}
```

## Frontend Gösterimi

Sonuç modalında yeni bölüm:

- Başlık: `Benzer piyasa örnekleri`
- Açıklama: bu kayıtların fiyatı birebir doğrulamak için değil, yapısal benzerlik göstermek için listelendiği belirtilir
- Her satırda:
  - `district / neighborhood`
  - `rooms • m2_gross m² • floor`
  - `similarity_reasons` chip'leri
  - `Kayıtlı kira`
  - `Benzerlik %`

Önemli tasarım kararı:

- fiyat ana odak yapılmamıştır
- önce neden benzer olduğu açıklanır
- fiyat yalnızca destekleyici bilgi olarak görünür

## Test Senaryoları

### 1. Keçiören / 4+1 / 150-155 m²

Sonuç:

- aynı ilçe ve aynı mahalleden örnekler önceliklendirildi
- ilk kayıt `4+1 / 155 m²`
- `similarity_reasons` mantıklı biçimde üretildi

### 2. Görselsiz submit

Sonuç:

- benzer piyasa örnekleri yine döndü
- sistem image branch olmadan da retrieval yapabildi

### 3. Explainability akışı içinde çalışma

`generate_prediction_explanations.py` ile test edildi.

Sonuç:

- tahmin
- confidence
- faktör açıklamaları
- `similar_listings`

aynı response içinde birlikte üretildi.

## Sınırlamalar

- Bu özellik gerçek ilan önerisi değildir; karar destek amaçlı piyasa karşılaştırmasıdır.
- Retrieval yalnızca eğitim veri setindeki kayıtlarla sınırlıdır.
- Similarity skoru resmi değerleme ölçütü değildir.
- Fotoğraf ve metin branch'i retrieval seçiminde ilk sürümde doğrudan kullanılmamaktadır.
- Çok seyrek mahallelerde ilçe ve genel yapısal özellikler daha baskın hale gelebilir.

## Doğrulama

Çalıştırılan komutlar:

```bash
python -B src/similar_listing_retrieval.py --input examples/sample_listing_input.json --json
python -B src/generate_prediction_explanations.py --input examples/sample_listing_input.json --json
```

Doğrulama sonucu:

- retrieval çıktısı üretildi
- `similarity_reasons` alanı dolu geldi
- explainability response içinde `similar_listings` alanı korunarak döndü
- sistem mevcut `/predict-with-explanations` akışını bozmadı
