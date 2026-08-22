# Predict Pixel Redesign ve Similarity Fix Raporu

## Kapsam

Bu turda ağırlıklı olarak frontend düzeni güncellenmiştir. Gerekli olduğu için benzer piyasa örnekleri retrieval katmanına küçük ve kontrollü bir backend iyileştirmesi eklenmiştir. Model dosyaları, eğitim pipeline'ı ve dataset içerikleri değiştirilmemiştir.

## 1. Predict Layout'unu Referans Görsele Yaklaştırma

`/predict` sayfası referans görseldeki geniş, ferah ve emlak ürünü hissi veren yapıya yaklaştırıldı.

Yapılan ana değişiklikler:

- sayfa ana container genişliği `max-w-[1520px]` seviyesine çıkarıldı
- kenar boşlukları daraltıldı, geniş ekranlar daha verimli kullanıldı
- hero, analiz barı ve form alanı aynı hizaya getirildi
- eski dar merkez hissi azaltıldı

## 2. Hero Bandı

Yeni hero bandı daha kısa ve daha düzenli hale getirildi.

İçerik:

- solda konut illüstrasyonu
- sağda şehir silüeti
- ortada `Kira tahmini` başlığı
- altında açıklama metni

Amaç, sayfanın üst bölümünü daha canlı göstermek ama gereksiz büyük bir afişe dönüştürmemekti.

## 3. Analizde Kullanılacaklar Barı

Hero altındaki yatay durum kartı referans görseldeki yapıya daha yakın hale getirildi.

Özellikler:

- 4 yatay öğe
- büyük yuvarlak ikon alanları
- küçük durum badge'leri
- masaüstünde öğeler arası ince kesikli bağlantı çizgileri
- mobilde 2x2 düzen

Öğe sırası:

1. Konum bilgileri
2. Konut özellikleri
3. Konut fotoğraflarını yükle
4. İlan açıklaması

## 4. Form + Sağ Fotoğraf Paneli Düzeni

Eski küçük tahmin sonucu paneli tamamen kaldırıldı. Onun yerine:

- solda geniş ana form paneli
- sağda ayrı fotoğraf paneli

kuruldu.

Grid mantığı:

- sol: `minmax(0,1fr)`
- sağ: `420-440px`

Bu sayede masaüstünde sağ panel alta düşmeden görünür kaldı.

## 5. Formun Kompaktlaştırılması

Formun aşağı doğru gereksiz uzamasını azaltmak için şu düzenlemeler yapıldı:

- konut özellikleri alanı 4 kolonlu kompakt grid olarak korundu
- gap değerleri azaltıldı
- input yükseklikleri küçük/orta seviyede bırakıldı
- açıklama alanı kısaltıldı
- açıklama bölümü daha dengeli 2 kolonlu yapıya alındı

Ek olarak:

- `home_type` kullanıcıdan gizlendi
- backend'e varsayılan olarak `Daire` gönderilmeye devam edildi

## 6. Sağ Fotoğraf Paneli

Fotoğraf alanı referans görseldeki gibi sağ tarafta daha net bir panel olarak yeniden düzenlendi.

Panel içeriği:

- `Konut fotoğrafları` başlığı
- 0/16 sayacı
- drag & drop alanı
- dosya seç butonu
- eklenen fotoğraflar thumbnail alanı
- güven notu
- masaüstünde alt bölümde `Tahmini Al` butonu

## 7. Sonuç Modalı

Tahmin sonucu hâlâ modal olarak açılıyor; ancak içerik hiyerarşisi temizlendi.

Modal ilk açıldığında ön planda görülen alanlar:

1. Beklenen kira aralığı
2. Merkez tahmin
3. Tahmin güven seviyesi

Detaylar accordion içinde tutuldu:

- `Tahmini etkileyen faktörler`
- `Benzer piyasa örnekleri`

Bu iki bölüm varsayılan olarak kapalı geliyor.

## 8. Benzer Piyasa Örnekleri UI Düzeltmesi

Kullanıcı güvenini düşüren ana sorun, benzer örneklerde fiyatın tek odak gibi görünmesiydi. Bu yüzden frontend gösterimi yeniden kurgulandı.

Yeni yaklaşım:

- fiyat artık tek başına ana odak değil
- önce lokasyon ve yapısal benzerlik gösteriliyor
- her örneğin altında `neden benzer?` mantığını anlatan küçük chip'ler yer alıyor
- fiyat etiketi daha geri planda `Kayıtlı kira` olarak gösteriliyor

Ek açıklama:

`Bu örnekler fiyatı birebir doğrulamak için değil, girilen ilana benzeyen geçmiş kayıtları göstermek için listelenir.`

## 9. similarity_reasons Eklendi

Backend retrieval katmanına `similarity_reasons` alanı eklendi.

Örnek nedenler:

- Aynı ilçe
- Aynı mahalle
- Oda tipi aynı
- Oda tipi yakın
- m² değeri yakın
- m² aralığı benzer
- Kat seviyesi benzer
- Bina yaşı yakın

Bu alan frontend modalında chip olarak gösteriliyor.

## 10. Similar Listing Retrieval İyileştirmesi

Benzer örnek seçiminde küçük bir kalite iyileştirmesi yapıldı.

Yeni kurallar:

- aynı district daha güçlü önceliklendirildi
- aynı neighborhood güçlü bonus almaya devam etti
- oda tipi çok farklıysa aday eleniyor
- m² farkı çok yüksekse aday eleniyor
- predicted rent ile aşırı uzak fiyatlı örnekler yumuşak biçimde zayıflatılıyor / filtreleniyor
- çok sert filtre sebebiyle sonuç kalmaması durumunda gevşek fallback korunuyor

Önemli not:

`price_try` benzerlik inputunda doğrudan kullanılmamaktadır; sadece sonuçların sunumunda ve hafif son-sıralama kontrolünde yardımcı olarak değerlendirilmiştir.

## 11. Home Type Davranışı

Frontend tarafında kullanıcıya `Konut tipi` alanı gösterilmemektedir; çünkü mevcut veri kümesinde pratik olarak tek değer `Daire` olarak kullanılmaktadır.

Korunan davranış:

- form submit edilirken `home_type = "Daire"` gönderilmeye devam eder
- backend contract bozulmaz

## 12. RentAI Adı

Frontend arayüzünde `Rent Agent` / `RentAgent` kalıntıları tarandı.

Son durum:

- kullanıcıya görünen katmanda marka adı `RentAI`
- ek alt açıklama `Kira Asistanı` olarak kalabilir

## 13. Test Sonuçları

Çalıştırılan komutlar:

```bash
cd frontend
npm run build

python -B src/similar_listing_retrieval.py --input examples/sample_listing_input.json --json
python -B src/generate_prediction_explanations.py --input examples/sample_listing_input.json --json
```

Doğrulama sonucu:

- `npm run build` geçti
- `/predict` sayfası build içinde başarıyla üretildi
- benzer piyasa örnekleri çıktısında `similarity_reasons` alanı geldi
- explainability çıktısında `similar_listings` alanı `similarity_reasons` ile birlikte döndü
- frontend içinde `Rent Agent` / `RentAgent` taraması temiz döndü

## 14. Özet

Bu tur sonunda `/predict` sayfası:

- referans görsele daha yakın
- daha geniş
- daha az boşluklu
- sağda net bir fotoğraf paneli olan
- modal içinde daha kontrollü sonuç gösteren
- benzer piyasa örneklerini daha açıklayıcı sunan

bir ürün yüzeyine dönüştürüldü.
