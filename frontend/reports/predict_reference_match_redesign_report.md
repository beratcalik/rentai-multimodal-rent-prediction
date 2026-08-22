# Predict Reference Match Redesign Raporu

## Kapsam

Bu turda `/predict` sayfası küçük yamalarla değil, yerleşim mantığı yeniden kurularak güncellenmiştir. Yalnızca frontend dosyalarına dokunulmuştur. Backend, model ve eğitim katmanına müdahale edilmemiştir.

## 1. Container Genişliği

Eski dar, ortalanmış yapı kaldırıldı.

Yeni yaklaşım:

- `width: calc(100% - 32px)` mobil
- `width: calc(100% - 48px)` küçük/orta ekran
- `width: calc(100% - 96px)` büyük ekran
- `max-width: 1500px`

Sonuç:

- kenar boşlukları belirgin biçimde azaldı
- hero, analiz barı ve ana form aynı genişlikte hizalandı
- referans görseldeki yatay yoğunluk hedeflendi

## 2. Hero Rewrite

Predict hero bölümü baştan yazıldı.

Yeni yapı:

- solda ev illüstrasyonu
- ortada başlık ve alt metin
- sağda şehir silüeti
- daha kompakt, daha yatay, daha az kart hissi veren bir üst alan

Başlık:

- `Kira tahmini`

Alt metin:

- `İlan bilgilerini ve fotoğrafları ekleyin, beklenen kira aralığını öğrenin.`

## 3. Analiz Bar Rewrite

`Analizde kullanılacaklar` alanı tek bir geniş yatay kart olarak yeniden tasarlandı.

Yeni yapı:

- 4 item aynı yüzey içinde
- büyük dairesel ikonlar
- başlık + küçük açıklama
- sağda badge
- desktopta item aralarında connector çizgileri
- mobilde 2x2 düzen

Bu bölüm artık küçük bağımsız kartlar değil, tek bir akış yüzeyi gibi davranıyor.

## 4. Form + Fotoğraf Grid Rewrite

Ana gövde referans görsele uygun şekilde yeniden kuruldu.

Grid:

- sol: `minmax(0, 1fr)`
- sağ: `440px`

Sol panel:

- geniş beyaz kart
- `rounded 18px`
- border + hafif shadow
- konum, konut özellikleri ve ilan açıklaması bölümleri

Sağ panel:

- ayrı fotoğraf kartı
- başlık + sayaç
- upload alanı
- thumbnail/empty state
- güven notu
- sağ panel altında submit butonu

Önemli:

- desktopta sağ panel aynı satırda kalır
- dar merkezi dashboard hissi kaldırılmıştır

## 5. Formun Kompaktlaştırılması

Form aşağı doğru gereksiz uzamasın diye yoğunluk yeniden ayarlandı.

Uygulanan yapı:

- konum: 3 kolon
- konut özellikleri: 4 kolon
- ilan açıklaması: 2 kolon

Stil:

- label: küçük/orta
- input yüksekliği: yaklaşık 40px
- daha kısa helper metinler
- section araları sıkılaştırıldı
- textarea kısaltıldı

## 6. Home Type Davranışı

`home_type` alanı frontend’de gösterilmemektedir.

Korunan davranış:

- backend’e varsayılan olarak `home_type = "Daire"` gönderilmeye devam eder

Böylece kullanıcı gereksiz tek seçenek görmez, API kontratı korunur.

## 7. Similar Listing UI Açıklama Chipleri

`Benzer piyasa örnekleri` bölümü güven kırıcı bir fiyat listesi gibi görünmemesi için yeniden kurgulandı.

Yeni gösterim:

- lokasyon
- `rooms • m² • floor`
- neden benzer chip’leri
- fiyat küçük alt satırda
- similarity score küçük destekleyici bilgi

Kullanılan açıklama:

`Bu örnekler fiyatı birebir doğrulamak için değil, girilen ilana benzeyen geçmiş kayıtları göstermek için listelenir.`

## 8. Frontend Similarity Fallback

Backend `similarity_reasons` alanı döndürse de, frontend tarafında ek fallback mantığı eklendi.

Backend nedenleri yoksa şu kurallardan üretim yapılır:

- aynı ilçe
- aynı mahalle
- aynı oda tipi
- m² yakın
- kat bilgisi yakın

Bu üretim sahte veri değildir; mevcut input ve response alanlarından türetilir.

## 9. Modal Accordion Davranışı

Sonuç modalında:

- `Tahmini etkileyen faktörler`
- `Benzer piyasa örnekleri`

varsayılan olarak kapalı tutulur.

Bu sayede modal ilk açıldığında:

- beklenen kira aralığı
- merkez tahmin
- güven seviyesi

öncelikli biçimde görünür.

## 10. Header ve Marka

Header opak beyaz tutuldu.

Korunan / güncellenen öğeler:

- marka adı: `RentAI`
- ortada nav
- sağda sarı CTA
- aktif nav underline

`Rent Agent` / `RentAgent` araması frontend katmanında temiz döndü.

## 11. Build Sonucu

Çalıştırılan komut:

```bash
cd frontend
npm run build
```

Sonuç:

- build başarıyla geçti
- `/`
- `/predict`
- `/methodology`
- `/api/predict`
- `/api/predict-with-explanations`

rotaları derleme çıktısında başarıyla üretildi.

## 12. Özet

Bu turda `/predict` ekranı küçük düzeltmelerle değil, referans görsele daha yakın olacak şekilde yeniden kurgulandı.

Son durum:

- daha geniş container
- daha net hero
- tek büyük analiz barı
- geniş sol form + sabit sağ fotoğraf paneli
- daha kompakt form ritmi
- daha açıklayıcı benzer piyasa örnekleri sunumu

Amaç, sayfayı dar SaaS kart hissinden çıkarıp gerçek bir emlak ürün yüzeyine yaklaştırmaktı.
