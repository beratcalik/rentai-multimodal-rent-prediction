## Product Landing Refactor Report

### Yapılan dosya değişiklikleri

- `app/page.tsx`
- `app/methodology/page.tsx`
- `components/layout/AppShell.tsx`
- `app/globals.css`
- `lib/constants.ts`

### Ana sayfadan kaldırılan veya geri plana alınan teknik içerikler

- Hero alanından `MAE`, `MAPE`, `R²` gibi model performans metrikleri çıkarıldı.
- Above-the-fold alandaki grafik ve performans kartları kaldırıldı.
- “Final multimodal performans”, “model snapshot” ve benzeri teknik söylemler ana sayfanın merkezinden çıkarıldı.
- Akademik/teknik anlatım yerine kullanıcı faydası, beklenen kira aralığı ve karar destek dili öne alındı.
- “Methodology / Metodoloji” ifadesi navbar’da geri plana çekilerek `Nasıl Çalışır` olarak sadeleştirildi.

### Yeni bilgi mimarisi

Ana sayfa artık şu yapı ile ilerliyor:

1. Hero
   - Başlık: `Evinizin beklenen kira aralığını dakikalar içinde görün`
   - Alt metin: ürün vaadi ve Ankara odaklı kullanım çerçevesi
   - Primary CTA: `Tahmini Hesapla`
   - Secondary CTA: `Nasıl çalışır?`
   - Sağ tarafta örnek ürün kartı:
     - `Beklenen kira aralığı: 38.000 TL – 44.000 TL`
     - Kullanılan veri tipleri
     - `Tahmin ekspertiz yerine geçmez.` notu

2. Trust strip
   - Fotoğrafların geçici işlendiği
   - Çok modlu veri kullanıldığı
   - Sonucun karar destek amaçlı olduğu

3. Nasıl çalışır?
   - İlan bilgilerini gir
   - Fotoğrafları ekle
   - Beklenen kira aralığını gör

4. Kullanım senaryoları
   - Ev sahibi
   - Gayrimenkul danışmanı
   - Yatırımcı

5. Neden güvenilir?
   - Gerçek Ankara ilan verisi
   - Çok modlu değerlendirme
   - Eksik bilgiyle ilerleyebilme
   - Karar destek odağı

6. FAQ / sınırlamalar
   - Ekspertiz değildir
   - Fotoğraflar kalıcı saklama amacıyla kullanılmaz
   - Şu an Ankara odaklıdır
   - Fotoğraf zorunlu değildir ama faydalıdır

### Navbar ve ürün dili revizyonu

- Ürün adı daha çok ürün yüzeyi gibi konumlandırıldı:
  - `Rent Agent`
  - alt başlık: `Kira Asistanı`
- Navbar linkleri sadeleştirildi:
  - `Ana Sayfa`
  - `Tahmin Yap`
  - `Nasıl Çalışır`
- Sağ üst CTA güncellendi:
  - `Tahmini Hesapla`
- Footer dili daha az teknik, daha çok ürün kullanım amacına odaklı hale getirildi.

### Methodology görünürlüğü ve yeni konumu

- Route aynı kaldı: `/methodology`
- Kullanıcıya sunulan adı sadeleşti: `Nasıl Çalışır`
- Sayfa içeriği tamamen kaldırılmadı; ama dashboard hissi veren ağır grafik/performans önceliği azaltıldı.
- Teknik içerik, ürün güvenini destekleyen açıklayıcı bir yapıda korunuyor:
  - Veri özeti
  - Modaliteler
  - Pipeline akışı
  - Performans güven notu
  - Ablation özeti
  - Sınırlamalar

### Test sonuçları

- `npm run build` başarıyla geçti.
- Route doğrulamaları yapıldı:
  - `GET /` -> `200`
  - `GET /predict` -> `200`
  - `GET /methodology` -> `200`
- `/predict` sayfasının form mantığına ve backend entegrasyonuna dokunulmadı.
- In-app browser ile localhost açma denemesi eklenti seviyesinde `ERR_BLOCKED_BY_CLIENT` verdi; bu nedenle route doğrulaması doğrudan HTTP istekleriyle tamamlandı.

### Kısa sonuç

Landing page artık teknik demo veya admin panel gibi değil; daha çok gerçek bir kira değerleme ürünü ile premium SaaS karışımı bir yüzey gibi davranıyor. Teknik güven notları tamamen yok edilmedi, ancak ana kullanıcı vaadinin önüne geçmeyecek şekilde geri plana alındı.
