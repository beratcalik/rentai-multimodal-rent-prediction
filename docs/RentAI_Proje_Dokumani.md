# RentAI: Multimodal Yapay Zeka ile Ankara Kira Tahmin Sistemi

**Bölüm:** Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği Bölümü  
**Ders:** YZM402 Araştırma Teknikleri II  
**Danışman:** Doç. Dr. Ramazan Yaşar  
**Öğrenci:** Berat Çalık - 22290395  
**GitHub:** <https://github.com/beratcalik/rentai-multimodal-rent-prediction>  
**Tarih:** 20 Mayıs 2026

---

## 1. Kapak

Bu doküman, RentAI projesinin amaç, veri seti, modelleme yaklaşımı, deneysel sonuçları ve ürünleştirme aşamalarını özetlemek amacıyla hazırlanmıştır.

## 2. Özet

RentAI, Ankara kiralık konut ilanları için kira tahmini üreten multimodal bir yapay zeka sistemidir. Sistem; konum ve konut özellikleri gibi yapılandırılmış verileri, ilan başlığı ve açıklama metnini, ayrıca fotoğrafları birlikte değerlendirir. Projede tabular baseline modellerinden başlanmış, daha sonra metin ve görsel branch’leri eklenerek performans adım adım iyileştirilmiştir.

Nihai model, tabular + text + image fusion yaklaşımıyla eğitilmiş ve test setinde **MAE 4.172,32 TL**, **RMSE 6.179,27 TL**, **R² 0,8477** ve **MAPE %12,50** sonucuna ulaşmıştır. Bu sonuç, matched tabular baseline modele göre anlamlı bir iyileşme sağlamıştır. Proje ayrıca açıklanabilirlik, güven skoru ve benzer piyasa örnekleri gibi karar destek katmanlarıyla gerçek zamanlı bir ürün demosu haline getirilmiştir.

## 3. Problem Tanımı ve Motivasyon

Ankara kiralık konut piyasasında fiyatlar; ilçe, mahalle, metrekare, bina yaşı, oda planı, ilanın anlatım dili ve görsel kalitesi gibi birçok değişkenden etkilenmektedir. Sadece tabular veriyle çalışan klasik modeller bu çok boyutlu sinyali tam olarak yakalayamamaktadır.

Özellikle ilan fotoğrafları, konutun bakım seviyesi, ferahlık algısı ve yaşam kalitesi hakkında dolaylı bilgi taşırken; açıklama metinleri de “lüks”, “ankastre”, “metroya yakın” gibi fiyat üzerinde etkili olabilecek ifadeler içermektedir. Bu nedenle projede, fiyat tahminini yalnızca sayısal alanlara değil, metin ve görsel veriye de dayandıran multimodal bir yaklaşım benimsenmiştir.

## 4. Veri Seti

Çalışmada kullanılan veri seti, Ankara kiralık emlak ilanlarından derlenmiştir. Eğitim ve deneylerde kullanılan train-ready multimodal veri seti aşağıdaki temel özelliklere sahiptir:

| Özellik | Değer |
| --- | --- |
| Toplam train-ready multimodal ilan | 6.389 |
| Toplam görsel | 99.148+ |
| İlçe sayısı | 16 |
| Mahalle sayısı | 361 |
| Ortalama görsel sayısı | 15,5 |

Veri türleri şunlardan oluşmaktadır:

- Konum bilgileri
- Konut özellikleri
- İlan açıklaması ve başlık
- Fotoğraflar

Bu yapı, tabular, text ve image branch’lerinin aynı ilan üzerinde birlikte çalışmasına olanak sağlamıştır.

## 5. Veri Ön İşleme

Veri ön işleme aşamasında aşağıdaki işlemler uygulanmıştır:

- Eksik veri analizi ve alan bazlı kontrol
- Uç değerlerin gözlemlenmesi ve raporlanması
- Bozuk veya okunamayan görsellerin ayıklanması
- `listing_id` üzerinden ilan ve görsellerin eşleştirilmesi
- Tüm veri akışının parquet tabanlı düzenli bir veri deposu yapısında tutulması

Ek olarak, `rooms` ve `floor` gibi alanlar modelleme öncesinde normalize edilmiştir. Bu sayede farklı yazım biçimleri daha tutarlı özellik temsillerine dönüştürülmüştür.

## 6. Modelleme Yaklaşımı

### 6.1 Tabular Baseline

İlk aşamada yalnızca tabular veri kullanan baseline modeller denenmiştir. HistGradientBoosting, XGBoost ve LightGBM gibi yöntemler karşılaştırılmış; amaç, multimodal genişletmeler için güçlü bir referans nokta oluşturmaktır.

### 6.2 Text Branch

Metin branch’inde ilan başlığı (`title`) ve açıklama (`description`) birleştirilmiştir. Bu metin üzerinde temizlik işlemleri uygulanmış, ardından:

- TF-IDF
- TruncatedSVD

yaklaşımıyla düşük boyutlu bir metin temsili elde edilmiştir.

### 6.3 Image Branch

İlk görsel deneylerde EfficientNet tabanlı embedding’ler kullanılmıştır. Daha sonra OpenCLIP tabanlı görsel temsilin daha güçlü sonuç verdiği görülmüştür. Nihai görsel akışta:

- image cap = 16
- CLIP mean+max embedding
- PCA ile boyut indirgeme

kullanılmıştır.

### 6.4 Fusion Model

Nihai modelde tabular, text ve image özellikleri aynı feature uzayında birleştirilmiş ve XGBoost Regressor ile kira fiyatı regresyonu yapılmıştır. Bu yapı, her modalitenin fiyat tahminine katkısını birlikte değerlendirebilen bütüncül bir model sunmuştur.

## 7. Deneysel Sonuçlar

| Model | MAE | RMSE | R2 | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Matched Tabular Baseline | 4700.08 | 7057.76 | 0.8013 | 13.79 |
| Best CLIP Fusion | 4346.62 | 6441.28 | 0.8345 | 12.95 |
| Final Multimodal Text+CLIP | 4172.32 | 6179.27 | 0.8477 | 12.50 |

Sonuçlar, final multimodal modelin baseline’a göre en iyi performansı verdiğini göstermektedir. Özellikle görsel branch’in ve metin branch’inin birlikte kullanılması, hata değerlerini düşürmüş ve açıklayıcılığı artırmıştır.

## 8. Ablation Analizi

| Durum | MAE |
| --- | ---: |
| Full Multimodal | 4172.32 |
| Text Branch Kapalı | 4482.72 |
| Image Branch Kapalı | 5494.53 |
| Text + Image Kapalı | 5965.51 |

Burada dikkat edilmesi gereken nokta, standalone tabular baseline ile final model içindeki ablation analizinin aynı şey olmadığıdır. Standalone baseline, ayrı eğitilmiş bağımsız bir tabular modeldir. Ablation analizi ise final multimodal pipeline içinde belirli branch’lerin kapatılmasıyla yapılmış karşılaştırmadır. Bu nedenle iki sonuç farklı amaçlarla yorumlanmalıdır.

## 9. Explainability ve Güven Skoru

Sistemde SHAP tabanlı yerel açıklanabilirlik katmanı kurulmuştur. Bu sayede her tahmin için:

- `top_positive_factors`
- `top_negative_factors`

üretilmektedir.

Ayrıca kullanıcıya doğrudan istatistiksel kesinlik iddiası taşımayan, karar destek amaçlı bir güven seviyesi sunulmaktadır. Bu katman:

- `confidence_score`
- `confidence_label`
- `confidence_reasons`

alanlarından oluşur. Bu yapı, kullanıcının yalnızca fiyatı değil, tahminin veri kalitesi ve benzerlik bağlamında ne kadar desteklendiğini de anlamasına yardımcı olmaktadır.

## 10. Benzer Piyasa Örnekleri

Tahmin sonucunun altında, girilen ilana benzeyen geçmiş ilanlardan 3-5 örnek gösterilmektedir. Bu modül KNN tabanlı retrieval yaklaşımıyla kurulmuştur. Benzerlik hesaplanırken konum, oda tipi, metrekare ve temel konut özellikleri kullanılmaktadır.

Önemli olarak, `price_try` benzerlik girişinde kullanılmamıştır. Çünkü kullanıcı yeni ilan için gerçek fiyatı bilmemektedir; bu alanın similarity inputunda kullanılması metodolojik olarak uygun değildir. `price_try` sadece referans amaçlı gösterilmektedir. Böylece kullanıcı, sistemin fiyatı ne tür piyasa örneklerine benzeterek verdiğini daha anlaşılır biçimde görebilmektedir.

## 11. Sistem Mimarisi

### Frontend

- Next.js
- TypeScript
- Tailwind CSS
- shadcn/ui

### Backend

- FastAPI

### Model Katmanı

- `final_multimodal_text_clip_model.joblib`
- OpenCLIP
- XGBoost

### Pipeline

İlan bilgileri + açıklama + fotoğraflar → preprocessing → model → tahmin + açıklama + confidence + benzer örnekler

Bu mimari, gerçek zamanlı bir ürün demosu ile araştırma çıktısını aynı yapıda birleştirmektedir.

## 12. Kullanıcı Arayüzü

Sistem için gerçek emlak portalı tarzında bir arayüz geliştirilmiştir. Kullanıcı arayüzünün temel özellikleri şunlardır:

- Dataset tabanlı seçim alanları
- Fotoğraf yükleme
- Beklenen kira aralığı gösterimi
- Tahmini etkileyen faktörler
- Güven seviyesi
- Benzer piyasa örnekleri

Bu yapı sayesinde sistem yalnızca model çıktısı veren bir demo olmaktan çıkarılmış, kullanıcıyla etkileşime girebilen bir değerleme arayüzüne dönüştürülmüştür.

## 13. Sonuç

Bu çalışma sonucunda, Ankara kiralık konut ilanları üzerinde çalışan multimodal bir kira tahmin sistemi başarıyla geliştirilmiştir. Deneyler, görsel branch’in anlamlı katkı sağladığını, text branch’in ise bu yapıya ek iyileştirme eklediğini göstermiştir. Nihai sistem; tahmin, açıklanabilirlik, güven seviyesi ve benzer piyasa örnekleri katmanlarıyla birlikte gerçek zamanlı ürün demosu seviyesine taşınmıştır.

## 14. Gelecek Çalışmalar

Gelecek aşamalarda aşağıdaki geliştirmeler planlanabilir:

- Daha fazla şehir desteği
- Daha büyük veri seti
- Zamansal fiyat değişimi modelleme
- Harita / GIS entegrasyonu
- Daha gelişmiş image ranking
- Mobil arayüz iyileştirmeleri

## 15. Kaynakça / Bağlantılar

- GitHub: <https://github.com/beratcalik/rentai-multimodal-rent-prediction>
- XGBoost: <https://xgboost.readthedocs.io/>
- OpenCLIP: <https://github.com/mlfoundations/open_clip>
- SHAP: <https://shap.readthedocs.io/>
- FastAPI: <https://fastapi.tiangolo.com/>
- Next.js: <https://nextjs.org/>

## 16. Kullanım Komutları

**Backend**

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Frontend**

```bash
cd frontend
npm run dev
```
