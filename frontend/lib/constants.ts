import {
  BarChart3,
  Building2,
  Camera,
  FileText,
  Home,
  ImageIcon,
  MapPinned,
  MessageSquareText,
  ScanSearch,
  type LucideIcon,
} from "lucide-react";

export const NAV_LINKS = [
  { href: "/", label: "Ana Sayfa" },
  { href: "/predict", label: "Tahmin Yap" },
  { href: "/methodology", label: "Nasıl Çalışır" },
] as const;

export const HOME_STEPS: Array<{
  title: string;
  description: string;
  icon: LucideIcon;
}> = [
  {
    title: "Konumu seç",
    description: "İlçe ve mahalleyi belirleyin.",
    icon: MapPinned,
  },
  {
    title: "Özellikleri gir",
    description: "Konut bilgilerini ve ilan metnini ekleyin.",
    icon: Building2,
  },
  {
    title: "Tahmini gör",
    description: "Beklenen kira aralığını aynı ekranda alın.",
    icon: BarChart3,
  },
] as const;

export const HOME_TRUST_NOTES = [
  "Fotoğraflar geçici işlenir.",
  "Sonuç karar destek amaçlıdır.",
  "Şu an Ankara verisiyle çalışır.",
] as const;

export const PREDICTION_MODALITIES: Array<{
  title: string;
  description: string;
  icon: LucideIcon;
}> = [
  {
    title: "Sayısal özellikler",
    description: "Konum, oda tipi, metrekare ve konut detayları birlikte değerlendirilir.",
    icon: Home,
  },
  {
    title: "İlan açıklaması",
    description: "Başlık ve açıklama, mülkün bağlamını güçlendiren ek sinyal sağlar.",
    icon: MessageSquareText,
  },
  {
    title: "Fotoğraflar",
    description: "Salon, mutfak, banyo ve cephe görselleri görsel kalite sinyali üretir.",
    icon: Camera,
  },
] as const;

export const PREDICTION_STEPS: Array<{
  id: string;
  title: string;
  icon: LucideIcon;
}> = [
  { id: "location", title: "Konum", icon: MapPinned },
  { id: "features", title: "Özellikler", icon: Building2 },
  { id: "description", title: "Açıklama", icon: FileText },
  { id: "photos", title: "Fotoğraflar", icon: Camera },
  { id: "prediction", title: "Tahmin", icon: BarChart3 },
] as const;

export const METHODOLOGY_STEPS: Array<{
  title: string;
  description: string;
  icon: LucideIcon;
}> = [
  {
    title: "İlan bilgileri toplanır",
    description: "Konum ve konut özellikleri tek form akışında alınır.",
    icon: MapPinned,
  },
  {
    title: "Metin ve görseller okunur",
    description: "Açıklama metni ve fotoğraflar modele ek bağlam sağlar.",
    icon: ScanSearch,
  },
  {
    title: "Kira tahmini üretilir",
    description: "Tüm veriler birlikte değerlendirilerek beklenen kira aralığı oluşturulur.",
    icon: BarChart3,
  },
] as const;

export const METHODOLOGY_DATA_TYPES: Array<{
  title: string;
  description: string;
  icon: LucideIcon;
}> = [
  {
    title: "Konum ve konut özellikleri",
    description: "İlçe, mahalle, m², oda tipi, kat, ısıtma, aidat ve benzeri yapısal alanlar.",
    icon: Building2,
  },
  {
    title: "İlan metni",
    description: "Başlık ve açıklama, görünmeyen kalite ve yaşam tarzı detaylarını destekler.",
    icon: FileText,
  },
  {
    title: "Fotoğraflar",
    description: "Mekân düzeni, yenilik seviyesi ve sunum kalitesi için görsel sinyal sağlar.",
    icon: ImageIcon,
  },
] as const;

export const METHODOLOGY_LIMITATIONS = [
  "Sistem şu an Ankara veri seti ile sınırlıdır.",
  "Tahmin çıktısı ekspertiz yerine geçmez; fiyat kararı için yardımcı referans sunar.",
  "Fotoğraf olmadan da çalışır, ancak fotoğraf eklemek tahmini güçlendirebilir.",
  "Veri setinde az temsil edilen sıra dışı mülklerde belirsizlik artabilir.",
] as const;

export const METHODOLOGY_TECHNICAL_NOTES: Array<{
  title: string;
  summary: string;
  details: string[];
}> = [
  {
    title: "Veri seti özeti",
    summary: "6.389 train-ready multimodal ilan ve 99 binden fazla görsel kullanıldı.",
    details: [
      "Ankara kiralık konut ilanlarından oluşturulmuş train-ready multimodal örnekler kullanıldı.",
      "Tabular alanlar, açıklama metni ve görseller aynı ilan düzeyinde birleştirildi.",
    ],
  },
  {
    title: "Model hattı",
    summary: "Tabular preprocessing, TF-IDF + SVD text branch, CLIP image embeddings ve XGBoost fusion kullanıldı.",
    details: [
      "Sayısal ve kategorik alanlar uygun doldurma ve encoding stratejileriyle işlendi.",
      "Metin tarafında başlık ve açıklama indirgenmiş vektör temsiline çevrildi.",
      "Fotoğraflar CLIP tabanlı embeddingler ile temsil edilip fusion modeline verildi.",
    ],
  },
  {
    title: "Performans notları",
    summary: "Final multimodal model, tabular baseline ve image-only fusion deneylerine göre daha güçlü sonuç verdi.",
    details: [
      "Tabular baseline MAE: 4.700,08 TL",
      "Best CLIP fusion MAE: 4.346,62 TL",
      "Final multimodal MAE: 4.172,32 TL",
      "Final multimodal RMSE: 6.179,27 TL",
      "Final multimodal MAPE: %12,50",
      "Final multimodal R²: 0,8477",
    ],
  },
  {
    title: "Ablation notları",
    summary: "Metin ve görsel dalların katkısı, kapatıldıklarında ölçülen hata artışıyla izlendi.",
    details: [
      "Text kapatılınca MAE: 4.482,72 TL",
      "Image kapatılınca MAE: 5.494,53 TL",
      "Text + image kapatılınca MAE: 5.965,51 TL",
    ],
  },
  {
    title: "Yorumlama sınırı",
    summary: "Çıktı, fiyatlama konuşmasını başlatan karar destek yüzeyi olarak değerlendirilmelidir.",
    details: [
      "Beklenen kira aralığı, resmi ekspertiz veya hukuki bağlayıcılık taşıyan rapor değildir.",
      "Özellikle çok sıra dışı ilan tiplerinde insan değerlendirmesiyle birlikte okunmalıdır.",
    ],
  },
] as const;

export const MODEL_COMPARISON = [
  {
    name: "Tabular baseline",
    mae: 4700.08,
  },
  {
    name: "Best CLIP fusion",
    mae: 4346.62,
  },
  {
    name: "Final multimodal",
    mae: 4172.32,
  },
] as const;

export const PREDICT_LOADING_STEPS = [
  "İlan bilgileri hazırlanıyor",
  "Fotoğraflar analiz ediliyor",
  "Kira tahmini oluşturuluyor",
] as const;
