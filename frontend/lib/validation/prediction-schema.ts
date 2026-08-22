import { z } from "zod";

export const MAX_PREDICTION_IMAGES = 16;
export const MAX_PREDICTION_IMAGE_SIZE_BYTES = 10 * 1024 * 1024;
export const ALLOWED_PREDICTION_IMAGE_TYPES = ["image/jpeg", "image/png"] as const;
export const ALLOWED_PREDICTION_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"] as const;

const requiredText = (label: string) => z.string().trim().min(1, `${label} gerekli.`);

const optionalSelectText = () => z.string().trim().optional().or(z.literal(""));

const optionalPositiveNumberText = (label: string) =>
  z
    .string()
    .trim()
    .optional()
    .or(z.literal(""))
    .refine((value) => !value || (Number.isFinite(Number(value)) && Number(value) >= 0), `${label} geçerli bir sayı olmalı.`);

const buildingAgeText = z
  .string()
  .trim()
  .optional()
  .or(z.literal(""))
  .refine((value) => !value || (Number.isInteger(Number(value)) && Number(value) >= 0 && Number(value) <= 100), "Bina yaşı 0 ile 100 arasında olmalı.");

const imageFileSchema = z
  .custom<File>((value) => typeof File !== "undefined" && value instanceof File, "Geçersiz görsel dosyası.")
  .refine(
    (file) => ALLOWED_PREDICTION_IMAGE_TYPES.includes(file.type as (typeof ALLOWED_PREDICTION_IMAGE_TYPES)[number]),
    "Sadece JPG, JPEG ve PNG dosyaları kabul edilir.",
  )
  .refine((file) => file.size <= MAX_PREDICTION_IMAGE_SIZE_BYTES, "Her görsel en fazla 10 MB olabilir.");

export const predictionFormSchema = z.object({
  city: requiredText("Şehir"),
  district: requiredText("İlçe"),
  neighborhood: requiredText("Mahalle"),
  rooms: requiredText("Oda tipi"),
  bathrooms: optionalSelectText(),
  m2_gross: z
    .string()
    .trim()
    .min(1, "Brüt m² gerekli.")
    .refine((value) => Number.isFinite(Number(value)) && Number(value) > 0, "Brüt m² sıfırdan büyük olmalı."),
  building_age: buildingAgeText,
  floor: optionalSelectText(),
  total_floors: optionalSelectText(),
  heating_type: optionalSelectText(),
  fuel_type: optionalSelectText(),
  is_furnished: optionalSelectText(),
  dues_try: optionalPositiveNumberText("Aidat"),
  home_type: optionalSelectText(),
  home_shape: optionalSelectText(),
  title: z.string().trim().min(5, "İlan başlığı gerekli."),
  description: z.string().trim().min(20, "İlan açıklaması gerekli."),
  images: z.array(imageFileSchema).max(MAX_PREDICTION_IMAGES, `En fazla ${MAX_PREDICTION_IMAGES} görsel ekleyebilirsiniz.`).default([]),
});

export type PredictionFormValues = z.infer<typeof predictionFormSchema>;

export const similarListingSchema = z.object({
  district: z.string(),
  neighborhood: z.string(),
  rooms: z.string(),
  m2_gross: z.number().nullable().optional(),
  building_age: z.number().nullable().optional(),
  floor: z.string().nullable().optional(),
  price_try: z.number().int(),
  price_formatted: z.string(),
  similarity_score: z.number().int().min(0).max(100),
  similarity_reasons: z.array(z.string()).optional().default([]),
});

export const predictionResponseSchema = z.object({
  predicted_rent_try: z.number().int(),
  predicted_rent_formatted: z.string(),
  used_image_count: z.number().int(),
  model_name: z.string(),
  warnings: z.array(z.string()),
  message: z.string(),
  confidence_score: z.number().int().min(0).max(100),
  confidence_label: z.enum(["Yüksek", "Orta", "Düşük"]).optional(),
  confidence_reasons: z.array(z.string()).optional().default([]),
  top_positive_factors: z.array(z.string()).default([]),
  top_negative_factors: z.array(z.string()).default([]),
  similar_listings: z.array(similarListingSchema).optional().default([]),
});

export type PredictionResponse = z.infer<typeof predictionResponseSchema>;
export type SimilarListing = z.infer<typeof similarListingSchema>;

export const predictionDefaultValues: PredictionFormValues = {
  city: "Ankara",
  district: "",
  neighborhood: "",
  rooms: "",
  bathrooms: "",
  m2_gross: "",
  building_age: "",
  floor: "",
  total_floors: "",
  heating_type: "",
  fuel_type: "",
  is_furnished: "",
  dues_try: "",
  home_type: "Daire",
  home_shape: "",
  title: "",
  description: "",
  images: [],
};
