import { predictionResponseSchema, type PredictionFormValues, type PredictionResponse } from "../validation/prediction-schema";

const PREDICTION_TIMEOUT_MS = 90_000;

export class PredictionApiError extends Error {
  status?: number;
  detail?: string;

  constructor(message: string, options?: { status?: number; detail?: string }) {
    super(message);
    this.name = "PredictionApiError";
    this.status = options?.status;
    this.detail = options?.detail;
  }
}

function getPredictionEndpoint() {
  return "/api/predict";
}

function buildPredictionFormData(formValues: PredictionFormValues, files: File[]) {
  const formData = new FormData();

  const fields: Array<keyof Omit<PredictionFormValues, "images">> = [
    "city",
    "district",
    "neighborhood",
    "rooms",
    "bathrooms",
    "m2_gross",
    "building_age",
    "floor",
    "total_floors",
    "heating_type",
    "fuel_type",
    "is_furnished",
    "dues_try",
    "home_type",
    "home_shape",
    "title",
    "description",
  ];

  for (const field of fields) {
    formData.append(field, formValues[field] ?? "");
  }

  for (const file of files) {
    formData.append("images", file, file.name);
  }

  return formData;
}

function mapBackendError(status: number, detail?: string) {
  if (status === 400 && detail) {
    return detail;
  }

  if (status === 422 && detail) {
    return detail;
  }

  if (status === 413) {
    return "Yüklenen dosyalar izin verilen boyut sınırını aştı.";
  }

  if (status >= 500) {
    return "Tahmin servisi şu anda isteği tamamlayamıyor. Lütfen birkaç saniye sonra tekrar deneyin.";
  }

  return detail || "Tahmin isteği işlenemedi. Lütfen alanları kontrol edip tekrar deneyin.";
}

async function parseErrorDetail(response: Response) {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") {
      return payload.detail;
    }

    if (typeof payload?.message === "string") {
      return payload.message;
    }
  } catch {
    // Fall through to text parse.
  }

  try {
    const text = await response.text();
    return text.trim() || undefined;
  } catch {
    return undefined;
  }
}

export async function createPrediction(formValues: PredictionFormValues, files: File[]): Promise<PredictionResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort("prediction-timeout"), PREDICTION_TIMEOUT_MS);

  try {
    const response = await fetch(getPredictionEndpoint(), {
      method: "POST",
      body: buildPredictionFormData(formValues, files),
      signal: controller.signal,
    });

    if (!response.ok) {
      const detail = await parseErrorDetail(response);
      throw new PredictionApiError(mapBackendError(response.status, detail), {
        status: response.status,
        detail,
      });
    }

    const payload = await response.json();
    return predictionResponseSchema.parse(payload);
  } catch (error) {
    if (error instanceof PredictionApiError) {
      throw error;
    }

    if (error instanceof DOMException && error.name === "AbortError") {
      throw new PredictionApiError("Tahmin isteği zaman aşımına uğradı. Lütfen kısa süre sonra tekrar deneyin.");
    }

    if (error instanceof TypeError) {
      throw new PredictionApiError("Tahmin servisine ulaşılamadı. Lütfen API'nin çalıştığından emin olun.");
    }

    throw new PredictionApiError("Tahmin oluşturulamadı. Lütfen tekrar deneyin.");
  } finally {
    clearTimeout(timeoutId);
  }
}
