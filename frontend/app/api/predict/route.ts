import { NextResponse } from "next/server";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

const FIELD_LABELS: Record<string, string> = {
  city: "şehir",
  district: "ilçe",
  neighborhood: "mahalle",
  rooms: "oda tipi",
  bathrooms: "banyo bilgisi",
  m2_gross: "brüt m² bilgisi",
  building_age: "bina yaşı",
  floor: "bulunduğu kat",
  total_floors: "toplam kat",
  heating_type: "ısıtma tipi",
  fuel_type: "yakıt tipi",
  is_furnished: "eşyalı bilgisi",
  dues_try: "aidat bilgisi",
  home_type: "konut tipi",
  home_shape: "konut şekli",
  title: "ilan başlığı",
  description: "açıklama",
  images: "fotoğraflar",
};

type ValidationDetail = {
  loc?: Array<string | number>;
  msg?: string;
};

function getBackendBaseUrl() {
  const value = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  return value && value.length > 0 ? value.replace(/\/+$/, "") : DEFAULT_API_BASE_URL;
}

function getFieldLabel(fieldName: string) {
  return FIELD_LABELS[fieldName] ?? "ilgili alan";
}

function normalizeValidationMessage(detail: ValidationDetail) {
  const fieldName = detail.loc?.[detail.loc.length - 1];
  const label = typeof fieldName === "string" ? getFieldLabel(fieldName) : "ilgili alan";
  const message = detail.msg?.toLocaleLowerCase("tr-TR") ?? "";

  if (message.includes("field required") || message.includes("missing")) {
    return `Lütfen ${label} girin.`;
  }

  if (message.includes("valid")) {
    return `Lütfen ${label} alanını kontrol edin.`;
  }

  return `Lütfen ${label} alanını kontrol edin.`;
}

async function parseBackendError(response: Response) {
  try {
    const payload = await response.json();

    if (typeof payload?.detail === "string") {
      return payload.detail;
    }

    if (Array.isArray(payload?.detail) && payload.detail.length > 0) {
      return normalizeValidationMessage(payload.detail[0] as ValidationDetail);
    }

    if (typeof payload?.message === "string") {
      return payload.message;
    }
  } catch {
    // Fall through to text parsing.
  }

  try {
    const text = await response.text();
    return text.trim() || undefined;
  } catch {
    return undefined;
  }
}

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const incomingFormData = await request.formData();
    const backendFormData = new FormData();

    for (const [key, value] of incomingFormData.entries()) {
      backendFormData.append(key, value);
    }

    const response = await fetch(`${getBackendBaseUrl()}/predict`, {
      method: "POST",
      body: backendFormData,
      cache: "no-store",
    });

    const payload = await response.text();

    if (!response.ok) {
      let detail: string | undefined;

      try {
        detail = await parseBackendError(new Response(payload, { status: response.status, headers: response.headers }));
      } catch {
        detail = undefined;
      }

      return NextResponse.json(
        {
          detail: detail || "Tahmin isteği işlenemedi. Lütfen alanları kontrol edip tekrar deneyin.",
        },
        {
          status: response.status,
        },
      );
    }

    return new NextResponse(payload, {
      status: response.status,
      headers: {
        "content-type": response.headers.get("content-type") ?? "application/json; charset=utf-8",
      },
    });
  } catch {
    return NextResponse.json(
      {
        detail: "Tahmin servisine ulaşılamadı. Lütfen API'nin çalıştığından emin olun.",
      },
      {
        status: 503,
      },
    );
  }
}
