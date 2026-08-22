"use client";

import { AlertTriangle, CheckCircle2, Home, LoaderCircle, ShieldCheck, Sparkles } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";

import { PREDICT_LOADING_STEPS, PREDICTION_MODALITIES } from "@/lib/constants";
import { formatNumberTr } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { PredictionResponse, SimilarListing } from "@/lib/validation/prediction-schema";

export type AnalysisStatus = "ready" | "missing" | "optional";

export type PredictionAnalysisItem = {
  label: string;
  status: AnalysisStatus;
};

type PredictionResultCardProps = {
  state:
    | { status: "empty" }
    | { status: "loading" }
    | { status: "error"; message: string }
    | { status: "success"; data: PredictionResponse };
  analysisItems: PredictionAnalysisItem[];
};

type ExplanationItem = {
  tone: "positive" | "negative";
  message: string;
};

type ResultSectionProps = {
  title: string;
  description?: string;
  children: ReactNode;
};

function ResultSection({ title, description, children }: ResultSectionProps) {
  return (
    <section className="border-t border-border pt-4">
      <div className="mb-3 space-y-1">
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
        {description ? <p className="text-xs leading-5 text-muted-foreground">{description}</p> : null}
      </div>
      {children}
    </section>
  );
}

function formatRangeLabel(value: number) {
  return `${formatNumberTr(value)} TL`;
}

function getStatusLabel(status: AnalysisStatus) {
  if (status === "ready") {
    return "Hazır";
  }

  if (status === "optional") {
    return "Opsiyonel";
  }

  return "Eksik";
}

function getStatusClassName(status: AnalysisStatus) {
  if (status === "ready") {
    return "bg-success/10 text-success";
  }

  if (status === "optional") {
    return "bg-slate-100 text-muted-foreground";
  }

  return "bg-warning/10 text-warning";
}

function buildExplanationItems(data: PredictionResponse): ExplanationItem[] {
  const positiveItems = data.top_positive_factors.slice(0, 3).map((message) => ({
    tone: "positive" as const,
    message,
  }));
  const negativeItems = data.top_negative_factors.slice(0, 2).map((message) => ({
    tone: "negative" as const,
    message,
  }));

  return [...positiveItems, ...negativeItems].slice(0, 5);
}

function buildSimilarListingMeta(item: SimilarListing) {
  const parts = [
    item.rooms,
    typeof item.m2_gross === "number" ? `${formatNumberTr(item.m2_gross)} m²` : "",
    item.floor ?? "",
  ].filter(Boolean);

  return parts.join(" • ");
}

function getConfidenceAppearance(label?: string) {
  if (label === "Yüksek") {
    return {
      textClassName: "text-success",
      badgeClassName: "bg-success/10 text-success",
      barClassName: "bg-success",
    };
  }

  if (label === "Düşük") {
    return {
      textClassName: "text-warning",
      badgeClassName: "bg-warning/10 text-warning",
      barClassName: "bg-warning",
    };
  }

  return {
    textClassName: "text-[#B45309]",
    badgeClassName: "bg-warning/10 text-warning",
    barClassName: "bg-warning",
  };
}

export function PredictionResultCard({ state, analysisItems }: PredictionResultCardProps) {
  const [loadingStepIndex, setLoadingStepIndex] = useState(0);

  useEffect(() => {
    if (state.status !== "loading") {
      setLoadingStepIndex(0);
      return;
    }

    const intervalId = window.setInterval(() => {
      setLoadingStepIndex((current) => (current + 1) % PREDICT_LOADING_STEPS.length);
    }, 1400);

    return () => window.clearInterval(intervalId);
  }, [state.status]);

  const rangeValues = useMemo(() => {
    if (state.status !== "success") {
      return null;
    }

    const center = state.data.predicted_rent_try;
    return {
      lower: Math.round(center * 0.9),
      upper: Math.round(center * 1.1),
    };
  }, [state]);

  const explanationItems = useMemo(() => {
    if (state.status !== "success") {
      return [];
    }

    return buildExplanationItems(state.data);
  }, [state]);

  const confidenceAppearance = useMemo(() => {
    if (state.status !== "success") {
      return getConfidenceAppearance("Orta");
    }

    return getConfidenceAppearance(state.data.confidence_label);
  }, [state]);

  const loadingProgress = ((loadingStepIndex + 1) / PREDICT_LOADING_STEPS.length) * 100;

  return (
    <Card className="overflow-hidden lg:sticky lg:top-20">
      <CardHeader className="space-y-1.5 border-b border-border px-4 py-4 sm:px-5">
        <CardTitle className="text-xl sm:text-[22px]">Tahmin sonucu</CardTitle>
        <CardDescription>Bilgileri girin, beklenen kira aralığını görün.</CardDescription>
      </CardHeader>

      <CardContent className="space-y-4 px-4 pb-4 pt-4 sm:px-5 sm:pb-5">
        {state.status === "loading" ? (
          <div className="space-y-4">
            <div className="rounded-xl border border-border bg-slate-50/70 px-4 py-4">
              <div className="flex items-start gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[#EAF2FF] text-[#0057B8]">
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                </div>
                <div className="min-w-0 flex-1 space-y-2">
                  <div className="text-sm font-semibold text-foreground">{PREDICT_LOADING_STEPS[loadingStepIndex]}…</div>
                  <div className="h-2 overflow-hidden rounded-full bg-slate-200">
                    <div className="h-full rounded-full bg-[#0057B8] transition-all duration-500" style={{ width: `${loadingProgress}%` }} />
                  </div>
                  <p className="text-xs leading-5 text-muted-foreground">
                    İlan metni, fotoğraflar ve benzer piyasa sinyalleri birlikte değerlendiriliyor.
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              {PREDICT_LOADING_STEPS.map((step, index) => {
                const isComplete = index < loadingStepIndex;
                const isActive = index === loadingStepIndex;

                return (
                  <div key={step} className="flex items-center gap-3 rounded-lg border border-border px-3 py-3">
                    <div
                      className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                        isComplete
                          ? "bg-success text-white"
                          : isActive
                            ? "bg-[#0057B8] text-white"
                            : "bg-slate-100 text-muted-foreground"
                      }`}
                    >
                      {isComplete ? <CheckCircle2 className="h-3.5 w-3.5" /> : index + 1}
                    </div>
                    <div className="min-w-0 flex-1 text-sm text-foreground">{step}</div>
                  </div>
                );
              })}
            </div>

            <div className="space-y-2 rounded-lg border border-border bg-white px-4 py-4">
              <div className="h-4 w-28 animate-pulse rounded-full bg-slate-100" />
              <div className="h-8 w-3/4 animate-pulse rounded-lg bg-slate-100" />
              <div className="h-3.5 w-2/3 animate-pulse rounded-full bg-slate-100" />
            </div>
          </div>
        ) : null}

        {state.status === "success" && rangeValues ? (
          <div className="space-y-4">
            <div className="rounded-xl border border-[#0B3A75]/12 bg-[#F8FBFF] px-4 py-4">
              <div className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">Beklenen kira aralığı</div>
              <div className="mt-2 break-words text-[24px] font-semibold tracking-[-0.03em] text-primary sm:text-[30px]">
                {formatRangeLabel(rangeValues.lower)} – {formatRangeLabel(rangeValues.upper)}
              </div>
              <div className="mt-2 text-sm text-muted-foreground">
                Merkez tahmin: <span className="font-medium text-foreground">{state.data.predicted_rent_formatted}</span>
              </div>
            </div>

            <div className="rounded-lg border border-border bg-white px-4 py-3">
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-muted-foreground">Kullanılan görsel sayısı</span>
                <span className="font-semibold text-foreground">{state.data.used_image_count}</span>
              </div>
            </div>

            <ResultSection title="Tahmin güven seviyesi" description="Bu skor, girilen bilgilerin veri setindeki benzer örneklerle ne kadar uyumlu olduğunu gösterir.">
              <div className="space-y-3 rounded-lg border border-border bg-white px-4 py-4">
                <div className="flex items-center justify-between gap-3">
                  <div className={`rounded-md px-2.5 py-1 text-xs font-semibold ${confidenceAppearance.badgeClassName}`}>
                    {state.data.confidence_label ?? "Orta"}
                  </div>
                  <div className={`text-sm font-semibold ${confidenceAppearance.textClassName}`}>{state.data.confidence_score}/100</div>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full transition-all ${confidenceAppearance.barClassName}`}
                    style={{ width: `${Math.max(6, Math.min(100, state.data.confidence_score))}%` }}
                  />
                </div>
                {state.data.confidence_reasons.length > 0 ? (
                  <div className="space-y-2">
                    {state.data.confidence_reasons.slice(0, 3).map((reason) => (
                      <div key={reason} className="text-xs leading-5 text-muted-foreground">
                        • {reason}
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>
            </ResultSection>

            {explanationItems.length > 0 ? (
              <ResultSection title="Tahmini etkileyen faktörler">
                <div className="space-y-2.5">
                  {explanationItems.map((item) => (
                    <div
                      key={`${item.tone}-${item.message}`}
                      className={`rounded-lg border px-3 py-3 text-sm leading-6 ${
                        item.tone === "positive"
                          ? "border-success/20 bg-success/5 text-success"
                          : "border-error/20 bg-error/5 text-error"
                      }`}
                    >
                      {item.message}
                    </div>
                  ))}
                </div>
              </ResultSection>
            ) : null}

            {state.data.similar_listings.length > 0 ? (
              <ResultSection
                title="Benzer piyasa örnekleri"
                description="Bu örnekler, girilen ilana benzer özelliklere sahip geçmiş ilanlardan seçilmiştir."
              >
                <div className="space-y-2.5">
                  {state.data.similar_listings.slice(0, 5).map((item) => (
                    <div
                      key={`${item.district}-${item.neighborhood}-${item.rooms}-${item.price_try}`}
                      className="rounded-lg border border-border bg-slate-50/60 px-3 py-3"
                    >
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                        <div className="min-w-0 flex-1">
                          <div className="break-words text-sm font-medium text-foreground">
                            {item.district} / {item.neighborhood}
                          </div>
                          <div className="mt-1 break-words text-xs leading-5 text-muted-foreground">{buildSimilarListingMeta(item)}</div>
                        </div>
                        <div className="inline-flex w-fit shrink-0 rounded-md bg-slate-100 px-2 py-1 text-xs font-semibold text-foreground">
                          %{item.similarity_score}
                        </div>
                      </div>
                      <div className="mt-2 text-sm text-muted-foreground">
                        İlan kirası: <span className="font-semibold text-foreground">{item.price_formatted}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </ResultSection>
            ) : null}

            {state.data.warnings.length > 0 ? (
              <ResultSection title="Bilgilendirmeler">
                <div className="rounded-lg border border-warning/20 bg-warning/5 px-4 py-3">
                  <div className="space-y-2">
                    {state.data.warnings.map((warning) => (
                      <p key={warning} className="text-xs leading-5 text-warning">
                        {warning}
                      </p>
                    ))}
                  </div>
                </div>
              </ResultSection>
            ) : null}

            <ResultSection title="Analizde kullanılan veri kaynakları">
              <div className="space-y-3">
                {PREDICTION_MODALITIES.map((item) => (
                  <div key={item.title} className="flex gap-3 rounded-lg border border-border bg-white px-3 py-3">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-primary">
                      <item.icon className="h-4 w-4" />
                    </div>
                    <div className="min-w-0">
                      <div className="text-sm font-medium text-foreground">{item.title}</div>
                      <div className="text-xs leading-5 text-muted-foreground">{item.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </ResultSection>

            <ResultSection title="Bilgilendirme">
              <div className="space-y-1">
                <div className="text-sm font-medium text-foreground">Kira tahmini oluşturuldu.</div>
                <div className="text-xs leading-5 text-muted-foreground">{state.data.message}</div>
                <div className="text-xs leading-5 text-muted-foreground">
                  Bu tahmin karar destek amaçlıdır, ekspertiz yerine geçmez.
                </div>
              </div>
            </ResultSection>
          </div>
        ) : null}

        {state.status === "error" ? (
          <div className="space-y-3">
            <div className="rounded-xl border border-error/20 bg-error/5 px-4 py-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
                <div className="space-y-1.5">
                  <div className="text-sm font-semibold text-foreground">Tahmin servisine ulaşılamadı</div>
                  <p className="text-sm leading-6 text-muted-foreground">{state.message}</p>
                </div>
              </div>
            </div>
            <div className="text-xs leading-5 text-muted-foreground">
              Servis tekrar erişilebilir olduğunda aynı ekrandan yeniden tahmin alabilirsiniz.
            </div>
          </div>
        ) : null}

        {state.status === "empty" ? (
          <div className="space-y-4">
            <div className="rounded-xl border border-border bg-slate-50/70 px-4 py-4">
              <div className="flex items-start gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[#FFF8CC] text-primary">
                  <Home className="h-4 w-4" />
                </div>
                <div className="space-y-1.5">
                  <div className="text-sm font-semibold text-foreground">Bilgileri girin ve tahmini alın.</div>
                  <p className="text-sm leading-6 text-muted-foreground">Sonuç burada gösterilecek.</p>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-border bg-white">
              <div className="border-b border-border px-4 py-3 text-sm font-semibold text-foreground">
                Analizde kullanılacaklar
              </div>
              <div className="space-y-3 px-4 py-4">
                {analysisItems.map((item) => (
                  <div key={item.label} className="flex items-center justify-between gap-3 text-sm">
                    <span className="text-muted-foreground">{item.label}</span>
                    <span className={`rounded-md px-2 py-1 text-xs font-medium ${getStatusClassName(item.status)}`}>
                      {getStatusLabel(item.status)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
