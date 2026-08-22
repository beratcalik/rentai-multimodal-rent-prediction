"use client";

import { CheckCircle2, Sparkles, X } from "lucide-react";
import { type ReactNode, useEffect, useMemo } from "react";
import { createPortal } from "react-dom";

import { ResultAccordion } from "@/components/predict/ResultAccordion";
import { Button } from "@/components/ui/button";
import type { PredictionFormValues, PredictionResponse, SimilarListing } from "@/lib/validation/prediction-schema";
import { formatNumberTr } from "@/lib/utils";

type PredictionResultDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  data: PredictionResponse | null;
  onContinueEditing?: () => void;
  queryContext?: Pick<PredictionFormValues, "district" | "neighborhood" | "rooms" | "m2_gross" | "floor"> | null;
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
    <section className="space-y-3 border-t border-border pt-5">
      <div className="space-y-1">
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
        {description ? <p className="text-xs leading-5 text-muted-foreground">{description}</p> : null}
      </div>
      {children}
    </section>
  );
}

function formatCurrencyLabel(value: number) {
  return `${formatNumberTr(value)} TL`;
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
    textClassName: "text-warning",
    badgeClassName: "bg-warning/10 text-warning",
    barClassName: "bg-warning",
  };
}

function buildFallbackSimilarityReasons(
  item: SimilarListing,
  queryContext?: Pick<PredictionFormValues, "district" | "neighborhood" | "rooms" | "m2_gross" | "floor"> | null,
) {
  if (item.similarity_reasons.length > 0) {
    return item.similarity_reasons;
  }

  if (!queryContext) {
    return [];
  }

  const reasons: string[] = [];
  const queryDistrict = (queryContext.district ?? "").trim().toLocaleLowerCase("tr-TR");
  const queryNeighborhood = (queryContext.neighborhood ?? "").trim().toLocaleLowerCase("tr-TR");
  const queryRooms = (queryContext.rooms ?? "").trim().toLocaleLowerCase("tr-TR");
  const queryFloor = (queryContext.floor ?? "").trim().toLocaleLowerCase("tr-TR");

  if (queryDistrict && item.district.trim().toLocaleLowerCase("tr-TR") === queryDistrict) {
    reasons.push("Aynı ilçe");
  }

  if (queryNeighborhood && item.neighborhood.trim().toLocaleLowerCase("tr-TR") === queryNeighborhood) {
    reasons.push("Aynı mahalle");
  }

  if (queryRooms && item.rooms.trim().toLocaleLowerCase("tr-TR") === queryRooms) {
    reasons.push("Aynı oda tipi");
  }

  const queryM2 = Number(queryContext.m2_gross);
  if (Number.isFinite(queryM2) && typeof item.m2_gross === "number" && queryM2 > 0) {
    const ratio = Math.abs(queryM2 - item.m2_gross) / Math.max(queryM2, item.m2_gross, 1);
    if (ratio <= 0.15) {
      reasons.push("m² yakın");
    }
  }

  if (queryFloor && item.floor && item.floor.trim().toLocaleLowerCase("tr-TR") === queryFloor) {
    reasons.push("Kat bilgisi yakın");
  }

  return reasons.slice(0, 4);
}

export function PredictionResultDialog({
  open,
  onOpenChange,
  data,
  onContinueEditing,
  queryContext,
}: PredictionResultDialogProps) {
  const explanationItems = useMemo(() => (data ? buildExplanationItems(data) : []), [data]);

  const rangeValues = useMemo(() => {
    if (!data) {
      return null;
    }

    return {
      lower: Math.round(data.predicted_rent_try * 0.9),
      upper: Math.round(data.predicted_rent_try * 1.1),
    };
  }, [data]);

  const confidenceAppearance = useMemo(() => getConfidenceAppearance(data?.confidence_label), [data?.confidence_label]);

  useEffect(() => {
    if (!open) {
      return;
    }

    const previousOverflow = document.body.style.overflow;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onOpenChange(false);
      }
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onOpenChange, open]);

  if (!open || !data || !rangeValues || typeof document === "undefined") {
    return null;
  }

  return createPortal(
    <div
      className="fixed inset-0 z-[90] flex items-end justify-center bg-slate-950/45 p-0 sm:items-center sm:p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="prediction-result-title"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onOpenChange(false);
        }
      }}
    >
      <div className="flex h-[100dvh] w-full max-w-[1120px] flex-col overflow-hidden rounded-none bg-white shadow-2xl sm:h-auto sm:max-h-[calc(100dvh-2rem)] sm:rounded-2xl">
        <div className="flex items-start justify-between gap-4 border-b border-border px-4 py-4 sm:px-6">
          <div className="space-y-1">
            <h2 id="prediction-result-title" className="text-[22px] font-semibold tracking-[-0.03em] text-foreground sm:text-[26px]">
              Kira tahmini hazır
            </h2>
            <p className="max-w-2xl text-sm leading-6 text-muted-foreground">
              Girilen bilgiler, açıklama metni ve fotoğraflar birlikte değerlendirilmiştir.
            </p>
          </div>

          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => onOpenChange(false)}
            aria-label="Sonuç penceresini kapat"
            className="shrink-0"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 sm:px-6 sm:py-5">
          <div className="space-y-5">
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1.45fr)_minmax(300px,0.95fr)]">
              <div className="rounded-2xl border border-[#0B3A75]/12 bg-[#F8FBFF] px-5 py-5">
                <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Beklenen kira aralığı</div>
                <div className="mt-2 break-words text-[28px] font-semibold tracking-[-0.04em] text-primary sm:text-[38px]">
                  {formatCurrencyLabel(rangeValues.lower)} – {formatCurrencyLabel(rangeValues.upper)}
                </div>
                <div className="mt-3 text-sm text-muted-foreground">
                  Merkez tahmin: <span className="font-semibold text-foreground">{data.predicted_rent_formatted}</span>
                </div>
                <div className="mt-2 text-xs leading-5 text-muted-foreground">
                  Analiz edilen görsel sayısı: <span className="font-medium text-foreground">{data.used_image_count}</span>
                </div>
              </div>

              <div className="rounded-2xl border border-border bg-white px-4 py-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-semibold text-foreground">Tahmin güven seviyesi</div>
                  <div className={`rounded-md px-2.5 py-1 text-xs font-semibold ${confidenceAppearance.badgeClassName}`}>
                    {data.confidence_label ?? "Orta"}
                  </div>
                </div>
                <div className="mt-3 flex items-center justify-between gap-3">
                  <div className="text-xs leading-5 text-muted-foreground">
                    Bu skor, girilen bilgilerin veri setindeki benzer örneklerle ne kadar uyumlu olduğunu gösterir.
                  </div>
                  <div className={`shrink-0 text-sm font-semibold ${confidenceAppearance.textClassName}`}>
                    {data.confidence_score}/100
                  </div>
                </div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full transition-all ${confidenceAppearance.barClassName}`}
                    style={{ width: `${Math.max(6, Math.min(100, data.confidence_score))}%` }}
                  />
                </div>
                {data.confidence_reasons.length > 0 ? (
                  <div className="mt-3 space-y-2">
                    {data.confidence_reasons.slice(0, 3).map((reason) => (
                      <div key={reason} className="text-xs leading-5 text-muted-foreground">
                        • {reason}
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>
            </div>

            <ResultSection title="Tahmin detayları">
              <div className="space-y-3">
                {explanationItems.length > 0 ? (
                  <ResultAccordion title="Tahmini etkileyen faktörler" count={explanationItems.length} defaultOpen={false}>
                    <div className="space-y-2.5">
                      {explanationItems.map((item) => (
                        <div
                          key={`${item.tone}-${item.message}`}
                          className={`rounded-xl border px-4 py-3 text-sm leading-6 ${
                            item.tone === "positive"
                              ? "border-success/20 bg-success/5 text-success"
                              : "border-error/20 bg-error/5 text-error"
                          }`}
                        >
                          {item.message}
                        </div>
                      ))}
                    </div>
                  </ResultAccordion>
                ) : null}

                {data.similar_listings.length > 0 ? (
                  <ResultAccordion title="Benzer piyasa örnekleri" count={Math.min(5, data.similar_listings.length)} defaultOpen={false}>
                    <div className="space-y-3">
                      <p className="text-xs leading-5 text-muted-foreground">
                        Bu örnekler fiyatı birebir doğrulamak için değil, girilen ilana benzeyen geçmiş kayıtları göstermek için listelenir.
                      </p>

                      {data.similar_listings.slice(0, 5).map((item) => {
                        const reasons = buildFallbackSimilarityReasons(item, queryContext);

                        return (
                          <div
                            key={`${item.district}-${item.neighborhood}-${item.rooms}-${item.price_try}`}
                            className="rounded-xl border border-border bg-slate-50/70 px-4 py-3"
                          >
                            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                              <div className="min-w-0 flex-1">
                                <div className="break-words text-sm font-semibold text-foreground">
                                  {item.district} / {item.neighborhood}
                                </div>
                                <div className="mt-1 break-words text-xs leading-5 text-muted-foreground">
                                  {buildSimilarListingMeta(item)}
                                </div>
                              </div>
                              <div className="inline-flex w-fit shrink-0 rounded-full bg-white px-2.5 py-1 text-xs font-semibold text-foreground">
                                %{item.similarity_score}
                              </div>
                            </div>

                            {reasons.length > 0 ? (
                              <div className="mt-3 flex flex-wrap gap-2">
                                {reasons.map((reason) => (
                                  <span
                                    key={`${item.district}-${item.neighborhood}-${reason}`}
                                    className="rounded-full bg-[#EEF4FF] px-2.5 py-1 text-[11px] font-medium text-primary"
                                  >
                                    {reason}
                                  </span>
                                ))}
                              </div>
                            ) : null}

                            <div className="mt-3 text-xs text-muted-foreground">
                              İlan kirası: <span className="font-semibold text-foreground">{item.price_formatted}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </ResultAccordion>
                ) : null}
              </div>
            </ResultSection>

            <ResultSection title="Bilgilendirme">
              <div className="space-y-3">
                <div className="rounded-xl border border-border bg-white px-4 py-3">
                  <div className="flex items-start gap-3">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#F8FBFF] text-primary">
                      <Sparkles className="h-4 w-4" />
                    </div>
                    <div className="space-y-1">
                      <div className="text-sm font-semibold text-foreground">Kira tahmini oluşturuldu.</div>
                      <div className="text-xs leading-5 text-muted-foreground">
                        İlan bilgileri, açıklama metni ve fotoğraflar birlikte değerlendirilerek sonuç üretildi.
                      </div>
                      <div className="text-xs leading-5 text-muted-foreground">
                        Bu tahmin karar destek amaçlıdır, resmi ekspertiz yerine geçmez.
                      </div>
                    </div>
                  </div>
                </div>

                {data.warnings.length > 0 ? (
                  <div className="rounded-xl border border-warning/20 bg-warning/5 px-4 py-3">
                    <div className="space-y-2">
                      {data.warnings.map((warning) => (
                        <div key={warning} className="text-xs leading-5 text-warning">
                          • {warning}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            </ResultSection>
          </div>
        </div>

        <div className="flex flex-col-reverse gap-2 border-t border-border px-4 py-4 sm:flex-row sm:justify-end sm:px-6">
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              onOpenChange(false);
              onContinueEditing?.();
            }}
          >
            Yeni tahmin yap
          </Button>
          <Button type="button" onClick={() => onOpenChange(false)}>
            <CheckCircle2 className="h-4 w-4" />
            Sonucu kapat
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
