"use client";

import { AlertTriangle, CheckCircle2, Home, LoaderCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { PREDICT_LOADING_STEPS, PREDICTION_MODALITIES } from "@/lib/constants";
import { formatNumberTr } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { PredictionResponse } from "@/lib/validation/prediction-schema";

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

export function PredictionResultCard({ state, analysisItems }: PredictionResultCardProps) {
  const [loadingStepIndex, setLoadingStepIndex] = useState(0);

  useEffect(() => {
    if (state.status !== "loading") {
      setLoadingStepIndex(0);
      return;
    }

    const intervalId = window.setInterval(() => {
      setLoadingStepIndex((current) => (current + 1) % PREDICT_LOADING_STEPS.length);
    }, 1500);

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

  return (
    <Card className="xl:sticky xl:top-20">
      <CardHeader className="space-y-2 border-b border-border">
        <CardTitle className="text-[22px]">Tahmin sonucu</CardTitle>
        <CardDescription>Bilgileri girin, beklenen kira aralığını görün.</CardDescription>
      </CardHeader>

      <CardContent className="space-y-4 pt-5">
        {state.status === "loading" ? (
          <div className="space-y-3">
            <div className="rounded-xl border border-border bg-slate-50/70 px-4 py-4">
              <div className="flex items-center gap-3 text-sm font-semibold text-foreground">
                <LoaderCircle className="h-4 w-4 animate-spin text-[#0057B8]" />
                {PREDICT_LOADING_STEPS[loadingStepIndex]}...
              </div>
            </div>

            {PREDICT_LOADING_STEPS.map((step, index) => {
              const isComplete = index < loadingStepIndex;
              const isActive = index === loadingStepIndex;

              return (
                <div key={step} className="flex items-center gap-3 rounded-lg border border-border px-3 py-3">
                  <div
                    className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-semibold ${
                      isComplete ? "bg-success text-white" : isActive ? "bg-[#0057B8] text-white" : "bg-slate-100 text-muted-foreground"
                    }`}
                  >
                    {isComplete ? <CheckCircle2 className="h-3.5 w-3.5" /> : index + 1}
                  </div>
                  <div className="text-sm text-foreground">{step}</div>
                </div>
              );
            })}
          </div>
        ) : null}

        {state.status === "success" && rangeValues ? (
          <div className="space-y-4">
            <div className="rounded-xl border border-[#0B3A75]/12 bg-[#F8FBFF] px-4 py-5">
              <div className="text-xs uppercase tracking-[0.12em] text-muted-foreground">Beklenen kira aralığı</div>
              <div className="mt-2 text-[30px] font-semibold tracking-[-0.03em] text-primary">
                {formatRangeLabel(rangeValues.lower)} – {formatRangeLabel(rangeValues.upper)}
              </div>
              <div className="mt-3 text-sm text-muted-foreground">
                Merkez tahmin: <span className="font-medium text-foreground">{state.data.predicted_rent_formatted}</span>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg border border-border px-3 py-3">
                <div className="text-xs text-muted-foreground">Kullanılan görsel sayısı</div>
                <div className="mt-1 text-sm font-semibold text-foreground">{state.data.used_image_count}</div>
              </div>
              <div className="rounded-lg border border-border px-3 py-3">
                <div className="text-xs text-muted-foreground">Tahmin modeli</div>
                <div className="mt-1 text-sm font-semibold text-foreground">Multimodal kira modeli</div>
              </div>
            </div>

            {state.data.warnings.length > 0 ? (
              <div className="rounded-lg border border-warning/20 bg-warning/5 px-4 py-3">
                <div className="mb-2 text-sm font-semibold text-foreground">Uyarılar</div>
                <div className="space-y-2">
                  {state.data.warnings.map((warning) => (
                    <p key={warning} className="text-xs leading-5 text-warning">
                      {warning}
                    </p>
                  ))}
                </div>
              </div>
            ) : null}

            <div className="space-y-3 rounded-lg border border-border px-4 py-4">
              <div className="text-sm font-semibold text-foreground">Analizde kullanılan veri kaynakları</div>
              {PREDICTION_MODALITIES.map((item) => (
                <div key={item.title} className="flex gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-primary">
                    <item.icon className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-foreground">{item.title}</div>
                    <div className="text-xs leading-5 text-muted-foreground">{item.description}</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="space-y-1">
              <div className="text-sm font-medium text-foreground">Kira tahmini oluşturuldu.</div>
              <div className="text-xs leading-5 text-muted-foreground">{state.data.message}</div>
              <div className="text-xs leading-5 text-muted-foreground">Bu tahmin karar destek amaçlıdır, ekspertiz yerine geçmez.</div>
            </div>
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
            <div className="text-xs leading-5 text-muted-foreground">Servis tekrar erişilebilir olduğunda aynı ekrandan yeniden tahmin alabilirsiniz.</div>
          </div>
        ) : null}

        {state.status === "empty" ? (
          <>
            <div className="rounded-xl border border-border bg-slate-50/70 px-4 py-5">
              <div className="flex items-start gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#FFF8CC] text-primary">
                  <Home className="h-4 w-4" />
                </div>
                <div className="space-y-1.5">
                  <div className="text-sm font-semibold text-foreground">Bilgileri girin ve tahmini alın.</div>
                  <p className="text-sm leading-6 text-muted-foreground">Sonuç burada gösterilecek.</p>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-border bg-white">
              <div className="border-b border-border px-4 py-3 text-sm font-semibold text-foreground">Analizde kullanılacaklar</div>
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
          </>
        ) : null}
      </CardContent>
    </Card>
  );
}
