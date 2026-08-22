"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertTriangle, LoaderCircle, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type FieldErrors, useForm } from "react-hook-form";

import { AnalysisStatusBar, type PredictionAnalysisItem } from "@/components/predict/AnalysisStatusBar";
import { CompactSection } from "@/components/predict/CompactSection";
import { LocationSelector } from "@/components/predict/LocationSelector";
import { MetadataSelect } from "@/components/predict/MetadataSelect";
import { NumericField } from "@/components/predict/NumericField";
import { PhotoPanel } from "@/components/predict/PhotoPanel";
import { PredictionResultDialog } from "@/components/predict/PredictionResultDialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { createPrediction, PredictionApiError } from "@/lib/api/prediction";
import { PREDICT_LOADING_STEPS } from "@/lib/constants";
import { loadPredictionMetadata } from "@/lib/meta/load-metadata";
import { predictionDefaultValues, predictionFormSchema, type PredictionFormValues } from "@/lib/validation/prediction-schema";

const FIELD_ORDER: Array<keyof PredictionFormValues> = [
  "city",
  "district",
  "neighborhood",
  "rooms",
  "m2_gross",
  "bathrooms",
  "building_age",
  "floor",
  "total_floors",
  "heating_type",
  "fuel_type",
  "is_furnished",
  "dues_try",
  "home_shape",
  "title",
  "description",
  "images",
];

function PredictWorkspaceSkeleton() {
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }).map((_, index) => (
          <div key={index} className="rounded-[18px] border border-border bg-white px-5 py-5 shadow-[0_12px_28px_rgba(14,42,89,0.05)]">
            <div className="flex items-center gap-3">
              <div className="h-14 w-14 animate-pulse rounded-full bg-slate-100" />
              <div className="min-w-0 flex-1 space-y-2">
                <div className="h-3.5 w-28 animate-pulse rounded-full bg-slate-100" />
                <div className="h-3 w-24 animate-pulse rounded-full bg-slate-100" />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_440px]">
        <Card className="rounded-[18px] border-[#DDE7F0] shadow-[0_14px_36px_rgba(14,42,89,0.06)]">
          <CardContent className="space-y-6 px-6 py-6">
            {Array.from({ length: 3 }).map((_, index) => (
              <div key={index} className={index > 0 ? "border-t border-border pt-5" : ""}>
                <div className="mb-4 h-4 w-32 animate-pulse rounded-full bg-slate-200" />
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {Array.from({ length: index === 2 ? 2 : 4 }).map((__, innerIndex) => (
                    <div key={innerIndex} className="space-y-2">
                      <div className="h-3.5 w-24 animate-pulse rounded-full bg-slate-100" />
                      <div className="h-10 animate-pulse rounded-lg bg-slate-100" />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="rounded-[18px] border-[#DDE7F0] shadow-[0_14px_36px_rgba(14,42,89,0.06)]">
          <CardContent className="space-y-4 px-6 py-5">
            <div className="h-4 w-32 animate-pulse rounded-full bg-slate-200" />
            <div className="h-52 animate-pulse rounded-[18px] bg-slate-100" />
            <div className="h-36 animate-pulse rounded-[18px] bg-slate-100" />
            <div className="h-12 animate-pulse rounded-xl bg-slate-100" />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export function PredictWorkspace() {
  const workspaceTopRef = useRef<HTMLDivElement>(null);
  const [isResultDialogOpen, setIsResultDialogOpen] = useState(false);
  const [loadingStepIndex, setLoadingStepIndex] = useState(0);
  const [lastSubmittedValues, setLastSubmittedValues] = useState<PredictionFormValues | null>(null);

  const form = useForm<PredictionFormValues>({
    resolver: zodResolver(predictionFormSchema),
    defaultValues: predictionDefaultValues,
    mode: "onBlur",
  });

  const watchedValues = form.watch();

  const metadataQuery = useQuery({
    queryKey: ["prediction-metadata"],
    queryFn: loadPredictionMetadata,
    staleTime: Number.POSITIVE_INFINITY,
  });

  const mutation = useMutation({
    mutationFn: async (values: PredictionFormValues) => createPrediction(values, values.images),
    onSuccess: () => {
      setIsResultDialogOpen(true);
    },
  });

  const setFieldValue = useCallback(
    (fieldName: keyof PredictionFormValues, value: string | File[]) => {
      form.setValue(fieldName, value as never, {
        shouldDirty: true,
        shouldTouch: true,
        shouldValidate: true,
      });
    },
    [form],
  );

  useEffect(() => {
    if (!metadataQuery.data?.locations.cities[0]) {
      return;
    }

    if (!form.getValues("city")) {
      setFieldValue("city", metadataQuery.data.locations.cities[0].value);
    }

    if (!form.getValues("home_type")) {
      setFieldValue("home_type", "Daire");
    }
  }, [form, metadataQuery.data, setFieldValue]);

  useEffect(() => {
    if (!mutation.isPending) {
      setLoadingStepIndex(0);
      return;
    }

    const intervalId = window.setInterval(() => {
      setLoadingStepIndex((current) => (current + 1) % PREDICT_LOADING_STEPS.length);
    }, 1400);

    return () => window.clearInterval(intervalId);
  }, [mutation.isPending]);

  useEffect(() => {
    if (!mutation.isError) {
      return;
    }

    window.requestAnimationFrame(() => {
      workspaceTopRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
  }, [mutation.isError]);

  const analysisItems = useMemo<PredictionAnalysisItem[]>(() => {
    const hasLocation = Boolean(watchedValues.city && watchedValues.district && watchedValues.neighborhood);
    const hasFeatures = Boolean(watchedValues.rooms && watchedValues.m2_gross);
    const hasTitle = Boolean(watchedValues.title.trim());
    const hasDescription = Boolean(watchedValues.description.trim());
    const hasImages = watchedValues.images.length > 0;

    return [
      { label: "Konum bilgileri", status: hasLocation ? "ready" : "missing" },
      { label: "Konut özellikleri", status: hasFeatures ? "ready" : "missing" },
      { label: "Konut fotoğraflarını yükle", status: hasImages ? "ready" : "optional" },
      { label: "İlan açıklaması", status: hasTitle && hasDescription ? "ready" : "missing" },
    ];
  }, [
    watchedValues.city,
    watchedValues.description,
    watchedValues.district,
    watchedValues.images.length,
    watchedValues.m2_gross,
    watchedValues.neighborhood,
    watchedValues.rooms,
    watchedValues.title,
  ]);

  const handleInvalidSubmit = useCallback((errors: FieldErrors<PredictionFormValues>) => {
    const firstField = FIELD_ORDER.find((field) => errors[field]);
    if (!firstField) {
      return;
    }

    const element = document.getElementById(firstField);
    element?.scrollIntoView({ behavior: "smooth", block: "center" });

    if (element instanceof HTMLElement) {
      window.setTimeout(() => {
        element.focus({ preventScroll: true });
      }, 120);
    }
  }, []);

  const handleCityChange = useCallback(
    (nextCity: string) => {
      setFieldValue("city", nextCity);
      setFieldValue("district", "");
      setFieldValue("neighborhood", "");
    },
    [setFieldValue],
  );

  const handleDistrictChange = useCallback(
    (nextDistrict: string) => {
      setFieldValue("district", nextDistrict);
      setFieldValue("neighborhood", "");
    },
    [setFieldValue],
  );

  const handleSubmit = form.handleSubmit(
    (values) => {
      mutation.reset();
      setIsResultDialogOpen(false);
      setLastSubmittedValues({
        ...values,
        home_type: values.home_type || "Daire",
      });
      mutation.mutate({
        ...values,
        home_type: values.home_type || "Daire",
      });
    },
    handleInvalidSubmit,
  );

  if (metadataQuery.isPending) {
    return <PredictWorkspaceSkeleton />;
  }

  if (metadataQuery.isError || !metadataQuery.data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Form seçenekleri yüklenemedi</CardTitle>
          <CardDescription>
            İlçe, mahalle ve konut özellikleri için gereken metadata alınamadı. Lütfen tekrar deneyin.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button type="button" onClick={() => void metadataQuery.refetch()}>
            <RefreshCw className="h-4 w-4" />
            Yeniden dene
          </Button>
        </CardContent>
      </Card>
    );
  }

  const metadata = metadataQuery.data;
  const loadingMessage = PREDICT_LOADING_STEPS[loadingStepIndex];
  const submissionErrorMessage = mutation.isError
    ? mutation.error instanceof PredictionApiError
      ? mutation.error.message
      : "Tahmin oluşturulamadı. Lütfen tekrar deneyin."
    : null;

  return (
    <div ref={workspaceTopRef} className="space-y-4 pb-24 md:pb-0">
      <AnalysisStatusBar items={analysisItems} />

      <form
        id="prediction-form"
        className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_440px]"
        onSubmit={handleSubmit}
        noValidate
      >
        <Card className="overflow-hidden rounded-[18px] border-[#DDE7F0] bg-white shadow-[0_14px_36px_rgba(14,42,89,0.06)]">
          <CardContent className="space-y-5 px-6 py-6">
            {submissionErrorMessage ? (
              <div className="rounded-xl border border-error/20 bg-error/5 px-4 py-3">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-error" />
                  <div className="space-y-1">
                    <div className="text-sm font-semibold text-foreground">Tahmin oluşturulamadı</div>
                    <p className="text-sm leading-6 text-muted-foreground">{submissionErrorMessage}</p>
                  </div>
                </div>
              </div>
            ) : null}

            {mutation.isPending ? (
              <div className="rounded-xl border border-primary/15 bg-[#F8FBFF] px-4 py-3">
                <div className="flex items-start gap-3">
                  <LoaderCircle className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-primary" />
                  <div className="space-y-1">
                    <div className="text-sm font-semibold text-foreground">{loadingMessage}…</div>
                    <p className="text-xs leading-5 text-muted-foreground">
                      Sonuç hazır olduğunda geniş sonuç penceresi otomatik olarak açılır.
                    </p>
                  </div>
                </div>
              </div>
            ) : null}

            <CompactSection title="Konum" description="Şehir, ilçe ve mahalle seçimi beklenen kira seviyesini doğrudan etkiler." first>
              <LocationSelector
                metadata={metadata.locations}
                city={watchedValues.city ?? ""}
                district={watchedValues.district ?? ""}
                neighborhood={watchedValues.neighborhood ?? ""}
                onCityChange={handleCityChange}
                onDistrictChange={handleDistrictChange}
                onNeighborhoodChange={(value) => setFieldValue("neighborhood", value)}
                errors={{
                  city: form.formState.errors.city?.message,
                  district: form.formState.errors.district?.message,
                  neighborhood: form.formState.errors.neighborhood?.message,
                }}
                disabled={mutation.isPending}
              />
            </CompactSection>

            <CompactSection title="Konut özellikleri" description="Konut tipini, kat bilgisini ve diğer temel nitelikleri seçin.">
              <div className="grid gap-x-5 gap-y-[14px] sm:grid-cols-2 lg:grid-cols-4">
                <MetadataSelect
                  id="rooms"
                  label="Oda tipi"
                  value={watchedValues.rooms ?? ""}
                  onValueChange={(value) => setFieldValue("rooms", value)}
                  options={metadata.categorical.rooms.options}
                  placeholder="Seçin"
                  errorMessage={form.formState.errors.rooms?.message}
                  disabled={mutation.isPending}
                  required
                />

                <NumericField
                  id="m2_gross"
                  label="Brüt m²"
                  value={watchedValues.m2_gross ?? ""}
                  onValueChange={(value) => setFieldValue("m2_gross", value)}
                  metadata={metadata.numeric.m2_gross}
                  errorMessage={form.formState.errors.m2_gross?.message}
                  disabled={mutation.isPending}
                  required
                  unitLabel="m²"
                />

                <MetadataSelect
                  id="bathrooms"
                  label="Banyo"
                  value={watchedValues.bathrooms ?? ""}
                  onValueChange={(value) => setFieldValue("bathrooms", value)}
                  options={metadata.categorical.bathrooms.options}
                  placeholder="Seçin"
                  errorMessage={form.formState.errors.bathrooms?.message}
                  disabled={mutation.isPending}
                />

                <div className="space-y-1.5">
                  <Label htmlFor="building_age" className="text-[13px] font-medium">
                    Bina yaşı
                  </Label>
                  <Input
                    id="building_age"
                    type="number"
                    min={0}
                    max={100}
                    step={1}
                    inputMode="numeric"
                    value={watchedValues.building_age ?? ""}
                    onChange={(event) => setFieldValue("building_age", event.target.value)}
                    placeholder="Örn. 8"
                    disabled={mutation.isPending}
                    aria-invalid={Boolean(form.formState.errors.building_age?.message)}
                    className={form.formState.errors.building_age?.message ? "border-error/40 focus-visible:border-error" : ""}
                  />
                  {form.formState.errors.building_age?.message ? (
                    <p className="text-[12px] text-error">{form.formState.errors.building_age.message}</p>
                  ) : (
                    <p className="text-[12px] text-muted-foreground">Bina yaşını girin.</p>
                  )}
                </div>

                <MetadataSelect
                  id="floor"
                  label="Bulunduğu kat"
                  value={watchedValues.floor ?? ""}
                  onValueChange={(value) => setFieldValue("floor", value)}
                  options={metadata.categorical.floor.options}
                  placeholder="Seçin"
                  errorMessage={form.formState.errors.floor?.message}
                  disabled={mutation.isPending}
                  searchable
                  searchPlaceholder="Kat ara"
                />

                <MetadataSelect
                  id="total_floors"
                  label="Toplam kat"
                  value={watchedValues.total_floors ?? ""}
                  onValueChange={(value) => setFieldValue("total_floors", value)}
                  options={metadata.categorical.total_floors.options}
                  placeholder="Seçin"
                  errorMessage={form.formState.errors.total_floors?.message}
                  disabled={mutation.isPending}
                />

                <MetadataSelect
                  id="heating_type"
                  label="Isıtma tipi"
                  value={watchedValues.heating_type ?? ""}
                  onValueChange={(value) => setFieldValue("heating_type", value)}
                  options={metadata.categorical.heating_type.options}
                  placeholder="Seçin"
                  errorMessage={form.formState.errors.heating_type?.message}
                  disabled={mutation.isPending}
                />

                <MetadataSelect
                  id="fuel_type"
                  label="Yakıt tipi"
                  value={watchedValues.fuel_type ?? ""}
                  onValueChange={(value) => setFieldValue("fuel_type", value)}
                  options={metadata.categorical.fuel_type.options}
                  placeholder="Seçin"
                  errorMessage={form.formState.errors.fuel_type?.message}
                  disabled={mutation.isPending}
                />

                <MetadataSelect
                  id="is_furnished"
                  label="Eşyalı mı?"
                  value={watchedValues.is_furnished ?? ""}
                  onValueChange={(value) => setFieldValue("is_furnished", value)}
                  options={metadata.categorical.is_furnished.options}
                  placeholder="Seçin"
                  errorMessage={form.formState.errors.is_furnished?.message}
                  disabled={mutation.isPending}
                />

                <NumericField
                  id="dues_try"
                  label="Aidat (TL)"
                  value={watchedValues.dues_try ?? ""}
                  onValueChange={(value) => setFieldValue("dues_try", value)}
                  metadata={metadata.numeric.dues_try}
                  errorMessage={form.formState.errors.dues_try?.message}
                  disabled={mutation.isPending}
                  unitLabel="TL"
                />

                <MetadataSelect
                  id="home_shape"
                  label="Konut şekli"
                  value={watchedValues.home_shape ?? ""}
                  onValueChange={(value) => setFieldValue("home_shape", value)}
                  options={metadata.categorical.home_shape.options}
                  placeholder="Seçin"
                  errorMessage={form.formState.errors.home_shape?.message}
                  disabled={mutation.isPending}
                />
              </div>
            </CompactSection>

            <CompactSection title="İlan açıklaması" description="İlan başlığı ve açıklama metni modeli daha iyi anlamamıza yardımcı olur.">
              <div className="grid gap-4 lg:grid-cols-[minmax(0,0.72fr)_minmax(0,1.28fr)]">
                <div className="space-y-1.5">
                  <Label htmlFor="title" className="text-[13px] font-medium">
                    İlan başlığı
                    <span className="ml-1 text-error">*</span>
                  </Label>
                  <Input
                    id="title"
                    placeholder="Örn. Çankaya'da geniş ve aydınlık 3+1 daire"
                    {...form.register("title")}
                    aria-invalid={Boolean(form.formState.errors.title?.message)}
                    disabled={mutation.isPending}
                    className={form.formState.errors.title?.message ? "border-error/40 focus-visible:border-error" : ""}
                  />
                  {form.formState.errors.title?.message ? (
                    <p className="text-[12px] text-error">{form.formState.errors.title.message}</p>
                  ) : null}
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor="description" className="text-[13px] font-medium">
                    Açıklama
                    <span className="ml-1 text-error">*</span>
                  </Label>
                  <Textarea
                    id="description"
                    placeholder="İlan açıklama metnini buraya yazın..."
                    {...form.register("description")}
                    aria-invalid={Boolean(form.formState.errors.description?.message)}
                    disabled={mutation.isPending}
                    className={
                      form.formState.errors.description?.message
                        ? "min-h-[84px] border-error/40 focus-visible:border-error"
                        : "min-h-[84px]"
                    }
                  />
                  {form.formState.errors.description?.message ? (
                    <p className="text-[12px] text-error">{form.formState.errors.description.message}</p>
                  ) : (
                    <p className="text-[12px] text-muted-foreground">Ulaşım, cephe, yenileme durumu ve site bilgileri tahmini destekler.</p>
                  )}
                </div>
              </div>
            </CompactSection>
          </CardContent>
        </Card>

        <PhotoPanel
          files={watchedValues.images}
          onChange={(nextFiles) => {
            setFieldValue("images", nextFiles);
            void form.trigger("images");
          }}
          errorMessage={form.formState.errors.images?.message}
          disabled={mutation.isPending}
          pending={mutation.isPending}
        />
      </form>

      <div className="mobile-safe-bottom fixed inset-x-0 bottom-0 z-40 border-t border-border bg-white/95 px-4 py-3 shadow-[0_-8px_24px_rgba(15,23,42,0.08)] backdrop-blur supports-[backdrop-filter]:bg-white/90 md:hidden">
        <div className="mx-auto w-full max-w-[1500px] space-y-2">
          <Button type="submit" form="prediction-form" variant="secondary" size="lg" disabled={mutation.isPending} className="w-full">
            {mutation.isPending ? (
              <>
                <LoaderCircle className="h-4 w-4 animate-spin" />
                Tahmin hazırlanıyor
              </>
            ) : (
              "Tahmini Al"
            )}
          </Button>
          <p className="text-center text-[11px] leading-4 text-muted-foreground">
            Eksik zorunlu alan varsa uyarı gösterilir.
          </p>
        </div>
      </div>

      <PredictionResultDialog
        open={isResultDialogOpen}
        onOpenChange={setIsResultDialogOpen}
        data={mutation.data ?? null}
        onContinueEditing={() => {
          workspaceTopRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }}
        queryContext={lastSubmittedValues ?? watchedValues}
      />
    </div>
  );
}
