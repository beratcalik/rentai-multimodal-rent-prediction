"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery } from "@tanstack/react-query";
import { LoaderCircle, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef } from "react";
import { type FieldErrors, useForm } from "react-hook-form";

import { createPrediction, PredictionApiError } from "@/lib/api/prediction";
import { loadPredictionMetadata } from "@/lib/meta/load-metadata";
import { predictionDefaultValues, predictionFormSchema, type PredictionFormValues } from "@/lib/validation/prediction-schema";
import { CompactSection } from "@/components/predict/CompactSection";
import { ImageUploader } from "@/components/predict/ImageUploader";
import { LocationSelector } from "@/components/predict/LocationSelector";
import { MetadataSelect } from "@/components/predict/MetadataSelect";
import { NumericField } from "@/components/predict/NumericField";
import { PredictionResultCard, type PredictionAnalysisItem } from "@/components/predict/PredictionResultCard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

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
  "home_type",
  "home_shape",
  "title",
  "description",
  "images",
];

function PredictWorkspaceSkeleton() {
  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
      <Card>
        <CardContent className="space-y-5 pt-5">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className={index > 0 ? "border-t border-border pt-5" : ""}>
              <div className="mb-4 h-4 w-32 animate-pulse rounded-full bg-slate-200" />
              <div className="grid gap-4 md:grid-cols-3">
                {Array.from({ length: index === 2 ? 2 : 3 }).map((__, innerIndex) => (
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
      <PredictionResultCard
        state={{ status: "empty" }}
        analysisItems={[
          { label: "Konum bilgileri", status: "missing" },
          { label: "Konut özellikleri", status: "missing" },
          { label: "İlan açıklaması", status: "missing" },
          { label: "Fotoğraflar", status: "optional" },
        ]}
      />
    </div>
  );
}

export function PredictWorkspace() {
  const resultPanelRef = useRef<HTMLDivElement>(null);

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
  }, [form, metadataQuery.data, setFieldValue]);

  const resultState = useMemo(() => {
    if (mutation.isPending) {
      return { status: "loading" as const };
    }

    if (mutation.isError) {
      const message =
        mutation.error instanceof PredictionApiError ? mutation.error.message : "Tahmin sonucu alınırken beklenmeyen bir hata oluştu.";

      return { status: "error" as const, message };
    }

    if (mutation.data) {
      return { status: "success" as const, data: mutation.data };
    }

    return { status: "empty" as const };
  }, [mutation.data, mutation.error, mutation.isError, mutation.isPending]);

  useEffect(() => {
    if (resultState.status !== "success" && resultState.status !== "error") {
      return;
    }

    window.requestAnimationFrame(() => {
      resultPanelRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
  }, [resultState.status]);

  const summaryText = useMemo(() => {
    const location =
      watchedValues.district && watchedValues.neighborhood ? `${watchedValues.district} / ${watchedValues.neighborhood}` : "Konum seçilmedi";
    const house = watchedValues.rooms && watchedValues.m2_gross ? `${watchedValues.rooms} · ${watchedValues.m2_gross} m²` : "Konut özeti eksik";
    return `${location} · ${house}`;
  }, [watchedValues.district, watchedValues.m2_gross, watchedValues.neighborhood, watchedValues.rooms]);

  const analysisItems = useMemo<PredictionAnalysisItem[]>(() => {
    const hasLocation = Boolean(watchedValues.city && watchedValues.district && watchedValues.neighborhood);
    const hasFeatures = Boolean(watchedValues.rooms && watchedValues.m2_gross);
    const hasText = Boolean(watchedValues.title.trim() && watchedValues.description.trim());
    const hasImages = watchedValues.images.length > 0;

    return [
      { label: "Konum bilgileri", status: hasLocation ? "ready" : "missing" },
      { label: "Konut özellikleri", status: hasFeatures ? "ready" : "missing" },
      { label: "İlan açıklaması", status: hasText ? "ready" : "missing" },
      { label: "Fotoğraflar", status: hasImages ? "ready" : "optional" },
    ];
  }, [watchedValues.city, watchedValues.description, watchedValues.district, watchedValues.images.length, watchedValues.m2_gross, watchedValues.neighborhood, watchedValues.rooms, watchedValues.title]);

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

  const handleSubmit = form.handleSubmit((values) => {
    mutation.reset();
    mutation.mutate(values);
  }, handleInvalidSubmit);

  if (metadataQuery.isPending) {
    return <PredictWorkspaceSkeleton />;
  }

  if (metadataQuery.isError || !metadataQuery.data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Form seçenekleri yüklenemedi</CardTitle>
          <CardDescription>İlçe, mahalle ve konut özellikleri için gereken metadata alınamadı. Lütfen tekrar deneyin.</CardDescription>
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

  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
      <Card>
        <CardContent className="pt-5">
          <form className="space-y-5" onSubmit={handleSubmit}>
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

            <CompactSection title="Konut özellikleri" description="Kullanıcıya mantıklı ve seçilebilir seçenekler sunulur.">
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
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
                  <Label htmlFor="building_age">Bina yaşı</Label>
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
                    <p className="text-xs text-error">{form.formState.errors.building_age.message}</p>
                  ) : (
                    <p className="text-xs text-muted-foreground">Bina yaşını girin.</p>
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
                  label="Aidat"
                  value={watchedValues.dues_try ?? ""}
                  onValueChange={(value) => setFieldValue("dues_try", value)}
                  metadata={metadata.numeric.dues_try}
                  errorMessage={form.formState.errors.dues_try?.message}
                  disabled={mutation.isPending}
                  unitLabel="TL"
                />

                <MetadataSelect
                  id="home_type"
                  label="Konut tipi"
                  value={watchedValues.home_type ?? ""}
                  onValueChange={(value) => setFieldValue("home_type", value)}
                  options={metadata.categorical.home_type.options}
                  placeholder="Seçin"
                  errorMessage={form.formState.errors.home_type?.message}
                  disabled={mutation.isPending}
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

            <CompactSection title="İlan metni" description="Başlık ve açıklama, görsellerin söylemediklerini tamamlar.">
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <Label htmlFor="title">
                    Başlık
                    <span className="ml-1 text-error">*</span>
                  </Label>
                  <Input
                    id="title"
                    placeholder="Örn. Merkezi konumda, geniş balkonlu aile dairesi"
                    {...form.register("title")}
                    aria-invalid={Boolean(form.formState.errors.title?.message)}
                    disabled={mutation.isPending}
                    className={form.formState.errors.title?.message ? "border-error/40 focus-visible:border-error" : ""}
                  />
                  {form.formState.errors.title?.message ? (
                    <p className="text-xs text-error">{form.formState.errors.title.message}</p>
                  ) : (
                    <p className="text-xs text-muted-foreground">Kısa ve açıklayıcı bir başlık yeterlidir.</p>
                  )}
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor="description">
                    Açıklama
                    <span className="ml-1 text-error">*</span>
                  </Label>
                  <Textarea
                    id="description"
                    placeholder="Cephe, ulaşım, yenileme durumu, otopark, asansör ve site bilgilerini paylaşın."
                    {...form.register("description")}
                    aria-invalid={Boolean(form.formState.errors.description?.message)}
                    disabled={mutation.isPending}
                    className={form.formState.errors.description?.message ? "min-h-[120px] border-error/40 focus-visible:border-error" : "min-h-[120px]"}
                  />
                  {form.formState.errors.description?.message ? (
                    <p className="text-xs text-error">{form.formState.errors.description.message}</p>
                  ) : (
                    <p className="text-xs text-muted-foreground">Fotoğrafta görünmeyen detaylar burada tahmini destekler.</p>
                  )}
                </div>
              </div>
            </CompactSection>

            <CompactSection title="Fotoğraflar" description="Fotoğraf eklemek zorunlu değildir, ancak tahmin kalitesini artırabilir.">
              <ImageUploader
                files={watchedValues.images}
                onChange={(nextFiles) => {
                  setFieldValue("images", nextFiles);
                  void form.trigger("images");
                }}
                errorMessage={form.formState.errors.images?.message}
                disabled={mutation.isPending}
              />
            </CompactSection>

            <CompactSection title="Tahmin" description={summaryText}>
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="text-sm text-muted-foreground">Bilgileri gözden geçirin. Sonuç sağ panelde gösterilecektir.</div>
                <Button type="submit" variant="secondary" size="lg" disabled={mutation.isPending} className="min-w-[180px]">
                  {mutation.isPending ? (
                    <>
                      <LoaderCircle className="h-4 w-4 animate-spin" />
                      Tahmin hazırlanıyor
                    </>
                  ) : (
                    "Tahmini Al"
                  )}
                </Button>
              </div>
            </CompactSection>
          </form>
        </CardContent>
      </Card>

      <div ref={resultPanelRef}>
        <PredictionResultCard state={resultState} analysisItems={analysisItems} />
      </div>
    </div>
  );
}
