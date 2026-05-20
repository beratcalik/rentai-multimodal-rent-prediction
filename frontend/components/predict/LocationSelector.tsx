"use client";

import { useEffect, useMemo } from "react";

import type { LocationsMetadata, MetadataOption } from "@/lib/meta/load-metadata";
import { MetadataSelect } from "@/components/predict/MetadataSelect";

type LocationSelectorProps = {
  metadata: LocationsMetadata;
  city: string;
  district: string;
  neighborhood: string;
  onCityChange: (value: string) => void;
  onDistrictChange: (value: string) => void;
  onNeighborhoodChange: (value: string) => void;
  errors?: {
    city?: string;
    district?: string;
    neighborhood?: string;
  };
  disabled?: boolean;
};

function toOptions(items: Array<{ value: string; label: string; count: number }>): MetadataOption[] {
  return items.map((item) => ({
    value: item.value,
    rawValue: item.value,
    label: item.label,
    count: item.count,
    isEmpty: false,
  }));
}

export function LocationSelector({
  metadata,
  city,
  district,
  neighborhood,
  onCityChange,
  onDistrictChange,
  onNeighborhoodChange,
  errors,
  disabled = false,
}: LocationSelectorProps) {
  useEffect(() => {
    if (!city && metadata.cities[0]) {
      onCityChange(metadata.cities[0].value);
    }
  }, [city, metadata.cities, onCityChange]);

  const selectedCity = useMemo(
    () => metadata.cities.find((item) => item.value === city) ?? metadata.cities[0],
    [city, metadata.cities],
  );

  const districtOptions = useMemo(() => toOptions(selectedCity?.districts ?? []), [selectedCity]);

  const selectedDistrict = useMemo(
    () => selectedCity?.districts.find((item) => item.value === district),
    [district, selectedCity],
  );

  const neighborhoodOptions = useMemo(() => toOptions(selectedDistrict?.neighborhoods ?? []), [selectedDistrict]);

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <MetadataSelect
        id="city"
        label="Şehir"
        value={city}
        onValueChange={onCityChange}
        options={toOptions(metadata.cities)}
        placeholder="Şehir seçin"
        helperText="Şu an Ankara verisiyle çalışır."
        errorMessage={errors?.city}
        disabled={disabled}
        required
      />

      <MetadataSelect
        id="district"
        label="İlçe"
        value={district}
        onValueChange={onDistrictChange}
        options={districtOptions}
        placeholder="İlçe seçin"
        helperText="İlçe değişirse mahalle listesi güncellenir."
        errorMessage={errors?.district}
        disabled={disabled}
        required
      />

      <MetadataSelect
        id="neighborhood"
        label="Mahalle"
        value={neighborhood}
        onValueChange={onNeighborhoodChange}
        options={neighborhoodOptions}
        placeholder={district ? "Mahalle seçin" : "Önce ilçe seçin"}
        helperText="Mahalle alanında arama yapabilirsiniz."
        errorMessage={errors?.neighborhood}
        disabled={disabled || !district}
        searchable
        searchPlaceholder="Mahalle ara"
        emptyMessage="Bu ilçe için mahalle bulunamadı."
        required
      />
    </div>
  );
}
