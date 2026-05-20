import { normalizeCategoricalOptions, normalizeLocationTree } from "./normalized-options";

export type RawMetaValue = string | number | boolean | null;

export type MetadataOption = {
  value: string;
  rawValue: RawMetaValue;
  label: string;
  count: number;
  isEmpty: boolean;
};

export type PredictionCategoricalField =
  | "rooms"
  | "floor"
  | "heating_type"
  | "fuel_type"
  | "home_type"
  | "home_shape"
  | "is_furnished"
  | "bathrooms"
  | "building_age"
  | "total_floors";

export type PredictionNumericField = "m2_gross" | "dues_try" | "price_try";

export type LocationNode = {
  value: string;
  label: string;
  count: number;
};

export type DistrictNode = LocationNode & {
  neighborhoods: LocationNode[];
};

export type CityNode = LocationNode & {
  districts: DistrictNode[];
};

export type LocationsMetadata = {
  generatedAt: string;
  sourceDataset: string;
  totalCities: number;
  totalDistricts: number;
  totalNeighborhoods: number;
  cities: CityNode[];
};

export type CategoricalFieldMetadata = {
  displayName: string;
  optionCount: number;
  options: MetadataOption[];
};

export type NumericFieldMetadata = {
  displayName: string;
  frontendInput: boolean;
  min: number;
  max: number;
  median: number;
  p05: number;
  p95: number;
  missingCount: number;
};

export type PredictionMetadata = {
  locations: LocationsMetadata;
  categorical: Record<PredictionCategoricalField, CategoricalFieldMetadata>;
  numeric: Record<PredictionNumericField, NumericFieldMetadata>;
};

type RawLocationNode = {
  value: string;
  label: string;
  count: number;
};

type RawDistrictNode = RawLocationNode & {
  neighborhoods: RawLocationNode[];
};

type RawCityNode = RawLocationNode & {
  districts: RawDistrictNode[];
};

type RawLocationsMetadata = {
  generated_at: string;
  source_dataset: string;
  total_cities: number;
  total_districts: number;
  total_neighborhoods: number;
  cities: RawCityNode[];
};

type RawMetadataOption = {
  value: RawMetaValue;
  label: string;
  count: number;
};

type RawCategoricalFieldMetadata = {
  display_name: string;
  option_count: number;
  options: RawMetadataOption[];
};

type RawCategoricalMetadata = {
  generated_at: string;
  source_dataset: string;
  fields: Record<PredictionCategoricalField, RawCategoricalFieldMetadata>;
};

type RawNumericFieldMetadata = {
  display_name: string;
  frontend_input: boolean;
  min: number;
  max: number;
  median: number;
  p05: number;
  p95: number;
  missing_count: number;
};

type RawNumericMetadata = {
  generated_at: string;
  source_dataset: string;
  fields: Record<PredictionNumericField, RawNumericFieldMetadata>;
};

export const trCollator = new Intl.Collator("tr-TR", {
  sensitivity: "base",
  numeric: true,
});

export function compareTrLocale(left: string, right: string) {
  return trCollator.compare(left, right);
}

export function sortByLabelTr<T extends { label: string }>(items: T[]) {
  return [...items].sort((left, right) => compareTrLocale(left.label, right.label));
}

export function normalizeMetaValue(value: RawMetaValue) {
  if (value === null || value === undefined) {
    return "";
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  return String(value);
}

function normalizeOption(option: RawMetadataOption): MetadataOption {
  return {
    value: normalizeMetaValue(option.value),
    rawValue: option.value,
    label: option.label,
    count: option.count,
    isEmpty: option.value === null,
  };
}

function normalizeLocationNode(node: RawLocationNode): LocationNode {
  return {
    value: node.value,
    label: node.label,
    count: node.count,
  };
}

function normalizeLocationsMetadata(raw: RawLocationsMetadata): LocationsMetadata {
  return {
    generatedAt: raw.generated_at,
    sourceDataset: raw.source_dataset,
    totalCities: raw.total_cities,
    totalDistricts: raw.total_districts,
    totalNeighborhoods: raw.total_neighborhoods,
    cities: normalizeLocationTree(
      raw.cities.map((city) => ({
        ...normalizeLocationNode(city),
        districts: city.districts.map((district) => ({
          ...normalizeLocationNode(district),
          neighborhoods: district.neighborhoods.map(normalizeLocationNode),
        })),
      })),
    ),
  };
}

function normalizeCategoricalMetadata(raw: RawCategoricalMetadata): PredictionMetadata["categorical"] {
  return Object.fromEntries(
    Object.entries(raw.fields).map(([fieldName, fieldMetadata]) => [
      fieldName,
      {
        displayName: fieldMetadata.display_name,
        optionCount: fieldMetadata.option_count,
        options: normalizeCategoricalOptions(fieldName, fieldMetadata.options.map(normalizeOption)),
      },
    ]),
  ) as PredictionMetadata["categorical"];
}

function normalizeNumericMetadata(raw: RawNumericMetadata): PredictionMetadata["numeric"] {
  return Object.fromEntries(
    Object.entries(raw.fields).map(([fieldName, fieldMetadata]) => [
      fieldName,
      {
        displayName: fieldMetadata.display_name,
        frontendInput: fieldMetadata.frontend_input,
        min: fieldMetadata.min,
        max: fieldMetadata.max,
        median: fieldMetadata.median,
        p05: fieldMetadata.p05,
        p95: fieldMetadata.p95,
        missingCount: fieldMetadata.missing_count,
      },
    ]),
  ) as PredictionMetadata["numeric"];
}

async function fetchMetadataJson<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`${path} yüklenemedi.`);
  }

  return (await response.json()) as T;
}

export async function loadPredictionMetadata(): Promise<PredictionMetadata> {
  const [locations, categorical, numeric] = await Promise.all([
    fetchMetadataJson<RawLocationsMetadata>("/meta/locations.json"),
    fetchMetadataJson<RawCategoricalMetadata>("/meta/categorical-options.json"),
    fetchMetadataJson<RawNumericMetadata>("/meta/numeric-ranges.json"),
  ]);

  return {
    locations: normalizeLocationsMetadata(locations),
    categorical: normalizeCategoricalMetadata(categorical),
    numeric: normalizeNumericMetadata(numeric),
  };
}
