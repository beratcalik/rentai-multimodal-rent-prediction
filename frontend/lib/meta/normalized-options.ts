type RawMetaValue = string | number | boolean | null;

type MetadataOptionLike = {
  value: string;
  rawValue: RawMetaValue;
  label: string;
  count: number;
  isEmpty: boolean;
};

type LocationNodeLike = {
  value: string;
  label: string;
  count: number;
};

type DistrictNodeLike = LocationNodeLike & {
  neighborhoods: LocationNodeLike[];
};

type CityNodeLike = LocationNodeLike & {
  districts: DistrictNodeLike[];
};

const EMPTY_LABEL = "Belirtilmemiş";

const trCollator = new Intl.Collator("tr-TR", {
  sensitivity: "base",
  numeric: true,
});

const ROOM_DEFINITIONS = [
  { value: "1+0", label: "1+0" },
  { value: "1+1", label: "1+1" },
  { value: "2+0", label: "2+0" },
  { value: "2+1", label: "2+1" },
  { value: "3+1", label: "3+1" },
  { value: "3+2", label: "3+2" },
  { value: "4+1", label: "4+1" },
  { value: "4+2", label: "4+2" },
  { value: "5+1", label: "5+1" },
  { value: "5+2", label: "5+2" },
  { value: "6+1", label: "6+1" },
  { value: "6+2", label: "6+2" },
  { value: "7+1", label: "7+1" },
  { value: "7+2", label: "7+2" },
  { value: "8+1", label: "8+1" },
  { value: "8+2", label: "8+2" },
  { value: "9+1", label: "9+ ve üzeri" },
] as const;

const BATHROOM_DEFINITIONS = [
  { value: "1", label: "1" },
  { value: "2", label: "2" },
  { value: "3", label: "3" },
  { value: "4", label: "4" },
  { value: "5+", label: "5+" },
] as const;

const TOTAL_FLOOR_DEFINITIONS = [
  ...Array.from({ length: 50 }, (_, index) => ({
    value: String(index + 1),
    label: String(index + 1),
  })),
  { value: "50+", label: "50+" },
] as const;

const FLOOR_DEFINITIONS = [
  { value: "Bahçe Katı", label: "Bahçe Katı" },
  { value: "Zemin Kat", label: "Zemin Kat" },
  { value: "Giriş Katı", label: "Giriş Katı" },
  { value: "Yüksek Giriş", label: "Yüksek Giriş" },
  ...Array.from({ length: 50 }, (_, index) => ({
    value: `${index + 1}. Kat`,
    label: `${index + 1}. Kat`,
  })),
  { value: "Çatı Katı", label: "Çatı Katı" },
  { value: "Kot 1", label: "Kot 1" },
  { value: "Kot 2", label: "Kot 2" },
  { value: "Kot 3", label: "Kot 3" },
  { value: "Bodrum Kat", label: "Bodrum Kat" },
  { value: "Teras Kat", label: "Teras Kat" },
] as const;

const FURNISHED_DEFINITIONS = [
  { value: "false", label: "Hayır", rawValue: false },
  { value: "true", label: "Evet", rawValue: true },
  { value: "", label: EMPTY_LABEL, rawValue: null },
] as const;

function sanitizeLabel(label: string | null | undefined) {
  const trimmed = (label ?? "").trim();
  return trimmed.length > 0 ? trimmed : EMPTY_LABEL;
}

function isEmptyLabel(label: string) {
  return label === EMPTY_LABEL;
}

function compareOptionLabels(left: { label: string }, right: { label: string }) {
  const leftIsEmpty = isEmptyLabel(left.label);
  const rightIsEmpty = isEmptyLabel(right.label);

  if (leftIsEmpty && !rightIsEmpty) {
    return 1;
  }

  if (!leftIsEmpty && rightIsEmpty) {
    return -1;
  }

  return trCollator.compare(left.label, right.label);
}

function compareLocationLabels(left: { label: string }, right: { label: string }) {
  const leftIsEmpty = isEmptyLabel(left.label);
  const rightIsEmpty = isEmptyLabel(right.label);

  if (leftIsEmpty && !rightIsEmpty) {
    return 1;
  }

  if (!leftIsEmpty && rightIsEmpty) {
    return -1;
  }

  return trCollator.compare(left.label, right.label);
}

function toSourceMap(options: MetadataOptionLike[]) {
  const map = new Map<string, MetadataOptionLike>();

  for (const option of options) {
    const normalizedLabel = sanitizeLabel(option.label);
    const normalizedValue = option.value.trim();

    map.set(normalizedValue, {
      ...option,
      value: normalizedValue,
      label: normalizedLabel,
      isEmpty: option.isEmpty || normalizedValue.length === 0,
    });
  }

  return map;
}

function buildCustomOptions(
  definitions: ReadonlyArray<{
    value: string;
    label: string;
    rawValue?: RawMetaValue;
  }>,
  sourceOptions: MetadataOptionLike[],
) {
  const sourceMap = toSourceMap(sourceOptions);

  return definitions.map((definition) => {
    const source = sourceMap.get(definition.value);

    return {
      value: definition.value,
      rawValue: definition.rawValue ?? definition.value,
      label: definition.label,
      count: source?.count ?? 0,
      isEmpty: definition.value.length === 0,
    };
  });
}

function sanitizeAndSortOptions(options: MetadataOptionLike[]) {
  return options
    .map((option) => ({
      ...option,
      value: option.value.trim(),
      label: sanitizeLabel(option.label),
      isEmpty: option.isEmpty || option.value.trim().length === 0,
    }))
    .sort(compareOptionLabels);
}

export function normalizeLocationTree(cities: CityNodeLike[]) {
  return cities
    .map((city) => ({
      ...city,
      label: sanitizeLabel(city.label),
      districts: [...city.districts]
        .map((district) => ({
          ...district,
          label: sanitizeLabel(district.label),
          neighborhoods: [...district.neighborhoods]
            .map((neighborhood) => ({
              ...neighborhood,
              label: sanitizeLabel(neighborhood.label),
            }))
            .sort(compareLocationLabels),
        }))
        .sort(compareLocationLabels),
    }))
    .sort(compareLocationLabels);
}

export function normalizeCategoricalOptions(fieldName: string, options: MetadataOptionLike[]) {
  if (fieldName === "rooms") {
    return buildCustomOptions(ROOM_DEFINITIONS, options);
  }

  if (fieldName === "bathrooms") {
    return buildCustomOptions(BATHROOM_DEFINITIONS, options);
  }

  if (fieldName === "total_floors") {
    return buildCustomOptions(TOTAL_FLOOR_DEFINITIONS, options);
  }

  if (fieldName === "floor") {
    return buildCustomOptions(FLOOR_DEFINITIONS, options);
  }

  if (fieldName === "is_furnished") {
    return buildCustomOptions(FURNISHED_DEFINITIONS, options);
  }

  return sanitizeAndSortOptions(options);
}
