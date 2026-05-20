from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version .* of 'numexpr'.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version .* of 'bottleneck'.*",
    category=UserWarning,
)

import pandas as pd


LOGGER = logging.getLogger("frontend_metadata_export")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "train_ready_multimodal.parquet"
FRONTEND_META_DIR = ROOT_DIR / "frontend" / "public" / "meta"
LOCATIONS_OUTPUT_PATH = FRONTEND_META_DIR / "locations.json"
CATEGORICAL_OUTPUT_PATH = FRONTEND_META_DIR / "categorical-options.json"
NUMERIC_OUTPUT_PATH = FRONTEND_META_DIR / "numeric-ranges.json"
REPORT_OUTPUT_PATH = ROOT_DIR / "reports" / "frontend_metadata_report.md"

LOCATION_COLUMNS = ["city", "district", "neighborhood"]
CATEGORICAL_FIELDS = [
    "rooms",
    "floor",
    "heating_type",
    "fuel_type",
    "home_type",
    "home_shape",
    "is_furnished",
    "bathrooms",
    "building_age",
    "total_floors",
]
NUMERIC_FIELDS = ["m2_gross", "dues_try", "price_try"]

DISPLAY_NAMES = {
    "city": "Şehir",
    "district": "İlçe",
    "neighborhood": "Mahalle",
    "rooms": "Oda tipi",
    "floor": "Bulunduğu kat",
    "heating_type": "Isıtma tipi",
    "fuel_type": "Yakıt tipi",
    "home_type": "Konut tipi",
    "home_shape": "Bulunduğu tip",
    "is_furnished": "Eşyalı mı?",
    "bathrooms": "Banyo sayısı",
    "building_age": "Bina yaşı",
    "total_floors": "Toplam kat",
    "m2_gross": "Brüt m²",
    "dues_try": "Aidat",
    "price_try": "Kira fiyatı",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_output_directories() -> None:
    FRONTEND_META_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    if hasattr(value, "item") and not isinstance(value, str):
        try:
            python_value = value.item()
        except ValueError:
            python_value = value
        if python_value is not value:
            return normalize_scalar(python_value)

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    return " ".join(text.split())


def get_display_label(value: Any, *, field_name: str) -> str:
    if value is None:
        return "Belirtilmemiş"

    if field_name == "is_furnished" and isinstance(value, bool):
        return "Evet" if value else "Hayır"

    return str(value)


def sort_option_items(items: list[tuple[Any, int]], *, field_name: str) -> list[tuple[Any, int]]:
    return sorted(
        items,
        key=lambda item: (
            item[0] is None,
            -item[1],
            get_display_label(item[0], field_name=field_name).casefold(),
        ),
    )


def build_option_records(series: pd.Series, *, field_name: str) -> list[dict[str, Any]]:
    counts: dict[Any, int] = {}
    for raw_value in series.tolist():
        value = normalize_scalar(raw_value)
        counts[value] = counts.get(value, 0) + 1

    records: list[dict[str, Any]] = []
    for value, count in sort_option_items(list(counts.items()), field_name=field_name):
        records.append(
            {
                "value": value,
                "label": get_display_label(value, field_name=field_name),
                "count": int(count),
            }
        )
    return records


def build_location_metadata(dataframe: pd.DataFrame) -> dict[str, Any]:
    location_frame = dataframe.loc[:, LOCATION_COLUMNS].copy()
    for column in LOCATION_COLUMNS:
        location_frame[column] = location_frame[column].map(normalize_scalar)

    city_option_records = build_option_records(location_frame["city"], field_name="city")
    district_option_records = build_option_records(location_frame["district"], field_name="district")
    neighborhood_option_records = build_option_records(location_frame["neighborhood"], field_name="neighborhood")

    district_records_by_city: dict[Any, list[dict[str, Any]]] = {}
    neighborhood_records_by_pair: dict[tuple[Any, Any], list[dict[str, Any]]] = {}

    for city_value, city_group in location_frame.groupby("city", dropna=False, sort=False):
        district_records_by_city[city_value] = build_option_records(city_group["district"], field_name="district")
        for district_value, district_group in city_group.groupby("district", dropna=False, sort=False):
            neighborhood_records_by_pair[(city_value, district_value)] = build_option_records(
                district_group["neighborhood"],
                field_name="neighborhood",
            )

    cities: list[dict[str, Any]] = []
    for city_record in city_option_records:
        city_value = city_record["value"]
        districts: list[dict[str, Any]] = []

        for district_record in district_records_by_city.get(city_value, []):
            district_value = district_record["value"]
            districts.append(
                {
                    **district_record,
                    "neighborhoods": neighborhood_records_by_pair.get((city_value, district_value), []),
                }
            )

        cities.append(
            {
                **city_record,
                "districts": districts,
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(DATASET_PATH),
        "total_cities": len(city_option_records),
        "total_districts": len(district_option_records),
        "total_neighborhoods": len(neighborhood_option_records),
        "cities": cities,
    }


def build_categorical_metadata(dataframe: pd.DataFrame) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field_name in CATEGORICAL_FIELDS:
        options = build_option_records(dataframe[field_name], field_name=field_name)
        fields[field_name] = {
            "display_name": DISPLAY_NAMES[field_name],
            "option_count": len(options),
            "options": options,
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(DATASET_PATH),
        "fields": fields,
    }


def normalize_numeric_stat(value: float) -> int | float | None:
    if pd.isna(value):
        return None
    numeric_value = float(value)
    if numeric_value.is_integer():
        return int(numeric_value)
    return round(numeric_value, 4)


def build_numeric_metadata(dataframe: pd.DataFrame) -> dict[str, Any]:
    fields: dict[str, Any] = {}

    for field_name in NUMERIC_FIELDS:
        numeric_series = pd.to_numeric(dataframe[field_name], errors="coerce")
        fields[field_name] = {
            "display_name": DISPLAY_NAMES[field_name],
            "frontend_input": field_name != "price_try",
            "min": normalize_numeric_stat(numeric_series.min()),
            "max": normalize_numeric_stat(numeric_series.max()),
            "median": normalize_numeric_stat(numeric_series.median()),
            "p05": normalize_numeric_stat(numeric_series.quantile(0.05)),
            "p95": normalize_numeric_stat(numeric_series.quantile(0.95)),
            "missing_count": int(numeric_series.isna().sum()),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(DATASET_PATH),
        "fields": fields,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_Veri yok_"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        escaped = [str(cell).replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def build_report(
    locations_metadata: dict[str, Any],
    categorical_metadata: dict[str, Any],
    numeric_metadata: dict[str, Any],
) -> str:
    first_city = locations_metadata["cities"][0] if locations_metadata["cities"] else None
    top_district_rows: list[list[str]] = []
    if first_city:
        for district_record in first_city["districts"][:10]:
            top_district_rows.append(
                [
                    district_record["label"],
                    f"{district_record['count']:,}",
                    f"{len(district_record['neighborhoods']):,}",
                ]
            )

    categorical_rows = [
        [
            field_payload["display_name"],
            field_name,
            f"{field_payload['option_count']:,}",
        ]
        for field_name, field_payload in categorical_metadata["fields"].items()
    ]

    numeric_rows = [
        [
            field_payload["display_name"],
            field_name,
            str(field_payload["min"]),
            str(field_payload["max"]),
            str(field_payload["median"]),
            str(field_payload["p05"]),
            str(field_payload["p95"]),
            f"{field_payload['missing_count']:,}",
        ]
        for field_name, field_payload in numeric_metadata["fields"].items()
    ]

    return "\n".join(
        [
            "# Frontend Metadata Raporu",
            "",
            "## Özet",
            "",
            f"- Çalışma zamanı: `{datetime.now().isoformat(timespec='seconds')}`",
            f"- Kaynak dataset: `{DATASET_PATH}`",
            f"- Üretilen lokasyon dosyası: `{LOCATIONS_OUTPUT_PATH}`",
            f"- Üretilen kategorik seçenek dosyası: `{CATEGORICAL_OUTPUT_PATH}`",
            f"- Üretilen sayısal aralık dosyası: `{NUMERIC_OUTPUT_PATH}`",
            "",
            "## Lokasyon Kapsamı",
            "",
            f"- Toplam city sayısı: **{locations_metadata['total_cities']:,}**",
            f"- Toplam district sayısı: **{locations_metadata['total_districts']:,}**",
            f"- Toplam neighborhood sayısı: **{locations_metadata['total_neighborhoods']:,}**",
            "",
            "### En çok örneği olan ilçeler",
            "",
            render_markdown_table(
                ["İlçe", "Örnek sayısı", "Neighborhood sayısı"],
                top_district_rows,
            ),
            "",
            "## Kategorik Alan Option Sayıları",
            "",
            render_markdown_table(
                ["Alan", "Field key", "Option sayısı"],
                categorical_rows,
            ),
            "",
            "## Numeric Range Özeti",
            "",
            render_markdown_table(
                ["Alan", "Field key", "Min", "Max", "Median", "P05", "P95", "Missing count"],
                numeric_rows,
            ),
            "",
            "## Frontend Kullanımı",
            "",
            "- `locations.json`, şehir -> ilçe -> mahalle seçimlerini bağımlı select veya autocomplete akışında beslemek için kullanılabilir.",
            "- `categorical-options.json`, serbest yazı yerine dataset temelli dropdown, combobox veya suggestion listeleri üretmek için kullanılabilir.",
            "- `numeric-ranges.json`, sayısal input placeholder, slider sınırı, yardımcı metin ve validation önerileri için referans sağlar.",
            "- `price_try` sadece referans aralık olarak tutulur; form input’u olarak kullanılmamalıdır.",
            "- Null veya boş değerler frontend tarafında `Belirtilmemiş` etiketiyle gösterilebilir.",
        ]
    )


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset bulunamadı: {DATASET_PATH}")

    columns = sorted(set(LOCATION_COLUMNS + CATEGORICAL_FIELDS + NUMERIC_FIELDS))
    LOGGER.info("Dataset okunuyor: %s", DATASET_PATH)
    dataframe = pd.read_parquet(DATASET_PATH, columns=columns)
    LOGGER.info("Okunan satır sayısı: %s", len(dataframe))
    return dataframe


def main() -> int:
    configure_logging()
    ensure_output_directories()

    dataframe = load_dataset()
    locations_metadata = build_location_metadata(dataframe)
    categorical_metadata = build_categorical_metadata(dataframe)
    numeric_metadata = build_numeric_metadata(dataframe)

    write_json(LOCATIONS_OUTPUT_PATH, locations_metadata)
    write_json(CATEGORICAL_OUTPUT_PATH, categorical_metadata)
    write_json(NUMERIC_OUTPUT_PATH, numeric_metadata)

    report_body = build_report(
        locations_metadata=locations_metadata,
        categorical_metadata=categorical_metadata,
        numeric_metadata=numeric_metadata,
    )
    REPORT_OUTPUT_PATH.write_text(report_body, encoding="utf-8")

    LOGGER.info("Frontend metadata export tamamlandı.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
