from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
PRICE_PRIORITY = [
    "price_try",
    "price",
    "rent_price",
    "monthly_rent",
    "fiyat",
    "target",
    "label",
]
TEXT_NAME_HINTS = ("title", "description", "desc", "text", "content", "summary", "caption")
TEXT_NAME_EXCLUDE = {
    "listing_id",
    "listing_no",
    "image_id",
    "run_id",
    "url",
    "source_url",
    "local_path",
    "valid_image_paths",
    "scraped_at",
    "updated_at",
    "validated_at",
    "ts",
}
HEAD_ROWS = 5
MAX_CELL_LEN = 120


def detect_parquet_engines() -> dict[str, bool]:
    return {
        "pyarrow": importlib.util.find_spec("pyarrow") is not None,
        "fastparquet": importlib.util.find_spec("fastparquet") is not None,
    }


def normalize_text(value: Any) -> str:
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(value: Any, max_len: int = MAX_CELL_LEN) -> str:
    text = normalize_text(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    return False


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def serialize_value(value: Any, max_len: int = MAX_CELL_LEN) -> str:
    if is_missing(value):
        return "NA"
    if isinstance(value, dict):
        return truncate_text(safe_json_dumps(value), max_len=max_len)
    if isinstance(value, (list, tuple, set)):
        return truncate_text(safe_json_dumps(list(value)), max_len=max_len)
    return truncate_text(value, max_len=max_len)


def normalize_path_key(value: Any) -> str:
    text = normalize_text(value).strip("\"'")
    text = text.replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    text = re.sub(r"^\./", "", text)
    return text.casefold()


def build_existing_file_index(project_root: Path) -> tuple[set[str], int]:
    existing_keys: set[str] = set()
    image_file_count = 0
    for file_path in project_root.rglob("*"):
        if not file_path.is_file():
            continue
        existing_keys.add(normalize_path_key(file_path.resolve()))
        try:
            rel_to_root = file_path.relative_to(project_root)
            existing_keys.add(normalize_path_key(rel_to_root))
            existing_keys.add(normalize_path_key(Path(project_root.name) / rel_to_root))
        except ValueError:
            pass
        if file_path.suffix.lower() in IMAGE_EXTENSIONS and "images" in {part.lower() for part in file_path.parts}:
            image_file_count += 1
    return existing_keys, image_file_count


def extract_path_items(value: Any) -> list[str]:
    if is_missing(value):
        return []

    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(extract_path_items(item))
        return items

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        if text[0] in "[(" and text[-1] in "])":
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    return extract_path_items(parsed)
                except Exception:
                    continue
        return [text]

    return [str(value)]


def path_exists(raw_path: str, existing_keys: set[str], project_root: Path) -> bool:
    if not raw_path:
        return False

    raw_key = normalize_path_key(raw_path)
    if raw_key in existing_keys:
        return True

    if raw_key.startswith("dataset/"):
        if raw_key[len("dataset/") :] in existing_keys:
            return True
    else:
        prefixed = normalize_path_key(Path(project_root.name) / raw_path)
        if prefixed in existing_keys:
            return True

    path_obj = Path(str(raw_path))
    if path_obj.is_absolute() and path_obj.exists():
        return True

    candidate = (project_root / path_obj).resolve()
    if normalize_path_key(candidate) in existing_keys:
        return True

    return False


def detect_listing_like_columns(columns: list[str]) -> list[str]:
    matches: list[str] = []
    for column in columns:
        lower = column.lower()
        if lower in {"listing_id", "listing_no"}:
            matches.append(column)
            continue
        if "ilan" in lower:
            matches.append(column)
            continue
        if "listing" in lower and ("id" in lower or "no" in lower):
            matches.append(column)
    return matches


def pick_price_target(columns: list[str]) -> dict[str, Any]:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in PRICE_PRIORITY:
        if candidate in lower_to_original:
            return {"column": lower_to_original[candidate], "derived": False}

    for column in columns:
        lower = column.lower()
        if "price_per_m2" in lower:
            return {"column": column, "derived": True}
    for column in columns:
        lower = column.lower()
        if "price" in lower or "fiyat" in lower:
            derived = any(token in lower for token in ("outlier", "per_m2", "ratio"))
            return {"column": column, "derived": derived}
    return {"column": None, "derived": False}


def detect_text_columns(df: pd.DataFrame) -> list[str]:
    detected: list[str] = []
    for column in df.columns:
        lower = column.lower()
        if lower in TEXT_NAME_EXCLUDE:
            continue
        if not (
            pd.api.types.is_object_dtype(df[column].dtype)
            or pd.api.types.is_string_dtype(df[column].dtype)
        ):
            continue

        if any(hint in lower for hint in TEXT_NAME_HINTS):
            detected.append(column)
            continue

        sample_values = [normalize_text(value) for value in df[column].dropna().head(25)]
        if not sample_values:
            continue
        average_length = sum(len(value) for value in sample_values) / len(sample_values)
        if average_length >= 80:
            detected.append(column)
    return detected


def detect_visual_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    visual_columns = {"local_path": [], "url": []}
    for column in df.columns:
        lower = column.lower()
        if lower.endswith("_count") or lower.endswith("_index") or lower in {"image_id"}:
            continue

        sample_items: list[str] = []
        for value in df[column].dropna().head(10):
            sample_items.extend(extract_path_items(value))
            if len(sample_items) >= 5:
                break

        if not sample_items:
            continue

        sample_items = sample_items[:5]
        looks_url = any(item.lower().startswith(("http://", "https://")) for item in sample_items)
        only_urls = all(item.lower().startswith(("http://", "https://")) for item in sample_items)
        looks_local = any(
            normalize_path_key(item).startswith(("images/", "dataset/images/"))
            or Path(item).suffix.lower() in IMAGE_EXTENSIONS
            for item in sample_items
        )
        name_matches = any(token in lower for token in ("image", "photo", "visual", "gallery", "thumbnail"))
        special_name = lower in {"local_path", "source_url", "valid_image_paths"}

        if (name_matches or special_name) and looks_local and not only_urls:
            visual_columns["local_path"].append(column)
        if (name_matches or special_name) and looks_url:
            visual_columns["url"].append(column)
    return visual_columns


def safe_duplicate_count(df: pd.DataFrame) -> tuple[int, str | None]:
    try:
        return int(df.duplicated().sum()), None
    except Exception as exc:
        normalized = df.copy()
        for column in normalized.columns:
            normalized[column] = normalized[column].map(
                lambda value: safe_json_dumps(value)
                if isinstance(value, (list, tuple, set, dict))
                else value
            )
        return int(normalized.duplicated().sum()), f"Fallback duplicate check used: {exc}"


def render_preview(df: pd.DataFrame) -> str:
    preview = df.head(HEAD_ROWS).copy()
    for column in preview.columns:
        preview[column] = preview[column].map(serialize_value)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        MAX_CELL_LEN,
    ):
        return preview.to_string(index=False)


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No data_"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        escaped = [str(cell).replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def format_pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0.00%"
    return f"{(part / whole) * 100:.2f}%"


def analyze_local_path_column(
    df: pd.DataFrame,
    column: str,
    existing_keys: set[str],
    project_root: Path,
) -> dict[str, Any]:
    listing_column = "listing_id" if "listing_id" in df.columns else None
    total_paths = 0
    valid_paths = 0
    broken_paths = 0
    sample_broken: list[str] = []
    valid_paths_per_listing: defaultdict[str, int] = defaultdict(int)

    for row_index, value in df[column].items():
        extracted_paths = extract_path_items(value)
        if not extracted_paths:
            continue
        listing_value = None
        if listing_column is not None:
            listing_value = df.at[row_index, listing_column]
            if is_missing(listing_value):
                listing_value = None
            elif listing_value is not None:
                listing_value = str(listing_value)

        for raw_path in extracted_paths:
            total_paths += 1
            exists = path_exists(raw_path, existing_keys, project_root)
            if exists:
                valid_paths += 1
                if listing_value is not None:
                    valid_paths_per_listing[listing_value] += 1
            else:
                broken_paths += 1
                if len(sample_broken) < 5:
                    sample_broken.append(truncate_text(raw_path, max_len=140))

    average_per_listing = None
    if valid_paths_per_listing:
        average_per_listing = sum(valid_paths_per_listing.values()) / len(valid_paths_per_listing)

    return {
        "column": column,
        "total_paths": total_paths,
        "valid_paths": valid_paths,
        "broken_paths": broken_paths,
        "average_valid_images_per_listing": average_per_listing,
        "listings_with_valid_images": len(valid_paths_per_listing),
        "sample_broken_paths": sample_broken,
    }


def infer_file_role(summary: dict[str, Any]) -> str:
    name = summary["relative_path"].lower()
    columns = set(summary["columns"])
    price_column = summary["price_target_candidate"]["column"]

    if "run_log" in name or {"run_id", "event", "error"} & columns == {"run_id", "event", "error"}:
        return "Operational run log"
    if "validation" in name or "report" in name or "is_train_ready_ml" in columns:
        return "Validation/report table"
    if "multimodal" in name or "valid_image_paths" in columns:
        return "Train-ready multimodal table"
    if "train_ready" in name and price_column:
        return "Train-ready tabular/text table"
    if "image" in name and "local_path" in columns:
        return "Image metadata table"
    if "listing" in name and price_column:
        return "Raw main listing table"
    if price_column:
        return "Price-bearing auxiliary table"
    return "Auxiliary table"


def infer_trainability(summary: dict[str, Any]) -> str:
    role = summary["role"]
    price_column = summary["price_target_candidate"]["column"]
    text_columns = summary["text_columns"]
    local_image_columns = summary["visual_columns"]["local_path"]

    if role == "Operational run log":
        return "Hayir. Bu dosya toplama/boru hatti logu; model egitimi icin ana veri degil."
    if role == "Validation/report table":
        return "Kismen. Kalite filtresi ve train-ready kararlarini destekler, ama ana egitim tablosu degil."
    if role == "Image metadata table":
        return "Kismen. Image branch icin yardimci tablo olarak kullanilabilir, ancak fiyat hedefi yok."
    if price_column and local_image_columns and text_columns:
        return "Evet. Multimodal egitim icin dogrudan aday."
    if price_column and text_columns:
        return "Evet. Tabular + text egitimi icin uygun."
    if price_column:
        return "Evet. Tabular egitim icin uygun, ancak ek onisleme gerekir."
    return "Kismen/Hayir. Hedef degisken veya ana ozellikler eksik."


def summarize_dataframe(
    parquet_path: Path,
    project_root: Path,
    df: pd.DataFrame,
    existing_keys: set[str],
) -> dict[str, Any]:
    columns = list(df.columns)
    listing_like_columns = detect_listing_like_columns(columns)
    price_target_candidate = pick_price_target(columns)
    text_columns = detect_text_columns(df)
    visual_columns = detect_visual_columns(df)
    missing_counts = {column: int(value) for column, value in df.isna().sum().items()}
    duplicate_count, duplicate_note = safe_duplicate_count(df)
    local_path_audits = [
        analyze_local_path_column(df, column, existing_keys, project_root)
        for column in visual_columns["local_path"]
    ]

    summary = {
        "path": str(parquet_path.resolve()),
        "relative_path": str(parquet_path.relative_to(project_root)),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": columns,
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_counts": missing_counts,
        "preview": render_preview(df),
        "duplicate_count": duplicate_count,
        "duplicate_note": duplicate_note,
        "listing_like_columns": listing_like_columns,
        "price_target_candidate": price_target_candidate,
        "visual_columns": visual_columns,
        "text_columns": text_columns,
        "unique_listing_count": None,
        "local_path_audits": local_path_audits,
        "read_error": None,
    }

    if "listing_id" in df.columns:
        summary["unique_listing_count"] = int(df["listing_id"].dropna().astype(str).nunique())

    summary["role"] = infer_file_role(summary)
    summary["trainability"] = infer_trainability(summary)
    return summary


def summarize_read_error(parquet_path: Path, project_root: Path, error: Exception) -> dict[str, Any]:
    return {
        "path": str(parquet_path.resolve()),
        "relative_path": str(parquet_path.relative_to(project_root)),
        "row_count": None,
        "column_count": None,
        "columns": [],
        "dtypes": {},
        "missing_counts": {},
        "preview": "",
        "duplicate_count": None,
        "duplicate_note": None,
        "listing_like_columns": [],
        "price_target_candidate": {"column": None, "derived": False},
        "visual_columns": {"local_path": [], "url": []},
        "text_columns": [],
        "unique_listing_count": None,
        "local_path_audits": [],
        "role": "Unreadable parquet",
        "trainability": "Hayir. Dosya okunamadi.",
        "read_error": repr(error),
    }


def analyze_relationships(dataframes: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []

    for left_name, right_name in combinations(sorted(dataframes), 2):
        left_df = dataframes[left_name]
        right_df = dataframes[right_name]
        common_columns = sorted(set(left_df.columns) & set(right_df.columns))

        relationship = {
            "left": left_name,
            "right": right_name,
            "common_columns": common_columns,
            "common_column_count": len(common_columns),
            "join_on_listing_id": False,
            "shared_listing_count": 0,
            "left_unique_listing_count": 0,
            "right_unique_listing_count": 0,
            "left_coverage": "0.00%",
            "right_coverage": "0.00%",
            "join_shape": None,
        }

        if "listing_id" in left_df.columns and "listing_id" in right_df.columns:
            left_ids = set(left_df["listing_id"].dropna().astype(str))
            right_ids = set(right_df["listing_id"].dropna().astype(str))
            shared_ids = left_ids & right_ids
            relationship["join_on_listing_id"] = True
            relationship["shared_listing_count"] = len(shared_ids)
            relationship["left_unique_listing_count"] = len(left_ids)
            relationship["right_unique_listing_count"] = len(right_ids)
            relationship["left_coverage"] = format_pct(len(shared_ids), len(left_ids))
            relationship["right_coverage"] = format_pct(len(shared_ids), len(right_ids))

            left_rows_per_id = len(left_df) / len(left_ids) if left_ids else 0.0
            right_rows_per_id = len(right_df) / len(right_ids) if right_ids else 0.0
            if left_rows_per_id <= 1.05 and right_rows_per_id <= 1.05:
                relationship["join_shape"] = "mostly one-to-one"
            elif left_rows_per_id > 1.05 and right_rows_per_id <= 1.05:
                relationship["join_shape"] = f"many-to-one from {left_name} to {right_name}"
            elif left_rows_per_id <= 1.05 and right_rows_per_id > 1.05:
                relationship["join_shape"] = f"many-to-one from {right_name} to {left_name}"
            else:
                relationship["join_shape"] = "many-to-many / event-style"

        relationships.append(relationship)

    return relationships


def choose_primary_files(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    readable = [summary for summary in summaries if summary["read_error"] is None]

    def by_relative_path(relative_path: str) -> dict[str, Any] | None:
        for summary in readable:
            if summary["relative_path"] == relative_path:
                return summary
        return None

    raw_main = by_relative_path("listings.parquet")
    if raw_main is None:
        raw_candidates = [summary for summary in readable if summary["role"] == "Raw main listing table"]
        raw_main = max(raw_candidates, key=lambda item: item["row_count"], default=None)

    tabular_train = by_relative_path("train_ready_ml.parquet")
    if tabular_train is None:
        train_candidates = [
            summary for summary in readable if summary["role"] == "Train-ready tabular/text table"
        ]
        tabular_train = max(train_candidates, key=lambda item: item["row_count"], default=None)

    multimodal_train = by_relative_path("train_ready_multimodal.parquet")
    if multimodal_train is None:
        multi_candidates = [
            summary for summary in readable if summary["role"] == "Train-ready multimodal table"
        ]
        multimodal_train = max(multi_candidates, key=lambda item: item["row_count"], default=None)

    image_table = by_relative_path("images.parquet")
    validation_table = by_relative_path("validation_report.parquet")
    run_log = by_relative_path("run_log.parquet")

    return {
        "raw_main": raw_main,
        "tabular_train": tabular_train,
        "multimodal_train": multimodal_train,
        "image_table": image_table,
        "validation_table": validation_table,
        "run_log": run_log,
    }


def estimate_total_listings(dataframes: dict[str, pd.DataFrame]) -> int:
    all_ids: set[str] = set()
    for df in dataframes.values():
        if "listing_id" not in df.columns:
            continue
        all_ids.update(df["listing_id"].dropna().astype(str).tolist())
    return len(all_ids)


def collect_consistency_notes(
    dataframes: dict[str, pd.DataFrame],
    primary_files: dict[str, Any],
) -> list[str]:
    notes: list[str] = []
    raw_main = primary_files["raw_main"]
    raw_listing_ids: set[str] = set()

    if raw_main and raw_main["relative_path"] in dataframes:
        raw_listing_ids = set(
            dataframes[raw_main["relative_path"]]["listing_id"].dropna().astype(str)
        )

    if "run_log.parquet" in dataframes and raw_listing_ids:
        run_log_ids = set(dataframes["run_log.parquet"]["listing_id"].dropna().astype(str))
        extra_run_log_ids = run_log_ids - raw_listing_ids
        if extra_run_log_ids:
            notes.append(
                f"`run_log.parquet` contains {len(extra_run_log_ids)} listing_id values that do not appear in the raw main listing table."
            )

    if "validation_report.parquet" in dataframes:
        validation_df = dataframes["validation_report.parquet"]
        if "train_ready_ml.parquet" in dataframes and "is_train_ready_ml" in validation_df.columns:
            validation_ml_ids = set(
                validation_df.loc[validation_df["is_train_ready_ml"], "listing_id"].dropna().astype(str)
            )
            train_ml_ids = set(dataframes["train_ready_ml.parquet"]["listing_id"].dropna().astype(str))
            diff = validation_ml_ids - train_ml_ids
            if diff:
                notes.append(
                    f"`validation_report.parquet` marks {len(validation_ml_ids):,} listings as `is_train_ready_ml=True`, "
                    f"but `train_ready_ml.parquet` contains {len(train_ml_ids):,}; {len(diff)} flagged IDs are missing from the train-ready file."
                )

        if "train_ready_multimodal.parquet" in dataframes and "is_train_ready_multimodal" in validation_df.columns:
            validation_mm_ids = set(
                validation_df.loc[
                    validation_df["is_train_ready_multimodal"], "listing_id"
                ].dropna().astype(str)
            )
            train_mm_ids = set(
                dataframes["train_ready_multimodal.parquet"]["listing_id"].dropna().astype(str)
            )
            diff = validation_mm_ids - train_mm_ids
            if diff:
                notes.append(
                    f"`validation_report.parquet` marks {len(validation_mm_ids):,} listings as `is_train_ready_multimodal=True`, "
                    f"but `train_ready_multimodal.parquet` contains {len(train_mm_ids):,}; {len(diff)} flagged IDs are missing from the train-ready multimodal file."
                )

    return notes


def recommend_baseline_columns(summary: dict[str, Any] | None) -> list[str]:
    if summary is None:
        return []
    preferred = [
        "city",
        "district",
        "neighborhood",
        "rooms",
        "bathrooms",
        "m2_gross",
        "m2_net",
        "building_age",
        "floor",
        "total_floors",
        "heating_type",
        "fuel_type",
        "is_furnished",
        "dues_try",
        "home_type",
        "home_shape",
        "image_count",
    ]
    return [
        column
        for column in preferred
        if column in summary["columns"]
        and summary["missing_counts"].get(column, summary["row_count"]) < summary["row_count"]
    ]


def recommend_multimodal_columns(summary: dict[str, Any] | None) -> list[str]:
    if summary is None:
        return []
    preferred = recommend_baseline_columns(summary) + [
        "title",
        "description",
        "valid_image_paths",
        "valid_image_count",
    ]
    return [column for column in preferred if column in summary["columns"]]


def evaluate_dataset_readiness(primary_files: dict[str, Any]) -> dict[str, str]:
    tabular = primary_files["tabular_train"]
    multimodal = primary_files["multimodal_train"]
    image_table = primary_files["image_table"]

    tabular_ready = (
        "Evet. `train_ready_ml.parquet` fiyat hedefi ve temel yapisal kolonlarla hazir."
        if tabular is not None
        else "Hayir. Net bir train-ready tabular dosya bulunamadi."
    )

    text_ready = (
        "Evet. `title` ve `description` kolonlari train-ready dosyalarda mevcut."
        if tabular is not None and {"title", "description"} <= set(tabular["columns"])
        else "Kismen. Metin kolonlari eksik veya daginik."
    )

    image_note = "Hayir. Dogrudan image branch dosyasi bulunamadi."
    if image_table is not None and image_table["local_path_audits"]:
        audit = max(image_table["local_path_audits"], key=lambda item: item["valid_paths"])
        if audit["valid_paths"] > 0:
            image_note = (
                "Evet. `images.parquet` uzerinden yerel goruntuler bulunuyor; ancak bozuk pathler de var."
            )
        else:
            image_note = "Kismen. Image metadata var ama dogrulanan yerel goruntu bulunamadi."

    multimodal_note = "Hayir. Train-ready multimodal dosya bulunamadi."
    if multimodal is not None and multimodal["local_path_audits"]:
        audit = max(multimodal["local_path_audits"], key=lambda item: item["valid_paths"])
        if audit["valid_paths"] > 0:
            multimodal_note = (
                "Evet. `train_ready_multimodal.parquet` fiyat + metin + goruntu pathlerini bir araya getiriyor."
            )
        else:
            multimodal_note = (
                "Kismen. Multimodal tablo var, fakat goruntu pathlerinin dosya sistemindeki karsiliklari sorunlu."
            )

    return {
        "tabular": tabular_ready,
        "text": text_ready,
        "image": image_note,
        "multimodal": multimodal_note,
    }


def build_report(
    project_root: Path,
    report_path: Path,
    parquet_paths: list[Path],
    summaries: list[dict[str, Any]],
    dataframes: dict[str, pd.DataFrame],
    relationships: list[dict[str, Any]],
    primary_files: dict[str, Any],
    readiness: dict[str, str],
    engine_status: dict[str, bool],
    image_file_count: int,
) -> str:
    union_listing_count = estimate_total_listings(dataframes)
    baseline_columns = recommend_baseline_columns(primary_files["tabular_train"])
    multimodal_columns = recommend_multimodal_columns(primary_files["multimodal_train"])
    consistency_notes = collect_consistency_notes(dataframes, primary_files)
    raw_main_listing_count = (
        primary_files["raw_main"]["unique_listing_count"] if primary_files["raw_main"] else None
    )
    target_column = None
    for candidate in (
        primary_files["multimodal_train"],
        primary_files["tabular_train"],
        primary_files["raw_main"],
    ):
        if candidate and candidate["price_target_candidate"]["column"]:
            target_column = candidate["price_target_candidate"]["column"]
            break

    lines: list[str] = []
    lines.append("# Dataset Summary")
    lines.append("")
    lines.append(f"- Project root: `{project_root}`")
    lines.append(f"- Report path: `{report_path}`")
    lines.append(f"- Total parquet files found: **{len(parquet_paths)}**")
    if raw_main_listing_count is not None:
        lines.append(f"- Main raw listing count: **{raw_main_listing_count:,}**")
    lines.append(f"- Union of listing_id values across all parquet files: **{union_listing_count:,}**")
    lines.append(f"- Image files physically present under `images/`: **{image_file_count:,}**")
    lines.append(
        f"- Parquet engines: `pyarrow={'yes' if engine_status['pyarrow'] else 'no'}`, "
        f"`fastparquet={'yes' if engine_status['fastparquet'] else 'no'}`"
    )
    lines.append("")

    lines.append("## Project Structure")
    lines.append("")
    lines.append("- Top-level parquet files are at the dataset root.")
    lines.append("- The `images/` directory contains the local image assets referenced by image-related parquet files.")
    lines.append("- No existing Python analysis script or requirements file was found in this workspace before this run.")
    lines.append("")

    lines.append("## Quick Summary")
    lines.append("")
    raw_main = primary_files["raw_main"]
    tabular_train = primary_files["tabular_train"]
    multimodal_train = primary_files["multimodal_train"]
    image_table = primary_files["image_table"]
    validation_table = primary_files["validation_table"]
    run_log = primary_files["run_log"]

    lines.append(
        f"- Raw main listing table: `{raw_main['relative_path']}`" if raw_main else "- Raw main listing table: not identified"
    )
    lines.append(
        f"- Recommended tabular training file: `{tabular_train['relative_path']}`"
        if tabular_train
        else "- Recommended tabular training file: not identified"
    )
    lines.append(
        f"- Recommended multimodal training file: `{multimodal_train['relative_path']}`"
        if multimodal_train
        else "- Recommended multimodal training file: not identified"
    )
    lines.append(
        f"- Image metadata table: `{image_table['relative_path']}`" if image_table else "- Image metadata table: not identified"
    )
    lines.append(
        f"- Validation/report table: `{validation_table['relative_path']}`"
        if validation_table
        else "- Validation/report table: not identified"
    )
    lines.append(f"- Process log table: `{run_log['relative_path']}`" if run_log else "- Process log table: not identified")
    lines.append(f"- Suggested target column: `{target_column}`" if target_column else "- Suggested target column: not identified")
    lines.append("")

    lines.append("## Dependency Note")
    lines.append("")
    if engine_status["pyarrow"] and engine_status["fastparquet"]:
        lines.append("- Both parquet backends are available.")
    elif engine_status["pyarrow"] or engine_status["fastparquet"]:
        lines.append("- Current run succeeded because at least one parquet backend is available.")
        lines.append(
            "- Requirements suggestion: add `pandas`, `pyarrow`, and optionally `fastparquet` if you want both engines reproducibly."
        )
    else:
        lines.append("- No parquet backend was detected. Reading parquet files will fail until a backend is installed.")
        lines.append("- Requirements suggestion: add `pandas` and `pyarrow` or `fastparquet`.")
    lines.append("")

    lines.append("## Per-File Analysis")
    lines.append("")
    for summary in summaries:
        lines.append(f"### `{summary['relative_path']}`")
        lines.append("")
        lines.append(f"- Absolute path: `{summary['path']}`")
        lines.append(f"- Role guess: {summary['role']}")
        if summary["read_error"] is not None:
            lines.append(f"- Read status: ERROR - `{summary['read_error']}`")
            lines.append("")
            continue

        lines.append(f"- Shape: **{summary['row_count']:,} rows x {summary['column_count']} columns**")
        lines.append(
            f"- Duplicate rows: {'yes' if summary['duplicate_count'] else 'no'}"
            + (f" ({summary['duplicate_count']:,})" if summary["duplicate_count"] else "")
        )
        if summary["duplicate_note"]:
            lines.append(f"- Duplicate note: {summary['duplicate_note']}")
        lines.append(
            f"- Listing-like columns: {', '.join(f'`{column}`' for column in summary['listing_like_columns'])}"
            if summary["listing_like_columns"]
            else "- Listing-like columns: none"
        )
        if summary["unique_listing_count"] is not None:
            lines.append(f"- Unique `listing_id` count: **{summary['unique_listing_count']:,}**")
        price_candidate = summary["price_target_candidate"]["column"]
        if price_candidate:
            suffix = " (derived metric)" if summary["price_target_candidate"]["derived"] else ""
            lines.append(f"- Possible price target: `{price_candidate}`{suffix}")
        else:
            lines.append("- Possible price target: none")
        local_visuals = summary["visual_columns"]["local_path"]
        url_visuals = summary["visual_columns"]["url"]
        lines.append(
            f"- Visual local path columns: {', '.join(f'`{column}`' for column in local_visuals)}"
            if local_visuals
            else "- Visual local path columns: none"
        )
        lines.append(
            f"- Visual URL columns: {', '.join(f'`{column}`' for column in url_visuals)}"
            if url_visuals
            else "- Visual URL columns: none"
        )
        lines.append(
            f"- Text columns: {', '.join(f'`{column}`' for column in summary['text_columns'])}"
            if summary["text_columns"]
            else "- Text columns: none"
        )
        lines.append(f"- Trainability: {summary['trainability']}")
        lines.append("")

        lines.append("#### Columns")
        lines.append("")
        lines.append(", ".join(f"`{column}`" for column in summary["columns"]) if summary["columns"] else "_No columns_")
        lines.append("")

        lines.append("#### Dtypes")
        lines.append("")
        dtype_rows = [[column, dtype] for column, dtype in summary["dtypes"].items()]
        lines.append(render_markdown_table(["Column", "Dtype"], dtype_rows))
        lines.append("")

        lines.append("#### Missing Values")
        lines.append("")
        missing_rows = [
            [column, f"{count:,}"]
            for column, count in sorted(
                summary["missing_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        lines.append(render_markdown_table(["Column", "Missing"], missing_rows))
        lines.append("")

        lines.append("#### First 5 Rows")
        lines.append("")
        lines.append("```text")
        lines.append(summary["preview"])
        lines.append("```")
        lines.append("")

        if summary["local_path_audits"]:
            lines.append("#### Image Path Audit")
            lines.append("")
            for audit in summary["local_path_audits"]:
                lines.append(f"- Column `{audit['column']}` total paths: **{audit['total_paths']:,}**")
                lines.append(f"- Existing local files: **{audit['valid_paths']:,}**")
                lines.append(f"- Broken local files: **{audit['broken_paths']:,}**")
                if audit["average_valid_images_per_listing"] is not None:
                    lines.append(
                        "- Average valid images per listing: "
                        f"**{audit['average_valid_images_per_listing']:.2f}**"
                    )
                    lines.append(
                        f"- Listings with at least one valid image: **{audit['listings_with_valid_images']:,}**"
                    )
                if audit["sample_broken_paths"]:
                    lines.append(
                        "- Sample broken paths: "
                        + ", ".join(f"`{path}`" for path in audit["sample_broken_paths"])
                    )
            lines.append("")

    lines.append("## Cross-File Relationships")
    lines.append("")
    for relationship in relationships:
        common_preview = ", ".join(f"`{column}`" for column in relationship["common_columns"][:12])
        if relationship["common_column_count"] > 12:
            common_preview += ", ..."
        if not common_preview:
            common_preview = "none"
        lines.append(f"### `{relationship['left']}` <-> `{relationship['right']}`")
        lines.append("")
        lines.append(f"- Common columns ({relationship['common_column_count']}): {common_preview}")
        if relationship["join_on_listing_id"]:
            lines.append("- Joinable on `listing_id`: yes")
            lines.append(f"- Shared listing_id count: **{relationship['shared_listing_count']:,}**")
            lines.append(
                f"- Coverage: left={relationship['left_coverage']}, right={relationship['right_coverage']}"
            )
            if relationship["join_shape"]:
                lines.append(f"- Relationship shape guess: {relationship['join_shape']}")
        else:
            lines.append("- Joinable on `listing_id`: no")
        lines.append("")

    lines.append("## File Roles")
    lines.append("")
    if raw_main:
        lines.append(f"- Main raw listing table: `{raw_main['relative_path']}`")
    if image_table:
        lines.append(f"- Image table: `{image_table['relative_path']}`")
    if validation_table:
        lines.append(f"- Validation/report table: `{validation_table['relative_path']}`")
    if tabular_train:
        lines.append(f"- Train-ready tabular/text table: `{tabular_train['relative_path']}`")
    if multimodal_train:
        lines.append(f"- Train-ready multimodal table: `{multimodal_train['relative_path']}`")
    if run_log:
        lines.append(f"- Crawl/process log table: `{run_log['relative_path']}`")
    lines.append("")

    lines.append("## Consistency Notes")
    lines.append("")
    if consistency_notes:
        for note in consistency_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No cross-file consistency anomalies were detected.")
    lines.append("")

    lines.append("## Dataset Readiness")
    lines.append("")
    lines.append(f"- Tabular model: {readiness['tabular']}")
    lines.append(f"- Text branch: {readiness['text']}")
    lines.append(f"- Image branch: {readiness['image']}")
    lines.append(f"- Multimodal fusion: {readiness['multimodal']}")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    lines.append(
        f"- Suggested target: `{target_column}`" if target_column else "- Suggested target: not identified"
    )
    lines.append(
        "- Baseline model feature columns: "
        + (", ".join(f"`{column}`" for column in baseline_columns) if baseline_columns else "not identified")
    )
    lines.append(
        "- Multimodal model feature columns: "
        + (
            ", ".join(f"`{column}`" for column in multimodal_columns)
            if multimodal_columns
            else "not identified"
        )
    )
    lines.append(
        "- Preprocessing note: `rooms` and `floor` still look string-like; "
        "`m2_net` appears fully missing in the main listing tables and should be dropped or reconstructed before training."
    )
    lines.append("- Join note: `listings.parquet` is the raw source of truth, while `train_ready_*` tables are filtered training subsets.")
    lines.append("")

    error_summaries = [summary for summary in summaries if summary["read_error"] is not None]
    lines.append("## Errors")
    lines.append("")
    if error_summaries:
        for summary in error_summaries:
            lines.append(f"- `{summary['relative_path']}` -> `{summary['read_error']}`")
    else:
        lines.append("- No parquet read errors occurred.")
    lines.append("")

    return "\n".join(lines)


def print_terminal_summary(
    parquet_count: int,
    estimated_listing_count: int,
    raw_main_listing_count: int | None,
    consistency_notes: list[str],
    primary_files: dict[str, Any],
    target_column: str | None,
    baseline_columns: list[str],
    multimodal_columns: list[str],
    report_path: Path,
) -> None:
    general_train = primary_files["multimodal_train"] or primary_files["tabular_train"] or primary_files["raw_main"]
    print(f"Toplam parquet dosyasi: {parquet_count}")
    if raw_main_listing_count is not None:
        print(f"Tahmini toplam ilan sayisi: {raw_main_listing_count:,}")
        print(f"Tum parquet union listing_id: {estimated_listing_count:,}")
    else:
        print(f"Tahmini toplam ilan sayisi: {estimated_listing_count:,}")
    print(
        "Ana egitim dosyasi: "
        + (general_train["relative_path"] if general_train else "belirlenemedi")
    )
    if primary_files["tabular_train"]:
        print(f"Baseline egitim dosyasi: {primary_files['tabular_train']['relative_path']}")
    if primary_files["multimodal_train"]:
        print(f"Multimodal egitim dosyasi: {primary_files['multimodal_train']['relative_path']}")
    print(f"Hedef degisken: {target_column or 'belirlenemedi'}")
    print(
        "Baseline onerilen kolonlar: "
        + (", ".join(baseline_columns) if baseline_columns else "belirlenemedi")
    )
    print(
        "Multimodal onerilen kolonlar: "
        + (", ".join(multimodal_columns) if multimodal_columns else "belirlenemedi")
    )
    if consistency_notes:
        print(f"Tutarlilik notu: {consistency_notes[0]}")
    print(f"Rapor: {report_path}")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = script_path.parent.parent
    default_report = default_root / "reports" / "dataset_summary.md"

    parser = argparse.ArgumentParser(description="Inspect all parquet files under a dataset project.")
    parser.add_argument("--root", type=Path, default=default_root, help="Project root to scan recursively.")
    parser.add_argument("--report", type=Path, default=default_report, help="Markdown report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.root.resolve()
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    engine_status = detect_parquet_engines()
    parquet_paths = sorted(project_root.rglob("*.parquet"))
    existing_keys, image_file_count = build_existing_file_index(project_root)

    summaries: list[dict[str, Any]] = []
    dataframes: dict[str, pd.DataFrame] = {}

    for parquet_path in parquet_paths:
        try:
            df = pd.read_parquet(parquet_path)
            dataframes[parquet_path.name] = df
            summaries.append(summarize_dataframe(parquet_path, project_root, df, existing_keys))
        except Exception as exc:
            summaries.append(summarize_read_error(parquet_path, project_root, exc))

    relationships = analyze_relationships(dataframes)
    primary_files = choose_primary_files(summaries)
    readiness = evaluate_dataset_readiness(primary_files)
    consistency_notes = collect_consistency_notes(dataframes, primary_files)

    report_body = build_report(
        project_root=project_root,
        report_path=report_path,
        parquet_paths=parquet_paths,
        summaries=summaries,
        dataframes=dataframes,
        relationships=relationships,
        primary_files=primary_files,
        readiness=readiness,
        engine_status=engine_status,
        image_file_count=image_file_count,
    )
    report_path.write_text(report_body, encoding="utf-8")

    estimated_listing_count = estimate_total_listings(dataframes)
    baseline_columns = recommend_baseline_columns(primary_files["tabular_train"])
    multimodal_columns = recommend_multimodal_columns(primary_files["multimodal_train"])
    target_column = None
    for candidate in (
        primary_files["multimodal_train"],
        primary_files["tabular_train"],
        primary_files["raw_main"],
    ):
        if candidate and candidate["price_target_candidate"]["column"]:
            target_column = candidate["price_target_candidate"]["column"]
            break

    print_terminal_summary(
        parquet_count=len(parquet_paths),
        estimated_listing_count=estimated_listing_count,
        raw_main_listing_count=primary_files["raw_main"]["unique_listing_count"] if primary_files["raw_main"] else None,
        consistency_notes=consistency_notes,
        primary_files=primary_files,
        target_column=target_column,
        baseline_columns=baseline_columns,
        multimodal_columns=multimodal_columns,
        report_path=report_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
