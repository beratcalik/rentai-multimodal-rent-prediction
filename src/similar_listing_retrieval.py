from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version '.*' or newer of 'numexpr'.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version '.*' or newer of 'bottleneck'.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.predict_single_listing import DEFAULT_INPUT_PATH, format_amount
from src.train_baseline import (
    normalize_furnished_flag,
    normalize_string,
    parse_floor_value,
    parse_room_count,
)


LOGGER = logging.getLogger("similar_listing_retrieval")

DEFAULT_DATASET_PATH = ROOT_DIR / "dataset" / "train_ready_multimodal.parquet"
LISTING_ID_COLUMN = "listing_id"
TARGET_COLUMN = "price_try"

RETRIEVAL_RAW_COLUMNS = [
    LISTING_ID_COLUMN,
    "district",
    "neighborhood",
    "rooms",
    "m2_gross",
    "building_age",
    "floor",
    "total_floors",
    "bathrooms",
    "heating_type",
    "is_furnished",
    "home_shape",
    TARGET_COLUMN,
]

RETRIEVAL_FEATURE_COLUMNS = [
    "district",
    "neighborhood",
    "rooms",
    "m2_gross",
    "building_age",
    "floor",
    "total_floors",
    "bathrooms",
    "heating_type",
    "is_furnished",
    "home_shape",
]

NUMERIC_FEATURE_COLUMNS = [
    "rooms",
    "m2_gross",
    "building_age",
    "floor",
    "total_floors",
    "bathrooms",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "district",
    "neighborhood",
    "heating_type",
    "is_furnished",
    "home_shape",
]

MAX_RESULTS = 5
LOW_SIMILARITY_THRESHOLD = 55
FALLBACK_SIMILARITY_THRESHOLD = 40
SEARCH_CANDIDATE_COUNT = 120

FEATURE_WEIGHTS = {
    "rooms": 2.6,
    "m2_gross": 2.8,
    "building_age": 1.0,
    "floor": 0.9,
    "total_floors": 0.7,
    "bathrooms": 0.9,
    "district_": 2.9,
    "neighborhood_": 3.4,
    "heating_type_": 0.8,
    "is_furnished_": 0.7,
    "home_shape_": 0.8,
}

HEURISTIC_MATCH_WEIGHTS = {
    "district": 24.0,
    "neighborhood": 20.0,
    "rooms": 16.0,
    "m2_gross": 18.0,
    "building_age": 6.0,
    "floor": 5.0,
    "total_floors": 4.0,
    "bathrooms": 4.0,
    "heating_type": 2.0,
    "is_furnished": 0.5,
    "home_shape": 0.5,
}

HEURISTIC_NUMERIC_TOLERANCES = {
    "rooms": 3.0,
    "m2_gross": 100.0,
    "building_age": 25.0,
    "floor": 12.0,
    "total_floors": 20.0,
    "bathrooms": 3.0,
}

STRICT_PRICE_GAP_RATIO = 0.60
RELAXED_PRICE_GAP_RATIO = 0.80
STRICT_M2_GAP_RATIO = 0.40
RELAXED_M2_GAP_RATIO = 0.55
STRICT_ROOM_GAP = 2.0
RELAXED_ROOM_GAP = 3.0


@dataclass
class SimilarityRuntime:
    raw_dataframe: pd.DataFrame
    cleaned_dataframe: pd.DataFrame
    preprocessor: ColumnTransformer
    weighted_matrix: np.ndarray
    nearest_neighbors: NearestNeighbors
    feature_names: list[str]
    feature_weights: np.ndarray
    distance_p10: float
    distance_p90: float


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve similar market examples for a single listing input."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input JSON path for the query listing.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Train-ready multimodal dataset used for retrieval.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON.",
    )
    return parser.parse_args()


def make_dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_input_payload(input_path: Path) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON bulunamadı: {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Input JSON tek bir ilan nesnesi olmalıdır.")
    return payload


def clean_retrieval_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.loc[:, RETRIEVAL_FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column == "is_furnished":
            frame[column] = frame[column].map(normalize_furnished_flag)
        else:
            frame[column] = frame[column].map(normalize_string)

    frame["rooms"] = frame["rooms"].map(parse_room_count)
    frame["floor"] = frame["floor"].map(parse_floor_value)

    for column in NUMERIC_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_dense_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURE_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURE_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_feature_weights(feature_names: list[str]) -> np.ndarray:
    weights = np.ones(len(feature_names), dtype=np.float32)
    for index, feature_name in enumerate(feature_names):
        if feature_name in FEATURE_WEIGHTS:
            weights[index] = np.float32(FEATURE_WEIGHTS[feature_name])
            continue

        for prefix, weight in FEATURE_WEIGHTS.items():
            if prefix.endswith("_") and feature_name.startswith(prefix):
                weights[index] = np.float32(weight)
                break
    return weights


def _normalize_distance_score(distance: float, p10: float, p90: float) -> int:
    if p90 <= p10:
        return int(np.clip(round(100 - distance * 10), 5, 99))

    ratio = (distance - p10) / max(p90 - p10, 1e-6)
    score = 100 - 90 * ratio
    return int(np.clip(round(score), 5, 99))


def _coerce_output_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if float(numeric_value).is_integer():
        return float(int(numeric_value))
    return round(numeric_value, 2)


def _normalize_match_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, str):
        return normalize_string(value)
    return normalize_string(str(value))


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _categorical_similarity(query_value: Any, candidate_value: Any) -> float | None:
    query_text = _normalize_match_text(query_value)
    candidate_text = _normalize_match_text(candidate_value)
    if not query_text or not candidate_text:
        return None
    return 1.0 if query_text == candidate_text else 0.0


def _numeric_similarity(query_value: Any, candidate_value: Any, tolerance: float) -> float | None:
    query_number = _safe_float(query_value)
    candidate_number = _safe_float(candidate_value)
    if query_number is None or candidate_number is None:
        return None

    difference = abs(query_number - candidate_number)
    similarity = max(0.0, 1.0 - (difference / max(tolerance, 1e-6)))
    return float(similarity)


def _relative_difference(query_value: Any, candidate_value: Any) -> float | None:
    query_number = _safe_float(query_value)
    candidate_number = _safe_float(candidate_value)
    if query_number is None or candidate_number is None:
        return None

    baseline = max(abs(query_number), abs(candidate_number), 1.0)
    return abs(query_number - candidate_number) / baseline


def _build_similarity_reasons(query_row: pd.Series, candidate_row: pd.Series) -> list[str]:
    reasons: list[str] = []

    if _normalize_match_text(query_row.get("district")) == _normalize_match_text(candidate_row.get("district")):
        reasons.append("Aynı ilçe")

    if _normalize_match_text(query_row.get("neighborhood")) == _normalize_match_text(candidate_row.get("neighborhood")):
        reasons.append("Aynı mahalle")

    room_gap = _numeric_similarity(
        query_row.get("rooms"),
        candidate_row.get("rooms"),
        HEURISTIC_NUMERIC_TOLERANCES["rooms"],
    )
    if room_gap is not None:
        room_difference = abs(float(query_row.get("rooms")) - float(candidate_row.get("rooms")))
        if room_difference < 0.1:
            reasons.append("Oda tipi aynı")
        elif room_difference <= 1.0:
            reasons.append("Oda tipi yakın")

    m2_gap_ratio = _relative_difference(query_row.get("m2_gross"), candidate_row.get("m2_gross"))
    if m2_gap_ratio is not None:
        if m2_gap_ratio <= 0.15:
            reasons.append("m² değeri yakın")
        elif m2_gap_ratio <= 0.30:
            reasons.append("m² aralığı benzer")

    floor_similarity = _numeric_similarity(
        query_row.get("floor"),
        candidate_row.get("floor"),
        HEURISTIC_NUMERIC_TOLERANCES["floor"],
    )
    if floor_similarity is not None and floor_similarity >= 0.75:
        reasons.append("Kat seviyesi benzer")

    building_age_similarity = _numeric_similarity(
        query_row.get("building_age"),
        candidate_row.get("building_age"),
        HEURISTIC_NUMERIC_TOLERANCES["building_age"],
    )
    if building_age_similarity is not None and building_age_similarity >= 0.8:
        reasons.append("Bina yaşı yakın")

    if _normalize_match_text(query_row.get("heating_type")) == _normalize_match_text(candidate_row.get("heating_type")):
        reasons.append("Isıtma tipi aynı")

    return reasons[:4] if reasons else ["Temel konut özellikleri benzer"]


def _passes_similarity_guardrails(
    query_row: pd.Series,
    candidate_row: pd.Series,
    predicted_rent_try: float | None,
    *,
    relaxed: bool,
) -> bool:
    max_room_gap = RELAXED_ROOM_GAP if relaxed else STRICT_ROOM_GAP
    max_m2_gap = RELAXED_M2_GAP_RATIO if relaxed else STRICT_M2_GAP_RATIO
    max_price_gap = RELAXED_PRICE_GAP_RATIO if relaxed else STRICT_PRICE_GAP_RATIO

    query_rooms = _safe_float(query_row.get("rooms"))
    candidate_rooms = _safe_float(candidate_row.get("rooms"))
    if query_rooms is not None and candidate_rooms is not None:
        if abs(query_rooms - candidate_rooms) > max_room_gap:
            return False

    m2_gap_ratio = _relative_difference(query_row.get("m2_gross"), candidate_row.get("m2_gross"))
    if m2_gap_ratio is not None and m2_gap_ratio > max_m2_gap:
        return False

    if predicted_rent_try is not None:
        candidate_price = _safe_float(candidate_row.get(TARGET_COLUMN))
        if candidate_price is not None:
            price_gap_ratio = _relative_difference(predicted_rent_try, candidate_price)
            if price_gap_ratio is not None and price_gap_ratio > max_price_gap:
                return False

    return True


def compute_similarity_score(query_row: pd.Series, candidate_row: pd.Series) -> int:
    weighted_score = 0.0
    total_weight = 0.0

    def maybe_add(score: float | None, weight: float) -> None:
        nonlocal weighted_score, total_weight
        if score is None:
            return
        weighted_score += score * weight
        total_weight += weight

    maybe_add(
        _categorical_similarity(query_row.get("district"), candidate_row.get("district")),
        HEURISTIC_MATCH_WEIGHTS["district"],
    )
    maybe_add(
        _categorical_similarity(query_row.get("neighborhood"), candidate_row.get("neighborhood")),
        HEURISTIC_MATCH_WEIGHTS["neighborhood"],
    )
    maybe_add(
        _numeric_similarity(query_row.get("rooms"), candidate_row.get("rooms"), HEURISTIC_NUMERIC_TOLERANCES["rooms"]),
        HEURISTIC_MATCH_WEIGHTS["rooms"],
    )
    maybe_add(
        _numeric_similarity(query_row.get("m2_gross"), candidate_row.get("m2_gross"), HEURISTIC_NUMERIC_TOLERANCES["m2_gross"]),
        HEURISTIC_MATCH_WEIGHTS["m2_gross"],
    )
    maybe_add(
        _numeric_similarity(
            query_row.get("building_age"),
            candidate_row.get("building_age"),
            HEURISTIC_NUMERIC_TOLERANCES["building_age"],
        ),
        HEURISTIC_MATCH_WEIGHTS["building_age"],
    )
    maybe_add(
        _numeric_similarity(query_row.get("floor"), candidate_row.get("floor"), HEURISTIC_NUMERIC_TOLERANCES["floor"]),
        HEURISTIC_MATCH_WEIGHTS["floor"],
    )
    maybe_add(
        _numeric_similarity(
            query_row.get("total_floors"),
            candidate_row.get("total_floors"),
            HEURISTIC_NUMERIC_TOLERANCES["total_floors"],
        ),
        HEURISTIC_MATCH_WEIGHTS["total_floors"],
    )
    maybe_add(
        _numeric_similarity(
            query_row.get("bathrooms"),
            candidate_row.get("bathrooms"),
            HEURISTIC_NUMERIC_TOLERANCES["bathrooms"],
        ),
        HEURISTIC_MATCH_WEIGHTS["bathrooms"],
    )
    maybe_add(
        _categorical_similarity(query_row.get("heating_type"), candidate_row.get("heating_type")),
        HEURISTIC_MATCH_WEIGHTS["heating_type"],
    )
    maybe_add(
        _categorical_similarity(query_row.get("is_furnished"), candidate_row.get("is_furnished")),
        HEURISTIC_MATCH_WEIGHTS["is_furnished"],
    )
    maybe_add(
        _categorical_similarity(query_row.get("home_shape"), candidate_row.get("home_shape")),
        HEURISTIC_MATCH_WEIGHTS["home_shape"],
    )

    if total_weight <= 0:
        return 0

    base_score = 100.0 * (weighted_score / total_weight)

    if _normalize_match_text(query_row.get("district")) == _normalize_match_text(candidate_row.get("district")):
        base_score += 6.0
    if _normalize_match_text(query_row.get("neighborhood")) == _normalize_match_text(candidate_row.get("neighborhood")):
        base_score += 8.0
    if _safe_float(query_row.get("m2_gross")) is not None and _safe_float(candidate_row.get("m2_gross")) is not None:
        if abs(float(query_row.get("m2_gross")) - float(candidate_row.get("m2_gross"))) <= 15:
            base_score += 4.0
    if _safe_float(query_row.get("rooms")) is not None and _safe_float(candidate_row.get("rooms")) is not None:
        if abs(float(query_row.get("rooms")) - float(candidate_row.get("rooms"))) < 0.1:
            base_score += 4.0

    return int(np.clip(round(base_score), 0, 99))


@lru_cache(maxsize=1)
def get_similarity_runtime(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> SimilarityRuntime:
    resolved_dataset_path = Path(dataset_path).resolve()
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"Retrieval dataset bulunamadı: {resolved_dataset_path}")

    LOGGER.info("Benzer ilan retrieval dataset yükleniyor: %s", resolved_dataset_path)
    raw_dataframe = pd.read_parquet(resolved_dataset_path, columns=RETRIEVAL_RAW_COLUMNS).copy()
    raw_dataframe = raw_dataframe.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    cleaned_dataframe = clean_retrieval_features(raw_dataframe)
    preprocessor = build_preprocessor()
    transformed_matrix = np.asarray(preprocessor.fit_transform(cleaned_dataframe), dtype=np.float32)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    feature_weights = build_feature_weights(feature_names)
    weighted_matrix = transformed_matrix * feature_weights

    n_neighbors = min(max(6, SEARCH_CANDIDATE_COUNT), len(raw_dataframe))
    nearest_neighbors = NearestNeighbors(metric="euclidean", n_neighbors=n_neighbors)
    nearest_neighbors.fit(weighted_matrix)

    calibration_neighbors = min(6, len(raw_dataframe))
    distances, _ = nearest_neighbors.kneighbors(weighted_matrix, n_neighbors=calibration_neighbors)
    non_self_distances = distances[:, 1:].reshape(-1) if calibration_neighbors > 1 else distances.reshape(-1)
    non_self_distances = non_self_distances[np.isfinite(non_self_distances)]
    if len(non_self_distances) == 0:
        distance_p10 = 0.0
        distance_p90 = 1.0
    else:
        distance_p10 = float(np.quantile(non_self_distances, 0.10))
        distance_p90 = float(np.quantile(non_self_distances, 0.90))

    return SimilarityRuntime(
        raw_dataframe=raw_dataframe,
        cleaned_dataframe=cleaned_dataframe,
        preprocessor=preprocessor,
        weighted_matrix=weighted_matrix,
        nearest_neighbors=nearest_neighbors,
        feature_names=feature_names,
        feature_weights=feature_weights,
        distance_p10=distance_p10,
        distance_p90=distance_p90,
    )


def prepare_query_features(input_data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    record = {
        column: input_data.get(column, np.nan)
        for column in RETRIEVAL_FEATURE_COLUMNS
    }
    raw_frame = pd.DataFrame([record])
    cleaned_frame = clean_retrieval_features(raw_frame)
    return raw_frame, cleaned_frame


def format_similar_listing(row: pd.Series, similarity_score: int, similarity_reasons: list[str]) -> dict[str, Any]:
    price_value = int(round(float(row[TARGET_COLUMN])))
    return {
        "district": str(row.get("district") or ""),
        "neighborhood": str(row.get("neighborhood") or ""),
        "rooms": str(row.get("rooms") or ""),
        "m2_gross": _coerce_output_float(row.get("m2_gross")),
        "building_age": _coerce_output_float(row.get("building_age")),
        "floor": None if pd.isna(row.get("floor")) else str(row.get("floor")),
        "price_try": price_value,
        "price_formatted": f"{format_amount(price_value)} TL",
        "similarity_score": int(similarity_score),
        "similarity_reasons": list(similarity_reasons),
    }


def retrieve_similar_listings_from_dict(
    input_data: dict[str, Any],
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    max_results: int = MAX_RESULTS,
    predicted_rent_try: float | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(input_data, dict):
        raise ValueError("retrieve_similar_listings_from_dict bir ilan sözlüğü bekler.")

    runtime = get_similarity_runtime(dataset_path)
    _, cleaned_query = prepare_query_features(input_data)
    query_series = cleaned_query.iloc[0]
    transformed_query = np.asarray(runtime.preprocessor.transform(cleaned_query), dtype=np.float32)
    weighted_query = transformed_query * runtime.feature_weights

    n_neighbors = min(SEARCH_CANDIDATE_COUNT, len(runtime.raw_dataframe))
    distances, indices = runtime.nearest_neighbors.kneighbors(weighted_query, n_neighbors=n_neighbors)
    query_listing_id = input_data.get(LISTING_ID_COLUMN)

    candidate_results: list[dict[str, Any]] = []

    for distance, candidate_index in zip(distances[0].tolist(), indices[0].tolist(), strict=False):
        candidate_row = runtime.raw_dataframe.iloc[int(candidate_index)]
        candidate_cleaned = runtime.cleaned_dataframe.iloc[int(candidate_index)]
        candidate_listing_id = candidate_row.get(LISTING_ID_COLUMN)
        if query_listing_id and candidate_listing_id == query_listing_id:
            continue

        knn_score = _normalize_distance_score(distance, runtime.distance_p10, runtime.distance_p90)
        heuristic_score = compute_similarity_score(query_series, candidate_cleaned)
        similarity_score = int(round((heuristic_score * 0.8) + (knn_score * 0.2)))
        if not _passes_similarity_guardrails(
            query_series,
            candidate_cleaned,
            predicted_rent_try,
            relaxed=True,
        ):
            continue

        similarity_reasons = _build_similarity_reasons(query_series, candidate_cleaned)
        candidate_result = format_similar_listing(candidate_row, similarity_score, similarity_reasons)
        candidate_result["_strict_match"] = _passes_similarity_guardrails(
            query_series,
            candidate_cleaned,
            predicted_rent_try,
            relaxed=False,
        )
        candidate_result["_same_district"] = (
            _normalize_match_text(query_series.get("district"))
            == _normalize_match_text(candidate_cleaned.get("district"))
        )
        candidate_result["_same_neighborhood"] = (
            _normalize_match_text(query_series.get("neighborhood"))
            == _normalize_match_text(candidate_cleaned.get("neighborhood"))
        )
        candidate_result["_price_gap_ratio"] = (
            _relative_difference(predicted_rent_try, candidate_row.get(TARGET_COLUMN))
            if predicted_rent_try is not None
            else 0.0
        )
        candidate_result["_signature"] = (
            str(candidate_row.get("district") or ""),
            str(candidate_row.get("neighborhood") or ""),
            str(candidate_row.get("rooms") or ""),
            _coerce_output_float(candidate_row.get("m2_gross")),
            int(round(float(candidate_row.get(TARGET_COLUMN)))),
        )
        candidate_results.append(candidate_result)

    candidate_results.sort(
        key=lambda item: (
            1 if item["_same_neighborhood"] else 0,
            1 if item["_same_district"] else 0,
            item["similarity_score"],
            -float(item["_price_gap_ratio"]),
        ),
        reverse=True,
    )

    def collect_results(min_score: int, *, strict_only: bool) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_signatures: set[tuple[Any, ...]] = set()
        for item in candidate_results:
            if int(item["similarity_score"]) < min_score:
                continue
            if strict_only and not item["_strict_match"]:
                continue
            signature = item["_signature"]
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            selected.append(
                {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "_signature",
                        "_strict_match",
                        "_same_district",
                        "_same_neighborhood",
                        "_price_gap_ratio",
                    }
                }
            )
            if len(selected) >= max_results:
                break
        return selected

    strict_results = collect_results(LOW_SIMILARITY_THRESHOLD, strict_only=True)
    if len(strict_results) >= min(3, max_results):
        return strict_results

    relaxed_results = collect_results(LOW_SIMILARITY_THRESHOLD, strict_only=False)
    if relaxed_results:
        merged_results: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, ...]] = set()
        for item in strict_results + relaxed_results:
            signature = (
                item.get("district"),
                item.get("neighborhood"),
                item.get("rooms"),
                item.get("m2_gross"),
                item.get("price_try"),
            )
            if signature in seen_keys:
                continue
            seen_keys.add(signature)
            merged_results.append(item)
            if len(merged_results) >= max_results:
                break
        if merged_results:
            return merged_results

    return collect_results(FALLBACK_SIMILARITY_THRESHOLD, strict_only=False)


def print_human_summary(similar_listings: list[dict[str, Any]]) -> None:
    if not similar_listings:
        print("Yeterli benzer örnek bulunamadı.")
        return

    print("Benzer piyasa örnekleri:")
    for index, item in enumerate(similar_listings, start=1):
        location = " / ".join(part for part in [item["district"], item["neighborhood"]] if part)
        details = " • ".join(
            part
            for part in [
                item.get("rooms") or "",
                f"{int(item['m2_gross'])} m²" if item.get("m2_gross") is not None else "",
                item.get("floor") or "",
            ]
            if part
        )
        print(f"{index}. {location}")
        if details:
            print(f"   {details}")
        print(f"   İlan kirası: {item['price_formatted']} | Benzerlik: %{item['similarity_score']}")
        if item.get("similarity_reasons"):
            print(f"   Neden benzer: {', '.join(str(reason) for reason in item['similarity_reasons'])}")


def main() -> int:
    configure_logging()
    args = parse_args()
    payload = load_input_payload(args.input.resolve())
    similar_listings = retrieve_similar_listings_from_dict(
        input_data=payload,
        dataset_path=args.dataset.resolve(),
    )

    if args.json:
        print(
            json.dumps(
                {"similar_listings": similar_listings},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_human_summary(similar_listings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
