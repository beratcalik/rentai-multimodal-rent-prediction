from __future__ import annotations

import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from train_baseline import (
    RANDOM_STATE,
    TARGET_COLUMN,
    calculate_metrics,
    configure_logging,
    split_dataset,
)
from train_clip_fusion_reduced import (
    build_image_processor,
    build_model_factories,
    build_tabular_preprocessor,
    render_model_status,
)
from train_image_fusion_reduced import (
    FUSION_TABULAR_COLUMNS,
    M2_BIN_EDGES,
    M2_BIN_LABELS,
    PRICE_BIN_EDGES,
    PRICE_BIN_LABELS,
    clean_tabular_features,
    dataframe_to_markdown_table,
    format_float,
    render_markdown_table,
    train_matched_tabular_baseline,
)

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Eksik bagimlilik bulundu. Lutfen once `python -m pip install -r requirements.txt` calistirin."
    ) from exc


LOGGER = logging.getLogger("final_multimodal_text_clip_training")

ROOT_DIR = Path(__file__).resolve().parent.parent
MULTIMODAL_DATASET_PATH = ROOT_DIR / "dataset" / "train_ready_multimodal.parquet"
CLIP_CAPS_EMBEDDINGS_PATH = ROOT_DIR / "dataset" / "clip_image_embeddings_caps.parquet"
MODEL_OUTPUT_PATH = ROOT_DIR / "models" / "final_multimodal_text_clip_model.joblib"
REPORT_OUTPUT_PATH = ROOT_DIR / "reports" / "final_multimodal_text_clip_results.md"

LISTING_ID_COLUMN = "listing_id"
TITLE_COLUMN = "title"
DESCRIPTION_COLUMN = "description"
IMAGE_CAP_COLUMN = "image_cap"
USED_IMAGE_COUNT_COLUMN = "used_image_count"
CLIP_REPRESENTATION_COLUMN = "clip_meanmax_embedding"
CLIP_CAP_VALUE = 16
PRIMARY_SELECTION_METRIC = "mae"
TEXT_TFIDF_MAX_FEATURES = [2000, 5000]
TEXT_SVD_DIMS = [16, 32, 64]
IMAGE_PCA_DIMS = [16, 32]
TEXT_SVD_PREFIX = "text_svd_"
IMAGE_REDUCED_PREFIX = "clip_reduced_"
TEXT_MIN_TITLE_TOKEN_COUNT = 1
TEXT_MIN_DESCRIPTION_TOKEN_COUNT = 3
TEXT_MIN_COMBINED_TOKEN_COUNT = 1

MATCHED_BASELINE_REFERENCE = {
    "mae": 4700.08,
    "rmse": 7057.76,
    "r2": 0.8013,
    "mape": 13.79,
}
BEST_CLIP_CAP_REFERENCE = {
    "mae": 4346.62,
    "rmse": 6441.28,
    "r2": 0.8345,
    "mape": 12.95,
}
BEST_SIGLIP_REFERENCE = {
    "mae": 4441.86,
    "rmse": 6484.50,
    "r2": 0.8323,
    "mape": 12.98,
}

BOILERPLATE_PATTERNS = [
    "telefonu goster",
    "detayli bilgi",
    "detaylar icin",
    "arayiniz",
    "arayin",
    "iletisime geciniz",
    "gayrimenkul",
    "emlak",
    "kahve icmeye",
    "yetkili",
    "danisman",
    "portfoy",
    "realty",
]


@dataclass
class ValidationTrial:
    text_max_features: int
    text_svd_dim: int
    image_pca_dim: int
    model_label: str
    validation_metrics: dict[str, float]
    factory: Callable[[], Any]


def ensure_output_directories() -> None:
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_base_multimodal_and_clip_embeddings(
    multimodal_path: Path,
    clip_caps_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], int]:
    if not multimodal_path.exists():
        raise FileNotFoundError(f"Multimodal dataset bulunamadi: {multimodal_path}")
    if not clip_caps_path.exists():
        raise FileNotFoundError(f"CLIP caps embedding parquet bulunamadi: {clip_caps_path}")

    LOGGER.info("Multimodal dataset okunuyor: %s", multimodal_path)
    multimodal_df = pd.read_parquet(multimodal_path)
    LOGGER.info("CLIP caps embedding dataset okunuyor: %s", clip_caps_path)
    clip_caps_df = pd.read_parquet(clip_caps_path)

    required_multimodal_columns = set(
        FUSION_TABULAR_COLUMNS
        + [TARGET_COLUMN, LISTING_ID_COLUMN, TITLE_COLUMN, DESCRIPTION_COLUMN]
    )
    missing_multimodal_columns = sorted(required_multimodal_columns - set(multimodal_df.columns))
    if missing_multimodal_columns:
        raise ValueError(f"Multimodal dataset icinde eksik kolonlar bulundu: {missing_multimodal_columns}")

    required_clip_columns = {
        LISTING_ID_COLUMN,
        IMAGE_CAP_COLUMN,
        USED_IMAGE_COUNT_COLUMN,
        CLIP_REPRESENTATION_COLUMN,
    }
    missing_clip_columns = sorted(required_clip_columns - set(clip_caps_df.columns))
    if missing_clip_columns:
        raise ValueError(f"CLIP caps dataset icinde eksik kolonlar bulundu: {missing_clip_columns}")

    clip_caps_df = clip_caps_df[clip_caps_df[IMAGE_CAP_COLUMN] == CLIP_CAP_VALUE].copy()
    if clip_caps_df.empty:
        raise ValueError(f"image_cap={CLIP_CAP_VALUE} icin hic CLIP embedding bulunamadi.")

    multimodal_df[LISTING_ID_COLUMN] = multimodal_df[LISTING_ID_COLUMN].astype(str)
    clip_caps_df[LISTING_ID_COLUMN] = clip_caps_df[LISTING_ID_COLUMN].astype(str)
    clip_caps_df = clip_caps_df.drop_duplicates(subset=[LISTING_ID_COLUMN], keep="first").reset_index(drop=True)

    embedding_vectors: list[np.ndarray] = []
    embedding_dimension: int | None = None
    for value in clip_caps_df[CLIP_REPRESENTATION_COLUMN]:
        vector = np.asarray(value, dtype=np.float32).ravel()
        if embedding_dimension is None:
            embedding_dimension = int(len(vector))
        elif len(vector) != embedding_dimension:
            raise ValueError("clip_meanmax_embedding kolonunda tutarsiz vektor boyutu bulundu.")
        embedding_vectors.append(vector)

    if embedding_dimension is None:
        raise ValueError("clip_meanmax_embedding kolonunda hic vektor bulunamadi.")

    valid_listing_ids = set(clip_caps_df[LISTING_ID_COLUMN].tolist())
    base_df = multimodal_df[multimodal_df[LISTING_ID_COLUMN].isin(valid_listing_ids)].copy()
    base_df = base_df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    clip_store = {
        "listing_ids": clip_caps_df[LISTING_ID_COLUMN].tolist(),
        "index_by_listing": {
            listing_id: index
            for index, listing_id in enumerate(clip_caps_df[LISTING_ID_COLUMN].tolist())
        },
        "used_image_counts": clip_caps_df[USED_IMAGE_COUNT_COLUMN].astype(int).to_numpy(dtype=np.int32, copy=False),
        "embedding_matrix": np.vstack(embedding_vectors).astype(np.float32, copy=False),
    }

    LOGGER.info("Joinable listing sayisi: %s", len(base_df))
    return base_df, clip_store, embedding_dimension


def collapse_whitespace(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def strip_combining_marks(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def clean_text_value(value: Any, min_token_count: int) -> str:
    text = collapse_whitespace(value)
    if not text:
        return ""

    text = strip_combining_marks(text)
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(rf"\b{re.escape(pattern)}\b", " ", text)

    text = re.sub(r"\b[a-z]{1,2}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text.split()) < min_token_count:
        return ""
    return text


def build_clean_text_columns(dataframe: pd.DataFrame) -> dict[str, pd.Series]:
    title_clean = dataframe[TITLE_COLUMN].map(
        lambda value: clean_text_value(value, min_token_count=TEXT_MIN_TITLE_TOKEN_COUNT)
    )
    description_clean = dataframe[DESCRIPTION_COLUMN].map(
        lambda value: clean_text_value(value, min_token_count=TEXT_MIN_DESCRIPTION_TOKEN_COUNT)
    )

    combined_text = []
    for title_value, description_value in zip(title_clean.tolist(), description_clean.tolist(), strict=False):
        combined = " ".join(part for part in [title_value, description_value] if part).strip()
        if len(combined.split()) < TEXT_MIN_COMBINED_TOKEN_COUNT:
            combined = title_value or description_value or ""
        combined_text.append(combined)

    return {
        "title_clean": title_clean,
        "description_clean": description_clean,
        "combined_text": pd.Series(combined_text, index=dataframe.index, dtype="object"),
    }


def extract_indices_for_split(clip_store: dict[str, Any], split_listing_ids: list[str]) -> np.ndarray:
    index_by_listing = clip_store["index_by_listing"]
    try:
        return np.asarray([index_by_listing[listing_id] for listing_id in split_listing_ids], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(f"Split icinde CLIP embedding bulunamayan listing bulundu: {exc}") from exc


def extract_image_matrix(clip_store: dict[str, Any], split_listing_ids: list[str]) -> np.ndarray:
    indices = extract_indices_for_split(clip_store, split_listing_ids)
    matrix = clip_store["embedding_matrix"][indices]
    return np.asarray(matrix, dtype=np.float32)


def build_text_cache(
    train_text: pd.Series,
    validation_text: pd.Series,
) -> dict[tuple[int, int], dict[str, Any]]:
    cache: dict[tuple[int, int], dict[str, Any]] = {}

    for max_features in TEXT_TFIDF_MAX_FEATURES:
        LOGGER.info("TF-IDF fit ediliyor | max_features=%s", max_features)
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=3,
        )
        X_train_tfidf = vectorizer.fit_transform(train_text)
        X_validation_tfidf = vectorizer.transform(validation_text)
        tfidf_feature_count = int(X_train_tfidf.shape[1])

        for requested_svd_dim in TEXT_SVD_DIMS:
            actual_svd_dim = min(
                requested_svd_dim,
                max(2, tfidf_feature_count - 1),
                max(2, int(X_train_tfidf.shape[0]) - 1),
            )
            LOGGER.info(
                "Text SVD fit ediliyor | max_features=%s | requested_dim=%s | actual_dim=%s",
                max_features,
                requested_svd_dim,
                actual_svd_dim,
            )
            svd = TruncatedSVD(n_components=actual_svd_dim, random_state=RANDOM_STATE)
            X_train_text = svd.fit_transform(X_train_tfidf).astype(np.float32, copy=False)
            X_validation_text = svd.transform(X_validation_tfidf).astype(np.float32, copy=False)
            cache[(max_features, requested_svd_dim)] = {
                "vectorizer": vectorizer,
                "svd": svd,
                "actual_svd_dim": actual_svd_dim,
                "tfidf_feature_count": tfidf_feature_count,
                "X_train_text": X_train_text,
                "X_validation_text": X_validation_text,
            }

    return cache


def fit_text_processor_for_final(
    train_validation_text: pd.Series,
    test_text: pd.Series,
    max_features: int,
    requested_svd_dim: int,
) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
    )
    X_train_validation_tfidf = vectorizer.fit_transform(train_validation_text)
    X_test_tfidf = vectorizer.transform(test_text)
    tfidf_feature_count = int(X_train_validation_tfidf.shape[1])
    actual_svd_dim = min(
        requested_svd_dim,
        max(2, tfidf_feature_count - 1),
        max(2, int(X_train_validation_tfidf.shape[0]) - 1),
    )
    svd = TruncatedSVD(n_components=actual_svd_dim, random_state=RANDOM_STATE)
    X_train_validation_text = svd.fit_transform(X_train_validation_tfidf).astype(np.float32, copy=False)
    X_test_text = svd.transform(X_test_tfidf).astype(np.float32, copy=False)
    return {
        "vectorizer": vectorizer,
        "svd": svd,
        "actual_svd_dim": actual_svd_dim,
        "tfidf_feature_count": tfidf_feature_count,
        "X_train_validation_text": X_train_validation_text,
        "X_test_text": X_test_text,
    }


def build_image_cache(
    train_listing_ids: list[str],
    validation_listing_ids: list[str],
    clip_store: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    train_image_matrix = extract_image_matrix(clip_store, train_listing_ids)
    validation_image_matrix = extract_image_matrix(clip_store, validation_listing_ids)
    cache: dict[int, dict[str, Any]] = {}

    for image_pca_dim in IMAGE_PCA_DIMS:
        LOGGER.info("Image PCA fit ediliyor | image_pca_dim=%s", image_pca_dim)
        image_processor = build_image_processor(image_pca_dim)
        X_train_image = image_processor.fit_transform(train_image_matrix).astype(np.float32, copy=False)
        X_validation_image = image_processor.transform(validation_image_matrix).astype(np.float32, copy=False)
        cache[image_pca_dim] = {
            "image_processor": image_processor,
            "X_train_image": X_train_image,
            "X_validation_image": X_validation_image,
        }

    return cache


def fit_image_processor_for_final(
    train_validation_listing_ids: list[str],
    test_listing_ids: list[str],
    clip_store: dict[str, Any],
    image_pca_dim: int,
) -> dict[str, Any]:
    train_validation_image_matrix = extract_image_matrix(clip_store, train_validation_listing_ids)
    test_image_matrix = extract_image_matrix(clip_store, test_listing_ids)
    image_processor = build_image_processor(image_pca_dim)
    X_train_validation_image = image_processor.fit_transform(train_validation_image_matrix).astype(np.float32, copy=False)
    X_test_image = image_processor.transform(test_image_matrix).astype(np.float32, copy=False)
    return {
        "image_processor": image_processor,
        "X_train_validation_image": X_train_validation_image,
        "X_test_image": X_test_image,
    }


def concatenate_multimodal_blocks(*blocks: np.ndarray) -> np.ndarray:
    fused = np.hstack(blocks)
    return np.asarray(fused, dtype=np.float32)


def train_validation_trials(
    candidates: list[dict[str, Any]],
    X_train_tabular: np.ndarray,
    X_validation_tabular: np.ndarray,
    text_cache: dict[tuple[int, int], dict[str, Any]],
    image_cache: dict[int, dict[str, Any]],
    y_train: pd.Series,
    y_validation: pd.Series,
    model_tracking: dict[str, dict[str, Any]],
) -> list[ValidationTrial]:
    results: list[ValidationTrial] = []

    for max_features in TEXT_TFIDF_MAX_FEATURES:
        for text_svd_dim in TEXT_SVD_DIMS:
            text_info = text_cache[(max_features, text_svd_dim)]
            X_train_text = text_info["X_train_text"]
            X_validation_text = text_info["X_validation_text"]

            for image_pca_dim in IMAGE_PCA_DIMS:
                image_info = image_cache[image_pca_dim]
                X_train_fused = concatenate_multimodal_blocks(
                    X_train_tabular,
                    X_train_text,
                    image_info["X_train_image"],
                )
                X_validation_fused = concatenate_multimodal_blocks(
                    X_validation_tabular,
                    X_validation_text,
                    image_info["X_validation_image"],
                )

                for candidate in candidates:
                    model_label = candidate["label"]
                    LOGGER.info(
                        "Model egitiliyor | tfidf=%s | text_svd=%s | image_pca=%s | %s",
                        max_features,
                        text_svd_dim,
                        image_pca_dim,
                        model_label,
                    )
                    try:
                        model = candidate["factory"]()
                        model.fit(X_train_fused, y_train)
                        validation_predictions = np.asarray(model.predict(X_validation_fused), dtype=float)
                        validation_metrics = calculate_metrics(y_validation, validation_predictions)
                        results.append(
                            ValidationTrial(
                                text_max_features=max_features,
                                text_svd_dim=text_svd_dim,
                                image_pca_dim=image_pca_dim,
                                model_label=model_label,
                                validation_metrics=validation_metrics,
                                factory=candidate["factory"],
                            )
                        )
                        LOGGER.info(
                            "Validation | tfidf=%s | text_svd=%s | image_pca=%s | %s | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
                            max_features,
                            text_svd_dim,
                            image_pca_dim,
                            model_label,
                            validation_metrics["mae"],
                            validation_metrics["rmse"],
                            validation_metrics["r2"],
                            validation_metrics["mape"],
                        )
                        model_tracking[model_label]["success_combinations"].append(
                            f"tfidf={max_features}|text={text_svd_dim}|image={image_pca_dim}"
                        )
                    except Exception as exc:  # pragma: no cover - defensive path
                        LOGGER.exception(
                            "Model denemesi basarisiz oldu | tfidf=%s | text_svd=%s | image_pca=%s | %s",
                            max_features,
                            text_svd_dim,
                            image_pca_dim,
                            model_label,
                        )
                        model_tracking[model_label]["failure_notes"].append(
                            f"tfidf={max_features}|text={text_svd_dim}|image={image_pca_dim}: "
                            f"{exc.__class__.__name__}: {exc}"
                        )

    if not results:
        raise RuntimeError("Hicbir final multimodal text+clip modeli basariyla egitilemedi.")

    return results


def select_best_trial(results: list[ValidationTrial]) -> ValidationTrial:
    best_result = min(results, key=lambda item: item.validation_metrics[PRIMARY_SELECTION_METRIC])
    LOGGER.info(
        "En iyi kombinasyon secildi: tfidf=%s + text_svd=%s + image_pca=%s + %s (validation MAE=%.2f)",
        best_result.text_max_features,
        best_result.text_svd_dim,
        best_result.image_pca_dim,
        best_result.model_label,
        best_result.validation_metrics["mae"],
    )
    return best_result


def build_validation_leaderboard(results: list[ValidationTrial]) -> pd.DataFrame:
    leaderboard_df = pd.DataFrame(
        [
            {
                "text_max_features": result.text_max_features,
                "text_svd_dim": result.text_svd_dim,
                "image_pca_dim": result.image_pca_dim,
                "model": result.model_label,
                "validation_mae": result.validation_metrics["mae"],
                "validation_rmse": result.validation_metrics["rmse"],
                "validation_r2": result.validation_metrics["r2"],
                "validation_mape": result.validation_metrics["mape"],
            }
            for result in results
        ]
    )
    return leaderboard_df.sort_values(
        by=[
            "validation_mae",
            "validation_rmse",
            "text_max_features",
            "text_svd_dim",
            "image_pca_dim",
            "model",
        ],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)


def build_text_config_summary(validation_leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for max_features in TEXT_TFIDF_MAX_FEATURES:
        for text_svd_dim in TEXT_SVD_DIMS:
            subset = validation_leaderboard[
                (validation_leaderboard["text_max_features"] == max_features)
                & (validation_leaderboard["text_svd_dim"] == text_svd_dim)
            ]
            if subset.empty:
                continue
            best_row = subset.iloc[0]
            rows.append(
                {
                    "text_max_features": max_features,
                    "text_svd_dim": text_svd_dim,
                    "best_image_pca_dim": int(best_row["image_pca_dim"]),
                    "best_model": best_row["model"],
                    "best_validation_mae": float(best_row["validation_mae"]),
                    "best_validation_rmse": float(best_row["validation_rmse"]),
                    "best_validation_r2": float(best_row["validation_r2"]),
                    "best_validation_mape": float(best_row["validation_mape"]),
                }
            )
    return pd.DataFrame(rows)


def build_image_pca_summary(validation_leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image_pca_dim in IMAGE_PCA_DIMS:
        subset = validation_leaderboard[validation_leaderboard["image_pca_dim"] == image_pca_dim]
        if subset.empty:
            continue
        best_row = subset.iloc[0]
        rows.append(
            {
                "image_pca_dim": image_pca_dim,
                "best_text_max_features": int(best_row["text_max_features"]),
                "best_text_svd_dim": int(best_row["text_svd_dim"]),
                "best_model": best_row["model"],
                "best_validation_mae": float(best_row["validation_mae"]),
                "best_validation_rmse": float(best_row["validation_rmse"]),
                "best_validation_r2": float(best_row["validation_r2"]),
                "best_validation_mape": float(best_row["validation_mape"]),
            }
        )
    return pd.DataFrame(rows)


def build_clip_reference_factory() -> Callable[[], Any]:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Best CLIP referans modelini yeniden egitmek icin XGBoost gerekli ama import edilemedi."
        ) from exc

    return lambda: XGBRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )


def build_prediction_frame(
    test_df: pd.DataFrame,
    baseline_predictions: np.ndarray,
    clip_predictions: np.ndarray,
    multimodal_predictions: np.ndarray,
) -> pd.DataFrame:
    frame = test_df.loc[:, [LISTING_ID_COLUMN, "district", "neighborhood", "rooms", "m2_gross"]].copy()
    frame["actual_price_try"] = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    frame["baseline_prediction"] = np.asarray(baseline_predictions, dtype=float)
    frame["clip_prediction"] = np.asarray(clip_predictions, dtype=float)
    frame["multimodal_prediction"] = np.asarray(multimodal_predictions, dtype=float)

    frame["baseline_abs_error"] = np.abs(frame["baseline_prediction"] - frame["actual_price_try"])
    frame["clip_abs_error"] = np.abs(frame["clip_prediction"] - frame["actual_price_try"])
    frame["multimodal_abs_error"] = np.abs(frame["multimodal_prediction"] - frame["actual_price_try"])

    frame["baseline_ape_pct"] = (
        frame["baseline_abs_error"] / frame["actual_price_try"].clip(lower=1e-8)
    ) * 100.0
    frame["clip_ape_pct"] = (
        frame["clip_abs_error"] / frame["actual_price_try"].clip(lower=1e-8)
    ) * 100.0
    frame["multimodal_ape_pct"] = (
        frame["multimodal_abs_error"] / frame["actual_price_try"].clip(lower=1e-8)
    ) * 100.0

    frame["price_range"] = pd.cut(
        frame["actual_price_try"],
        bins=PRICE_BIN_EDGES,
        labels=PRICE_BIN_LABELS,
        include_lowest=True,
        right=False,
    )
    frame["m2_range"] = pd.cut(
        frame["m2_gross"],
        bins=M2_BIN_EDGES,
        labels=M2_BIN_LABELS,
        include_lowest=True,
        right=False,
    )
    frame["multimodal_residual"] = frame["multimodal_prediction"] - frame["actual_price_try"]
    frame["abs_error_gain_vs_clip"] = frame["clip_abs_error"] - frame["multimodal_abs_error"]
    frame["abs_error_gain_vs_baseline"] = frame["baseline_abs_error"] - frame["multimodal_abs_error"]
    return frame


def summarize_reference_improvement(
    prediction_frame: pd.DataFrame,
    group_column: str,
    reference_prefix: str,
    target_prefix: str,
    min_samples: int = 1,
) -> pd.DataFrame:
    summary_df = prediction_frame.copy()
    summary_df[group_column] = summary_df[group_column].astype("object").where(
        summary_df[group_column].notna(),
        "Unknown",
    )

    grouped = (
        summary_df.groupby(group_column, dropna=False)
        .agg(
            sample_count=(LISTING_ID_COLUMN, "size"),
            reference_mae=(f"{reference_prefix}_abs_error", "mean"),
            target_mae=(f"{target_prefix}_abs_error", "mean"),
            reference_mape=(f"{reference_prefix}_ape_pct", "mean"),
            target_mape=(f"{target_prefix}_ape_pct", "mean"),
        )
        .reset_index()
        .rename(columns={group_column: "group"})
    )
    grouped = grouped[grouped["sample_count"] >= min_samples].copy()
    grouped["mae_improvement"] = grouped["reference_mae"] - grouped["target_mae"]
    grouped["mape_improvement"] = grouped["reference_mape"] - grouped["target_mape"]
    grouped = grouped.sort_values("mae_improvement", ascending=False).reset_index(drop=True)
    return grouped


def summarize_binned_reference_improvement(
    prediction_frame: pd.DataFrame,
    group_column: str,
    label_order: list[str],
    reference_prefix: str,
    target_prefix: str,
) -> pd.DataFrame:
    grouped = summarize_reference_improvement(
        prediction_frame=prediction_frame,
        group_column=group_column,
        reference_prefix=reference_prefix,
        target_prefix=target_prefix,
        min_samples=1,
    )
    grouped["group"] = pd.Categorical(grouped["group"], categories=label_order, ordered=True)
    grouped = grouped.sort_values("group").reset_index(drop=True)
    grouped["group"] = grouped["group"].astype(str)
    grouped = grouped[grouped["group"] != "nan"].reset_index(drop=True)
    return grouped


def select_worst_cases(prediction_frame: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    worst_df = prediction_frame.sort_values("multimodal_abs_error", ascending=False).head(top_n).copy()
    return worst_df[
        [
            LISTING_ID_COLUMN,
            "district",
            "neighborhood",
            "rooms",
            "m2_gross",
            "actual_price_try",
            "clip_prediction",
            "multimodal_prediction",
            "multimodal_residual",
            "multimodal_abs_error",
            "multimodal_ape_pct",
            "abs_error_gain_vs_clip",
        ]
    ].reset_index(drop=True)


def compute_branch_ablations(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
    tabular_dim: int,
    text_dim: int,
    image_dim: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], str]:
    text_start = tabular_dim
    text_end = tabular_dim + text_dim
    image_start = text_end
    image_end = image_start + image_dim

    X_no_text = np.array(X_test, copy=True)
    X_no_text[:, text_start:text_end] = 0.0
    no_text_predictions = np.asarray(model.predict(X_no_text), dtype=float)
    no_text_metrics = calculate_metrics(y_test, no_text_predictions)

    X_no_image = np.array(X_test, copy=True)
    X_no_image[:, image_start:image_end] = 0.0
    no_image_predictions = np.asarray(model.predict(X_no_image), dtype=float)
    no_image_metrics = calculate_metrics(y_test, no_image_predictions)

    X_tabular_only = np.array(X_test, copy=True)
    X_tabular_only[:, text_start:text_end] = 0.0
    X_tabular_only[:, image_start:image_end] = 0.0
    tabular_only_predictions = np.asarray(model.predict(X_tabular_only), dtype=float)
    tabular_only_metrics = calculate_metrics(y_test, tabular_only_predictions)

    note = (
        "Ablasyonlar reduced text ve reduced image bloklarini sifira cekerek olculdu; "
        "boylece tabular blok sabit tutulurken text ve image katkisi ayrica gozlemlendi."
    )
    return no_text_metrics, no_image_metrics, tabular_only_metrics, note


def build_reference_comparison_table(
    baseline_subset_metrics: dict[str, float],
    clip_reference_metrics: dict[str, float],
    final_metrics: dict[str, float],
) -> pd.DataFrame:
    rows = [
        {
            "row": "Matched tabular baseline",
            "mae": baseline_subset_metrics["mae"],
            "rmse": baseline_subset_metrics["rmse"],
            "r2": baseline_subset_metrics["r2"],
            "mape": baseline_subset_metrics["mape"],
        },
        {
            "row": "Historical CLIP best",
            "mae": BEST_CLIP_CAP_REFERENCE["mae"],
            "rmse": BEST_CLIP_CAP_REFERENCE["rmse"],
            "r2": BEST_CLIP_CAP_REFERENCE["r2"],
            "mape": BEST_CLIP_CAP_REFERENCE["mape"],
        },
        {
            "row": "Rebuilt CLIP best",
            "mae": clip_reference_metrics["mae"],
            "rmse": clip_reference_metrics["rmse"],
            "r2": clip_reference_metrics["r2"],
            "mape": clip_reference_metrics["mape"],
        },
        {
            "row": "Historical SigLIP best",
            "mae": BEST_SIGLIP_REFERENCE["mae"],
            "rmse": BEST_SIGLIP_REFERENCE["rmse"],
            "r2": BEST_SIGLIP_REFERENCE["r2"],
            "mape": BEST_SIGLIP_REFERENCE["mape"],
        },
        {
            "row": "Final multimodal text+clip",
            "mae": final_metrics["mae"],
            "rmse": final_metrics["rmse"],
            "r2": final_metrics["r2"],
            "mape": final_metrics["mape"],
        },
        {
            "row": "Improvement vs matched baseline",
            "mae": baseline_subset_metrics["mae"] - final_metrics["mae"],
            "rmse": baseline_subset_metrics["rmse"] - final_metrics["rmse"],
            "r2": final_metrics["r2"] - baseline_subset_metrics["r2"],
            "mape": baseline_subset_metrics["mape"] - final_metrics["mape"],
        },
        {
            "row": "Improvement vs CLIP best",
            "mae": clip_reference_metrics["mae"] - final_metrics["mae"],
            "rmse": clip_reference_metrics["rmse"] - final_metrics["rmse"],
            "r2": final_metrics["r2"] - clip_reference_metrics["r2"],
            "mape": clip_reference_metrics["mape"] - final_metrics["mape"],
        },
    ]
    return pd.DataFrame(rows)


def build_final_commentary(
    best_result: ValidationTrial,
    final_metrics: dict[str, float],
    baseline_subset_metrics: dict[str, float],
    clip_reference_metrics: dict[str, float],
    no_text_metrics: dict[str, float],
    no_image_metrics: dict[str, float],
    tabular_only_metrics: dict[str, float],
) -> str:
    commentary_bits: list[str] = []

    if final_metrics["mae"] < baseline_subset_metrics["mae"]:
        commentary_bits.append(
            f"matched tabular baseline'a gore MAE tarafinda {baseline_subset_metrics['mae'] - final_metrics['mae']:.2f} TRY iyilesme var"
        )
    else:
        commentary_bits.append(
            f"matched tabular baseline'a gore MAE tarafinda {final_metrics['mae'] - baseline_subset_metrics['mae']:.2f} TRY kotulesme var"
        )

    if final_metrics["mae"] < clip_reference_metrics["mae"]:
        commentary_bits.append(
            f"CLIP referansina gore MAE {clip_reference_metrics['mae'] - final_metrics['mae']:.2f} TRY daha iyi"
        )
    else:
        commentary_bits.append(
            f"CLIP referansina gore MAE {final_metrics['mae'] - clip_reference_metrics['mae']:.2f} TRY daha zayif"
        )

    commentary_bits.append(
        f"en iyi validation kombinasyonu TF-IDF max_features={best_result.text_max_features}, text_svd={best_result.text_svd_dim}, image_pca={best_result.image_pca_dim}, model={best_result.model_label}"
    )

    if no_text_metrics["mae"] > final_metrics["mae"]:
        commentary_bits.append(
            f"text branch ablasyonda MAE {no_text_metrics['mae'] - final_metrics['mae']:.2f} kadar kotulestigi icin text sinyali ek fayda uretiyor"
        )
    else:
        commentary_bits.append(
            f"text branch ablasyonda MAE {final_metrics['mae'] - no_text_metrics['mae']:.2f} kadar iyilestigi icin text block henuz en verimli sekilde kullanilmiyor olabilir"
        )

    if no_image_metrics["mae"] > final_metrics["mae"]:
        commentary_bits.append(
            f"image branch ablasyonda MAE {no_image_metrics['mae'] - final_metrics['mae']:.2f} kadar kotulestigi icin CLIP image sinyali kritik katkida bulunuyor"
        )
    else:
        commentary_bits.append(
            f"image branch ablasyonda MAE {final_metrics['mae'] - no_image_metrics['mae']:.2f} kadar iyilestigi icin image block henuz optimizasyon alani barindiriyor"
        )

    if tabular_only_metrics["mae"] > final_metrics["mae"]:
        commentary_bits.append(
            f"text+image birlikte kaldirildiginda MAE {tabular_only_metrics['mae'] - final_metrics['mae']:.2f} kadar kotulesti"
        )
    else:
        commentary_bits.append(
            f"text+image birlikte kaldirildiginda MAE {final_metrics['mae'] - tabular_only_metrics['mae']:.2f} kadar iyilesti; bu anormal gorunuyorsa config tekrar gozden gecirilmeli"
        )

    return "; ".join(commentary_bits) + "."


def build_report(
    join_count: int,
    embedding_dimension: int,
    best_result: ValidationTrial,
    validation_leaderboard: pd.DataFrame,
    text_config_summary_df: pd.DataFrame,
    image_pca_summary_df: pd.DataFrame,
    final_metrics: dict[str, float],
    baseline_subset_metrics: dict[str, float],
    clip_reference_metrics: dict[str, float],
    comparison_table: pd.DataFrame,
    district_improvement_df: pd.DataFrame,
    price_improvement_df: pd.DataFrame,
    m2_improvement_df: pd.DataFrame,
    no_text_metrics: dict[str, float],
    no_image_metrics: dict[str, float],
    tabular_only_metrics: dict[str, float],
    ablation_note: str,
    worst_cases_df: pd.DataFrame,
    model_tracking: dict[str, dict[str, Any]],
) -> str:
    final_commentary = build_final_commentary(
        best_result=best_result,
        final_metrics=final_metrics,
        baseline_subset_metrics=baseline_subset_metrics,
        clip_reference_metrics=clip_reference_metrics,
        no_text_metrics=no_text_metrics,
        no_image_metrics=no_image_metrics,
        tabular_only_metrics=tabular_only_metrics,
    )

    report_lines = [
        "# Final Multimodal Text+CLIP Results",
        "",
        "## Ozet",
        "",
        f"- Multimodal source: `{MULTIMODAL_DATASET_PATH}`",
        f"- CLIP caps embedding source: `{CLIP_CAPS_EMBEDDINGS_PATH}`",
        f"- Sabit image_cap: **{CLIP_CAP_VALUE}**",
        f"- Sabit image representation: **{CLIP_REPRESENTATION_COLUMN}**",
        f"- Ham CLIP embedding dimension: **{embedding_dimension}**",
        f"- Join sonucu kalan ornek sayisi: **{join_count:,}**",
        f"- Denenen TF-IDF max_features: **{', '.join(str(value) for value in TEXT_TFIDF_MAX_FEATURES)}**",
        f"- Denenen text SVD dim: **{', '.join(str(value) for value in TEXT_SVD_DIMS)}**",
        f"- Denenen image PCA dim: **{', '.join(str(value) for value in IMAGE_PCA_DIMS)}**",
        f"- En iyi kombinasyon: **tfidf={best_result.text_max_features} + text_svd={best_result.text_svd_dim} + image_pca={best_result.image_pca_dim} + {best_result.model_label}**",
        f"- Kaydedilen model bundle: `{MODEL_OUTPUT_PATH}`",
        "",
        "## Validation Leaderboard",
        "",
        dataframe_to_markdown_table(validation_leaderboard, digits=4),
        "",
        "## Text Feature Config Comparison",
        "",
        dataframe_to_markdown_table(text_config_summary_df, digits=4),
        "",
        "## Image PCA Comparison",
        "",
        dataframe_to_markdown_table(image_pca_summary_df, digits=4),
        "",
        "## Final Test Sonuclari",
        "",
        render_markdown_table(
            ["Metric", "Value"],
            [
                ["MAE", format_float(final_metrics["mae"])],
                ["RMSE", format_float(final_metrics["rmse"])],
                ["R2", format_float(final_metrics["r2"], digits=4)],
                ["MAPE (%)", format_float(final_metrics["mape"])],
            ],
        ),
        "",
        "## Baseline vs CLIP vs Multimodal Karsilastirmasi",
        "",
        dataframe_to_markdown_table(comparison_table, digits=4),
        "",
        "## District Bazli Improvement (CLIP vs Multimodal)",
        "",
        dataframe_to_markdown_table(district_improvement_df, digits=4),
        "",
        "## Price Range Bazli Improvement (CLIP vs Multimodal)",
        "",
        dataframe_to_markdown_table(price_improvement_df, digits=4),
        "",
        "## m2 Range Bazli Improvement (CLIP vs Multimodal)",
        "",
        dataframe_to_markdown_table(m2_improvement_df, digits=4),
        "",
        "## Text Branch Ablation",
        "",
        f"- Final multimodal MAE: **{final_metrics['mae']:.2f}**",
        f"- Text branch sifirlaninca MAE: **{no_text_metrics['mae']:.2f}**",
        f"- Text branch sifirlaninca RMSE: **{no_text_metrics['rmse']:.2f}**",
        f"- Text branch sifirlaninca R2: **{no_text_metrics['r2']:.4f}**",
        f"- Text branch sifirlaninca MAPE: **{no_text_metrics['mape']:.2f}%**",
        "",
        "## Image Branch Ablation",
        "",
        f"- Image branch sifirlaninca MAE: **{no_image_metrics['mae']:.2f}**",
        f"- Image branch sifirlaninca RMSE: **{no_image_metrics['rmse']:.2f}**",
        f"- Image branch sifirlaninca R2: **{no_image_metrics['r2']:.4f}**",
        f"- Image branch sifirlaninca MAPE: **{no_image_metrics['mape']:.2f}%**",
        "",
        "## Text+Image Birlikte Ablation",
        "",
        f"- Text+image birlikte sifirlaninca MAE: **{tabular_only_metrics['mae']:.2f}**",
        f"- Text+image birlikte sifirlaninca RMSE: **{tabular_only_metrics['rmse']:.2f}**",
        f"- Text+image birlikte sifirlaninca R2: **{tabular_only_metrics['r2']:.4f}**",
        f"- Text+image birlikte sifirlaninca MAPE: **{tabular_only_metrics['mape']:.2f}%**",
        f"- Ablation notu: {ablation_note}",
        "",
        "## En Yuksek Hata Yapan 20 Ilan",
        "",
        dataframe_to_markdown_table(worst_cases_df, digits=4),
        "",
        "## Model Durumu",
        "",
        render_model_status(model_tracking),
        "",
        "## Sonuc Yorumu",
        "",
        f"- {final_commentary}",
    ]
    return "\n".join(report_lines)


def save_model_bundle(
    best_result: ValidationTrial,
    tabular_preprocessor: Any,
    text_vectorizer: TfidfVectorizer,
    text_svd: TruncatedSVD,
    image_processor: Any,
    model: Any,
    feature_names: list[str],
    join_count: int,
    embedding_dimension: int,
    final_metrics: dict[str, float],
    baseline_subset_metrics: dict[str, float],
    clip_reference_metrics: dict[str, float],
    model_tracking: dict[str, dict[str, Any]],
) -> None:
    bundle = {
        "model_name": best_result.model_label,
        "image_cap": CLIP_CAP_VALUE,
        "image_representation": CLIP_REPRESENTATION_COLUMN,
        "text_max_features": best_result.text_max_features,
        "text_svd_dim": best_result.text_svd_dim,
        "image_pca_dim": best_result.image_pca_dim,
        "tabular_preprocessor": tabular_preprocessor,
        "text_vectorizer": text_vectorizer,
        "text_svd": text_svd,
        "image_processor": image_processor,
        "regressor": model,
        "feature_names": feature_names,
        "tabular_feature_columns": FUSION_TABULAR_COLUMNS,
        "text_columns": [TITLE_COLUMN, DESCRIPTION_COLUMN],
        "join_count": join_count,
        "embedding_dimension": embedding_dimension,
        "final_metrics": final_metrics,
        "baseline_subset_metrics": baseline_subset_metrics,
        "clip_reference_metrics": clip_reference_metrics,
        "reference_metrics": {
            "matched_tabular_baseline": MATCHED_BASELINE_REFERENCE,
            "best_clip_cap": BEST_CLIP_CAP_REFERENCE,
            "best_siglip": BEST_SIGLIP_REFERENCE,
        },
        "boilerplate_patterns": BOILERPLATE_PATTERNS,
        "model_tracking": model_tracking,
    }
    LOGGER.info("Model bundle kaydediliyor: %s", MODEL_OUTPUT_PATH)
    joblib.dump(bundle, MODEL_OUTPUT_PATH)


def main() -> int:
    configure_logging()
    ensure_output_directories()

    base_df, clip_store, embedding_dimension = load_base_multimodal_and_clip_embeddings(
        multimodal_path=MULTIMODAL_DATASET_PATH,
        clip_caps_path=CLIP_CAPS_EMBEDDINGS_PATH,
    )
    splits = split_dataset(base_df)

    train_df = splits.train.copy()
    validation_df = splits.validation.copy()
    test_df = splits.test.copy()

    y_train = train_df[TARGET_COLUMN]
    y_validation = validation_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    train_listing_ids = train_df[LISTING_ID_COLUMN].astype(str).tolist()
    validation_listing_ids = validation_df[LISTING_ID_COLUMN].astype(str).tolist()
    test_listing_ids = test_df[LISTING_ID_COLUMN].astype(str).tolist()

    LOGGER.info("Tabular preprocessor fit ediliyor...")
    train_tabular = clean_tabular_features(train_df)
    validation_tabular = clean_tabular_features(validation_df)
    validation_tabular_preprocessor = build_tabular_preprocessor()
    X_train_tabular = np.asarray(
        validation_tabular_preprocessor.fit_transform(train_tabular),
        dtype=np.float32,
    )
    X_validation_tabular = np.asarray(
        validation_tabular_preprocessor.transform(validation_tabular),
        dtype=np.float32,
    )
    LOGGER.info("Donusmus tabular feature sayisi: %s", X_train_tabular.shape[1])

    LOGGER.info("Text kolonlari temizleniyor ve cache hazirlaniyor...")
    train_text = build_clean_text_columns(train_df)["combined_text"]
    validation_text = build_clean_text_columns(validation_df)["combined_text"]
    text_cache = build_text_cache(train_text=train_text, validation_text=validation_text)

    LOGGER.info("Image PCA cache hazirlaniyor...")
    image_cache = build_image_cache(
        train_listing_ids=train_listing_ids,
        validation_listing_ids=validation_listing_ids,
        clip_store=clip_store,
    )

    candidates, model_tracking = build_model_factories()
    validation_trials = train_validation_trials(
        candidates=candidates,
        X_train_tabular=X_train_tabular,
        X_validation_tabular=X_validation_tabular,
        text_cache=text_cache,
        image_cache=image_cache,
        y_train=y_train,
        y_validation=y_validation,
        model_tracking=model_tracking,
    )
    best_result = select_best_trial(validation_trials)
    validation_leaderboard = build_validation_leaderboard(validation_trials)
    text_config_summary_df = build_text_config_summary(validation_leaderboard)
    image_pca_summary_df = build_image_pca_summary(validation_leaderboard)

    LOGGER.info(
        "Final model yeniden egitiliyor | tfidf=%s | text_svd=%s | image_pca=%s | %s",
        best_result.text_max_features,
        best_result.text_svd_dim,
        best_result.image_pca_dim,
        best_result.model_label,
    )
    train_validation_df = pd.concat([train_df, validation_df], axis=0, ignore_index=True)
    y_train_validation = train_validation_df[TARGET_COLUMN]
    train_validation_listing_ids = train_validation_df[LISTING_ID_COLUMN].astype(str).tolist()

    train_validation_tabular = clean_tabular_features(train_validation_df)
    test_tabular = clean_tabular_features(test_df)
    final_tabular_preprocessor = build_tabular_preprocessor()
    X_train_validation_tabular = np.asarray(
        final_tabular_preprocessor.fit_transform(train_validation_tabular),
        dtype=np.float32,
    )
    X_test_tabular = np.asarray(
        final_tabular_preprocessor.transform(test_tabular),
        dtype=np.float32,
    )

    train_validation_text = build_clean_text_columns(train_validation_df)["combined_text"]
    test_text = build_clean_text_columns(test_df)["combined_text"]
    final_text_info = fit_text_processor_for_final(
        train_validation_text=train_validation_text,
        test_text=test_text,
        max_features=best_result.text_max_features,
        requested_svd_dim=best_result.text_svd_dim,
    )

    final_image_info = fit_image_processor_for_final(
        train_validation_listing_ids=train_validation_listing_ids,
        test_listing_ids=test_listing_ids,
        clip_store=clip_store,
        image_pca_dim=best_result.image_pca_dim,
    )

    X_train_validation_fused = concatenate_multimodal_blocks(
        X_train_validation_tabular,
        final_text_info["X_train_validation_text"],
        final_image_info["X_train_validation_image"],
    )
    X_test_fused = concatenate_multimodal_blocks(
        X_test_tabular,
        final_text_info["X_test_text"],
        final_image_info["X_test_image"],
    )

    final_model = best_result.factory()
    final_model.fit(X_train_validation_fused, y_train_validation)
    multimodal_predictions = np.asarray(final_model.predict(X_test_fused), dtype=float)
    final_metrics = calculate_metrics(y_test, multimodal_predictions)
    LOGGER.info(
        "Final test | tfidf=%s | text_svd=%s | image_pca=%s | %s | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
        best_result.text_max_features,
        best_result.text_svd_dim,
        best_result.image_pca_dim,
        best_result.model_label,
        final_metrics["mae"],
        final_metrics["rmse"],
        final_metrics["r2"],
        final_metrics["mape"],
    )

    LOGGER.info("Matched tabular baseline yeniden egitiliyor...")
    baseline_predictions, baseline_subset_metrics = train_matched_tabular_baseline(
        train_validation_df=train_validation_df,
        test_df=test_df,
    )

    LOGGER.info("Best CLIP referansi ayni splitte yeniden egitiliyor...")
    clip_reference_factory = build_clip_reference_factory()
    clip_reference_image_info = fit_image_processor_for_final(
        train_validation_listing_ids=train_validation_listing_ids,
        test_listing_ids=test_listing_ids,
        clip_store=clip_store,
        image_pca_dim=16,
    )
    X_train_validation_clip = concatenate_multimodal_blocks(
        X_train_validation_tabular,
        clip_reference_image_info["X_train_validation_image"],
    )
    X_test_clip = concatenate_multimodal_blocks(
        X_test_tabular,
        clip_reference_image_info["X_test_image"],
    )
    clip_reference_model = clip_reference_factory()
    clip_reference_model.fit(X_train_validation_clip, y_train_validation)
    clip_predictions = np.asarray(clip_reference_model.predict(X_test_clip), dtype=float)
    clip_reference_metrics = calculate_metrics(y_test, clip_predictions)

    prediction_frame = build_prediction_frame(
        test_df=test_df,
        baseline_predictions=baseline_predictions,
        clip_predictions=clip_predictions,
        multimodal_predictions=multimodal_predictions,
    )
    district_improvement_df = summarize_reference_improvement(
        prediction_frame=prediction_frame,
        group_column="district",
        reference_prefix="clip",
        target_prefix="multimodal",
        min_samples=5,
    )
    price_improvement_df = summarize_binned_reference_improvement(
        prediction_frame=prediction_frame,
        group_column="price_range",
        label_order=PRICE_BIN_LABELS,
        reference_prefix="clip",
        target_prefix="multimodal",
    )
    m2_improvement_df = summarize_binned_reference_improvement(
        prediction_frame=prediction_frame,
        group_column="m2_range",
        label_order=M2_BIN_LABELS,
        reference_prefix="clip",
        target_prefix="multimodal",
    )
    worst_cases_df = select_worst_cases(prediction_frame, top_n=20)

    no_text_metrics, no_image_metrics, tabular_only_metrics, ablation_note = compute_branch_ablations(
        model=final_model,
        X_test=X_test_fused,
        y_test=y_test,
        tabular_dim=int(X_test_tabular.shape[1]),
        text_dim=int(final_text_info["X_test_text"].shape[1]),
        image_dim=int(final_image_info["X_test_image"].shape[1]),
    )

    comparison_table = build_reference_comparison_table(
        baseline_subset_metrics=baseline_subset_metrics,
        clip_reference_metrics=clip_reference_metrics,
        final_metrics=final_metrics,
    )
    report_text = build_report(
        join_count=len(base_df),
        embedding_dimension=embedding_dimension,
        best_result=best_result,
        validation_leaderboard=validation_leaderboard,
        text_config_summary_df=text_config_summary_df,
        image_pca_summary_df=image_pca_summary_df,
        final_metrics=final_metrics,
        baseline_subset_metrics=baseline_subset_metrics,
        clip_reference_metrics=clip_reference_metrics,
        comparison_table=comparison_table,
        district_improvement_df=district_improvement_df,
        price_improvement_df=price_improvement_df,
        m2_improvement_df=m2_improvement_df,
        no_text_metrics=no_text_metrics,
        no_image_metrics=no_image_metrics,
        tabular_only_metrics=tabular_only_metrics,
        ablation_note=ablation_note,
        worst_cases_df=worst_cases_df,
        model_tracking=model_tracking,
    )
    LOGGER.info("Rapor kaydediliyor: %s", REPORT_OUTPUT_PATH)
    REPORT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")

    tabular_feature_names = [
        str(name)
        for name in final_tabular_preprocessor.get_feature_names_out()
    ]
    text_feature_names = [
        f"{TEXT_SVD_PREFIX}{index:03d}"
        for index in range(int(final_text_info["X_train_validation_text"].shape[1]))
    ]
    image_feature_names = [
        f"{IMAGE_REDUCED_PREFIX}{index:03d}"
        for index in range(int(final_image_info["X_train_validation_image"].shape[1]))
    ]
    save_model_bundle(
        best_result=best_result,
        tabular_preprocessor=final_tabular_preprocessor,
        text_vectorizer=final_text_info["vectorizer"],
        text_svd=final_text_info["svd"],
        image_processor=final_image_info["image_processor"],
        model=final_model,
        feature_names=tabular_feature_names + text_feature_names + image_feature_names,
        join_count=len(base_df),
        embedding_dimension=embedding_dimension,
        final_metrics=final_metrics,
        baseline_subset_metrics=baseline_subset_metrics,
        clip_reference_metrics=clip_reference_metrics,
        model_tracking=model_tracking,
    )

    LOGGER.info("Final multimodal text+clip egitimi tamamlandi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
