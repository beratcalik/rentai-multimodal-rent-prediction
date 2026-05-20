from __future__ import annotations

import logging
import sys
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
    USED_IMAGE_COUNT_LABELS,
    build_image_processor,
    build_model_factories,
    build_prediction_frame,
    build_tabular_preprocessor,
    render_model_status,
)
from train_image_fusion_reduced import (
    FUSION_TABULAR_COLUMNS,
    M2_BIN_LABELS,
    PRICE_BIN_LABELS,
    clean_tabular_features,
    compute_reduced_image_ablation,
    concatenate_feature_blocks,
    dataframe_to_markdown_table,
    format_float,
    render_markdown_table,
    select_worst_cases,
    summarize_binned_improvement,
    summarize_group_improvement,
    train_matched_tabular_baseline,
)


LOGGER = logging.getLogger("clip_fusion_caps_ablation_training")

ROOT_DIR = Path(__file__).resolve().parent.parent
MULTIMODAL_DATASET_PATH = ROOT_DIR / "dataset" / "train_ready_multimodal.parquet"
CLIP_CAPS_EMBEDDINGS_PATH = ROOT_DIR / "dataset" / "clip_image_embeddings_caps.parquet"
MODEL_OUTPUT_PATH = ROOT_DIR / "models" / "clip_fusion_caps_ablation_model.joblib"
REPORT_OUTPUT_PATH = ROOT_DIR / "reports" / "clip_fusion_caps_ablation_results.md"

LISTING_ID_COLUMN = "listing_id"
IMAGE_CAP_COLUMN = "image_cap"
USED_IMAGE_COUNT_COLUMN = "used_image_count"
REDUCER_NAME = "PCA"
PRIMARY_SELECTION_METRIC = "mae"
REDUCED_IMAGE_PREFIX = "clip_caps_reduced_"

CAP_VALUES = [4, 8, 12, 16]
REDUCED_IMAGE_DIMS = [16, 32, 64]

REPRESENTATION_COLUMNS = {
    "clip_mean_embedding": "clip_mean_embedding",
    "clip_max_embedding": "clip_max_embedding",
    "clip_meanmax_embedding": "clip_meanmax_embedding",
}
REPRESENTATION_PREFIXES = {
    "clip_mean_embedding": "clip_caps_mean_emb_",
    "clip_max_embedding": "clip_caps_max_emb_",
    "clip_meanmax_embedding": "clip_caps_meanmax_emb_",
}

MATCHED_BASELINE_REFERENCE = {
    "mae": 4700.08,
    "rmse": 7057.76,
    "r2": 0.8013,
    "mape": 13.79,
}
PREVIOUS_CLIP_REFERENCE = {
    "image_cap": 16,
    "representation": "clip_mean_embedding",
    "image_dim": 16,
    "model": "XGBRegressor",
    "mae": 4381.56,
    "rmse": 6359.81,
    "r2": 0.8387,
    "mape": 13.10,
}


@dataclass
class ValidationTrial:
    image_cap: int
    representation_name: str
    model_label: str
    image_dim: int
    validation_metrics: dict[str, float]
    factory: Callable[[], Any]


def ensure_output_directories() -> None:
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_base_multimodal_and_caps(
    multimodal_path: Path,
    caps_embeddings_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not multimodal_path.exists():
        raise FileNotFoundError(f"Multimodal dataset bulunamadi: {multimodal_path}")
    if not caps_embeddings_path.exists():
        raise FileNotFoundError(f"CLIP caps embedding parquet bulunamadi: {caps_embeddings_path}")

    LOGGER.info("Multimodal dataset okunuyor: %s", multimodal_path)
    multimodal_df = pd.read_parquet(multimodal_path)
    LOGGER.info("CLIP caps embeddings okunuyor: %s", caps_embeddings_path)
    caps_df = pd.read_parquet(caps_embeddings_path)

    required_multimodal_columns = set(FUSION_TABULAR_COLUMNS + [TARGET_COLUMN, LISTING_ID_COLUMN])
    missing_multimodal_columns = sorted(required_multimodal_columns - set(multimodal_df.columns))
    if missing_multimodal_columns:
        raise ValueError(f"Multimodal dataset icinde eksik kolonlar bulundu: {missing_multimodal_columns}")

    required_embedding_columns = {
        LISTING_ID_COLUMN,
        IMAGE_CAP_COLUMN,
        USED_IMAGE_COUNT_COLUMN,
        *REPRESENTATION_COLUMNS.values(),
    }
    missing_embedding_columns = sorted(required_embedding_columns - set(caps_df.columns))
    if missing_embedding_columns:
        raise ValueError(f"CLIP caps embedding dataset icinde eksik kolonlar bulundu: {missing_embedding_columns}")

    caps_df = caps_df[caps_df[IMAGE_CAP_COLUMN].isin(CAP_VALUES)].copy()
    if caps_df.empty:
        raise ValueError("CLIP caps embedding dataset icinde hedef image_cap degerleri bulunamadi.")

    listing_cap_counts = (
        caps_df.groupby(LISTING_ID_COLUMN)[IMAGE_CAP_COLUMN]
        .nunique()
    )
    valid_listing_ids = set(listing_cap_counts[listing_cap_counts == len(CAP_VALUES)].index.astype(str))
    if not valid_listing_ids:
        raise ValueError("Tum cap degerleri icin ortak listing bulunamadi.")

    multimodal_df[LISTING_ID_COLUMN] = multimodal_df[LISTING_ID_COLUMN].astype(str)
    base_df = multimodal_df[multimodal_df[LISTING_ID_COLUMN].isin(valid_listing_ids)].copy()
    base_df = base_df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    caps_df[LISTING_ID_COLUMN] = caps_df[LISTING_ID_COLUMN].astype(str)
    caps_df = caps_df[caps_df[LISTING_ID_COLUMN].isin(valid_listing_ids)].reset_index(drop=True)

    LOGGER.info("Joinable listing sayisi: %s", len(base_df))
    return base_df, caps_df


def build_cap_store(
    caps_df: pd.DataFrame,
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    cap_store: dict[int, dict[str, Any]] = {}
    representation_dimensions: dict[str, int] = {}

    for image_cap in CAP_VALUES:
        cap_df = caps_df[caps_df[IMAGE_CAP_COLUMN] == image_cap].copy()
        cap_df = cap_df.drop_duplicates(subset=[LISTING_ID_COLUMN], keep="first").reset_index(drop=True)
        if cap_df.empty:
            raise ValueError(f"image_cap={image_cap} icin hic embedding satiri bulunamadi.")

        listing_ids = cap_df[LISTING_ID_COLUMN].astype(str).tolist()
        index_by_listing = {listing_id: index for index, listing_id in enumerate(listing_ids)}
        used_image_counts = cap_df[USED_IMAGE_COUNT_COLUMN].astype(int).to_numpy(dtype=np.int32, copy=False)

        representation_matrices: dict[str, np.ndarray] = {}
        for representation_name, source_column in REPRESENTATION_COLUMNS.items():
            vectors: list[np.ndarray] = []
            vector_dim: int | None = None

            for value in cap_df[source_column]:
                vector = np.asarray(value, dtype=np.float32).ravel()
                if vector_dim is None:
                    vector_dim = int(len(vector))
                elif len(vector) != vector_dim:
                    raise ValueError(
                        f"{source_column} kolonunda tutarsiz vektor boyutu bulundu (image_cap={image_cap})."
                    )
                vectors.append(vector)

            if vector_dim is None:
                raise ValueError(f"{source_column} kolonunda hic vektor bulunamadi (image_cap={image_cap}).")

            previous_dim = representation_dimensions.get(representation_name)
            if previous_dim is None:
                representation_dimensions[representation_name] = vector_dim
            elif previous_dim != vector_dim:
                raise ValueError(
                    f"{representation_name} icin cap bazinda tutarsiz embedding dimension bulundu: "
                    f"{previous_dim} vs {vector_dim}."
                )

            representation_matrices[representation_name] = np.vstack(vectors).astype(np.float32, copy=False)

        cap_store[image_cap] = {
            "listing_ids": listing_ids,
            "index_by_listing": index_by_listing,
            "used_image_counts": used_image_counts,
            "representation_matrices": representation_matrices,
        }

    return cap_store, representation_dimensions


def extract_indices_for_split(cap_info: dict[str, Any], split_listing_ids: list[str]) -> np.ndarray:
    index_by_listing = cap_info["index_by_listing"]
    try:
        return np.asarray([index_by_listing[listing_id] for listing_id in split_listing_ids], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(f"Split icinde embedding bulunamayan listing bulundu: {exc}") from exc


def extract_image_matrix(
    cap_info: dict[str, Any],
    representation_name: str,
    split_listing_ids: list[str],
) -> np.ndarray:
    indices = extract_indices_for_split(cap_info, split_listing_ids)
    matrix = cap_info["representation_matrices"][representation_name][indices]
    return np.asarray(matrix, dtype=np.float32)


def extract_used_image_count(
    cap_info: dict[str, Any],
    split_listing_ids: list[str],
) -> np.ndarray:
    indices = extract_indices_for_split(cap_info, split_listing_ids)
    counts = cap_info["used_image_counts"][indices]
    return np.asarray(counts, dtype=np.int32)


def train_validation_trials(
    candidates: list[dict[str, Any]],
    X_train_tabular: np.ndarray,
    X_validation_tabular: np.ndarray,
    y_train: pd.Series,
    y_validation: pd.Series,
    train_listing_ids: list[str],
    validation_listing_ids: list[str],
    cap_store: dict[int, dict[str, Any]],
    model_tracking: dict[str, dict[str, Any]],
) -> list[ValidationTrial]:
    results: list[ValidationTrial] = []

    for image_cap in CAP_VALUES:
        cap_info = cap_store[image_cap]

        for representation_name in REPRESENTATION_COLUMNS:
            train_image_matrix = extract_image_matrix(
                cap_info=cap_info,
                representation_name=representation_name,
                split_listing_ids=train_listing_ids,
            )
            validation_image_matrix = extract_image_matrix(
                cap_info=cap_info,
                representation_name=representation_name,
                split_listing_ids=validation_listing_ids,
            )

            for image_dim in REDUCED_IMAGE_DIMS:
                LOGGER.info(
                    "%s reduction fit ediliyor | image_cap=%s | representation=%s | image_dim=%s",
                    REDUCER_NAME,
                    image_cap,
                    representation_name,
                    image_dim,
                )
                image_processor = build_image_processor(image_dim)
                X_train_image = image_processor.fit_transform(train_image_matrix)
                X_validation_image = image_processor.transform(validation_image_matrix)

                X_train_fused = concatenate_feature_blocks(X_train_tabular, X_train_image)
                X_validation_fused = concatenate_feature_blocks(X_validation_tabular, X_validation_image)

                for candidate in candidates:
                    model_label = candidate["label"]
                    LOGGER.info(
                        "Model egitiliyor | image_cap=%s | representation=%s | image_dim=%s | %s",
                        image_cap,
                        representation_name,
                        image_dim,
                        model_label,
                    )
                    try:
                        model = candidate["factory"]()
                        model.fit(X_train_fused, y_train)
                        validation_predictions = np.asarray(model.predict(X_validation_fused), dtype=float)
                        validation_metrics = calculate_metrics(y_validation, validation_predictions)
                        results.append(
                            ValidationTrial(
                                image_cap=image_cap,
                                representation_name=representation_name,
                                model_label=model_label,
                                image_dim=image_dim,
                                validation_metrics=validation_metrics,
                                factory=candidate["factory"],
                            )
                        )
                        LOGGER.info(
                            "Validation | cap=%s | %s | image_dim=%s | %s | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
                            image_cap,
                            representation_name,
                            image_dim,
                            model_label,
                            validation_metrics["mae"],
                            validation_metrics["rmse"],
                            validation_metrics["r2"],
                            validation_metrics["mape"],
                        )
                        model_tracking[model_label]["success_combinations"].append(
                            f"cap={image_cap}|{representation_name}@{image_dim}"
                        )
                    except Exception as exc:  # pragma: no cover - defensive path
                        LOGGER.exception(
                            "Model denemesi basarisiz oldu | image_cap=%s | representation=%s | image_dim=%s | %s",
                            image_cap,
                            representation_name,
                            image_dim,
                            model_label,
                        )
                        model_tracking[model_label]["failure_notes"].append(
                            f"cap={image_cap}|{representation_name}@{image_dim}: {exc.__class__.__name__}: {exc}"
                        )

    if not results:
        raise RuntimeError("Hicbir CLIP cap fusion modeli basariyla egitilemedi.")

    return results


def select_best_trial(results: list[ValidationTrial]) -> ValidationTrial:
    best_result = min(results, key=lambda item: item.validation_metrics[PRIMARY_SELECTION_METRIC])
    LOGGER.info(
        "En iyi kombinasyon secildi: cap=%s + %s + %s + image_dim=%s (validation MAE=%.2f)",
        best_result.image_cap,
        best_result.representation_name,
        best_result.model_label,
        best_result.image_dim,
        best_result.validation_metrics["mae"],
    )
    return best_result


def build_validation_leaderboard(results: list[ValidationTrial]) -> pd.DataFrame:
    leaderboard_df = pd.DataFrame(
        [
            {
                "image_cap": result.image_cap,
                "representation": result.representation_name,
                "image_dim": result.image_dim,
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
        by=["validation_mae", "validation_rmse", "image_cap", "representation", "image_dim", "model"],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)


def build_cap_summary(validation_leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image_cap in CAP_VALUES:
        subset = validation_leaderboard[validation_leaderboard["image_cap"] == image_cap]
        if subset.empty:
            continue
        best_row = subset.iloc[0]
        rows.append(
            {
                "image_cap": image_cap,
                "best_representation": best_row["representation"],
                "best_model": best_row["model"],
                "best_image_dim": int(best_row["image_dim"]),
                "best_validation_mae": float(best_row["validation_mae"]),
                "best_validation_rmse": float(best_row["validation_rmse"]),
                "best_validation_r2": float(best_row["validation_r2"]),
                "best_validation_mape": float(best_row["validation_mape"]),
            }
        )
    return pd.DataFrame(rows)


def build_representation_summary(validation_leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for representation_name in REPRESENTATION_COLUMNS:
        subset = validation_leaderboard[validation_leaderboard["representation"] == representation_name]
        if subset.empty:
            continue
        best_row = subset.iloc[0]
        rows.append(
            {
                "representation": representation_name,
                "best_image_cap": int(best_row["image_cap"]),
                "best_model": best_row["model"],
                "best_image_dim": int(best_row["image_dim"]),
                "best_validation_mae": float(best_row["validation_mae"]),
                "best_validation_rmse": float(best_row["validation_rmse"]),
                "best_validation_r2": float(best_row["validation_r2"]),
                "best_validation_mape": float(best_row["validation_mape"]),
            }
        )
    return pd.DataFrame(rows)


def build_image_dim_summary(validation_leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image_dim in REDUCED_IMAGE_DIMS:
        subset = validation_leaderboard[validation_leaderboard["image_dim"] == image_dim]
        if subset.empty:
            continue
        best_row = subset.iloc[0]
        rows.append(
            {
                "image_dim": image_dim,
                "best_image_cap": int(best_row["image_cap"]),
                "best_representation": best_row["representation"],
                "best_model": best_row["model"],
                "best_validation_mae": float(best_row["validation_mae"]),
                "best_validation_rmse": float(best_row["validation_rmse"]),
                "best_validation_r2": float(best_row["validation_r2"]),
                "best_validation_mape": float(best_row["validation_mape"]),
            }
        )
    return pd.DataFrame(rows)


def build_reference_comparison_table(
    baseline_subset_metrics: dict[str, float],
    cap_metrics: dict[str, float],
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
            "row": "Previous CLIP best",
            "mae": PREVIOUS_CLIP_REFERENCE["mae"],
            "rmse": PREVIOUS_CLIP_REFERENCE["rmse"],
            "r2": PREVIOUS_CLIP_REFERENCE["r2"],
            "mape": PREVIOUS_CLIP_REFERENCE["mape"],
        },
        {
            "row": "Best CLIP cap model",
            "mae": cap_metrics["mae"],
            "rmse": cap_metrics["rmse"],
            "r2": cap_metrics["r2"],
            "mape": cap_metrics["mape"],
        },
        {
            "row": "Improvement vs matched baseline",
            "mae": baseline_subset_metrics["mae"] - cap_metrics["mae"],
            "rmse": baseline_subset_metrics["rmse"] - cap_metrics["rmse"],
            "r2": cap_metrics["r2"] - baseline_subset_metrics["r2"],
            "mape": baseline_subset_metrics["mape"] - cap_metrics["mape"],
        },
        {
            "row": "Improvement vs previous CLIP best",
            "mae": PREVIOUS_CLIP_REFERENCE["mae"] - cap_metrics["mae"],
            "rmse": PREVIOUS_CLIP_REFERENCE["rmse"] - cap_metrics["rmse"],
            "r2": cap_metrics["r2"] - PREVIOUS_CLIP_REFERENCE["r2"],
            "mape": PREVIOUS_CLIP_REFERENCE["mape"] - cap_metrics["mape"],
        },
    ]
    return pd.DataFrame(rows)


def build_final_commentary(
    best_result: ValidationTrial,
    cap_metrics: dict[str, float],
    baseline_subset_metrics: dict[str, float],
    image_ablation_metrics: dict[str, float],
) -> str:
    comparison_bits: list[str] = []

    if cap_metrics["mae"] < baseline_subset_metrics["mae"]:
        comparison_bits.append(
            f"matched tabular baseline'a gore MAE tarafinda {baseline_subset_metrics['mae'] - cap_metrics['mae']:.2f} TRY iyilesme var"
        )
    else:
        comparison_bits.append(
            f"matched tabular baseline'a gore MAE tarafinda {cap_metrics['mae'] - baseline_subset_metrics['mae']:.2f} TRY kotulesme var"
        )

    if cap_metrics["mae"] < PREVIOUS_CLIP_REFERENCE["mae"]:
        comparison_bits.append(
            f"onceki CLIP best referansina gore MAE {PREVIOUS_CLIP_REFERENCE['mae'] - cap_metrics['mae']:.2f} TRY daha iyi"
        )
    else:
        comparison_bits.append(
            f"onceki CLIP best referansina gore MAE {cap_metrics['mae'] - PREVIOUS_CLIP_REFERENCE['mae']:.2f} TRY daha zayif"
        )

    if best_result.image_cap != PREVIOUS_CLIP_REFERENCE["image_cap"]:
        comparison_bits.append(
            f"en iyi validation kombinasyonu image_cap={best_result.image_cap} ile geldi; bu, ilk {best_result.image_cap} gorselin 16 cap'e gore daha faydali oldugunu gosteriyor olabilir"
        )
    else:
        comparison_bits.append(
            "en iyi validation kombinasyonu yine image_cap=16 ile geldi; yani daha dusuk cap'ler mevcut CLIP kurulumunu gecemedi"
        )

    if image_ablation_metrics["mae"] > cap_metrics["mae"]:
        comparison_bits.append(
            f"image branch ablasyonda MAE {image_ablation_metrics['mae'] - cap_metrics['mae']:.2f} kadar kotulestigi icin CLIP image sinyali modele net pozitif katki veriyor"
        )
    else:
        comparison_bits.append(
            f"ablasyonda MAE {cap_metrics['mae'] - image_ablation_metrics['mae']:.2f} kadar iyilestigi icin image block henuz en verimli sekilde kullanilmiyor olabilir"
        )

    return "; ".join(comparison_bits) + "."


def build_report(
    join_count: int,
    representation_dimensions: dict[str, int],
    best_result: ValidationTrial,
    validation_leaderboard: pd.DataFrame,
    cap_summary_df: pd.DataFrame,
    representation_summary_df: pd.DataFrame,
    image_dim_summary_df: pd.DataFrame,
    cap_metrics: dict[str, float],
    baseline_subset_metrics: dict[str, float],
    comparison_table: pd.DataFrame,
    district_improvement_df: pd.DataFrame,
    price_improvement_df: pd.DataFrame,
    m2_improvement_df: pd.DataFrame,
    used_image_count_df: pd.DataFrame,
    image_ablation_metrics: dict[str, float],
    image_ablation_note: str,
    worst_cases_df: pd.DataFrame,
    model_tracking: dict[str, dict[str, Any]],
) -> str:
    final_commentary = build_final_commentary(
        best_result=best_result,
        cap_metrics=cap_metrics,
        baseline_subset_metrics=baseline_subset_metrics,
        image_ablation_metrics=image_ablation_metrics,
    )

    representation_dimension_text = ", ".join(
        f"{name}={dimension}"
        for name, dimension in representation_dimensions.items()
    )

    report_lines = [
        "# CLIP Fusion Caps Ablation Results",
        "",
        "## Ozet",
        "",
        f"- Multimodal source: `{MULTIMODAL_DATASET_PATH}`",
        f"- CLIP caps embedding source: `{CLIP_CAPS_EMBEDDINGS_PATH}`",
        f"- Kaydedilen model bundle: `{MODEL_OUTPUT_PATH}`",
        f"- Join sonucu kalan ornek sayisi: **{join_count:,}**",
        f"- Denenen image_cap degerleri: **{', '.join(str(value) for value in CAP_VALUES)}**",
        f"- Denenen representationlar: **{', '.join(REPRESENTATION_COLUMNS.keys())}**",
        f"- Representation dimensionlari: **{representation_dimension_text}**",
        f"- Reducer: **{REDUCER_NAME}**",
        f"- Denenen image_dim degerleri: **{', '.join(str(value) for value in REDUCED_IMAGE_DIMS)}**",
        f"- En iyi kombinasyon: **image_cap={best_result.image_cap} + {best_result.representation_name} + {best_result.model_label} + image_dim={best_result.image_dim}**",
        "",
        "## Validation Leaderboard",
        "",
        dataframe_to_markdown_table(validation_leaderboard, digits=4),
        "",
        "## Cap Bazli Skorlar",
        "",
        dataframe_to_markdown_table(cap_summary_df, digits=4),
        "",
        "## Representation Bazli Skorlar",
        "",
        dataframe_to_markdown_table(representation_summary_df, digits=4),
        "",
        "## Image Dim Bazli Skorlar",
        "",
        dataframe_to_markdown_table(image_dim_summary_df, digits=4),
        "",
        "## Final Test Sonuclari",
        "",
        render_markdown_table(
            ["Metric", "Value"],
            [
                ["MAE", format_float(cap_metrics["mae"])],
                ["RMSE", format_float(cap_metrics["rmse"])],
                ["R2", format_float(cap_metrics["r2"], digits=4)],
                ["MAPE (%)", format_float(cap_metrics["mape"])],
            ],
        ),
        "",
        "## Matched Baseline ve Previous CLIP Best Karsilastirmasi",
        "",
        dataframe_to_markdown_table(comparison_table, digits=4),
        "",
        "## District Bazli Improvement",
        "",
        dataframe_to_markdown_table(district_improvement_df, digits=4),
        "",
        "## Price Range Bazli Improvement",
        "",
        dataframe_to_markdown_table(price_improvement_df, digits=4),
        "",
        "## m2 Range Bazli Improvement",
        "",
        dataframe_to_markdown_table(m2_improvement_df, digits=4),
        "",
        "## Used Image Count Analizi",
        "",
        dataframe_to_markdown_table(used_image_count_df, digits=4),
        "",
        "## Image Branch Ablation",
        "",
        f"- Final best-cap CLIP fusion MAE: **{cap_metrics['mae']:.2f}**",
        f"- Ablation sonrasi MAE: **{image_ablation_metrics['mae']:.2f}**",
        f"- Ablation sonrasi RMSE: **{image_ablation_metrics['rmse']:.2f}**",
        f"- Ablation sonrasi R2: **{image_ablation_metrics['r2']:.4f}**",
        f"- Ablation sonrasi MAPE: **{image_ablation_metrics['mape']:.2f}%**",
        f"- Ablation notu: {image_ablation_note}",
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
    image_processor: Any,
    model: Any,
    feature_names: list[str],
    join_count: int,
    representation_dimensions: dict[str, int],
    cap_metrics: dict[str, float],
    baseline_subset_metrics: dict[str, float],
    model_tracking: dict[str, dict[str, Any]],
) -> None:
    bundle = {
        "model_name": best_result.model_label,
        "image_cap": best_result.image_cap,
        "representation_name": best_result.representation_name,
        "reducer_name": REDUCER_NAME,
        "reduced_image_dim": best_result.image_dim,
        "tabular_preprocessor": tabular_preprocessor,
        "image_processor": image_processor,
        "regressor": model,
        "feature_names": feature_names,
        "tabular_feature_columns": FUSION_TABULAR_COLUMNS,
        "join_count": join_count,
        "representation_dimensions": representation_dimensions,
        "cap_metrics": cap_metrics,
        "baseline_subset_metrics": baseline_subset_metrics,
        "reference_metrics": {
            "matched_tabular_baseline": MATCHED_BASELINE_REFERENCE,
            "previous_clip_best": PREVIOUS_CLIP_REFERENCE,
        },
        "model_tracking": model_tracking,
    }
    LOGGER.info("Model bundle kaydediliyor: %s", MODEL_OUTPUT_PATH)
    joblib.dump(bundle, MODEL_OUTPUT_PATH)


def main() -> int:
    configure_logging()
    ensure_output_directories()

    base_df, caps_df = load_base_multimodal_and_caps(
        multimodal_path=MULTIMODAL_DATASET_PATH,
        caps_embeddings_path=CLIP_CAPS_EMBEDDINGS_PATH,
    )
    cap_store, representation_dimensions = build_cap_store(caps_df)
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

    candidates, model_tracking = build_model_factories()
    validation_trials = train_validation_trials(
        candidates=candidates,
        X_train_tabular=X_train_tabular,
        X_validation_tabular=X_validation_tabular,
        y_train=y_train,
        y_validation=y_validation,
        train_listing_ids=train_listing_ids,
        validation_listing_ids=validation_listing_ids,
        cap_store=cap_store,
        model_tracking=model_tracking,
    )
    best_result = select_best_trial(validation_trials)
    validation_leaderboard = build_validation_leaderboard(validation_trials)
    cap_summary_df = build_cap_summary(validation_leaderboard)
    representation_summary_df = build_representation_summary(validation_leaderboard)
    image_dim_summary_df = build_image_dim_summary(validation_leaderboard)

    LOGGER.info(
        "Final model yeniden egitiliyor | cap=%s | %s | %s | image_dim=%s",
        best_result.image_cap,
        best_result.representation_name,
        best_result.model_label,
        best_result.image_dim,
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

    selected_cap_info = cap_store[best_result.image_cap]
    train_validation_image_matrix = extract_image_matrix(
        cap_info=selected_cap_info,
        representation_name=best_result.representation_name,
        split_listing_ids=train_validation_listing_ids,
    )
    test_image_matrix = extract_image_matrix(
        cap_info=selected_cap_info,
        representation_name=best_result.representation_name,
        split_listing_ids=test_listing_ids,
    )

    final_image_processor = build_image_processor(best_result.image_dim)
    X_train_validation_image = final_image_processor.fit_transform(train_validation_image_matrix)
    X_test_image = final_image_processor.transform(test_image_matrix)

    X_train_validation_fused = concatenate_feature_blocks(
        X_train_validation_tabular,
        X_train_validation_image,
    )
    X_test_fused = concatenate_feature_blocks(
        X_test_tabular,
        X_test_image,
    )

    final_model = best_result.factory()
    final_model.fit(X_train_validation_fused, y_train_validation)
    cap_predictions = np.asarray(final_model.predict(X_test_fused), dtype=float)
    cap_metrics = calculate_metrics(y_test, cap_predictions)
    LOGGER.info(
        "Final test | cap=%s | %s | %s | image_dim=%s | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
        best_result.image_cap,
        best_result.representation_name,
        best_result.model_label,
        best_result.image_dim,
        cap_metrics["mae"],
        cap_metrics["rmse"],
        cap_metrics["r2"],
        cap_metrics["mape"],
    )

    baseline_predictions, baseline_subset_metrics = train_matched_tabular_baseline(
        train_validation_df=train_validation_df,
        test_df=test_df,
    )

    test_analysis_df = test_df.copy()
    test_analysis_df[USED_IMAGE_COUNT_COLUMN] = extract_used_image_count(
        cap_info=selected_cap_info,
        split_listing_ids=test_listing_ids,
    )
    prediction_frame = build_prediction_frame(
        test_df=test_analysis_df,
        baseline_predictions=baseline_predictions,
        fusion_predictions=cap_predictions,
    )

    district_improvement_df = summarize_group_improvement(
        prediction_frame=prediction_frame,
        group_column="district",
        min_samples=5,
    )
    price_improvement_df = summarize_binned_improvement(
        prediction_frame=prediction_frame,
        group_column="price_range",
        label_order=PRICE_BIN_LABELS,
    )
    m2_improvement_df = summarize_binned_improvement(
        prediction_frame=prediction_frame,
        group_column="m2_range",
        label_order=M2_BIN_LABELS,
    )
    used_image_count_df = summarize_binned_improvement(
        prediction_frame=prediction_frame,
        group_column="used_image_count_range",
        label_order=USED_IMAGE_COUNT_LABELS,
    )
    worst_cases_df = select_worst_cases(prediction_frame, top_n=20)

    image_ablation_metrics, image_ablation_note = compute_reduced_image_ablation(
        model=final_model,
        X_test=X_test_fused,
        y_test=y_test,
        image_dim=best_result.image_dim,
    )
    comparison_table = build_reference_comparison_table(
        baseline_subset_metrics=baseline_subset_metrics,
        cap_metrics=cap_metrics,
    )

    feature_names = [str(name) for name in final_tabular_preprocessor.get_feature_names_out()]
    feature_names.extend(
        f"{REDUCED_IMAGE_PREFIX}{index}" for index in range(best_result.image_dim)
    )

    report_body = build_report(
        join_count=len(base_df),
        representation_dimensions=representation_dimensions,
        best_result=best_result,
        validation_leaderboard=validation_leaderboard,
        cap_summary_df=cap_summary_df,
        representation_summary_df=representation_summary_df,
        image_dim_summary_df=image_dim_summary_df,
        cap_metrics=cap_metrics,
        baseline_subset_metrics=baseline_subset_metrics,
        comparison_table=comparison_table,
        district_improvement_df=district_improvement_df,
        price_improvement_df=price_improvement_df,
        m2_improvement_df=m2_improvement_df,
        used_image_count_df=used_image_count_df,
        image_ablation_metrics=image_ablation_metrics,
        image_ablation_note=image_ablation_note,
        worst_cases_df=worst_cases_df,
        model_tracking=model_tracking,
    )

    save_model_bundle(
        best_result=best_result,
        tabular_preprocessor=final_tabular_preprocessor,
        image_processor=final_image_processor,
        model=final_model,
        feature_names=feature_names,
        join_count=len(base_df),
        representation_dimensions=representation_dimensions,
        cap_metrics=cap_metrics,
        baseline_subset_metrics=baseline_subset_metrics,
        model_tracking=model_tracking,
    )

    LOGGER.info("Rapor kaydediliyor: %s", REPORT_OUTPUT_PATH)
    REPORT_OUTPUT_PATH.write_text(report_body, encoding="utf-8")
    LOGGER.info("Tamamlandi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
