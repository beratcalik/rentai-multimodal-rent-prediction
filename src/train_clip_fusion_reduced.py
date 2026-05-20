from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import partial
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
from train_image_fusion_reduced import (
    FUSION_TABULAR_COLUMNS,
    M2_BIN_EDGES,
    M2_BIN_LABELS,
    PRICE_BIN_EDGES,
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

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Eksik bagimlilik bulundu. Lutfen once `python -m pip install -r requirements.txt` calistirin."
    ) from exc


LOGGER = logging.getLogger("clip_fusion_reduced_training")

ROOT_DIR = Path(__file__).resolve().parent.parent
MULTIMODAL_DATASET_PATH = ROOT_DIR / "dataset" / "train_ready_multimodal.parquet"
CLIP_EMBEDDINGS_PATH = ROOT_DIR / "dataset" / "clip_image_embeddings.parquet"
MODEL_OUTPUT_PATH = ROOT_DIR / "models" / "clip_fusion_reduced_model.joblib"
REPORT_OUTPUT_PATH = ROOT_DIR / "reports" / "clip_fusion_reduced_results.md"

LISTING_ID_COLUMN = "listing_id"
USED_IMAGE_COUNT_COLUMN = "used_image_count"
REDUCER_NAME = "PCA"
REDUCED_IMAGE_DIMS = [16, 32, 64, 128]
PRIMARY_SELECTION_METRIC = "mae"
REDUCED_IMAGE_PREFIX = "clip_reduced_"

REPRESENTATION_COLUMNS = {
    "clip_mean_embedding": "clip_mean_embedding",
    "clip_max_embedding": "clip_max_embedding",
    "clip_meanmax_embedding": "clip_meanmax_embedding",
}
REPRESENTATION_PREFIXES = {
    "clip_mean_embedding": "clip_mean_emb_",
    "clip_max_embedding": "clip_max_emb_",
    "clip_meanmax_embedding": "clip_meanmax_emb_",
}
USED_IMAGE_COUNT_BIN_EDGES = [0, 5, 9, 13, 17]
USED_IMAGE_COUNT_LABELS = ["1-4 images", "5-8 images", "9-12 images", "13-16 images"]

MATCHED_BASELINE_REFERENCE = {
    "mae": 4700.08,
    "rmse": 7057.76,
    "r2": 0.8013,
    "mape": 13.79,
}
EFFICIENTNET_6_REFERENCE = {
    "mae": 4555.73,
    "rmse": 6842.20,
    "r2": 0.8133,
    "mape": 13.53,
}
EFFICIENTNET_ALL_REFERENCE = {
    "mae": 4570.65,
    "rmse": 6673.16,
    "r2": 0.8224,
    "mape": 13.59,
}


@dataclass
class ValidationTrial:
    representation_name: str
    model_label: str
    image_dim: int
    validation_metrics: dict[str, float]
    factory: Callable[[], Any]


def ensure_output_directories() -> None:
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_tabular_preprocessor() -> ColumnTransformer:
    from train_image_fusion_reduced import build_tabular_preprocessor as build_shared_preprocessor

    return build_shared_preprocessor()


def load_and_join_datasets(
    multimodal_path: Path,
    embeddings_path: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, int]]:
    if not multimodal_path.exists():
        raise FileNotFoundError(f"Multimodal dataset bulunamadi: {multimodal_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"CLIP embeddings parquet bulunamadi: {embeddings_path}")

    LOGGER.info("Multimodal dataset okunuyor: %s", multimodal_path)
    multimodal_df = pd.read_parquet(multimodal_path)
    LOGGER.info("CLIP embeddings okunuyor: %s", embeddings_path)
    embeddings_df = pd.read_parquet(embeddings_path)

    required_multimodal_columns = set(FUSION_TABULAR_COLUMNS + [TARGET_COLUMN, LISTING_ID_COLUMN])
    missing_multimodal_columns = sorted(required_multimodal_columns - set(multimodal_df.columns))
    if missing_multimodal_columns:
        raise ValueError(f"Multimodal dataset icinde eksik kolonlar bulundu: {missing_multimodal_columns}")

    required_embedding_columns = {LISTING_ID_COLUMN, USED_IMAGE_COUNT_COLUMN, *REPRESENTATION_COLUMNS.values()}
    missing_embedding_columns = sorted(required_embedding_columns - set(embeddings_df.columns))
    if missing_embedding_columns:
        raise ValueError(f"CLIP embedding dataset icinde eksik kolonlar bulundu: {missing_embedding_columns}")

    embeddings_df = embeddings_df.drop_duplicates(subset=[LISTING_ID_COLUMN], keep="first").reset_index(drop=True)

    expanded_frames: list[pd.DataFrame] = [embeddings_df.loc[:, [LISTING_ID_COLUMN, USED_IMAGE_COUNT_COLUMN]].copy()]
    representation_feature_map: dict[str, list[str]] = {}
    representation_dimensions: dict[str, int] = {}

    for representation_name, source_column in REPRESENTATION_COLUMNS.items():
        vectors: list[np.ndarray] = []
        vector_dim: int | None = None

        for value in embeddings_df[source_column]:
            vector = np.asarray(value, dtype=np.float32).ravel()
            if vector_dim is None:
                vector_dim = int(len(vector))
            elif len(vector) != vector_dim:
                raise ValueError(f"{source_column} kolonunda tutarsiz vektor boyutu bulundu.")
            vectors.append(vector)

        if vector_dim is None:
            raise ValueError(f"{source_column} kolonunda hic vektor bulunamadi.")

        representation_dimensions[representation_name] = vector_dim
        feature_columns = [
            f"{REPRESENTATION_PREFIXES[representation_name]}{index}"
            for index in range(vector_dim)
        ]
        representation_feature_map[representation_name] = feature_columns
        expanded_frames.append(pd.DataFrame(np.vstack(vectors), columns=feature_columns))

    expanded_embeddings_df = pd.concat(expanded_frames, axis=1)
    expanded_embeddings_df = expanded_embeddings_df.loc[
        :,
        ~expanded_embeddings_df.columns.duplicated(),
    ].copy()

    joined_df = multimodal_df.merge(
        expanded_embeddings_df,
        on=LISTING_ID_COLUMN,
        how="inner",
    )
    joined_df = joined_df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    LOGGER.info("Join sonrasi ornek sayisi: %s", len(joined_df))
    return joined_df, representation_feature_map, representation_dimensions


def extract_image_matrix(dataframe: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return np.asarray(dataframe.loc[:, feature_columns], dtype=np.float32)


def build_image_processor(image_dim: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "reducer",
                PCA(
                    n_components=image_dim,
                    svd_solver="randomized",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_model_factories() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    model_tracking: dict[str, dict[str, Any]] = {
        "XGBRegressor": {
            "import_status": "not_attempted",
            "success_combinations": [],
            "failure_notes": [],
        },
        "LightGBMRegressor": {
            "import_status": "not_attempted",
            "success_combinations": [],
            "failure_notes": [],
        },
        "HistGradientBoostingRegressor": {
            "import_status": "ok",
            "success_combinations": [],
            "failure_notes": [],
        },
    }

    try:
        from xgboost import XGBRegressor

        model_tracking["XGBRegressor"]["import_status"] = "ok"
        candidates.append(
            {
                "label": "XGBRegressor",
                "factory": partial(
                    XGBRegressor,
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
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        model_tracking["XGBRegressor"]["import_status"] = (
            f"import hatasi: {exc.__class__.__name__}: {exc}"
        )

    try:
        from lightgbm import LGBMRegressor

        model_tracking["LightGBMRegressor"]["import_status"] = "ok"
        candidates.append(
            {
                "label": "LightGBMRegressor",
                "factory": partial(
                    LGBMRegressor,
                    n_estimators=700,
                    learning_rate=0.03,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    objective="regression",
                    n_jobs=-1,
                    verbosity=-1,
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        model_tracking["LightGBMRegressor"]["import_status"] = (
            f"import hatasi: {exc.__class__.__name__}: {exc}"
        )

    candidates.append(
        {
            "label": "HistGradientBoostingRegressor",
            "factory": partial(
                HistGradientBoostingRegressor,
                learning_rate=0.05,
                max_iter=400,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            ),
        }
    )

    return candidates, model_tracking


def train_validation_trials(
    candidates: list[dict[str, Any]],
    X_train_tabular: np.ndarray,
    X_validation_tabular: np.ndarray,
    y_train: pd.Series,
    y_validation: pd.Series,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    representation_feature_map: dict[str, list[str]],
    model_tracking: dict[str, dict[str, Any]],
) -> list[ValidationTrial]:
    results: list[ValidationTrial] = []

    for representation_name, feature_columns in representation_feature_map.items():
        train_image_matrix = extract_image_matrix(train_df, feature_columns)
        validation_image_matrix = extract_image_matrix(validation_df, feature_columns)

        for image_dim in REDUCED_IMAGE_DIMS:
            LOGGER.info(
                "%s reduction fit ediliyor | representation=%s | image_dim=%s",
                REDUCER_NAME,
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
                    "Model egitiliyor | representation=%s | image_dim=%s | %s",
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
                            representation_name=representation_name,
                            model_label=model_label,
                            image_dim=image_dim,
                            validation_metrics=validation_metrics,
                            factory=candidate["factory"],
                        )
                    )
                    LOGGER.info(
                        "Validation | %s | image_dim=%s | %s | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
                        representation_name,
                        image_dim,
                        model_label,
                        validation_metrics["mae"],
                        validation_metrics["rmse"],
                        validation_metrics["r2"],
                        validation_metrics["mape"],
                    )
                    model_tracking[model_label]["success_combinations"].append(
                        f"{representation_name}@{image_dim}"
                    )
                except Exception as exc:  # pragma: no cover - defensive path
                    LOGGER.exception(
                        "Model denemesi basarisiz oldu | representation=%s | image_dim=%s | %s",
                        representation_name,
                        image_dim,
                        model_label,
                    )
                    model_tracking[model_label]["failure_notes"].append(
                        f"{representation_name}@{image_dim}: {exc.__class__.__name__}: {exc}"
                    )

    if not results:
        raise RuntimeError("Hicbir CLIP fusion modeli basariyla egitilemedi.")

    return results


def select_best_trial(results: list[ValidationTrial]) -> ValidationTrial:
    best_result = min(results, key=lambda item: item.validation_metrics[PRIMARY_SELECTION_METRIC])
    LOGGER.info(
        "En iyi kombinasyon secildi: %s + %s + image_dim=%s (validation MAE=%.2f)",
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
        by=["validation_mae", "validation_rmse", "representation", "image_dim", "model"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def build_representation_summary(validation_leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for representation_name in REPRESENTATION_COLUMNS:
        subset = validation_leaderboard[
            validation_leaderboard["representation"] == representation_name
        ]
        if subset.empty:
            continue
        best_row = subset.iloc[0]
        rows.append(
            {
                "representation": representation_name,
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
                "best_representation": best_row["representation"],
                "best_model": best_row["model"],
                "best_validation_mae": float(best_row["validation_mae"]),
                "best_validation_rmse": float(best_row["validation_rmse"]),
                "best_validation_r2": float(best_row["validation_r2"]),
                "best_validation_mape": float(best_row["validation_mape"]),
            }
        )
    return pd.DataFrame(rows)


def build_prediction_frame(
    test_df: pd.DataFrame,
    baseline_predictions: np.ndarray,
    fusion_predictions: np.ndarray,
) -> pd.DataFrame:
    frame = test_df.loc[
        :,
        [
            LISTING_ID_COLUMN,
            "district",
            "neighborhood",
            "rooms",
            "m2_gross",
            USED_IMAGE_COUNT_COLUMN,
        ],
    ].copy()
    frame["actual_price_try"] = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    frame["baseline_prediction"] = np.asarray(baseline_predictions, dtype=float)
    frame["fusion_prediction"] = np.asarray(fusion_predictions, dtype=float)
    frame["baseline_abs_error"] = np.abs(frame["baseline_prediction"] - frame["actual_price_try"])
    frame["fusion_abs_error"] = np.abs(frame["fusion_prediction"] - frame["actual_price_try"])
    frame["baseline_ape_pct"] = (
        frame["baseline_abs_error"] / frame["actual_price_try"].clip(lower=1e-8)
    ) * 100.0
    frame["fusion_ape_pct"] = (
        frame["fusion_abs_error"] / frame["actual_price_try"].clip(lower=1e-8)
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
    frame["used_image_count_range"] = pd.cut(
        frame[USED_IMAGE_COUNT_COLUMN],
        bins=USED_IMAGE_COUNT_BIN_EDGES,
        labels=USED_IMAGE_COUNT_LABELS,
        include_lowest=True,
        right=False,
    )
    frame["fusion_residual"] = frame["fusion_prediction"] - frame["actual_price_try"]
    frame["abs_error_gain"] = frame["baseline_abs_error"] - frame["fusion_abs_error"]
    return frame


def build_reference_comparison_table(
    baseline_subset_metrics: dict[str, float],
    clip_metrics: dict[str, float],
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
            "row": "EfficientNet 6-image reduced fusion",
            "mae": EFFICIENTNET_6_REFERENCE["mae"],
            "rmse": EFFICIENTNET_6_REFERENCE["rmse"],
            "r2": EFFICIENTNET_6_REFERENCE["r2"],
            "mape": EFFICIENTNET_6_REFERENCE["mape"],
        },
        {
            "row": "EfficientNet all-image reduced fusion",
            "mae": EFFICIENTNET_ALL_REFERENCE["mae"],
            "rmse": EFFICIENTNET_ALL_REFERENCE["rmse"],
            "r2": EFFICIENTNET_ALL_REFERENCE["r2"],
            "mape": EFFICIENTNET_ALL_REFERENCE["mape"],
        },
        {
            "row": "CLIP reduced fusion",
            "mae": clip_metrics["mae"],
            "rmse": clip_metrics["rmse"],
            "r2": clip_metrics["r2"],
            "mape": clip_metrics["mape"],
        },
        {
            "row": "Improvement vs matched baseline",
            "mae": baseline_subset_metrics["mae"] - clip_metrics["mae"],
            "rmse": baseline_subset_metrics["rmse"] - clip_metrics["rmse"],
            "r2": clip_metrics["r2"] - baseline_subset_metrics["r2"],
            "mape": baseline_subset_metrics["mape"] - clip_metrics["mape"],
        },
        {
            "row": "Improvement vs EfficientNet 6-image",
            "mae": EFFICIENTNET_6_REFERENCE["mae"] - clip_metrics["mae"],
            "rmse": EFFICIENTNET_6_REFERENCE["rmse"] - clip_metrics["rmse"],
            "r2": clip_metrics["r2"] - EFFICIENTNET_6_REFERENCE["r2"],
            "mape": EFFICIENTNET_6_REFERENCE["mape"] - clip_metrics["mape"],
        },
        {
            "row": "Improvement vs EfficientNet all-image",
            "mae": EFFICIENTNET_ALL_REFERENCE["mae"] - clip_metrics["mae"],
            "rmse": EFFICIENTNET_ALL_REFERENCE["rmse"] - clip_metrics["rmse"],
            "r2": clip_metrics["r2"] - EFFICIENTNET_ALL_REFERENCE["r2"],
            "mape": EFFICIENTNET_ALL_REFERENCE["mape"] - clip_metrics["mape"],
        },
    ]
    return pd.DataFrame(rows)


def render_model_status(model_tracking: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for model_name in ["XGBRegressor", "LightGBMRegressor", "HistGradientBoostingRegressor"]:
        tracking = model_tracking[model_name]
        import_status = tracking["import_status"]
        if import_status != "ok":
            lines.append(f"- `{model_name}`: {import_status}")
            continue

        success_values = sorted(set(str(value) for value in tracking["success_combinations"]))
        if success_values:
            lines.append(
                f"- `{model_name}`: calisti; denenen kombinasyonlar arasinda {', '.join(success_values[:8])}"
                + (" ..." if len(success_values) > 8 else "")
            )
            continue

        if tracking["failure_notes"]:
            lines.append(f"- `{model_name}`: import edildi ama calismadi; {tracking['failure_notes'][0]}")
            continue

        lines.append(f"- `{model_name}`: import edildi ama sonuc kaydi olusmadi.")

    return "\n".join(lines) if lines else "_Veri yok_"


def build_final_commentary(
    clip_metrics: dict[str, float],
    baseline_subset_metrics: dict[str, float],
    image_ablation_metrics: dict[str, float],
) -> str:
    comparison_bits: list[str] = []

    if clip_metrics["mae"] < baseline_subset_metrics["mae"]:
        comparison_bits.append(
            f"matched tabular baseline'a gore MAE tarafinda {baseline_subset_metrics['mae'] - clip_metrics['mae']:.2f} TRY iyilesme var"
        )
    else:
        comparison_bits.append(
            f"matched tabular baseline'a gore MAE tarafinda {clip_metrics['mae'] - baseline_subset_metrics['mae']:.2f} TRY kotulesme var"
        )

    if clip_metrics["mae"] < EFFICIENTNET_6_REFERENCE["mae"]:
        comparison_bits.append(
            f"EfficientNet 6-image referansina gore MAE {EFFICIENTNET_6_REFERENCE['mae'] - clip_metrics['mae']:.2f} TRY daha iyi"
        )
    else:
        comparison_bits.append(
            f"EfficientNet 6-image referansina gore MAE {clip_metrics['mae'] - EFFICIENTNET_6_REFERENCE['mae']:.2f} TRY daha zayif"
        )

    if clip_metrics["rmse"] < EFFICIENTNET_ALL_REFERENCE["rmse"]:
        comparison_bits.append(
            f"all-image EfficientNet referansina gore RMSE {EFFICIENTNET_ALL_REFERENCE['rmse'] - clip_metrics['rmse']:.2f} kadar daha iyi"
        )
    else:
        comparison_bits.append(
            f"all-image EfficientNet referansina gore RMSE {clip_metrics['rmse'] - EFFICIENTNET_ALL_REFERENCE['rmse']:.2f} kadar daha zayif"
        )

    if image_ablation_metrics["mae"] > clip_metrics["mae"]:
        comparison_bits.append(
            f"image branch ablasyonda MAE {image_ablation_metrics['mae'] - clip_metrics['mae']:.2f} kadar kotulestigi icin CLIP image sinyali modele net pozitif katki veriyor"
        )
    else:
        comparison_bits.append(
            f"ablasyonda MAE {clip_metrics['mae'] - image_ablation_metrics['mae']:.2f} kadar iyilestigi icin CLIP image block henuz en verimli sekilde kullanilmiyor olabilir"
        )

    return "; ".join(comparison_bits) + "."


def build_report(
    join_count: int,
    representation_dimensions: dict[str, int],
    best_result: ValidationTrial,
    validation_leaderboard: pd.DataFrame,
    representation_summary_df: pd.DataFrame,
    image_dim_summary_df: pd.DataFrame,
    clip_metrics: dict[str, float],
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
        clip_metrics=clip_metrics,
        baseline_subset_metrics=baseline_subset_metrics,
        image_ablation_metrics=image_ablation_metrics,
    )

    representation_dimension_text = ", ".join(
        f"{name}={dimension}"
        for name, dimension in representation_dimensions.items()
    )

    report_lines = [
        "# CLIP Fusion Reduced Results",
        "",
        "## Ozet",
        "",
        f"- Multimodal source: `{MULTIMODAL_DATASET_PATH}`",
        f"- CLIP embedding source: `{CLIP_EMBEDDINGS_PATH}`",
        f"- Kaydedilen model bundle: `{MODEL_OUTPUT_PATH}`",
        f"- Join sonucu kalan ornek sayisi: **{join_count:,}**",
        f"- Deneen representationlar: **{', '.join(REPRESENTATION_COLUMNS.keys())}**",
        f"- Representation dimensionlari: **{representation_dimension_text}**",
        f"- Reducer: **{REDUCER_NAME}**",
        f"- Deneen image_dim degerleri: **{', '.join(str(value) for value in REDUCED_IMAGE_DIMS)}**",
        f"- En iyi kombinasyon: **{best_result.representation_name} + {best_result.model_label} + image_dim={best_result.image_dim}**",
        "",
        "## Validation Leaderboard",
        "",
        dataframe_to_markdown_table(validation_leaderboard, digits=4),
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
                ["MAE", format_float(clip_metrics["mae"])],
                ["RMSE", format_float(clip_metrics["rmse"])],
                ["R2", format_float(clip_metrics["r2"], digits=4)],
                ["MAPE (%)", format_float(clip_metrics["mape"])],
            ],
        ),
        "",
        "## Matched Baseline ve EfficientNet Referanslari Karsilastirmasi",
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
        f"- Final CLIP fusion MAE: **{clip_metrics['mae']:.2f}**",
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
    tabular_preprocessor: ColumnTransformer,
    image_processor: Pipeline,
    model: Any,
    feature_names: list[str],
    join_count: int,
    representation_dimensions: dict[str, int],
    clip_metrics: dict[str, float],
    baseline_subset_metrics: dict[str, float],
    model_tracking: dict[str, dict[str, Any]],
) -> None:
    bundle = {
        "model_name": best_result.model_label,
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
        "clip_metrics": clip_metrics,
        "baseline_subset_metrics": baseline_subset_metrics,
        "reference_metrics": {
            "matched_tabular_baseline": MATCHED_BASELINE_REFERENCE,
            "efficientnet_6_image": EFFICIENTNET_6_REFERENCE,
            "efficientnet_all_image": EFFICIENTNET_ALL_REFERENCE,
        },
        "model_tracking": model_tracking,
    }
    LOGGER.info("Model bundle kaydediliyor: %s", MODEL_OUTPUT_PATH)
    joblib.dump(bundle, MODEL_OUTPUT_PATH)


def main() -> int:
    configure_logging()
    ensure_output_directories()

    joined_df, representation_feature_map, representation_dimensions = load_and_join_datasets(
        multimodal_path=MULTIMODAL_DATASET_PATH,
        embeddings_path=CLIP_EMBEDDINGS_PATH,
    )
    splits = split_dataset(joined_df)

    train_df = splits.train.copy()
    validation_df = splits.validation.copy()
    test_df = splits.test.copy()

    y_train = train_df[TARGET_COLUMN]
    y_validation = validation_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

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
        train_df=train_df,
        validation_df=validation_df,
        representation_feature_map=representation_feature_map,
        model_tracking=model_tracking,
    )
    best_result = select_best_trial(validation_trials)
    validation_leaderboard = build_validation_leaderboard(validation_trials)
    representation_summary_df = build_representation_summary(validation_leaderboard)
    image_dim_summary_df = build_image_dim_summary(validation_leaderboard)

    LOGGER.info(
        "Final model yeniden egitiliyor | %s + %s + image_dim=%s",
        best_result.representation_name,
        best_result.model_label,
        best_result.image_dim,
    )
    train_validation_df = pd.concat([train_df, validation_df], axis=0, ignore_index=True)
    y_train_validation = train_validation_df[TARGET_COLUMN]

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

    best_feature_columns = representation_feature_map[best_result.representation_name]
    train_validation_image_matrix = extract_image_matrix(train_validation_df, best_feature_columns)
    test_image_matrix = extract_image_matrix(test_df, best_feature_columns)

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
    clip_predictions = np.asarray(final_model.predict(X_test_fused), dtype=float)
    clip_metrics = calculate_metrics(y_test, clip_predictions)
    LOGGER.info(
        "Final test | %s | %s | image_dim=%s | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
        best_result.representation_name,
        best_result.model_label,
        best_result.image_dim,
        clip_metrics["mae"],
        clip_metrics["rmse"],
        clip_metrics["r2"],
        clip_metrics["mape"],
    )

    baseline_predictions, baseline_subset_metrics = train_matched_tabular_baseline(
        train_validation_df=train_validation_df,
        test_df=test_df,
    )
    prediction_frame = build_prediction_frame(
        test_df=test_df,
        baseline_predictions=baseline_predictions,
        fusion_predictions=clip_predictions,
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
        clip_metrics=clip_metrics,
    )

    feature_names = [str(name) for name in final_tabular_preprocessor.get_feature_names_out()]
    feature_names.extend(
        f"{REDUCED_IMAGE_PREFIX}{index}" for index in range(best_result.image_dim)
    )

    report_body = build_report(
        join_count=len(joined_df),
        representation_dimensions=representation_dimensions,
        best_result=best_result,
        validation_leaderboard=validation_leaderboard,
        representation_summary_df=representation_summary_df,
        image_dim_summary_df=image_dim_summary_df,
        clip_metrics=clip_metrics,
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
        join_count=len(joined_df),
        representation_dimensions=representation_dimensions,
        clip_metrics=clip_metrics,
        baseline_subset_metrics=baseline_subset_metrics,
        model_tracking=model_tracking,
    )

    LOGGER.info("Rapor kaydediliyor: %s", REPORT_OUTPUT_PATH)
    REPORT_OUTPUT_PATH.write_text(report_body, encoding="utf-8")
    LOGGER.info("Tamamlandi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
