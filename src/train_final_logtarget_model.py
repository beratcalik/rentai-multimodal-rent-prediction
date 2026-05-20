from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from train_baseline import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    build_preprocessor as build_baseline_preprocessor,
    calculate_metrics,
    configure_logging,
    normalize_furnished_flag,
    normalize_string,
    parse_floor_value,
    parse_room_count,
    split_dataset,
)

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Eksik bagimlilik bulundu. Lutfen once `python -m pip install -r requirements.txt` calistirin."
    ) from exc

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "xgboost bulunamadi. Lutfen aktif Python interpreter icinde xgboost kurulu oldugunu dogrulayin."
    ) from exc


LOGGER = logging.getLogger("final_logtarget_training")

ROOT_DIR = Path(__file__).resolve().parent.parent
MULTIMODAL_DATASET_PATH = ROOT_DIR / "dataset" / "train_ready_multimodal.parquet"
IMAGE_EMBEDDINGS_PATH = ROOT_DIR / "dataset" / "image_embeddings.parquet"
MODEL_OUTPUT_PATH = ROOT_DIR / "models" / "final_logtarget_model.joblib"
REPORT_OUTPUT_PATH = ROOT_DIR / "reports" / "final_logtarget_results.md"

PREVIOUS_REDUCED_FUSION_METRICS = {
    "mae": 4555.73,
    "rmse": 6842.20,
    "r2": 0.8133,
    "mape": 13.53,
}
PREVIOUS_ALL_IMAGE_REDUCED_METRICS = {
    "mae": 4570.65,
    "rmse": 6673.16,
    "r2": 0.8224,
    "mape": 13.59,
}

VALID_IMAGE_COUNT_COLUMN = "valid_image_count"
IMAGE_EMBEDDING_COLUMN = "image_embedding"
IMAGE_FEATURE_PREFIX = "image_emb_"
REDUCED_IMAGE_PREFIX = "image_reduced_"
REDUCER_NAME = "PCA"
REDUCED_IMAGE_DIM = 16
PRIMARY_SELECTION_METRIC = "mae"

FUSION_TABULAR_COLUMNS = FEATURE_COLUMNS + [VALID_IMAGE_COUNT_COLUMN]
FUSION_NUMERIC_COLUMNS = NUMERIC_COLUMNS + [VALID_IMAGE_COUNT_COLUMN]
FUSION_CATEGORICAL_COLUMNS = list(CATEGORICAL_COLUMNS)

PRICE_BIN_EDGES = [0, 20_000, 30_000, 40_000, 50_000, 75_000, np.inf]
PRICE_BIN_LABELS = [
    "0-20k TRY",
    "20k-30k TRY",
    "30k-40k TRY",
    "40k-50k TRY",
    "50k-75k TRY",
    "75k+ TRY",
]
M2_BIN_EDGES = [0, 75, 100, 125, 150, 200, np.inf]
M2_BIN_LABELS = [
    "0-75 m2",
    "75-100 m2",
    "100-125 m2",
    "125-150 m2",
    "150-200 m2",
    "200+ m2",
]

HIGH_PRICE_THRESHOLD = 75_000.0
LUXURY_PRICE_THRESHOLD = 75_000.0
LUXURY_M2_THRESHOLD = 180.0
LUXURY_ROOMS_THRESHOLD = 5.0
EARLY_STOPPING_ROUNDS = 50


@dataclass
class ValidationTrial:
    candidate_label: str
    validation_metrics: dict[str, float]
    effective_n_estimators: int
    used_early_stopping: bool


def ensure_output_directories() -> None:
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - sklearn compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_and_join_datasets(
    multimodal_path: Path,
    embeddings_path: Path,
) -> tuple[pd.DataFrame, list[str], int]:
    if not multimodal_path.exists():
        raise FileNotFoundError(f"Multimodal dataset bulunamadi: {multimodal_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Image embeddings parquet bulunamadi: {embeddings_path}")

    LOGGER.info("Multimodal dataset okunuyor: %s", multimodal_path)
    multimodal_df = pd.read_parquet(multimodal_path)
    LOGGER.info("Image embeddings okunuyor: %s", embeddings_path)
    embeddings_df = pd.read_parquet(embeddings_path)

    required_multimodal_columns = set(FUSION_TABULAR_COLUMNS + [TARGET_COLUMN, "listing_id"])
    missing_multimodal_columns = sorted(required_multimodal_columns - set(multimodal_df.columns))
    if missing_multimodal_columns:
        raise ValueError(f"Multimodal dataset icinde eksik kolonlar bulundu: {missing_multimodal_columns}")

    required_embedding_columns = {"listing_id", IMAGE_EMBEDDING_COLUMN}
    missing_embedding_columns = sorted(required_embedding_columns - set(embeddings_df.columns))
    if missing_embedding_columns:
        raise ValueError(f"Embedding dataset icinde eksik kolonlar bulundu: {missing_embedding_columns}")

    embeddings_df = embeddings_df.drop_duplicates(subset=["listing_id"], keep="first").reset_index(drop=True)

    embedding_vectors: list[np.ndarray] = []
    embedding_dimension: int | None = None
    for value in embeddings_df[IMAGE_EMBEDDING_COLUMN]:
        vector = np.asarray(value, dtype=np.float32).ravel()
        if embedding_dimension is None:
            embedding_dimension = int(len(vector))
        elif len(vector) != embedding_dimension:
            raise ValueError("image_embedding kolonunda tutarsiz vektor boyutu bulundu.")
        embedding_vectors.append(vector)

    if embedding_dimension is None:
        raise ValueError("image_embedding kolonunda hic vektor bulunamadi.")

    embedding_matrix = np.vstack(embedding_vectors)
    image_feature_columns = [f"{IMAGE_FEATURE_PREFIX}{index}" for index in range(embedding_dimension)]
    embedding_expanded_df = pd.DataFrame(embedding_matrix, columns=image_feature_columns)
    embeddings_expanded_df = pd.concat(
        [embeddings_df.loc[:, ["listing_id"]].reset_index(drop=True), embedding_expanded_df],
        axis=1,
    )

    joined_df = multimodal_df.merge(embeddings_expanded_df, on="listing_id", how="inner")
    joined_df = joined_df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    LOGGER.info("Join sonrasi ornek sayisi: %s", len(joined_df))
    return joined_df, image_feature_columns, embedding_dimension


def clean_tabular_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.loc[:, FUSION_TABULAR_COLUMNS].copy()

    for column in FUSION_CATEGORICAL_COLUMNS:
        if column == "is_furnished":
            frame[column] = frame[column].map(normalize_furnished_flag)
        else:
            frame[column] = frame[column].map(normalize_string)

    frame["rooms"] = frame["rooms"].map(parse_room_count)
    frame["floor"] = frame["floor"].map(parse_floor_value)

    for column in FUSION_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def build_tabular_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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
            ("numeric", numeric_transformer, FUSION_NUMERIC_COLUMNS),
            ("categorical", categorical_transformer, FUSION_CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def extract_image_matrix(dataframe: pd.DataFrame, image_feature_columns: list[str]) -> np.ndarray:
    return np.asarray(dataframe.loc[:, image_feature_columns], dtype=np.float32)


def build_image_processor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "reducer",
                PCA(
                    n_components=REDUCED_IMAGE_DIM,
                    svd_solver="randomized",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def concatenate_feature_blocks(tabular_block: np.ndarray, image_block: np.ndarray) -> np.ndarray:
    fused_block = np.hstack([tabular_block, image_block])
    return np.asarray(fused_block, dtype=np.float32)


def build_xgb_params(n_estimators: int = 700) -> dict[str, Any]:
    return {
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
        "eval_metric": "rmse",
    }


def to_log_target(y: pd.Series) -> np.ndarray:
    return np.log1p(y.to_numpy(dtype=float))


def inverse_log_predictions(predictions_log: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.asarray(predictions_log, dtype=float)), a_min=0.0, a_max=None)


def train_validation_trials(
    X_train_fused: np.ndarray,
    y_train: pd.Series,
    X_validation_fused: np.ndarray,
    y_validation: pd.Series,
) -> list[ValidationTrial]:
    y_train_log = to_log_target(y_train)
    y_validation_log = to_log_target(y_validation)
    trials: list[ValidationTrial] = []

    LOGGER.info("Validation adayi egitiliyor: XGBRegressor")
    base_model = XGBRegressor(**build_xgb_params())
    base_model.fit(X_train_fused, y_train_log)
    base_predictions = inverse_log_predictions(base_model.predict(X_validation_fused))
    base_metrics = calculate_metrics(y_validation, base_predictions)
    trials.append(
        ValidationTrial(
            candidate_label="XGBRegressor",
            validation_metrics=base_metrics,
            effective_n_estimators=build_xgb_params()["n_estimators"],
            used_early_stopping=False,
        )
    )
    LOGGER.info(
        "Validation | XGBRegressor | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
        base_metrics["mae"],
        base_metrics["rmse"],
        base_metrics["r2"],
        base_metrics["mape"],
    )

    LOGGER.info("Validation adayi egitiliyor: XGBRegressorEarlyStopping")
    early_stop_model = XGBRegressor(
        **build_xgb_params(),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    early_stop_model.fit(
        X_train_fused,
        y_train_log,
        eval_set=[(X_validation_fused, y_validation_log)],
        verbose=False,
    )
    early_stop_predictions = inverse_log_predictions(early_stop_model.predict(X_validation_fused))
    early_stop_metrics = calculate_metrics(y_validation, early_stop_predictions)
    best_iteration = getattr(early_stop_model, "best_iteration", None)
    effective_n_estimators = (
        int(best_iteration) + 1
        if best_iteration is not None and int(best_iteration) >= 0
        else build_xgb_params()["n_estimators"]
    )
    trials.append(
        ValidationTrial(
            candidate_label="XGBRegressorEarlyStopping",
            validation_metrics=early_stop_metrics,
            effective_n_estimators=effective_n_estimators,
            used_early_stopping=True,
        )
    )
    LOGGER.info(
        "Validation | XGBRegressorEarlyStopping | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%% | best_n_estimators=%s",
        early_stop_metrics["mae"],
        early_stop_metrics["rmse"],
        early_stop_metrics["r2"],
        early_stop_metrics["mape"],
        effective_n_estimators,
    )

    return trials


def select_best_trial(trials: list[ValidationTrial]) -> ValidationTrial:
    best_result = min(trials, key=lambda item: item.validation_metrics[PRIMARY_SELECTION_METRIC])
    LOGGER.info(
        "En iyi validation adayi secildi: %s (MAE=%.2f)",
        best_result.candidate_label,
        best_result.validation_metrics["mae"],
    )
    return best_result


def build_validation_leaderboard(trials: list[ValidationTrial]) -> pd.DataFrame:
    rows = [
        {
            "candidate": trial.candidate_label,
            "effective_n_estimators": trial.effective_n_estimators,
            "used_early_stopping": trial.used_early_stopping,
            "validation_mae": trial.validation_metrics["mae"],
            "validation_rmse": trial.validation_metrics["rmse"],
            "validation_r2": trial.validation_metrics["r2"],
            "validation_mape": trial.validation_metrics["mape"],
        }
        for trial in trials
    ]
    return pd.DataFrame(rows).sort_values("validation_mae", ascending=True).reset_index(drop=True)


def train_matched_tabular_baseline(
    train_validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    LOGGER.info("Matched tabular baseline train ediliyor: HistGradientBoostingRegressor")
    baseline_preprocessor = build_baseline_preprocessor()
    X_train_validation = baseline_preprocessor.fit_transform(train_validation_df[FEATURE_COLUMNS])
    X_test = baseline_preprocessor.transform(test_df[FEATURE_COLUMNS])

    baseline_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    baseline_model.fit(X_train_validation, train_validation_df[TARGET_COLUMN])
    predictions = np.asarray(baseline_model.predict(X_test), dtype=float)
    metrics = calculate_metrics(test_df[TARGET_COLUMN], predictions)
    return predictions, metrics


def build_prediction_frame(
    test_df: pd.DataFrame,
    baseline_predictions: np.ndarray,
    fusion_predictions: np.ndarray,
) -> pd.DataFrame:
    frame = test_df.loc[:, ["listing_id", "district", "neighborhood", "rooms", "m2_gross"]].copy()
    frame["rooms_count"] = test_df["rooms"].map(parse_room_count)
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
    frame["high_price_segment"] = frame["actual_price_try"] >= HIGH_PRICE_THRESHOLD
    frame["luxury_segment"] = (
        (frame["actual_price_try"] >= LUXURY_PRICE_THRESHOLD)
        | (frame["m2_gross"] >= LUXURY_M2_THRESHOLD)
        | (frame["rooms_count"] >= LUXURY_ROOMS_THRESHOLD)
    )
    frame["fusion_residual"] = frame["fusion_prediction"] - frame["actual_price_try"]
    frame["abs_error_gain"] = frame["baseline_abs_error"] - frame["fusion_abs_error"]
    return frame


def summarize_group_improvement(
    prediction_frame: pd.DataFrame,
    group_column: str,
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
            sample_count=("listing_id", "size"),
            baseline_mae=("baseline_abs_error", "mean"),
            fusion_mae=("fusion_abs_error", "mean"),
            baseline_mape=("baseline_ape_pct", "mean"),
            fusion_mape=("fusion_ape_pct", "mean"),
        )
        .reset_index()
        .rename(columns={group_column: "group"})
    )
    grouped = grouped[grouped["sample_count"] >= min_samples].copy()
    grouped["mae_improvement"] = grouped["baseline_mae"] - grouped["fusion_mae"]
    grouped["mape_improvement"] = grouped["baseline_mape"] - grouped["fusion_mape"]
    grouped = grouped.sort_values("mae_improvement", ascending=False).reset_index(drop=True)
    return grouped


def summarize_binned_improvement(
    prediction_frame: pd.DataFrame,
    group_column: str,
    label_order: list[str],
) -> pd.DataFrame:
    grouped = summarize_group_improvement(
        prediction_frame=prediction_frame,
        group_column=group_column,
        min_samples=1,
    )
    grouped["group"] = pd.Categorical(grouped["group"], categories=label_order, ordered=True)
    grouped = grouped.sort_values("group").reset_index(drop=True)
    grouped["group"] = grouped["group"].astype(str)
    grouped = grouped[grouped["group"] != "nan"].reset_index(drop=True)
    return grouped


def summarize_segment_metrics(
    prediction_frame: pd.DataFrame,
    segment_name: str,
    mask: pd.Series,
) -> pd.DataFrame:
    segment_df = prediction_frame.loc[mask].copy()
    if segment_df.empty:
        return pd.DataFrame(
            columns=[
                "segment",
                "sample_count",
                "baseline_mae",
                "fusion_mae",
                "baseline_rmse",
                "fusion_rmse",
                "baseline_r2",
                "fusion_r2",
                "baseline_mape",
                "fusion_mape",
                "mae_improvement",
                "rmse_improvement",
                "r2_improvement",
                "mape_improvement",
            ]
        )

    actual = segment_df["actual_price_try"]
    baseline_predictions = segment_df["baseline_prediction"].to_numpy(dtype=float)
    fusion_predictions = segment_df["fusion_prediction"].to_numpy(dtype=float)

    baseline_metrics = calculate_metrics(actual, baseline_predictions)
    fusion_metrics = calculate_metrics(actual, fusion_predictions)

    return pd.DataFrame(
        [
            {
                "segment": segment_name,
                "sample_count": int(len(segment_df)),
                "baseline_mae": baseline_metrics["mae"],
                "fusion_mae": fusion_metrics["mae"],
                "baseline_rmse": baseline_metrics["rmse"],
                "fusion_rmse": fusion_metrics["rmse"],
                "baseline_r2": baseline_metrics["r2"],
                "fusion_r2": fusion_metrics["r2"],
                "baseline_mape": baseline_metrics["mape"],
                "fusion_mape": fusion_metrics["mape"],
                "mae_improvement": baseline_metrics["mae"] - fusion_metrics["mae"],
                "rmse_improvement": baseline_metrics["rmse"] - fusion_metrics["rmse"],
                "r2_improvement": fusion_metrics["r2"] - baseline_metrics["r2"],
                "mape_improvement": baseline_metrics["mape"] - fusion_metrics["mape"],
            }
        ]
    )


def summarize_residuals(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, float | str | int]] = []
    segment_map = {
        "overall": pd.Series(np.ones(len(prediction_frame), dtype=bool), index=prediction_frame.index),
        "high_price_75k_plus": prediction_frame["high_price_segment"],
        "luxury_proxy": prediction_frame["luxury_segment"],
    }

    for segment_name, mask in segment_map.items():
        segment_df = prediction_frame.loc[mask].copy()
        if segment_df.empty:
            continue
        residuals = segment_df["fusion_residual"].to_numpy(dtype=float)
        summaries.append(
            {
                "segment": segment_name,
                "sample_count": int(len(segment_df)),
                "mean_residual_try": float(np.mean(residuals)),
                "median_residual_try": float(np.median(residuals)),
                "std_residual_try": float(np.std(residuals)),
                "mean_abs_error_try": float(np.mean(np.abs(residuals))),
                "underprediction_share_pct": float(np.mean(residuals < 0) * 100.0),
                "overprediction_share_pct": float(np.mean(residuals > 0) * 100.0),
            }
        )

    return pd.DataFrame(summaries)


def select_worst_cases(prediction_frame: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    worst_df = prediction_frame.sort_values("fusion_abs_error", ascending=False).head(top_n).copy()
    return worst_df[
        [
            "listing_id",
            "district",
            "neighborhood",
            "rooms",
            "m2_gross",
            "actual_price_try",
            "baseline_prediction",
            "fusion_prediction",
            "fusion_residual",
            "fusion_abs_error",
            "fusion_ape_pct",
            "abs_error_gain",
        ]
    ].reset_index(drop=True)


def compute_reduced_image_ablation(
    model: XGBRegressor,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> tuple[dict[str, float], str]:
    X_without_image = np.array(X_test, copy=True)
    X_without_image[:, -REDUCED_IMAGE_DIM:] = 0.0
    no_image_predictions_log = np.asarray(model.predict(X_without_image), dtype=float)
    no_image_predictions = inverse_log_predictions(no_image_predictions_log)
    no_image_metrics = calculate_metrics(y_test, no_image_predictions)
    return no_image_metrics, (
        "Reduced image block standardized ve PCA ile olusturuldugu icin ablasyon testinde son image block sifira cekildi; "
        "bu yaklasim image branch katkisini ayirmak icin kullanildi."
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


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def dataframe_to_markdown_table(dataframe: pd.DataFrame, digits: int = 2) -> str:
    if dataframe.empty:
        return "_Veri yok_"

    rows: list[list[str]] = []
    for _, row in dataframe.iterrows():
        rendered_row: list[str] = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                rendered_row.append(format_float(float(value), digits=digits))
            else:
                rendered_row.append(str(value))
        rows.append(rendered_row)

    return render_markdown_table(list(dataframe.columns), rows)


def build_comparison_table(
    baseline_subset_metrics: dict[str, float],
    previous_reduced_metrics: dict[str, float],
    previous_all_image_metrics: dict[str, float],
    fusion_metrics: dict[str, float],
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
            "row": "Previous 6-image reduced fusion",
            "mae": previous_reduced_metrics["mae"],
            "rmse": previous_reduced_metrics["rmse"],
            "r2": previous_reduced_metrics["r2"],
            "mape": previous_reduced_metrics["mape"],
        },
        {
            "row": "Previous all-image reduced fusion",
            "mae": previous_all_image_metrics["mae"],
            "rmse": previous_all_image_metrics["rmse"],
            "r2": previous_all_image_metrics["r2"],
            "mape": previous_all_image_metrics["mape"],
        },
        {
            "row": "Log-target final fusion",
            "mae": fusion_metrics["mae"],
            "rmse": fusion_metrics["rmse"],
            "r2": fusion_metrics["r2"],
            "mape": fusion_metrics["mape"],
        },
        {
            "row": "Improvement vs matched baseline",
            "mae": baseline_subset_metrics["mae"] - fusion_metrics["mae"],
            "rmse": baseline_subset_metrics["rmse"] - fusion_metrics["rmse"],
            "r2": fusion_metrics["r2"] - baseline_subset_metrics["r2"],
            "mape": baseline_subset_metrics["mape"] - fusion_metrics["mape"],
        },
        {
            "row": "Improvement vs previous 6-image reduced",
            "mae": previous_reduced_metrics["mae"] - fusion_metrics["mae"],
            "rmse": previous_reduced_metrics["rmse"] - fusion_metrics["rmse"],
            "r2": fusion_metrics["r2"] - previous_reduced_metrics["r2"],
            "mape": previous_reduced_metrics["mape"] - fusion_metrics["mape"],
        },
        {
            "row": "Improvement vs previous all-image reduced",
            "mae": previous_all_image_metrics["mae"] - fusion_metrics["mae"],
            "rmse": previous_all_image_metrics["rmse"] - fusion_metrics["rmse"],
            "r2": fusion_metrics["r2"] - previous_all_image_metrics["r2"],
            "mape": previous_all_image_metrics["mape"] - fusion_metrics["mape"],
        },
    ]
    return pd.DataFrame(rows)


def build_report(
    join_count: int,
    original_embedding_dimension: int,
    best_result: ValidationTrial,
    validation_leaderboard: pd.DataFrame,
    fusion_metrics: dict[str, float],
    baseline_subset_metrics: dict[str, float],
    comparison_table: pd.DataFrame,
    high_price_df: pd.DataFrame,
    luxury_df: pd.DataFrame,
    district_improvement_df: pd.DataFrame,
    price_improvement_df: pd.DataFrame,
    m2_improvement_df: pd.DataFrame,
    residual_summary_df: pd.DataFrame,
    image_ablation_metrics: dict[str, float],
    image_ablation_note: str,
    worst_cases_df: pd.DataFrame,
) -> str:
    if image_ablation_metrics["mae"] > fusion_metrics["mae"]:
        ablation_comment = (
            f"Image branch sifirlandiginda MAE {image_ablation_metrics['mae'] - fusion_metrics['mae']:.2f} kadar kotulesiyor; "
            "image embeddingler modele net pozitif katkili."
        )
    else:
        ablation_comment = (
            f"Image branch sifirlandiginda MAE {fusion_metrics['mae'] - image_ablation_metrics['mae']:.2f} kadar iyilesiyor; "
            "image signal henuz tam verimli kullanilmiyor olabilir."
        )

    logtarget_note = (
        f"Egitim hedefi `log1p(price_try)` olarak olusturuldu; tum validation ve test skorları `expm1` sonrasi gercek TRY fiyatlari uzerinde hesaplandi. "
        f"Secilen aday: **{best_result.candidate_label}**, effective_n_estimators=**{best_result.effective_n_estimators}**."
    )

    report_lines = [
        "# Final Log-Target Fusion Results",
        "",
        "## Ozet",
        "",
        f"- Multimodal source: `{MULTIMODAL_DATASET_PATH}`",
        f"- Image embedding source: `{IMAGE_EMBEDDINGS_PATH}`",
        f"- Kaydedilen model bundle: `{MODEL_OUTPUT_PATH}`",
        f"- Join sonucu kalan ornek sayisi: **{join_count:,}**",
        f"- Original image embedding dimension: **{original_embedding_dimension}**",
        f"- Reducer: **{REDUCER_NAME}**",
        f"- Reduced image_dim: **{REDUCED_IMAGE_DIM}**",
        f"- En iyi validation adayi: **{best_result.candidate_label}**",
        "",
        "## Validation Leaderboard",
        "",
        dataframe_to_markdown_table(validation_leaderboard, digits=4),
        "",
        "## Final Test Metrics",
        "",
        render_markdown_table(
            ["Metric", "Value"],
            [
                ["MAE", format_float(fusion_metrics["mae"])],
                ["RMSE", format_float(fusion_metrics["rmse"])],
                ["R2", format_float(fusion_metrics["r2"], digits=4)],
                ["MAPE (%)", format_float(fusion_metrics["mape"])],
            ],
        ),
        "",
        "## Log-Target Etkisi",
        "",
        f"- {logtarget_note}",
        "",
        "## Baseline ve Onceki Fusion Karsilastirmasi",
        "",
        dataframe_to_markdown_table(comparison_table, digits=4),
        "",
        "## High-Price Segment",
        "",
        "- Tanim: `actual_price_try >= 75,000 TRY`",
        dataframe_to_markdown_table(high_price_df, digits=4),
        "",
        "## Luxury Segment",
        "",
        "- Tanim: luxury proxy = `actual_price_try >= 75,000` veya `m2_gross >= 180` veya `rooms_count >= 5`",
        dataframe_to_markdown_table(luxury_df, digits=4),
        "",
        "## District Bazli Improvement",
        "",
        dataframe_to_markdown_table(district_improvement_df, digits=4),
        "",
        "## Price-Range Bazli Improvement",
        "",
        dataframe_to_markdown_table(price_improvement_df, digits=4),
        "",
        "## m2-Range Bazli Improvement",
        "",
        dataframe_to_markdown_table(m2_improvement_df, digits=4),
        "",
        "## Residual Analysis",
        "",
        dataframe_to_markdown_table(residual_summary_df, digits=4),
        "",
        "## Image Branch Ablation",
        "",
        f"- Log-target fusion MAE: **{fusion_metrics['mae']:.2f}**",
        f"- Image branch sifirlandiginda MAE: **{image_ablation_metrics['mae']:.2f}**",
        f"- Image branch sifirlandiginda MAPE: **{image_ablation_metrics['mape']:.2f}%**",
        f"- Ablasyon yorumu: {image_ablation_note}",
        f"- Yorum: {ablation_comment}",
        "",
        "## En Yuksek Hata Yapan 20 Ilan",
        "",
        dataframe_to_markdown_table(worst_cases_df, digits=4),
        "",
        "## Final Yorum",
        "",
        (
            f"- Log-target model, matched tabular baseline'e gore MAE farkini {baseline_subset_metrics['mae'] - fusion_metrics['mae']:.2f} olarak degistirdi."
            if fusion_metrics["mae"] < baseline_subset_metrics["mae"]
            else f"- Log-target model, matched tabular baseline'in gerisinde kaldi; MAE farki {fusion_metrics['mae'] - baseline_subset_metrics['mae']:.2f}."
        ),
        (
            f"- Onceki 6-image reduced fusion referansina gore MAE degisimi: {PREVIOUS_REDUCED_FUSION_METRICS['mae'] - fusion_metrics['mae']:.2f}."
        ),
        (
            f"- Onceki all-image reduced fusion referansina gore MAE degisimi: {PREVIOUS_ALL_IMAGE_REDUCED_METRICS['mae'] - fusion_metrics['mae']:.2f}."
        ),
        "- Bu kosuda temel hedef, long-tail kira dagiliminda ozellikle yuksek fiyat segmentini daha dengeli ogrenmekti; high-price ve luxury tablolari bu etkiyi ozetliyor.",
    ]
    return "\n".join(report_lines)


def save_model_bundle(
    best_result: ValidationTrial,
    tabular_preprocessor: ColumnTransformer,
    image_processor: Pipeline,
    model: XGBRegressor,
    feature_names: list[str],
    join_count: int,
    original_embedding_dimension: int,
    fusion_metrics: dict[str, float],
    baseline_subset_metrics: dict[str, float],
) -> None:
    bundle = {
        "model_name": "XGBRegressor",
        "candidate_label": best_result.candidate_label,
        "reducer_name": REDUCER_NAME,
        "reduced_image_dim": REDUCED_IMAGE_DIM,
        "target_transform": "log1p_expm1",
        "effective_n_estimators": best_result.effective_n_estimators,
        "used_early_stopping_for_selection": best_result.used_early_stopping,
        "tabular_preprocessor": tabular_preprocessor,
        "image_processor": image_processor,
        "regressor": model,
        "feature_names": feature_names,
        "tabular_feature_columns": FUSION_TABULAR_COLUMNS,
        "join_count": join_count,
        "original_embedding_dimension": original_embedding_dimension,
        "fusion_metrics": fusion_metrics,
        "baseline_subset_metrics": baseline_subset_metrics,
        "previous_reduced_fusion_metrics": PREVIOUS_REDUCED_FUSION_METRICS,
        "previous_all_image_reduced_metrics": PREVIOUS_ALL_IMAGE_REDUCED_METRICS,
    }
    LOGGER.info("Model bundle kaydediliyor: %s", MODEL_OUTPUT_PATH)
    joblib.dump(bundle, MODEL_OUTPUT_PATH)


def main() -> int:
    configure_logging()
    ensure_output_directories()

    joined_df, image_feature_columns, original_embedding_dimension = load_and_join_datasets(
        multimodal_path=MULTIMODAL_DATASET_PATH,
        embeddings_path=IMAGE_EMBEDDINGS_PATH,
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

    train_image_matrix = extract_image_matrix(train_df, image_feature_columns)
    validation_image_matrix = extract_image_matrix(validation_df, image_feature_columns)

    validation_image_processor = build_image_processor()
    X_train_image = validation_image_processor.fit_transform(train_image_matrix)
    X_validation_image = validation_image_processor.transform(validation_image_matrix)
    X_train_fused = concatenate_feature_blocks(X_train_tabular, X_train_image)
    X_validation_fused = concatenate_feature_blocks(X_validation_tabular, X_validation_image)

    validation_trials = train_validation_trials(
        X_train_fused=X_train_fused,
        y_train=y_train,
        X_validation_fused=X_validation_fused,
        y_validation=y_validation,
    )
    best_result = select_best_trial(validation_trials)
    validation_leaderboard = build_validation_leaderboard(validation_trials)

    LOGGER.info(
        "Final model yeniden egitiliyor | effective_n_estimators=%s",
        best_result.effective_n_estimators,
    )
    train_validation_df = pd.concat([train_df, validation_df], axis=0, ignore_index=True)
    y_train_validation = train_validation_df[TARGET_COLUMN]
    y_train_validation_log = to_log_target(y_train_validation)

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

    train_validation_image_matrix = extract_image_matrix(train_validation_df, image_feature_columns)
    test_image_matrix = extract_image_matrix(test_df, image_feature_columns)

    final_image_processor = build_image_processor()
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

    final_model = XGBRegressor(**build_xgb_params(n_estimators=best_result.effective_n_estimators))
    final_model.fit(X_train_validation_fused, y_train_validation_log)
    fusion_predictions_log = np.asarray(final_model.predict(X_test_fused), dtype=float)
    fusion_predictions = inverse_log_predictions(fusion_predictions_log)
    fusion_metrics = calculate_metrics(y_test, fusion_predictions)
    LOGGER.info(
        "Final test | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
        fusion_metrics["mae"],
        fusion_metrics["rmse"],
        fusion_metrics["r2"],
        fusion_metrics["mape"],
    )

    baseline_predictions, baseline_subset_metrics = train_matched_tabular_baseline(
        train_validation_df=train_validation_df,
        test_df=test_df,
    )
    prediction_frame = build_prediction_frame(
        test_df=test_df,
        baseline_predictions=baseline_predictions,
        fusion_predictions=fusion_predictions,
    )

    high_price_df = summarize_segment_metrics(
        prediction_frame=prediction_frame,
        segment_name="high_price_75k_plus",
        mask=prediction_frame["high_price_segment"],
    )
    luxury_df = summarize_segment_metrics(
        prediction_frame=prediction_frame,
        segment_name="luxury_proxy",
        mask=prediction_frame["luxury_segment"],
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
    residual_summary_df = summarize_residuals(prediction_frame)
    worst_cases_df = select_worst_cases(prediction_frame, top_n=20)

    image_ablation_metrics, image_ablation_note = compute_reduced_image_ablation(
        model=final_model,
        X_test=X_test_fused,
        y_test=y_test,
    )
    comparison_table = build_comparison_table(
        baseline_subset_metrics=baseline_subset_metrics,
        previous_reduced_metrics=PREVIOUS_REDUCED_FUSION_METRICS,
        previous_all_image_metrics=PREVIOUS_ALL_IMAGE_REDUCED_METRICS,
        fusion_metrics=fusion_metrics,
    )

    feature_names = [str(name) for name in final_tabular_preprocessor.get_feature_names_out()]
    feature_names.extend(f"{REDUCED_IMAGE_PREFIX}{index}" for index in range(REDUCED_IMAGE_DIM))

    report_body = build_report(
        join_count=len(joined_df),
        original_embedding_dimension=original_embedding_dimension,
        best_result=best_result,
        validation_leaderboard=validation_leaderboard,
        fusion_metrics=fusion_metrics,
        baseline_subset_metrics=baseline_subset_metrics,
        comparison_table=comparison_table,
        high_price_df=high_price_df,
        luxury_df=luxury_df,
        district_improvement_df=district_improvement_df,
        price_improvement_df=price_improvement_df,
        m2_improvement_df=m2_improvement_df,
        residual_summary_df=residual_summary_df,
        image_ablation_metrics=image_ablation_metrics,
        image_ablation_note=image_ablation_note,
        worst_cases_df=worst_cases_df,
    )

    save_model_bundle(
        best_result=best_result,
        tabular_preprocessor=final_tabular_preprocessor,
        image_processor=final_image_processor,
        model=final_model,
        feature_names=feature_names,
        join_count=len(joined_df),
        original_embedding_dimension=original_embedding_dimension,
        fusion_metrics=fusion_metrics,
        baseline_subset_metrics=baseline_subset_metrics,
    )
    LOGGER.info("Rapor kaydediliyor: %s", REPORT_OUTPUT_PATH)
    REPORT_OUTPUT_PATH.write_text(report_body, encoding="utf-8")

    LOGGER.info("Tamamlandi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
