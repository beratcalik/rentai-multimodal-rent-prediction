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
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Eksik bagimlilik bulundu. Lutfen once `python -m pip install -r requirements.txt` calistirin."
    ) from exc


LOGGER = logging.getLogger("image_fusion_reduced_training")

ROOT_DIR = Path(__file__).resolve().parent.parent
MULTIMODAL_DATASET_PATH = ROOT_DIR / "dataset" / "train_ready_multimodal.parquet"
IMAGE_EMBEDDINGS_PATH = ROOT_DIR / "dataset" / "image_embeddings.parquet"
MODEL_OUTPUT_PATH = ROOT_DIR / "models" / "image_fusion_reduced_model.joblib"
REPORT_OUTPUT_PATH = ROOT_DIR / "reports" / "image_fusion_reduced_results.md"

VALID_IMAGE_COUNT_COLUMN = "valid_image_count"
IMAGE_EMBEDDING_COLUMN = "image_embedding"
IMAGE_FEATURE_PREFIX = "image_emb_"
REDUCED_IMAGE_PREFIX = "image_reduced_"
REDUCER_NAME = "PCA"
REDUCED_IMAGE_DIMS = [16, 32, 64, 128, 256]
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


@dataclass
class ValidationTrial:
    model_label: str
    image_dim: int
    validation_metrics: dict[str, float]
    factory: Callable[[], Any]


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

    joined_df = multimodal_df.merge(
        embeddings_expanded_df,
        on="listing_id",
        how="inner",
    )
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


def concatenate_feature_blocks(tabular_block: np.ndarray, image_block: np.ndarray) -> np.ndarray:
    fused_block = np.hstack([tabular_block, image_block])
    return np.asarray(fused_block, dtype=np.float32)


def build_model_factories() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    optional_model_tracking: dict[str, dict[str, Any]] = {
        "LightGBMRegressor": {
            "import_status": "not_attempted",
            "success_dims": [],
            "failure_notes": [],
        },
        "XGBRegressor": {
            "import_status": "not_attempted",
            "success_dims": [],
            "failure_notes": [],
        },
    }

    try:
        from lightgbm import LGBMRegressor

        optional_model_tracking["LightGBMRegressor"]["import_status"] = "ok"
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
                    objective="regression",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        optional_model_tracking["LightGBMRegressor"]["import_status"] = (
            f"import hatasi: {exc.__class__.__name__}: {exc}"
        )

    try:
        from xgboost import XGBRegressor

        optional_model_tracking["XGBRegressor"]["import_status"] = "ok"
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
        optional_model_tracking["XGBRegressor"]["import_status"] = (
            f"import hatasi: {exc.__class__.__name__}: {exc}"
        )

    candidates.extend(
        [
            {
                "label": "HistGradientBoostingRegressor",
                "factory": partial(
                    HistGradientBoostingRegressor,
                    learning_rate=0.05,
                    max_iter=400,
                    min_samples_leaf=20,
                    random_state=RANDOM_STATE,
                ),
            },
            {
                "label": "Ridge",
                "factory": partial(Ridge, alpha=5.0),
            },
            {
                "label": "RandomForestRegressor",
                "factory": partial(
                    RandomForestRegressor,
                    n_estimators=250,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            },
            {
                "label": "MLPRegressor",
                "factory": partial(
                    MLPRegressor,
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    learning_rate_init=0.001,
                    batch_size=128,
                    early_stopping=True,
                    max_iter=250,
                    n_iter_no_change=20,
                    random_state=RANDOM_STATE,
                ),
            },
        ]
    )
    return candidates, optional_model_tracking


def train_validation_trials(
    candidates: list[dict[str, Any]],
    X_train_tabular: np.ndarray,
    train_image_matrix: np.ndarray,
    y_train: pd.Series,
    X_validation_tabular: np.ndarray,
    validation_image_matrix: np.ndarray,
    y_validation: pd.Series,
    optional_model_tracking: dict[str, dict[str, Any]],
) -> list[ValidationTrial]:
    results: list[ValidationTrial] = []

    for image_dim in REDUCED_IMAGE_DIMS:
        LOGGER.info("%s reduction fit ediliyor | image_dim=%s", REDUCER_NAME, image_dim)
        image_processor = build_image_processor(image_dim)
        X_train_image = image_processor.fit_transform(train_image_matrix)
        X_validation_image = image_processor.transform(validation_image_matrix)

        X_train_fused = concatenate_feature_blocks(X_train_tabular, X_train_image)
        X_validation_fused = concatenate_feature_blocks(X_validation_tabular, X_validation_image)

        for candidate in candidates:
            model_label = candidate["label"]
            LOGGER.info("Model egitiliyor | image_dim=%s | %s", image_dim, model_label)
            try:
                model = candidate["factory"]()
                model.fit(X_train_fused, y_train)
                validation_predictions = np.asarray(model.predict(X_validation_fused), dtype=float)
                validation_metrics = calculate_metrics(y_validation, validation_predictions)
                results.append(
                    ValidationTrial(
                        model_label=model_label,
                        image_dim=image_dim,
                        validation_metrics=validation_metrics,
                        factory=candidate["factory"],
                    )
                )
                LOGGER.info(
                    "Validation | image_dim=%s | %s | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
                    image_dim,
                    model_label,
                    validation_metrics["mae"],
                    validation_metrics["rmse"],
                    validation_metrics["r2"],
                    validation_metrics["mape"],
                )
                if model_label in optional_model_tracking:
                    optional_model_tracking[model_label]["success_dims"].append(image_dim)
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.exception(
                    "Model denemesi basarisiz oldu | image_dim=%s | %s",
                    image_dim,
                    model_label,
                )
                if model_label in optional_model_tracking:
                    optional_model_tracking[model_label]["failure_notes"].append(
                        f"image_dim={image_dim} egitim hatasi: {exc.__class__.__name__}: {exc}"
                    )

    if not results:
        raise RuntimeError("Hicbir reduced image fusion modeli basariyla egitilemedi.")

    return results


def select_best_trial(results: list[ValidationTrial]) -> ValidationTrial:
    best_result = min(results, key=lambda item: item.validation_metrics[PRIMARY_SELECTION_METRIC])
    LOGGER.info(
        "En iyi kombinasyon secildi: %s + image_dim=%s (validation MAE=%.2f)",
        best_result.model_label,
        best_result.image_dim,
        best_result.validation_metrics["mae"],
    )
    return best_result


def build_validation_leaderboard(results: list[ValidationTrial]) -> pd.DataFrame:
    rows = [
        {
            "reducer": REDUCER_NAME,
            "image_dim": result.image_dim,
            "model": result.model_label,
            "validation_mae": result.validation_metrics["mae"],
            "validation_rmse": result.validation_metrics["rmse"],
            "validation_r2": result.validation_metrics["r2"],
            "validation_mape": result.validation_metrics["mape"],
        }
        for result in results
    ]
    leaderboard_df = pd.DataFrame(rows)
    return leaderboard_df.sort_values(
        by=["validation_mae", "validation_rmse", "image_dim", "model"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def build_image_dim_summary(validation_leaderboard: pd.DataFrame) -> pd.DataFrame:
    if validation_leaderboard.empty:
        return pd.DataFrame(
            columns=[
                "image_dim",
                "best_model",
                "best_validation_mae",
                "best_validation_rmse",
                "best_validation_r2",
                "best_validation_mape",
            ]
        )

    summary_rows: list[dict[str, Any]] = []
    for image_dim in REDUCED_IMAGE_DIMS:
        image_dim_df = validation_leaderboard[validation_leaderboard["image_dim"] == image_dim]
        if image_dim_df.empty:
            continue
        best_row = image_dim_df.iloc[0]
        summary_rows.append(
            {
                "image_dim": image_dim,
                "best_model": best_row["model"],
                "best_validation_mae": float(best_row["validation_mae"]),
                "best_validation_rmse": float(best_row["validation_rmse"]),
                "best_validation_r2": float(best_row["validation_r2"]),
                "best_validation_mape": float(best_row["validation_mape"]),
            }
        )

    return pd.DataFrame(summary_rows)


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
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
    image_dim: int,
) -> tuple[dict[str, float], str]:
    if image_dim <= 0:
        raise ValueError("Image dim pozitif olmali.")

    X_without_image = np.array(X_test, copy=True)
    X_without_image[:, -image_dim:] = 0.0
    no_image_predictions = np.asarray(model.predict(X_without_image), dtype=float)
    no_image_metrics = calculate_metrics(y_test, no_image_predictions)
    return no_image_metrics, (
        "Reduced image block standardized ve PCA ile olusturuldugu icin ablasyon testinde son image block sifira cekildi; "
        "bu yaklasim reduced image branch katkisini ayirmak icin kullanildi."
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
            "row": "Reduced image fusion model",
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
    ]
    return pd.DataFrame(rows)


def render_optional_model_status(optional_model_tracking: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for model_name in ["LightGBMRegressor", "XGBRegressor"]:
        tracking = optional_model_tracking[model_name]
        import_status = tracking["import_status"]
        success_dims = sorted(set(int(value) for value in tracking["success_dims"]))
        failure_notes = tracking["failure_notes"]

        if import_status != "ok":
            lines.append(f"- `{model_name}`: {import_status}")
            continue

        if success_dims:
            dims_text = ", ".join(str(value) for value in success_dims)
            if failure_notes:
                lines.append(
                    f"- `{model_name}`: calisti; validation denemelerinde image_dim={dims_text} icin egitildi. Ilk hata notu: {failure_notes[0]}"
                )
            else:
                lines.append(
                    f"- `{model_name}`: calisti; validation denemelerinde image_dim={dims_text} icin egitildi."
                )
            continue

        if failure_notes:
            lines.append(
                f"- `{model_name}`: import edildi ama calismadi; {failure_notes[0]}"
            )
            continue

        lines.append(
            f"- `{model_name}`: import edildi ama validation denemelerine ulasmadan bilinmeyen bir durum olustu."
        )

    return "\n".join(lines) if lines else "_Veri yok_"


def build_report(
    join_count: int,
    original_embedding_dimension: int,
    best_result: ValidationTrial,
    validation_leaderboard: pd.DataFrame,
    image_dim_summary_df: pd.DataFrame,
    fusion_metrics: dict[str, float],
    baseline_subset_metrics: dict[str, float],
    comparison_table: pd.DataFrame,
    district_improvement_df: pd.DataFrame,
    price_improvement_df: pd.DataFrame,
    m2_improvement_df: pd.DataFrame,
    image_ablation_metrics: dict[str, float],
    image_ablation_note: str,
    worst_cases_df: pd.DataFrame,
    optional_model_tracking: dict[str, dict[str, Any]],
) -> str:
    optional_model_section = render_optional_model_status(optional_model_tracking)

    if image_ablation_metrics["mae"] > fusion_metrics["mae"] and fusion_metrics["mae"] >= baseline_subset_metrics["mae"]:
        contribution_comment = (
            f"Reduced image block model icinde pozitif sinyal uretiyor; block sifirlandiginda MAE {image_ablation_metrics['mae'] - fusion_metrics['mae']:.2f} kadar kotulesiyor. "
            "Buna ragmen mevcut fusion kurulumunda matched tabular baseline halen daha iyi."
        )
    elif image_ablation_metrics["mae"] > fusion_metrics["mae"]:
        contribution_comment = (
            f"Reduced image block net pozitif katkili; block sifirlandiginda MAE {image_ablation_metrics['mae'] - fusion_metrics['mae']:.2f} kadar kotulesiyor."
        )
    else:
        contribution_comment = (
            f"Bu kosuda reduced image block sifirlandiginda MAE {fusion_metrics['mae'] - image_ablation_metrics['mae']:.2f} kadar iyilesiyor; "
            "image branch sinyali henuz en verimli sekilde kullanilmiyor olabilir."
        )

    report_lines = [
        "# Reduced Image Fusion Results",
        "",
        "## Ozet",
        "",
        f"- Multimodal source: `{MULTIMODAL_DATASET_PATH}`",
        f"- Image embedding source: `{IMAGE_EMBEDDINGS_PATH}`",
        f"- Kaydedilen model bundle: `{MODEL_OUTPUT_PATH}`",
        f"- Join sonucu kalan ornek sayisi: **{join_count:,}**",
        f"- Original image embedding dimension: **{original_embedding_dimension}**",
        f"- Reducer: **{REDUCER_NAME}**",
        f"- Denenen reduced image_dim degerleri: **{', '.join(str(value) for value in REDUCED_IMAGE_DIMS)}**",
        f"- En iyi kombinasyon: **{best_result.model_label} + image_dim={best_result.image_dim}**",
        "",
        "## Validation Leaderboard",
        "",
        dataframe_to_markdown_table(validation_leaderboard, digits=4),
        "",
        "## Image Dim Bazli Skorlar",
        "",
        dataframe_to_markdown_table(image_dim_summary_df, digits=4),
        "",
        "## Test Sonuclari",
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
        "## Matched Baseline vs Reduced Image Fusion",
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
        "## Image Branch Katkisi",
        "",
        f"- Matched tabular baseline MAE: **{baseline_subset_metrics['mae']:.2f}**",
        f"- Reduced image fusion MAE: **{fusion_metrics['mae']:.2f}**",
        f"- Reduced image block sifirlandiginda MAE: **{image_ablation_metrics['mae']:.2f}**",
        f"- Reduced image block sifirlandiginda MAPE: **{image_ablation_metrics['mape']:.2f}%**",
        f"- Ablasyon yorumu: {image_ablation_note}",
        f"- Yorum: {contribution_comment}",
        "",
        "## En Yuksek Hata Yapan 20 Ilan",
        "",
        dataframe_to_markdown_table(worst_cases_df, digits=4),
        "",
        "## LightGBM ve XGBoost Durumu",
        "",
        optional_model_section,
    ]
    return "\n".join(report_lines)


def save_model_bundle(
    best_result: ValidationTrial,
    tabular_preprocessor: ColumnTransformer,
    image_processor: Pipeline,
    model: Any,
    feature_names: list[str],
    join_count: int,
    original_embedding_dimension: int,
    fusion_metrics: dict[str, float],
    baseline_subset_metrics: dict[str, float],
    optional_model_tracking: dict[str, dict[str, Any]],
) -> None:
    bundle = {
        "model_name": best_result.model_label,
        "reducer_name": REDUCER_NAME,
        "reduced_image_dim": best_result.image_dim,
        "tabular_preprocessor": tabular_preprocessor,
        "image_processor": image_processor,
        "regressor": model,
        "feature_names": feature_names,
        "tabular_feature_columns": FUSION_TABULAR_COLUMNS,
        "join_count": join_count,
        "original_embedding_dimension": original_embedding_dimension,
        "fusion_metrics": fusion_metrics,
        "baseline_subset_metrics": baseline_subset_metrics,
        "optional_model_tracking": optional_model_tracking,
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

    candidates, optional_model_tracking = build_model_factories()
    validation_trials = train_validation_trials(
        candidates=candidates,
        X_train_tabular=X_train_tabular,
        train_image_matrix=train_image_matrix,
        y_train=y_train,
        X_validation_tabular=X_validation_tabular,
        validation_image_matrix=validation_image_matrix,
        y_validation=y_validation,
        optional_model_tracking=optional_model_tracking,
    )
    best_result = select_best_trial(validation_trials)
    validation_leaderboard = build_validation_leaderboard(validation_trials)
    image_dim_summary_df = build_image_dim_summary(validation_leaderboard)

    LOGGER.info(
        "Final model yeniden egitiliyor | %s + image_dim=%s",
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

    train_validation_image_matrix = extract_image_matrix(train_validation_df, image_feature_columns)
    test_image_matrix = extract_image_matrix(test_df, image_feature_columns)

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
    fusion_predictions = np.asarray(final_model.predict(X_test_fused), dtype=float)
    fusion_metrics = calculate_metrics(y_test, fusion_predictions)
    LOGGER.info(
        "Final test | %s + image_dim=%s | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
        best_result.model_label,
        best_result.image_dim,
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
    worst_cases_df = select_worst_cases(prediction_frame, top_n=20)

    image_ablation_metrics, image_ablation_note = compute_reduced_image_ablation(
        model=final_model,
        X_test=X_test_fused,
        y_test=y_test,
        image_dim=best_result.image_dim,
    )
    comparison_table = build_comparison_table(
        baseline_subset_metrics=baseline_subset_metrics,
        fusion_metrics=fusion_metrics,
    )

    feature_names = [str(name) for name in final_tabular_preprocessor.get_feature_names_out()]
    feature_names.extend(
        f"{REDUCED_IMAGE_PREFIX}{index}" for index in range(best_result.image_dim)
    )

    report_body = build_report(
        join_count=len(joined_df),
        original_embedding_dimension=original_embedding_dimension,
        best_result=best_result,
        validation_leaderboard=validation_leaderboard,
        image_dim_summary_df=image_dim_summary_df,
        fusion_metrics=fusion_metrics,
        baseline_subset_metrics=baseline_subset_metrics,
        comparison_table=comparison_table,
        district_improvement_df=district_improvement_df,
        price_improvement_df=price_improvement_df,
        m2_improvement_df=m2_improvement_df,
        image_ablation_metrics=image_ablation_metrics,
        image_ablation_note=image_ablation_note,
        worst_cases_df=worst_cases_df,
        optional_model_tracking=optional_model_tracking,
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
        optional_model_tracking=optional_model_tracking,
    )
    LOGGER.info("Rapor kaydediliyor: %s", REPORT_OUTPUT_PATH)
    REPORT_OUTPUT_PATH.write_text(report_body, encoding="utf-8")

    LOGGER.info("Tamamlandi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
