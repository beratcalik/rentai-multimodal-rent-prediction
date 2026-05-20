from __future__ import annotations

import logging
import re
import sys
import unicodedata
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

import train_baseline as train_baseline_module
from train_baseline import (
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    FEATURE_COLUMNS,
    MODEL_OUTPUT_PATH as BASELINE_MODEL_PATH,
    NUMERIC_COLUMNS,
    PREPROCESSOR_OUTPUT_PATH as BASELINE_PREPROCESSOR_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    calculate_metrics,
    configure_logging,
    load_dataset,
    normalize_furnished_flag,
    normalize_string,
    parse_floor_value,
    parse_room_count,
    split_dataset,
)

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Eksik bagimlilik bulundu. Lutfen once `python -m pip install -r requirements.txt` calistirin."
    ) from exc


LOGGER = logging.getLogger("text_model_training")

ROOT_DIR = Path(__file__).resolve().parent.parent
TEXT_MODEL_OUTPUT_PATH = ROOT_DIR / "models" / "text_model.joblib"
TEXT_REPORT_OUTPUT_PATH = ROOT_DIR / "reports" / "text_model_results.md"

TITLE_COLUMN = "title"
DESCRIPTION_COLUMN = "description"
TEXT_MIN_TOKEN_COUNT = 2
TEXT_COLUMNS = [TITLE_COLUMN, DESCRIPTION_COLUMN]

TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
SVD_COMPONENTS = 256

BASELINE_REFERENCE_METRICS = {
    "mae": 4683.35,
    "rmse": 6911.91,
    "r2": 0.7632,
    "mape": 14.85,
}

PRICE_BAND_LUXURY_THRESHOLD = 75_000.0
LUXURY_KEYWORDS = {
    "ebeveyn",
    "guvenlik",
    "lux",
    "luks",
    "manzarali",
    "rezidans",
    "residence",
    "teras",
    "ultra",
}

BOILERPLATE_PATTERNS = [
    "telefonu goster",
    "detayli bilgi",
    "arayiniz",
    "gayrimenkul",
    "emlak",
    "kahve icmeye",
]


def ensure_output_directories() -> None:
    TEXT_MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEXT_REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_sparse_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover - sklearn compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_tabular_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_sparse_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def clean_tabular_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.loc[:, FEATURE_COLUMNS].copy()

    for column in CATEGORICAL_COLUMNS:
        if column == "is_furnished":
            frame[column] = frame[column].map(normalize_furnished_flag)
        else:
            frame[column] = frame[column].map(normalize_string)

    frame["rooms"] = frame["rooms"].map(parse_room_count)
    frame["floor"] = frame["floor"].map(parse_floor_value)

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def collapse_whitespace(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def strip_combining_marks(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def clean_text_value(value: Any) -> str:
    text = collapse_whitespace(value)
    if not text:
        return ""

    text = strip_combining_marks(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(rf"\b{re.escape(pattern)}\b", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    if len(text.split()) < TEXT_MIN_TOKEN_COUNT:
        return ""
    return text


def build_clean_text_columns(dataframe: pd.DataFrame) -> dict[str, pd.Series]:
    title_clean = dataframe[TITLE_COLUMN].map(clean_text_value)
    description_clean = dataframe[DESCRIPTION_COLUMN].map(clean_text_value)
    combined = (title_clean + " " + description_clean).str.strip()
    combined = combined.map(lambda text: text if len(text.split()) >= TEXT_MIN_TOKEN_COUNT else "")

    return {
        "title_clean": title_clean,
        "description_clean": description_clean,
        "combined_text": combined,
    }


def ensure_csr_matrix(matrix: Any) -> sp.csr_matrix:
    if sp.issparse(matrix):
        return matrix.tocsr()
    return sp.csr_matrix(matrix)


def to_dense_array(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def fit_feature_artifacts(fit_df: pd.DataFrame, representation: str) -> dict[str, Any]:
    if representation not in {"tfidf", "svd"}:
        raise ValueError(f"Bilinmeyen representation: {representation}")

    tabular_preprocessor = build_tabular_preprocessor()
    cleaned_tabular = clean_tabular_features(fit_df)
    tabular_preprocessor.fit(cleaned_tabular)
    tabular_feature_names = [str(name) for name in tabular_preprocessor.get_feature_names_out()]

    cleaned_text = build_clean_text_columns(fit_df)["combined_text"]
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
    )
    text_tfidf = vectorizer.fit_transform(cleaned_text)
    raw_text_feature_names = [str(name) for name in vectorizer.get_feature_names_out()]

    artifacts: dict[str, Any] = {
        "representation": representation,
        "tabular_preprocessor": tabular_preprocessor,
        "text_vectorizer": vectorizer,
        "raw_text_feature_names": raw_text_feature_names,
        "tabular_feature_names": tabular_feature_names,
        "text_svd": None,
        "text_feature_names": [f"text_tfidf__{name}" for name in raw_text_feature_names],
    }

    if representation == "svd":
        component_count = min(SVD_COMPONENTS, max(2, text_tfidf.shape[1] - 1))
        svd = TruncatedSVD(n_components=component_count, random_state=RANDOM_STATE)
        svd.fit(text_tfidf)
        artifacts["text_svd"] = svd
        artifacts["text_feature_names"] = [
            f"text_svd_{index:03d}"
            for index in range(component_count)
        ]

    artifacts["combined_feature_names"] = (
        tabular_feature_names + list(artifacts["text_feature_names"])
    )
    return artifacts


def transform_with_artifacts(artifacts: dict[str, Any], dataframe: pd.DataFrame) -> Any:
    cleaned_tabular = clean_tabular_features(dataframe)
    tabular_matrix = artifacts["tabular_preprocessor"].transform(cleaned_tabular)
    cleaned_text = build_clean_text_columns(dataframe)["combined_text"]
    tfidf_matrix = artifacts["text_vectorizer"].transform(cleaned_text)

    if artifacts["representation"] == "tfidf":
        return sp.hstack(
            [ensure_csr_matrix(tabular_matrix), tfidf_matrix],
            format="csr",
        )

    dense_tabular = to_dense_array(tabular_matrix)
    text_svd = artifacts["text_svd"].transform(tfidf_matrix)
    return np.hstack([dense_tabular, text_svd])


def build_candidate_specs() -> tuple[list[dict[str, Any]], dict[str, str]]:
    candidates: list[dict[str, Any]] = [
        {
            "label": "HistGradientBoosting + SVD",
            "representation": "svd",
            "factory": partial(
                HistGradientBoostingRegressor,
                learning_rate=0.05,
                max_iter=400,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            ),
            "supports_token_coefficients": False,
        },
        {
            "label": "Ridge + TFIDF",
            "representation": "tfidf",
            "factory": partial(
                Ridge,
                alpha=3.0,
            ),
            "supports_token_coefficients": True,
        },
        {
            "label": "Ridge + SVD",
            "representation": "svd",
            "factory": partial(
                Ridge,
                alpha=3.0,
            ),
            "supports_token_coefficients": False,
        },
        {
            "label": "ElasticNet + TFIDF",
            "representation": "tfidf",
            "factory": partial(
                ElasticNet,
                alpha=0.0005,
                l1_ratio=0.20,
                max_iter=5000,
                selection="random",
                tol=1e-3,
                random_state=RANDOM_STATE,
            ),
            "supports_token_coefficients": True,
        },
        {
            "label": "ElasticNet + SVD",
            "representation": "svd",
            "factory": partial(
                ElasticNet,
                alpha=0.0010,
                l1_ratio=0.20,
                max_iter=5000,
                selection="random",
                tol=1e-3,
                random_state=RANDOM_STATE,
            ),
            "supports_token_coefficients": False,
        },
    ]
    skipped_models: dict[str, str] = {}

    try:
        from lightgbm import LGBMRegressor

        candidates.append(
            {
                "label": "LightGBM + SVD",
                "representation": "svd",
                "factory": partial(
                    LGBMRegressor,
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                "supports_token_coefficients": False,
            }
        )
    except ModuleNotFoundError:
        skipped_models["LightGBM + SVD"] = "lightgbm kurulu degil"

    try:
        from xgboost import XGBRegressor

        candidates.append(
            {
                "label": "XGBoost + SVD",
                "representation": "svd",
                "factory": partial(
                    XGBRegressor,
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=0,
                ),
                "supports_token_coefficients": False,
            }
        )
    except ModuleNotFoundError:
        skipped_models["XGBoost + SVD"] = "xgboost kurulu degil"

    return candidates, skipped_models


def build_representation_cache(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for representation in ("tfidf", "svd"):
        LOGGER.info("Feature artifacts fit ediliyor: %s", representation)
        artifacts = fit_feature_artifacts(train_df, representation)
        cache[representation] = {
            "artifacts": artifacts,
            "train_X": transform_with_artifacts(artifacts, train_df),
            "validation_X": transform_with_artifacts(artifacts, validation_df),
        }
    return cache


def train_candidate_models(
    candidates: list[dict[str, Any]],
    representation_cache: dict[str, dict[str, Any]],
    y_train: pd.Series,
    y_validation: pd.Series,
    skipped_models: dict[str, str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for candidate in candidates:
        LOGGER.info("Model egitiliyor: %s", candidate["label"])
        try:
            model = candidate["factory"]()
            cache_item = representation_cache[candidate["representation"]]
            model.fit(cache_item["train_X"], y_train)
            validation_predictions = model.predict(cache_item["validation_X"])
            validation_metrics = calculate_metrics(y_validation, validation_predictions)
            LOGGER.info(
                "%s validation | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
                candidate["label"],
                validation_metrics["mae"],
                validation_metrics["rmse"],
                validation_metrics["r2"],
                validation_metrics["mape"],
            )
            results.append(
                {
                    "label": candidate["label"],
                    "representation": candidate["representation"],
                    "validation_metrics": validation_metrics,
                    "factory": candidate["factory"],
                    "supports_token_coefficients": candidate["supports_token_coefficients"],
                }
            )
        except Exception as exc:  # pragma: no cover - defensive path
            skipped_models[candidate["label"]] = f"egitim hatasi: {exc.__class__.__name__}: {exc}"
            LOGGER.exception("Model atlandi: %s", candidate["label"])

    if not results:
        raise RuntimeError("Hicbir text-enhanced model basariyla egitilemedi.")

    return results


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    best_result = min(results, key=lambda item: item["validation_metrics"]["mae"])
    LOGGER.info(
        "En iyi candidate secildi: %s (validation MAE=%.2f)",
        best_result["label"],
        best_result["validation_metrics"]["mae"],
    )
    return best_result


def load_baseline_artifacts() -> tuple[Any, Any]:
    setattr(sys.modules["__main__"], "PropertyFeatureCleaner", train_baseline_module.PropertyFeatureCleaner)

    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    baseline_preprocessor = joblib.load(BASELINE_PREPROCESSOR_PATH)
    return baseline_model, baseline_preprocessor


def build_baseline_predictions(test_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    baseline_model, baseline_preprocessor = load_baseline_artifacts()
    transformed_test = baseline_preprocessor.transform(test_df[FEATURE_COLUMNS])
    predictions = np.asarray(baseline_model.predict(transformed_test), dtype=float)
    metrics = calculate_metrics(test_df[TARGET_COLUMN], predictions)
    return predictions, metrics


def compute_feature_importance(
    model: Any,
    feature_names: list[str],
    X_test: Any,
    y_test: pd.Series,
) -> tuple[str, np.ndarray, np.ndarray | None]:
    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float).ravel()
        importances = np.abs(coefficients)
        return "absolute_coefficients", importances, coefficients

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float).ravel()
        return "native_feature_importances", importances, None

    sample_size = min(len(y_test), 400)
    if sp.issparse(X_test):
        X_sample = X_test[:sample_size]
    else:
        X_sample = np.asarray(X_test)[:sample_size]

    permutation = permutation_importance(
        model,
        X_sample,
        y_test.iloc[:sample_size],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importances = np.asarray(permutation.importances_mean, dtype=float).ravel()
    return "permutation_importance", importances, None


def build_top_feature_table(
    feature_names: list[str],
    importances: np.ndarray,
    signed_values: np.ndarray | None = None,
    top_n: int = 20,
    prefix_filter: str | None = None,
) -> pd.DataFrame:
    feature_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    if signed_values is not None:
        feature_df["signed_value"] = signed_values

    if prefix_filter is not None:
        feature_df = feature_df[feature_df["feature"].str.startswith(prefix_filter)].copy()

    feature_df = feature_df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return feature_df


def extract_linear_token_tables(
    model: Any,
    artifacts: dict[str, Any],
    top_n: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficients = np.asarray(model.coef_, dtype=float).ravel()
    start_index = len(artifacts["tabular_feature_names"])
    token_names = list(artifacts["raw_text_feature_names"])
    token_coefficients = coefficients[start_index : start_index + len(token_names)]

    token_df = pd.DataFrame(
        {
            "token": token_names,
            "coefficient": token_coefficients,
            "abs_coefficient": np.abs(token_coefficients),
        }
    )
    positive_df = token_df.sort_values("coefficient", ascending=False).head(top_n).reset_index(drop=True)
    negative_df = token_df.sort_values("coefficient", ascending=True).head(top_n).reset_index(drop=True)
    absolute_df = token_df.sort_values("abs_coefficient", ascending=False).head(top_n).reset_index(drop=True)
    return positive_df, negative_df, absolute_df


def build_svd_component_token_table(
    artifacts: dict[str, Any],
    feature_importance_table: pd.DataFrame,
    top_n: int = 10,
    tokens_per_component: int = 8,
) -> pd.DataFrame:
    if artifacts["representation"] != "svd" or artifacts["text_svd"] is None:
        return pd.DataFrame()

    svd_components = artifacts["text_svd"].components_
    raw_tokens = np.asarray(artifacts["raw_text_feature_names"])

    text_rows: list[dict[str, Any]] = []
    for _, row in feature_importance_table.iterrows():
        feature_name = str(row["feature"])
        if not feature_name.startswith("text_svd_"):
            continue

        component_index = int(feature_name.rsplit("_", 1)[-1])
        component_weights = svd_components[component_index]
        top_token_indices = np.argsort(np.abs(component_weights))[::-1][:tokens_per_component]
        top_tokens = [str(raw_tokens[index]) for index in top_token_indices]
        text_rows.append(
            {
                "component": feature_name,
                "importance": float(row["importance"]),
                "top_tokens": ", ".join(top_tokens),
            }
        )
        if len(text_rows) >= top_n:
            break

    return pd.DataFrame(text_rows)


def build_prediction_frame(
    test_df: pd.DataFrame,
    baseline_predictions: np.ndarray,
    text_predictions: np.ndarray,
) -> pd.DataFrame:
    clean_text = build_clean_text_columns(test_df)
    frame = test_df.loc[:, ["listing_id", "district", TITLE_COLUMN, DESCRIPTION_COLUMN]].copy()
    frame["combined_clean_text"] = clean_text["combined_text"]
    frame["actual_price_try"] = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    frame["baseline_prediction"] = np.asarray(baseline_predictions, dtype=float)
    frame["text_prediction"] = np.asarray(text_predictions, dtype=float)
    frame["baseline_abs_error"] = np.abs(frame["baseline_prediction"] - frame["actual_price_try"])
    frame["text_abs_error"] = np.abs(frame["text_prediction"] - frame["actual_price_try"])
    frame["baseline_ape_pct"] = (
        frame["baseline_abs_error"] / frame["actual_price_try"].clip(lower=1e-8)
    ) * 100.0
    frame["text_ape_pct"] = (
        frame["text_abs_error"] / frame["actual_price_try"].clip(lower=1e-8)
    ) * 100.0
    return frame


def summarize_district_improvement(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        prediction_frame.groupby("district", dropna=False)
        .agg(
            sample_count=("listing_id", "size"),
            baseline_mae=("baseline_abs_error", "mean"),
            text_mae=("text_abs_error", "mean"),
            baseline_mape=("baseline_ape_pct", "mean"),
            text_mape=("text_ape_pct", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["sample_count"] >= 5].copy()
    grouped["mae_improvement"] = grouped["baseline_mae"] - grouped["text_mae"]
    grouped["mape_improvement"] = grouped["baseline_mape"] - grouped["text_mape"]
    grouped = grouped.sort_values("mae_improvement", ascending=False).reset_index(drop=True)
    return grouped


def contains_luxury_keyword(clean_text: str) -> bool:
    tokens = set(clean_text.split())
    return any(keyword in tokens for keyword in LUXURY_KEYWORDS)


def build_segment_comparison_table(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    high_price_mask = prediction_frame["actual_price_try"] >= PRICE_BAND_LUXURY_THRESHOLD
    keyword_mask = prediction_frame["combined_clean_text"].map(contains_luxury_keyword)
    luxury_proxy_mask = high_price_mask | keyword_mask

    segment_specs = [
        ("All test listings", np.ones(len(prediction_frame), dtype=bool)),
        ("Luxury proxy", luxury_proxy_mask.to_numpy(dtype=bool)),
        ("High-price (>=75k)", high_price_mask.to_numpy(dtype=bool)),
        ("Luxury keywords", keyword_mask.to_numpy(dtype=bool)),
    ]

    rows: list[dict[str, Any]] = []
    for segment_name, mask in segment_specs:
        subset = prediction_frame.loc[mask]
        if subset.empty:
            continue
        rows.append(
            {
                "segment": segment_name,
                "sample_count": int(len(subset)),
                "baseline_mae": float(subset["baseline_abs_error"].mean()),
                "text_mae": float(subset["text_abs_error"].mean()),
                "mae_improvement": float(subset["baseline_abs_error"].mean() - subset["text_abs_error"].mean()),
                "baseline_mape": float(subset["baseline_ape_pct"].mean()),
                "text_mape": float(subset["text_ape_pct"].mean()),
                "mape_improvement": float(subset["baseline_ape_pct"].mean() - subset["text_ape_pct"].mean()),
            }
        )

    return pd.DataFrame(rows)


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


def build_validation_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "candidate": result["label"],
            "representation": result["representation"],
            "validation_mae": result["validation_metrics"]["mae"],
            "validation_rmse": result["validation_metrics"]["rmse"],
            "validation_r2": result["validation_metrics"]["r2"],
            "validation_mape": result["validation_metrics"]["mape"],
        }
        for result in results
    ]
    validation_df = pd.DataFrame(rows)
    return validation_df.sort_values("validation_mae", ascending=True).reset_index(drop=True)


def build_test_comparison_table(
    baseline_metrics: dict[str, float],
    text_metrics: dict[str, float],
) -> pd.DataFrame:
    rows = [
        {
            "model": "Baseline HistGradientBoosting",
            "mae": baseline_metrics["mae"],
            "rmse": baseline_metrics["rmse"],
            "r2": baseline_metrics["r2"],
            "mape": baseline_metrics["mape"],
        },
        {
            "model": "Best text-enhanced model",
            "mae": text_metrics["mae"],
            "rmse": text_metrics["rmse"],
            "r2": text_metrics["r2"],
            "mape": text_metrics["mape"],
        },
        {
            "model": "Improvement vs baseline",
            "mae": baseline_metrics["mae"] - text_metrics["mae"],
            "rmse": baseline_metrics["rmse"] - text_metrics["rmse"],
            "r2": text_metrics["r2"] - baseline_metrics["r2"],
            "mape": baseline_metrics["mape"] - text_metrics["mape"],
        },
    ]
    return pd.DataFrame(rows)


def build_feature_space_summary(
    final_artifacts: dict[str, Any],
    train_validation_df: pd.DataFrame,
) -> pd.DataFrame:
    clean_text = build_clean_text_columns(train_validation_df)
    combined_text = clean_text["combined_text"]
    blank_count = int((combined_text == "").sum())

    rows = [
        {"metric": "train_validation_samples", "value": int(len(train_validation_df))},
        {"metric": "tabular_feature_count", "value": int(len(final_artifacts["tabular_feature_names"]))},
        {"metric": "raw_tfidf_feature_count", "value": int(len(final_artifacts["raw_text_feature_names"]))},
        {"metric": "final_feature_count", "value": int(len(final_artifacts["combined_feature_names"]))},
        {"metric": "representation", "value": final_artifacts["representation"]},
        {"metric": "clean_text_blank_count", "value": blank_count},
        {"metric": "clean_text_blank_ratio_pct", "value": float((blank_count / len(train_validation_df)) * 100.0)},
    ]
    if final_artifacts["text_svd"] is not None:
        rows.append(
            {
                "metric": "svd_components",
                "value": int(final_artifacts["text_svd"].n_components),
            }
        )

    return pd.DataFrame(rows)


def build_report(
    best_result: dict[str, Any],
    validation_table: pd.DataFrame,
    baseline_metrics: dict[str, float],
    text_metrics: dict[str, float],
    comparison_table: pd.DataFrame,
    district_improvement_table: pd.DataFrame,
    luxury_comparison_table: pd.DataFrame,
    feature_space_summary: pd.DataFrame,
    overall_feature_importance: pd.DataFrame,
    text_feature_importance: pd.DataFrame,
    positive_tokens: pd.DataFrame,
    negative_tokens: pd.DataFrame,
    absolute_tokens: pd.DataFrame,
    svd_component_tokens: pd.DataFrame,
    skipped_models: dict[str, str],
    best_raw_linear_label: str | None,
) -> str:
    skipped_section = "_Atlanan model yok_"
    if skipped_models:
        skipped_section = "\n".join(
            f"- `{model_name}`: {reason}"
            for model_name, reason in sorted(skipped_models.items())
        )

    baseline_mape_gain = baseline_metrics["mape"] - text_metrics["mape"]
    baseline_mae_gain = baseline_metrics["mae"] - text_metrics["mae"]

    report_lines = [
        "# Text-Enhanced Tabular Model Results",
        "",
        "## Ozet",
        "",
        f"- Dataset: `{DATASET_PATH}`",
        f"- Kaydedilen model bundle: `{TEXT_MODEL_OUTPUT_PATH}`",
        f"- Kaydedilen rapor: `{TEXT_REPORT_OUTPUT_PATH}`",
        f"- En iyi candidate: **{best_result['label']}**",
        f"- Secilen text representation: **{best_result['representation']}**",
        f"- Baseline MAE referansi: **{BASELINE_REFERENCE_METRICS['mae']:.2f}**",
        f"- Baseline MAPE referansi: **{BASELINE_REFERENCE_METRICS['mape']:.2f}%**",
        f"- Test MAE improvement: **{baseline_mae_gain:.2f}**",
        f"- Test MAPE improvement: **{baseline_mape_gain:.2f} puan**",
        "",
        "## Validation Leaderboard",
        "",
        dataframe_to_markdown_table(validation_table, digits=4),
        "",
        "## Baseline vs Text Karsilastirmasi",
        "",
        dataframe_to_markdown_table(comparison_table, digits=4),
        "",
        "## District Bazli Improvement",
        "",
        dataframe_to_markdown_table(district_improvement_table, digits=4),
        "",
        "## Luxury Listing Improvement",
        "",
        "- Luxury proxy tanimi: `actual price >= 75k` veya temizlenmis metinde `lux/luks/ultra/rezidans/ebeveyn/guvenlik/teras/manzarali` tokenlarindan biri geciyor.",
        "",
        dataframe_to_markdown_table(luxury_comparison_table, digits=4),
        "",
        "## Feature Space Ozet",
        "",
        dataframe_to_markdown_table(feature_space_summary, digits=4),
        "",
        "## Overall Feature Importance",
        "",
        dataframe_to_markdown_table(overall_feature_importance, digits=6),
        "",
        "## Text Feature Importance",
        "",
        dataframe_to_markdown_table(text_feature_importance, digits=6),
        "",
        "## En Faydali Text Tokenlari",
        "",
        (
            f"- En iyi yorumlanabilir raw-TFIDF lineer model: **{best_raw_linear_label}**"
            if best_raw_linear_label
            else "- Raw-TFIDF lineer model bulunamadi."
        ),
        "",
        "### Pozitif Fiyat Sinyali Veren Tokenlar",
        "",
        dataframe_to_markdown_table(positive_tokens, digits=6),
        "",
        "### Negatif Fiyat Sinyali Veren Tokenlar",
        "",
        dataframe_to_markdown_table(negative_tokens, digits=6),
        "",
        "### En Guclu Mutlak Token Katsayilari",
        "",
        dataframe_to_markdown_table(absolute_tokens, digits=6),
        "",
        "## SVD Text Component Analizi",
        "",
        dataframe_to_markdown_table(svd_component_tokens, digits=6),
        "",
        "## Kullanilan Text Cleaning Kurallari",
        "",
        "- Lowercase uygulandi.",
        "- Unicode normalize + combining mark temizligi yapildi.",
        "- Fazla whitespace ve noktalama sadelestirildi.",
        "- Su boilerplate kaliplari kaldirildi: `telefonu goster`, `detayli bilgi`, `arayiniz`, `gayrimenkul`, `emlak`, `kahve icmeye`.",
        f"- {TEXT_MIN_TOKEN_COUNT} token altindaki cok kisa metinler bos kabul edildi.",
        "",
        "## Atlanan Modeller",
        "",
        skipped_section,
    ]

    return "\n".join(report_lines)


def save_model_bundle(
    best_result: dict[str, Any],
    final_artifacts: dict[str, Any],
    final_model: Any,
    baseline_metrics: dict[str, float],
    text_metrics: dict[str, float],
) -> None:
    bundle = {
        "model_name": best_result["label"],
        "representation": best_result["representation"],
        "regressor": final_model,
        "tabular_preprocessor": final_artifacts["tabular_preprocessor"],
        "text_vectorizer": final_artifacts["text_vectorizer"],
        "text_svd": final_artifacts["text_svd"],
        "tabular_feature_names": final_artifacts["tabular_feature_names"],
        "raw_text_feature_names": final_artifacts["raw_text_feature_names"],
        "combined_feature_names": final_artifacts["combined_feature_names"],
        "text_cleaning_rules": {
            "boilerplate_patterns": BOILERPLATE_PATTERNS,
            "min_token_count": TEXT_MIN_TOKEN_COUNT,
            "unicode_normalization": "NFKD + combining mark removal",
        },
        "baseline_metrics": baseline_metrics,
        "text_metrics": text_metrics,
    }

    LOGGER.info("Model bundle kaydediliyor: %s", TEXT_MODEL_OUTPUT_PATH)
    joblib.dump(bundle, TEXT_MODEL_OUTPUT_PATH)


def main() -> int:
    configure_logging()
    ensure_output_directories()

    dataframe = load_dataset(DATASET_PATH)
    splits = split_dataset(dataframe)

    train_df = splits.train.copy()
    validation_df = splits.validation.copy()
    test_df = splits.test.copy()

    y_train = train_df[TARGET_COLUMN]
    y_validation = validation_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    candidates, skipped_models = build_candidate_specs()
    representation_cache = build_representation_cache(train_df, validation_df)
    results = train_candidate_models(
        candidates=candidates,
        representation_cache=representation_cache,
        y_train=y_train,
        y_validation=y_validation,
        skipped_models=skipped_models,
    )
    best_result = select_best_result(results)

    train_validation_df = pd.concat([train_df, validation_df], axis=0, ignore_index=True)
    y_train_validation = train_validation_df[TARGET_COLUMN]

    LOGGER.info("En iyi candidate train + validation ile yeniden egitiliyor...")
    final_artifacts = fit_feature_artifacts(train_validation_df, best_result["representation"])
    X_train_validation = transform_with_artifacts(final_artifacts, train_validation_df)
    X_test = transform_with_artifacts(final_artifacts, test_df)

    final_model = best_result["factory"]()
    final_model.fit(X_train_validation, y_train_validation)
    text_predictions = np.asarray(final_model.predict(X_test), dtype=float)
    text_metrics = calculate_metrics(y_test, text_predictions)
    LOGGER.info(
        "Final test | %s | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
        best_result["label"],
        text_metrics["mae"],
        text_metrics["rmse"],
        text_metrics["r2"],
        text_metrics["mape"],
    )

    baseline_predictions, baseline_metrics = build_baseline_predictions(test_df)
    prediction_frame = build_prediction_frame(
        test_df=test_df,
        baseline_predictions=baseline_predictions,
        text_predictions=text_predictions,
    )

    validation_table = build_validation_table(results)
    comparison_table = build_test_comparison_table(baseline_metrics, text_metrics)
    district_improvement_table = summarize_district_improvement(prediction_frame)
    luxury_comparison_table = build_segment_comparison_table(prediction_frame)
    feature_space_summary = build_feature_space_summary(final_artifacts, train_validation_df)

    importance_method, importance_values, signed_values = compute_feature_importance(
        model=final_model,
        feature_names=final_artifacts["combined_feature_names"],
        X_test=X_test,
        y_test=y_test,
    )
    overall_feature_importance = build_top_feature_table(
        feature_names=final_artifacts["combined_feature_names"],
        importances=importance_values,
        signed_values=signed_values,
        top_n=20,
    )
    text_feature_importance = build_top_feature_table(
        feature_names=final_artifacts["combined_feature_names"],
        importances=importance_values,
        signed_values=signed_values,
        top_n=20,
        prefix_filter="text_",
    )

    best_raw_linear_candidates = [
        result for result in results
        if result["representation"] == "tfidf" and result["supports_token_coefficients"]
    ]
    best_raw_linear_label: str | None = None
    positive_tokens = pd.DataFrame()
    negative_tokens = pd.DataFrame()
    absolute_tokens = pd.DataFrame()

    if best_raw_linear_candidates:
        best_raw_linear_result = min(
            best_raw_linear_candidates,
            key=lambda item: item["validation_metrics"]["mae"],
        )
        best_raw_linear_label = best_raw_linear_result["label"]

        if best_result["label"] == best_raw_linear_label:
            raw_linear_artifacts = final_artifacts
            raw_linear_model = final_model
        else:
            raw_linear_artifacts = fit_feature_artifacts(train_validation_df, "tfidf")
            raw_linear_model = best_raw_linear_result["factory"]()
            X_train_validation_raw = transform_with_artifacts(raw_linear_artifacts, train_validation_df)
            raw_linear_model.fit(X_train_validation_raw, y_train_validation)

        positive_tokens, negative_tokens, absolute_tokens = extract_linear_token_tables(
            model=raw_linear_model,
            artifacts=raw_linear_artifacts,
            top_n=15,
        )

    svd_component_tokens = build_svd_component_token_table(
        artifacts=final_artifacts,
        feature_importance_table=overall_feature_importance,
        top_n=10,
        tokens_per_component=8,
    )

    report_body = build_report(
        best_result=best_result,
        validation_table=validation_table,
        baseline_metrics=baseline_metrics,
        text_metrics=text_metrics,
        comparison_table=comparison_table,
        district_improvement_table=district_improvement_table,
        luxury_comparison_table=luxury_comparison_table,
        feature_space_summary=feature_space_summary,
        overall_feature_importance=overall_feature_importance,
        text_feature_importance=text_feature_importance,
        positive_tokens=positive_tokens,
        negative_tokens=negative_tokens,
        absolute_tokens=absolute_tokens,
        svd_component_tokens=svd_component_tokens,
        skipped_models=skipped_models,
        best_raw_linear_label=best_raw_linear_label,
    )

    save_model_bundle(
        best_result=best_result,
        final_artifacts=final_artifacts,
        final_model=final_model,
        baseline_metrics=baseline_metrics,
        text_metrics=text_metrics,
    )
    LOGGER.info("Rapor kaydediliyor: %s", TEXT_REPORT_OUTPUT_PATH)
    TEXT_REPORT_OUTPUT_PATH.write_text(report_body, encoding="utf-8")

    LOGGER.info("Feature importance yontemi: %s", importance_method)
    LOGGER.info("Tamamlandi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
