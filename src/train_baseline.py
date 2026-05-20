from __future__ import annotations

import logging
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ModuleNotFoundError as exc:  # pragma: no cover - handled for user experience
    raise SystemExit(
        "Eksik bagimlilik bulundu. Lutfen once `python -m pip install -r requirements.txt` calistirin."
    ) from exc


LOGGER = logging.getLogger("baseline_training")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "train_ready_ml.parquet"
MODEL_OUTPUT_PATH = ROOT_DIR / "models" / "baseline_model.joblib"
PREPROCESSOR_OUTPUT_PATH = ROOT_DIR / "models" / "baseline_preprocessor.joblib"
REPORT_OUTPUT_PATH = ROOT_DIR / "reports" / "baseline_results.md"

TARGET_COLUMN = "price_try"
RANDOM_STATE = 42
PRIMARY_SELECTION_METRIC = "mae"
ROOMS_OUTLIER_THRESHOLD = 12

FEATURE_COLUMNS = [
    "city",
    "district",
    "neighborhood",
    "rooms",
    "bathrooms",
    "m2_gross",
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

NUMERIC_COLUMNS = [
    "rooms",
    "bathrooms",
    "m2_gross",
    "building_age",
    "floor",
    "total_floors",
    "dues_try",
    "image_count",
]

CATEGORICAL_COLUMNS = [
    "city",
    "district",
    "neighborhood",
    "heating_type",
    "fuel_type",
    "is_furnished",
    "home_type",
    "home_shape",
]


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class ModelResult:
    name: str
    validation_metrics: dict[str, float]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility path
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def normalize_string(value: Any) -> Any:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    text = re.sub(r"\s+", " ", text)
    return text


def parse_room_count(value: Any) -> float:
    normalized = normalize_string(value)
    if pd.isna(normalized):
        return np.nan

    match = re.fullmatch(r"(\d+)\s*\+\s*(\d+)", str(normalized))
    if match:
        total_rooms = int(match.group(1)) + int(match.group(2))
        if total_rooms > ROOMS_OUTLIER_THRESHOLD:
            return np.nan
        return float(total_rooms)

    fallback_match = re.search(r"\d+", str(normalized))
    if fallback_match:
        parsed = float(fallback_match.group(0))
        if parsed > ROOMS_OUTLIER_THRESHOLD:
            return np.nan
        return parsed

    return np.nan


def parse_floor_value(value: Any) -> float:
    normalized = normalize_string(value)
    if pd.isna(normalized):
        return np.nan

    text = str(normalized).casefold()
    zero_floor_tokens = {
        "giriş katı",
        "zemin kat",
        "zemin",
        "bahçe katı",
        "yüksek giriş",
    }
    unknown_tokens = {
        "ara kat",
        "en üst kat",
        "yarı bodrum",
        "bodrum",
        "bodrum ve zemin",
        "çatı katı",
        "teras katı",
        "villa katı",
    }

    if text in zero_floor_tokens:
        return 0.0
    if text in unknown_tokens:
        return np.nan

    numeric_match = re.fullmatch(r"(\d+)\.\s*kat", text)
    if numeric_match:
        return float(numeric_match.group(1))

    if text == "21 ve üzeri":
        return 21.0

    return np.nan


def normalize_furnished_flag(value: Any) -> Any:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return "Evet" if value else "Hayir"

    text = str(value).strip().casefold()
    if text in {"true", "evet", "yes", "1"}:
        return "Evet"
    if text in {"false", "hayır", "hayir", "no", "0"}:
        return "Hayir"
    return normalize_string(value)


class PropertyFeatureCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, feature_columns: list[str]):
        self.feature_columns = feature_columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PropertyFeatureCleaner":
        self._validate_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X)
        frame = X.loc[:, self.feature_columns].copy()

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

    def _validate_columns(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PropertyFeatureCleaner bir pandas DataFrame bekliyor.")
        missing = sorted(set(self.feature_columns) - set(X.columns))
        if missing:
            raise ValueError(f"Beklenen feature kolonlari eksik: {missing}")


def ensure_output_directories() -> None:
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    LOGGER.info("Dataset okunuyor: %s", dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset bulunamadi: {dataset_path}")

    dataframe = pd.read_parquet(dataset_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Dataset icinde eksik kolonlar bulundu: {missing_columns}")

    dataframe = dataframe.dropna(subset=[TARGET_COLUMN]).copy()
    LOGGER.info("Dataset boyutu: %s satir x %s kolon", dataframe.shape[0], dataframe.shape[1])
    return dataframe


def split_dataset(dataframe: pd.DataFrame) -> DatasetSplits:
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    LOGGER.info(
        "Split tamamlandi | train=%s, validation=%s, test=%s",
        len(train_df),
        len(validation_df),
        len(test_df),
    )
    return DatasetSplits(
        train=train_df.reset_index(drop=True),
        validation=validation_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def build_preprocessor() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("feature_cleaner", PropertyFeatureCleaner(FEATURE_COLUMNS)),
            ("column_transformer", column_transformer),
        ]
    )


def build_model_factories() -> tuple[dict[str, Callable[[], Any]], dict[str, str]]:
    factories: dict[str, Callable[[], Any]] = {
        "RandomForestRegressor": partial(
            RandomForestRegressor,
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoostingRegressor": partial(
            HistGradientBoostingRegressor,
            learning_rate=0.05,
            max_iter=400,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
    }
    skipped_models: dict[str, str] = {}

    try:
        from lightgbm import LGBMRegressor

        factories["LightGBMRegressor"] = partial(
            LGBMRegressor,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ModuleNotFoundError:
        skipped_models["LightGBMRegressor"] = "lightgbm kurulu degil"

    try:
        from xgboost import XGBRegressor

        factories["XGBoostRegressor"] = partial(
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
        )
    except ModuleNotFoundError:
        skipped_models["XGBoostRegressor"] = "xgboost kurulu degil"

    return factories, skipped_models


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true_array = y_true.to_numpy(dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true_array, y_pred_array))
    rmse = float(math.sqrt(mean_squared_error(y_true_array, y_pred_array)))
    r2 = float(r2_score(y_true_array, y_pred_array))

    denominator = np.where(np.abs(y_true_array) < 1e-8, np.nan, np.abs(y_true_array))
    mape = float(np.nanmean(np.abs((y_true_array - y_pred_array) / denominator)) * 100.0)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def train_candidate_models(
    model_factories: dict[str, Callable[[], Any]],
    X_train_processed: np.ndarray,
    y_train: pd.Series,
    X_validation_processed: np.ndarray,
    y_validation: pd.Series,
    skipped_models: dict[str, str],
) -> list[ModelResult]:
    results: list[ModelResult] = []

    for model_name, factory in model_factories.items():
        LOGGER.info("%s egitiliyor...", model_name)
        try:
            model = factory()
            model.fit(X_train_processed, y_train)
            validation_predictions = model.predict(X_validation_processed)
            validation_metrics = calculate_metrics(y_validation, validation_predictions)
            LOGGER.info(
                "%s validation | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
                model_name,
                validation_metrics["mae"],
                validation_metrics["rmse"],
                validation_metrics["r2"],
                validation_metrics["mape"],
            )
            results.append(ModelResult(name=model_name, validation_metrics=validation_metrics))
        except Exception as exc:  # pragma: no cover - defensive against optional libraries
            skipped_models[model_name] = f"egitim hatasi: {exc.__class__.__name__}: {exc}"
            LOGGER.exception("%s egitimi atlandi.", model_name)

    if not results:
        raise RuntimeError("Hicbir model basariyla egitilemedi.")

    return results


def select_best_model(results: list[ModelResult]) -> ModelResult:
    best_result = min(results, key=lambda item: item.validation_metrics[PRIMARY_SELECTION_METRIC])
    LOGGER.info(
        "En iyi model secildi: %s (validation MAE=%.2f)",
        best_result.name,
        best_result.validation_metrics["mae"],
    )
    return best_result


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    transformer = preprocessor.named_steps["column_transformer"]
    feature_names = transformer.get_feature_names_out()
    return [str(name) for name in feature_names]


def compute_feature_importance(
    model: Any,
    feature_names: list[str],
    X_test_processed: np.ndarray,
    y_test: pd.Series,
) -> tuple[str | None, list[dict[str, float]]]:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        method = "native_feature_importances"
    else:
        sample_size = min(len(y_test), 1000)
        permutation = permutation_importance(
            model,
            X_test_processed[:sample_size],
            y_test.iloc[:sample_size],
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        importances = np.asarray(permutation.importances_mean, dtype=float)
        method = "permutation_importance"

    importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    importance_series = importance_series.head(20)

    records = [
        {"feature": str(feature_name), "importance": float(importance_value)}
        for feature_name, importance_value in importance_series.items()
    ]
    return method, records


def build_error_analysis(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    test_predictions: np.ndarray,
) -> dict[str, Any]:
    error_frame = test_df.loc[
        :,
        ["listing_id", "district", "neighborhood", "rooms", "m2_gross", "floor", "image_count"],
    ].copy()
    error_frame["actual_price_try"] = y_test.to_numpy(dtype=float)
    error_frame["predicted_price_try"] = np.asarray(test_predictions, dtype=float)
    error_frame["signed_error"] = error_frame["predicted_price_try"] - error_frame["actual_price_try"]
    error_frame["abs_error"] = error_frame["signed_error"].abs()
    error_frame["ape_pct"] = (
        error_frame["abs_error"] / error_frame["actual_price_try"].clip(lower=1e-8)
    ) * 100.0

    error_quantiles = error_frame["abs_error"].quantile([0.50, 0.75, 0.90, 0.95]).to_dict()
    worst_predictions = error_frame.sort_values("abs_error", ascending=False).head(10)

    district_summary = (
        error_frame.groupby("district", dropna=False)
        .agg(
            sample_count=("district", "size"),
            mae=("abs_error", "mean"),
            mean_signed_error=("signed_error", "mean"),
            mape=("ape_pct", "mean"),
        )
        .reset_index()
    )
    district_summary = district_summary[district_summary["sample_count"] >= 5]
    district_summary = district_summary.sort_values("mae", ascending=False).head(10)

    return {
        "summary": {
            "mean_actual": float(error_frame["actual_price_try"].mean()),
            "mean_prediction": float(error_frame["predicted_price_try"].mean()),
            "mean_signed_error": float(error_frame["signed_error"].mean()),
            "median_abs_error": float(error_quantiles.get(0.50, np.nan)),
            "p90_abs_error": float(error_quantiles.get(0.90, np.nan)),
            "p95_abs_error": float(error_quantiles.get(0.95, np.nan)),
        },
        "worst_predictions": worst_predictions,
        "district_summary": district_summary,
    }


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


def format_metrics_table(results: list[ModelResult]) -> str:
    rows: list[list[str]] = []
    for result in sorted(results, key=lambda item: item.validation_metrics[PRIMARY_SELECTION_METRIC]):
        rows.append(
            [
                result.name,
                format_float(result.validation_metrics["mae"]),
                format_float(result.validation_metrics["rmse"]),
                format_float(result.validation_metrics["r2"], digits=4),
                format_float(result.validation_metrics["mape"]),
            ]
        )
    return render_markdown_table(
        headers=["Model", "Validation MAE", "Validation RMSE", "Validation R2", "Validation MAPE (%)"],
        rows=rows,
    )


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


def build_report(
    dataset_path: Path,
    splits: DatasetSplits,
    validation_results: list[ModelResult],
    best_validation_result: ModelResult,
    final_model_name: str,
    test_metrics: dict[str, float],
    transformed_feature_count: int,
    skipped_models: dict[str, str],
    feature_importance_method: str | None,
    feature_importance_rows: list[dict[str, float]],
    error_analysis: dict[str, Any],
) -> str:
    original_feature_list = ", ".join(f"`{column}`" for column in FEATURE_COLUMNS)
    numeric_feature_list = ", ".join(f"`{column}`" for column in NUMERIC_COLUMNS)
    categorical_feature_list = ", ".join(f"`{column}`" for column in CATEGORICAL_COLUMNS)

    feature_importance_table = render_markdown_table(
        ["Feature", "Importance"],
        [
            [row["feature"], format_float(row["importance"], digits=6)]
            for row in feature_importance_rows
        ],
    )

    worst_prediction_columns = [
        "listing_id",
        "district",
        "neighborhood",
        "rooms",
        "m2_gross",
        "floor",
        "image_count",
        "actual_price_try",
        "predicted_price_try",
        "abs_error",
        "ape_pct",
    ]
    worst_prediction_table = dataframe_to_markdown_table(
        error_analysis["worst_predictions"][worst_prediction_columns],
        digits=2,
    )
    district_table = dataframe_to_markdown_table(error_analysis["district_summary"], digits=2)

    skipped_section = "_Atlanan model yok_"
    if skipped_models:
        skipped_section = "\n".join(
            f"- `{model_name}`: {reason}"
            for model_name, reason in sorted(skipped_models.items())
        )

    report_lines = [
        "# Baseline Model Sonuclari",
        "",
        "## Ozet",
        "",
        f"- Calisma zamani: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Dataset: `{dataset_path}`",
        f"- Hedef degisken: `{TARGET_COLUMN}`",
        f"- Model secim metriği: validation `{PRIMARY_SELECTION_METRIC.upper()}`",
        f"- Kaydedilen model: `{MODEL_OUTPUT_PATH}`",
        f"- Kaydedilen preprocessing pipeline: `{PREPROCESSOR_OUTPUT_PATH}`",
        f"- En iyi model: **{best_validation_result.name}**",
        f"- Test icin kaydedilen final model: **{final_model_name}**",
        "",
        "## Veri ve Split Bilgileri",
        "",
        f"- Toplam ornek sayisi: **{len(splits.train) + len(splits.validation) + len(splits.test):,}**",
        f"- Train sayisi: **{len(splits.train):,}**",
        f"- Validation sayisi: **{len(splits.validation):,}**",
        f"- Test sayisi: **{len(splits.test):,}**",
        f"- Ham feature sayisi: **{len(FEATURE_COLUMNS)}**",
        f"- Donusmus feature sayisi: **{transformed_feature_count:,}**",
        "",
        "## Kullanilan Feature'lar",
        "",
        f"- Tum feature'lar: {original_feature_list}",
        f"- Sayisallastirilan kolonlar: {numeric_feature_list}",
        f"- One-hot encode edilen kategorik kolonlar: {categorical_feature_list}",
        "- `rooms` kolonu `3+1 -> 4` mantigiyla sayisallastirildi.",
        "- `rooms` icin toplam oda sayisi 12'yi asan acikca anomalik degerler `NaN` kabul edildi.",
        "- `floor` kolonu icin `Giris Kati`, `Zemin`, `Bahce Kati`, `Yuksek Giris` degerleri `0` kabul edildi.",
        "- Belirsiz veya baglama bagli kat degerleri (`Ara Kat`, `En Ust Kat`, `Kot 1`, `Bodrum`, `Cati Kati` vb.) `NaN` birakildi.",
        "- Numerik eksikler median ile, kategorik eksikler most-frequent ile dolduruldu.",
        "",
        "## Validation Sonuclari",
        "",
        format_metrics_table(validation_results),
        "",
        "## Test Sonuclari",
        "",
        render_markdown_table(
            ["Metric", "Value"],
            [
                ["MAE", format_float(test_metrics["mae"])],
                ["RMSE", format_float(test_metrics["rmse"])],
                ["R2", format_float(test_metrics["r2"], digits=4)],
                ["MAPE (%)", format_float(test_metrics["mape"])],
            ],
        ),
        "",
        "## Feature Importance",
        "",
        (
            f"- Yontem: `{feature_importance_method}`"
            if feature_importance_method
            else "- Feature importance hesaplanamadi."
        ),
        "",
        feature_importance_table if feature_importance_rows else "_Feature importance verisi yok_",
        "",
        "## Hata Analizi",
        "",
        render_markdown_table(
            ["Ozet", "Deger"],
            [
                ["Ortalama gercek fiyat", format_float(error_analysis["summary"]["mean_actual"])],
                ["Ortalama tahmin", format_float(error_analysis["summary"]["mean_prediction"])],
                ["Ortalama signed error", format_float(error_analysis["summary"]["mean_signed_error"])],
                ["Median absolute error", format_float(error_analysis["summary"]["median_abs_error"])],
                ["P90 absolute error", format_float(error_analysis["summary"]["p90_abs_error"])],
                ["P95 absolute error", format_float(error_analysis["summary"]["p95_abs_error"])],
            ],
        ),
        "",
        "### En Yuksek Hata Ureten Test Ornekleri",
        "",
        worst_prediction_table,
        "",
        "### Ilce Bazli Test Hata Ozeti",
        "",
        district_table,
        "",
        "## Atlanan Modeller",
        "",
        skipped_section,
        "",
        "## Notlar",
        "",
        "- Validation sonuclari model secimi icin train split uzerinde egitilen modellerden hesaplandi.",
        "- Kaydedilen final model, en iyi algoritma secildikten sonra `train + validation` verisi ile yeniden egitildi.",
        "- Test sonuclari sadece final model icin, hic gorulmemis test split uzerinde raporlandi.",
    ]

    return "\n".join(report_lines)


def save_artifacts(model: Any, preprocessor: Pipeline, report_body: str) -> None:
    LOGGER.info("Model kaydediliyor: %s", MODEL_OUTPUT_PATH)
    joblib.dump(model, MODEL_OUTPUT_PATH)

    LOGGER.info("Preprocessor kaydediliyor: %s", PREPROCESSOR_OUTPUT_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_OUTPUT_PATH)

    LOGGER.info("Rapor kaydediliyor: %s", REPORT_OUTPUT_PATH)
    REPORT_OUTPUT_PATH.write_text(report_body, encoding="utf-8")


def main() -> int:
    configure_logging()
    ensure_output_directories()

    dataframe = load_dataset(DATASET_PATH)
    splits = split_dataset(dataframe)

    X_train = splits.train[FEATURE_COLUMNS]
    y_train = splits.train[TARGET_COLUMN]
    X_validation = splits.validation[FEATURE_COLUMNS]
    y_validation = splits.validation[TARGET_COLUMN]
    X_test = splits.test[FEATURE_COLUMNS]
    y_test = splits.test[TARGET_COLUMN]

    LOGGER.info("Train split icin preprocessing fit ediliyor...")
    shared_preprocessor = build_preprocessor()
    X_train_processed = shared_preprocessor.fit_transform(X_train)
    X_validation_processed = shared_preprocessor.transform(X_validation)
    LOGGER.info("Donusmus feature sayisi: %s", X_train_processed.shape[1])

    model_factories, skipped_models = build_model_factories()
    validation_results = train_candidate_models(
        model_factories=model_factories,
        X_train_processed=X_train_processed,
        y_train=y_train,
        X_validation_processed=X_validation_processed,
        y_validation=y_validation,
        skipped_models=skipped_models,
    )
    best_validation_result = select_best_model(validation_results)

    LOGGER.info("%s modeli train + validation ile yeniden egitiliyor...", best_validation_result.name)
    final_preprocessor = build_preprocessor()
    train_validation_df = pd.concat([splits.train, splits.validation], axis=0, ignore_index=True)
    X_train_validation = train_validation_df[FEATURE_COLUMNS]
    y_train_validation = train_validation_df[TARGET_COLUMN]

    X_train_validation_processed = final_preprocessor.fit_transform(X_train_validation)
    X_test_processed = final_preprocessor.transform(X_test)

    final_model = model_factories[best_validation_result.name]()
    final_model.fit(X_train_validation_processed, y_train_validation)
    test_predictions = final_model.predict(X_test_processed)
    test_metrics = calculate_metrics(y_test, test_predictions)
    LOGGER.info(
        "Final test | %s | MAE=%.2f | RMSE=%.2f | R2=%.4f | MAPE=%.2f%%",
        best_validation_result.name,
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["r2"],
        test_metrics["mape"],
    )

    feature_names = get_feature_names(final_preprocessor)
    feature_importance_method, feature_importance_rows = compute_feature_importance(
        model=final_model,
        feature_names=feature_names,
        X_test_processed=X_test_processed,
        y_test=y_test,
    )
    error_analysis = build_error_analysis(
        test_df=splits.test,
        y_test=y_test,
        test_predictions=test_predictions,
    )

    report_body = build_report(
        dataset_path=DATASET_PATH,
        splits=splits,
        validation_results=validation_results,
        best_validation_result=best_validation_result,
        final_model_name=best_validation_result.name,
        test_metrics=test_metrics,
        transformed_feature_count=len(feature_names),
        skipped_models=skipped_models,
        feature_importance_method=feature_importance_method,
        feature_importance_rows=feature_importance_rows,
        error_analysis=error_analysis,
    )
    save_artifacts(final_model, final_preprocessor, report_body)

    LOGGER.info("Tamamlandi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
