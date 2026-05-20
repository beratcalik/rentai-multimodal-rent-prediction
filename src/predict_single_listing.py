from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
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

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import pandas as pd
import torch

from extract_clip_image_embeddings import (
    MAX_IMAGES_PER_LISTING,
    build_model as build_clip_model,
    l2_normalize,
    load_image_tensor,
    parse_image_path_list,
    resolve_image_path,
    select_device,
)
from train_final_multimodal_text_clip import (
    DESCRIPTION_COLUMN,
    TITLE_COLUMN,
    build_clean_text_columns,
)
from train_image_fusion_reduced import clean_tabular_features


LOGGER = logging.getLogger("single_listing_prediction")

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "final_multimodal_text_clip_model.joblib"
DEFAULT_INPUT_PATH = ROOT_DIR / "examples" / "sample_listing_input.json"

IMAGE_PATHS_COLUMN = "image_paths"
DEFAULT_BATCH_SIZE = 16
DEFAULT_PREDICTION_MESSAGE = (
    "Tahmin ilan bilgileri, aciklama metni ve fotograflar birlikte analiz edilerek uretildi."
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict rent for a single listing using the final multimodal tabular+text+CLIP model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the input listing JSON file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model bundle.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used while encoding listing images.",
    )
    return parser.parse_args()


def resolve_model_path(model_path: str | Path | None) -> Path:
    if model_path is None:
        return DEFAULT_MODEL_PATH.resolve()
    return Path(model_path).resolve()


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle bulunamadi: {model_path}")

    LOGGER.info("Model bundle yukleniyor: %s", model_path)
    bundle = joblib.load(model_path)
    required_keys = {
        "model_name",
        "tabular_preprocessor",
        "text_vectorizer",
        "text_svd",
        "image_processor",
        "regressor",
        "tabular_feature_columns",
        "embedding_dimension",
        "image_cap",
        "image_representation",
    }
    missing_keys = sorted(required_keys - set(bundle.keys()))
    if missing_keys:
        raise ValueError(f"Model bundle icinde eksik anahtarlar bulundu: {missing_keys}")
    return bundle


@lru_cache(maxsize=4)
def _load_model_bundle_cached(model_path_text: str) -> dict[str, Any]:
    return load_model_bundle(Path(model_path_text))


def get_model_bundle(model_path: str | Path | None = None) -> dict[str, Any]:
    resolved_model_path = resolve_model_path(model_path)
    return _load_model_bundle_cached(str(resolved_model_path))


@lru_cache(maxsize=1)
def get_clip_runtime() -> tuple[torch.device, Any, Any]:
    device = select_device()
    LOGGER.info("Kullanilan cihaz: %s", device.type)
    clip_model, image_preprocess = build_clip_model(device)
    return device, clip_model, image_preprocess


def load_input_payload(input_path: Path) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON bulunamadi: {input_path}")

    LOGGER.info("Input JSON okunuyor: %s", input_path)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Input JSON tek bir ilan nesnesi olmalidir.")

    return payload


def normalize_payload(
    payload: dict[str, Any],
    expected_tabular_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    inference_warnings: list[str] = []
    extra_fields = sorted(
        set(payload.keys()) - set(expected_tabular_columns) - {TITLE_COLUMN, DESCRIPTION_COLUMN, IMAGE_PATHS_COLUMN}
    )
    if extra_fields:
        inference_warnings.append(f"Ignore edilen fazladan alanlar: {', '.join(extra_fields)}")

    record: dict[str, Any] = {}
    for column in expected_tabular_columns:
        if column in payload:
            record[column] = payload[column]
        else:
            record[column] = np.nan
            inference_warnings.append(f"Eksik tabular alan imputer'a birakildi: {column}")

    for text_column in [TITLE_COLUMN, DESCRIPTION_COLUMN]:
        value = payload.get(text_column, "")
        if value in (None, ""):
            inference_warnings.append(f"Bos text alani fallback olarak kullanilacak: {text_column}")
            value = ""
        record[text_column] = value

    raw_image_paths = payload.get(IMAGE_PATHS_COLUMN, [])
    image_paths = parse_image_path_list(raw_image_paths)
    if IMAGE_PATHS_COLUMN not in payload:
        inference_warnings.append("image_paths alani gelmedi; image branch zero-vector fallback kullanacak.")
    elif not image_paths:
        inference_warnings.append("image_paths bos veya parse edilemedi; image branch zero-vector fallback kullanacak.")

    dataframe = pd.DataFrame([record])
    return dataframe, image_paths, inference_warnings


def build_image_embedding(
    image_paths: list[str],
    embedding_dimension: int,
    max_images: int,
    batch_size: int,
    device: torch.device,
    clip_model: Any,
    image_preprocess: Any,
) -> tuple[np.ndarray, int, list[str]]:
    inference_warnings: list[str] = []
    if not image_paths:
        return np.zeros((1, embedding_dimension), dtype=np.float32), 0, inference_warnings

    if len(image_paths) > max_images:
        inference_warnings.append(
            f"{len(image_paths)} gorsel verildi; en fazla {max_images} gorsel kullanilacagi icin yalnizca ilk {max_images} okunabilir gorsel dikkate alinacak."
        )

    usable_tensors: list[torch.Tensor] = []
    used_image_count = 0
    broken_image_count = 0

    for raw_path in image_paths:
        if used_image_count >= max_images:
            break

        resolved_path = resolve_image_path(raw_path, ROOT_DIR)
        if resolved_path is None:
            broken_image_count += 1
            continue

        try:
            usable_tensors.append(load_image_tensor(resolved_path, image_preprocess))
            used_image_count += 1
        except Exception:
            broken_image_count += 1

    if broken_image_count > 0:
        inference_warnings.append(f"{broken_image_count} gorsel okunamadi veya bulunamadi; kalanlar kullanildi.")

    if used_image_count == 0:
        inference_warnings.append("Hic okunabilir gorsel bulunamadi; image branch zero-vector fallback kullanacak.")
        return np.zeros((1, embedding_dimension), dtype=np.float32), 0, inference_warnings

    batches: list[np.ndarray] = []
    for start in range(0, len(usable_tensors), max(1, batch_size)):
        batch_tensor = torch.stack(usable_tensors[start : start + batch_size], dim=0).to(
            device,
            non_blocking=device.type == "cuda",
        )

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    batch_output = clip_model.encode_image(batch_tensor)
            else:
                batch_output = clip_model.encode_image(batch_tensor)

        batch_embeddings = batch_output.detach().float().cpu().numpy().astype(np.float32, copy=False)
        batches.append(l2_normalize(batch_embeddings))

    image_embeddings = np.vstack(batches).astype(np.float32, copy=False)
    mean_embedding = l2_normalize(image_embeddings.mean(axis=0, dtype=np.float32, keepdims=True))[0]
    max_embedding = l2_normalize(image_embeddings.max(axis=0, keepdims=True))[0]
    meanmax_embedding = np.concatenate([mean_embedding, max_embedding], axis=0).astype(np.float32, copy=False)
    meanmax_embedding = l2_normalize(meanmax_embedding[None, :])[0]

    if len(meanmax_embedding) != embedding_dimension:
        raise ValueError(
            f"Beklenen image embedding boyutu {embedding_dimension}, hesaplanan boyut {len(meanmax_embedding)}."
        )

    return meanmax_embedding[None, :], used_image_count, inference_warnings


def transform_features(
    bundle: dict[str, Any],
    dataframe: pd.DataFrame,
    image_embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cleaned_tabular = clean_tabular_features(dataframe)
    X_tabular = np.asarray(
        bundle["tabular_preprocessor"].transform(cleaned_tabular),
        dtype=np.float32,
    )

    cleaned_text = build_clean_text_columns(dataframe)["combined_text"]
    X_text_tfidf = bundle["text_vectorizer"].transform(cleaned_text)
    X_text = np.asarray(
        bundle["text_svd"].transform(X_text_tfidf),
        dtype=np.float32,
    )

    X_image = np.asarray(
        bundle["image_processor"].transform(image_embedding),
        dtype=np.float32,
    )

    X_fused = np.hstack([X_tabular, X_text, X_image]).astype(np.float32, copy=False)
    return X_tabular, X_text, X_image, X_fused


def format_amount(value: float) -> str:
    rounded = int(round(float(value)))
    return f"{rounded:,}".replace(",", ".")


def predict_from_dict(
    input_data: dict[str, Any],
    model_path: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    if not isinstance(input_data, dict):
        raise ValueError("predict_from_dict bir ilan sozlugu bekler.")

    bundle = get_model_bundle(model_path=model_path)
    dataframe, image_paths, inference_warnings = normalize_payload(
        payload=input_data,
        expected_tabular_columns=list(bundle["tabular_feature_columns"]),
    )

    cleaned_text = build_clean_text_columns(dataframe)["combined_text"].iloc[0]
    if not cleaned_text:
        inference_warnings.append("Title/description temizleme sonrasi bos kaldi; text branch sparse fallback kullanacak.")

    device, clip_model, image_preprocess = get_clip_runtime()
    image_embedding, used_image_count, image_warnings = build_image_embedding(
        image_paths=image_paths,
        embedding_dimension=int(bundle["embedding_dimension"]),
        max_images=min(int(bundle.get("image_cap", MAX_IMAGES_PER_LISTING)), MAX_IMAGES_PER_LISTING),
        batch_size=max(1, int(batch_size)),
        device=device,
        clip_model=clip_model,
        image_preprocess=image_preprocess,
    )
    inference_warnings.extend(image_warnings)

    _, _, _, X_fused = transform_features(
        bundle=bundle,
        dataframe=dataframe,
        image_embedding=image_embedding,
    )
    raw_prediction = float(np.asarray(bundle["regressor"].predict(X_fused), dtype=float).ravel()[0])
    rounded_prediction = int(round(raw_prediction))

    return {
        "predicted_rent_try": rounded_prediction,
        "predicted_rent_formatted": f"{format_amount(raw_prediction)} TL",
        "used_image_count": used_image_count,
        "model_name": str(bundle["model_name"]),
        "warnings": inference_warnings,
        "message": DEFAULT_PREDICTION_MESSAGE,
        "raw_prediction_try": raw_prediction,
    }


def print_result(prediction_result: dict[str, Any]) -> None:
    print(f"Model adi: {prediction_result['model_name']}")
    print(f"Tahmini kira: {format_amount(prediction_result['predicted_rent_try'])} TRY")
    print(f"Kullanilan gorsel sayisi: {prediction_result['used_image_count']}")
    warnings_list = list(prediction_result.get("warnings", []))
    if warnings_list:
        print("Uyarilar:")
        for warning_text in warnings_list:
            print(f"- {warning_text}")
    else:
        print("Uyarilar: yok")


def main() -> int:
    configure_logging()
    args = parse_args()

    payload = load_input_payload(args.input.resolve())
    prediction_result = predict_from_dict(
        input_data=payload,
        model_path=args.model.resolve(),
        batch_size=args.batch_size,
    )
    print_result(prediction_result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
