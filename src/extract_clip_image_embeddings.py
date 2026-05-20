from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm


LOGGER = logging.getLogger("clip_image_embedding_extraction")

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = ROOT_DIR / "dataset" / "train_ready_multimodal.parquet"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "dataset" / "clip_image_embeddings.parquet"
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "clip_image_embedding_report.md"

LISTING_ID_COLUMN = "listing_id"
IMAGE_PATHS_COLUMN = "valid_image_paths"
DEFAULT_BATCH_SIZE = 64
MAX_IMAGES_PER_LISTING = 16
OPENCLIP_MODEL_NAME = "ViT-B-16"
OPENCLIP_PRETRAINED_TAG = "laion2b_s34b_b88k"


@dataclass
class ExtractionStats:
    total_listings: int = 0
    extracted_listings: int = 0
    skipped_listings: int = 0
    broken_image_count: int = 0
    total_used_images: int = 0
    embedding_dimension: int = 0
    meanmax_dimension: int = 0
    device_name: str = "cpu"
    duration_seconds: float = 0.0


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OpenCLIP image embeddings per listing using up to 16 readable images."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input multimodal parquet path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output embeddings parquet path.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Output markdown report path.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for OpenCLIP inference.",
    )
    parser.add_argument(
        "--max-images-per-listing",
        type=int,
        default=MAX_IMAGES_PER_LISTING,
        help="Maximum readable images to use per listing.",
    )
    parser.add_argument(
        "--limit-listings",
        type=int,
        default=None,
        help="Optional cap for smoke tests; when omitted all listings are processed.",
    )
    return parser.parse_args()


def ensure_output_directories(output_path: Path, report_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)


def load_dataset(input_path: Path, limit_listings: int | None = None) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet bulunamadi: {input_path}")

    LOGGER.info("Multimodal dataset okunuyor: %s", input_path)
    dataframe = pd.read_parquet(input_path)

    required_columns = {LISTING_ID_COLUMN, IMAGE_PATHS_COLUMN}
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Dataset icinde eksik kolonlar bulundu: {missing_columns}")

    if limit_listings is not None:
        dataframe = dataframe.head(limit_listings).copy()

    LOGGER.info("Islenecek listing sayisi: %s", len(dataframe))
    return dataframe.reset_index(drop=True)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.cuda.get_device_name(0)
            return torch.device("cuda")
        except Exception as exc:
            LOGGER.warning("CUDA kullanilamadi, CPU'ya dusuluyor: %s", exc)
    return torch.device("cpu")


def parse_image_path_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]

    if isinstance(value, tuple):
        return [str(item) for item in value]

    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
        except Exception:
            continue

    return [text]


def resolve_image_path(raw_path: str, project_root: Path) -> Path | None:
    if not raw_path:
        return None

    path_obj = Path(raw_path)
    candidate_paths = []

    if path_obj.is_absolute():
        candidate_paths.append(path_obj)
    else:
        candidate_paths.append(path_obj)
        candidate_paths.append(project_root / path_obj)

        raw_text = raw_path.replace("\\", "/").strip("/")
        candidate_paths.append(project_root / raw_text)
        if raw_text.startswith("dataset/"):
            candidate_paths.append(project_root / raw_text)
        else:
            candidate_paths.append(project_root / "dataset" / raw_text)

    seen: set[str] = set()
    for candidate in candidate_paths:
        candidate_text = str(candidate)
        if candidate_text in seen:
            continue
        seen.add(candidate_text)
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def load_image_tensor(image_path: Path, image_preprocess: Any) -> torch.Tensor:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return image_preprocess(rgb_image)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def flush_batch(
    model: Any,
    device: torch.device,
    batch_tensors: list[torch.Tensor],
    batch_listing_ids: list[str],
    embedding_sums: dict[str, np.ndarray],
    embedding_maxes: dict[str, np.ndarray],
    embedding_counts: dict[str, int],
) -> int:
    if not batch_tensors:
        return 0

    batch_tensor = torch.stack(batch_tensors, dim=0).to(device, non_blocking=device.type == "cuda")

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model.encode_image(batch_tensor)
        else:
            outputs = model.encode_image(batch_tensor)

    output_array = outputs.detach().float().cpu().numpy()
    output_array = l2_normalize(output_array.astype(np.float32, copy=False))

    for listing_id, embedding in zip(batch_listing_ids, output_array, strict=True):
        if listing_id not in embedding_sums:
            embedding_sums[listing_id] = embedding.astype(np.float64, copy=True)
            embedding_maxes[listing_id] = embedding.astype(np.float32, copy=True)
            embedding_counts[listing_id] = 1
        else:
            embedding_sums[listing_id] += embedding
            embedding_maxes[listing_id] = np.maximum(embedding_maxes[listing_id], embedding)
            embedding_counts[listing_id] += 1

    processed_image_count = len(batch_tensors)
    batch_tensors.clear()
    batch_listing_ids.clear()
    return processed_image_count


def extract_embeddings(
    dataframe: pd.DataFrame,
    model: Any,
    device: torch.device,
    image_preprocess: Any,
    batch_size: int,
    max_images_per_listing: int,
) -> tuple[pd.DataFrame, ExtractionStats]:
    stats = ExtractionStats(
        total_listings=int(len(dataframe)),
        device_name=(
            f"cuda ({torch.cuda.get_device_name(device)})"
            if device.type == "cuda"
            else device.type
        ),
    )

    embedding_sums: dict[str, np.ndarray] = {}
    embedding_maxes: dict[str, np.ndarray] = {}
    embedding_counts: dict[str, int] = {}
    used_image_counts: dict[str, int] = {}
    batch_tensors: list[torch.Tensor] = []
    batch_listing_ids: list[str] = []
    running_total_used_images = 0

    progress_bar = tqdm(
        dataframe.itertuples(index=False),
        total=len(dataframe),
        desc="Processing listings",
        unit="listing",
        mininterval=5.0,
    )

    for row in progress_bar:
        listing_id = str(getattr(row, LISTING_ID_COLUMN))
        raw_paths = parse_image_path_list(getattr(row, IMAGE_PATHS_COLUMN))

        used_for_listing = 0
        for raw_path in raw_paths:
            if used_for_listing >= max_images_per_listing:
                break

            resolved_path = resolve_image_path(raw_path, ROOT_DIR)
            if resolved_path is None:
                stats.broken_image_count += 1
                continue

            try:
                image_tensor = load_image_tensor(resolved_path, image_preprocess)
            except Exception:
                stats.broken_image_count += 1
                continue

            batch_tensors.append(image_tensor)
            batch_listing_ids.append(listing_id)
            used_for_listing += 1

            if len(batch_tensors) >= batch_size:
                flush_batch(
                    model=model,
                    device=device,
                    batch_tensors=batch_tensors,
                    batch_listing_ids=batch_listing_ids,
                    embedding_sums=embedding_sums,
                    embedding_maxes=embedding_maxes,
                    embedding_counts=embedding_counts,
                )

        if used_for_listing > 0:
            used_image_counts[listing_id] = used_for_listing
            running_total_used_images += used_for_listing

        progress_bar.set_postfix(
            extracted=len(used_image_counts),
            skipped=(stats.total_listings - len(used_image_counts)),
            broken=stats.broken_image_count,
            used_images=running_total_used_images,
        )

    flush_batch(
        model=model,
        device=device,
        batch_tensors=batch_tensors,
        batch_listing_ids=batch_listing_ids,
        embedding_sums=embedding_sums,
        embedding_maxes=embedding_maxes,
        embedding_counts=embedding_counts,
    )
    progress_bar.close()

    rows: list[dict[str, Any]] = []
    for listing_id in dataframe[LISTING_ID_COLUMN].astype(str).tolist():
        if listing_id not in embedding_sums:
            continue

        mean_embedding = embedding_sums[listing_id] / max(embedding_counts[listing_id], 1)
        max_embedding = embedding_maxes[listing_id]
        mean_embedding = l2_normalize(mean_embedding.astype(np.float32, copy=False)[None, :])[0]
        max_embedding = l2_normalize(max_embedding.astype(np.float32, copy=False)[None, :])[0]
        meanmax_embedding = np.concatenate([mean_embedding, max_embedding], axis=0).astype(np.float32, copy=False)
        meanmax_embedding = l2_normalize(meanmax_embedding[None, :])[0]

        rows.append(
            {
                LISTING_ID_COLUMN: listing_id,
                "clip_mean_embedding": mean_embedding.tolist(),
                "clip_max_embedding": max_embedding.tolist(),
                "clip_meanmax_embedding": meanmax_embedding.tolist(),
                "used_image_count": int(used_image_counts[listing_id]),
            }
        )

    embedding_df = pd.DataFrame(rows)

    stats.extracted_listings = int(len(embedding_df))
    stats.total_used_images = int(sum(used_image_counts.values()))
    if not embedding_df.empty:
        stats.embedding_dimension = int(len(embedding_df.iloc[0]["clip_mean_embedding"]))
        stats.meanmax_dimension = int(len(embedding_df.iloc[0]["clip_meanmax_embedding"]))
    stats.skipped_listings = int(stats.total_listings - stats.extracted_listings)

    return embedding_df, stats


def format_duration(duration_seconds: float) -> str:
    total_seconds = int(round(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_report(
    stats: ExtractionStats,
    input_path: Path,
    output_path: Path,
    batch_size: int,
    max_images_per_listing: int,
) -> str:
    average_used_images = (
        stats.total_used_images / stats.extracted_listings
        if stats.extracted_listings
        else 0.0
    )

    lines = [
        "# CLIP Image Embedding Report",
        "",
        "## Summary",
        "",
        f"- Input dataset: `{input_path}`",
        f"- Output parquet: `{output_path}`",
        f"- Encoder: `open_clip` {OPENCLIP_MODEL_NAME} / `{OPENCLIP_PRETRAINED_TAG}`",
        "- Image processing: model-native OpenCLIP preprocess",
        "- Listing representation: mean pooled, max pooled, and mean+max concatenated embeddings",
        f"- Device: **{stats.device_name}**",
        f"- Batch size: **{batch_size}**",
        f"- Max images per listing: **{max_images_per_listing}**",
        f"- Sure: **{format_duration(stats.duration_seconds)}** ({stats.duration_seconds:.2f} saniye)",
        "",
        "## Metrics",
        "",
        f"- Toplam ilan sayisi: **{stats.total_listings:,}**",
        f"- Embedding cikarilan ilan sayisi: **{stats.extracted_listings:,}**",
        f"- Skip edilen ilan sayisi: **{stats.skipped_listings:,}**",
        f"- Toplam kullanilan gorsel sayisi: **{stats.total_used_images:,}**",
        f"- Ortalama kullanilan gorsel sayisi: **{average_used_images:.2f}**",
        f"- Embedding dimension: **{stats.embedding_dimension}**",
        f"- Mean+max dimension: **{stats.meanmax_dimension}**",
        f"- Hata veren / okunamayan gorsel sayisi: **{stats.broken_image_count:,}**",
    ]

    return "\n".join(lines)


def save_outputs(
    embedding_df: pd.DataFrame,
    output_path: Path,
    report_body: str,
    report_path: Path,
) -> None:
    LOGGER.info("Embedding parquet kaydediliyor: %s", output_path)
    embedding_df.to_parquet(output_path, index=False)

    LOGGER.info("Markdown rapor kaydediliyor: %s", report_path)
    report_path.write_text(report_body, encoding="utf-8")


def build_model(device: torch.device) -> tuple[Any, Any]:
    LOGGER.info(
        "OpenCLIP model yukleniyor: %s / %s",
        OPENCLIP_MODEL_NAME,
        OPENCLIP_PRETRAINED_TAG,
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL_NAME,
        pretrained=OPENCLIP_PRETRAINED_TAG,
        device=device,
    )
    model.eval()
    return model, preprocess


def main() -> int:
    configure_logging()
    args = parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()

    ensure_output_directories(output_path=output_path, report_path=report_path)

    dataframe = load_dataset(input_path=input_path, limit_listings=args.limit_listings)
    device = select_device()
    LOGGER.info("Kullanilan cihaz: %s", device.type)

    model, image_preprocess = build_model(device)

    started_at = time.perf_counter()
    embedding_df, stats = extract_embeddings(
        dataframe=dataframe,
        model=model,
        device=device,
        image_preprocess=image_preprocess,
        batch_size=args.batch_size,
        max_images_per_listing=args.max_images_per_listing,
    )
    stats.duration_seconds = time.perf_counter() - started_at

    report_body = build_report(
        stats=stats,
        input_path=input_path,
        output_path=output_path,
        batch_size=args.batch_size,
        max_images_per_listing=args.max_images_per_listing,
    )
    save_outputs(
        embedding_df=embedding_df,
        output_path=output_path,
        report_body=report_body,
        report_path=report_path,
    )

    LOGGER.info("Tamamlandi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
