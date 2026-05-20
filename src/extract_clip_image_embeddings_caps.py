from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from extract_clip_image_embeddings import (
    DEFAULT_BATCH_SIZE,
    IMAGE_PATHS_COLUMN,
    LISTING_ID_COLUMN,
    OPENCLIP_MODEL_NAME,
    OPENCLIP_PRETRAINED_TAG,
    ROOT_DIR,
    build_model,
    configure_logging,
    ensure_output_directories,
    l2_normalize,
    load_dataset,
    load_image_tensor,
    parse_image_path_list,
    resolve_image_path,
    select_device,
)


LOGGER = logging.getLogger("clip_image_embedding_caps_extraction")

DEFAULT_INPUT_PATH = ROOT_DIR / "dataset" / "train_ready_multimodal.parquet"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "dataset" / "clip_image_embeddings_caps.parquet"
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "clip_image_embedding_caps_report.md"
DEFAULT_CAP_VALUES = [4, 8, 12, 16]


@dataclass
class ExtractionStats:
    total_listings: int = 0
    extracted_listings: int = 0
    skipped_listings: int = 0
    broken_image_count: int = 0
    total_processed_images: int = 0
    embedding_dimension: int = 0
    meanmax_dimension: int = 0
    device_name: str = "cpu"
    duration_seconds: float = 0.0
    per_cap_extracted_listings: dict[int, int] = field(default_factory=dict)
    per_cap_total_used_images: dict[int, int] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OpenCLIP image embeddings for multiple image caps in a single pass."
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
        "--caps",
        type=int,
        nargs="+",
        default=DEFAULT_CAP_VALUES,
        help="Image cap values to evaluate.",
    )
    parser.add_argument(
        "--limit-listings",
        type=int,
        default=None,
        help="Optional cap for smoke tests; when omitted all listings are processed.",
    )
    return parser.parse_args()


def normalize_caps(cap_values: list[int]) -> list[int]:
    normalized = sorted({int(value) for value in cap_values if int(value) > 0})
    if not normalized:
        raise ValueError("En az bir pozitif image cap degeri gerekli.")
    return normalized


def flush_batch(
    model: Any,
    device: torch.device,
    batch_tensors: list[torch.Tensor],
    batch_entries: list[tuple[str, int]],
    cap_values: list[int],
    embedding_sums: dict[tuple[str, int], np.ndarray],
    embedding_maxes: dict[tuple[str, int], np.ndarray],
    embedding_counts: dict[tuple[str, int], int],
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

    for (listing_id, image_index), embedding in zip(batch_entries, output_array, strict=True):
        for image_cap in cap_values:
            if image_index > image_cap:
                continue

            key = (listing_id, image_cap)
            if key not in embedding_sums:
                embedding_sums[key] = embedding.astype(np.float64, copy=True)
                embedding_maxes[key] = embedding.astype(np.float32, copy=True)
                embedding_counts[key] = 1
            else:
                embedding_sums[key] += embedding
                embedding_maxes[key] = np.maximum(embedding_maxes[key], embedding)
                embedding_counts[key] += 1

    processed_image_count = len(batch_tensors)
    batch_tensors.clear()
    batch_entries.clear()
    return processed_image_count


def extract_embeddings_for_caps(
    dataframe: pd.DataFrame,
    model: Any,
    device: torch.device,
    image_preprocess: Any,
    batch_size: int,
    cap_values: list[int],
) -> tuple[pd.DataFrame, ExtractionStats]:
    max_cap = max(cap_values)
    stats = ExtractionStats(
        total_listings=int(len(dataframe)),
        device_name=(
            f"cuda ({torch.cuda.get_device_name(device)})"
            if device.type == "cuda"
            else device.type
        ),
    )

    embedding_sums: dict[tuple[str, int], np.ndarray] = {}
    embedding_maxes: dict[tuple[str, int], np.ndarray] = {}
    embedding_counts: dict[tuple[str, int], int] = {}
    used_image_counts: dict[str, int] = {}
    batch_tensors: list[torch.Tensor] = []
    batch_entries: list[tuple[str, int]] = []
    running_total_processed_images = 0

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
            if used_for_listing >= max_cap:
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

            used_for_listing += 1
            batch_tensors.append(image_tensor)
            batch_entries.append((listing_id, used_for_listing))

            if len(batch_tensors) >= batch_size:
                flush_batch(
                    model=model,
                    device=device,
                    batch_tensors=batch_tensors,
                    batch_entries=batch_entries,
                    cap_values=cap_values,
                    embedding_sums=embedding_sums,
                    embedding_maxes=embedding_maxes,
                    embedding_counts=embedding_counts,
                )

        if used_for_listing > 0:
            used_image_counts[listing_id] = used_for_listing
            running_total_processed_images += used_for_listing

        progress_bar.set_postfix(
            extracted=len(used_image_counts),
            skipped=(stats.total_listings - len(used_image_counts)),
            broken=stats.broken_image_count,
            processed_images=running_total_processed_images,
        )

    flush_batch(
        model=model,
        device=device,
        batch_tensors=batch_tensors,
        batch_entries=batch_entries,
        cap_values=cap_values,
        embedding_sums=embedding_sums,
        embedding_maxes=embedding_maxes,
        embedding_counts=embedding_counts,
    )
    progress_bar.close()

    rows: list[dict[str, Any]] = []
    per_cap_extracted_listings = {image_cap: 0 for image_cap in cap_values}
    per_cap_total_used_images = {image_cap: 0 for image_cap in cap_values}

    for listing_id in dataframe[LISTING_ID_COLUMN].astype(str).tolist():
        listing_available_images = int(used_image_counts.get(listing_id, 0))
        if listing_available_images <= 0:
            continue

        for image_cap in cap_values:
            key = (listing_id, image_cap)
            if key not in embedding_sums:
                continue

            mean_embedding = embedding_sums[key] / max(embedding_counts[key], 1)
            max_embedding = embedding_maxes[key]
            mean_embedding = l2_normalize(mean_embedding.astype(np.float32, copy=False)[None, :])[0]
            max_embedding = l2_normalize(max_embedding.astype(np.float32, copy=False)[None, :])[0]
            meanmax_embedding = np.concatenate([mean_embedding, max_embedding], axis=0).astype(np.float32, copy=False)
            meanmax_embedding = l2_normalize(meanmax_embedding[None, :])[0]

            used_count = min(listing_available_images, image_cap)
            per_cap_extracted_listings[image_cap] += 1
            per_cap_total_used_images[image_cap] += used_count

            rows.append(
                {
                    LISTING_ID_COLUMN: listing_id,
                    "image_cap": int(image_cap),
                    "clip_mean_embedding": mean_embedding.tolist(),
                    "clip_max_embedding": max_embedding.tolist(),
                    "clip_meanmax_embedding": meanmax_embedding.tolist(),
                    "used_image_count": int(used_count),
                }
            )

    embedding_df = pd.DataFrame(rows)

    stats.extracted_listings = int(len(used_image_counts))
    stats.skipped_listings = int(stats.total_listings - stats.extracted_listings)
    stats.total_processed_images = int(sum(used_image_counts.values()))
    stats.per_cap_extracted_listings = per_cap_extracted_listings
    stats.per_cap_total_used_images = per_cap_total_used_images
    if not embedding_df.empty:
        stats.embedding_dimension = int(len(embedding_df.iloc[0]["clip_mean_embedding"]))
        stats.meanmax_dimension = int(len(embedding_df.iloc[0]["clip_meanmax_embedding"]))

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
    cap_values: list[int],
) -> str:
    cap_rows = []
    for image_cap in cap_values:
        extracted_count = stats.per_cap_extracted_listings.get(image_cap, 0)
        total_used_images = stats.per_cap_total_used_images.get(image_cap, 0)
        average_used_images = total_used_images / extracted_count if extracted_count else 0.0
        cap_rows.append(
            f"| {image_cap} | {extracted_count:,} | {total_used_images:,} | {average_used_images:.2f} |"
        )

    cap_table = "\n".join(
        [
            "| image_cap | extracted_listings | total_used_images | avg_used_images |",
            "| --- | --- | --- | --- |",
            *cap_rows,
        ]
    ) if cap_rows else "_Veri yok_"

    lines = [
        "# CLIP Image Embedding Caps Report",
        "",
        "## Summary",
        "",
        f"- Input dataset: `{input_path}`",
        f"- Output parquet: `{output_path}`",
        f"- Encoder: `open_clip` {OPENCLIP_MODEL_NAME} / `{OPENCLIP_PRETRAINED_TAG}`",
        "- Image processing: model-native OpenCLIP preprocess",
        "- Listing representation: mean pooled, max pooled, and mean+max concatenated embeddings",
        f"- Tested image caps: **{', '.join(str(value) for value in cap_values)}**",
        f"- Device: **{stats.device_name}**",
        f"- Batch size: **{batch_size}**",
        f"- Sure: **{format_duration(stats.duration_seconds)}** ({stats.duration_seconds:.2f} saniye)",
        "",
        "## Metrics",
        "",
        f"- Toplam ilan sayisi: **{stats.total_listings:,}**",
        f"- Embedding cikarilan ilan sayisi: **{stats.extracted_listings:,}**",
        f"- Skip edilen ilan sayisi: **{stats.skipped_listings:,}**",
        f"- Bir kez encode edilen toplam gorsel sayisi: **{stats.total_processed_images:,}**",
        f"- Embedding dimension: **{stats.embedding_dimension}**",
        f"- Mean+max dimension: **{stats.meanmax_dimension}**",
        f"- Hata veren / okunamayan gorsel sayisi: **{stats.broken_image_count:,}**",
        "",
        "## Cap Breakdown",
        "",
        cap_table,
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


def main() -> int:
    configure_logging()
    args = parse_args()

    cap_values = normalize_caps(args.caps)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()

    ensure_output_directories(output_path=output_path, report_path=report_path)

    dataframe = load_dataset(input_path=input_path, limit_listings=args.limit_listings)
    device = select_device()
    LOGGER.info("Kullanilan cihaz: %s", device.type)

    model, image_preprocess = build_model(device)

    started_at = time.perf_counter()
    embedding_df, stats = extract_embeddings_for_caps(
        dataframe=dataframe,
        model=model,
        device=device,
        image_preprocess=image_preprocess,
        batch_size=args.batch_size,
        cap_values=cap_values,
    )
    stats.duration_seconds = time.perf_counter() - started_at

    report_body = build_report(
        stats=stats,
        input_path=input_path,
        output_path=output_path,
        batch_size=args.batch_size,
        cap_values=cap_values,
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
