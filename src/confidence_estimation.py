from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


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

LOGGER = logging.getLogger("confidence_estimation")

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = ROOT_DIR / "dataset" / "train_ready_multimodal.parquet"

CITY_COLUMN = "city"
DISTRICT_COLUMN = "district"
NEIGHBORHOOD_COLUMN = "neighborhood"
ROOMS_COLUMN = "rooms"
M2_GROSS_COLUMN = "m2_gross"
TITLE_COLUMN = "title"
DESCRIPTION_COLUMN = "description"
TARGET_COLUMN = "price_try"

WEIGHT_COMPLETENESS = 0.25
WEIGHT_VISUAL = 0.20
WEIGHT_LOCATION = 0.20
WEIGHT_PRICE = 0.20
WEIGHT_STABILITY = 0.15


@dataclass(frozen=True)
class ConfidenceProfile:
    row_count: int
    district_counts: dict[str, int]
    neighborhood_counts: dict[str, int]
    district_reference_count: float
    neighborhood_reference_count: float
    price_p05: float
    price_p95: float


@dataclass(frozen=True)
class ConfidenceEstimate:
    score: int
    label: str
    reasons: list[str]
    components: dict[str, int]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def _normalize_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _normalize_numeric(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(str(value).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None


def _normalize_location_key(value: Any) -> str:
    return _normalize_text(value)


def _count_reference(value_counts: dict[str, int]) -> float:
    if not value_counts:
        return 1.0
    counts = pd.Series(list(value_counts.values()), dtype="float64")
    return float(max(counts.quantile(0.95), 1.0))


@lru_cache(maxsize=1)
def get_confidence_profile(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> ConfidenceProfile:
    resolved_path = Path(dataset_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Confidence dataset bulunamadı: {resolved_path}")

    LOGGER.info("Confidence profili yükleniyor: %s", resolved_path)
    dataframe = pd.read_parquet(
        resolved_path,
        columns=[DISTRICT_COLUMN, NEIGHBORHOOD_COLUMN, TARGET_COLUMN],
    )
    price_series = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce").dropna()
    district_counts = {
        str(key): int(value)
        for key, value in dataframe[DISTRICT_COLUMN].dropna().astype(str).str.strip().value_counts().items()
    }
    neighborhood_counts = {
        str(key): int(value)
        for key, value in dataframe[NEIGHBORHOOD_COLUMN].dropna().astype(str).str.strip().value_counts().items()
    }

    return ConfidenceProfile(
        row_count=len(dataframe),
        district_counts=district_counts,
        neighborhood_counts=neighborhood_counts,
        district_reference_count=_count_reference(district_counts),
        neighborhood_reference_count=_count_reference(neighborhood_counts),
        price_p05=float(price_series.quantile(0.05)),
        price_p95=float(price_series.quantile(0.95)),
    )


def _completeness_score(
    input_row: Mapping[str, Any],
    *,
    cleaned_text: str,
    used_image_count: int,
) -> int:
    checks = {
        CITY_COLUMN: 10,
        DISTRICT_COLUMN: 12,
        NEIGHBORHOOD_COLUMN: 12,
        ROOMS_COLUMN: 14,
        M2_GROSS_COLUMN: 16,
        TITLE_COLUMN: 12,
        DESCRIPTION_COLUMN: 14,
        "cleaned_text": 6,
        "image_presence": 4,
    }

    score = 0
    if not _is_missing(input_row.get(CITY_COLUMN)):
        score += checks[CITY_COLUMN]
    if not _is_missing(input_row.get(DISTRICT_COLUMN)):
        score += checks[DISTRICT_COLUMN]
    if not _is_missing(input_row.get(NEIGHBORHOOD_COLUMN)):
        score += checks[NEIGHBORHOOD_COLUMN]
    if not _is_missing(input_row.get(ROOMS_COLUMN)):
        score += checks[ROOMS_COLUMN]

    m2_value = _normalize_numeric(input_row.get(M2_GROSS_COLUMN))
    if m2_value is not None and m2_value > 0:
        score += checks[M2_GROSS_COLUMN]

    if len(_normalize_text(input_row.get(TITLE_COLUMN))) >= 5:
        score += checks[TITLE_COLUMN]
    if len(_normalize_text(input_row.get(DESCRIPTION_COLUMN))) >= 20:
        score += checks[DESCRIPTION_COLUMN]
    if cleaned_text:
        score += checks["cleaned_text"]
    if used_image_count > 0:
        score += checks["image_presence"]

    return int(np.clip(round(score), 0, 100))


def _visual_score(used_image_count: int) -> int:
    if used_image_count <= 0:
        return 20
    if used_image_count <= 3:
        return int(35 + used_image_count * 10)
    if used_image_count <= 8:
        return int(70 + (used_image_count - 4) * 4)
    if used_image_count <= 16:
        return int(min(98, 88 + (used_image_count - 9) * 1.5))
    return 98


def _location_density_score(input_row: Mapping[str, Any], profile: ConfidenceProfile) -> int:
    district = _normalize_location_key(input_row.get(DISTRICT_COLUMN))
    neighborhood = _normalize_location_key(input_row.get(NEIGHBORHOOD_COLUMN))
    if not district and not neighborhood:
        return 10

    district_count = profile.district_counts.get(district, 0)
    neighborhood_count = profile.neighborhood_counts.get(neighborhood, 0)

    district_ratio = min(district_count / max(profile.district_reference_count, 1.0), 1.0)
    neighborhood_ratio = min(neighborhood_count / max(profile.neighborhood_reference_count, 1.0), 1.0)

    district_score = 15 + 85 * np.sqrt(district_ratio) if district else 10
    neighborhood_score = 15 + 85 * np.sqrt(neighborhood_ratio) if neighborhood else 10

    if district and neighborhood:
        score = 0.4 * district_score + 0.6 * neighborhood_score
    else:
        score = max(district_score, neighborhood_score)
    return int(np.clip(round(score), 0, 100))


def _price_range_score(predicted_rent_try: float, profile: ConfidenceProfile) -> int:
    p05 = profile.price_p05
    p95 = profile.price_p95
    typical_span = max(p95 - p05, 1.0)

    if p05 <= predicted_rent_try <= p95:
        return 100

    if predicted_rent_try < p05:
        distance = p05 - predicted_rent_try
    else:
        distance = predicted_rent_try - p95

    penalty_ratio = min(distance / typical_span, 1.25)
    score = 100 - 65 * penalty_ratio
    return int(np.clip(round(score), 20, 100))


def _stability_score(ablation_predictions: Mapping[str, float] | None) -> int:
    if not ablation_predictions:
        return 60

    full_prediction = float(ablation_predictions.get("full", 0.0))
    if abs(full_prediction) < 1e-6:
        return 60

    diff_no_text = abs(float(ablation_predictions.get("no_text", full_prediction)) - full_prediction) / abs(full_prediction)
    diff_no_image = abs(float(ablation_predictions.get("no_image", full_prediction)) - full_prediction) / abs(full_prediction)
    diff_tabular_only = abs(float(ablation_predictions.get("tabular_only", full_prediction)) - full_prediction) / abs(full_prediction)

    weighted_difference = 0.4 * diff_no_text + 0.4 * diff_no_image + 0.2 * diff_tabular_only
    score = 100 - 180 * weighted_difference
    return int(np.clip(round(score), 20, 100))


def _label_for_score(score: int) -> str:
    if score >= 80:
        return "Yüksek"
    if score >= 60:
        return "Orta"
    return "Düşük"


def _reason_specs() -> dict[str, dict[str, float | str]]:
    return {
        "completeness": {
            "weight": WEIGHT_COMPLETENESS,
            "positive": "İlan bilgileri büyük ölçüde eksiksiz paylaşıldı.",
            "neutral": "Temel ilan bilgileri büyük ölçüde tamamlandı.",
            "negative": "Bazı temel ilan bilgileri eksik olduğu için güven seviyesi düşüyor.",
        },
        "visual": {
            "weight": WEIGHT_VISUAL,
            "positive": "Yeterli sayıda fotoğraf analiz edildi.",
            "neutral": "Fotoğraf sayısı orta seviyede olduğu için görsel sinyal kısmen destek sağlıyor.",
            "negative": "Fotoğraf sayısı sınırlı olduğu için görsel sinyal zayıf kalıyor.",
        },
        "location": {
            "weight": WEIGHT_LOCATION,
            "positive": "Konum bilgileri eğitim verisinde güçlü temsil ediliyor.",
            "neutral": "Seçilen ilçe ve mahalle eğitim verisinde yeterli örnek içeriyor.",
            "negative": "Seçilen konum eğitim verisinde daha sınırlı temsil ediliyor.",
        },
        "price_range": {
            "weight": WEIGHT_PRICE,
            "positive": "Tahmin fiyatı eğitim verisindeki tipik aralıkta.",
            "neutral": "Tahmin fiyatı eğitim verisindeki geniş aralığa uyumlu görünüyor.",
            "negative": "Tahmin fiyatı eğitim verisindeki tipik aralığın dışında kalıyor.",
        },
        "stability": {
            "weight": WEIGHT_STABILITY,
            "positive": "Metin ve fotoğraf katkıları çıkarıldığında sonuç görece stabil kalıyor.",
            "neutral": "Modaliteler arası katkı dengeli görünüyor.",
            "negative": "Metin veya fotoğraf katkısı çıkarıldığında sonuç belirgin değişiyor.",
        },
    }


def _build_reasons(components: dict[str, int], label: str) -> list[str]:
    specs = _reason_specs()

    weighted_components = [
        (
            name,
            score,
            float(specs[name]["weight"]),
        )
        for name, score in components.items()
    ]

    reasons: list[str] = []

    if label == "Yüksek":
        for name, score, _ in sorted(weighted_components, key=lambda item: (item[1], item[2]), reverse=True):
            spec = specs[name]
            if score >= 80:
                reasons.append(str(spec["positive"]))
            elif score >= 68:
                reasons.append(str(spec["neutral"]))
            if len(reasons) == 3:
                break
    elif label == "Düşük":
        for name, score, _ in sorted(weighted_components, key=lambda item: (item[1], -item[2])):
            spec = specs[name]
            if score <= 60:
                reasons.append(str(spec["negative"]))
            elif score <= 72:
                reasons.append(str(spec["neutral"]))
            if len(reasons) == 3:
                break
    else:
        strengths: list[str] = []
        limiters: list[str] = []
        for name, score, _ in sorted(weighted_components, key=lambda item: (item[1], item[2]), reverse=True):
            spec = specs[name]
            if score >= 78:
                strengths.append(str(spec["positive"]))
            elif score >= 68:
                strengths.append(str(spec["neutral"]))
        for name, score, _ in sorted(weighted_components, key=lambda item: (item[1], -item[2])):
            spec = specs[name]
            if score <= 68:
                limiters.append(str(spec["negative"]))
        reasons.extend(strengths[:1])
        reasons.extend(limiters[:2])
        if len(reasons) < 3:
            for item in strengths[1:]:
                if item not in reasons:
                    reasons.append(item)
                if len(reasons) == 3:
                    break

    if not reasons:
        reasons = ["Girilen bilgiler veri setindeki benzer örneklerle kısmen uyumlu görünüyor."]

    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
        if len(deduped) == 3:
            break
    return deduped


def estimate_confidence(
    *,
    input_row: Mapping[str, Any],
    cleaned_text: str,
    used_image_count: int,
    predicted_rent_try: float,
    ablation_predictions: Mapping[str, float] | None = None,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> ConfidenceEstimate:
    profile = get_confidence_profile(dataset_path)

    components = {
        "completeness": _completeness_score(
            input_row,
            cleaned_text=cleaned_text,
            used_image_count=used_image_count,
        ),
        "visual": _visual_score(used_image_count),
        "location": _location_density_score(input_row, profile),
        "price_range": _price_range_score(predicted_rent_try, profile),
        "stability": _stability_score(ablation_predictions),
    }

    weighted_score = (
        WEIGHT_COMPLETENESS * components["completeness"]
        + WEIGHT_VISUAL * components["visual"]
        + WEIGHT_LOCATION * components["location"]
        + WEIGHT_PRICE * components["price_range"]
        + WEIGHT_STABILITY * components["stability"]
    )
    score = int(np.clip(round(weighted_score), 0, 100))
    label = _label_for_score(score)
    reasons = _build_reasons(components, label)

    return ConfidenceEstimate(
        score=score,
        label=label,
        reasons=reasons,
        components={name: int(value) for name, value in components.items()},
    )
