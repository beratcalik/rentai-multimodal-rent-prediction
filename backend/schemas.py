from __future__ import annotations

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    name: str = Field(examples=["Rent Agent API"])
    status: str = Field(examples=["ok"])
    model: str = Field(examples=["final_multimodal_text_clip_model"])


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class SimilarListing(BaseModel):
    district: str
    neighborhood: str
    rooms: str
    m2_gross: float | None = None
    building_age: float | None = None
    floor: str | None = None
    price_try: int
    price_formatted: str
    similarity_score: int
    similarity_reasons: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    predicted_rent_try: int = Field(examples=[41700])
    predicted_rent_formatted: str = Field(examples=["41.700 TL"])
    used_image_count: int = Field(examples=[16])
    model_name: str = Field(examples=["XGBRegressor"])
    warnings: list[str] = Field(default_factory=list)
    message: str = Field(
        default="Tahmin ilan bilgileri, açıklama metni ve fotoğraflar birlikte analiz edilerek üretildi."
    )


class PredictWithExplanationsResponse(PredictResponse):
    confidence_score: int = Field(examples=[84])
    confidence_label: str = Field(examples=["Yüksek"])
    confidence_reasons: list[str] = Field(default_factory=list)
    top_positive_factors: list[str] = Field(default_factory=list)
    top_negative_factors: list[str] = Field(default_factory=list)
    similar_listings: list[SimilarListing] = Field(default_factory=list)
