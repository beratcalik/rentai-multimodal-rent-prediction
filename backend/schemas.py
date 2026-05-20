from __future__ import annotations

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    name: str = Field(examples=["Rent Agent API"])
    status: str = Field(examples=["ok"])
    model: str = Field(examples=["final_multimodal_text_clip_model"])


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class PredictResponse(BaseModel):
    predicted_rent_try: int = Field(examples=[41700])
    predicted_rent_formatted: str = Field(examples=["41.700 TL"])
    used_image_count: int = Field(examples=[16])
    model_name: str = Field(examples=["XGBRegressor"])
    warnings: list[str] = Field(default_factory=list)
    message: str = Field(
        default="Tahmin ilan bilgileri, açıklama metni ve fotoğraflar birlikte analiz edilerek üretildi."
    )
