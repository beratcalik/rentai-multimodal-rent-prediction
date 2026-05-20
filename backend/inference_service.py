from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.predict_single_listing import (
    DEFAULT_MODEL_PATH,
    get_clip_runtime,
    get_model_bundle,
    predict_from_dict,
)


LOGGER = logging.getLogger("rent_agent_inference_service")

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FINAL_MODEL_NAME = "final_multimodal_text_clip_model"


class InferenceService:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = (model_path or DEFAULT_MODEL_PATH).resolve()

    def get_model_name(self) -> str:
        return DEFAULT_FINAL_MODEL_NAME

    def get_model_bundle(self) -> dict[str, Any]:
        return get_model_bundle(self.model_path)

    def warm_runtime(self) -> None:
        LOGGER.info("Inference runtime on yukleme kontrolu baslatiliyor.")
        _ = self.get_model_bundle()
        _ = get_clip_runtime()

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        return predict_from_dict(
            input_data=input_data,
            model_path=self.model_path,
        )


_SERVICE = InferenceService()


def get_inference_service() -> InferenceService:
    return _SERVICE
