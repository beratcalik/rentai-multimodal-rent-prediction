from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.inference_service import get_inference_service
from backend.schemas import HealthResponse, PredictResponse, RootResponse


LOGGER = logging.getLogger("rent_agent_api")

ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_ROOT = ROOT_DIR / "backend" / "uploads"
TEMP_UPLOAD_ROOT = UPLOAD_ROOT / "temp"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_IMAGE_COUNT = 16
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
API_MESSAGE = "Tahmin ilan bilgileri, açıklama metni ve fotoğraflar birlikte analiz edilerek üretildi."


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def ensure_upload_directories() -> None:
    TEMP_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


configure_logging()
ensure_upload_directories()

app = FastAPI(
    title="Rent Agent API",
    version="0.1.0",
    description="Production-like FastAPI surface for the final Rent Agent multimodal inference model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_temp_request_dir() -> Path:
    request_dir = TEMP_UPLOAD_ROOT / uuid4().hex
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def cleanup_temp_dir(temp_dir: Path | None) -> None:
    if temp_dir is None:
        return
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        LOGGER.warning("Temp klasoru temizlenemedi: %s", temp_dir, exc_info=True)


async def persist_uploaded_images(images: list[UploadFile] | None) -> tuple[list[str], int, list[str], Path | None]:
    if not images:
        return [], 0, [], None

    request_dir = build_temp_request_dir()
    warnings: list[str] = []
    saved_paths: list[str] = []
    total_uploaded_count = len(images)
    images_to_process = images[:MAX_IMAGE_COUNT]

    if total_uploaded_count > MAX_IMAGE_COUNT:
        warnings.append(
            f"{total_uploaded_count} gorsel gonderildi; ilk {MAX_IMAGE_COUNT} dosya kullanildi, kalanlar ignore edildi."
        )

    try:
        for index, uploaded_image in enumerate(images_to_process, start=1):
            original_name = uploaded_image.filename or ""
            suffix = Path(original_name).suffix.lower()
            if suffix not in ALLOWED_IMAGE_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Gecersiz dosya turu: {original_name or 'isimsiz dosya'}. Yalnizca jpg, jpeg ve png kabul edilir.",
                )

            if uploaded_image.content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Desteklenmeyen content-type: {uploaded_image.content_type}. Yalnizca JPEG ve PNG kabul edilir.",
                )

            file_bytes = await uploaded_image.read()
            if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"{original_name or 'Gorsel'} 10MB limitini asiyor.",
                )

            safe_path = request_dir / f"image_{index:02d}_{uuid4().hex}{suffix}"
            safe_path.write_bytes(file_bytes)
            saved_paths.append(str(safe_path))

    except Exception:
        cleanup_temp_dir(request_dir)
        raise
    finally:
        for uploaded_image in images:
            await uploaded_image.close()

    return saved_paths, total_uploaded_count, warnings, request_dir


def build_prediction_payload(
    *,
    city: str | None,
    district: str | None,
    neighborhood: str | None,
    rooms: str | None,
    bathrooms: str | None,
    m2_gross: str | None,
    building_age: str | None,
    floor: str | None,
    total_floors: str | None,
    heating_type: str | None,
    fuel_type: str | None,
    is_furnished: str | None,
    dues_try: str | None,
    home_type: str | None,
    home_shape: str | None,
    title: str | None,
    description: str | None,
    image_paths: list[str],
    image_count: int,
    valid_image_count: int,
) -> dict[str, str | int | list[str] | None]:
    return {
        "city": city,
        "district": district,
        "neighborhood": neighborhood,
        "rooms": rooms,
        "bathrooms": bathrooms,
        "m2_gross": m2_gross,
        "building_age": building_age,
        "floor": floor,
        "total_floors": total_floors,
        "heating_type": heating_type,
        "fuel_type": fuel_type,
        "is_furnished": is_furnished,
        "dues_try": dues_try,
        "home_type": home_type,
        "home_shape": home_shape,
        "image_count": image_count,
        "valid_image_count": valid_image_count,
        "title": title,
        "description": description,
        "image_paths": image_paths,
    }


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    service = get_inference_service()
    return RootResponse(
        name="Rent Agent API",
        status="ok",
        model=service.get_model_name(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict(
    city: Annotated[str | None, Form()] = None,
    district: Annotated[str | None, Form()] = None,
    neighborhood: Annotated[str | None, Form()] = None,
    rooms: Annotated[str | None, Form()] = None,
    bathrooms: Annotated[str | None, Form()] = None,
    m2_gross: Annotated[str | None, Form()] = None,
    building_age: Annotated[str | None, Form()] = None,
    floor: Annotated[str | None, Form()] = None,
    total_floors: Annotated[str | None, Form()] = None,
    heating_type: Annotated[str | None, Form()] = None,
    fuel_type: Annotated[str | None, Form()] = None,
    is_furnished: Annotated[str | None, Form()] = None,
    dues_try: Annotated[str | None, Form()] = None,
    home_type: Annotated[str | None, Form()] = None,
    home_shape: Annotated[str | None, Form()] = None,
    title: Annotated[str | None, Form()] = None,
    description: Annotated[str | None, Form()] = None,
    images: Annotated[list[UploadFile] | None, File()] = None,
) -> PredictResponse:
    service = get_inference_service()
    temp_dir: Path | None = None

    try:
        saved_image_paths, raw_image_count, request_warnings, temp_dir = await persist_uploaded_images(images)
        prediction_payload = build_prediction_payload(
            city=city,
            district=district,
            neighborhood=neighborhood,
            rooms=rooms,
            bathrooms=bathrooms,
            m2_gross=m2_gross,
            building_age=building_age,
            floor=floor,
            total_floors=total_floors,
            heating_type=heating_type,
            fuel_type=fuel_type,
            is_furnished=is_furnished,
            dues_try=dues_try,
            home_type=home_type,
            home_shape=home_shape,
            title=title,
            description=description,
            image_paths=saved_image_paths,
            image_count=raw_image_count,
            valid_image_count=len(saved_image_paths),
        )

        prediction_result = service.predict(prediction_payload)
        merged_warnings = request_warnings + list(prediction_result.get("warnings", []))

        return PredictResponse(
            predicted_rent_try=int(prediction_result["predicted_rent_try"]),
            predicted_rent_formatted=str(prediction_result["predicted_rent_formatted"]),
            used_image_count=int(prediction_result["used_image_count"]),
            model_name=str(prediction_result["model_name"]),
            warnings=merged_warnings,
            message=API_MESSAGE,
        )
    except HTTPException:
        raise
    except Exception:
        LOGGER.exception("Tahmin istegi islenirken beklenmeyen hata olustu.")
        raise HTTPException(
            status_code=500,
            detail="Tahmin uretilemedi. Lutfen girdileri kontrol edip tekrar deneyin.",
        )
    finally:
        cleanup_temp_dir(temp_dir)
