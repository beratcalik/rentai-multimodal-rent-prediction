from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import shap
import torch
from shap.explainers import _tree as shap_tree

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.confidence_estimation import estimate_confidence
from src.extract_clip_image_embeddings import OPENCLIP_MODEL_NAME, l2_normalize
from src.predict_single_listing import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTION_MESSAGE,
    PreparedPredictionArtifacts,
    format_amount,
    get_clip_runtime,
    prepare_prediction_artifacts,
)
from src.similar_listing_retrieval import retrieve_similar_listings_from_dict
from src.train_final_multimodal_text_clip import (
    DESCRIPTION_COLUMN,
    TITLE_COLUMN,
    collapse_whitespace,
    strip_combining_marks,
)


LOGGER = logging.getLogger("prediction_explanations")

DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "explainability_report.md"
MAX_TOP_FACTORS = 5

POSITIVE_IMAGE_CONCEPTS = [
    ("modern apartment interior", "Görseller daha modern ve bakımlı bir ev algısı oluşturuyor."),
    ("bright apartment interior", "Görseller aydınlık ve ferah bir ev algısı oluşturuyor."),
    ("spacious living room", "Görseller daha geniş yaşam alanları sinyali veriyor."),
    ("renovated kitchen and bathroom", "Görseller yenilenmiş iç mekân hissi veriyor."),
]

NEGATIVE_IMAGE_CONCEPTS = [
    ("dated apartment interior", "Görseller daha eski veya yenileme ihtiyacı olan bir görünüm sinyali veriyor."),
    ("dim apartment interior", "Görseller daha karanlık alanlar sinyali veriyor."),
    ("empty apartment room", "Görseller daha sade ve boş alanlar izlenimi veriyor."),
    ("unfinished apartment interior", "Görseller tamamlanmamış veya mütevazı bir görünüm sinyali veriyor."),
]

EXPLANATION_TEXT_STOPWORDS = {
    "alternatifler",
    "bakiniz",
    "bina",
    "bilgi",
    "caddeye",
    "daire",
    "diger",
    "ekibimiz",
    "genis",
    "ilanlarimiza",
    "kat",
    "m2",
    "memur",
    "ofisimize",
    "sahibi",
    "sokak",
    "tarafindan",
    "tecrubemiz",
    "ucretsiz",
    "yasam",
    "yillik",
}


@dataclass
class FactorExplanation:
    label: str
    message: str
    score: float
    modality: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SHAP-based local explanations for a single multimodal rent prediction."
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the explanation payload as JSON instead of a human summary.",
    )
    return parser.parse_args()


def load_input_payload(input_path: Path) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON bulunamadı: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Input JSON tek bir ilan nesnesi olmalıdır.")
    return payload


def _normalize_xgb_base_score(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.startswith("[") and normalized.endswith("]"):
            normalized = normalized[1:-1]
        return float(normalized)
    return float(value)


def ensure_shap_xgboost_compatibility() -> None:
    if getattr(shap_tree.XGBTreeModelLoader, "_rent_agent_patched", False):
        return

    def patched_init(self, xgb_model) -> None:  # type: ignore[no-untyped-def]
        import xgboost as xgb

        shap_tree._check_xgboost_version(xgb.__version__)
        model: xgb.Booster = xgb_model

        raw = xgb_model.save_raw(raw_format="ubj")
        with io.BytesIO(raw) as fd:
            jmodel = shap_tree.decode_ubjson_buffer(fd)

        learner = jmodel["learner"]
        learner_model_param = learner["learner_model_param"]
        objective = learner["objective"]

        booster = learner["gradient_booster"]
        n_classes = max(int(learner_model_param["num_class"]), 1)
        n_targets = max(int(learner_model_param["num_target"]), 1)
        n_targets = max(n_targets, n_classes)

        if "gbtree" in booster and "model" not in booster:
            booster = booster["gbtree"]

        if booster["model"].get("iteration_indptr", None) is not None:
            iteration_indptr = np.asarray(booster["model"]["iteration_indptr"], dtype=np.int32)
            diff = np.diff(iteration_indptr)
        else:
            n_parallel_trees = int(booster["model"]["gbtree_model_param"]["num_parallel_tree"])
            diff = np.repeat(n_targets * n_parallel_trees, model.num_boosted_rounds())
        if np.any(diff != diff[0]):
            raise ValueError("vector-leaf is not yet supported.:", diff)

        self.n_trees_per_iter = int(diff[0])
        self.n_targets = n_targets
        base_score = _normalize_xgb_base_score(learner_model_param["base_score"])
        self.base_score = base_score
        assert self.n_trees_per_iter > 0

        self.name_obj = objective["name"]
        self.name_gbm = booster["name"]
        if self.name_obj in ("binary:logistic", "reg:logistic"):
            self.base_score = shap_tree.scipy.special.logit(base_score)
        elif self.name_obj in (
            "reg:gamma",
            "reg:tweedie",
            "count:poisson",
            "survival:cox",
            "survival:aft",
        ):
            self.base_score = np.log(self.base_score)
        else:
            self.base_score = base_score

        self.num_feature = int(learner_model_param["num_feature"])
        self.num_class = int(learner_model_param["num_class"])

        trees = booster["model"]["trees"]
        self.num_trees = len(trees)

        self.node_parents = []
        self.node_cleft = []
        self.node_cright = []
        self.node_sindex = []
        self.children_default = []
        self.sum_hess = []

        self.values = []
        self.thresholds = []
        self.threshold_types = []
        self.features = []

        self.split_types = []
        self.categories = []

        feature_types = model.feature_types
        if feature_types is not None:
            cat_feature_indices: np.ndarray = np.where(np.asarray(feature_types) == "c")[0]
            if len(cat_feature_indices) == 0:
                self.cat_feature_indices = None
            else:
                self.cat_feature_indices = cat_feature_indices
        else:
            self.cat_feature_indices = None

        def to_integers(data: list[int]) -> np.ndarray:
            assert isinstance(data, list)
            return np.asanyarray(data, dtype=np.uint8)

        for tree in trees:
            self.node_parents.append(np.asarray(tree["parents"]))
            self.node_cleft.append(np.asarray(tree["left_children"], dtype=np.int32))
            self.node_cright.append(np.asarray(tree["right_children"], dtype=np.int32))
            self.node_sindex.append(np.asarray(tree["split_indices"], dtype=np.uint32))

            base_weight = np.asarray(tree["base_weights"], dtype=np.float32)
            if base_weight.size != self.node_cleft[-1].size:
                raise ValueError("vector-leaf is not yet supported.")

            default_left = to_integers(tree["default_left"])
            default_child = np.where(default_left == 1, self.node_cleft[-1], self.node_cright[-1]).astype(np.int64)
            self.children_default.append(default_child)
            self.sum_hess.append(np.asarray(tree["sum_hessian"], dtype=np.float64))

            is_leaf = self.node_cleft[-1] == -1
            split_cond = np.asarray(tree["split_conditions"], dtype=np.float32)
            leaf_weight = np.where(is_leaf, split_cond, 0.0)
            thresholds = np.where(is_leaf, 0.0, split_cond)
            thresholds = np.where(is_leaf, 0.0, np.nextafter(thresholds, -np.float32(np.inf)))
            threshold_types = np.zeros_like(thresholds, dtype=np.int32)

            self.values.append(leaf_weight.reshape(leaf_weight.size, 1))
            self.thresholds.append(thresholds)
            self.threshold_types.append(threshold_types)

            split_idx = np.asarray(tree["split_indices"], dtype=np.int64)
            self.features.append(split_idx)

            split_types = to_integers(tree["split_type"])
            self.split_types.append(split_types)
            cat_segments: list[int] = tree["categories_segments"]
            cat_sizes: list[int] = tree["categories_sizes"]
            cat_nodes: list[int] = tree["categories_nodes"]
            assert len(cat_segments) == len(cat_sizes) == len(cat_nodes)
            cats = tree["categories"]

            tree_categories = self.parse_categories(cat_nodes, cat_segments, cat_sizes, cats, self.node_cleft[-1])
            self.categories.append(tree_categories)

    shap_tree.XGBTreeModelLoader.__init__ = patched_init
    shap_tree.XGBTreeModelLoader._rent_agent_patched = True


@lru_cache(maxsize=4)
def get_tree_explainer(model_path: str | Path | None = None) -> shap.TreeExplainer:
    ensure_shap_xgboost_compatibility()
    from src.predict_single_listing import get_model_bundle

    bundle = get_model_bundle(model_path)
    return shap.TreeExplainer(bundle["regressor"])


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _coerce_display_value(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "Evet" if value else "Hayır"
    if isinstance(value, str):
        normalized = value.strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
            numeric_value = float(normalized)
            if numeric_value.is_integer():
                return str(int(numeric_value))
            return str(round(numeric_value, 2)).rstrip("0").rstrip(".")
        return normalized
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return str(round(float(value), 2))
    return str(value).strip()


def _join_with_conjunction(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} ve {items[1]}"
    return f"{', '.join(items[:-1])} ve {items[-1]}"


def _wrap_terms(terms: list[str]) -> str:
    wrapped = [f"“{term}”" for term in terms]
    return _join_with_conjunction(wrapped)


def _build_token_display_map(artifacts: PreparedPredictionArtifacts) -> dict[str, str]:
    raw_text = " ".join(
        part
        for part in [
            collapse_whitespace(artifacts.dataframe.iloc[0].get(TITLE_COLUMN, "")),
            collapse_whitespace(artifacts.dataframe.iloc[0].get(DESCRIPTION_COLUMN, "")),
        ]
        if part
    ).lower()
    token_map: dict[str, str] = {}
    for raw_token in re.findall(r"[0-9a-zA-ZÇĞİÖŞÜçğıöşü]+", raw_text):
        normalized = strip_combining_marks(raw_token.lower())
        if normalized and normalized not in token_map:
            token_map[normalized] = raw_token
    return token_map


def _render_text_term(term: str, token_display_map: dict[str, str]) -> str:
    parts = []
    for piece in term.split():
        parts.append(token_display_map.get(piece, piece))
    return " ".join(parts)


def _build_tabular_group_scores(artifacts: PreparedPredictionArtifacts, shap_values: np.ndarray) -> list[FactorExplanation]:
    bundle = artifacts.bundle
    tabular_feature_names = list(bundle["feature_names"][: artifacts.feature_blocks.tabular.shape[1]])
    tabular_scores = shap_values[: len(tabular_feature_names)]
    score_by_feature = {
        feature_name: float(score)
        for feature_name, score in zip(tabular_feature_names, tabular_scores, strict=False)
    }
    row = artifacts.dataframe.iloc[0]

    def sum_prefixes(prefixes: tuple[str, ...]) -> float:
        return float(
            sum(
                score
                for feature_name, score in score_by_feature.items()
                if any(feature_name.startswith(prefix) for prefix in prefixes)
            )
        )

    def maybe_add(
        label: str,
        score: float,
        positive_message: str,
        negative_message: str,
        output: list[FactorExplanation],
    ) -> None:
        if abs(score) < 1e-6:
            return
        output.append(
            FactorExplanation(
                label=label,
                message=positive_message if score >= 0 else negative_message,
                score=score,
                modality="tabular",
            )
        )

    location_bits = [
        _coerce_display_value(row.get("district")),
        _coerce_display_value(row.get("neighborhood")),
    ]
    location_bits = [item for item in location_bits if item]
    location_label = " / ".join(location_bits) or _coerce_display_value(row.get("city")) or "Konum"

    location_score = sum_prefixes(("city_", "district_", "neighborhood_"))
    rooms_score = float(score_by_feature.get("rooms", 0.0))
    size_score = float(score_by_feature.get("m2_gross", 0.0))
    bathrooms_score = float(score_by_feature.get("bathrooms", 0.0))
    building_age_score = float(score_by_feature.get("building_age", 0.0))
    floor_score = float(score_by_feature.get("floor", 0.0) + score_by_feature.get("total_floors", 0.0))
    heating_score = sum_prefixes(("heating_type_", "fuel_type_"))
    furnishing_score = sum_prefixes(("is_furnished_",))
    dues_score = float(score_by_feature.get("dues_try", 0.0))
    home_score = sum_prefixes(("home_type_", "home_shape_"))

    factors: list[FactorExplanation] = []

    if location_label:
        maybe_add(
            "location",
            location_score,
            f"{location_label} lokasyonu tahmini yukarı çekiyor.",
            f"{location_label} lokasyonu tahmini aşağı çekiyor.",
            factors,
        )

    rooms_value = _coerce_display_value(row.get("rooms"))
    if rooms_value:
        maybe_add(
            "rooms",
            rooms_score,
            f"{rooms_value} oda planı tahmini yukarı çekiyor.",
            f"{rooms_value} oda planı tahmini aşağı çekiyor.",
            factors,
        )

    m2_value = _coerce_display_value(row.get("m2_gross"))
    if m2_value:
        maybe_add(
            "m2_gross",
            size_score,
            f"Brüt {m2_value} m² büyüklük tahmini yukarı çekiyor.",
            f"Brüt {m2_value} m² büyüklük tahmini aşağı çekiyor.",
            factors,
        )

    bathroom_value = _coerce_display_value(row.get("bathrooms"))
    if bathroom_value:
        maybe_add(
            "bathrooms",
            bathrooms_score,
            f"{bathroom_value} banyo bilgisi tahmini yukarı çekiyor.",
            f"{bathroom_value} banyo bilgisi tahmini aşağı çekiyor.",
            factors,
        )

    building_age_value = row.get("building_age")
    if not _is_missing(building_age_value):
        building_age_number = float(building_age_value)
        if building_age_number <= 5:
            building_age_label = "Görece yeni bina yaşı"
        elif building_age_number >= 20:
            building_age_label = "İleri bina yaşı"
        else:
            building_age_label = "Bina yaşı"
        maybe_add(
            "building_age",
            building_age_score,
            f"{building_age_label} tahmini yukarı çekiyor.",
            f"{building_age_label} tahmini aşağı çekiyor.",
            factors,
        )

    floor_value = _coerce_display_value(row.get("floor"))
    total_floors_value = _coerce_display_value(row.get("total_floors"))
    floor_label = _join_with_conjunction([item for item in [floor_value, total_floors_value and f"{total_floors_value} katlı bina"] if item])
    if floor_label:
        maybe_add(
            "floor",
            floor_score,
            f"{floor_label} bilgisi tahmini yukarı çekiyor.",
            f"{floor_label} bilgisi tahmini aşağı çekiyor.",
            factors,
        )

    heating_parts = [_coerce_display_value(row.get("heating_type")), _coerce_display_value(row.get("fuel_type"))]
    heating_parts = [item for item in heating_parts if item]
    if heating_parts:
        heating_label = _join_with_conjunction(heating_parts)
        maybe_add(
            "heating",
            heating_score,
            f"{heating_label} tahmini yukarı çekiyor.",
            f"{heating_label} tahmini aşağı çekiyor.",
            factors,
        )

    furnished_value = row.get("is_furnished")
    if not _is_missing(furnished_value):
        if isinstance(furnished_value, (bool, np.bool_)):
            furnishing_label = "Eşyalı olması" if furnished_value else "Eşyasız olması"
        else:
            furnishing_label = f"{_coerce_display_value(furnished_value)} bilgisi"
        maybe_add(
            "is_furnished",
            furnishing_score,
            f"{furnishing_label} tahmini yukarı çekiyor.",
            f"{furnishing_label} tahmini aşağı çekiyor.",
            factors,
        )

    dues_value = row.get("dues_try")
    if not _is_missing(dues_value):
        maybe_add(
            "dues_try",
            dues_score,
            "Aidat seviyesi tahmini yukarı çekiyor.",
            "Aidat seviyesi tahmini aşağı çekiyor.",
            factors,
        )

    home_parts = [_coerce_display_value(row.get("home_type")), _coerce_display_value(row.get("home_shape"))]
    home_parts = [item for item in home_parts if item]
    if home_parts:
        home_label = _join_with_conjunction(home_parts)
        maybe_add(
            "home_profile",
            home_score,
            f"{home_label} bilgisi tahmini yukarı çekiyor.",
            f"{home_label} bilgisi tahmini aşağı çekiyor.",
            factors,
        )

    return factors


def _build_text_factor_explanations(artifacts: PreparedPredictionArtifacts, shap_values: np.ndarray) -> list[FactorExplanation]:
    if not artifacts.cleaned_text:
        return []

    bundle = artifacts.bundle
    tabular_dim = artifacts.feature_blocks.tabular.shape[1]
    text_dim = artifacts.feature_blocks.text_svd.shape[1]
    text_scores = np.asarray(shap_values[tabular_dim : tabular_dim + text_dim], dtype=np.float32)
    if not np.any(np.abs(text_scores) > 1e-6):
        return []

    tfidf_matrix = artifacts.feature_blocks.text_tfidf
    if getattr(tfidf_matrix, "nnz", 0) == 0:
        return []

    vectorizer_feature_names = np.asarray(bundle["text_vectorizer"].get_feature_names_out(), dtype=object)
    svd_components = np.asarray(bundle["text_svd"].components_, dtype=np.float32)
    tfidf_indices = np.asarray(tfidf_matrix.indices, dtype=np.int32)
    tfidf_data = np.asarray(tfidf_matrix.data, dtype=np.float32)
    projected_scores = svd_components[:, tfidf_indices].T @ text_scores
    token_scores = tfidf_data * projected_scores

    aggregated_scores: dict[str, float] = {}
    for index, score in zip(tfidf_indices.tolist(), token_scores.tolist(), strict=False):
        token = str(vectorizer_feature_names[index]).strip()
        if not token or re.fullmatch(r"\d+", token) or token in EXPLANATION_TEXT_STOPWORDS:
            continue
        aggregated_scores[token] = aggregated_scores.get(token, 0.0) + float(score)

    if not aggregated_scores:
        return []

    token_display_map = _build_token_display_map(artifacts)

    positive_terms = [
        _render_text_term(token, token_display_map)
        for token, score in sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)
        if score > 0
    ][:3]
    negative_terms = [
        _render_text_term(token, token_display_map)
        for token, score in sorted(aggregated_scores.items(), key=lambda item: item[1])
        if score < 0
    ][:3]

    total_text_score = float(text_scores.sum())
    positive_score = float(sum(score for score in aggregated_scores.values() if score > 0))
    negative_score = float(sum(score for score in aggregated_scores.values() if score < 0))

    factors: list[FactorExplanation] = []
    if positive_terms and positive_score > 0:
        factors.append(
            FactorExplanation(
                label="text_positive",
                message=f"İlan metnindeki {_wrap_terms(positive_terms)} ifadeleri tahmini yukarı çekiyor.",
                score=positive_score if total_text_score >= 0 else positive_score * 0.75,
                modality="text",
            )
        )
    if negative_terms and negative_score < 0:
        factors.append(
            FactorExplanation(
                label="text_negative",
                message=f"İlan metnindeki {_wrap_terms(negative_terms)} ifadeleri tahmini aşağı çekiyor.",
                score=negative_score if total_text_score <= 0 else negative_score * 0.75,
                modality="text",
            )
        )
    return factors


@lru_cache(maxsize=1)
def get_clip_prompt_bank() -> dict[str, tuple[list[str], np.ndarray]]:
    device, clip_model, _ = get_clip_runtime()
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL_NAME)
    prompt_banks = {
        "positive": POSITIVE_IMAGE_CONCEPTS,
        "negative": NEGATIVE_IMAGE_CONCEPTS,
    }

    encoded_banks: dict[str, tuple[list[str], np.ndarray]] = {}
    for key, concept_items in prompt_banks.items():
        prompts = [prompt for prompt, _ in concept_items]
        text_tokens = tokenizer(prompts).to(device)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    text_features = clip_model.encode_text(text_tokens)
            else:
                text_features = clip_model.encode_text(text_tokens)

        encoded_banks[key] = (
            [message for _, message in concept_items],
            l2_normalize(text_features.detach().float().cpu().numpy().astype(np.float32, copy=False)),
        )

    return encoded_banks


def _build_image_factor_explanations(artifacts: PreparedPredictionArtifacts, shap_values: np.ndarray) -> list[FactorExplanation]:
    if artifacts.image_artifacts.used_image_count == 0:
        return []

    image_dim = artifacts.feature_blocks.image_reduced.shape[1]
    if image_dim == 0:
        return []

    image_scores = np.asarray(shap_values[-image_dim:], dtype=np.float32)
    total_score = float(image_scores.sum())
    if abs(total_score) < 1e-6:
        return []

    prompt_bank = get_clip_prompt_bank()
    mean_embedding = np.asarray(artifacts.image_artifacts.mean_embedding, dtype=np.float32)
    if not np.any(mean_embedding):
        generic_message = (
            "Fotoğraflar modelde olumlu bir kalite sinyali oluşturuyor."
            if total_score >= 0
            else "Fotoğraflar modelde daha mütevazı bir kalite sinyali oluşturuyor."
        )
        return [FactorExplanation(label="image", message=generic_message, score=total_score, modality="image")]

    bank_key = "positive" if total_score >= 0 else "negative"
    messages, prompt_embeddings = prompt_bank[bank_key]
    similarities = prompt_embeddings @ mean_embedding[0]
    best_index = int(np.argmax(similarities))
    return [
        FactorExplanation(
            label="image",
            message=messages[best_index],
            score=total_score,
            modality="image",
        )
    ]


def _predict_with_blocks(bundle: dict[str, Any], x_tabular: np.ndarray, x_text: np.ndarray, x_image: np.ndarray) -> float:
    fused = np.hstack([x_tabular, x_text, x_image]).astype(np.float32, copy=False)
    return float(np.asarray(bundle["regressor"].predict(fused), dtype=float).ravel()[0])


def compute_ablation_predictions(artifacts: PreparedPredictionArtifacts) -> dict[str, float]:
    bundle = artifacts.bundle
    x_tabular = artifacts.feature_blocks.tabular
    x_text = artifacts.feature_blocks.text_svd
    x_image = artifacts.feature_blocks.image_reduced
    zero_text = np.zeros_like(x_text)
    zero_image = np.zeros_like(x_image)

    return {
        "full": artifacts.raw_prediction_try,
        "no_text": _predict_with_blocks(bundle, x_tabular, zero_text, x_image),
        "no_image": _predict_with_blocks(bundle, x_tabular, x_text, zero_image),
        "tabular_only": _predict_with_blocks(bundle, x_tabular, zero_text, zero_image),
    }


def build_explanation_factors(artifacts: PreparedPredictionArtifacts, shap_values: np.ndarray) -> tuple[list[FactorExplanation], list[FactorExplanation]]:
    tabular_factors = _build_tabular_group_scores(artifacts, shap_values)
    text_factors = _build_text_factor_explanations(artifacts, shap_values)
    image_factors = _build_image_factor_explanations(artifacts, shap_values)

    all_factors = tabular_factors + text_factors + image_factors
    positive = sorted((factor for factor in all_factors if factor.score > 0), key=lambda item: item.score, reverse=True)
    negative = sorted((factor for factor in all_factors if factor.score < 0), key=lambda item: item.score)
    return positive[:MAX_TOP_FACTORS], negative[:MAX_TOP_FACTORS]


def predict_with_explanations_from_dict(
    input_data: dict[str, Any],
    model_path: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    artifacts = prepare_prediction_artifacts(
        input_data=input_data,
        model_path=model_path,
        batch_size=batch_size,
    )
    explainer = get_tree_explainer(model_path=model_path)
    shap_output = explainer.shap_values(artifacts.feature_blocks.fused)
    shap_values = np.asarray(shap_output, dtype=np.float32).reshape(-1)

    positive_factors, negative_factors = build_explanation_factors(artifacts, shap_values)
    ablation_predictions = compute_ablation_predictions(artifacts)
    similar_listings = retrieve_similar_listings_from_dict(
        input_data,
        predicted_rent_try=artifacts.raw_prediction_try,
    )
    confidence_estimate = estimate_confidence(
        input_row=artifacts.dataframe.iloc[0].to_dict(),
        cleaned_text=artifacts.cleaned_text,
        used_image_count=artifacts.image_artifacts.used_image_count,
        predicted_rent_try=artifacts.raw_prediction_try,
        ablation_predictions=ablation_predictions,
    )
    rounded_prediction = int(round(artifacts.raw_prediction_try))

    return {
        "predicted_rent_try": rounded_prediction,
        "predicted_rent_formatted": f"{format_amount(artifacts.raw_prediction_try)} TL",
        "used_image_count": artifacts.image_artifacts.used_image_count,
        "model_name": str(artifacts.bundle["model_name"]),
        "warnings": list(artifacts.warnings),
        "message": DEFAULT_PREDICTION_MESSAGE,
        "raw_prediction_try": artifacts.raw_prediction_try,
        "confidence_score": confidence_estimate.score,
        "confidence_label": confidence_estimate.label,
        "confidence_reasons": list(confidence_estimate.reasons),
        "top_positive_factors": [factor.message for factor in positive_factors],
        "top_negative_factors": [factor.message for factor in negative_factors],
        "similar_listings": similar_listings,
        "ablation_predictions": {key: round(value, 2) for key, value in ablation_predictions.items()},
    }


def print_human_summary(result: dict[str, Any]) -> None:
    print(f"Tahmini kira: {result['predicted_rent_formatted']}")
    print(f"Tahmin güven seviyesi: {result['confidence_label']} ({result['confidence_score']}/100)")
    print(f"Kullanılan görsel sayısı: {result['used_image_count']}")
    if result.get("confidence_reasons"):
        print("Güven nedenleri:")
        for item in result["confidence_reasons"]:
            print(f"- {item}")
    if result["top_positive_factors"]:
        print("Pozitif etkiler:")
        for item in result["top_positive_factors"]:
            print(f"- {item}")
    if result["top_negative_factors"]:
        print("Negatif etkiler:")
        for item in result["top_negative_factors"]:
            print(f"- {item}")
    if result["warnings"]:
        print("Uyarılar:")
        for item in result["warnings"]:
            print(f"- {item}")
    if result.get("similar_listings"):
        print("Benzer piyasa örnekleri:")
        for item in result["similar_listings"]:
            location = " / ".join(part for part in [item.get("district"), item.get("neighborhood")] if part)
            print(f"- {location}: {item['price_formatted']} (%{item['similarity_score']})")


def main() -> int:
    configure_logging()
    args = parse_args()
    payload = load_input_payload(args.input.resolve())
    result = predict_with_explanations_from_dict(
        input_data=payload,
        model_path=args.model.resolve(),
        batch_size=args.batch_size,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
