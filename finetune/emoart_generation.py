import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset


DEFAULT_IMAGE_SIZE = 384
DEFAULT_PATCH_SIZE = 16
_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
DEFAULT_PROMPT_TEMPLATE = "default"
DEFAULT_ART_TEXTURE_MODE = "off"
DEFAULT_ART_TEXTURE_FIELDS = "all"
PROMPT_TEMPLATE_CHOICES = ("default", "high_brushstroke", "v2_minimal")
ART_TEXTURE_MODE_CHOICES = ("off", "balanced", "strong")
ART_TEXTURE_FIELD_CHOICES = ("all", "brushstroke_only")

_STYLE_MEDIUM_HINTS = {
    "abstract expressionism": ["oil paint handling", "gestural paint layers", "canvas texture"],
    "expressionism": ["oil paint handling", "visible pigment variation", "textured paint surface"],
    "impressionism": ["broken brushwork", "layered paint", "canvas grain"],
    "post-impressionism": ["layered pigment", "directional brushwork", "paint surface texture"],
    "fauvism": ["bold pigment application", "visible brushwork", "paint texture"],
    "baroque": ["oil glazing", "layered paint depth", "canvas texture"],
    "rococo": ["refined oil paint layering", "soft brush transitions", "canvas texture"],
    "renaissance": ["oil tempera finish", "layered surface depth", "painted texture"],
    "ink and wash painting": ["ink wash gradation", "absorbent paper grain", "brush ink bleed"],
    "ukiyo-e": ["printed texture", "paper grain", "woodblock surface"],
    "native art": ["dense mark making", "pigment dots", "paint surface texture"],
    "pointillism": ["stippled pigment", "dense micro marks", "paint surface texture"],
}
_GENERIC_TEXTURE_TAGS = {
    "balanced": [
        "visible brushwork",
        "layered pigment variation",
        "tactile painted surface",
        "non-flat tonal transitions",
        "rich local texture",
    ],
    "strong": [
        "visible expressive brushstrokes",
        "thick layered pigment",
        "tactile canvas or paper grain",
        "painterly edge variation",
        "dense local texture detail",
        "material depth in the paint surface",
    ],
}


def compute_image_token_count(image_size: int, patch_size: int = DEFAULT_PATCH_SIZE) -> int:
    validate_image_generation_geometry(image_size=image_size, patch_size=patch_size)
    return (image_size // patch_size) ** 2


def validate_image_generation_geometry(
    image_size: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    expected_token_count: Optional[int] = None,
) -> int:
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")
    if image_size % patch_size != 0:
        raise ValueError(
            f"image_size={image_size} must be divisible by patch_size={patch_size}."
        )
    token_count = (image_size // patch_size) ** 2
    if expected_token_count is not None and expected_token_count != token_count:
        raise ValueError(
            "image token geometry mismatch: "
            f"image_size={image_size}, patch_size={patch_size} implies "
            f"{token_count} tokens, but expected_token_count={expected_token_count}."
        )
    return token_count


def resolve_image_token_count(
    image_size: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    requested_token_count: Optional[int] = None,
) -> int:
    if requested_token_count is None:
        return validate_image_generation_geometry(image_size=image_size, patch_size=patch_size)
    return validate_image_generation_geometry(
        image_size=image_size,
        patch_size=patch_size,
        expected_token_count=requested_token_count,
    )


DEFAULT_IMAGE_TOKEN_COUNT = compute_image_token_count(DEFAULT_IMAGE_SIZE, DEFAULT_PATCH_SIZE)


def normalize_image_path(image_path: str) -> str:
    return image_path.replace("\\", os.sep).replace("/", os.sep)

def resolve_image_path(images_root: str, image_path: str) -> str:
    normalized = normalize_image_path(image_path).lstrip(os.sep)
    root_name = os.path.basename(os.path.normpath(images_root))
    if normalized.startswith(root_name + os.sep):
        normalized = normalized[len(root_name + os.sep) :]
    return os.path.join(images_root, normalized)


def safe_get(data: Dict[str, Any], *keys: str) -> Optional[Any]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _field_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return any(_field_non_empty(v) for v in value.values())
    return True


def _stringify_list(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value).strip()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).replace("\n", " ").split())
    return text.strip(" ,.;")


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip(" ,.;")


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        normalized = cleaned.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(cleaned)
    return deduped


def extract_style_name(request_id: str) -> str:
    if "_request-" not in request_id:
        return ""
    return _clean_text(request_id.split("_request-", 1)[0])


def _contains_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    lowered = _clean_text(text).lower()
    return any(keyword in lowered for keyword in keywords)


def _infer_medium_tags(style_name: str, brushstroke_text: str, line_quality_text: str) -> List[str]:
    tags: List[str] = []
    style_key = style_name.lower()
    for candidate, medium_tags in _STYLE_MEDIUM_HINTS.items():
        if candidate in style_key:
            tags.extend(medium_tags)
    if _contains_any_keyword(brushstroke_text, ["rough", "energetic", "broken", "textured", "gestural"]):
        tags.append("textured paint handling")
    if _contains_any_keyword(brushstroke_text, ["dab", "stipple", "dot", "pointill"]):
        tags.append("dense mark making")
    if _contains_any_keyword(line_quality_text, ["soft", "broken"]):
        tags.append("organic edge variation")
    return _dedupe_keep_order(tags)


def _infer_texture_tags(
    style_name: str,
    brushstroke_text: str,
    emotional_impact: str,
    line_quality_text: str,
) -> List[str]:
    tags: List[str] = []
    if _contains_any_keyword(brushstroke_text, ["rough", "energetic", "gestural", "broken", "textured"]):
        tags.extend(["visible brushstrokes", "layered paint texture"])
    if _contains_any_keyword(brushstroke_text, ["thick", "heavy", "impasto"]):
        tags.extend(["thick pigment build-up", "paint ridges"])
    if _contains_any_keyword(brushstroke_text, ["fine", "meticulous", "delicate", "detailed"]):
        tags.extend(["fine-grained surface detail", "micro texture variation"])
    if _contains_any_keyword(brushstroke_text, ["smooth", "blending", "seamless"]):
        tags.append("subtle paint surface variation")
    if _contains_any_keyword(brushstroke_text, ["dab", "stippl", "dot", "pointill"]):
        tags.extend(["dense dotted pigment", "micro mark clusters"])
    if _contains_any_keyword(line_quality_text, ["broken", "varied", "organic"]):
        tags.append("irregular painterly edges")
    if _contains_any_keyword(emotional_impact, ["texture", "brushwork", "movement", "depth"]):
        tags.append("material brushwork depth")
    if "expressionism" in style_name.lower():
        tags.append("gestural surface energy")
    return _dedupe_keep_order(tags)


def extract_prompt_metadata(description: Dict[str, Any], request_id: str = "") -> Dict[str, Any]:
    visual_attributes = safe_get(description, "second_section", "visual_attributes") or {}
    brushstroke_text = _clean_text(visual_attributes.get("brushstroke", ""))
    line_quality_text = _clean_text(visual_attributes.get("line_quality", ""))
    emotional_impact = _clean_text(safe_get(description, "second_section", "emotional_impact"))
    style_name = extract_style_name(request_id)
    texture_tags = _infer_texture_tags(
        style_name=style_name,
        brushstroke_text=brushstroke_text,
        emotional_impact=emotional_impact,
        line_quality_text=line_quality_text,
    )
    medium_tags = _infer_medium_tags(
        style_name=style_name,
        brushstroke_text=brushstroke_text,
        line_quality_text=line_quality_text,
    )
    return {
        "style_name": style_name,
        "brushstroke_text": brushstroke_text,
        "line_quality_text": line_quality_text,
        "texture_tags": texture_tags,
        "medium_tags": medium_tags,
    }


def _build_texture_clauses(
    metadata: Dict[str, Any],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    art_texture_mode: str = DEFAULT_ART_TEXTURE_MODE,
    art_texture_fields: str = DEFAULT_ART_TEXTURE_FIELDS,
) -> List[str]:
    clauses: List[str] = []
    if (
        prompt_template == "v2_minimal"
        and art_texture_mode == DEFAULT_ART_TEXTURE_MODE
        and art_texture_fields == DEFAULT_ART_TEXTURE_FIELDS
    ):
        return clauses

    brushstroke_text = _clean_text(metadata.get("brushstroke_text", ""))
    medium_tags = _dedupe_keep_order(metadata.get("medium_tags", []))
    texture_tags = _dedupe_keep_order(metadata.get("texture_tags", []))

    if art_texture_mode != DEFAULT_ART_TEXTURE_MODE:
        texture_tags.extend(_GENERIC_TEXTURE_TAGS[art_texture_mode])

    if art_texture_fields == "brushstroke_only":
        if brushstroke_text:
            clauses.append("brushwork emphasis: " + brushstroke_text)
        if texture_tags:
            clauses.append("surface detail: " + ", ".join(_dedupe_keep_order(texture_tags)))
    else:
        if medium_tags:
            clauses.append("painting medium: " + ", ".join(medium_tags))
        if texture_tags:
            clauses.append("surface detail: " + ", ".join(_dedupe_keep_order(texture_tags)))
    if prompt_template == "high_brushstroke":
        clauses.append(
            "painterly rendering with visible brush marks, layered pigment depth, tactile surface variation, nuanced local texture, and non-flat shading"
        )

    return clauses


def _clause_label(clause: str) -> str:
    if ":" not in clause:
        return clause.strip().lower()
    return clause.split(":", 1)[0].strip().lower()


def _filter_duplicate_texture_clauses(base_prompt: str, clauses: List[str]) -> List[str]:
    lowered_prompt = base_prompt.lower()
    filtered: List[str] = []
    seen_labels = set()
    for clause in clauses:
        label = _clause_label(clause)
        if label in seen_labels:
            continue
        if label and f"{label}:" in lowered_prompt:
            continue
        filtered.append(clause)
        seen_labels.add(label)
    return filtered


def _split_prompt_clauses(prompt: str) -> List[str]:
    return [_clean_text(part) for part in prompt.split(".") if _clean_text(part)]


def _simplify_intro_clause(clause: str) -> str:
    normalized = clause.strip(" .")
    lowered = normalized.lower()
    if lowered == "expressive artwork":
        return "artwork"
    if ", expressive artwork" in lowered:
        prefix = normalized[: lowered.index(", expressive artwork")].strip(" ,.;")
        if prefix:
            return f"{prefix} artwork"
    return normalized


def _compact_existing_prompt(base_prompt: str) -> str:
    clauses = _split_prompt_clauses(base_prompt)
    if not clauses:
        return base_prompt

    kept: List[str] = []
    for clause in clauses:
        lowered = clause.lower()
        label = _clause_label(clause)

        if not kept:
            kept.append(_simplify_intro_clause(clause))
            continue

        if label in {"emotional effect", "visual style", "healing atmosphere", "painting medium", "surface detail"}:
            continue
        if lowered.startswith("mood "):
            continue
        if lowered == "coherent composition, rich detail, strong emotional presence":
            continue
        if ":" in clause:
            continue

        kept.append(clause)
        break

    return ". ".join(part.strip(" .") for part in kept if part.strip()) + "."


def _mean_color(image: Image.Image) -> tuple[int, int, int]:
    image_np = np.asarray(image.convert("RGB"), dtype=np.float32)
    mean_rgb = image_np.mean(axis=(0, 1))
    return tuple(int(channel) for channel in mean_rgb.tolist())


def build_generation_prompt(
    description: Dict[str, Any],
    request_id: str = "",
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    art_texture_mode: str = DEFAULT_ART_TEXTURE_MODE,
    art_texture_fields: str = DEFAULT_ART_TEXTURE_FIELDS,
) -> str:
    first_section = _truncate_words(
        _clean_text(safe_get(description, "first_section", "description")),
        28,
    )
    emotional_impact = _truncate_words(
        _clean_text(safe_get(description, "second_section", "emotional_impact")),
        18,
    )
    visual_attributes = safe_get(description, "second_section", "visual_attributes") or {}
    dominant_emotion = _clean_text(safe_get(description, "third_section", "dominant_emotion"))
    arousal = _clean_text(safe_get(description, "third_section", "emotional_arousal_level"))
    valence = _clean_text(safe_get(description, "third_section", "emotional_valence"))
    healing_effects = _clean_text(_stringify_list(safe_get(description, "third_section", "healing_effects")))

    prompt_metadata = extract_prompt_metadata(description, request_id=request_id)
    style_hint = prompt_metadata["style_name"]

    if prompt_template == "v2_minimal":
        prompt_parts = []
        if style_hint:
            prompt_parts.append(f"{style_hint} artwork")
        else:
            prompt_parts.append("artwork")
        if first_section:
            prompt_parts.append(first_section)
        prompt_parts.extend(
            _build_texture_clauses(
                metadata=prompt_metadata,
                prompt_template=prompt_template,
                art_texture_mode=art_texture_mode,
                art_texture_fields=art_texture_fields,
            )
        )
        return ". ".join(part.strip(" .") for part in prompt_parts if part.strip()) + "."

    visual_parts = []
    visual_key_map = {
        "brushstroke": "brushwork",
        "color": "color palette",
        "composition": "composition",
        "light_and_shadow": "lighting",
        "line_quality": "line quality",
    }
    for key, label in visual_key_map.items():
        value = _truncate_words(_clean_text(visual_attributes.get(key, "")), 10)
        if _field_non_empty(value):
            visual_parts.append(f"{label}: {value}")

    prompt_parts = []
    if style_hint:
        prompt_parts.append(f"{style_hint}, expressive artwork")
    else:
        prompt_parts.append("expressive artwork")
    if first_section:
        prompt_parts.append(first_section)
    if emotional_impact:
        prompt_parts.append(f"emotional effect: {emotional_impact}")
    emotion_tags = []
    if dominant_emotion:
        emotion_tags.append(f"mood {dominant_emotion.lower()}")
    if arousal:
        emotion_tags.append(f"{arousal.lower()} arousal")
    if valence:
        emotion_tags.append(f"{valence.lower()} tone")
    if emotion_tags:
        prompt_parts.append(", ".join(emotion_tags))
    if visual_parts:
        prompt_parts.append("visual style: " + "; ".join(visual_parts))
    if healing_effects:
        prompt_parts.append(f"healing atmosphere: {healing_effects}")
    prompt_parts.extend(
        _build_texture_clauses(
            metadata=prompt_metadata,
            prompt_template=prompt_template,
            art_texture_mode=art_texture_mode,
            art_texture_fields=art_texture_fields,
        )
    )
    prompt_parts.append("coherent composition, rich detail, strong emotional presence")

    return ". ".join(part.strip(" .") for part in prompt_parts if part.strip()) + "."


def build_prompt_from_record(
    record: Dict[str, Any],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    art_texture_mode: str = DEFAULT_ART_TEXTURE_MODE,
    art_texture_fields: str = DEFAULT_ART_TEXTURE_FIELDS,
) -> str:
    base_prompt = _clean_text(record.get("prompt", ""))
    if not base_prompt:
        return build_generation_prompt(
            description=record.get("description") or {},
            request_id=record.get("request_id", ""),
            prompt_template=prompt_template,
            art_texture_mode=art_texture_mode,
            art_texture_fields=art_texture_fields,
        )
    if prompt_template == "v2_minimal":
        compact_prompt = _compact_existing_prompt(base_prompt)
        if art_texture_mode == DEFAULT_ART_TEXTURE_MODE and art_texture_fields == DEFAULT_ART_TEXTURE_FIELDS:
            return compact_prompt
        style_name = _clean_text(record.get("style_name", "")) or extract_style_name(record.get("request_id", ""))
        brushstroke_text = _clean_text(record.get("brushstroke_text", ""))
        line_quality_text = _clean_text(record.get("line_quality_text", ""))
        metadata = {
            "style_name": style_name,
            "brushstroke_text": brushstroke_text,
            "line_quality_text": line_quality_text,
            "texture_tags": record.get("texture_tags", [])
            or _infer_texture_tags(
                style_name=style_name,
                brushstroke_text=brushstroke_text,
                emotional_impact="",
                line_quality_text=line_quality_text,
            ),
            "medium_tags": record.get("medium_tags", [])
            or _infer_medium_tags(
                style_name=style_name,
                brushstroke_text=brushstroke_text,
                line_quality_text=line_quality_text,
            ),
        }
        clauses = _build_texture_clauses(
            metadata=metadata,
            prompt_template=prompt_template,
            art_texture_mode=art_texture_mode,
            art_texture_fields=art_texture_fields,
        )
        clauses = _filter_duplicate_texture_clauses(base_prompt=compact_prompt, clauses=clauses)
        if not clauses:
            return compact_prompt
        return compact_prompt.rstrip(" .") + ". " + ". ".join(clauses) + "."
    if (
        prompt_template == DEFAULT_PROMPT_TEMPLATE
        and art_texture_mode == DEFAULT_ART_TEXTURE_MODE
        and art_texture_fields == DEFAULT_ART_TEXTURE_FIELDS
    ):
        return base_prompt

    style_name = _clean_text(record.get("style_name", "")) or extract_style_name(record.get("request_id", ""))
    brushstroke_text = _clean_text(record.get("brushstroke_text", ""))
    line_quality_text = _clean_text(record.get("line_quality_text", ""))
    metadata = {
        "style_name": style_name,
        "brushstroke_text": brushstroke_text,
        "line_quality_text": line_quality_text,
        "texture_tags": record.get("texture_tags", [])
        or _infer_texture_tags(
            style_name=style_name,
            brushstroke_text=brushstroke_text,
            emotional_impact="",
            line_quality_text=line_quality_text,
        ),
        "medium_tags": record.get("medium_tags", [])
        or _infer_medium_tags(
            style_name=style_name,
            brushstroke_text=brushstroke_text,
            line_quality_text=line_quality_text,
        ),
    }
    clauses = _build_texture_clauses(
        metadata=metadata,
        prompt_template=prompt_template,
        art_texture_mode=art_texture_mode,
        art_texture_fields=art_texture_fields,
    )
    clauses = _filter_duplicate_texture_clauses(base_prompt=base_prompt, clauses=clauses)
    if not clauses:
        return base_prompt
    return base_prompt.rstrip(" .") + ". " + ". ".join(clauses) + "."


def load_annotation(annotation_path: str) -> List[Dict[str, Any]]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_records(
    records: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    return train_records, val_records


def audit_generation_records(
    annotations: Iterable[Dict[str, Any]],
    images_root: str,
    skip_bad_images: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    valid_records: List[Dict[str, Any]] = []
    bad_records: List[Dict[str, Any]] = []
    annotations = list(annotations)
    stats = {
        "total_samples": len(annotations),
        "missing_images": 0,
        "bad_images": 0,
        "missing_request_id": 0,
        "missing_first_section_description": 0,
        "missing_second_section_emotional_impact": 0,
        "missing_visual_attributes": 0,
        "missing_dominant_emotion": 0,
        "missing_emotional_arousal_level": 0,
        "missing_emotional_valence": 0,
        "missing_healing_effects": 0,
    }

    for row in annotations:
        request_id = row.get("request_id", "")
        description = row.get("description") or {}
        image_path = resolve_image_path(images_root, row.get("image_path", ""))

        if not _field_non_empty(request_id):
            stats["missing_request_id"] += 1
        if not _field_non_empty(safe_get(description, "first_section", "description")):
            stats["missing_first_section_description"] += 1
        if not _field_non_empty(safe_get(description, "second_section", "emotional_impact")):
            stats["missing_second_section_emotional_impact"] += 1
        if not _field_non_empty(safe_get(description, "second_section", "visual_attributes")):
            stats["missing_visual_attributes"] += 1
        if not _field_non_empty(safe_get(description, "third_section", "dominant_emotion")):
            stats["missing_dominant_emotion"] += 1
        if not _field_non_empty(safe_get(description, "third_section", "emotional_arousal_level")):
            stats["missing_emotional_arousal_level"] += 1
        if not _field_non_empty(safe_get(description, "third_section", "emotional_valence")):
            stats["missing_emotional_valence"] += 1
        if not _field_non_empty(safe_get(description, "third_section", "healing_effects")):
            stats["missing_healing_effects"] += 1

        if not os.path.exists(image_path):
            stats["missing_images"] += 1
            bad_records.append({"request_id": request_id, "reason": "missing_image", "image_path": image_path})
            if skip_bad_images:
                continue

        if skip_bad_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
                stats["bad_images"] += 1
                bad_records.append(
                    {"request_id": request_id, "reason": "bad_image", "image_path": image_path, "error": str(exc)}
                )
                continue

        prompt_metadata = extract_prompt_metadata(description, request_id=request_id)
        prompt = build_generation_prompt(description, request_id=request_id)
        if not _field_non_empty(prompt):
            bad_records.append({"request_id": request_id, "reason": "empty_prompt", "image_path": image_path})
            continue

        valid_records.append(
            {
                "request_id": request_id,
                "image_path": image_path,
                "prompt": prompt,
                "style_name": prompt_metadata["style_name"],
                "brushstroke_text": prompt_metadata["brushstroke_text"],
                "line_quality_text": prompt_metadata["line_quality_text"],
                "texture_tags": prompt_metadata["texture_tags"],
                "medium_tags": prompt_metadata["medium_tags"],
            }
        )

    stats["trainable_samples"] = len(valid_records)
    return valid_records, stats, bad_records


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_emoart_generation_manifests(
    annotation_path: str,
    images_root: str,
    output_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None,
    skip_bad_images: bool = True,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    annotations = load_annotation(annotation_path)
    valid_records, stats, bad_records = audit_generation_records(
        annotations=annotations,
        images_root=images_root,
        skip_bad_images=skip_bad_images,
    )

    if max_samples is not None:
        valid_records = valid_records[:max_samples]
        stats["trainable_samples_after_cap"] = len(valid_records)
    else:
        stats["trainable_samples_after_cap"] = len(valid_records)

    train_records, val_records = split_records(valid_records, val_ratio=val_ratio, seed=seed)
    stats["train_split_size"] = len(train_records)
    stats["val_split_size"] = len(val_records)
    stats["seed"] = seed
    stats["val_ratio"] = val_ratio

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    stats_path = os.path.join(output_dir, "audit_summary.json")
    bad_path = os.path.join(output_dir, "bad_samples.jsonl")

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    write_json(stats_path, stats)
    write_jsonl(bad_path, bad_records)

    return {
        "train_path": train_path,
        "val_path": val_path,
        "stats_path": stats_path,
        "bad_path": bad_path,
        "stats": stats,
    }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def preprocess_image_for_vq(
    image: Image.Image,
    image_size: int = DEFAULT_IMAGE_SIZE,
    mode: str = "crop",
) -> torch.Tensor:
    image = image.convert("RGB")
    if mode == "crop":
        image = ImageOps.fit(
            image,
            (image_size, image_size),
            method=_BICUBIC,
            centering=(0.5, 0.5),
        )
    else:
        image = ImageOps.pad(
            image,
            (image_size, image_size),
            method=_BICUBIC,
            color=_mean_color(image),
            centering=(0.5, 0.5),
        )
    image_np = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
    return image_tensor


@dataclass
class EmoArtGenerationSample:
    request_id: str
    image_path: str
    prompt: str
    record: Dict[str, Any]


class EmoArtGenerationJsonlDataset(Dataset):
    def __init__(self, manifest_path: str):
        self.records = load_jsonl(manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EmoArtGenerationSample:
        record = self.records[index]
        return EmoArtGenerationSample(
            request_id=record["request_id"],
            image_path=record["image_path"],
            prompt=record["prompt"],
            record=record,
        )


class JanusGenerationDataCollator:
    def __init__(
        self,
        processor,
        image_size: int = DEFAULT_IMAGE_SIZE,
        image_preprocess_mode: str = "crop",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        art_texture_mode: str = DEFAULT_ART_TEXTURE_MODE,
        art_texture_fields: str = DEFAULT_ART_TEXTURE_FIELDS,
        art_texture_prob: float = 0.0,
    ):
        self.processor = processor
        self.image_size = image_size
        self.image_preprocess_mode = image_preprocess_mode
        self.prompt_template = prompt_template
        self.art_texture_mode = art_texture_mode
        self.art_texture_fields = art_texture_fields
        self.art_texture_prob = art_texture_prob

    def build_prompt_text(self, prompt: str) -> str:
        conversation = [
            {"role": "User", "content": prompt},
            {"role": "Assistant", "content": ""},
        ]
        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        return sft_format + self.processor.image_start_tag

    def should_enhance_prompt(self) -> bool:
        return (
            self.prompt_template != DEFAULT_PROMPT_TEMPLATE
            or self.art_texture_mode != DEFAULT_ART_TEXTURE_MODE
            or self.art_texture_fields != DEFAULT_ART_TEXTURE_FIELDS
        )

    def render_prompt(self, feature: EmoArtGenerationSample) -> str:
        if not self.should_enhance_prompt():
            return feature.prompt

        if self.prompt_template != DEFAULT_PROMPT_TEMPLATE:
            use_texture_variant = (
                self.art_texture_mode != DEFAULT_ART_TEXTURE_MODE
                or self.art_texture_fields != DEFAULT_ART_TEXTURE_FIELDS
            ) and self.art_texture_prob > 0.0 and random.random() <= self.art_texture_prob
            return build_prompt_from_record(
                feature.record,
                prompt_template=self.prompt_template,
                art_texture_mode=self.art_texture_mode if use_texture_variant else DEFAULT_ART_TEXTURE_MODE,
                art_texture_fields=self.art_texture_fields if use_texture_variant else DEFAULT_ART_TEXTURE_FIELDS,
            )

        if self.art_texture_prob <= 0.0 or random.random() > self.art_texture_prob:
            return feature.prompt
        return build_prompt_from_record(
            feature.record,
            prompt_template=self.prompt_template,
            art_texture_mode=self.art_texture_mode,
            art_texture_fields=self.art_texture_fields,
        )

    def __call__(self, features: List[EmoArtGenerationSample]) -> Dict[str, Any]:
        prompt_ids: List[torch.Tensor] = []
        pixel_values: List[torch.Tensor] = []
        prompts: List[str] = []
        request_ids: List[str] = []
        image_paths: List[str] = []

        for feature in features:
            with Image.open(feature.image_path) as image:
                pixel_values.append(
                    preprocess_image_for_vq(
                        image,
                        image_size=self.image_size,
                        mode=self.image_preprocess_mode,
                    )
                )

            rendered_prompt = self.render_prompt(feature)
            prompt_text = self.build_prompt_text(rendered_prompt)
            prompt_id = torch.tensor(self.processor.tokenizer.encode(prompt_text), dtype=torch.long)
            prompt_ids.append(prompt_id)
            prompts.append(rendered_prompt)
            request_ids.append(feature.request_id)
            image_paths.append(feature.image_path)

        return {
            "prompt_ids": prompt_ids,
            "pixel_values": torch.stack(pixel_values, dim=0),
            "prompts": prompts,
            "request_ids": request_ids,
            "image_paths": image_paths,
        }
