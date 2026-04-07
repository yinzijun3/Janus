import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


DEFAULT_PROMPT = (
    "Analyze this artwork and return a JSON object with keys "
    "`scene_description`, `emotional_impact`, `visual_attributes`, "
    "`dominant_emotion`, `emotional_arousal_level`, `emotional_valence`, "
    "and `healing_effects`."
)


def normalize_image_path(image_path: str) -> str:
    return image_path.replace("\\", os.sep).replace("/", os.sep)


def safe_get(data: Dict[str, Any], *keys: str) -> Optional[Any]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def build_target_record(description: Dict[str, Any]) -> Dict[str, Any]:
    visual_attributes = safe_get(description, "second_section", "visual_attributes") or {}
    healing_effects = safe_get(description, "third_section", "healing_effects") or []
    if not isinstance(healing_effects, list):
        healing_effects = [str(healing_effects)]

    return {
        "scene_description": safe_get(description, "first_section", "description") or "",
        "emotional_impact": safe_get(description, "second_section", "emotional_impact") or "",
        "visual_attributes": {
            "brushstroke": visual_attributes.get("brushstroke", ""),
            "color": visual_attributes.get("color", ""),
            "composition": visual_attributes.get("composition", ""),
            "light_and_shadow": visual_attributes.get("light_and_shadow", ""),
            "line_quality": visual_attributes.get("line_quality", ""),
        },
        "dominant_emotion": safe_get(description, "third_section", "dominant_emotion") or "",
        "emotional_arousal_level": safe_get(description, "third_section", "emotional_arousal_level") or "",
        "emotional_valence": safe_get(description, "third_section", "emotional_valence") or "",
        "healing_effects": healing_effects,
    }


def build_target_text(description: Dict[str, Any]) -> str:
    return json.dumps(build_target_record(description), ensure_ascii=False, sort_keys=False)


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


def audit_records(
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
        request_id = row.get("request_id")
        description = row.get("description") or {}
        image_path = os.path.join(images_root, normalize_image_path(row.get("image_path", "")))

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

        target_record = build_target_record(description)
        target_text = json.dumps(target_record, ensure_ascii=False)

        valid_records.append(
            {
                "request_id": request_id,
                "image_path": image_path,
                "prompt": DEFAULT_PROMPT,
                "target_text": target_text,
                "target_record": target_record,
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


def prepare_emoart_manifests(
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
    valid_records, stats, bad_records = audit_records(
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


@dataclass
class EmoArtSample:
    request_id: str
    image_path: str
    prompt: str
    target_text: str
    target_record: Dict[str, Any]


class EmoArtJsonlDataset(Dataset):
    def __init__(self, manifest_path: str):
        self.records = load_jsonl(manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EmoArtSample:
        record = self.records[index]
        return EmoArtSample(
            request_id=record["request_id"],
            image_path=record["image_path"],
            prompt=record["prompt"],
            target_text=record["target_text"],
            target_record=record["target_record"],
        )


class JanusVLDataCollator:
    def __init__(
        self,
        processor,
        max_seq_len: int = 512,
        ignore_index: int = -100,
    ):
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index

    def _truncate_input_ids(self, prepare) -> Any:
        if len(prepare.input_ids) <= self.max_seq_len:
            return prepare
        prepare.input_ids = prepare.input_ids[: self.max_seq_len]
        return prepare

    def __call__(self, features: List[EmoArtSample]) -> Dict[str, Any]:
        full_prepares = []
        prompt_lengths = []
        request_ids = []
        references = []
        image_paths = []

        for feature in features:
            with Image.open(feature.image_path) as image:
                pil_image = image.convert("RGB")

                prompt_conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{feature.prompt}",
                        "images": [feature.image_path],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                full_conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{feature.prompt}",
                        "images": [feature.image_path],
                    },
                    {"role": "<|Assistant|>", "content": feature.target_text},
                ]

                prompt_prepare = self.processor.process_one(
                    conversations=prompt_conversation,
                    images=[pil_image],
                )
                full_prepare = self.processor.process_one(
                    conversations=full_conversation,
                    images=[pil_image],
                )

            min_required_seq_len = int(prompt_prepare.num_image_tokens.sum().item()) + 32
            if self.max_seq_len <= min_required_seq_len:
                raise ValueError(
                    f"max_seq_len={self.max_seq_len} is too small for Janus image tokens; "
                    f"use a value greater than {min_required_seq_len}."
                )

            prompt_prepare = self._truncate_input_ids(prompt_prepare)
            full_prepare = self._truncate_input_ids(full_prepare)
            prompt_lengths.append(min(len(prompt_prepare.input_ids), len(full_prepare.input_ids)))
            full_prepares.append(full_prepare)
            request_ids.append(feature.request_id)
            references.append(feature.target_record)
            image_paths.append(feature.image_path)

        batch = self.processor.batchify(full_prepares)
        labels = batch.input_ids.clone()
        labels[batch.attention_mask == 0] = self.ignore_index
        labels[batch.images_seq_mask] = self.ignore_index

        seq_width = batch.input_ids.shape[1]
        for index, prepare in enumerate(full_prepares):
            seq_len = len(prepare.input_ids)
            prompt_len = prompt_lengths[index]
            left_pad = seq_width - seq_len
            labels[index, left_pad : left_pad + prompt_len] = self.ignore_index

        batch["labels"] = labels
        batch["request_ids"] = request_ids
        batch["references"] = references
        batch["image_paths"] = image_paths
        return batch


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
