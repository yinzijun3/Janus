"""Datasets and collators for JanusFlow-Art."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from finetune.janusflow_art_prompting import build_art_prompt, coerce_list, normalize_text


_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
MISSING_TOKEN = "__missing__"


def infer_subject_exclusion_hint(row: Dict[str, Any]) -> float:
    """Infer whether a sample should receive stronger portrait/head exclusion.

    The primary path uses manifest metadata produced by the official rebuild.
    When prompt-only evaluation records do not carry those explicit fields,
    fall back to a lightweight textual heuristic so inference-time behavior
    stays aligned with the training-time intent.
    """

    if bool(row.get("head_sensitive")):
        return 1.0
    if bool(row.get("style_portrait_dominant")):
        return 1.0
    try:
        if float(row.get("portrait_likelihood", 0.0)) >= 0.5:
            return 1.0
    except (TypeError, ValueError):
        pass
    heuristic_text = " ".join(
        str(value)
        for value in (
            row.get("request_id"),
            row.get("style_name"),
            row.get("composition_text"),
            row.get("rendered_prompt"),
            row.get("prompt"),
            row.get("description"),
        )
        if value
    ).lower()
    keyword_hits = [
        "portrait",
        "baroque",
        "face",
        "head",
        "bust",
        "half-length",
        "close-up",
        "single figure",
        "self-portrait",
        "woman",
        "man",
        "girl",
        "boy",
        "violinist",
        "scholar",
    ]
    return 1.0 if any(keyword in heuristic_text for keyword in keyword_hits) else 0.0


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load JSONL records from disk."""

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_optional_image_path(base_dir: str, value: Any) -> Optional[str]:
    """Resolve an optional image path against a base directory."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if os.path.isabs(text):
        return text
    return os.path.join(base_dir, text)


def existing_image_path_or_none(value: Optional[str]) -> Optional[str]:
    """Return the path only when it exists on disk."""

    if not value:
        return None
    return value if os.path.exists(value) else None


@dataclass
class ParsedArtRecord:
    """Canonical in-memory representation of one JanusFlow-Art sample."""

    request_id: str
    image_path: Optional[str]
    prompt: str
    style_label: str
    period_label: str
    medium_label: str
    reference_style_image: Optional[str]
    brushstroke_descriptors: List[str]
    texture_descriptors: List[str]
    composition_hints: List[str]
    texture_metadata: Dict[str, Any]
    raw_record: Dict[str, Any]


def parse_art_record(row: Dict[str, Any], *, base_dir: str = "") -> ParsedArtRecord:
    """Parse a raw manifest row into the unified JanusFlow-Art schema."""

    request_id = normalize_text(
        row.get("request_id") or row.get("id") or row.get("record_id")
    ) or "unknown_record"
    style_label = normalize_text(row.get("style_label") or row.get("style_name"))
    period_label = normalize_text(row.get("period_label"))
    medium_candidates = (
        [normalize_text(row.get("medium_label"))]
        + coerce_list(row.get("medium_tags"))
        + coerce_list(row.get("medium"))
    )
    medium_label = next((item for item in medium_candidates if item), "")
    texture_metadata = row.get("texture_metadata") or {}
    brushstroke_descriptors = coerce_list(
        row.get("brushstroke_descriptors")
        or row.get("brushstroke_text")
        or texture_metadata.get("brushstroke")
    )
    texture_descriptors = coerce_list(
        row.get("texture_descriptors")
        or row.get("texture_tags")
        or texture_metadata.get("texture")
        or texture_metadata.get("surface")
    )
    composition_hints = coerce_list(
        row.get("composition_hints")
        or texture_metadata.get("composition")
    )
    normalized_row = dict(row)
    normalized_row["style_label"] = style_label
    normalized_row["period_label"] = period_label
    normalized_row["medium_label"] = medium_label
    normalized_row["brushstroke_descriptors"] = brushstroke_descriptors
    normalized_row["texture_descriptors"] = texture_descriptors
    normalized_row["composition_hints"] = composition_hints
    explicit_rendered_prompt = normalize_text(
        row.get("rendered_prompt") or row.get("prompt_rendered")
    )
    prompt = build_art_prompt(normalized_row, template_name="conservative")
    if explicit_rendered_prompt:
        prompt = explicit_rendered_prompt
        normalized_row["rendered_prompt"] = explicit_rendered_prompt
    elif normalize_text(row.get("prompt")):
        prompt = build_art_prompt(
            {**normalized_row, "content": row.get("prompt")},
            template_name="conservative",
        )
    image_path = resolve_optional_image_path(base_dir, row.get("image_path"))
    reference_style_image = resolve_optional_image_path(
        base_dir,
        row.get("reference_style_image"),
    )
    return ParsedArtRecord(
        request_id=request_id,
        image_path=image_path,
        prompt=prompt,
        style_label=style_label,
        period_label=period_label,
        medium_label=medium_label,
        reference_style_image=existing_image_path_or_none(reference_style_image),
        brushstroke_descriptors=brushstroke_descriptors,
        texture_descriptors=texture_descriptors,
        composition_hints=composition_hints,
        texture_metadata=texture_metadata,
        raw_record=normalized_row,
    )


def build_label_vocabs(records: Iterable[ParsedArtRecord]) -> Dict[str, Dict[str, int]]:
    """Build string-to-index vocabularies for style metadata fields."""

    fields = {
        "style_label": [MISSING_TOKEN],
        "period_label": [MISSING_TOKEN],
        "medium_label": [MISSING_TOKEN],
    }
    for record in records:
        for field_name in fields:
            value = normalize_text(getattr(record, field_name))
            if value and value not in fields[field_name]:
                fields[field_name].append(value)
    return {
        field_name: {value: index for index, value in enumerate(values)}
        for field_name, values in fields.items()
    }


def encode_label_id(vocab: Dict[str, int], value: str) -> int:
    """Encode one label value into an integer ID."""

    normalized = normalize_text(value)
    if not normalized:
        return vocab[MISSING_TOKEN]
    return vocab.get(normalized, vocab[MISSING_TOKEN])


def preprocess_rgb_image(
    image: Image.Image,
    *,
    image_size: int,
    mode: str,
) -> torch.Tensor:
    """Preprocess an RGB image into a `[-1, 1]` tensor."""

    image = image.convert("RGB")
    if mode == "pad":
        background = tuple(int(channel) for channel in np.asarray(image).mean(axis=(0, 1)).tolist())
        image = ImageOps.pad(
            image,
            (image_size, image_size),
            method=_BICUBIC,
            color=background,
            centering=(0.5, 0.5),
        )
    else:
        image = ImageOps.fit(
            image,
            (image_size, image_size),
            method=_BICUBIC,
            centering=(0.5, 0.5),
        )
    image_np = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    # [H, W, 3] -> [3, H, W]
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous()


class ArtGenerationJsonlDataset(Dataset):
    """Image generation dataset backed by JSONL manifests."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = str(manifest_path)
        base_dir = str(Path(self.manifest_path).parent)
        parsed_records = [
            parse_art_record(row, base_dir=base_dir)
            for row in load_jsonl(self.manifest_path)
        ]
        self.missing_image_records: List[str] = []
        self.records: List[ParsedArtRecord] = []
        for record in parsed_records:
            if existing_image_path_or_none(record.image_path) is None:
                self.missing_image_records.append(record.request_id)
                continue
            self.records.append(
                ParsedArtRecord(
                    **{
                        **record.__dict__,
                        "image_path": existing_image_path_or_none(record.image_path),
                    }
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ParsedArtRecord:
        return self.records[index]


class ArtPromptJsonlDataset(Dataset):
    """Prompt-only dataset used by sampling and evaluation scripts."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = str(manifest_path)
        base_dir = str(Path(self.manifest_path).parent)
        self.records = [
            parse_art_record(row, base_dir=base_dir)
            for row in load_jsonl(self.manifest_path)
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ParsedArtRecord:
        return self.records[index]


class JanusFlowArtDataCollator:
    """Collate parsed art records into JanusFlow-Art training batches."""

    def __init__(
        self,
        *,
        image_size: int,
        image_preprocess_mode: str,
        reference_image_size: int,
        prompt_template: str,
        label_vocabs: Dict[str, Dict[str, int]],
        use_target_as_reference_when_missing: bool = False,
    ):
        self.image_size = image_size
        self.image_preprocess_mode = image_preprocess_mode
        self.reference_image_size = reference_image_size
        self.prompt_template = prompt_template
        self.label_vocabs = label_vocabs
        self.use_target_as_reference_when_missing = use_target_as_reference_when_missing

    def _render_prompt(self, feature: ParsedArtRecord) -> str:
        explicit_rendered_prompt = normalize_text(
            feature.raw_record.get("rendered_prompt")
            or feature.raw_record.get("prompt_rendered")
        )
        if explicit_rendered_prompt:
            return explicit_rendered_prompt
        return build_art_prompt(feature.raw_record, template_name=self.prompt_template)

    def __call__(self, features: List[ParsedArtRecord]) -> Dict[str, Any]:
        target_images: List[torch.Tensor] = []
        reference_images: List[torch.Tensor] = []
        has_reference_flags: List[bool] = []
        rendered_prompts: List[str] = []
        request_ids: List[str] = []
        image_paths: List[str] = []
        reference_image_paths: List[Optional[str]] = []
        raw_records: List[Dict[str, Any]] = []
        subject_exclusion_hints: List[float] = []
        style_ids: List[int] = []
        period_ids: List[int] = []
        medium_ids: List[int] = []

        for feature in features:
            image_path = existing_image_path_or_none(feature.image_path)
            if not image_path:
                raise ValueError(f"Training record {feature.request_id} is missing image_path.")
            with Image.open(image_path) as image:
                target_images.append(
                    preprocess_rgb_image(
                        image,
                        image_size=self.image_size,
                        mode=self.image_preprocess_mode,
                    )
                )
            reference_style_image = existing_image_path_or_none(feature.reference_style_image)
            if reference_style_image:
                with Image.open(reference_style_image) as image:
                    reference_images.append(
                        preprocess_rgb_image(
                            image,
                            image_size=self.reference_image_size,
                            mode="crop",
                        )
                    )
                has_reference_flags.append(True)
                reference_image_paths.append(reference_style_image)
            elif self.use_target_as_reference_when_missing:
                with Image.open(image_path) as image:
                    reference_images.append(
                        preprocess_rgb_image(
                            image,
                            image_size=self.reference_image_size,
                            mode="crop",
                        )
                    )
                has_reference_flags.append(True)
                reference_image_paths.append(image_path)
            else:
                reference_images.append(torch.zeros((3, self.reference_image_size, self.reference_image_size)))
                has_reference_flags.append(False)
                reference_image_paths.append(None)

            rendered_prompts.append(self._render_prompt(feature))
            request_ids.append(feature.request_id)
            image_paths.append(image_path)
            raw_records.append(feature.raw_record)
            subject_exclusion_hints.append(infer_subject_exclusion_hint(feature.raw_record))
            style_ids.append(encode_label_id(self.label_vocabs["style_label"], feature.style_label))
            period_ids.append(encode_label_id(self.label_vocabs["period_label"], feature.period_label))
            medium_ids.append(encode_label_id(self.label_vocabs["medium_label"], feature.medium_label))

        return {
            "target_images": torch.stack(target_images, dim=0),
            "reference_style_images": torch.stack(reference_images, dim=0),
            "has_reference_style_image": torch.tensor(has_reference_flags, dtype=torch.bool),
            "rendered_prompts": rendered_prompts,
            "request_ids": request_ids,
            "image_paths": image_paths,
            "reference_image_paths": reference_image_paths,
            "records": raw_records,
            "subject_exclusion_hints": torch.tensor(subject_exclusion_hints, dtype=torch.float32),
            "style_label_ids": torch.tensor(style_ids, dtype=torch.long),
            "period_label_ids": torch.tensor(period_ids, dtype=torch.long),
            "medium_label_ids": torch.tensor(medium_ids, dtype=torch.long),
        }


class JanusFlowArtPromptCollator:
    """Collator used for prompt-only sampling and evaluation batches."""

    def __init__(
        self,
        *,
        prompt_template: str,
        reference_image_size: int,
        label_vocabs: Dict[str, Dict[str, int]],
    ):
        self.prompt_template = prompt_template
        self.reference_image_size = reference_image_size
        self.label_vocabs = label_vocabs

    def _render_prompt(self, feature: ParsedArtRecord) -> str:
        explicit_rendered_prompt = normalize_text(
            feature.raw_record.get("rendered_prompt")
            or feature.raw_record.get("prompt_rendered")
        )
        if explicit_rendered_prompt:
            return explicit_rendered_prompt
        return build_art_prompt(feature.raw_record, template_name=self.prompt_template)

    def __call__(self, features: List[ParsedArtRecord]) -> Dict[str, Any]:
        reference_images: List[torch.Tensor] = []
        has_reference_flags: List[bool] = []
        reference_image_paths: List[Optional[str]] = []
        for feature in features:
            reference_style_image = existing_image_path_or_none(feature.reference_style_image)
            if reference_style_image:
                with Image.open(reference_style_image) as image:
                    reference_images.append(
                        preprocess_rgb_image(
                            image,
                            image_size=self.reference_image_size,
                            mode="crop",
                        )
                    )
                has_reference_flags.append(True)
                reference_image_paths.append(reference_style_image)
            else:
                reference_images.append(torch.zeros((3, self.reference_image_size, self.reference_image_size)))
                has_reference_flags.append(False)
                reference_image_paths.append(None)

        return {
            "rendered_prompts": [
                self._render_prompt(feature)
                for feature in features
            ],
            "request_ids": [feature.request_id for feature in features],
            "image_paths": [feature.image_path for feature in features],
            "reference_image_paths": reference_image_paths,
            "records": [feature.raw_record for feature in features],
            "subject_exclusion_hints": torch.tensor(
                [infer_subject_exclusion_hint(feature.raw_record) for feature in features],
                dtype=torch.float32,
            ),
            "reference_style_images": torch.stack(reference_images, dim=0),
            "has_reference_style_image": torch.tensor(has_reference_flags, dtype=torch.bool),
            "style_label_ids": torch.tensor(
                [
                    encode_label_id(self.label_vocabs["style_label"], feature.style_label)
                    for feature in features
                ],
                dtype=torch.long,
            ),
            "period_label_ids": torch.tensor(
                [
                    encode_label_id(self.label_vocabs["period_label"], feature.period_label)
                    for feature in features
                ],
                dtype=torch.long,
            ),
            "medium_label_ids": torch.tensor(
                [
                    encode_label_id(self.label_vocabs["medium_label"], feature.medium_label)
                    for feature in features
                ],
                dtype=torch.long,
            ),
        }
