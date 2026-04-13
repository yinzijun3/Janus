"""Rebuild EmoArt manifests from the complete official annotation source.

This script creates a richer master manifest plus balanced train/val and
continuation subsets tuned for the current JanusFlow brushstroke research
direction: stronger visible texture without sacrificing portrait/head
structure.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from analyze_emoart_data_risks import analyze_image
from finetune.emoart_generation import (
    extract_prompt_metadata,
    extract_style_name,
    resolve_image_path,
    safe_get,
)


PORTRAIT_KEYWORDS = (
    "portrait",
    "face",
    "head",
    "bust",
    "self-portrait",
    "figure",
    "figures",
    "half-length",
    "close-up",
    "single figure",
    "woman",
    "man",
    "girl",
    "boy",
    "person",
)
CENTRAL_SUBJECT_KEYWORDS = (
    "central focus",
    "centrally focused",
    "centered on",
    "centering on",
    "centers the",
    "dominant figure",
    "single figure",
    "bust",
    "half-length",
    "close-up",
    "frontal",
)
TEXTURE_STRONG_TAGS = {
    "visible brushstrokes",
    "layered paint texture",
    "dense dotted pigment",
    "micro mark clusters",
    "irregular painterly edges",
    "material brushwork depth",
    "gestural surface energy",
    "thick pigment build-up",
    "paint ridges",
}
TEXTURE_MEDIUM_TAGS = {
    "subtle paint surface variation",
    "fine-grained surface detail",
    "micro texture variation",
    "paint surface texture",
}
MEDIUM_TEXTURE_TAGS = {
    "textured paint handling",
    "visible pigment variation",
    "textured paint surface",
    "layered surface depth",
    "painted texture",
    "canvas texture",
    "paper grain",
    "printed texture",
}
TEXTURE_HIGH_DETAIL_PHRASES = (
    "rough",
    "textured",
    "stipple",
    "dense",
    "dotted",
    "irregular",
    "material",
    "layered",
    "organic edge",
    "broken",
    "energetic",
    "gestural",
    "thick",
    "impasto",
)
TEXTURE_LOW_DETAIL_PHRASES = (
    "fine and delicate",
    "fine, delicate",
    "smooth and defined",
    "smooth and refined",
    "soft and smooth",
    "thin and smooth",
    "fine and precise",
    "smooth and even",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild balanced EmoArt manifests from annotation.official.json."
    )
    parser.add_argument(
        "--annotation-path",
        default="/root/autodl-tmp/data/emoart_5k/annotation.official.json",
    )
    parser.add_argument(
        "--images-root",
        default="/root/autodl-tmp/data/emoart_5k/Images",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/data/emoart_5k/gen_full_official_rebuild_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-per-style", type=int, default=10)
    parser.add_argument("--texture-per-style", type=int, default=28)
    parser.add_argument("--portrait-per-style", type=int, default=20)
    parser.add_argument("--max-frame-risk", type=float, default=0.55)
    parser.add_argument("--border-ratio", type=float, default=0.08)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split()).strip(" ,.;")


def clean_clause(text: str) -> str:
    return normalize_text(text).rstrip(".")


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        cleaned = normalize_text(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def contains_any(text: str, keywords: Sequence[str]) -> bool:
    lowered = normalize_text(text).lower()
    return any(keyword in lowered for keyword in keywords)


def truncate_words(text: str, max_words: int) -> str:
    words = normalize_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).strip(" ,.;")


def stringify_two(values: Sequence[str]) -> str:
    picked = dedupe_keep_order(values)[:2]
    return ", ".join(picked)


def image_high_frequency_score(image_path: Path) -> Dict[str, float]:
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    gradients = dx + dy
    # Whole-image scalar used later as one component of texture scoring.
    grad_mean = float(gradients.mean())
    grad_std = float(gradients.std())
    return {
        "image_hf_grad_mean": grad_mean,
        "image_hf_grad_std": grad_std,
        "image_hf_raw": grad_mean + 0.25 * grad_std,
    }


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def score_texturedness(
    *,
    brushstroke_text: str,
    line_quality_text: str,
    texture_tags: Sequence[str],
    medium_tags: Sequence[str],
) -> float:
    score = 0.0
    texture_tags_lower = {normalize_text(tag).lower() for tag in texture_tags if normalize_text(tag)}
    medium_tags_lower = {normalize_text(tag).lower() for tag in medium_tags if normalize_text(tag)}
    brushstroke_lower = normalize_text(brushstroke_text).lower()
    line_lower = normalize_text(line_quality_text).lower()

    score += 0.9 * len(texture_tags_lower & TEXTURE_STRONG_TAGS)
    score += 0.45 * len(texture_tags_lower & TEXTURE_MEDIUM_TAGS)
    score += 0.65 * len(medium_tags_lower & MEDIUM_TEXTURE_TAGS)

    if contains_any(brushstroke_lower, TEXTURE_HIGH_DETAIL_PHRASES):
        score += 1.1
    if contains_any(line_lower, TEXTURE_HIGH_DETAIL_PHRASES):
        score += 0.55
    if contains_any(brushstroke_lower, TEXTURE_LOW_DETAIL_PHRASES):
        score -= 0.8
    if contains_any(line_lower, TEXTURE_LOW_DETAIL_PHRASES):
        score -= 0.4
    return score


def build_content_safe_prompt(record: Dict[str, Any]) -> str:
    clauses = []
    style_name = normalize_text(record.get("style_name"))
    content_text = truncate_words(record.get("content_text", ""), 28)
    composition_text = truncate_words(record.get("composition_text", ""), 14)
    medium_clause = stringify_two(record.get("medium_tags", []))
    if style_name:
        clauses.append(f"{style_name} artwork")
    else:
        clauses.append("artwork")
    if content_text:
        clauses.append(content_text)
    if medium_clause:
        clauses.append(f"material: {medium_clause}")
    if composition_text:
        clauses.append(f"composition cues: {composition_text}")
    return ". ".join(clean_clause(clause) for clause in clauses if clean_clause(clause)) + "."


def build_texture_push_prompt(record: Dict[str, Any]) -> str:
    base_prompt = build_content_safe_prompt(record).rstrip(" .")
    clauses = [base_prompt]
    brushstroke_text = truncate_words(record.get("brushstroke_text", ""), 12)
    texture_clause = stringify_two(record.get("texture_tags", []))
    if brushstroke_text:
        clauses.append(f"brushwork cue: {brushstroke_text}")
    if texture_clause:
        clauses.append(f"surface detail: {texture_clause}")
    return ". ".join(clean_clause(clause) for clause in clauses if clean_clause(clause)) + "."


def build_portrait_safe_prompt(record: Dict[str, Any]) -> str:
    base_prompt = build_content_safe_prompt(record).rstrip(" .")
    clauses = [base_prompt]
    restrained_clause = truncate_words(record.get("brushstroke_text", ""), 10)
    if restrained_clause:
        clauses.append(f"brushwork cue: {restrained_clause}")
    else:
        medium_clause = stringify_two(record.get("medium_tags", []))
        if medium_clause:
            clauses.append(f"material cue: {medium_clause}")
    clauses.append("stable facial structure, coherent anatomy, preserved composition")
    return ". ".join(clean_clause(clause) for clause in clauses if clean_clause(clause)) + "."


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_quality_flags(record: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    if not normalize_text(record.get("brushstroke_text")):
        flags.append("missing_brushstroke_text")
    if not normalize_text(record.get("line_quality_text")):
        flags.append("missing_line_quality_text")
    if not normalize_text(record.get("composition_text")):
        flags.append("missing_composition_text")
    if float(record.get("frame_risk_score", 0.0)) > 0.55:
        flags.append("high_frame_risk")
    if float(record.get("portrait_likelihood", 0.0)) > 0.0:
        flags.append("portrait_candidate")
    return flags


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def choose_rendered_prompt(row: Dict[str, Any], prompt_key: str) -> Dict[str, Any]:
    updated = dict(row)
    updated["prompt"] = updated[prompt_key]
    updated["rendered_prompt"] = updated[prompt_key]
    updated["prompt_variant"] = prompt_key
    return updated


def split_train_val(
    records: List[Dict[str, Any]],
    *,
    seed: int,
    val_per_style: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["style_name"]].append(row)

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    for style_name, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda row: row["request_id"])
        rng = random.Random(f"{seed}:{style_name}")
        rng.shuffle(ordered)
        val_split = [choose_rendered_prompt(row, "prompt_v2_content_safe") for row in ordered[:val_per_style]]
        train_split = [choose_rendered_prompt(row, "prompt_v2_content_safe") for row in ordered[val_per_style:]]
        val_rows.extend(val_split)
        train_rows.extend(train_split)
    return train_rows, val_rows


def bucket_select(
    source_rows: List[Dict[str, Any]],
    *,
    desired_count: int,
    key_fn,
    selected_ids: set[str],
) -> List[Dict[str, Any]]:
    picked: List[Dict[str, Any]] = []
    for row in sorted(source_rows, key=key_fn):
        if row["request_id"] in selected_ids:
            continue
        picked.append(row)
        selected_ids.add(row["request_id"])
        if len(picked) >= desired_count:
            break
    return picked


def build_texture_subset(
    train_rows: List[Dict[str, Any]],
    *,
    max_frame_risk: float,
    per_style: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        grouped[row["style_name"]].append(row)

    output_rows: List[Dict[str, Any]] = []
    for style_name, items in sorted(grouped.items()):
        filtered = [row for row in items if float(row["frame_risk_score"]) <= max_frame_risk]
        texture_scores = [float(row["texture_score"]) for row in filtered]
        q40 = quantile(texture_scores, 0.40)
        q80 = quantile(texture_scores, 0.80)
        selected_ids: set[str] = set()
        style_output_rows: List[Dict[str, Any]] = []

        high_texture_head = [
            row for row in filtered if bool(row["head_sensitive"])
        ]
        high_texture_non_head = [
            row for row in filtered if not bool(row["head_sensitive"])
        ]
        anchor_rows = [
            row for row in filtered
            if q40 <= float(row["texture_score"]) <= q80
        ]

        style_is_portrait_dominant = bool(items[0]["style_portrait_dominant"])
        if style_is_portrait_dominant:
            style_output_rows.extend(
                choose_rendered_prompt(row, "prompt_v2_portrait_safe")
                for row in bucket_select(
                    high_texture_head,
                    desired_count=12,
                    key_fn=lambda row: (-float(row["texture_score"]), float(row["frame_risk_score"]), row["request_id"]),
                    selected_ids=selected_ids,
                )
            )
            style_output_rows.extend(
                choose_rendered_prompt(row, "prompt_v2_texture_push")
                for row in bucket_select(
                    high_texture_non_head,
                    desired_count=8,
                    key_fn=lambda row: (-float(row["texture_score"]), float(row["frame_risk_score"]), row["request_id"]),
                    selected_ids=selected_ids,
                )
            )
        else:
            style_output_rows.extend(
                choose_rendered_prompt(row, "prompt_v2_texture_push")
                for row in bucket_select(
                    high_texture_non_head,
                    desired_count=12,
                    key_fn=lambda row: (-float(row["texture_score"]), float(row["frame_risk_score"]), row["request_id"]),
                    selected_ids=selected_ids,
                )
            )
            style_output_rows.extend(
                choose_rendered_prompt(row, "prompt_v2_portrait_safe")
                for row in bucket_select(
                    high_texture_head,
                    desired_count=8,
                    key_fn=lambda row: (-float(row["texture_score"]), float(row["frame_risk_score"]), row["request_id"]),
                    selected_ids=selected_ids,
                )
            )

        anchor_selected = bucket_select(
            anchor_rows,
            desired_count=8,
            key_fn=lambda row: (
                abs(float(row["texture_score"]) - quantile(texture_scores, 0.50)),
                float(row["frame_risk_score"]),
                row["request_id"],
            ),
            selected_ids=selected_ids,
        )
        style_output_rows.extend(
            choose_rendered_prompt(
                row,
                "prompt_v2_portrait_safe" if bool(row["head_sensitive"]) else "prompt_v2_texture_push",
            )
            for row in anchor_selected
        )

        current_style_count = len(style_output_rows)
        if current_style_count < per_style:
            fallback_rows = bucket_select(
                filtered,
                desired_count=per_style - current_style_count,
                key_fn=lambda row: (-float(row["texture_score"]), float(row["frame_risk_score"]), row["request_id"]),
                selected_ids=selected_ids,
            )
            style_output_rows.extend(
                choose_rendered_prompt(
                    row,
                    "prompt_v2_portrait_safe" if bool(row["head_sensitive"]) else "prompt_v2_texture_push",
                )
                for row in fallback_rows
            )
        if len(style_output_rows) < per_style and style_output_rows:
            repeated_rows = list(style_output_rows)
            while len(style_output_rows) < per_style:
                template = repeated_rows[(len(style_output_rows) - len(repeated_rows)) % len(repeated_rows)]
                repeated = dict(template)
                repeated["repeat_index"] = int(repeated.get("repeat_index", 0)) + 1
                style_output_rows.append(repeated)
        output_rows.extend(style_output_rows[:per_style])
    return output_rows


def build_portrait_subset(
    train_rows: List[Dict[str, Any]],
    *,
    max_frame_risk: float,
    per_style: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        if bool(row["style_portrait_dominant"]):
            grouped[row["style_name"]].append(row)

    output_rows: List[Dict[str, Any]] = []
    for style_name, items in sorted(grouped.items()):
        filtered = [
            row for row in items
            if bool(row["head_sensitive"]) and float(row["frame_risk_score"]) <= max_frame_risk
        ]
        texture_scores = [float(row["texture_score"]) for row in filtered]
        q40 = quantile(texture_scores, 0.40)
        q80 = quantile(texture_scores, 0.80)
        selected = sorted(
            filtered,
            key=lambda row: (
                -float(row["structure_risk_score"]),
                0 if q40 <= float(row["texture_score"]) <= q80 else 1,
                float(row["frame_risk_score"]),
                -float(row["texture_score"]),
                row["request_id"],
            ),
        )[:per_style]
        output_rows.extend(choose_rendered_prompt(row, "prompt_v2_portrait_safe") for row in selected)
    return output_rows


def summarize_styles(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["style_name"]].append(row)
    style_rows = []
    for style_name, items in sorted(grouped.items()):
        portrait_ratio = sum(1 for row in items if float(row["portrait_likelihood"]) > 0.0) / max(len(items), 1)
        head_sensitive_ratio = sum(1 for row in items if bool(row["head_sensitive"])) / max(len(items), 1)
        style_rows.append(
            {
                "style_name": style_name,
                "count": len(items),
                "portrait_ratio": round(portrait_ratio, 4),
                "head_sensitive_ratio": round(head_sensitive_ratio, 4),
                "avg_texture_score": round(
                    sum(float(row["texture_score"]) for row in items) / max(len(items), 1),
                    4,
                ),
                "avg_frame_risk_score": round(
                    sum(float(row["frame_risk_score"]) for row in items) / max(len(items), 1),
                    4,
                ),
                "style_portrait_dominant": bool(items[0]["style_portrait_dominant"]),
            }
        )
    return {
        "num_styles": len(style_rows),
        "styles": style_rows,
    }


def summarize_subjects(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "num_records": len(records),
        "portrait_candidates": sum(1 for row in records if float(row["portrait_likelihood"]) > 0.0),
        "head_sensitive_records": sum(1 for row in records if bool(row["head_sensitive"])),
        "portrait_dominant_styles": sorted(
            {row["style_name"] for row in records if bool(row["style_portrait_dominant"])}
        ),
        "prompt_variant_counts": Counter(row.get("prompt_variant", "none") for row in records),
    }


def build_data_card(
    *,
    valid_records: List[Dict[str, Any]],
    bad_records: List[Dict[str, Any]],
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    texture_rows: List[Dict[str, Any]],
    portrait_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> str:
    style_counter = Counter(row["style_name"] for row in valid_records)
    portrait_dominant_styles = sorted(
        {row["style_name"] for row in valid_records if bool(row["style_portrait_dominant"])}
    )
    lines = [
        "# EmoArt Full Rebuild v1",
        "",
        "## Summary",
        f"- output_dir: `{output_dir}`",
        f"- master_manifest_count: `{len(valid_records)}`",
        f"- bad_record_count: `{len(bad_records)}`",
        f"- train_full_balanced_v1_count: `{len(train_rows)}`",
        f"- val_balanced_v1_count: `{len(val_rows)}`",
        f"- train_texture_balanced_v2_count: `{len(texture_rows)}`",
        f"- train_portrait_structure_v1_count: `{len(portrait_rows)}`",
        "",
        "## Method",
        "- Source of truth is `annotation.official.json`; damaged legacy `annotation.json` is ignored.",
        "- Style labels are extracted from `request_id` and no longer inherited from older intermediate manifests.",
        "- `texture_score` is built from textural metadata and image high-frequency statistics only; no style bonus is used.",
        "- `head_sensitive` is derived from portrait and central-subject heuristics so continuation subsets do not over-prune portrait-safe anchor cases.",
        "",
        "## Top-Level Counts",
    ]
    for style_name, count in style_counter.most_common(12):
        lines.append(f"- `{style_name}`: `{count}`")
    lines.extend(
        [
            "",
            "## Portrait-Dominant Styles",
            "- " + ", ".join(portrait_dominant_styles),
            "",
            "## Prompt Policy",
            "- `prompt_v2_content_safe`: main balanced training",
            "- `prompt_v2_texture_push`: non-head-sensitive texture continuation",
            "- `prompt_v2_portrait_safe`: head-sensitive and portrait-structure continuation",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(Path(args.annotation_path))
    valid_records: List[Dict[str, Any]] = []
    bad_records: List[Dict[str, Any]] = []
    image_hf_values: List[float] = []

    for row in annotations:
        request_id = normalize_text(row.get("request_id"))
        description = row.get("description") or {}
        image_path = Path(resolve_image_path(args.images_root, row.get("image_path", "")))
        if not image_path.exists():
            bad_records.append(
                {
                    "request_id": request_id,
                    "image_path": str(image_path),
                    "reason": "missing_image",
                }
            )
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, FileNotFoundError) as exc:
            bad_records.append(
                {
                    "request_id": request_id,
                    "image_path": str(image_path),
                    "reason": "bad_image",
                    "error": str(exc),
                }
            )
            continue

        prompt_metadata = extract_prompt_metadata(description, request_id=request_id)
        image_stats = analyze_image(image_path, border_ratio=args.border_ratio)
        hf_stats = image_high_frequency_score(image_path)

        style_name = normalize_text(extract_style_name(request_id))
        content_text = normalize_text(safe_get(description, "first_section", "description"))
        visual_attributes = safe_get(description, "second_section", "visual_attributes") or {}
        brushstroke_text = normalize_text(
            prompt_metadata.get("brushstroke_text") or visual_attributes.get("brushstroke")
        )
        line_quality_text = normalize_text(
            prompt_metadata.get("line_quality_text") or visual_attributes.get("line_quality")
        )
        composition_text = normalize_text(visual_attributes.get("composition"))
        color_text = normalize_text(visual_attributes.get("color"))
        emotional_impact = normalize_text(safe_get(description, "second_section", "emotional_impact"))
        texture_tags = dedupe_keep_order(prompt_metadata.get("texture_tags", []))
        medium_tags = dedupe_keep_order(prompt_metadata.get("medium_tags", []))
        portrait_likelihood = 1.0 if contains_any(
            " ".join(
                [
                    json.dumps(description, ensure_ascii=False),
                    content_text,
                    composition_text,
                    emotional_impact,
                    brushstroke_text,
                ]
            ),
            PORTRAIT_KEYWORDS,
        ) else 0.0
        central_subject_hint = 1.0 if contains_any(composition_text, CENTRAL_SUBJECT_KEYWORDS) else 0.0
        text_texture_score = score_texturedness(
            brushstroke_text=brushstroke_text,
            line_quality_text=line_quality_text,
            texture_tags=texture_tags,
            medium_tags=medium_tags,
        )
        image_hf_values.append(hf_stats["image_hf_raw"])

        valid_records.append(
            {
                "request_id": request_id,
                "image_path": str(image_path),
                "style_name": style_name,
                "description": description,
                "content_text": content_text,
                "brushstroke_text": brushstroke_text,
                "line_quality_text": line_quality_text,
                "composition_text": composition_text,
                "color_text": color_text,
                "emotional_impact": emotional_impact,
                "texture_tags": texture_tags,
                "medium_tags": medium_tags,
                "text_texture_score": round(text_texture_score, 4),
                "image_hf_grad_mean": round(hf_stats["image_hf_grad_mean"], 4),
                "image_hf_grad_std": round(hf_stats["image_hf_grad_std"], 4),
                "image_hf_raw": round(hf_stats["image_hf_raw"], 4),
                "frame_risk_score": round(float(image_stats["frame_risk_score"]), 4),
                "portrait_likelihood": portrait_likelihood,
                "central_subject_hint": central_subject_hint,
            }
        )

    hf_p05 = quantile(image_hf_values, 0.05)
    hf_p95 = quantile(image_hf_values, 0.95)
    style_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in valid_records:
        style_to_rows[row["style_name"]].append(row)
    style_portrait_ratio = {
        style_name: (
            sum(1 for row in items if float(row["portrait_likelihood"]) > 0.0) / max(len(items), 1)
        )
        for style_name, items in style_to_rows.items()
    }

    for row in valid_records:
        image_hf_norm = 0.0 if hf_p95 <= hf_p05 else clip01(
            (float(row["image_hf_raw"]) - hf_p05) / (hf_p95 - hf_p05)
        )
        style_is_portrait_dominant = style_portrait_ratio[row["style_name"]] >= 0.70
        structure_risk_score = (
            0.55 * float(row["portrait_likelihood"])
            + 0.25 * float(row["central_subject_hint"])
            + 0.20 * float(style_is_portrait_dominant)
        )
        texture_score = float(row["text_texture_score"]) + 1.5 * image_hf_norm
        row["image_hf_norm"] = round(image_hf_norm, 4)
        row["texture_score"] = round(texture_score, 4)
        row["style_portrait_dominant"] = style_is_portrait_dominant
        row["structure_risk_score"] = round(structure_risk_score, 4)
        row["head_sensitive"] = structure_risk_score >= 0.60
        row["prompt_v2_content_safe"] = build_content_safe_prompt(row)
        row["prompt_v2_texture_push"] = build_texture_push_prompt(row)
        row["prompt_v2_portrait_safe"] = build_portrait_safe_prompt(row)
        row["quality_flags"] = build_quality_flags(row)

    valid_records.sort(key=lambda row: row["request_id"])
    train_rows, val_rows = split_train_val(
        valid_records,
        seed=args.seed,
        val_per_style=args.val_per_style,
    )
    texture_rows = build_texture_subset(
        train_rows,
        max_frame_risk=args.max_frame_risk,
        per_style=args.texture_per_style,
    )
    portrait_rows = build_portrait_subset(
        train_rows,
        max_frame_risk=args.max_frame_risk,
        per_style=args.portrait_per_style,
    )

    master_path = output_dir / "master_manifest.jsonl"
    train_path = output_dir / "train_full_balanced_v1.jsonl"
    val_path = output_dir / "val_balanced_v1.jsonl"
    texture_path = output_dir / "train_texture_balanced_v2.jsonl"
    portrait_path = output_dir / "train_portrait_structure_v1.jsonl"
    bad_path = output_dir / "bad_samples.jsonl"
    style_summary_path = output_dir / "style_summary.json"
    subject_summary_path = output_dir / "subject_summary.json"
    data_card_path = output_dir / "data_card.md"

    write_jsonl(master_path, valid_records)
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(texture_path, texture_rows)
    write_jsonl(portrait_path, portrait_rows)
    write_jsonl(bad_path, bad_records)
    write_json(style_summary_path, summarize_styles(valid_records))
    write_json(
        subject_summary_path,
        {
            "master": summarize_subjects(valid_records),
            "train_full_balanced_v1": summarize_subjects(train_rows),
            "val_balanced_v1": summarize_subjects(val_rows),
            "train_texture_balanced_v2": summarize_subjects(texture_rows),
            "train_portrait_structure_v1": summarize_subjects(portrait_rows),
        },
    )
    data_card_path.write_text(
        build_data_card(
            valid_records=valid_records,
            bad_records=bad_records,
            train_rows=train_rows,
            val_rows=val_rows,
            texture_rows=texture_rows,
            portrait_rows=portrait_rows,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )

    summary = {
        "annotation_path": args.annotation_path,
        "images_root": args.images_root,
        "output_dir": str(output_dir),
        "num_annotations": len(annotations),
        "num_valid_records": len(valid_records),
        "num_bad_records": len(bad_records),
        "train_full_balanced_v1_count": len(train_rows),
        "val_balanced_v1_count": len(val_rows),
        "train_texture_balanced_v2_count": len(texture_rows),
        "train_portrait_structure_v1_count": len(portrait_rows),
        "num_styles": len({row["style_name"] for row in valid_records}),
        "portrait_dominant_style_count": len(
            {row["style_name"] for row in valid_records if bool(row["style_portrait_dominant"])}
        ),
        "hf_p05": round(hf_p05, 4),
        "hf_p95": round(hf_p95, 4),
        "paths": {
            "master_manifest": str(master_path),
            "train_full_balanced_v1": str(train_path),
            "val_balanced_v1": str(val_path),
            "train_texture_balanced_v2": str(texture_path),
            "train_portrait_structure_v1": str(portrait_path),
            "style_summary": str(style_summary_path),
            "subject_summary": str(subject_summary_path),
            "data_card": str(data_card_path),
            "bad_samples": str(bad_path),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
