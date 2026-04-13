"""Prompt parsing and slot-based prompt construction for JanusFlow-Art."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


def normalize_text(value: Any) -> str:
    """Normalize arbitrary values into a compact string."""

    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split()).strip(" ,.;")


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    """Deduplicate a sequence while preserving the first occurrence order."""

    seen = set()
    result: List[str] = []
    for item in items:
        normalized = normalize_text(item)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def coerce_list(value: Any) -> List[str]:
    """Convert a potentially scalar or nested value into a flat list of strings."""

    if value is None:
        return []
    if isinstance(value, list):
        return dedupe_keep_order(str(item) for item in value)
    if isinstance(value, tuple):
        return dedupe_keep_order(str(item) for item in value)
    if isinstance(value, dict):
        items: List[str] = []
        for inner_value in value.values():
            items.extend(coerce_list(inner_value))
        return dedupe_keep_order(items)
    text = normalize_text(value)
    return [text] if text else []


def safe_get(data: Dict[str, Any], *keys: str) -> Any:
    """Safely resolve a nested dictionary path."""

    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


@dataclass
class PromptSlots:
    """Named prompt slots used by JanusFlow-Art templates."""

    content: str = ""
    style: str = ""
    period: str = ""
    medium: str = ""
    brushstroke: List[str] = field(default_factory=list)
    texture: List[str] = field(default_factory=list)
    composition: List[str] = field(default_factory=list)


def build_prompt_slots(record: Dict[str, Any]) -> PromptSlots:
    """Construct prompt slots from a raw art record."""

    description = record.get("description") or {}
    texture_metadata = record.get("texture_metadata") or {}

    content = normalize_text(
        record.get("content")
        or record.get("prompt")
        or safe_get(description, "first_section", "description")
    )
    style = normalize_text(record.get("style_label") or record.get("style_name"))
    period = normalize_text(record.get("period_label"))
    medium = normalize_text(
        record.get("medium_label")
        or record.get("medium")
        or safe_get(texture_metadata, "medium")
    )

    brushstroke = dedupe_keep_order(
        coerce_list(record.get("brushstroke_descriptors"))
        + coerce_list(record.get("brushstroke_text"))
        + coerce_list(safe_get(texture_metadata, "brushstroke"))
        + coerce_list(safe_get(description, "second_section", "visual_attributes", "brushstroke"))
    )
    texture = dedupe_keep_order(
        coerce_list(record.get("texture_descriptors"))
        + coerce_list(record.get("texture_tags"))
        + coerce_list(safe_get(texture_metadata, "texture"))
        + coerce_list(safe_get(texture_metadata, "surface"))
    )
    composition = dedupe_keep_order(
        coerce_list(record.get("composition_hints"))
        + coerce_list(safe_get(texture_metadata, "composition"))
        + coerce_list(safe_get(description, "second_section", "visual_attributes", "composition"))
    )
    return PromptSlots(
        content=content,
        style=style,
        period=period,
        medium=medium,
        brushstroke=brushstroke,
        texture=texture,
        composition=composition,
    )


def build_art_prompt(record: Dict[str, Any], template_name: str = "conservative") -> str:
    """Render the final generation prompt from parsed slots."""

    slots = build_prompt_slots(record)
    if template_name == "strong_style":
        parts: List[str] = []
        if slots.style:
            parts.append(f"{slots.style} artwork")
        if slots.period:
            parts.append(f"period: {slots.period}")
        if slots.medium:
            parts.append(f"medium: {slots.medium}")
        if slots.content:
            parts.append(slots.content)
        if slots.brushstroke:
            parts.append("brushwork: " + ", ".join(slots.brushstroke))
        if slots.texture:
            parts.append("surface detail: " + ", ".join(slots.texture))
        if slots.composition:
            parts.append("composition: " + ", ".join(slots.composition))
        parts.append(
            "prioritize tactile material presence, visible artistic handling, rich local texture, and cohesive overall composition"
        )
        return ". ".join(part.strip(" .") for part in parts if part.strip()) + "."

    parts = []
    if slots.style:
        parts.append(f"{slots.style} artwork")
    elif slots.medium:
        parts.append(f"{slots.medium} artwork")
    else:
        parts.append("artwork")
    if slots.content:
        parts.append(slots.content)
    if slots.period:
        parts.append(f"period influence: {slots.period}")
    if slots.medium:
        parts.append(f"material: {slots.medium}")
    if slots.brushstroke:
        parts.append("brushwork cues: " + ", ".join(slots.brushstroke[:3]))
    if slots.texture:
        parts.append("texture cues: " + ", ".join(slots.texture[:3]))
    if slots.composition:
        parts.append("composition cues: " + ", ".join(slots.composition[:3]))
    return ". ".join(part.strip(" .") for part in parts if part.strip()) + "."


def build_style_proxy_prompt(record: Dict[str, Any]) -> str:
    """Build a style-only prompt used for style-consistency proxy evaluation."""

    slots = build_prompt_slots(record)
    parts = []
    if slots.style:
        parts.append(slots.style)
    if slots.period:
        parts.append(slots.period)
    if slots.medium:
        parts.append(slots.medium)
    if slots.brushstroke:
        parts.append(", ".join(slots.brushstroke[:2]))
    if slots.texture:
        parts.append(", ".join(slots.texture[:2]))
    if not parts:
        return "artwork"
    return ", ".join(part for part in parts if part)
