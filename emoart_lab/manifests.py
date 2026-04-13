from pathlib import Path
from typing import Any, Dict, List, Optional

from emoart_lab.io_utils import load_jsonl, write_json, write_jsonl
from emoart_lab.layout import resolve_family_manifest_path, resolve_texture_manifest_path
from emoart_lab.schemas import ExpertConfig


def record_key(row: Dict[str, Any]) -> Optional[str]:
    for key in ("request_id", "id", "record_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def build_family_manifests(
    train_manifest: Path,
    val_manifest: Path,
    experts: Dict[str, ExpertConfig],
    output_dir: Path,
) -> Dict[str, Any]:
    style_to_expert: Dict[str, str] = {}
    for expert_name, expert_config in experts.items():
        for style_name in expert_config.styles:
            style_to_expert[style_name] = expert_name

    summary: Dict[str, Any] = {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "style_to_expert": style_to_expert,
        "experts": {},
        "ignored_style_counts": {"train": {}, "val": {}},
    }
    for expert_name, expert_config in experts.items():
        summary["experts"][expert_name] = {
            "styles": expert_config.styles,
            "description": expert_config.description,
            "train_count": 0,
            "val_count": 0,
            "train_style_counts": {},
            "val_style_counts": {},
            "paths": {},
        }

    for split_name, manifest_path in (("train", train_manifest), ("val", val_manifest)):
        rows = load_jsonl(manifest_path)
        buckets: Dict[str, List[Dict[str, Any]]] = {expert_name: [] for expert_name in experts}
        ignored_counts: Dict[str, int] = {}
        style_counts: Dict[str, Dict[str, int]] = {expert_name: {} for expert_name in experts}
        for row in rows:
            style_name = row.get("style_name", "UNKNOWN")
            expert_name = style_to_expert.get(style_name)
            if expert_name is None:
                ignored_counts[style_name] = ignored_counts.get(style_name, 0) + 1
                continue
            tagged = dict(row)
            tagged["style_expert"] = expert_name
            buckets[expert_name].append(tagged)
            style_counts[expert_name][style_name] = style_counts[expert_name].get(style_name, 0) + 1

        for expert_name, expert_rows in buckets.items():
            path = output_dir / expert_name / f"{split_name}.jsonl"
            write_jsonl(path, expert_rows)
            summary["experts"][expert_name][f"{split_name}_count"] = len(expert_rows)
            summary["experts"][expert_name][f"{split_name}_style_counts"] = dict(
                sorted(style_counts[expert_name].items())
            )
            summary["experts"][expert_name]["paths"][split_name] = str(path)
        summary["ignored_style_counts"][split_name] = dict(sorted(ignored_counts.items()))

    write_json(output_dir / "family_summary.json", summary)
    return summary


def build_texture_refine_manifests(
    experts: Dict[str, ExpertConfig],
    track_dir: Path,
    texture_rich_manifest: Path,
) -> Dict[str, Any]:
    texture_rows = load_jsonl(texture_rich_manifest)
    texture_index: Dict[str, Dict[str, Any]] = {}
    for row in texture_rows:
        key = record_key(row)
        if key and key not in texture_index:
            texture_index[key] = row

    summary: Dict[str, Any] = {
        "texture_rich_manifest": str(texture_rich_manifest),
        "texture_rich_count": len(texture_rows),
        "experts": {},
    }
    for expert_name in experts:
        family_path = resolve_family_manifest_path(track_dir, expert_name, "train")
        family_rows = load_jsonl(family_path)
        kept_rows: List[Dict[str, Any]] = []
        missing_record_key_count = 0
        for row in family_rows:
            key = record_key(row)
            if key is None:
                missing_record_key_count += 1
                continue
            texture_row = texture_index.get(key)
            if texture_row is None:
                continue
            merged = dict(row)
            for merge_key, merge_value in texture_row.items():
                if merge_key not in merged:
                    merged[merge_key] = merge_value
            kept_rows.append(merged)

        output_manifest = resolve_texture_manifest_path(track_dir, expert_name)
        write_jsonl(output_manifest, kept_rows)
        summary["experts"][expert_name] = {
            "family_train_manifest": str(family_path),
            "output_manifest": str(output_manifest),
            "family_train_count": len(family_rows),
            "texture_refine_count": len(kept_rows),
            "missing_record_key_count": missing_record_key_count,
        }

    write_json(track_dir / "data" / "texture_refine" / "texture_refine_summary.json", summary)
    return summary

