"""Evaluation entrypoint for JanusFlow-Art."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from finetune.janusflow_art_config import apply_common_cli_overrides, load_yaml_config
from finetune.janusflow_art_data import ArtPromptJsonlDataset, JanusFlowArtPromptCollator
from finetune.janusflow_art_prompting import build_style_proxy_prompt
from finetune.janusflow_art_runtime import (
    JanusFlowArtPipeline,
    save_image_grid,
    save_json,
    save_jsonl,
    tensor_to_pil_images,
)


def _disable_enabled_flags(node):
    """Recursively set `enabled: false` on nested config mappings."""

    if isinstance(node, dict):
        if "enabled" in node:
            node["enabled"] = False
        for value in node.values():
            _disable_enabled_flags(value)
    elif isinstance(node, list):
        for value in node:
            _disable_enabled_flags(value)


def _sha256_file(path: str) -> str:
    """Return a file's SHA256 hash."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate JanusFlow-Art checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--lora-scale", type=float, default=None)
    return parser.parse_args()


def maybe_load_clip_bundle(model_name: str):
    """Load CLIP only when the optional dependency stack is available."""

    try:
        from transformers import AutoProcessor, CLIPModel
    except Exception as exc:
        print({"event": "clip_unavailable", "stage": "import", "error": str(exc)})
        return None
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = model.to(device).eval()
        except RuntimeError as exc:
            if device != "cuda":
                raise
            torch.cuda.empty_cache()
            print({"event": "clip_fallback_cpu", "error": str(exc)})
            model = model.to("cpu").eval()
        return model, processor
    except Exception as exc:
        print({"event": "clip_unavailable", "stage": "load", "error": str(exc)})
        return None


@torch.inference_mode()
def compute_clip_similarity(clip_bundle, image, text: str) -> Optional[float]:
    """Compute a CLIP image-text similarity score."""

    if clip_bundle is None:
        return None
    clip_model, clip_processor = clip_bundle
    max_positions = getattr(getattr(clip_model, "config", None), "text_config", None)
    max_positions = getattr(max_positions, "max_position_embeddings", 77)
    batch = clip_processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_positions,
    )
    batch = {
        key: value.to(clip_model.device)
        for key, value in batch.items()
    }
    outputs = clip_model(**batch)
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    return float((image_embeds * text_embeds).sum(dim=-1).item())


def grayscale_array(image, image_size: int) -> np.ndarray:
    """Convert one image into a grayscale NumPy array."""

    return np.asarray(
        image.resize((image_size, image_size)).convert("L"),
        dtype=np.float32,
    )


def laplacian_variance(image, image_size: int) -> float:
    """Sharpness proxy based on Laplacian variance."""

    gray = grayscale_array(image, image_size)
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(lap))


def high_frequency_energy(image, image_size: int) -> float:
    """Detail proxy based on high-frequency spectral energy."""

    gray = grayscale_array(image, image_size)
    gray = gray - gray.mean()
    spectrum = np.abs(np.fft.rfft2(gray))
    rows, cols = spectrum.shape
    y = np.linspace(0.0, 1.0, rows, endpoint=False)[:, None]
    x = np.linspace(0.0, 1.0, cols, endpoint=False)[None, :]
    radius = np.sqrt(x**2 + y**2)
    return float(spectrum[radius >= 0.35].mean()) if np.any(radius >= 0.35) else 0.0


def pair_pixel_difference(image_a, image_b) -> Dict[str, float]:
    """Measure how different two RGB images are at the pixel level.

    Returns absolute-difference statistics on the original output resolution so
    near-identical baseline/tuned pairs are easy to spot in `summary.json`.
    """

    array_a = np.asarray(image_a.convert("RGB"), dtype=np.int16)
    array_b = np.asarray(image_b.convert("RGB"), dtype=np.int16)
    diff = np.abs(array_a - array_b)
    return {
        "pair_mean_abs_diff": float(diff.mean()),
        "pair_max_abs_diff": float(diff.max()),
        "pair_same_pixels": bool(np.array_equal(array_a, array_b)),
    }


def build_base_evaluation_config(config: Dict[str, object]) -> Dict[str, object]:
    """Return a copy of an experiment config with all learned art modules disabled."""

    base_config = copy.deepcopy(config)
    freeze_cfg = base_config.setdefault("freeze", {})
    freeze_cfg["train_art_projectors"] = False
    _disable_enabled_flags(freeze_cfg.get("language_lora", {}))

    conditioning_cfg = base_config.setdefault("conditioning", {})
    _disable_enabled_flags(conditioning_cfg)
    conditioning_cfg["enabled"] = False

    brush_cfg = base_config.setdefault("brush", {})
    _disable_enabled_flags(brush_cfg)

    loss_weights = base_config.setdefault("loss_weights", {})
    for key in (
        "semantic_align",
        "art_align",
        "style_cls",
        "texture_aux",
        "laplacian",
        "sobel",
        "fft_high_frequency",
        "low_frequency_consistency",
        "texture_stats_orientation",
        "texture_stats_band",
        "texture_stats_density",
        "texture_stats_pigment",
        "stroke_theta",
        "stroke_length",
        "stroke_width",
        "stroke_alpha",
        "stroke_objectness",
        "stroke_blank_suppression",
        "stroke_support_ceiling",
        "stroke_prototype",
        "stroke_occupancy_bce",
        "stroke_anchor_outside_support",
        "stroke_anchor_sparsity",
        "stroke_anchor_small_component",
        "stroke_anchor_component_overflow",
        "stroke_anchor_inside_exclusion",
        "stroke_exclusion_overlap",
        "slot_anchor_outside_support",
        "slot_anchor_inside_exclusion",
        "slot_component_conflict",
        "slot_count_sparsity",
        "basis_usage_entropy",
        "gate_l1",
        "gate_tv",
        "protected_gate_l1",
    ):
        loss_weights[key] = 0.0
    return base_config


def build_zoom_crop_bundle(output_dir: str, rows: List[Dict[str, object]]) -> None:
    """Export fixed-region zoom comparison crops for canonical qualitative review."""

    from PIL import Image, ImageDraw, ImageFont

    crop_specs = {
        "classical_portrait": [("face_window", (0.03, 0.10, 0.70, 0.72))],
        "baroque_drama": [("face_candle_violin", (0.18, 0.16, 0.82, 0.78))],
        "impressionist_garden": [("foliage_cluster", (0.22, 0.26, 0.82, 0.82))],
        "ukiyoe_evening": [("roof_outline", (0.36, 0.18, 0.92, 0.66))],
        "ink_wash_mountain": [
            ("upper_sparse", (0.16, 0.04, 0.84, 0.48)),
            ("lower_tree_band", (0.00, 0.56, 1.00, 1.00)),
        ],
    }
    zoom_dir = Path(output_dir) / "zoom_crops"
    compare_dir = zoom_dir / "compare_cards"
    compare_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    def find_prefix(request_id: str) -> str | None:
        for prefix in crop_specs:
            if request_id.startswith(prefix):
                return prefix
        return None

    for row in rows:
        prefix = find_prefix(str(row["request_id"]))
        if prefix is None:
            continue
        with Image.open(str(row["baseline_path"])).convert("RGB") as base_image:
            base_rgb = base_image.copy()
        with Image.open(str(row["tuned_path"])).convert("RGB") as tuned_image:
            tuned_rgb = tuned_image.copy()
        for crop_name, normalized_box in crop_specs[prefix]:
            left = int(base_rgb.width * normalized_box[0])
            top = int(base_rgb.height * normalized_box[1])
            right = int(base_rgb.width * normalized_box[2])
            bottom = int(base_rgb.height * normalized_box[3])
            base_crop = base_rgb.crop((left, top, right, bottom)).resize((256, 256))
            tuned_crop = tuned_rgb.crop((left, top, right, bottom)).resize((256, 256))
            card = Image.new("RGB", (544, 330), color=(245, 242, 236))
            draw = ImageDraw.Draw(card)
            draw.rectangle((0, 0, 544, 56), fill=(28, 30, 34))
            draw.text((16, 10), f"{row['request_id']} / seed {row['seed']}", fill=(235, 235, 235), font=font)
            draw.text((16, 30), crop_name, fill=(235, 235, 235), font=font)
            draw.rectangle((16, 56, 272, 84), fill=(60, 66, 74))
            draw.rectangle((288, 56, 544 - 16, 84), fill=(60, 66, 74))
            draw.text((26, 64), "BASELINE", fill=(255, 255, 255), font=font)
            draw.text((298, 64), "TUNED", fill=(255, 255, 255), font=font)
            card.paste(base_crop, (16, 84))
            card.paste(tuned_crop, (288, 84))
            card.save(compare_dir / f"{row['request_id']}_seed_{int(row['seed']):05d}_{crop_name}.png")


def write_markdown_report(path: str, summary: Dict[str, object], rows: List[Dict[str, object]]) -> None:
    """Write a compact evaluation markdown report."""

    lines = ["# JanusFlow-Art Evaluation", ""]
    lines.append("## Summary")
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Samples")
    for row in rows:
        lines.append(f"### {row['request_id']} / seed {row['seed']}")
        lines.append(f"- Prompt: {row['prompt']}")
        lines.append(f"- Baseline path: `{row['baseline_path']}`")
        lines.append(f"- Tuned path: `{row['tuned_path']}`")
        if row.get("reference_path"):
            lines.append(f"- Reference path: `{row['reference_path']}`")
        lines.append(
            f"- Prompt similarity baseline/tuned: {row['prompt_clip_baseline']} / {row['prompt_clip_tuned']}"
        )
        lines.append(
            f"- Style similarity baseline/tuned: {row['style_clip_baseline']} / {row['style_clip_tuned']}"
        )
        lines.append(
            f"- Laplacian variance baseline/tuned: {row['laplacian_baseline']:.4f} / {row['laplacian_tuned']:.4f}"
        )
        lines.append(
            f"- High-frequency baseline/tuned: {row['hf_energy_baseline']:.4f} / {row['hf_energy_tuned']:.4f}"
        )
        lines.append(
            f"- Pair pixel abs diff mean/max: {row['pair_mean_abs_diff']:.4f} / {row['pair_max_abs_diff']:.4f}"
        )
        lines.append(f"- Pair exact pixel match: {row['pair_same_pixels']}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def build_blind_review_bundle(output_dir: str, rows: List[Dict[str, object]]) -> None:
    """Export a simple blind-review packet for human comparison."""

    from PIL import Image, ImageDraw, ImageFont

    blind_dir = Path(output_dir) / "blind_review"
    blind_dir.mkdir(parents=True, exist_ok=True)
    pair_dir = blind_dir / "paired_cards"
    pair_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = Path(output_dir) / "compare_cards"
    compare_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    sheet_lines = [
        "# JanusFlow-Art Blind Review Sheet",
        "",
        "| Item | Prompt | A | B | Preferred | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rng = random.Random(42)
    font = ImageFont.load_default()

    def build_pair_card(
        output_path: Path,
        index: int,
        prompt: str,
        left_path: str,
        right_path: str,
        left_label: str,
        right_label: str,
    ) -> str:
        with Image.open(left_path).convert("RGB") as left_image:
            left_rgb = left_image.copy()
        with Image.open(right_path).convert("RGB") as right_image:
            right_rgb = right_image.copy()

        target_width = max(left_rgb.width, right_rgb.width)
        target_height = max(left_rgb.height, right_rgb.height)
        prompt_lines = textwrap.wrap(prompt, width=70) or [prompt]
        prompt_block_height = 26 + 18 * len(prompt_lines)
        label_band_height = 28
        gutter = 16
        canvas_width = target_width * 2 + gutter * 3
        canvas_height = prompt_block_height + label_band_height + target_height + gutter * 2
        card = Image.new("RGB", (canvas_width, canvas_height), color=(245, 242, 236))
        draw = ImageDraw.Draw(card)

        draw.rectangle((0, 0, canvas_width, prompt_block_height), fill=(28, 30, 34))
        draw.text((gutter, 8), f"Item {index:03d}", fill=(235, 235, 235), font=font)
        for line_index, line in enumerate(prompt_lines):
            draw.text(
                (gutter, 28 + 18 * line_index),
                line,
                fill=(235, 235, 235),
                font=font,
            )

        top = prompt_block_height + gutter
        left_x = gutter
        right_x = gutter * 2 + target_width
        draw.rectangle((left_x, top - label_band_height, left_x + target_width, top), fill=(60, 66, 74))
        draw.rectangle((right_x, top - label_band_height, right_x + target_width, top), fill=(60, 66, 74))
        draw.text((left_x + 10, top - label_band_height + 7), left_label, fill=(255, 255, 255), font=font)
        draw.text((right_x + 10, top - label_band_height + 7), right_label, fill=(255, 255, 255), font=font)

        card.paste(left_rgb.resize((target_width, target_height)), (left_x, top))
        card.paste(right_rgb.resize((target_width, target_height)), (right_x, top))

        card.save(output_path)
        return str(output_path)

    for index, row in enumerate(rows):
        pair = [
            ("A", row["baseline_path"], "baseline"),
            ("B", row["tuned_path"], "tuned"),
        ]
        rng.shuffle(pair)
        shuffled = {label: path for label, path, _ in pair}
        answer_key = {label: kind for label, _, kind in pair}
        pair_card_path = build_pair_card(
            output_path=pair_dir / f"{index:03d}.png",
            index=index,
            prompt=row["prompt"],
            left_path=shuffled["A"],
            right_path=shuffled["B"],
            left_label="A",
            right_label="B",
        )
        compare_card_path = build_pair_card(
            output_path=compare_dir / f"{index:03d}_{row['request_id']}_seed_{row['seed']}.png",
            index=index,
            prompt=row["prompt"],
            left_path=row["baseline_path"],
            right_path=row["tuned_path"],
            left_label="BASELINE",
            right_label="TUNED",
        )
        manifest_rows.append(
            {
                "index": index,
                "request_id": row["request_id"],
                "seed": row["seed"],
                "prompt": row["prompt"],
                "A": shuffled["A"],
                "B": shuffled["B"],
                "pair_card": pair_card_path,
                "compare_card": compare_card_path,
                "answer_key": answer_key,
            }
        )
        sheet_lines.append(
            f"| {index:03d} | {row['prompt']} | A | B |  | `{Path(pair_card_path).name}` |"
        )
    save_json(blind_dir / "packet_manifest.json", {"items": manifest_rows})
    with open(blind_dir / "manual_review_sheet.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(sheet_lines) + "\n")


def collect_canonical_baseline_result(
    rows: List[Dict[str, object]],
    canonical_cfg: Dict[str, object],
) -> Dict[str, object] | None:
    """Collect and validate the canonical baseline sample for an evaluation run."""

    if not canonical_cfg or not bool(canonical_cfg.get("enabled", False)):
        return None
    target_request_id = str(canonical_cfg.get("request_id", ""))
    target_seed = int(canonical_cfg.get("seed", 42))
    matched_row = next(
        (
            row for row in rows
            if row["request_id"] == target_request_id and int(row["seed"]) == target_seed
        ),
        None,
    )
    if matched_row is None:
        raise RuntimeError(
            f"Canonical baseline sample not found for request_id={target_request_id!r}, seed={target_seed}."
        )
    actual_hash = _sha256_file(str(matched_row["baseline_path"]))
    expected_hash = canonical_cfg.get("expected_sha256")
    result = {
        "request_id": target_request_id,
        "seed": target_seed,
        "baseline_path": matched_row["baseline_path"],
        "actual_sha256": actual_hash,
        "expected_sha256": expected_hash,
        "match": expected_hash is None or str(expected_hash).lower() == actual_hash.lower(),
    }
    if expected_hash is not None and not result["match"]:
        raise RuntimeError(
            "Canonical baseline hash mismatch: "
            f"expected {expected_hash}, got {actual_hash} for {matched_row['baseline_path']}."
        )
    return result


def main() -> None:
    args = parse_args()
    config = apply_common_cli_overrides(
        load_yaml_config(args.config),
        output_root=args.output_dir,
        checkpoint=args.checkpoint,
        prompt_file=args.prompt_file,
        model_path=args.model_path,
        vae_path=args.vae_path,
        lora_scale=args.lora_scale,
    )
    eval_cfg = config["evaluation"]
    output_dir = os.path.join(
        config["experiment"]["output_root"],
        eval_cfg.get("output_subdir", "evaluation"),
    )
    os.makedirs(output_dir, exist_ok=True)
    baseline_dir = os.path.join(output_dir, "baseline")
    tuned_dir = os.path.join(output_dir, "tuned")
    reference_dir = os.path.join(output_dir, "reference")
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(tuned_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)
    print(
        {
            "event": "evaluation_start",
            "config": args.config,
            "checkpoint": args.checkpoint,
            "output_dir": output_dir,
        }
    )

    tuned_pipeline = JanusFlowArtPipeline(config, checkpoint_path=args.checkpoint, training=False)
    base_config = build_base_evaluation_config(config)
    base_pipeline = JanusFlowArtPipeline(
        base_config,
        label_vocabs=tuned_pipeline.label_vocabs,
        checkpoint_path=None,
        training=False,
    )
    prompt_dataset = ArtPromptJsonlDataset(eval_cfg["prompt_file"])
    collator = JanusFlowArtPromptCollator(
        prompt_template=config["data"].get("prompt_template", "conservative"),
        reference_image_size=int(config["data"].get("reference_image_size", 224)),
        label_vocabs=tuned_pipeline.label_vocabs,
    )
    batch = collator(list(prompt_dataset.records))
    clip_bundle = maybe_load_clip_bundle(eval_cfg.get("clip_model_name", "openai/clip-vit-base-patch32"))
    rows: List[Dict[str, object]] = []

    for seed in eval_cfg.get("seeds", [42]):
        base_images = base_pipeline.sample_images(
            prompts=batch["rendered_prompts"],
            style_label_ids=batch["style_label_ids"],
            period_label_ids=batch["period_label_ids"],
            medium_label_ids=batch["medium_label_ids"],
            reference_style_images=batch["reference_style_images"],
            has_reference_style_image=batch["has_reference_style_image"],
            subject_exclusion_hints=batch["subject_exclusion_hints"].to(
                device=base_pipeline.device,
                dtype=base_pipeline.dtype,
            ),
            seed=int(seed),
            num_inference_steps=int(eval_cfg.get("num_inference_steps", 30)),
            cfg_weight=float(eval_cfg.get("cfg_weight", 2.0)),
            image_size=int(config["data"].get("image_size", 384)),
        )
        tuned_images = tuned_pipeline.sample_images(
            prompts=batch["rendered_prompts"],
            style_label_ids=batch["style_label_ids"],
            period_label_ids=batch["period_label_ids"],
            medium_label_ids=batch["medium_label_ids"],
            reference_style_images=batch["reference_style_images"],
            has_reference_style_image=batch["has_reference_style_image"],
            subject_exclusion_hints=batch["subject_exclusion_hints"].to(
                device=tuned_pipeline.device,
                dtype=tuned_pipeline.dtype,
            ),
            seed=int(seed),
            num_inference_steps=int(eval_cfg.get("num_inference_steps", 30)),
            cfg_weight=float(eval_cfg.get("cfg_weight", 2.0)),
            image_size=int(config["data"].get("image_size", 384)),
        )
        base_pils = tensor_to_pil_images(base_images)
        tuned_pils = tensor_to_pil_images(tuned_images)

        for index, (base_image, tuned_image, record) in enumerate(
            zip(base_pils, tuned_pils, batch["records"])
        ):
            stem = f"{int(seed):05d}_{index:03d}"
            base_path = os.path.join(baseline_dir, stem + ".png")
            tuned_path = os.path.join(tuned_dir, stem + ".png")
            base_image.save(base_path)
            tuned_image.save(tuned_path)
            reference_path = None
            source_reference_path = batch["reference_image_paths"][index] or batch["image_paths"][index]
            if source_reference_path and os.path.exists(source_reference_path):
                from PIL import Image

                reference_path = os.path.join(reference_dir, stem + ".png")
                with Image.open(source_reference_path) as image:
                    image.convert("RGB").save(reference_path)
            style_prompt = build_style_proxy_prompt(record)
            pair_diff = pair_pixel_difference(base_image, tuned_image)
            rows.append(
                {
                    "seed": int(seed),
                    "index": index,
                    "request_id": batch["request_ids"][index],
                    "prompt": batch["rendered_prompts"][index],
                    "baseline_path": base_path,
                    "tuned_path": tuned_path,
                    "reference_path": reference_path,
                    "prompt_clip_baseline": compute_clip_similarity(clip_bundle, base_image, batch["rendered_prompts"][index]),
                    "prompt_clip_tuned": compute_clip_similarity(clip_bundle, tuned_image, batch["rendered_prompts"][index]),
                    "style_clip_baseline": compute_clip_similarity(clip_bundle, base_image, style_prompt),
                    "style_clip_tuned": compute_clip_similarity(clip_bundle, tuned_image, style_prompt),
                    "laplacian_baseline": laplacian_variance(base_image, int(config["data"].get("image_size", 384))),
                    "laplacian_tuned": laplacian_variance(tuned_image, int(config["data"].get("image_size", 384))),
                    "hf_energy_baseline": high_frequency_energy(base_image, int(config["data"].get("image_size", 384))),
                    "hf_energy_tuned": high_frequency_energy(tuned_image, int(config["data"].get("image_size", 384))),
                    **pair_diff,
                }
            )

        save_image_grid(base_pils, os.path.join(baseline_dir, f"grid_seed_{int(seed):05d}.png"), columns=min(2, len(base_pils)))
        save_image_grid(tuned_pils, os.path.join(tuned_dir, f"grid_seed_{int(seed):05d}.png"), columns=min(2, len(tuned_pils)))

    summary = {
        "num_rows": len(rows),
        "checkpoint": args.checkpoint,
        "baseline_model": "JanusFlow base model with JanusFlow-Art/LoRA/brush modules disabled",
        "avg_prompt_clip_baseline": float(
            np.mean([row["prompt_clip_baseline"] for row in rows if row["prompt_clip_baseline"] is not None])
        ) if any(row["prompt_clip_baseline"] is not None for row in rows) else None,
        "avg_prompt_clip_tuned": float(
            np.mean([row["prompt_clip_tuned"] for row in rows if row["prompt_clip_tuned"] is not None])
        ) if any(row["prompt_clip_tuned"] is not None for row in rows) else None,
        "avg_style_clip_baseline": float(
            np.mean([row["style_clip_baseline"] for row in rows if row["style_clip_baseline"] is not None])
        ) if any(row["style_clip_baseline"] is not None for row in rows) else None,
        "avg_style_clip_tuned": float(
            np.mean([row["style_clip_tuned"] for row in rows if row["style_clip_tuned"] is not None])
        ) if any(row["style_clip_tuned"] is not None for row in rows) else None,
        "avg_laplacian_baseline": float(np.mean([row["laplacian_baseline"] for row in rows])) if rows else None,
        "avg_laplacian_tuned": float(np.mean([row["laplacian_tuned"] for row in rows])) if rows else None,
        "avg_hf_energy_baseline": float(np.mean([row["hf_energy_baseline"] for row in rows])) if rows else None,
        "avg_hf_energy_tuned": float(np.mean([row["hf_energy_tuned"] for row in rows])) if rows else None,
        "avg_pair_pixel_abs_diff": float(np.mean([row["pair_mean_abs_diff"] for row in rows])) if rows else None,
        "min_pair_pixel_abs_diff": float(np.min([row["pair_mean_abs_diff"] for row in rows])) if rows else None,
        "max_pair_pixel_abs_diff": float(np.max([row["pair_mean_abs_diff"] for row in rows])) if rows else None,
        "same_pixel_pairs": int(sum(1 for row in rows if row["pair_same_pixels"])) if rows else 0,
        "reference_path_count": int(sum(1 for row in rows if row["reference_path"])) if rows else 0,
        "brush_runtime": tuned_pipeline.brush_runtime_summary(),
    }
    canonical_result = collect_canonical_baseline_result(
        rows,
        eval_cfg.get("canonical_baseline", {}),
    )
    if canonical_result is not None:
        summary["canonical_baseline"] = canonical_result
    save_json(os.path.join(output_dir, "summary.json"), summary)
    save_jsonl(os.path.join(output_dir, "comparison.jsonl"), rows)
    write_markdown_report(os.path.join(output_dir, "report.md"), summary, rows)
    if eval_cfg.get("save_blind_review_bundle", True):
        build_blind_review_bundle(output_dir, rows)
    build_zoom_crop_bundle(output_dir, rows)
    print(
        {
            "event": "evaluation_complete",
            "output_dir": output_dir,
            "num_rows": len(rows),
            "summary_path": os.path.join(output_dir, "summary.json"),
        }
    )


if __name__ == "__main__":
    main()
