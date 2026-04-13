"""Training entrypoint for JanusFlow-Art."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from finetune.janusflow_art_config import apply_common_cli_overrides, load_yaml_config
from finetune.janusflow_art_data import (
    ArtGenerationJsonlDataset,
    JanusFlowArtDataCollator,
    build_label_vocabs,
)
from finetune.brush_proxy_targets import build_brush_proxy_targets, build_stroke_proxy_targets
from finetune.janusflow_art_losses import (
    compute_art_alignment_loss,
    compute_basis_usage_entropy_loss,
    compute_fft_high_frequency_loss,
    compute_flow_matching_loss,
    compute_laplacian_loss,
    compute_low_frequency_consistency_loss,
    compute_sobel_gradient_loss,
    compute_slot_anchor_inside_exclusion_loss,
    compute_slot_anchor_outside_support_loss,
    compute_slot_component_conflict_loss,
    compute_slot_count_sparsity_loss,
    compute_stroke_anchor_component_overflow_loss,
    compute_stroke_anchor_inside_exclusion_loss,
    compute_stroke_anchor_outside_support_loss,
    compute_stroke_anchor_small_component_loss,
    compute_stroke_anchor_sparsity_loss,
    compute_stroke_field_alpha_loss,
    compute_stroke_field_blank_suppression_loss,
    compute_stroke_field_length_loss,
    compute_stroke_field_objectness_loss,
    compute_stroke_field_orientation_loss,
    compute_stroke_field_prototype_loss,
    compute_stroke_field_support_ceiling_loss,
    compute_stroke_field_width_loss,
    compute_stroke_exclusion_overlap_loss,
    compute_stroke_occupancy_bce_loss,
    compute_style_classification_loss,
    compute_texture_aux_loss,
    compute_texture_statistics_band_loss,
    compute_texture_statistics_density_loss,
    compute_texture_statistics_orientation_loss,
    compute_texture_statistics_pigment_loss,
)
from finetune.janusflow_art_runtime import (
    JanusFlowArtPipeline,
    count_parameters,
    get_dtype,
    load_checkpoint_state_dict_if_available,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JanusFlow-Art.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--skip-final-eval", action="store_true")
    return parser.parse_args()


def move_batch_to_device(batch: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    """Move tensor-valued batch fields onto the target device."""

    updated = dict(batch)
    updated["target_images"] = batch["target_images"].to(device=device, dtype=dtype)
    updated["reference_style_images"] = batch["reference_style_images"].to(device=device, dtype=dtype)
    updated["has_reference_style_image"] = batch["has_reference_style_image"].to(device)
    updated["style_label_ids"] = batch["style_label_ids"].to(device)
    updated["period_label_ids"] = batch["period_label_ids"].to(device)
    updated["medium_label_ids"] = batch["medium_label_ids"].to(device)
    updated["subject_exclusion_hints"] = batch["subject_exclusion_hints"].to(device=device, dtype=dtype)
    return updated


def log_rendered_prompts(path: str, batch: Dict[str, Any], split: str) -> None:
    """Append rendered prompts to the prompt log."""

    with open(path, "a", encoding="utf-8") as handle:
        for request_id, prompt in zip(batch["request_ids"], batch["rendered_prompts"]):
            handle.write(
                json.dumps(
                    {
                        "split": split,
                        "request_id": request_id,
                        "prompt": prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_resume_global_step(checkpoint_path: Optional[str]) -> int:
    """Return the saved training step for a checkpoint, or zero when absent."""

    if not checkpoint_path:
        return 0
    checkpoint_dir = Path(checkpoint_path)
    summary_path = checkpoint_dir / "checkpoint_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                step = json.load(handle).get("step")
            if step is not None:
                return max(int(step), 0)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    name = checkpoint_dir.name
    if name.startswith("checkpoint-"):
        try:
            return max(int(name.rsplit("-", 1)[-1]), 0)
        except ValueError:
            return 0
    return 0


def compute_total_loss(
    pipeline: JanusFlowArtPipeline,
    batch: Dict[str, Any],
    *,
    flow_loss_type: str,
    loss_weights: Dict[str, float],
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute the full JanusFlow-Art training loss for one batch."""

    def maybe_repeat_proxy_targets(
        targets: Dict[str, torch.Tensor],
        target_batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Repeat proxy targets when CFG-expanded batches expose more rows."""

        current_batch_size = next(iter(targets.values())).shape[0]
        if current_batch_size == target_batch_size:
            return targets
        repeat_factor = target_batch_size // current_batch_size
        updated: Dict[str, torch.Tensor] = {}
        for key, value in targets.items():
            repeats = [repeat_factor] + [1] * (value.ndim - 1)
            updated[key] = value.repeat(*repeats)
        return updated

    target_latents = pipeline.encode_target_images_to_latents(batch["target_images"])
    noise_latents = torch.randn_like(target_latents)
    timesteps = torch.rand((target_latents.shape[0],), device=pipeline.device, dtype=pipeline.dtype)
    while timesteps.ndim < target_latents.ndim:
        timesteps = timesteps.unsqueeze(-1)
    z_t = (1.0 - timesteps) * noise_latents + timesteps * target_latents
    target_velocity = target_latents - noise_latents
    scalar_timesteps = timesteps.view(target_latents.shape[0])

    prompt_ids = pipeline.tokenize_prompts(batch["rendered_prompts"])
    style_output = pipeline.encode_style_batch(
        style_label_ids=batch["style_label_ids"],
        period_label_ids=batch["period_label_ids"],
        medium_label_ids=batch["medium_label_ids"],
        reference_style_images=batch["reference_style_images"],
        has_reference_style_image=batch["has_reference_style_image"],
    )
    brush_output = pipeline.encode_brush_batch(
        reference_style_images=batch["reference_style_images"],
        has_reference_style_image=batch["has_reference_style_image"],
    )
    prediction = pipeline.predict_velocity_from_prompt_ids(
        prompt_ids=prompt_ids,
        latents=z_t,
        timesteps=scalar_timesteps,
        style_output=style_output,
        brush_output=brush_output,
        subject_exclusion_hints=batch.get("subject_exclusion_hints"),
    )

    flow_loss = compute_flow_matching_loss(
        predicted_velocity=prediction.conditioned_velocity,
        target_velocity=target_velocity,
        loss_type=flow_loss_type,
    )
    total_loss = loss_weights.get("flow", 1.0) * flow_loss
    metrics = {
        "flow_loss": float(flow_loss.item()),
    }

    if style_output is not None and loss_weights.get("art_align", 0.0) > 0.0:
        predicted_embedding = pipeline.project_art_embedding(prediction.conditioned_llm_tokens)
        art_align_loss = compute_art_alignment_loss(
            predicted_embedding,
            style_output.global_embedding,
        )
        total_loss = total_loss + loss_weights["art_align"] * art_align_loss
        metrics["art_align_loss"] = float(art_align_loss.item())

    if style_output is not None and loss_weights.get("style_cls", 0.0) > 0.0:
        predicted_embedding = pipeline.project_art_embedding(prediction.conditioned_llm_tokens)
        style_logits = pipeline.style_classifier_head(predicted_embedding)
        style_cls_loss = compute_style_classification_loss(
            style_logits,
            batch["style_label_ids"],
            ignore_index=0,
        )
        total_loss = total_loss + loss_weights["style_cls"] * style_cls_loss
        metrics["style_cls_loss"] = float(style_cls_loss.item())

    if style_output is not None and loss_weights.get("texture_aux", 0.0) > 0.0:
        predicted_texture = pipeline.project_texture_embedding(prediction.conditioned_decoder_tokens)
        target_texture = style_output.local_tokens.mean(dim=1)
        texture_aux_loss = compute_texture_aux_loss(predicted_texture, target_texture)
        total_loss = total_loss + loss_weights["texture_aux"] * texture_aux_loss
        metrics["texture_aux_loss"] = float(texture_aux_loss.item())

    if loss_weights.get("laplacian", 0.0) > 0.0:
        laplacian_loss = compute_laplacian_loss(
            prediction.conditioned_velocity,
            target_velocity,
        )
        total_loss = total_loss + loss_weights["laplacian"] * laplacian_loss
        metrics["laplacian_loss"] = float(laplacian_loss.item())

    if loss_weights.get("sobel", 0.0) > 0.0:
        sobel_loss = compute_sobel_gradient_loss(
            prediction.conditioned_velocity,
            target_velocity,
        )
        total_loss = total_loss + loss_weights["sobel"] * sobel_loss
        metrics["sobel_loss"] = float(sobel_loss.item())

    if loss_weights.get("fft_high_frequency", 0.0) > 0.0:
        fft_loss = compute_fft_high_frequency_loss(
            prediction.conditioned_velocity,
            target_velocity,
            min_radius=float(pipeline.config.get("losses", {}).get("fft_min_radius", 0.35)),
        )
        total_loss = total_loss + loss_weights["fft_high_frequency"] * fft_loss
        metrics["fft_high_frequency_loss"] = float(fft_loss.item())

    if loss_weights.get("low_frequency_consistency", 0.0) > 0.0:
        with torch.no_grad():
            base_prediction = pipeline.predict_velocity_from_prompt_ids(
                prompt_ids=prompt_ids,
                latents=z_t,
                timesteps=scalar_timesteps,
                style_output=style_output,
                brush_output=brush_output,
                apply_brush_modules=False,
                subject_exclusion_hints=batch.get("subject_exclusion_hints"),
            )
        low_frequency_loss = compute_low_frequency_consistency_loss(
            prediction.conditioned_velocity,
            base_prediction.conditioned_velocity,
            kernel_size=5,
            sigma=1.0,
        )
        total_loss = total_loss + loss_weights["low_frequency_consistency"] * low_frequency_loss
        metrics["low_frequency_consistency_loss"] = float(low_frequency_loss.item())

    texture_statistics = pipeline.latest_texture_statistics()
    if texture_statistics is not None:
        proxy_targets = build_brush_proxy_targets(
            batch["target_images"],
            grid_size=int(texture_statistics.orientation_logits.shape[-1]),
            orientation_bins=int(texture_statistics.orientation_logits.shape[1]),
        )
        proxy_targets = maybe_repeat_proxy_targets(
            proxy_targets,
            texture_statistics.orientation_logits.shape[0],
        )
        if loss_weights.get("texture_stats_orientation", 0.0) > 0.0:
            orientation_loss = compute_texture_statistics_orientation_loss(
                texture_statistics.orientation_logits,
                proxy_targets["orientation_target"],
            )
            total_loss = total_loss + loss_weights["texture_stats_orientation"] * orientation_loss
            metrics["texture_stats_orientation_loss"] = float(orientation_loss.item())
        if loss_weights.get("texture_stats_band", 0.0) > 0.0:
            band_loss = compute_texture_statistics_band_loss(
                texture_statistics.band_logits,
                proxy_targets["band_target"],
            )
            total_loss = total_loss + loss_weights["texture_stats_band"] * band_loss
            metrics["texture_stats_band_loss"] = float(band_loss.item())
        if loss_weights.get("texture_stats_density", 0.0) > 0.0:
            density_loss = compute_texture_statistics_density_loss(
                texture_statistics.density_map,
                proxy_targets["density_target"],
            )
            total_loss = total_loss + loss_weights["texture_stats_density"] * density_loss
            metrics["texture_stats_density_loss"] = float(density_loss.item())
        if loss_weights.get("texture_stats_pigment", 0.0) > 0.0:
            pigment_loss = compute_texture_statistics_pigment_loss(
                texture_statistics.pigment_map,
                proxy_targets["pigment_target"],
            )
            total_loss = total_loss + loss_weights["texture_stats_pigment"] * pigment_loss
            metrics["texture_stats_pigment_loss"] = float(pigment_loss.item())

    stroke_fields = pipeline.latest_stroke_fields()
    if stroke_fields is not None:
        stroke_head = pipeline.brush_stroke_field_head
        if stroke_head is None:
            stroke_head = pipeline.brush_support_locked_primitive_head
        if stroke_head is None:
            stroke_head = pipeline.brush_slot_based_anchor_set_head
        anchors = stroke_head.prototype_anchor_metadata() if stroke_head is not None else None
        stroke_targets = build_stroke_proxy_targets(
            batch["target_images"],
            grid_size=int(stroke_fields.theta.shape[-1]),
            prototype_orientations=None if anchors is None else anchors["orientation"],
            prototype_lengths=None if anchors is None else anchors["length"],
            prototype_widths=None if anchors is None else anchors["width"],
        )
        stroke_targets = maybe_repeat_proxy_targets(stroke_targets, stroke_fields.theta.shape[0])
        stroke_weight = stroke_targets["alpha_target"]
        if loss_weights.get("stroke_theta", 0.0) > 0.0:
            stroke_theta_loss = compute_stroke_field_orientation_loss(
                stroke_fields.theta,
                stroke_targets["angle_target"],
                weight=stroke_weight,
            )
            total_loss = total_loss + loss_weights["stroke_theta"] * stroke_theta_loss
            metrics["stroke_theta_loss"] = float(stroke_theta_loss.item())
        if loss_weights.get("stroke_length", 0.0) > 0.0:
            stroke_length_loss = compute_stroke_field_length_loss(
                stroke_fields.length,
                stroke_targets["length_target"],
                weight=stroke_weight,
            )
            total_loss = total_loss + loss_weights["stroke_length"] * stroke_length_loss
            metrics["stroke_length_loss"] = float(stroke_length_loss.item())
        if loss_weights.get("stroke_width", 0.0) > 0.0:
            stroke_width_loss = compute_stroke_field_width_loss(
                stroke_fields.width,
                stroke_targets["width_target"],
                weight=stroke_weight,
            )
            total_loss = total_loss + loss_weights["stroke_width"] * stroke_width_loss
            metrics["stroke_width_loss"] = float(stroke_width_loss.item())
        if loss_weights.get("stroke_alpha", 0.0) > 0.0:
            stroke_alpha_loss = compute_stroke_field_alpha_loss(
                stroke_fields.alpha,
                stroke_targets["alpha_target"],
            )
            total_loss = total_loss + loss_weights["stroke_alpha"] * stroke_alpha_loss
            metrics["stroke_alpha_loss"] = float(stroke_alpha_loss.item())
        if loss_weights.get("stroke_objectness", 0.0) > 0.0:
            stroke_objectness_loss = compute_stroke_field_objectness_loss(
                stroke_fields.objectness,
                stroke_targets["objectness_target"],
            )
            total_loss = total_loss + loss_weights["stroke_objectness"] * stroke_objectness_loss
            metrics["stroke_objectness_loss"] = float(stroke_objectness_loss.item())
        if loss_weights.get("stroke_blank_suppression", 0.0) > 0.0:
            stroke_blank_suppression_loss = compute_stroke_field_blank_suppression_loss(
                stroke_fields.objectness,
                stroke_targets["blank_region_target"],
            )
            total_loss = total_loss + loss_weights["stroke_blank_suppression"] * stroke_blank_suppression_loss
            metrics["stroke_blank_suppression_loss"] = float(stroke_blank_suppression_loss.item())
        if loss_weights.get("stroke_support_ceiling", 0.0) > 0.0:
            stroke_support_ceiling_loss = compute_stroke_field_support_ceiling_loss(
                stroke_fields.objectness,
                stroke_targets["support_dilated_target"],
            )
            total_loss = total_loss + loss_weights["stroke_support_ceiling"] * stroke_support_ceiling_loss
            metrics["stroke_support_ceiling_loss"] = float(stroke_support_ceiling_loss.item())
        if (
            loss_weights.get("stroke_prototype", 0.0) > 0.0
            and "prototype_target" in stroke_targets
        ):
            stroke_prototype_loss = compute_stroke_field_prototype_loss(
                stroke_fields.prototype_logits,
                stroke_targets["prototype_target"],
            )
            total_loss = total_loss + loss_weights["stroke_prototype"] * stroke_prototype_loss
            metrics["stroke_prototype_loss"] = float(stroke_prototype_loss.item())
        if loss_weights.get("stroke_occupancy_bce", 0.0) > 0.0 and hasattr(stroke_fields, "occupancy_logits"):
            stroke_occupancy_bce_loss = compute_stroke_occupancy_bce_loss(
                stroke_fields.occupancy_logits,
                stroke_targets["objectness_target"],
            )
            total_loss = total_loss + loss_weights["stroke_occupancy_bce"] * stroke_occupancy_bce_loss
            metrics["stroke_occupancy_bce_loss"] = float(stroke_occupancy_bce_loss.item())
        if loss_weights.get("stroke_anchor_outside_support", 0.0) > 0.0 and hasattr(stroke_fields, "anchor_map"):
            stroke_anchor_outside_support_loss = compute_stroke_anchor_outside_support_loss(
                stroke_fields.anchor_map,
                stroke_targets["support_dilated_target"],
            )
            total_loss = total_loss + (
                loss_weights["stroke_anchor_outside_support"] * stroke_anchor_outside_support_loss
            )
            metrics["stroke_anchor_outside_support_loss"] = float(stroke_anchor_outside_support_loss.item())
        if loss_weights.get("stroke_anchor_sparsity", 0.0) > 0.0 and hasattr(stroke_fields, "anchor_map"):
            stroke_anchor_sparsity_loss = compute_stroke_anchor_sparsity_loss(stroke_fields.anchor_map)
            total_loss = total_loss + loss_weights["stroke_anchor_sparsity"] * stroke_anchor_sparsity_loss
            metrics["stroke_anchor_sparsity_loss"] = float(stroke_anchor_sparsity_loss.item())
        if (
            loss_weights.get("stroke_anchor_small_component", 0.0) > 0.0
            and hasattr(stroke_fields, "small_component_mask")
        ):
            stroke_anchor_small_component_loss = compute_stroke_anchor_small_component_loss(
                stroke_fields.anchor_map,
                stroke_fields.small_component_mask,
            )
            total_loss = total_loss + (
                loss_weights["stroke_anchor_small_component"] * stroke_anchor_small_component_loss
            )
            metrics["stroke_anchor_small_component_loss"] = float(
                stroke_anchor_small_component_loss.item()
            )
        if (
            loss_weights.get("stroke_anchor_component_overflow", 0.0) > 0.0
            and hasattr(stroke_fields, "component_overflow_score")
        ):
            stroke_anchor_component_overflow_loss = compute_stroke_anchor_component_overflow_loss(
                stroke_fields.component_overflow_score,
            )
            total_loss = total_loss + (
                loss_weights["stroke_anchor_component_overflow"] * stroke_anchor_component_overflow_loss
            )
            metrics["stroke_anchor_component_overflow_loss"] = float(
                stroke_anchor_component_overflow_loss.item()
            )
        if (
            loss_weights.get("stroke_anchor_inside_exclusion", 0.0) > 0.0
            and hasattr(stroke_fields, "exclusion_map")
        ):
            stroke_anchor_inside_exclusion_loss = compute_stroke_anchor_inside_exclusion_loss(
                stroke_fields.anchor_map,
                stroke_fields.exclusion_map,
            )
            total_loss = total_loss + (
                loss_weights["stroke_anchor_inside_exclusion"] * stroke_anchor_inside_exclusion_loss
            )
            metrics["stroke_anchor_inside_exclusion_loss"] = float(
                stroke_anchor_inside_exclusion_loss.item()
            )
        if (
            loss_weights.get("stroke_exclusion_overlap", 0.0) > 0.0
            and hasattr(stroke_fields, "exclusion_map")
            and hasattr(stroke_fields, "occupancy_logits")
        ):
            stroke_exclusion_overlap_loss = compute_stroke_exclusion_overlap_loss(
                stroke_fields.occupancy_logits,
                stroke_fields.exclusion_map,
            )
            total_loss = total_loss + (
                loss_weights["stroke_exclusion_overlap"] * stroke_exclusion_overlap_loss
            )
            metrics["stroke_exclusion_overlap_loss"] = float(
                stroke_exclusion_overlap_loss.item()
            )
        if (
            loss_weights.get("slot_anchor_outside_support", 0.0) > 0.0
            and hasattr(stroke_fields, "slot_valid_probs")
            and hasattr(stroke_fields, "slot_support_mass")
        ):
            slot_anchor_outside_support_loss = compute_slot_anchor_outside_support_loss(
                stroke_fields.slot_valid_probs,
                stroke_fields.slot_support_mass,
            )
            total_loss = total_loss + (
                loss_weights["slot_anchor_outside_support"] * slot_anchor_outside_support_loss
            )
            metrics["slot_anchor_outside_support_loss"] = float(
                slot_anchor_outside_support_loss.item()
            )
        if (
            loss_weights.get("slot_anchor_inside_exclusion", 0.0) > 0.0
            and hasattr(stroke_fields, "slot_valid_probs")
            and hasattr(stroke_fields, "slot_exclusion_mass")
        ):
            slot_anchor_inside_exclusion_loss = compute_slot_anchor_inside_exclusion_loss(
                stroke_fields.slot_valid_probs,
                stroke_fields.slot_exclusion_mass,
            )
            total_loss = total_loss + (
                loss_weights["slot_anchor_inside_exclusion"] * slot_anchor_inside_exclusion_loss
            )
            metrics["slot_anchor_inside_exclusion_loss"] = float(
                slot_anchor_inside_exclusion_loss.item()
            )
        if (
            loss_weights.get("slot_component_conflict", 0.0) > 0.0
            and hasattr(stroke_fields, "slot_component_conflict_score")
        ):
            slot_component_conflict_loss = compute_slot_component_conflict_loss(
                stroke_fields.slot_component_conflict_score
            )
            total_loss = total_loss + (
                loss_weights["slot_component_conflict"] * slot_component_conflict_loss
            )
            metrics["slot_component_conflict_loss"] = float(slot_component_conflict_loss.item())
        if (
            loss_weights.get("slot_count_sparsity", 0.0) > 0.0
            and hasattr(stroke_fields, "slot_valid_probs")
        ):
            slot_count_sparsity_loss = compute_slot_count_sparsity_loss(
                stroke_fields.slot_valid_probs
            )
            total_loss = total_loss + (
                loss_weights["slot_count_sparsity"] * slot_count_sparsity_loss
            )
            metrics["slot_count_sparsity_loss"] = float(slot_count_sparsity_loss.item())

    gate_losses = pipeline.brush_gate_regularization_losses()
    if loss_weights.get("gate_l1", 0.0) > 0.0 and "gate_l1" in gate_losses:
        gate_l1_loss = gate_losses["gate_l1"]
        total_loss = total_loss + loss_weights["gate_l1"] * gate_l1_loss
        metrics["gate_l1_loss"] = float(gate_l1_loss.item())

    if loss_weights.get("gate_tv", 0.0) > 0.0 and "gate_tv" in gate_losses:
        gate_tv_loss = gate_losses["gate_tv"]
        total_loss = total_loss + loss_weights["gate_tv"] * gate_tv_loss
        metrics["gate_tv_loss"] = float(gate_tv_loss.item())

    if loss_weights.get("protected_gate_l1", 0.0) > 0.0 and "protected_gate_l1" in gate_losses:
        protected_gate_l1_loss = gate_losses["protected_gate_l1"]
        total_loss = total_loss + loss_weights["protected_gate_l1"] * protected_gate_l1_loss
        metrics["protected_gate_l1_loss"] = float(protected_gate_l1_loss.item())

    if loss_weights.get("basis_usage_entropy", 0.0) > 0.0 and "basis_usage_entropy" in gate_losses:
        basis_entropy_loss = compute_basis_usage_entropy_loss(gate_losses["basis_usage_entropy"])
        total_loss = total_loss + loss_weights["basis_usage_entropy"] * basis_entropy_loss
        metrics["basis_usage_entropy_loss"] = float(basis_entropy_loss.item())

    return total_loss, metrics


def evaluate(
    pipeline: JanusFlowArtPipeline,
    dataloader: DataLoader,
    *,
    dtype: torch.dtype,
    flow_loss_type: str,
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    """Run validation with the same flow objective used during training."""

    pipeline.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, pipeline.device, dtype)
            with (
                torch.autocast(device_type="cuda", dtype=dtype)
                if torch.cuda.is_available()
                else nullcontext()
            ):
                loss, metrics = compute_total_loss(
                    pipeline,
                    batch,
                    flow_loss_type=flow_loss_type,
                    loss_weights=loss_weights,
                )
            total_loss += float(loss.item())
            total_steps += 1
    pipeline.train()
    return {"eval_loss": total_loss / max(total_steps, 1)}


def main() -> None:
    args = parse_args()
    config = apply_common_cli_overrides(
        load_yaml_config(args.config),
        max_steps=args.max_steps,
        output_root=args.output_dir,
        checkpoint=args.resume_from_checkpoint,
        init_checkpoint=args.init_from_checkpoint,
        model_path=args.model_path,
        vae_path=args.vae_path,
        skip_final_eval=args.skip_final_eval,
    )

    output_dir = config["experiment"]["output_root"]
    os.makedirs(output_dir, exist_ok=True)
    save_json(Path(output_dir) / "train_config.json", config)
    print(
        {
            "event": "train_start",
            "config": args.config,
            "output_dir": output_dir,
        }
    )

    set_seed(int(config["training"].get("seed", 42)))
    train_dataset = ArtGenerationJsonlDataset(config["data"]["train_manifest"])
    val_dataset = ArtGenerationJsonlDataset(config["data"]["val_manifest"])
    label_vocabs = build_label_vocabs(list(train_dataset.records) + list(val_dataset.records))
    print(
        {
            "event": "datasets_ready",
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "train_missing_image_records": len(getattr(train_dataset, "missing_image_records", [])),
            "val_missing_image_records": len(getattr(val_dataset, "missing_image_records", [])),
            "num_style_labels": len(label_vocabs.get("style_label", {})),
            "num_period_labels": len(label_vocabs.get("period_label", {})),
            "num_medium_labels": len(label_vocabs.get("medium_label", {})),
        }
    )

    resume_checkpoint = args.resume_from_checkpoint or config.get("checkpoint", {}).get("resume_from")
    init_checkpoint = args.init_from_checkpoint or config.get("training", {}).get("init_from_checkpoint")
    pipeline_checkpoint = resume_checkpoint or init_checkpoint
    print(
        {
            "event": "checkpoint_resolution",
            "resume_from_checkpoint": resume_checkpoint,
            "init_from_checkpoint": init_checkpoint,
            "pipeline_restore_source": pipeline_checkpoint,
        }
    )
    pipeline = JanusFlowArtPipeline(
        config,
        label_vocabs=label_vocabs,
        checkpoint_path=pipeline_checkpoint,
        checkpoint_strict=bool(resume_checkpoint),
        training=True,
    )
    print(
        {
            "event": "pipeline_ready",
            "device": str(pipeline.device),
            "dtype": str(pipeline.dtype),
            "parameter_summary": count_parameters(pipeline),
            "brush_runtime": pipeline.brush_runtime_summary(),
        }
    )
    collator = JanusFlowArtDataCollator(
        image_size=int(config["data"].get("image_size", 384)),
        image_preprocess_mode=config["data"].get("image_preprocess_mode", "crop"),
        reference_image_size=int(config["data"].get("reference_image_size", 224)),
        prompt_template=config["data"].get("prompt_template", "conservative"),
        label_vocabs=pipeline.label_vocabs,
        use_target_as_reference_when_missing=bool(
            config["data"].get("use_target_as_reference_when_missing", False)
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"].get("per_device_train_batch_size", 1)),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 2)),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("per_device_eval_batch_size", 1)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 2)),
        collate_fn=collator,
    )
    optimizer = pipeline.build_optimizer()

    training_cfg = config["training"]
    total_update_steps_per_epoch = math.ceil(
        len(train_loader) / max(int(training_cfg.get("gradient_accumulation_steps", 1)), 1)
    )
    planned_steps = math.ceil(total_update_steps_per_epoch * float(training_cfg.get("num_epochs", 1.0)))
    if int(training_cfg.get("max_steps", 0)) > 0:
        planned_steps = int(training_cfg["max_steps"])
    warmup_steps = int(planned_steps * float(training_cfg.get("warmup_ratio", 0.03)))

    if training_cfg.get("scheduler_type", "cosine") == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=planned_steps,
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=planned_steps,
        )
    print(
        {
            "event": "optimizer_ready",
            "planned_steps": planned_steps,
            "warmup_steps": warmup_steps,
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        }
    )

    optimizer_state = load_checkpoint_state_dict_if_available(resume_checkpoint, "optimizer.pt")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = load_checkpoint_state_dict_if_available(resume_checkpoint, "scheduler.pt")
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    dtype = get_dtype(config["model"].get("dtype", "auto"))
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and torch.cuda.is_available()))
    log_path = os.path.join(output_dir, "train_log.jsonl")
    prompt_log_path = os.path.join(output_dir, "prompt_log.jsonl")
    global_step = load_resume_global_step(resume_checkpoint)
    flow_loss_type = training_cfg.get("flow_loss_type", "mse")
    loss_weights = {key: float(value) for key, value in config.get("loss_weights", {}).items()}
    if global_step > 0:
        print({"event": "resume_global_step", "global_step": global_step})

    pipeline.train()
    optimizer.zero_grad(set_to_none=True)
    with open(log_path, "a", encoding="utf-8") as log_handle:
        for epoch in range(math.ceil(float(training_cfg.get("num_epochs", 1.0)))):
            if global_step >= planned_steps:
                break
            for step, batch in enumerate(train_loader, start=1):
                if global_step >= planned_steps:
                    break
                batch = move_batch_to_device(batch, pipeline.device, dtype)
                log_rendered_prompts(prompt_log_path, batch, split="train")
                step_start = time.time()
                accumulation_steps = max(int(training_cfg.get("gradient_accumulation_steps", 1)), 1)
                remainder_steps = len(train_loader) % accumulation_steps
                in_final_partial_window = remainder_steps > 0 and step > len(train_loader) - remainder_steps
                accumulation_window = remainder_steps if in_final_partial_window else accumulation_steps
                with (
                    torch.autocast(device_type="cuda", dtype=dtype)
                    if torch.cuda.is_available()
                    else nullcontext()
                ):
                    loss, metrics = compute_total_loss(
                        pipeline,
                        batch,
                        flow_loss_type=flow_loss_type,
                        loss_weights=loss_weights,
                    )
                    scaled_loss = loss / accumulation_window

                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if step % accumulation_steps != 0 and step != len(train_loader):
                    continue

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(pipeline.parameters(), float(training_cfg.get("max_grad_norm", 1.0))).item())
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(pipeline.parameters(), float(training_cfg.get("max_grad_norm", 1.0))).item())
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": float(loss.item()),
                    "grad_norm": round(grad_norm, 6),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "step_time_sec": round(time.time() - step_start, 4),
                    **metrics,
                }
                print(record)
                log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_handle.flush()

                if global_step % int(config["logging"].get("eval_steps", 200)) == 0 and len(val_dataset) > 0:
                    eval_metrics = evaluate(
                        pipeline,
                        val_loader,
                        dtype=dtype,
                        flow_loss_type=flow_loss_type,
                        loss_weights=loss_weights,
                    )
                    eval_record = {"global_step": global_step, **eval_metrics}
                    print(eval_record)
                    log_handle.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
                    log_handle.flush()

                if global_step % int(config["logging"].get("save_steps", 200)) == 0:
                    checkpoint_dir = pipeline.save_checkpoint(
                        output_dir,
                        step=global_step,
                        config_path=args.config,
                    )
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

    final_checkpoint = pipeline.save_checkpoint(output_dir, step=None, config_path=args.config)
    torch.save(optimizer.state_dict(), os.path.join(final_checkpoint, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(final_checkpoint, "scheduler.pt"))
    final_metrics = {
        "final_checkpoint": final_checkpoint,
        "parameter_summary": count_parameters(pipeline),
        "brush_runtime": pipeline.brush_runtime_summary(),
    }
    if len(val_dataset) > 0 and bool(config["logging"].get("run_final_eval", True)):
        final_metrics.update(
            evaluate(
                pipeline,
                val_loader,
                dtype=dtype,
                flow_loss_type=flow_loss_type,
                loss_weights=loss_weights,
            )
        )
    print(final_metrics)
    save_json(Path(output_dir) / "final_summary.json", final_metrics)


if __name__ == "__main__":
    main()
