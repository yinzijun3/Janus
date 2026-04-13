"""Shared runtime utilities for JanusFlow-Art."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune.janusflow_art_data import MISSING_TOKEN
from janus.janusflow.models.brush import (
    AnchoredOutputVelocityTextureHead,
    BrushAdapter,
    BrushConditionInjector,
    BrushFeatureMapAdapter,
    BrushReferenceOutput,
    OutputVelocityTextureHead,
    PatchBrushReferenceEncoder,
    SlotBasedAnchorSetBrushHead,
    SpatialHFBrushHead,
    SupportLockedPrimitiveBrushHead,
    StrokeFieldBrushHead,
    TextureStatisticsBrushHead,
)
from janus.janusflow.models.brush.patch_reference_encoder import (
    mask_brush_reference_output,
    repeat_brush_reference_output,
)
from janus.janusflow.models.conditioning import StyleConditionInjector
from janus.janusflow.models.style_encoder import (
    LabelStyleEncoder,
    ReferenceStyleImageEncoder,
    StyleEncoderOutput,
)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a human-readable dtype name to a torch dtype."""

    normalized = str(dtype_name).lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_lm_backbone(language_model: nn.Module) -> nn.Module:
    """Return the decoder-only backbone used for JanusFlow hidden-state access."""

    backbone = language_model
    if hasattr(language_model, "get_base_model"):
        backbone = language_model.get_base_model()
    if hasattr(backbone, "model"):
        return backbone.model
    return backbone


def get_output_last_hidden_state(outputs: Any) -> torch.Tensor:
    """Extract the final hidden states from either model or causal-LM outputs.

    Some JanusFlow codepaths call the decoder backbone directly and receive a
    `BaseModelOutputWithPast` with `last_hidden_state`, while others may still
    surface a `CausalLMOutputWithPast` that only exposes `hidden_states`.
    """

    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    raise AttributeError("language model outputs do not expose final hidden states")


def count_parameters(module: nn.Module) -> Dict[str, float]:
    """Return total and trainable parameter counts for a module tree."""

    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": trainable / max(total, 1),
    }


def ensure_label_vocabs(
    label_vocabs: Optional[Dict[str, Dict[str, int]]],
) -> Dict[str, Dict[str, int]]:
    """Provide a minimal label-vocab structure when no vocab is supplied."""

    if label_vocabs is not None:
        return label_vocabs
    return {
        "style_label": {MISSING_TOKEN: 0},
        "period_label": {MISSING_TOKEN: 0},
        "medium_label": {MISSING_TOKEN: 0},
    }


@dataclass
class VelocityPredictionOutput:
    """Outputs returned by the JanusFlow-Art denoiser pass."""

    conditioned_velocity: torch.Tensor
    unconditioned_velocity: torch.Tensor
    conditioned_llm_tokens: torch.Tensor
    conditioned_decoder_tokens: torch.Tensor
    conditioned_decoder_feature_map: torch.Tensor
    style_output: Optional[StyleEncoderOutput]
    brush_output: Optional[BrushReferenceOutput]


class JanusFlowArtPipeline(nn.Module):
    """High-level JanusFlow-Art model wrapper used by train, sample, and eval."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        label_vocabs: Optional[Dict[str, Dict[str, int]]] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = True,
        training: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.training_mode = training
        self.checkpoint_path = checkpoint_path
        self.checkpoint_strict = bool(checkpoint_strict)
        self.label_vocabs = ensure_label_vocabs(label_vocabs)
        self._lora_base_scaling: Dict[Tuple[str, str], float] = {}

        art_state = self._load_art_state_preview(checkpoint_path)
        if art_state and art_state.get("label_vocabs"):
            self.label_vocabs = art_state["label_vocabs"]

        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = get_dtype(config["model"].get("dtype", "auto"))

        self.processor = self._build_processor()
        self.model = self._build_base_model()
        self.vae = self._build_vae()

        conditioning_cfg = config.get("conditioning", {})
        self.style_hidden_size = int(conditioning_cfg.get("style_hidden_size", 512))

        self.label_style_encoder = self._build_label_style_encoder(conditioning_cfg)
        self.reference_style_encoder = self._build_reference_style_encoder(conditioning_cfg)
        self.style_injector = self._build_style_injector(conditioning_cfg)
        brush_cfg = config.get("brush", {})
        self.brush_reference_encoder = self._build_brush_reference_encoder(brush_cfg)
        self.brush_condition_injector = self._build_brush_condition_injector(brush_cfg)
        self.brush_adapter = self._build_brush_adapter(brush_cfg)
        self.brush_feature_map_adapter = self._build_brush_feature_map_adapter(brush_cfg)
        self.brush_spatial_hf_head = self._build_brush_spatial_hf_head(brush_cfg)
        self.brush_texture_statistics_head = self._build_brush_texture_statistics_head(brush_cfg)
        self.brush_stroke_field_head = self._build_brush_stroke_field_head(brush_cfg)
        self.brush_support_locked_primitive_head = self._build_brush_support_locked_primitive_head(brush_cfg)
        self.brush_slot_based_anchor_set_head = self._build_brush_slot_based_anchor_set_head(brush_cfg)
        self.brush_output_velocity_head = self._build_brush_output_velocity_head(brush_cfg)
        self.brush_anchored_output_velocity_head = self._build_brush_anchored_output_velocity_head(brush_cfg)

        self.art_projector = nn.Linear(2048, self.style_hidden_size)
        self.texture_projector = nn.Linear(768, self.style_hidden_size)
        self.style_classifier_head = nn.Linear(
            self.style_hidden_size,
            len(self.label_vocabs["style_label"]),
        )

        self.to(device=self.device_name, dtype=self.dtype)
        self._freeze_everything()
        restore_language_model = self._has_checkpoint_language_model(checkpoint_path)
        self._configure_trainable_modules(skip_language_lora=restore_language_model)
        if checkpoint_path:
            self._restore_checkpoint(checkpoint_path, training=training)
        self._configure_language_model_runtime(training=training)
        if not training:
            self._apply_language_lora_inference_scale()
        # Keep the actual module mode aligned with the intended runtime role.
        # `training_mode` controls JanusFlow-specific CFG behavior, while the
        # module `.training` flag controls dropout and other layer behavior.
        # If we leave inference pipelines in train mode, LoRA dropout remains
        # active and adapter-vs-base comparisons become unreliable.
        super().train(training)

    def _load_art_state_preview(self, checkpoint_path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not checkpoint_path:
            return None
        art_state_path = Path(checkpoint_path) / "art_state.pt"
        if not art_state_path.exists():
            return None
        return torch.load(art_state_path, map_location="cpu")

    @staticmethod
    def _has_checkpoint_language_model(checkpoint_path: Optional[str]) -> bool:
        """Return whether a checkpoint already contains a saved PEFT language model."""

        if not checkpoint_path:
            return False
        return (Path(checkpoint_path) / "language_model" / "adapter_config.json").exists()

    def _build_processor(self):
        from janus.janusflow.models import VLChatProcessor

        return VLChatProcessor.from_pretrained(self.config["model"]["model_path"])

    def _build_base_model(self):
        from janus.janusflow.models import MultiModalityCausalLM

        model = MultiModalityCausalLM.from_pretrained(
            self.config["model"]["model_path"],
            trust_remote_code=True,
        )
        return model

    def _build_vae(self):
        from diffusers.models import AutoencoderKL
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
        from safetensors.torch import load_file as load_safetensors

        vae_path = Path(self.config["model"]["vae_path"])
        single_file_candidates = (
            vae_path / "sdxl_vae.safetensors",
            vae_path / "diffusion_pytorch_model.safetensors",
            vae_path / "diffusion_pytorch_model.bin",
        )

        if vae_path.is_dir():
            single_file_path = next((path for path in single_file_candidates if path.exists()), None)
            if single_file_path is not None and single_file_path.name == "sdxl_vae.safetensors":
                # This local SDXL VAE bundle stores LDM-style weights alongside a
                # diffusers `config.json`, so rebuild the diffusers module from
                # config and convert the checkpoint keys locally.
                config_dict, _ = AutoencoderKL.load_config(str(vae_path), return_unused_kwargs=True)
                checkpoint = load_safetensors(str(single_file_path))
                converted_checkpoint = convert_ldm_vae_checkpoint(checkpoint, config_dict)
                vae = AutoencoderKL.from_config(config_dict)
                vae.load_state_dict(converted_checkpoint)
            else:
                vae = AutoencoderKL.from_pretrained(str(vae_path))
        else:
            vae = AutoencoderKL.from_single_file(str(vae_path))
        vae.eval()
        for parameter in vae.parameters():
            parameter.requires_grad = False
        return vae

    def _build_label_style_encoder(self, conditioning_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        label_cfg = conditioning_cfg.get("label_encoder", {})
        needs_label_encoder = bool(label_cfg.get("enabled", False)) or any(
            float(self.config.get("loss_weights", {}).get(key, 0.0)) > 0.0
            for key in ("art_align", "style_cls", "texture_aux")
        )
        if not needs_label_encoder:
            return None
        return LabelStyleEncoder(
            num_style_labels=len(self.label_vocabs["style_label"]),
            num_period_labels=len(self.label_vocabs["period_label"]),
            num_medium_labels=len(self.label_vocabs["medium_label"]),
            hidden_size=self.style_hidden_size,
            num_global_tokens=int(conditioning_cfg.get("num_global_tokens", 4)),
            num_local_tokens=int(conditioning_cfg.get("num_local_tokens", 8)),
            dropout=float(label_cfg.get("dropout", 0.0)),
        )

    def _build_reference_style_encoder(self, conditioning_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        reference_cfg = conditioning_cfg.get("reference_encoder", {})
        if not bool(reference_cfg.get("enabled", False)):
            return None
        return ReferenceStyleImageEncoder(
            hidden_size=self.style_hidden_size,
            num_global_tokens=int(conditioning_cfg.get("num_global_tokens", 4)),
            num_local_tokens=int(conditioning_cfg.get("num_local_tokens", 8)),
            base_channels=int(reference_cfg.get("base_channels", 64)),
        )

    def _build_style_injector(self, conditioning_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        injector_cfg = conditioning_cfg.get("injector", {})
        if not bool(injector_cfg.get("enabled", False)):
            return None
        return StyleConditionInjector(
            style_hidden_size=self.style_hidden_size,
            llm_hidden_size=2048,
            decoder_hidden_size=768,
            global_scale=float(injector_cfg.get("global_scale", 1.0)),
            local_scale=float(injector_cfg.get("local_scale", 1.0)),
            dropout=float(injector_cfg.get("dropout", 0.0)),
        )

    def _build_brush_reference_encoder(self, brush_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        """Build the optional patch-level brush reference encoder."""

        reference_cfg = brush_cfg.get("reference_encoder", {})
        if not bool(reference_cfg.get("enabled", False)):
            return None
        return PatchBrushReferenceEncoder(
            hidden_size=int(brush_cfg.get("style_hidden_size", self.style_hidden_size)),
            base_channels=int(reference_cfg.get("base_channels", 64)),
            num_coarse_tokens=int(reference_cfg.get("num_coarse_tokens", 4)),
            num_mid_tokens=int(reference_cfg.get("num_mid_tokens", 8)),
            num_fine_tokens=int(reference_cfg.get("num_fine_tokens", 16)),
        )

    def _build_brush_condition_injector(self, brush_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        """Build the optional decoder-local brush condition injector."""

        injector_cfg = brush_cfg.get("condition_injector", {})
        if not bool(injector_cfg.get("enabled", False)):
            return None
        return BrushConditionInjector(
            style_hidden_size=int(brush_cfg.get("style_hidden_size", self.style_hidden_size)),
            decoder_hidden_size=int(brush_cfg.get("decoder_hidden_size", 768)),
            use_mid_tokens=bool(injector_cfg.get("use_mid_tokens", True)),
            use_fine_tokens=bool(injector_cfg.get("use_fine_tokens", True)),
            cross_attention_scale=float(injector_cfg.get("cross_attention_scale", 0.25)),
            dropout=float(injector_cfg.get("dropout", 0.0)),
        )

    def _build_brush_adapter(self, brush_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        """Build the optional decoder-side brush adapter."""

        adapter_cfg = brush_cfg.get("adapter", {})
        if not bool(adapter_cfg.get("enabled", False)):
            return None
        expert_cfg = brush_cfg.get("decoder_experts", {})
        expert_count = int(expert_cfg.get("num_experts", 0)) if bool(expert_cfg.get("enabled", False)) else 0
        return BrushAdapter(
            decoder_hidden_size=int(brush_cfg.get("decoder_hidden_size", 768)),
            style_hidden_size=int(brush_cfg.get("style_hidden_size", self.style_hidden_size)),
            bottleneck_channels=int(adapter_cfg.get("bottleneck_channels", 192)),
            kernel_size=int(adapter_cfg.get("kernel_size", 3)),
            residual_init=float(adapter_cfg.get("residual_init", 0.02)),
            residual_scale=float(adapter_cfg.get("residual_scale", 1.0)),
            dropout=float(adapter_cfg.get("dropout", 0.0)),
            expert_count=expert_count,
        )

    def _build_brush_feature_map_adapter(self, brush_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        """Build the optional decoder feature-map brush adapter."""

        feature_cfg = brush_cfg.get("feature_map_adapter", {})
        if not bool(feature_cfg.get("enabled", False)):
            return None
        return BrushFeatureMapAdapter(
            feature_channels=int(brush_cfg.get("decoder_hidden_size", 768)),
            style_hidden_size=int(brush_cfg.get("style_hidden_size", self.style_hidden_size)),
            hidden_channels=int(feature_cfg.get("hidden_channels", 384)),
            kernel_size=int(feature_cfg.get("kernel_size", 3)),
            residual_init=float(feature_cfg.get("residual_init", 0.05)),
            residual_scale=float(feature_cfg.get("residual_scale", 1.0)),
            dropout=float(feature_cfg.get("dropout", 0.0)),
        )

    def _build_brush_spatial_hf_head(self, brush_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        """Build the optional spatially gated high-frequency brush head."""

        head_cfg = brush_cfg.get("spatial_hf_head", {})
        if not bool(head_cfg.get("enabled", False)):
            return None
        return SpatialHFBrushHead(
            feature_channels=int(brush_cfg.get("decoder_hidden_size", 768)),
            hidden_channels=int(head_cfg.get("hidden_channels", 512)),
            kernel_size=int(head_cfg.get("kernel_size", 5)),
            residual_init=float(head_cfg.get("residual_init", 0.18)),
            residual_scale=float(head_cfg.get("residual_scale", 1.5)),
            dropout=float(head_cfg.get("dropout", 0.02)),
            gate_bias_init=float(head_cfg.get("gate_bias_init", -2.0)),
        )

    def _build_brush_output_velocity_head(self, brush_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        """Build the optional output-space velocity texture head."""

        head_cfg = brush_cfg.get("output_velocity_head", {})
        if not bool(head_cfg.get("enabled", False)):
            return None
        return OutputVelocityTextureHead(
            in_channels=4,
            hidden_channels=int(head_cfg.get("hidden_channels", 64)),
            kernel_size=int(head_cfg.get("kernel_size", 5)),
            residual_scale=float(head_cfg.get("residual_scale", 0.35)),
            dropout=float(head_cfg.get("dropout", 0.02)),
            gate_bias_init=float(head_cfg.get("gate_bias_init", -2.2)),
        )

    def _build_brush_texture_statistics_head(self, brush_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        """Build the optional texture-statistics brush head."""

        head_cfg = brush_cfg.get("texture_statistics_head", {})
        if not bool(head_cfg.get("enabled", False)):
            return None
        return TextureStatisticsBrushHead(
            feature_channels=int(brush_cfg.get("decoder_hidden_size", 768)),
            hidden_channels=int(head_cfg.get("hidden_channels", 512)),
            basis_channels=int(head_cfg.get("basis_channels", 96)),
            orientation_bins=int(head_cfg.get("orientation_bins", 8)),
            scale_bins=int(head_cfg.get("scale_bins", 3)),
            band_count=int(head_cfg.get("band_count", 3)),
            kernel_size=int(head_cfg.get("kernel_size", 5)),
            residual_init=float(head_cfg.get("residual_init", 0.18)),
            residual_scale=float(head_cfg.get("residual_scale", 1.5)),
            dropout=float(head_cfg.get("dropout", 0.02)),
            gate_bias_init=float(head_cfg.get("gate_bias_init", -2.0)),
        )

    def _build_brush_stroke_field_head(self, brush_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        """Build the optional stroke-field brush head."""

        head_cfg = brush_cfg.get("stroke_field_head", {})
        if not bool(head_cfg.get("enabled", False)):
            return None
        return StrokeFieldBrushHead(
            feature_channels=int(brush_cfg.get("decoder_hidden_size", 768)),
            hidden_channels=int(head_cfg.get("hidden_channels", 384)),
            prototype_count=int(head_cfg.get("prototype_count", 8)),
            renderer_kernel_size=int(head_cfg.get("renderer_kernel_size", 9)),
            render_channels=int(head_cfg.get("render_channels", 8)),
            topk_prototypes=int(head_cfg.get("topk_prototypes", 2)),
            support_gate_enabled=bool(head_cfg.get("support_gate_enabled", False)),
            support_threshold=float(head_cfg.get("support_threshold", 0.20)),
            support_temperature=float(head_cfg.get("support_temperature", 0.08)),
            support_dilation=int(head_cfg.get("support_dilation", 5)),
            support_mix=float(head_cfg.get("support_mix", 0.65)),
            support_hard_mask_enabled=bool(head_cfg.get("support_hard_mask_enabled", False)),
            support_opening_kernel=int(head_cfg.get("support_opening_kernel", 0)),
            residual_scale=float(head_cfg.get("residual_scale", 0.5)),
            dropout=float(head_cfg.get("dropout", 0.02)),
        )

    def _build_brush_support_locked_primitive_head(self, brush_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        """Build the optional support-locked primitive stroke head."""

        head_cfg = brush_cfg.get("support_locked_primitive_head", {})
        if not bool(head_cfg.get("enabled", False)):
            return None
        return SupportLockedPrimitiveBrushHead(
            feature_channels=int(brush_cfg.get("decoder_hidden_size", 768)),
            hidden_channels=int(head_cfg.get("hidden_channels", 512)),
            prototype_count=int(head_cfg.get("prototype_count", 16)),
            render_channels=int(head_cfg.get("render_channels", 12)),
            renderer_kernel_size=int(head_cfg.get("renderer_kernel_size", 13)),
            top_k_anchors=int(head_cfg.get("top_k_anchors", 24)),
            support_threshold=float(head_cfg.get("support_threshold", 0.28)),
            support_mix=float(head_cfg.get("support_mix", 0.80)),
            support_opening_kernel=int(head_cfg.get("support_opening_kernel", 3)),
            support_dilation=int(head_cfg.get("support_dilation", 3)),
            component_aware_enabled=bool(head_cfg.get("component_aware_enabled", False)),
            min_component_area=int(head_cfg.get("min_component_area", 4)),
            large_component_area=int(head_cfg.get("large_component_area", 20)),
            max_anchors_per_component_small=int(head_cfg.get("max_anchors_per_component_small", 1)),
            max_anchors_per_component_large=int(head_cfg.get("max_anchors_per_component_large", 2)),
            subject_safe_exclusion_enabled=bool(head_cfg.get("subject_safe_exclusion_enabled", False)),
            exclusion_logit_bias=float(head_cfg.get("exclusion_logit_bias", -6.0)),
            exclusion_upper_center_weight=float(head_cfg.get("exclusion_upper_center_weight", 1.0)),
            exclusion_dilation=int(head_cfg.get("exclusion_dilation", 3)),
            residual_scale=float(head_cfg.get("residual_scale", 0.50)),
            dropout=float(head_cfg.get("dropout", 0.02)),
        )

    def _build_brush_slot_based_anchor_set_head(self, brush_cfg: Dict[str, Any]) -> Optional[nn.Module]:
        """Build the optional slot-based anchor-set primitive stroke head."""

        head_cfg = brush_cfg.get("slot_based_anchor_set_head", {})
        if not bool(head_cfg.get("enabled", False)):
            return None
        return SlotBasedAnchorSetBrushHead(
            feature_channels=int(brush_cfg.get("decoder_hidden_size", 768)),
            hidden_channels=int(head_cfg.get("hidden_channels", 512)),
            slot_count=int(head_cfg.get("slot_count", 16)),
            prototype_count=int(head_cfg.get("prototype_count", 16)),
            render_channels=int(head_cfg.get("render_channels", 12)),
            support_threshold=float(head_cfg.get("support_threshold", 0.28)),
            support_mix=float(head_cfg.get("support_mix", 0.80)),
            support_opening_kernel=int(head_cfg.get("support_opening_kernel", 3)),
            support_dilation=int(head_cfg.get("support_dilation", 3)),
            min_component_area=int(head_cfg.get("min_component_area", 4)),
            large_component_area=int(head_cfg.get("large_component_area", 20)),
            max_anchors_per_component_small=int(head_cfg.get("max_anchors_per_component_small", 1)),
            max_anchors_per_component_large=int(head_cfg.get("max_anchors_per_component_large", 2)),
            subject_safe_exclusion_enabled=bool(head_cfg.get("subject_safe_exclusion_enabled", True)),
            exclusion_logit_bias=float(head_cfg.get("exclusion_logit_bias", -6.0)),
            exclusion_upper_center_weight=float(head_cfg.get("exclusion_upper_center_weight", 1.0)),
            exclusion_dilation=int(head_cfg.get("exclusion_dilation", 3)),
            valid_anchor_threshold=float(head_cfg.get("valid_anchor_threshold", 0.35)),
            residual_scale=float(head_cfg.get("residual_scale", 0.50)),
            dropout=float(head_cfg.get("dropout", 0.02)),
        )

    def _build_brush_anchored_output_velocity_head(
        self,
        brush_cfg: Dict[str, Any],
    ) -> Optional[nn.Module]:
        """Build the optional structure-anchored output velocity texture head."""

        head_cfg = brush_cfg.get("anchored_output_velocity_head", {})
        if not bool(head_cfg.get("enabled", False)):
            return None
        return AnchoredOutputVelocityTextureHead(
            in_channels=4,
            hidden_channels=int(head_cfg.get("hidden_channels", 96)),
            kernel_size=int(head_cfg.get("kernel_size", 5)),
            residual_scale=float(head_cfg.get("residual_scale", 0.35)),
            dropout=float(head_cfg.get("dropout", 0.02)),
            gate_bias_init=float(head_cfg.get("gate_bias_init", -2.2)),
            anchor_kernel_size=int(head_cfg.get("anchor_kernel_size", 9)),
            anchor_sigma=float(head_cfg.get("anchor_sigma", 2.0)),
            anchor_strength=float(head_cfg.get("anchor_strength", 0.85)),
            anchor_threshold=float(head_cfg.get("anchor_threshold", 0.30)),
            anchor_sharpness=float(head_cfg.get("anchor_sharpness", 12.0)),
            min_edit_scale=float(head_cfg.get("min_edit_scale", 0.10)),
            center_prior_strength=float(head_cfg.get("center_prior_strength", 0.0)),
            center_prior_sigma_x=float(head_cfg.get("center_prior_sigma_x", 0.22)),
            center_prior_sigma_y=float(head_cfg.get("center_prior_sigma_y", 0.18)),
            center_prior_y_offset=float(head_cfg.get("center_prior_y_offset", -0.10)),
        )

    def _freeze_everything(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def _configure_trainable_modules(self, *, skip_language_lora: bool = False) -> None:
        freeze_cfg = self.config.get("freeze", {})
        for module_name in freeze_cfg.get("trainable_base_modules", []):
            if hasattr(self.model, module_name):
                module = getattr(self.model, module_name)
                for parameter in module.parameters():
                    parameter.requires_grad = True

        if self.label_style_encoder is not None:
            for parameter in self.label_style_encoder.parameters():
                parameter.requires_grad = True
        if self.reference_style_encoder is not None:
            for parameter in self.reference_style_encoder.parameters():
                parameter.requires_grad = True
        if self.style_injector is not None:
            for parameter in self.style_injector.parameters():
                parameter.requires_grad = True
        if bool(freeze_cfg.get("train_art_projectors", True)):
            for module in (self.art_projector, self.texture_projector, self.style_classifier_head):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        for module in (
            self.brush_reference_encoder,
            self.brush_condition_injector,
            self.brush_adapter,
            self.brush_feature_map_adapter,
            self.brush_spatial_hf_head,
            self.brush_texture_statistics_head,
            self.brush_stroke_field_head,
            self.brush_support_locked_primitive_head,
            self.brush_slot_based_anchor_set_head,
            self.brush_output_velocity_head,
            self.brush_anchored_output_velocity_head,
        ):
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = True

        if skip_language_lora:
            return

        self._attach_fresh_language_lora()

    def _attach_fresh_language_lora(self) -> None:
        """Wrap the language model with a new LoRA adapter from config."""

        freeze_cfg = self.config.get("freeze", {})
        lora_cfg = freeze_cfg.get("language_lora", {})
        if not bool(lora_cfg.get("enabled", False)):
            return
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=int(lora_cfg.get("r", 32)),
            lora_alpha=int(lora_cfg.get("alpha", 64)),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora)

    def _configure_language_model_runtime(self, *, training: bool) -> None:
        """Apply train/eval-specific LoRA settings after attach or checkpoint restore."""

        freeze_cfg = self.config.get("freeze", {})
        lora_cfg = freeze_cfg.get("language_lora", {})
        if not bool(lora_cfg.get("enabled", False)):
            return
        if training:
            self._restrict_lora_to_last_layers(int(lora_cfg.get("train_last_n_layers", 0)))
        if self.config.get("training", {}).get("gradient_checkpointing", False):
            self.model.language_model.gradient_checkpointing_enable()
            self.model.language_model.enable_input_require_grads()
            self.model.language_model.config.use_cache = False

    def _apply_language_lora_inference_scale(self) -> None:
        """Scale restored LoRA adapters at inference without editing checkpoint weights."""

        freeze_cfg = self.config.get("freeze", {})
        lora_cfg = freeze_cfg.get("language_lora", {})
        if not bool(lora_cfg.get("enabled", False)):
            return
        scale = float(lora_cfg.get("inference_scale", 1.0))
        for module_name, module in self.model.language_model.named_modules():
            scaling = getattr(module, "scaling", None)
            if not isinstance(scaling, dict):
                continue
            for adapter_name, base_value in list(scaling.items()):
                key = (module_name, adapter_name)
                if key not in self._lora_base_scaling:
                    self._lora_base_scaling[key] = float(base_value)
                scaling[adapter_name] = self._lora_base_scaling[key] * scale

    def _restrict_lora_to_last_layers(self, last_n_layers: int) -> None:
        if last_n_layers <= 0:
            return
        backbone = get_lm_backbone(self.model.language_model)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            return
        total_layers = len(layers)
        cutoff = max(total_layers - last_n_layers, 0)
        for name, parameter in self.model.language_model.named_parameters():
            if "lora_" not in name:
                continue
            layer_index = self._extract_layer_index(name)
            if layer_index is not None and layer_index < cutoff:
                parameter.requires_grad = False

    @staticmethod
    def _extract_layer_index(name: str) -> Optional[int]:
        marker = ".layers."
        if marker not in name:
            return None
        suffix = name.split(marker, 1)[1]
        layer_text = suffix.split(".", 1)[0]
        if not layer_text.isdigit():
            return None
        return int(layer_text)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def build_generation_prompt_text(self, prompt: str) -> str:
        """Render the JanusFlow generation prompt text with the official image-start tag.

        JanusFlow's public demo app appends `image_start_tag` and then replaces that
        final token with the timestep embedding at sampling time. Using the generic
        generation tag here leads to a noticeably different token prefix and can
        degrade image fidelity for both the untouched base model and tuned checkpoints.
        """

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

    def tokenize_prompts(self, prompts: List[str]) -> List[torch.LongTensor]:
        """Tokenize rendered prompts for JanusFlow generation."""

        token_rows: List[torch.LongTensor] = []
        for prompt in prompts:
            prompt_text = self.build_generation_prompt_text(prompt)
            token_rows.append(
                torch.tensor(
                    self.processor.tokenizer.encode(prompt_text),
                    dtype=torch.long,
                    device=self.device,
                )
            )
        return token_rows

    def encode_target_images_to_latents(self, target_images: torch.Tensor) -> torch.Tensor:
        """Encode RGB targets into SDXL VAE latents."""

        with torch.no_grad():
            latents = self.vae.encode(target_images.to(device=self.device, dtype=self.dtype)).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def decode_latents_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode SDXL latents back to image tensors in `[0, 1]` space."""

        decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        return decoded.clamp_(-1.0, 1.0).mul(0.5).add(0.5)

    def encode_style_batch(
        self,
        *,
        style_label_ids: torch.Tensor,
        period_label_ids: torch.Tensor,
        medium_label_ids: torch.Tensor,
        reference_style_images: Optional[torch.Tensor] = None,
        has_reference_style_image: Optional[torch.Tensor] = None,
    ) -> Optional[StyleEncoderOutput]:
        """Encode label and reference-image style inputs into a unified output."""

        outputs: List[StyleEncoderOutput] = []
        if self.label_style_encoder is not None:
            outputs.append(
                self.label_style_encoder(
                    style_label_ids=style_label_ids.to(self.device),
                    period_label_ids=period_label_ids.to(self.device),
                    medium_label_ids=medium_label_ids.to(self.device),
                )
            )
        if (
            self.reference_style_encoder is not None
            and reference_style_images is not None
            and has_reference_style_image is not None
        ):
            reference_output = self.reference_style_encoder(
                reference_style_images.to(device=self.device, dtype=self.dtype)
            )
            mask = has_reference_style_image.to(self.device).float().unsqueeze(-1)
            reference_output = StyleEncoderOutput(
                global_embedding=reference_output.global_embedding * mask,
                global_tokens=reference_output.global_tokens * mask.unsqueeze(-1),
                local_tokens=reference_output.local_tokens * mask.unsqueeze(-1),
                style_logits=None,
            )
            outputs.append(reference_output)

        if not outputs:
            return None
        if len(outputs) == 1:
            return outputs[0]

        global_embedding = torch.stack(
            [output.global_embedding for output in outputs], dim=0
        ).mean(dim=0)
        global_tokens = torch.stack(
            [output.global_tokens for output in outputs], dim=0
        ).mean(dim=0)
        local_tokens = torch.stack(
            [output.local_tokens for output in outputs], dim=0
        ).mean(dim=0)
        style_logits = None
        for output in outputs:
            if output.style_logits is not None:
                style_logits = output.style_logits
                break
        return StyleEncoderOutput(
            global_embedding=global_embedding,
            global_tokens=global_tokens,
            local_tokens=local_tokens,
            style_logits=style_logits,
        )

    def encode_brush_batch(
        self,
        *,
        reference_style_images: Optional[torch.Tensor] = None,
        has_reference_style_image: Optional[torch.Tensor] = None,
    ) -> Optional[BrushReferenceOutput]:
        """Encode optional brush-reference images into multi-scale texture tokens."""

        if (
            self.brush_reference_encoder is None
            or reference_style_images is None
            or has_reference_style_image is None
        ):
            return None
        reference_images = reference_style_images.to(device=self.device, dtype=self.dtype)
        reference_images = self._prepare_brush_reference_images(reference_images)
        brush_output = self.brush_reference_encoder(
            reference_images
        )
        return mask_brush_reference_output(
            brush_output,
            has_reference_style_image.to(self.device),
        )

    def _prepare_brush_reference_images(self, reference_images: torch.Tensor) -> torch.Tensor:
        """Optionally keep only high-frequency residuals from brush references."""

        reference_cfg = self.config.get("brush", {}).get("reference_encoder", {})
        high_pass_cfg = reference_cfg.get("high_pass", {})
        if not bool(high_pass_cfg.get("enabled", False)):
            return reference_images
        kernel_size = max(int(high_pass_cfg.get("kernel_size", 15)), 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        gain = float(high_pass_cfg.get("gain", 2.0))
        padding = kernel_size // 2
        # [batch, 3, height, width] -> high-frequency residual at the same shape.
        padded_images = F.pad(reference_images.float(), (padding, padding, padding, padding), mode="reflect")
        low_frequency = F.avg_pool2d(
            padded_images,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        ).to(dtype=reference_images.dtype)
        high_frequency = (reference_images - low_frequency) * gain
        return high_frequency.clamp(-1.0, 1.0)

    def _repeat_cfg_style_output(
        self,
        style_output: Optional[StyleEncoderOutput],
    ) -> Optional[StyleEncoderOutput]:
        if style_output is None:
            return None
        return StyleEncoderOutput(
            global_embedding=torch.cat(
                [style_output.global_embedding, style_output.global_embedding], dim=0
            ),
            global_tokens=torch.cat(
                [style_output.global_tokens, style_output.global_tokens], dim=0
            ),
            local_tokens=torch.cat(
                [style_output.local_tokens, style_output.local_tokens], dim=0
            ),
            style_logits=style_output.style_logits,
        )

    def _repeat_cfg_brush_output(
        self,
        brush_output: Optional[BrushReferenceOutput],
    ) -> Optional[BrushReferenceOutput]:
        if brush_output is None:
            return None
        return repeat_brush_reference_output(brush_output)

    def _apply_brush_modules_to_decoder_tokens(
        self,
        decoder_tokens: torch.Tensor,
        brush_output: Optional[BrushReferenceOutput],
    ) -> torch.Tensor:
        """Apply decoder-local brush modules with CFG-aware conditioning.

        The default is to apply brush modules only to the conditioned CFG half.
        Applying the same local texture signal to both conditioned and
        unconditioned rows can make the signal common-mode and suppress it in
        the final CFG combination.
        """

        if self.brush_condition_injector is None and self.brush_adapter is None:
            return decoder_tokens

        brush_cfg = self.config.get("brush", {})
        if bool(brush_cfg.get("condition_unconditioned", False)):
            cfg_brush_output = self._repeat_cfg_brush_output(brush_output)
            if self.brush_condition_injector is not None:
                decoder_tokens = self.brush_condition_injector(decoder_tokens, cfg_brush_output)
            if self.brush_adapter is not None:
                decoder_tokens = self.brush_adapter(decoder_tokens, cfg_brush_output)
            return decoder_tokens

        conditioned_tokens, unconditioned_tokens = decoder_tokens.chunk(2, dim=0)
        if self.brush_condition_injector is not None:
            conditioned_tokens = self.brush_condition_injector(conditioned_tokens, brush_output)
        if self.brush_adapter is not None:
            conditioned_tokens = self.brush_adapter(conditioned_tokens, brush_output)
        return torch.cat([conditioned_tokens, unconditioned_tokens], dim=0)

    def _apply_brush_modules_to_feature_map(
        self,
        decoder_feature_map: torch.Tensor,
        brush_output: Optional[BrushReferenceOutput],
        *,
        apply_brush_modules: bool = True,
        subject_exclusion_hints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply local brush modules directly on decoder feature maps."""

        if not apply_brush_modules:
            return decoder_feature_map

        has_feature_map_module = any(
            module is not None
            for module in (
                self.brush_feature_map_adapter,
                self.brush_spatial_hf_head,
                self.brush_texture_statistics_head,
                self.brush_stroke_field_head,
                self.brush_support_locked_primitive_head,
                self.brush_slot_based_anchor_set_head,
            )
        )
        if not has_feature_map_module:
            return decoder_feature_map

        brush_cfg = self.config.get("brush", {})
        if bool(brush_cfg.get("condition_unconditioned", False)):
            cfg_brush_output = self._repeat_cfg_brush_output(brush_output)
            if self.brush_feature_map_adapter is not None:
                decoder_feature_map = self.brush_feature_map_adapter(decoder_feature_map, cfg_brush_output)
            if self.brush_spatial_hf_head is not None:
                decoder_feature_map = self.brush_spatial_hf_head(decoder_feature_map)
            if self.brush_texture_statistics_head is not None:
                decoder_feature_map = self.brush_texture_statistics_head(decoder_feature_map)
            if self.brush_stroke_field_head is not None:
                decoder_feature_map = self.brush_stroke_field_head(decoder_feature_map)
            if self.brush_support_locked_primitive_head is not None:
                repeated_hints = None
                if subject_exclusion_hints is not None:
                    repeated_hints = torch.cat([subject_exclusion_hints, subject_exclusion_hints], dim=0)
                decoder_feature_map = self.brush_support_locked_primitive_head(
                    decoder_feature_map,
                    subject_exclusion_hints=repeated_hints,
                )
            if self.brush_slot_based_anchor_set_head is not None:
                repeated_hints = None
                if subject_exclusion_hints is not None:
                    repeated_hints = torch.cat([subject_exclusion_hints, subject_exclusion_hints], dim=0)
                decoder_feature_map = self.brush_slot_based_anchor_set_head(
                    decoder_feature_map,
                    subject_exclusion_hints=repeated_hints,
                )
            return decoder_feature_map

        conditioned_map, unconditioned_map = decoder_feature_map.chunk(2, dim=0)
        if self.brush_feature_map_adapter is not None:
            conditioned_map = self.brush_feature_map_adapter(conditioned_map, brush_output)
        if self.brush_spatial_hf_head is not None:
            conditioned_map = self.brush_spatial_hf_head(conditioned_map)
        if self.brush_texture_statistics_head is not None:
            conditioned_map = self.brush_texture_statistics_head(conditioned_map)
        if self.brush_stroke_field_head is not None:
            conditioned_map = self.brush_stroke_field_head(conditioned_map)
        if self.brush_support_locked_primitive_head is not None:
            conditioned_map = self.brush_support_locked_primitive_head(
                conditioned_map,
                subject_exclusion_hints=subject_exclusion_hints,
            )
        if self.brush_slot_based_anchor_set_head is not None:
            conditioned_map = self.brush_slot_based_anchor_set_head(
                conditioned_map,
                subject_exclusion_hints=subject_exclusion_hints,
            )
        return torch.cat([conditioned_map, unconditioned_map], dim=0)

    def _apply_brush_modules_to_velocity(
        self,
        pred_velocity: torch.Tensor,
        *,
        apply_brush_modules: bool = True,
    ) -> torch.Tensor:
        """Apply optional brush modules directly on predicted velocity outputs."""

        if not apply_brush_modules:
            return pred_velocity

        active_velocity_head = self.brush_anchored_output_velocity_head or self.brush_output_velocity_head
        if active_velocity_head is None:
            return pred_velocity

        brush_cfg = self.config.get("brush", {})
        if bool(brush_cfg.get("condition_unconditioned", False)):
            return active_velocity_head(pred_velocity)

        conditioned_velocity, unconditioned_velocity = pred_velocity.chunk(2, dim=0)
        conditioned_velocity = active_velocity_head(conditioned_velocity)
        return torch.cat([conditioned_velocity, unconditioned_velocity], dim=0)

    def brush_runtime_summary(self) -> Dict[str, Any]:
        """Return concise diagnostics for enabled brush modules."""

        adapter_gate = None
        adapter_scale = None
        feature_map_gate = None
        feature_map_scale = None
        spatial_hf_gate = None
        spatial_hf_scale = None
        texture_statistics_gate = None
        texture_statistics_scale = None
        output_velocity_scale = None
        anchored_output_velocity_scale = None
        if self.brush_adapter is not None:
            adapter_gate = float(self.brush_adapter.residual_gate.detach().float().cpu().item())
            adapter_scale = float(getattr(self.brush_adapter, "residual_scale", 1.0))
        if self.brush_feature_map_adapter is not None:
            feature_map_gate = float(self.brush_feature_map_adapter.residual_gate.detach().float().cpu().item())
            feature_map_scale = float(getattr(self.brush_feature_map_adapter, "residual_scale", 1.0))
        if self.brush_spatial_hf_head is not None:
            spatial_hf_gate = float(self.brush_spatial_hf_head.residual_gate.detach().float().cpu().item())
            spatial_hf_scale = float(getattr(self.brush_spatial_hf_head, "residual_scale", 1.0))
        if self.brush_texture_statistics_head is not None:
            texture_statistics_gate = float(
                self.brush_texture_statistics_head.residual_gate.detach().float().cpu().item()
            )
            texture_statistics_scale = float(
                getattr(self.brush_texture_statistics_head, "residual_scale", 1.0)
            )
        if self.brush_output_velocity_head is not None:
            output_velocity_scale = float(getattr(self.brush_output_velocity_head, "residual_scale", 1.0))
        if self.brush_anchored_output_velocity_head is not None:
            anchored_output_velocity_scale = float(
                getattr(self.brush_anchored_output_velocity_head, "residual_scale", 1.0)
            )
        return {
            "brush_adapter_enabled": self.brush_adapter is not None,
            "brush_reference_encoder_enabled": self.brush_reference_encoder is not None,
            "brush_condition_injector_enabled": self.brush_condition_injector is not None,
            "brush_feature_map_adapter_enabled": self.brush_feature_map_adapter is not None,
            "brush_spatial_hf_head_enabled": self.brush_spatial_hf_head is not None,
            "brush_texture_statistics_head_enabled": self.brush_texture_statistics_head is not None,
            "brush_stroke_field_head_enabled": self.brush_stroke_field_head is not None,
            "brush_support_locked_primitive_head_enabled": self.brush_support_locked_primitive_head is not None,
            "brush_slot_based_anchor_set_head_enabled": self.brush_slot_based_anchor_set_head is not None,
            "brush_output_velocity_head_enabled": self.brush_output_velocity_head is not None,
            "brush_anchored_output_velocity_head_enabled": self.brush_anchored_output_velocity_head is not None,
            "brush_condition_unconditioned": bool(
                self.config.get("brush", {}).get("condition_unconditioned", False)
            ),
            "brush_adapter_residual_gate": adapter_gate,
            "brush_adapter_residual_scale": adapter_scale,
            "brush_adapter_effective_gate": (
                adapter_gate * adapter_scale
                if adapter_gate is not None and adapter_scale is not None
                else None
            ),
            "brush_feature_map_residual_gate": feature_map_gate,
            "brush_feature_map_residual_scale": feature_map_scale,
            "brush_feature_map_effective_gate": (
                feature_map_gate * feature_map_scale
                if feature_map_gate is not None and feature_map_scale is not None
                else None
            ),
            "brush_spatial_hf_residual_gate": spatial_hf_gate,
            "brush_spatial_hf_residual_scale": spatial_hf_scale,
            "brush_spatial_hf_effective_gate": (
                spatial_hf_gate * spatial_hf_scale
                if spatial_hf_gate is not None and spatial_hf_scale is not None
                else None
            ),
            "brush_texture_statistics_residual_gate": texture_statistics_gate,
            "brush_texture_statistics_residual_scale": texture_statistics_scale,
            "brush_texture_statistics_effective_gate": (
                texture_statistics_gate * texture_statistics_scale
                if texture_statistics_gate is not None and texture_statistics_scale is not None
                else None
            ),
            "brush_support_locked_primitive_residual_scale": (
                getattr(self.brush_support_locked_primitive_head, "residual_scale", None)
                if self.brush_support_locked_primitive_head is not None
                else None
            ),
            "brush_slot_based_anchor_set_residual_scale": (
                getattr(self.brush_slot_based_anchor_set_head, "residual_scale", None)
                if self.brush_slot_based_anchor_set_head is not None
                else None
            ),
            "brush_output_velocity_residual_scale": output_velocity_scale,
            "brush_anchored_output_velocity_residual_scale": anchored_output_velocity_scale,
        }

    def brush_gate_regularization_losses(self) -> Dict[str, torch.Tensor]:
        """Return gate regularization losses for active gated brush heads."""

        losses: Dict[str, torch.Tensor] = {}
        for module in (
            self.brush_spatial_hf_head,
            self.brush_texture_statistics_head,
            self.brush_stroke_field_head,
            self.brush_support_locked_primitive_head,
            self.brush_slot_based_anchor_set_head,
            self.brush_output_velocity_head,
            self.brush_anchored_output_velocity_head,
        ):
            if module is None:
                continue
            gate_l1 = module.gate_l1_loss()
            gate_tv = module.gate_tv_loss()
            protected_gate_l1 = getattr(module, "protected_gate_l1_loss", lambda: None)()
            if gate_l1 is not None:
                losses["gate_l1"] = losses.get("gate_l1", gate_l1.new_zeros(())) + gate_l1
            if gate_tv is not None:
                losses["gate_tv"] = losses.get("gate_tv", gate_tv.new_zeros(())) + gate_tv
            if protected_gate_l1 is not None:
                losses["protected_gate_l1"] = losses.get(
                    "protected_gate_l1",
                    protected_gate_l1.new_zeros(()),
                ) + protected_gate_l1
            basis_usage_entropy = getattr(module, "basis_usage_entropy_loss", lambda: None)()
            if basis_usage_entropy is not None:
                losses["basis_usage_entropy"] = losses.get(
                    "basis_usage_entropy",
                    basis_usage_entropy.new_zeros(()),
                ) + basis_usage_entropy
        return losses

    def latest_texture_statistics(self) -> Optional[Any]:
        """Return cached texture statistics from the active statistics head."""

        if self.brush_texture_statistics_head is None:
            return None
        return self.brush_texture_statistics_head.latest_statistics()

    def latest_stroke_fields(self) -> Optional[Any]:
        """Return cached stroke-field predictions from the active stroke head."""

        if self.brush_slot_based_anchor_set_head is not None:
            return self.brush_slot_based_anchor_set_head.latest_output()
        if self.brush_support_locked_primitive_head is not None:
            return self.brush_support_locked_primitive_head.latest_output()
        if self.brush_stroke_field_head is None:
            return None
        return self.brush_stroke_field_head.latest_output()

    def _build_cfg_llm_inputs(
        self,
        prompt_ids: List[torch.LongTensor],
        z_tokens: torch.Tensor,
        t_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build JanusFlow CFG inputs exactly like the public demo.

        Shapes:
        - `prompt_ids[i]`: `[prompt_len]`, where the final token is `image_start_tag`
        - output `llm_embeds`: `[2 * batch, prefix_len + 1 + latent_tokens, hidden]`
        - output `attention_mask`: `[2 * batch, prefix_len + 1 + latent_tokens]`
        """

        token_embedding = self.model.language_model.get_input_embeddings()
        batch_size = len(prompt_ids)
        max_prompt_len = max(int(prompt_id.shape[0]) for prompt_id in prompt_ids)

        # Match the demo CFG pattern: conditioned rows keep the whole prompt,
        # unconditioned rows keep token 0 and replace the rest with pad tokens.
        prompt_token_batch = torch.full(
            (batch_size * 2, max_prompt_len),
            fill_value=int(self.processor.pad_id),
            device=self.device,
            dtype=torch.long,
        )
        for row_index, prompt_id in enumerate(prompt_ids):
            prompt_len = int(prompt_id.shape[0])
            prompt_token_batch[row_index, :prompt_len] = prompt_id
            prompt_token_batch[row_index + batch_size, :prompt_len] = prompt_id
            if not (self.training_mode and not bool(self.config.get("training", {}).get("use_cfg_training", True))):
                prompt_token_batch[row_index + batch_size, 1:prompt_len] = int(self.processor.pad_id)

        # The final `<image_start_tag>` token is replaced by `t_embed`, just like the demo.
        prefix_embeds = token_embedding(prompt_token_batch)[:, :-1, :].to(dtype=self.dtype)
        prefix_len = prefix_embeds.shape[1]
        latent_token_count = z_tokens.shape[1]
        hidden_size = z_tokens.shape[2]

        attention_mask = torch.ones(
            (batch_size * 2, prefix_len + 1 + latent_token_count),
            device=self.device,
            dtype=torch.long,
        )
        for row_index, prompt_id in enumerate(prompt_ids):
            prompt_len = int(prompt_id.shape[0])
            conditioned_prefix_len = max(prompt_len - 1, 1)
            attention_mask[row_index, conditioned_prefix_len:prefix_len] = 0
            if not (self.training_mode and not bool(self.config.get("training", {}).get("use_cfg_training", True))):
                attention_mask[row_index + batch_size, 1:conditioned_prefix_len] = 0
                attention_mask[row_index + batch_size, conditioned_prefix_len:prefix_len] = 0
            else:
                attention_mask[row_index + batch_size, conditioned_prefix_len:prefix_len] = 0

        llm_embeds = torch.cat([prefix_embeds, t_embed.unsqueeze(1), z_tokens], dim=1)
        return llm_embeds, attention_mask

    def predict_velocity_from_prompt_ids(
        self,
        *,
        prompt_ids: List[torch.LongTensor],
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        style_output: Optional[StyleEncoderOutput] = None,
        brush_output: Optional[BrushReferenceOutput] = None,
        apply_brush_modules: bool = True,
        subject_exclusion_hints: Optional[torch.Tensor] = None,
    ) -> VelocityPredictionOutput:
        """Run one JanusFlow-Art denoiser pass on interpolated latents.

        Shapes:
        - `latents`: `[batch, 4, latent_h, latent_w]`
        - `timesteps`: `[batch]` in the normalized `[0, 1]` interval
        """

        cfg_style_output = self._repeat_cfg_style_output(style_output)

        latent_input = torch.cat([latents, latents], dim=0).to(device=self.device, dtype=self.dtype)
        scaled_timesteps = (timesteps * 1000.0).to(device=self.device, dtype=self.dtype)
        scaled_timesteps = torch.cat([scaled_timesteps, scaled_timesteps], dim=0)

        z_feature_map, t_embed, hs = self.model.vision_gen_enc_model(latent_input, scaled_timesteps)
        # `[2 * batch, 768, 24, 24]` -> `[2 * batch, 576, 768]`
        z_tokens = z_feature_map.view(z_feature_map.shape[0], z_feature_map.shape[1], -1).permute(0, 2, 1)
        z_tokens = self.model.vision_gen_enc_aligner(z_tokens)
        if self.style_injector is not None and cfg_style_output is not None:
            injector_cfg = self.config.get("conditioning", {}).get("injector", {})
            if injector_cfg.get("use_global", False):
                z_tokens, t_embed = self.style_injector.inject_generation_context(
                    z_tokens,
                    t_embed,
                    global_tokens=cfg_style_output.global_tokens,
                    global_embedding=cfg_style_output.global_embedding,
                )

        llm_embeds, attention_mask = self._build_cfg_llm_inputs(
            prompt_ids,
            z_tokens,
            t_embed,
        )
        # Use the wrapped language model directly so PEFT/LoRA adapters
        # participate in both training and sampling. Unwrapping to the raw
        # decoder backbone is still useful for layer introspection, but it can
        # bypass adapter logic during forward passes.
        outputs = self.model.language_model(
            inputs_embeds=llm_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        llm_hidden_states = get_output_last_hidden_state(outputs)
        llm_latent_tokens = llm_hidden_states[:, -z_tokens.shape[1] :, :]
        decoder_tokens = self.model.vision_gen_dec_aligner(
            self.model.vision_gen_dec_aligner_norm(llm_latent_tokens)
        )
        if self.style_injector is not None and cfg_style_output is not None:
            injector_cfg = self.config.get("conditioning", {}).get("injector", {})
            if injector_cfg.get("use_local", False):
                decoder_tokens = self.style_injector.inject_decoder_tokens(
                    decoder_tokens,
                    local_tokens=cfg_style_output.local_tokens,
                )
        decoder_tokens = self._apply_brush_modules_to_decoder_tokens(
            decoder_tokens,
            brush_output,
        ) if apply_brush_modules else decoder_tokens

        side = int(math.sqrt(decoder_tokens.shape[1]))
        # `[2 * batch, 576, 768]` -> `[2 * batch, 768, 24, 24]`
        decoder_feature_map = decoder_tokens.reshape(
            decoder_tokens.shape[0],
            side,
            side,
            decoder_tokens.shape[-1],
        ).permute(0, 3, 1, 2)
        decoder_feature_map = self._apply_brush_modules_to_feature_map(
            decoder_feature_map,
            brush_output,
            apply_brush_modules=apply_brush_modules,
            subject_exclusion_hints=subject_exclusion_hints,
        )
        pred_velocity = self.model.vision_gen_dec_model(
            decoder_feature_map,
            [tensor for tensor in hs],
            t_embed,
        )
        pred_velocity = self._apply_brush_modules_to_velocity(
            pred_velocity,
            apply_brush_modules=apply_brush_modules,
        )

        conditioned_velocity, unconditioned_velocity = pred_velocity.chunk(2, dim=0)
        conditioned_llm_tokens, _ = llm_latent_tokens.chunk(2, dim=0)
        conditioned_decoder_tokens, _ = decoder_tokens.chunk(2, dim=0)
        conditioned_decoder_feature_map, _ = decoder_feature_map.chunk(2, dim=0)
        return VelocityPredictionOutput(
            conditioned_velocity=conditioned_velocity,
            unconditioned_velocity=unconditioned_velocity,
            conditioned_llm_tokens=conditioned_llm_tokens,
            conditioned_decoder_tokens=conditioned_decoder_tokens,
            conditioned_decoder_feature_map=conditioned_decoder_feature_map,
            style_output=style_output,
            brush_output=brush_output,
        )

    def project_art_embedding(self, conditioned_llm_tokens: torch.Tensor) -> torch.Tensor:
        """Project conditioned LLM image tokens into the style-embedding space."""

        pooled = conditioned_llm_tokens.mean(dim=1)
        return self.art_projector(pooled)

    def project_texture_embedding(self, conditioned_decoder_tokens: torch.Tensor) -> torch.Tensor:
        """Project conditioned decoder tokens into the style-embedding space."""

        pooled = conditioned_decoder_tokens.mean(dim=1)
        return self.texture_projector(pooled)

    @torch.inference_mode()
    def sample_images(
        self,
        *,
        prompts: List[str],
        style_label_ids: torch.Tensor,
        period_label_ids: torch.Tensor,
        medium_label_ids: torch.Tensor,
        reference_style_images: Optional[torch.Tensor],
        has_reference_style_image: Optional[torch.Tensor],
        seed: int,
        num_inference_steps: int,
        cfg_weight: float,
        image_size: int,
        subject_exclusion_hints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample images for a prompt batch using JanusFlow rectified-flow ODE integration."""

        set_seed(seed)
        batch_size = len(prompts)
        prompt_ids = self.tokenize_prompts(prompts)
        style_output = self.encode_style_batch(
            style_label_ids=style_label_ids,
            period_label_ids=period_label_ids,
            medium_label_ids=medium_label_ids,
            reference_style_images=reference_style_images,
            has_reference_style_image=has_reference_style_image,
        )
        brush_output = self.encode_brush_batch(
            reference_style_images=reference_style_images,
            has_reference_style_image=has_reference_style_image,
        )
        latent_size = image_size // 8
        z = torch.randn(
            (batch_size, 4, latent_size, latent_size),
            device=self.device,
            dtype=self.dtype,
        )
        dt = 1.0 / max(num_inference_steps, 1)
        for step in range(num_inference_steps):
            timestep_value = torch.full(
                (batch_size,),
                fill_value=float(step) / max(num_inference_steps, 1),
                device=self.device,
                dtype=self.dtype,
            )
            prediction = self.predict_velocity_from_prompt_ids(
                prompt_ids=prompt_ids,
                latents=z,
                timesteps=timestep_value,
                style_output=style_output,
                brush_output=brush_output,
                subject_exclusion_hints=subject_exclusion_hints,
            )
            velocity = cfg_weight * prediction.conditioned_velocity - (cfg_weight - 1.0) * prediction.unconditioned_velocity
            z = z + dt * velocity
        return self.decode_latents_to_images(z)

    def _restore_checkpoint(self, checkpoint_path: str, *, training: bool) -> None:
        checkpoint_dir = Path(checkpoint_path)
        language_model_dir = checkpoint_dir / "language_model"
        if (language_model_dir / "adapter_config.json").exists():
            from peft import PeftModel

            self.model.language_model = PeftModel.from_pretrained(
                self.model.language_model,
                str(language_model_dir),
                is_trainable=training,
            )
        art_state_path = checkpoint_dir / "art_state.pt"
        if art_state_path.exists():
            payload = torch.load(art_state_path, map_location="cpu")
            strict = self.checkpoint_strict
            self._load_optional_state_dict(self.model.vision_gen_enc_aligner, payload.get("vision_gen_enc_aligner"), strict=strict)
            self._load_optional_state_dict(self.model.vision_gen_dec_aligner_norm, payload.get("vision_gen_dec_aligner_norm"), strict=strict)
            self._load_optional_state_dict(self.model.vision_gen_dec_aligner, payload.get("vision_gen_dec_aligner"), strict=strict)
            self._load_optional_state_dict(self.label_style_encoder, payload.get("label_style_encoder"), strict=strict)
            self._load_optional_state_dict(self.reference_style_encoder, payload.get("reference_style_encoder"), strict=strict)
            self._load_optional_state_dict(self.style_injector, payload.get("style_injector"), strict=strict)
            self._load_optional_state_dict(self.brush_reference_encoder, payload.get("brush_reference_encoder"), strict=strict)
            self._load_optional_state_dict(self.brush_condition_injector, payload.get("brush_condition_injector"), strict=strict)
            self._load_optional_state_dict(self.brush_adapter, payload.get("brush_adapter"), strict=strict)
            self._load_optional_state_dict(self.brush_feature_map_adapter, payload.get("brush_feature_map_adapter"), strict=strict)
            self._load_optional_state_dict(self.brush_spatial_hf_head, payload.get("brush_spatial_hf_head"), strict=strict)
            self._load_optional_state_dict(
                self.brush_texture_statistics_head,
                payload.get("brush_texture_statistics_head"),
                strict=strict,
            )
            self._load_optional_state_dict(
                self.brush_stroke_field_head,
                payload.get("brush_stroke_field_head"),
                strict=strict,
            )
            self._load_optional_state_dict(
                self.brush_support_locked_primitive_head,
                payload.get("brush_support_locked_primitive_head"),
                strict=strict,
            )
            self._load_optional_state_dict(
                self.brush_slot_based_anchor_set_head,
                payload.get("brush_slot_based_anchor_set_head"),
                strict=strict,
            )
            self._load_optional_state_dict(self.brush_output_velocity_head, payload.get("brush_output_velocity_head"), strict=strict)
            self._load_optional_state_dict(
                self.brush_anchored_output_velocity_head,
                payload.get("brush_anchored_output_velocity_head"),
                strict=strict,
            )
            self._load_optional_state_dict(self.art_projector, payload.get("art_projector"), strict=strict)
            self._load_optional_state_dict(self.texture_projector, payload.get("texture_projector"), strict=strict)
            self._load_optional_state_dict(self.style_classifier_head, payload.get("style_classifier_head"), strict=strict)

    @staticmethod
    def _load_optional_state_dict(
        module: Optional[nn.Module],
        state_dict: Optional[Dict[str, Any]],
        *,
        strict: bool = True,
    ) -> None:
        if module is None or state_dict is None:
            return
        if strict:
            module.load_state_dict(state_dict)
            return
        current_state = module.state_dict()
        compatible_state = {}
        mismatched_keys = []
        for key, value in state_dict.items():
            current_value = current_state.get(key)
            if current_value is None:
                continue
            if tuple(current_value.shape) != tuple(value.shape):
                mismatched_keys.append(key)
                continue
            compatible_state[key] = value
        missing_keys, unexpected_keys = module.load_state_dict(compatible_state, strict=False)
        if mismatched_keys or missing_keys or unexpected_keys:
            print(
                {
                    "event": "partial_checkpoint_restore",
                    "module": module.__class__.__name__,
                    "loaded_keys": len(compatible_state),
                    "mismatched_keys": mismatched_keys,
                    "missing_keys": list(missing_keys),
                    "unexpected_keys": list(unexpected_keys),
                }
            )

    def build_optimizer(self) -> torch.optim.Optimizer:
        """Build an AdamW optimizer for the currently trainable parameters."""

        train_cfg = self.config.get("training", {})
        lora_parameters: List[torch.nn.Parameter] = []
        other_parameters: List[torch.nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if "lora_" in name:
                lora_parameters.append(parameter)
            else:
                other_parameters.append(parameter)

        parameter_groups = []
        if lora_parameters:
            parameter_groups.append(
                {
                    "params": lora_parameters,
                    "lr": float(train_cfg.get("learning_rate", 2.0e-5)),
                    "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
                }
            )
        if other_parameters:
            parameter_groups.append(
                {
                    "params": other_parameters,
                    "lr": float(train_cfg.get("generation_learning_rate", train_cfg.get("learning_rate", 2.0e-5))),
                    "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
                }
            )
        return torch.optim.AdamW(parameter_groups)

    def save_checkpoint(self, output_dir: str, *, step: Optional[int] = None, config_path: Optional[str] = None) -> str:
        """Save the current JanusFlow-Art checkpoint."""

        checkpoint_name = f"checkpoint-{step}" if step is not None else "final_checkpoint"
        checkpoint_dir = Path(output_dir) / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.processor.save_pretrained(str(checkpoint_dir / "processor"))
        if hasattr(self.model.language_model, "peft_config") and hasattr(self.model.language_model, "save_pretrained"):
            self.model.language_model.save_pretrained(str(checkpoint_dir / "language_model"))

        payload = {
            "vision_gen_enc_aligner": self.model.vision_gen_enc_aligner.state_dict(),
            "vision_gen_dec_aligner_norm": self.model.vision_gen_dec_aligner_norm.state_dict(),
            "vision_gen_dec_aligner": self.model.vision_gen_dec_aligner.state_dict(),
            "label_style_encoder": self.label_style_encoder.state_dict() if self.label_style_encoder is not None else None,
            "reference_style_encoder": self.reference_style_encoder.state_dict() if self.reference_style_encoder is not None else None,
            "style_injector": self.style_injector.state_dict() if self.style_injector is not None else None,
            "brush_reference_encoder": self.brush_reference_encoder.state_dict() if self.brush_reference_encoder is not None else None,
            "brush_condition_injector": self.brush_condition_injector.state_dict() if self.brush_condition_injector is not None else None,
            "brush_adapter": self.brush_adapter.state_dict() if self.brush_adapter is not None else None,
            "brush_feature_map_adapter": self.brush_feature_map_adapter.state_dict() if self.brush_feature_map_adapter is not None else None,
            "brush_spatial_hf_head": self.brush_spatial_hf_head.state_dict() if self.brush_spatial_hf_head is not None else None,
            "brush_texture_statistics_head": (
                self.brush_texture_statistics_head.state_dict()
                if self.brush_texture_statistics_head is not None
                else None
            ),
            "brush_stroke_field_head": (
                self.brush_stroke_field_head.state_dict()
                if self.brush_stroke_field_head is not None
                else None
            ),
            "brush_support_locked_primitive_head": (
                self.brush_support_locked_primitive_head.state_dict()
                if self.brush_support_locked_primitive_head is not None
                else None
            ),
            "brush_slot_based_anchor_set_head": (
                self.brush_slot_based_anchor_set_head.state_dict()
                if self.brush_slot_based_anchor_set_head is not None
                else None
            ),
            "brush_output_velocity_head": self.brush_output_velocity_head.state_dict() if self.brush_output_velocity_head is not None else None,
            "brush_anchored_output_velocity_head": (
                self.brush_anchored_output_velocity_head.state_dict()
                if self.brush_anchored_output_velocity_head is not None
                else None
            ),
            "art_projector": self.art_projector.state_dict(),
            "texture_projector": self.texture_projector.state_dict(),
            "style_classifier_head": self.style_classifier_head.state_dict(),
            "label_vocabs": self.label_vocabs,
        }
        torch.save(payload, checkpoint_dir / "art_state.pt")
        summary = {
            "step": step,
            "parameter_summary": count_parameters(self),
            "brush_runtime": self.brush_runtime_summary(),
            "checkpoint_dir": str(checkpoint_dir),
            "config_path": config_path,
        }
        with open(checkpoint_dir / "checkpoint_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        return str(checkpoint_dir)


def load_checkpoint_state_dict_if_available(checkpoint_path: Optional[str], name: str) -> Optional[Dict[str, Any]]:
    """Load an optimizer or scheduler state dict when it exists."""

    if not checkpoint_path:
        return None
    path = Path(checkpoint_path) / name
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write JSON to disk with UTF-8 encoding."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write JSONL rows to disk."""

    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tensor_to_pil_images(images: torch.Tensor) -> List["Image.Image"]:
    """Convert `[batch, 3, height, width]` image tensors into PIL images."""

    from PIL import Image

    array = images.detach().cpu().clamp(0.0, 1.0).mul(255.0).byte().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(item) for item in array]


def save_image_grid(images: List["Image.Image"], path: str, *, columns: int = 2) -> None:
    """Save a simple PIL image grid."""

    from PIL import Image

    if not images:
        raise ValueError("Cannot save an empty image grid.")
    columns = max(columns, 1)
    rows = math.ceil(len(images) / columns)
    width, height = images[0].size
    grid = Image.new("RGB", (columns * width, rows * height))
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * height
        grid.paste(image, (x, y))
    grid.save(path)
