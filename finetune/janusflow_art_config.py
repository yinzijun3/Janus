"""Configuration helpers for JanusFlow-Art."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def deep_merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two mapping objects and return a new dictionary."""

    merged = copy.deepcopy(base)
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {path} must load into a mapping.")
    return payload


def set_nested_value(payload: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested dictionary value from a dotted key path."""

    current = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = current.get(part)
        if not isinstance(node, dict):
            node = {}
            current[part] = node
        current = node
    current[parts[-1]] = value


def apply_common_cli_overrides(
    config: Dict[str, Any],
    *,
    max_steps: Optional[int] = None,
    output_root: Optional[str] = None,
    checkpoint: Optional[str] = None,
    init_checkpoint: Optional[str] = None,
    prompt_file: Optional[str] = None,
    num_samples: Optional[int] = None,
    model_path: Optional[str] = None,
    vae_path: Optional[str] = None,
    skip_final_eval: Optional[bool] = None,
    lora_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """Apply common CLI overrides used by train, sample, and eval entrypoints."""

    updated = copy.deepcopy(config)
    if max_steps is not None:
        set_nested_value(updated, "training.max_steps", int(max_steps))
    if output_root is not None:
        set_nested_value(updated, "experiment.output_root", output_root)
    if checkpoint is not None:
        set_nested_value(updated, "checkpoint.resume_from", checkpoint)
    if init_checkpoint is not None:
        set_nested_value(updated, "training.init_from_checkpoint", init_checkpoint)
    if prompt_file is not None:
        set_nested_value(updated, "sampling.prompt_file", prompt_file)
        set_nested_value(updated, "evaluation.prompt_file", prompt_file)
    if num_samples is not None:
        set_nested_value(updated, "evaluation.num_samples", int(num_samples))
    if model_path is not None:
        set_nested_value(updated, "model.model_path", model_path)
    if vae_path is not None:
        set_nested_value(updated, "model.vae_path", vae_path)
    if skip_final_eval is not None:
        set_nested_value(updated, "logging.run_final_eval", not bool(skip_final_eval))
    if lora_scale is not None:
        set_nested_value(updated, "freeze.language_lora.inference_scale", float(lora_scale))
    return updated
