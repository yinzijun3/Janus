"""Sampling entrypoint for JanusFlow-Art."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from finetune.janusflow_art_config import apply_common_cli_overrides, load_yaml_config
from finetune.janusflow_art_data import (
    ArtPromptJsonlDataset,
    JanusFlowArtPromptCollator,
    build_label_vocabs,
)
from finetune.janusflow_art_runtime import (
    JanusFlowArtPipeline,
    save_image_grid,
    save_json,
    save_jsonl,
    tensor_to_pil_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images with JanusFlow-Art.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--lora-scale", type=float, default=None)
    return parser.parse_args()


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
    sampling_cfg = config["sampling"]
    output_dir = os.path.join(
        config["experiment"]["output_root"],
        sampling_cfg.get("output_subdir", "sampled"),
    )
    os.makedirs(output_dir, exist_ok=True)
    print(
        {
            "event": "sampling_start",
            "config": args.config,
            "checkpoint": args.checkpoint,
            "output_dir": output_dir,
        }
    )

    pipeline = JanusFlowArtPipeline(
        config,
        label_vocabs=None,
        checkpoint_path=args.checkpoint,
        training=False,
    )
    prompt_dataset = ArtPromptJsonlDataset(sampling_cfg["prompt_file"])
    if len(prompt_dataset) == 0:
        raise ValueError("Prompt file is empty; cannot sample JanusFlow-Art outputs.")
    collator = JanusFlowArtPromptCollator(
        prompt_template=config["data"].get("prompt_template", "conservative"),
        reference_image_size=int(config["data"].get("reference_image_size", 224)),
        label_vocabs=pipeline.label_vocabs or build_label_vocabs(prompt_dataset.records),
    )
    batch = collator(list(prompt_dataset.records))
    save_json(os.path.join(output_dir, "sampling_config.json"), config)

    sample_rows = []
    for seed in sampling_cfg.get("seeds", [42]):
        images = pipeline.sample_images(
            prompts=batch["rendered_prompts"],
            style_label_ids=batch["style_label_ids"],
            period_label_ids=batch["period_label_ids"],
            medium_label_ids=batch["medium_label_ids"],
            reference_style_images=batch["reference_style_images"],
            has_reference_style_image=batch["has_reference_style_image"],
            subject_exclusion_hints=batch["subject_exclusion_hints"].to(device=pipeline.device, dtype=pipeline.dtype),
            seed=int(seed),
            num_inference_steps=int(sampling_cfg.get("num_inference_steps", 30)),
            cfg_weight=float(sampling_cfg.get("cfg_weight", 2.0)),
            image_size=int(config["data"].get("image_size", 384)),
        )
        pil_images = tensor_to_pil_images(images)
        seed_dir = Path(output_dir) / f"seed_{int(seed):05d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(pil_images):
            image_path = seed_dir / f"{index:03d}.png"
            image.save(image_path)
            sample_rows.append(
                {
                    "seed": int(seed),
                    "index": index,
                    "request_id": batch["request_ids"][index],
                    "prompt": batch["rendered_prompts"][index],
                    "reference_style_image": batch["reference_image_paths"][index],
                    "image_path": str(image_path),
                }
            )
        if sampling_cfg.get("save_grid", True):
            save_image_grid(
                pil_images,
                str(seed_dir / "grid.png"),
                columns=min(2, max(1, len(pil_images))),
            )

    save_jsonl(os.path.join(output_dir, "samples.jsonl"), sample_rows)
    if sampling_cfg.get("save_prompt_log", True):
        save_jsonl(
            os.path.join(output_dir, "prompt_log.jsonl"),
            [
                {
                    "request_id": request_id,
                    "prompt": prompt,
                }
                for request_id, prompt in zip(batch["request_ids"], batch["rendered_prompts"])
            ],
        )
    print(
        {
            "event": "sampling_complete",
            "output_dir": output_dir,
            "num_prompts": len(batch["rendered_prompts"]),
            "num_seeds": len(sampling_cfg.get("seeds", [42])),
        }
    )


if __name__ == "__main__":
    main()
