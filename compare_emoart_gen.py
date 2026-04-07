import argparse
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import PIL.Image
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from finetune.emoart_generation import (
    ART_TEXTURE_FIELD_CHOICES,
    ART_TEXTURE_MODE_CHOICES,
    DEFAULT_ART_TEXTURE_FIELDS,
    DEFAULT_ART_TEXTURE_MODE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_CHOICES,
    build_prompt_from_record,
    load_jsonl,
    preprocess_image_for_vq,
)
from janus.models import MultiModalityCausalLM, VLChatProcessor
from train_emoart_gen_lora import (
    build_generation_inputs,
    get_dtype,
    get_lm_backbone,
    load_generation_modules,
    set_seed,
)


DEFAULT_MODEL_PATH = os.environ.get(
    "JANUS_MODEL_PATH",
    "/root/autodl-tmp/hf_cache/hub/models--deepseek-ai--Janus-Pro-1B/snapshots/960ab33191f61342a4c60ae74d8dc356a39fafcb",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Janus generation outputs before and after a LoRA adapter.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--parallel-size", type=int, default=1)
    parser.add_argument("--cfg-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--image-preprocess-mode", choices=["pad", "crop"], default="crop")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-token-num-per-image", type=int, default=576)
    parser.add_argument("--sample-strategy", choices=["greedy", "sample"], default="greedy")
    parser.add_argument(
        "--prompt-template",
        choices=PROMPT_TEMPLATE_CHOICES,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template variant used during generation.",
    )
    parser.add_argument(
        "--art-texture-mode",
        choices=ART_TEXTURE_MODE_CHOICES,
        default=DEFAULT_ART_TEXTURE_MODE,
        help="Optional painterly texture emphasis for evaluation prompts.",
    )
    parser.add_argument(
        "--art-texture-fields",
        choices=ART_TEXTURE_FIELD_CHOICES,
        default=DEFAULT_ART_TEXTURE_FIELDS,
        help="Which metadata-derived texture fields are allowed into the evaluation prompt.",
    )
    parser.add_argument("--use-clipscore", action="store_true")
    parser.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def load_model(model_path: str, dtype: torch.dtype) -> tuple[MultiModalityCausalLM, VLChatProcessor]:
    processor = VLChatProcessor.from_pretrained(model_path)
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = model.to(dtype).cuda().eval()
    return model, processor


def apply_adapter(model: MultiModalityCausalLM, adapter_path: str) -> MultiModalityCausalLM:
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        adapter_path,
        is_trainable=False,
    )
    loaded_modules = load_generation_modules(model, adapter_path)
    print({"loaded_generation_modules": loaded_modules})
    model.eval()
    return model


def format_generation_prompt(processor: VLChatProcessor, prompt: str) -> str:
    conversation = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    return sft_format + processor.image_start_tag


def maybe_load_clipscore(model_name: str):
    try:
        from transformers import AutoProcessor, CLIPModel
    except Exception:
        return None

    try:
        clip_processor = AutoProcessor.from_pretrained(model_name)
        clip_model = CLIPModel.from_pretrained(model_name).cuda().eval()
        return clip_model, clip_processor
    except Exception as exc:
        print({"clipscore_unavailable": str(exc)})
        return None


@torch.inference_mode()
def compute_clipscore(clip_bundle, image: PIL.Image.Image, prompt: str) -> Optional[float]:
    if clip_bundle is None:
        return None
    clip_model, clip_processor = clip_bundle
    batch = clip_processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    batch = {key: value.cuda() for key, value in batch.items()}
    outputs = clip_model(**batch)
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    return float((image_embeds * text_embeds).sum(dim=-1).item())


@torch.inference_mode()
def generate_images(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    prompt: str,
    parallel_size: int,
    cfg_weight: float,
    temperature: float,
    image_token_num_per_image: int,
    image_size: int,
    patch_size: int,
    seed: int,
    sample_strategy: str,
):
    set_seed(seed)
    device = next(model.parameters()).device
    prompt_text = format_generation_prompt(processor, prompt)
    input_ids = processor.tokenizer.encode(prompt_text)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.long, device=device)
    for index in range(parallel_size * 2):
        tokens[index, :] = input_ids
        if index % 2 != 0 and len(input_ids) > 2:
            tokens[index, 1:-1] = processor.pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.long, device=device)
    lm_backbone = get_lm_backbone(model.language_model)

    outputs = None
    for step in range(image_token_num_per_image):
        outputs = lm_backbone(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if step != 0 else None,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = model.gen_head(hidden_states[:, -1, :])
        cond_logits = logits[0::2, :]
        uncond_logits = logits[1::2, :]
        logits = uncond_logits + cfg_weight * (cond_logits - uncond_logits)

        if sample_strategy == "sample":
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated_tokens[:, step] = next_token.squeeze(-1)
        next_token = torch.cat([next_token, next_token], dim=1).view(-1)
        image_embeds = model.prepare_gen_img_embeds(next_token)
        inputs_embeds = image_embeds.unsqueeze(1)

    decoded = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, image_size // patch_size, image_size // patch_size],
    )
    decoded = decoded.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    decoded = np.clip((decoded + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    images = [PIL.Image.fromarray(decoded[index]) for index in range(parallel_size)]
    token_rows = [generated_tokens[index].detach().cpu() for index in range(parallel_size)]
    return images, token_rows


def image_mae(image_a: PIL.Image.Image, image_b: PIL.Image.Image, image_size: int) -> float:
    arr_a = np.asarray(image_a.resize((image_size, image_size), PIL.Image.BICUBIC), dtype=np.float32)
    arr_b = np.asarray(image_b.resize((image_size, image_size), PIL.Image.BICUBIC), dtype=np.float32)
    return float(np.abs(arr_a - arr_b).mean())


def image_psnr(image_a: PIL.Image.Image, image_b: PIL.Image.Image, image_size: int) -> float:
    arr_a = np.asarray(image_a.resize((image_size, image_size), PIL.Image.BICUBIC), dtype=np.float32)
    arr_b = np.asarray(image_b.resize((image_size, image_size), PIL.Image.BICUBIC), dtype=np.float32)
    mse = float(np.mean((arr_a - arr_b) ** 2))
    if mse == 0.0:
        return 99.0
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def _grayscale_array(image: PIL.Image.Image, image_size: int) -> np.ndarray:
    return np.asarray(
        image.resize((image_size, image_size), PIL.Image.BICUBIC).convert("L"),
        dtype=np.float32,
    )


def image_gradient_energy(image: PIL.Image.Image, image_size: int) -> float:
    gray = _grayscale_array(image, image_size)
    grad_x = np.diff(gray, axis=1)
    grad_y = np.diff(gray, axis=0)
    return float(np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y)))


def image_laplacian_variance(image: PIL.Image.Image, image_size: int) -> float:
    gray = _grayscale_array(image, image_size)
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(lap))


def image_high_frequency_energy(image: PIL.Image.Image, image_size: int) -> float:
    gray = _grayscale_array(image, image_size)
    gray = gray - gray.mean()
    spectrum = np.abs(np.fft.rfft2(gray))
    rows, cols = spectrum.shape
    y = np.linspace(0.0, 1.0, rows, endpoint=False)[:, None]
    x = np.linspace(0.0, 1.0, cols, endpoint=False)[None, :]
    radius = np.sqrt(x**2 + y**2)
    high_band = spectrum[radius >= 0.35]
    if high_band.size == 0:
        return 0.0
    return float(high_band.mean())


def texture_metrics(image: PIL.Image.Image, image_size: int) -> Dict[str, float]:
    return {
        "gradient_energy": image_gradient_energy(image, image_size),
        "laplacian_variance": image_laplacian_variance(image, image_size),
        "high_frequency_energy": image_high_frequency_energy(image, image_size),
    }


def texture_abs_errors(
    image: PIL.Image.Image,
    reference_image: PIL.Image.Image,
    image_size: int,
) -> Dict[str, float]:
    image_metrics = texture_metrics(image, image_size)
    reference_metrics = texture_metrics(reference_image, image_size)
    return {
        key: abs(image_metrics[key] - reference_metrics[key])
        for key in image_metrics
    }


@torch.inference_mode()
def encode_reference_tokens(
    model: MultiModalityCausalLM,
    image_path: str,
    image_size: int,
    image_preprocess_mode: str,
) -> torch.LongTensor:
    with PIL.Image.open(image_path) as image:
        image_tensor = preprocess_image_for_vq(
            image.convert("RGB"),
            image_size=image_size,
            mode=image_preprocess_mode,
        ).unsqueeze(0)
    image_tensor = image_tensor.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    _, _, (_, _, token_ids) = model.gen_vision_model.encode(image_tensor)
    return token_ids.view(-1).detach().cpu().long()


def token_match_ratio(predicted: torch.LongTensor, reference: torch.LongTensor, prefix: int = 0) -> float:
    if prefix > 0:
        predicted = predicted[:prefix]
        reference = reference[:prefix]
    return float((predicted == reference).float().mean().item())


@torch.inference_mode()
def reference_token_nll(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    prompt: str,
    reference_tokens: torch.LongTensor,
) -> float:
    device = next(model.parameters()).device
    prompt_text = format_generation_prompt(processor, prompt)
    prompt_ids = torch.tensor(processor.tokenizer.encode(prompt_text), dtype=torch.long, device=device)
    reference_tokens = reference_tokens.to(device)
    inputs_embeds, attention_mask, target_positions, target_ids = build_generation_inputs(
        model=model,
        prompt_ids=[prompt_ids],
        image_token_ids=reference_tokens.unsqueeze(0),
    )
    lm_backbone = get_lm_backbone(model.language_model)
    outputs = lm_backbone(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state[0]
    start = target_positions[0]
    stop = start + target_ids[0].numel()
    logits = model.gen_head(hidden_states[start:stop, :]).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids[0].unsqueeze(-1)).squeeze(-1)
    return float((-token_log_probs.mean()).item())


def write_markdown_report(path: str, summary: Dict[str, object], rows: List[Dict[str, object]]) -> None:
    lines = ["# Janus EmoArt Generation Comparison", ""]
    lines.append("## Summary")
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Samples")
    for row in rows:
        lines.append(f"### {row['index']:02d} - {row['request_id']}")
        lines.append(f"- Prompt: {row['prompt']}")
        lines.append(f"- Before: `{row['before_path']}`")
        lines.append(f"- After: `{row['after_path']}`")
        lines.append(f"- Target: `{row['target_path']}`")
        lines.append(
            f"- Token match before/after: {row['token_match_before']:.4f} / {row['token_match_after']:.4f}"
        )
        lines.append(
            f"- Prefix64 match before/after: {row['token_match_prefix64_before']:.4f} / {row['token_match_prefix64_after']:.4f}"
        )
        lines.append(
            f"- MAE before/after: {row['reference_mae_before']:.4f} / {row['reference_mae_after']:.4f}"
        )
        lines.append(
            f"- PSNR before/after: {row['reference_psnr_before']:.4f} / {row['reference_psnr_after']:.4f}"
        )
        lines.append(
            f"- Reference token NLL before/after: {row['reference_token_nll_before']:.4f} / {row['reference_token_nll_after']:.4f}"
        )
        lines.append(
            f"- Gradient energy abs error before/after: {row['reference_gradient_energy_abs_error_before']:.4f} / {row['reference_gradient_energy_abs_error_after']:.4f}"
        )
        lines.append(
            f"- Laplacian abs error before/after: {row['reference_laplacian_variance_abs_error_before']:.4f} / {row['reference_laplacian_variance_abs_error_after']:.4f}"
        )
        lines.append(
            f"- High-frequency abs error before/after: {row['reference_high_frequency_energy_abs_error_before']:.4f} / {row['reference_high_frequency_energy_abs_error_after']:.4f}"
        )
        if row.get("clipscore_before") is not None:
            lines.append(
                f"- CLIPScore before/after: {row['clipscore_before']:.4f} / {row['clipscore_after']:.4f}"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    before_dir = os.path.join(args.output_dir, "before")
    after_dir = os.path.join(args.output_dir, "after")
    target_dir = os.path.join(args.output_dir, "target")
    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    dtype = get_dtype(args.dtype)
    base_model, processor = load_model(args.model_path, dtype)
    adapted_model, _ = load_model(args.model_path, dtype)
    adapted_model = apply_adapter(adapted_model, args.adapter_path)
    clip_bundle = maybe_load_clipscore(args.clip_model_name) if args.use_clipscore else None

    records = load_jsonl(args.manifest_path)
    random.Random(args.seed).shuffle(records)
    selected_records = records[: args.num_samples]

    comparison_rows: List[Dict[str, object]] = []
    for index, record in enumerate(selected_records):
        prompt = build_prompt_from_record(
            record,
            prompt_template=args.prompt_template,
            art_texture_mode=args.art_texture_mode,
            art_texture_fields=args.art_texture_fields,
        )
        row_seed = args.seed + index

        base_images, base_tokens = generate_images(
            model=base_model,
            processor=processor,
            prompt=prompt,
            parallel_size=args.parallel_size,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            image_token_num_per_image=args.image_token_num_per_image,
            image_size=args.image_size,
            patch_size=args.patch_size,
            seed=row_seed,
            sample_strategy=args.sample_strategy,
        )
        adapted_images, adapted_tokens = generate_images(
            model=adapted_model,
            processor=processor,
            prompt=prompt,
            parallel_size=args.parallel_size,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            image_token_num_per_image=args.image_token_num_per_image,
            image_size=args.image_size,
            patch_size=args.patch_size,
            seed=row_seed,
            sample_strategy=args.sample_strategy,
        )
        base_image = base_images[0]
        adapted_image = adapted_images[0]
        base_token_row = base_tokens[0]
        adapted_token_row = adapted_tokens[0]
        reference_tokens = encode_reference_tokens(
            base_model,
            record["image_path"],
            args.image_size,
            args.image_preprocess_mode,
        )

        with PIL.Image.open(record["image_path"]) as reference_image:
            reference_rgb = reference_image.convert("RGB")

        before_path = os.path.join(before_dir, f"{index:02d}.png")
        after_path = os.path.join(after_dir, f"{index:02d}.png")
        target_path = os.path.join(target_dir, f"{index:02d}.png")
        base_image.save(before_path)
        adapted_image.save(after_path)
        reference_rgb.save(target_path)

        before_mae = image_mae(base_image, reference_rgb, args.image_size)
        after_mae = image_mae(adapted_image, reference_rgb, args.image_size)
        before_after_mae = image_mae(base_image, adapted_image, args.image_size)
        before_psnr = image_psnr(base_image, reference_rgb, args.image_size)
        after_psnr = image_psnr(adapted_image, reference_rgb, args.image_size)
        base_texture_metrics = texture_metrics(base_image, args.image_size)
        adapted_texture_metrics = texture_metrics(adapted_image, args.image_size)
        reference_texture_metrics = texture_metrics(reference_rgb, args.image_size)
        base_texture_errors = texture_abs_errors(base_image, reference_rgb, args.image_size)
        adapted_texture_errors = texture_abs_errors(adapted_image, reference_rgb, args.image_size)
        token_before = token_match_ratio(base_token_row, reference_tokens)
        token_after = token_match_ratio(adapted_token_row, reference_tokens)
        token_prefix64_before = token_match_ratio(base_token_row, reference_tokens, prefix=64)
        token_prefix64_after = token_match_ratio(adapted_token_row, reference_tokens, prefix=64)
        token_prefix128_before = token_match_ratio(base_token_row, reference_tokens, prefix=128)
        token_prefix128_after = token_match_ratio(adapted_token_row, reference_tokens, prefix=128)
        reference_nll_before = reference_token_nll(base_model, processor, prompt, reference_tokens)
        reference_nll_after = reference_token_nll(adapted_model, processor, prompt, reference_tokens)
        clipscore_before = compute_clipscore(clip_bundle, base_image, prompt)
        clipscore_after = compute_clipscore(clip_bundle, adapted_image, prompt)
        comparison_rows.append(
            {
                "index": index,
                "request_id": record["request_id"],
                "prompt": prompt,
                "image_path": record["image_path"],
                "before_path": before_path,
                "after_path": after_path,
                "target_path": target_path,
                "reference_mae_before": before_mae,
                "reference_mae_after": after_mae,
                "reference_psnr_before": before_psnr,
                "reference_psnr_after": after_psnr,
                "before_after_mae": before_after_mae,
                "token_match_before": token_before,
                "token_match_after": token_after,
                "token_match_prefix64_before": token_prefix64_before,
                "token_match_prefix64_after": token_prefix64_after,
                "token_match_prefix128_before": token_prefix128_before,
                "token_match_prefix128_after": token_prefix128_after,
                "reference_token_nll_before": reference_nll_before,
                "reference_token_nll_after": reference_nll_after,
                "gradient_energy_before": base_texture_metrics["gradient_energy"],
                "gradient_energy_after": adapted_texture_metrics["gradient_energy"],
                "gradient_energy_reference": reference_texture_metrics["gradient_energy"],
                "laplacian_variance_before": base_texture_metrics["laplacian_variance"],
                "laplacian_variance_after": adapted_texture_metrics["laplacian_variance"],
                "laplacian_variance_reference": reference_texture_metrics["laplacian_variance"],
                "high_frequency_energy_before": base_texture_metrics["high_frequency_energy"],
                "high_frequency_energy_after": adapted_texture_metrics["high_frequency_energy"],
                "high_frequency_energy_reference": reference_texture_metrics["high_frequency_energy"],
                "reference_gradient_energy_abs_error_before": base_texture_errors["gradient_energy"],
                "reference_gradient_energy_abs_error_after": adapted_texture_errors["gradient_energy"],
                "reference_laplacian_variance_abs_error_before": base_texture_errors["laplacian_variance"],
                "reference_laplacian_variance_abs_error_after": adapted_texture_errors["laplacian_variance"],
                "reference_high_frequency_energy_abs_error_before": base_texture_errors["high_frequency_energy"],
                "reference_high_frequency_energy_abs_error_after": adapted_texture_errors["high_frequency_energy"],
                "clipscore_before": clipscore_before,
                "clipscore_after": clipscore_after,
                "improved_vs_reference": after_mae < before_mae,
                "improved_token_match": token_after > token_before,
                "improved_psnr": after_psnr > before_psnr,
                "improved_reference_nll": reference_nll_after < reference_nll_before,
                "improved_gradient_energy": adapted_texture_errors["gradient_energy"] < base_texture_errors["gradient_energy"],
                "improved_laplacian_variance": adapted_texture_errors["laplacian_variance"] < base_texture_errors["laplacian_variance"],
                "improved_high_frequency_energy": adapted_texture_errors["high_frequency_energy"] < base_texture_errors["high_frequency_energy"],
            }
        )

    output_jsonl = os.path.join(args.output_dir, "comparison.jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in comparison_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_samples": len(comparison_rows),
        "adapter_path": args.adapter_path,
        "avg_reference_mae_before": float(np.mean([row["reference_mae_before"] for row in comparison_rows])),
        "avg_reference_mae_after": float(np.mean([row["reference_mae_after"] for row in comparison_rows])),
        "avg_reference_psnr_before": float(np.mean([row["reference_psnr_before"] for row in comparison_rows])),
        "avg_reference_psnr_after": float(np.mean([row["reference_psnr_after"] for row in comparison_rows])),
        "avg_token_match_before": float(np.mean([row["token_match_before"] for row in comparison_rows])),
        "avg_token_match_after": float(np.mean([row["token_match_after"] for row in comparison_rows])),
        "avg_prefix64_match_before": float(np.mean([row["token_match_prefix64_before"] for row in comparison_rows])),
        "avg_prefix64_match_after": float(np.mean([row["token_match_prefix64_after"] for row in comparison_rows])),
        "avg_reference_token_nll_before": float(np.mean([row["reference_token_nll_before"] for row in comparison_rows])),
        "avg_reference_token_nll_after": float(np.mean([row["reference_token_nll_after"] for row in comparison_rows])),
        "avg_reference_gradient_energy_abs_error_before": float(
            np.mean([row["reference_gradient_energy_abs_error_before"] for row in comparison_rows])
        ),
        "avg_reference_gradient_energy_abs_error_after": float(
            np.mean([row["reference_gradient_energy_abs_error_after"] for row in comparison_rows])
        ),
        "avg_reference_laplacian_variance_abs_error_before": float(
            np.mean([row["reference_laplacian_variance_abs_error_before"] for row in comparison_rows])
        ),
        "avg_reference_laplacian_variance_abs_error_after": float(
            np.mean([row["reference_laplacian_variance_abs_error_after"] for row in comparison_rows])
        ),
        "avg_reference_high_frequency_energy_abs_error_before": float(
            np.mean([row["reference_high_frequency_energy_abs_error_before"] for row in comparison_rows])
        ),
        "avg_reference_high_frequency_energy_abs_error_after": float(
            np.mean([row["reference_high_frequency_energy_abs_error_after"] for row in comparison_rows])
        ),
        "avg_before_after_mae": float(np.mean([row["before_after_mae"] for row in comparison_rows])),
        "changed_outputs": int(sum(row["before_after_mae"] > 0.0 for row in comparison_rows)),
        "improved_vs_reference_count": int(sum(bool(row["improved_vs_reference"]) for row in comparison_rows)),
        "improved_token_match_count": int(sum(bool(row["improved_token_match"]) for row in comparison_rows)),
        "improved_psnr_count": int(sum(bool(row["improved_psnr"]) for row in comparison_rows)),
        "improved_reference_nll_count": int(sum(bool(row["improved_reference_nll"]) for row in comparison_rows)),
        "improved_gradient_energy_count": int(sum(bool(row["improved_gradient_energy"]) for row in comparison_rows)),
        "improved_laplacian_variance_count": int(
            sum(bool(row["improved_laplacian_variance"]) for row in comparison_rows)
        ),
        "improved_high_frequency_energy_count": int(
            sum(bool(row["improved_high_frequency_energy"]) for row in comparison_rows)
        ),
        "sample_strategy": args.sample_strategy,
        "prompt_template": args.prompt_template,
        "art_texture_mode": args.art_texture_mode,
    }
    clip_before_values = [row["clipscore_before"] for row in comparison_rows if row["clipscore_before"] is not None]
    clip_after_values = [row["clipscore_after"] for row in comparison_rows if row["clipscore_after"] is not None]
    if clip_before_values and clip_after_values:
        summary["avg_clipscore_before"] = float(np.mean(clip_before_values))
        summary["avg_clipscore_after"] = float(np.mean(clip_after_values))
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_markdown_report(os.path.join(args.output_dir, "report.md"), summary, comparison_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
