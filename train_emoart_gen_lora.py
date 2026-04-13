import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from finetune.emoart_generation import (
    ART_TEXTURE_FIELD_CHOICES,
    ART_TEXTURE_MODE_CHOICES,
    DEFAULT_PATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_ART_TEXTURE_FIELDS,
    DEFAULT_ART_TEXTURE_MODE,
    DEFAULT_PROMPT_TEMPLATE,
    EmoArtGenerationJsonlDataset,
    JanusGenerationDataCollator,
    PROMPT_TEMPLATE_CHOICES,
    compute_image_token_count,
    validate_image_generation_geometry,
)
from janus.models import MultiModalityCausalLM, VLChatProcessor


DEFAULT_MODEL_PATH = os.environ.get(
    "JANUS_MODEL_PATH",
    "/root/autodl-tmp/hf_cache/hub/models--deepseek-ai--Janus-Pro-1B/snapshots/960ab33191f61342a4c60ae74d8dc356a39fafcb",
)
GENERATION_MODULE_MODE_CHOICES = ("full", "head_only", "frozen")


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Janus image generation on EmoArt-5k.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-name", default="printblue/EmoArt-5k")
    parser.add_argument("--train-data", required=True, help="Path to the generation training manifest JSONL.")
    parser.add_argument("--val-data", required=True, help="Path to the generation validation manifest JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--image-preprocess-mode", choices=["pad", "crop"], default="crop")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--generation-learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument(
        "--train-generation-modules",
        nargs="+",
        default=["gen_head", "gen_aligner", "gen_embed"],
        help="Generation-specific modules to keep trainable alongside LoRA.",
    )
    parser.add_argument(
        "--generation-module-mode",
        choices=GENERATION_MODULE_MODE_CHOICES,
        default="head_only",
        help="Conservative default: keep generation-side training limited unless a broader sweep is intentional.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--scheduler-type", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--prompt-template",
        choices=PROMPT_TEMPLATE_CHOICES,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template variant used at train/eval time.",
    )
    parser.add_argument(
        "--art-texture-mode",
        choices=ART_TEXTURE_MODE_CHOICES,
        default=DEFAULT_ART_TEXTURE_MODE,
        help="Optional painterly texture emphasis injected into prompts.",
    )
    parser.add_argument(
        "--art-texture-fields",
        choices=ART_TEXTURE_FIELD_CHOICES,
        default=DEFAULT_ART_TEXTURE_FIELDS,
        help="Which metadata-derived texture fields are allowed into the enhanced prompt.",
    )
    parser.add_argument(
        "--art-texture-prob",
        type=float,
        default=0.0,
        help="Probability of swapping a batch sample to the texture-enhanced prompt variant.",
    )
    parser.add_argument(
        "--retention-manifest",
        default=None,
        help="Optional JSONL of prompt-only generic prompts used for retention-aware distillation.",
    )
    parser.add_argument(
        "--retention-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the retention KL loss. Set > 0 to enable retention-aware training.",
    )
    parser.add_argument(
        "--retention-batch-size",
        type=int,
        default=2,
        help="Prompt batch size for retention updates.",
    )
    parser.add_argument(
        "--retention-num-workers",
        type=int,
        default=0,
        help="Data loader workers for prompt-only retention manifest.",
    )
    parser.add_argument(
        "--retention-token-prefix",
        type=int,
        default=64,
        help="Number of teacher-generated image tokens used for retention distillation.",
    )
    parser.add_argument(
        "--retention-temperature",
        type=float,
        default=1.0,
        help="Temperature used for student/teacher KL distillation.",
    )
    parser.add_argument(
        "--retention-cfg-weight",
        type=float,
        default=5.0,
        help="CFG weight used when generating teacher token prefixes for retention prompts.",
    )
    parser.add_argument(
        "--retention-sample-strategy",
        choices=["greedy", "sample"],
        default="greedy",
        help="Sampling strategy for teacher prefix generation on retention prompts.",
    )
    parser.add_argument(
        "--retention-seed",
        type=int,
        default=1234,
        help="Base seed for retention prompt token-prefix generation.",
    )
    parser.add_argument(
        "--rehearsal-manifest",
        default=None,
        help="Optional JSONL of prompt-only generic/art prompts used for mixed-domain rehearsal.",
    )
    parser.add_argument(
        "--rehearsal-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the mixed-domain rehearsal CE loss.",
    )
    parser.add_argument(
        "--rehearsal-batch-size",
        type=int,
        default=8,
        help="Prompt batch size for mixed-domain rehearsal updates.",
    )
    parser.add_argument(
        "--rehearsal-num-workers",
        type=int,
        default=0,
        help="Data loader workers for prompt-only rehearsal manifest.",
    )
    parser.add_argument(
        "--rehearsal-token-count",
        type=int,
        default=32,
        help="Number of teacher-generated image tokens used as pseudo targets for rehearsal.",
    )
    parser.add_argument(
        "--rehearsal-cfg-weight",
        type=float,
        default=5.0,
        help="CFG weight used for teacher generation on rehearsal prompts.",
    )
    parser.add_argument(
        "--rehearsal-sample-strategy",
        choices=["greedy", "sample"],
        default="greedy",
        help="Sampling strategy for teacher generation on rehearsal prompts.",
    )
    parser.add_argument(
        "--rehearsal-seed",
        type=int,
        default=4321,
        help="Base seed for teacher generation on rehearsal prompts.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PromptOnlyJsonlDataset(Dataset):
    def __init__(self, manifest_path: str):
        self.records = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


class JanusPromptOnlyCollator:
    def __init__(self, processor):
        self.processor = processor

    def build_prompt_text(self, prompt: str) -> str:
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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_ids: List[torch.Tensor] = []
        prompts: List[str] = []
        record_ids: List[str] = []
        for feature in features:
            prompt = feature["prompt"]
            prompt_text = self.build_prompt_text(prompt)
            prompt_ids.append(torch.tensor(self.processor.tokenizer.encode(prompt_text), dtype=torch.long))
            prompts.append(prompt)
            record_ids.append(feature.get("id") or feature.get("request_id") or f"prompt_{len(record_ids):04d}")
        return {
            "prompt_ids": prompt_ids,
            "prompts": prompts,
            "record_ids": record_ids,
        }


def get_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_lm_backbone(language_model):
    backbone = language_model
    if hasattr(language_model, "get_base_model"):
        backbone = language_model.get_base_model()
    if hasattr(backbone, "model"):
        return backbone.model
    return backbone


def count_parameters(model) -> Dict[str, float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": trainable / max(total, 1),
    }


def resolve_generation_module_names(args) -> List[str]:
    if args.generation_module_mode == "frozen":
        return []
    if args.generation_module_mode == "head_only":
        return ["gen_head"]
    module_names = [name for name in args.train_generation_modules if name and name.lower() != "none"]
    return module_names


def mark_generation_modules_trainable(model: MultiModalityCausalLM, module_names: Iterable[str]) -> List[str]:
    enabled = []
    for module_name in module_names:
        if not hasattr(model, module_name):
            continue
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = True
        enabled.append(module_name)
    return enabled


def save_generation_modules(model: MultiModalityCausalLM, output_dir: str, module_names: Iterable[str]) -> None:
    state = {}
    names = []
    for module_name in module_names:
        if hasattr(model, module_name):
            state[module_name] = getattr(model, module_name).state_dict()
            names.append(module_name)
    torch.save({"module_names": names, "state_dict": state}, os.path.join(output_dir, "generation_modules.pt"))


def load_generation_modules(model: MultiModalityCausalLM, adapter_dir: str) -> List[str]:
    path = os.path.join(adapter_dir, "generation_modules.pt")
    if not os.path.exists(path):
        return []
    payload = torch.load(path, map_location="cpu")
    loaded = []
    for module_name, state_dict in payload.get("state_dict", {}).items():
        if hasattr(model, module_name):
            getattr(model, module_name).load_state_dict(state_dict)
            loaded.append(module_name)
    return loaded


def build_optimizer(model, args):
    lora_parameters = []
    generation_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora_parameters.append(parameter)
        else:
            generation_parameters.append(parameter)

    parameter_groups = []
    if lora_parameters:
        parameter_groups.append(
            {
                "params": lora_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    if generation_parameters:
        parameter_groups.append(
            {
                "params": generation_parameters,
                "lr": args.generation_learning_rate,
                "weight_decay": args.weight_decay,
            }
        )

    return AdamW(parameter_groups)


def build_model(args, dtype):
    processor = VLChatProcessor.from_pretrained(args.model_path)
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = model.to(dtype).cuda()

    for parameter in model.parameters():
        parameter.requires_grad = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if args.resume_from_checkpoint:
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            args.resume_from_checkpoint,
            is_trainable=True,
        )
        loaded_modules = load_generation_modules(model, args.resume_from_checkpoint)
        if loaded_modules:
            print({"loaded_generation_modules": loaded_modules})
    else:
        model.language_model = get_peft_model(model.language_model, lora_config)

    generation_module_names = resolve_generation_module_names(args)
    enabled_generation_modules = mark_generation_modules_trainable(
        model,
        module_names=generation_module_names,
    )

    if args.gradient_checkpointing:
        model.language_model.gradient_checkpointing_enable()
        model.language_model.enable_input_require_grads()
        model.language_model.config.use_cache = False

    stats = count_parameters(model)
    print(
        {
            "enabled_generation_modules": enabled_generation_modules,
            "total_parameters": stats["total_parameters"],
            "trainable_parameters": stats["trainable_parameters"],
            "trainable_ratio": round(stats["trainable_ratio"], 6),
        }
    )
    return model, processor, enabled_generation_modules


def build_teacher_model(model_path: str, dtype: torch.dtype) -> MultiModalityCausalLM:
    teacher: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    teacher = teacher.to(dtype).cuda().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def move_batch_to_device(batch, device, dtype):
    batch["pixel_values"] = batch["pixel_values"].to(device=device, dtype=dtype)
    batch["prompt_ids"] = [prompt_ids.to(device) for prompt_ids in batch["prompt_ids"]]
    return batch


def move_prompt_batch_to_device(batch, device):
    batch["prompt_ids"] = [prompt_ids.to(device) for prompt_ids in batch["prompt_ids"]]
    return batch


def encode_images_to_tokens(model: MultiModalityCausalLM, pixel_values: torch.Tensor) -> torch.LongTensor:
    with torch.no_grad():
        _, _, (_, _, image_token_ids) = model.gen_vision_model.encode(pixel_values)
    image_token_ids = image_token_ids.view(pixel_values.size(0), -1).long()
    return image_token_ids


def build_generation_inputs(
    model: MultiModalityCausalLM,
    prompt_ids: List[torch.LongTensor],
    image_token_ids: torch.LongTensor,
):
    token_embedding = model.language_model.get_input_embeddings()
    batch_size = len(prompt_ids)
    hidden_size = token_embedding.weight.shape[1]
    device = token_embedding.weight.device

    per_sample_embeds: List[torch.Tensor] = []
    target_positions: List[int] = []
    target_ids: List[torch.LongTensor] = []
    max_seq_len = 0

    for sample_index in range(batch_size):
        prompt_id = prompt_ids[sample_index]
        code_ids = image_token_ids[sample_index]

        prompt_embeds = token_embedding(prompt_id)
        previous_code_ids = code_ids[:-1]
        previous_img_embeds = model.prepare_gen_img_embeds(previous_code_ids)
        sequence_embeds = torch.cat([prompt_embeds, previous_img_embeds], dim=0)

        per_sample_embeds.append(sequence_embeds)
        target_positions.append(prompt_embeds.size(0) - 1)
        target_ids.append(code_ids)
        max_seq_len = max(max_seq_len, sequence_embeds.size(0))

    inputs_embeds = token_embedding.weight.new_zeros((batch_size, max_seq_len, hidden_size))
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)

    for index, sequence_embeds in enumerate(per_sample_embeds):
        seq_len = sequence_embeds.size(0)
        inputs_embeds[index, :seq_len] = sequence_embeds
        attention_mask[index, :seq_len] = 1

    return inputs_embeds, attention_mask, target_positions, target_ids


def compute_generation_logits(
    model: MultiModalityCausalLM,
    prompt_ids: List[torch.LongTensor],
    image_token_ids: torch.LongTensor,
):
    inputs_embeds, attention_mask, target_positions, target_ids = build_generation_inputs(
        model=model,
        prompt_ids=prompt_ids,
        image_token_ids=image_token_ids,
    )
    lm_backbone = get_lm_backbone(model.language_model)
    outputs = lm_backbone(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state

    logits_list = []
    label_list = []
    for batch_index, labels in enumerate(target_ids):
        start = target_positions[batch_index]
        stop = start + labels.size(0)
        logits = model.gen_head(hidden_states[batch_index, start:stop, :])
        logits_list.append(logits)
        label_list.append(labels)

    flat_logits = torch.cat(logits_list, dim=0).float()
    flat_labels = torch.cat(label_list, dim=0)
    return flat_logits, flat_labels


def compute_generation_loss_and_metrics(
    model: MultiModalityCausalLM,
    prompt_ids: List[torch.LongTensor],
    image_token_ids: torch.LongTensor,
):
    flat_logits, flat_labels = compute_generation_logits(
        model=model,
        prompt_ids=prompt_ids,
        image_token_ids=image_token_ids,
    )
    loss = F.cross_entropy(flat_logits, flat_labels)
    predictions = flat_logits.argmax(dim=-1)
    token_accuracy = (predictions == flat_labels).float().mean()
    return loss, {"token_accuracy": float(token_accuracy.item())}


@torch.inference_mode()
def generate_token_prefix(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    prompt_id: torch.LongTensor,
    token_count: int,
    cfg_weight: float,
    sample_strategy: str,
    seed: int,
) -> torch.LongTensor:
    set_seed(seed)
    device = prompt_id.device
    tokens = torch.zeros((2, len(prompt_id)), dtype=torch.long, device=device)
    tokens[0, :] = prompt_id
    tokens[1, :] = prompt_id
    if len(prompt_id) > 2:
        tokens[1, 1:-1] = processor.pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((token_count,), dtype=torch.long, device=device)
    lm_backbone = get_lm_backbone(model.language_model)

    outputs = None
    for step in range(token_count):
        outputs = lm_backbone(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if step != 0 else None,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = model.gen_head(hidden_states[:, -1, :]).float()
        cond_logits = logits[0:1, :]
        uncond_logits = logits[1:2, :]
        mixed_logits = uncond_logits + cfg_weight * (cond_logits - uncond_logits)

        if sample_strategy == "sample":
            probs = torch.softmax(mixed_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(mixed_logits, dim=-1, keepdim=True)

        generated_tokens[step] = next_token.squeeze(0).squeeze(0)
        repeated = next_token.repeat(1, 2).view(-1)
        image_embeds = model.prepare_gen_img_embeds(repeated)
        inputs_embeds = image_embeds.unsqueeze(1)

    return generated_tokens


def compute_retention_kl_loss(
    student_model: MultiModalityCausalLM,
    teacher_model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    prompt_ids: List[torch.LongTensor],
    token_prefix: int,
    cfg_weight: float,
    sample_strategy: str,
    temperature: float,
    seed_base: int,
) -> torch.Tensor:
    teacher_token_rows: List[torch.LongTensor] = []
    for index, prompt_id in enumerate(prompt_ids):
        teacher_token_rows.append(
            generate_token_prefix(
                model=teacher_model,
                processor=processor,
                prompt_id=prompt_id,
                token_count=token_prefix,
                cfg_weight=cfg_weight,
                sample_strategy=sample_strategy,
                seed=seed_base + index,
            )
        )
    teacher_tokens = torch.stack(teacher_token_rows, dim=0)

    with torch.no_grad():
        teacher_logits, _ = compute_generation_logits(
            model=teacher_model,
            prompt_ids=prompt_ids,
            image_token_ids=teacher_tokens,
        )

    student_logits, _ = compute_generation_logits(
        model=student_model,
        prompt_ids=prompt_ids,
        image_token_ids=teacher_tokens,
    )

    teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def compute_rehearsal_ce_loss(
    student_model: MultiModalityCausalLM,
    teacher_model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    prompt_ids: List[torch.LongTensor],
    token_count: int,
    cfg_weight: float,
    sample_strategy: str,
    seed_base: int,
):
    teacher_token_rows: List[torch.LongTensor] = []
    for index, prompt_id in enumerate(prompt_ids):
        teacher_token_rows.append(
            generate_token_prefix(
                model=teacher_model,
                processor=processor,
                prompt_id=prompt_id,
                token_count=token_count,
                cfg_weight=cfg_weight,
                sample_strategy=sample_strategy,
                seed=seed_base + index,
            )
        )
    teacher_tokens = torch.stack(teacher_token_rows, dim=0)
    loss, metrics = compute_generation_loss_and_metrics(
        model=student_model,
        prompt_ids=prompt_ids,
        image_token_ids=teacher_tokens,
    )
    return loss, metrics


def evaluate(model, dataloader, dtype):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_steps = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device, dtype)
            image_token_ids = encode_images_to_tokens(model, batch["pixel_values"])
            with (
                torch.autocast(device_type="cuda", dtype=dtype)
                if torch.cuda.is_available()
                else nullcontext()
            ):
                loss, metrics = compute_generation_loss_and_metrics(
                    model=model,
                    prompt_ids=batch["prompt_ids"],
                    image_token_ids=image_token_ids,
                )
            total_loss += float(loss.item())
            total_accuracy += metrics["token_accuracy"]
            total_steps += 1

    model.train()
    return {
        "eval_loss": total_loss / max(total_steps, 1),
        "eval_token_accuracy": total_accuracy / max(total_steps, 1),
    }


def next_retention_batch(retention_loader, retention_iterator):
    if retention_loader is None:
        return None, retention_iterator
    try:
        batch = next(retention_iterator)
    except StopIteration:
        retention_iterator = iter(retention_loader)
        batch = next(retention_iterator)
    return batch, retention_iterator


def save_checkpoint(model, processor, optimizer, scheduler, output_dir, step, args):
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.language_model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(os.path.join(checkpoint_dir, "processor"))
    save_generation_modules(model, checkpoint_dir, resolve_generation_module_names(args))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
    with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump({"global_step": step, "args": vars(args)}, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    resolved_image_token_count = compute_image_token_count(args.image_size, DEFAULT_PATCH_SIZE)
    validate_image_generation_geometry(
        image_size=args.image_size,
        patch_size=DEFAULT_PATCH_SIZE,
        expected_token_count=resolved_image_token_count,
    )
    args.image_token_num_per_image = resolved_image_token_count
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    dtype = get_dtype(args.dtype)
    train_dataset = EmoArtGenerationJsonlDataset(args.train_data)
    val_dataset = EmoArtGenerationJsonlDataset(args.val_data)
    retention_dataset = None
    if args.retention_manifest:
        retention_dataset = PromptOnlyJsonlDataset(args.retention_manifest)
    rehearsal_dataset = None
    if args.rehearsal_manifest:
        rehearsal_dataset = PromptOnlyJsonlDataset(args.rehearsal_manifest)

    model, processor, enabled_generation_modules = build_model(args, dtype)
    teacher_model = None
    if (
        (retention_dataset is not None and args.retention_loss_weight > 0.0)
        or (rehearsal_dataset is not None and args.rehearsal_loss_weight > 0.0)
    ):
        teacher_model = build_teacher_model(args.model_path, dtype)
    collator = JanusGenerationDataCollator(
        processor=processor,
        image_size=args.image_size,
        image_preprocess_mode=args.image_preprocess_mode,
        prompt_template=args.prompt_template,
        art_texture_mode=args.art_texture_mode,
        art_texture_fields=args.art_texture_fields,
        art_texture_prob=args.art_texture_prob,
    )
    retention_collator = (
        JanusPromptOnlyCollator(processor=processor)
        if (retention_dataset is not None or rehearsal_dataset is not None)
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    retention_loader = None
    if retention_dataset is not None:
        retention_loader = DataLoader(
            retention_dataset,
            batch_size=args.retention_batch_size,
            shuffle=True,
            num_workers=args.retention_num_workers,
            collate_fn=retention_collator,
        )
    rehearsal_loader = None
    if rehearsal_dataset is not None:
        rehearsal_loader = DataLoader(
            rehearsal_dataset,
            batch_size=args.rehearsal_batch_size,
            shuffle=True,
            num_workers=args.rehearsal_num_workers,
            collate_fn=retention_collator,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    total_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    planned_steps = math.ceil(total_update_steps_per_epoch * args.num_epochs)
    if args.max_steps and args.max_steps > 0:
        planned_steps = args.max_steps
    target_total_steps = planned_steps
    warmup_steps = int(planned_steps * args.warmup_ratio)

    optimizer = build_optimizer(model, args)
    if args.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=planned_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=planned_steps,
        )

    global_step = 0
    resumed_global_step = 0
    if args.resume_from_checkpoint:
        optimizer_path = os.path.join(args.resume_from_checkpoint, "optimizer.pt")
        scheduler_path = os.path.join(args.resume_from_checkpoint, "scheduler.pt")
        state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                global_step = json.load(f).get("global_step", 0)
                resumed_global_step = global_step

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and torch.cuda.is_available()))
    device = next(model.parameters()).device
    retention_iterator = iter(retention_loader) if retention_loader is not None else None
    rehearsal_iterator = iter(rehearsal_loader) if rehearsal_loader is not None else None

    model.train()
    optimizer.zero_grad(set_to_none=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "resolved_train_generation_modules": resolve_generation_module_names(args),
                "enabled_generation_modules": enabled_generation_modules,
                "parameter_summary": count_parameters(model),
                "target_total_steps": target_total_steps,
                "resumed_global_step": resumed_global_step,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(log_path, "a", encoding="utf-8") as log_file:
        if global_step >= target_total_steps:
            resume_record = {
                "global_step": global_step,
                "target_total_steps": target_total_steps,
                "message": "resume checkpoint already reached target_total_steps; skipping train loop",
            }
            print(resume_record)
            log_file.write(json.dumps(resume_record, ensure_ascii=False) + "\n")
            log_file.flush()
        for epoch in range(math.ceil(args.num_epochs)):
            if global_step >= target_total_steps:
                break
            accumulation_start_time = None
            for step, batch in enumerate(train_loader, start=1):
                if global_step >= target_total_steps:
                    break
                if (step - 1) % args.gradient_accumulation_steps == 0:
                    accumulation_start_time = time.time()
                remainder_micro_batches = len(train_loader) % args.gradient_accumulation_steps
                in_final_partial_window = (
                    remainder_micro_batches > 0
                    and step > len(train_loader) - remainder_micro_batches
                )
                accumulation_window_size = (
                    remainder_micro_batches
                    if in_final_partial_window
                    else args.gradient_accumulation_steps
                )
                batch = move_batch_to_device(batch, device, dtype)
                image_token_ids = encode_images_to_tokens(model, batch["pixel_values"])

                with (
                    torch.autocast(device_type="cuda", dtype=dtype)
                    if torch.cuda.is_available()
                    else nullcontext()
                ):
                    loss, metrics = compute_generation_loss_and_metrics(
                        model=model,
                        prompt_ids=batch["prompt_ids"],
                        image_token_ids=image_token_ids,
                    )
                    scaled_loss = loss / accumulation_window_size

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss detected at global_step={global_step}: {loss.item()}")

                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                is_accumulation_boundary = (
                    step % args.gradient_accumulation_steps == 0 or step == len(train_loader)
                )
                if not is_accumulation_boundary:
                    continue

                retention_loss_value: Optional[float] = None
                if retention_loader is not None and teacher_model is not None and args.retention_loss_weight > 0.0:
                    retention_batch, retention_iterator = next_retention_batch(retention_loader, retention_iterator)
                    retention_batch = move_prompt_batch_to_device(retention_batch, device)
                    with (
                        torch.autocast(device_type="cuda", dtype=dtype)
                        if torch.cuda.is_available()
                        else nullcontext()
                    ):
                        retention_loss = compute_retention_kl_loss(
                            student_model=model,
                            teacher_model=teacher_model,
                            processor=processor,
                            prompt_ids=retention_batch["prompt_ids"],
                            token_prefix=min(args.retention_token_prefix, resolved_image_token_count),
                            cfg_weight=args.retention_cfg_weight,
                            sample_strategy=args.retention_sample_strategy,
                            temperature=args.retention_temperature,
                            seed_base=args.retention_seed + global_step * max(args.retention_batch_size, 1),
                        )
                        scaled_retention_loss = retention_loss * args.retention_loss_weight
                    retention_loss_value = float(retention_loss.item())
                    if scaler.is_enabled():
                        scaler.scale(scaled_retention_loss).backward()
                    else:
                        scaled_retention_loss.backward()

                rehearsal_loss_value: Optional[float] = None
                rehearsal_token_accuracy: Optional[float] = None
                if rehearsal_loader is not None and teacher_model is not None and args.rehearsal_loss_weight > 0.0:
                    rehearsal_batch, rehearsal_iterator = next_retention_batch(rehearsal_loader, rehearsal_iterator)
                    rehearsal_batch = move_prompt_batch_to_device(rehearsal_batch, device)
                    with (
                        torch.autocast(device_type="cuda", dtype=dtype)
                        if torch.cuda.is_available()
                        else nullcontext()
                    ):
                        rehearsal_loss, rehearsal_metrics = compute_rehearsal_ce_loss(
                            student_model=model,
                            teacher_model=teacher_model,
                            processor=processor,
                            prompt_ids=rehearsal_batch["prompt_ids"],
                            token_count=min(args.rehearsal_token_count, resolved_image_token_count),
                            cfg_weight=args.rehearsal_cfg_weight,
                            sample_strategy=args.rehearsal_sample_strategy,
                            seed_base=args.rehearsal_seed + global_step * max(args.rehearsal_batch_size, 1),
                        )
                        scaled_rehearsal_loss = rehearsal_loss * args.rehearsal_loss_weight
                    rehearsal_loss_value = float(rehearsal_loss.item())
                    rehearsal_token_accuracy = float(rehearsal_metrics["token_accuracy"])
                    if scaler.is_enabled():
                        scaler.scale(scaled_rehearsal_loss).backward()
                    else:
                        scaled_rehearsal_loss.backward()

                grad_norm = 0.0
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                max_memory_gb = 0.0
                if torch.cuda.is_available():
                    max_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

                log_record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": float(loss.item()),
                    "token_accuracy": metrics["token_accuracy"],
                    "lr_lora": float(scheduler.get_last_lr()[0]),
                    "lr_generation": float(scheduler.get_last_lr()[-1]),
                    "grad_norm": round(grad_norm, 6),
                    "step_time_sec": round(time.time() - (accumulation_start_time or time.time()), 4),
                    "max_memory_gb": round(max_memory_gb, 3),
                }
                if retention_loss_value is not None:
                    log_record["retention_kl_loss"] = retention_loss_value
                if rehearsal_loss_value is not None:
                    log_record["rehearsal_ce_loss"] = rehearsal_loss_value
                if rehearsal_token_accuracy is not None:
                    log_record["rehearsal_token_accuracy"] = rehearsal_token_accuracy
                print(log_record)
                log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                log_file.flush()

                if global_step % args.eval_steps == 0 and len(val_dataset) > 0:
                    eval_metrics = evaluate(model, val_loader, dtype)
                    eval_record = {"global_step": global_step, **eval_metrics}
                    print(eval_record)
                    log_file.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
                    log_file.flush()

                if global_step % args.save_steps == 0:
                    save_checkpoint(model, processor, optimizer, scheduler, args.output_dir, global_step, args)

                if global_step >= target_total_steps:
                    break

    final_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(final_dir, exist_ok=True)
    model.language_model.save_pretrained(final_dir)
    processor.save_pretrained(os.path.join(final_dir, "processor"))
    save_generation_modules(model, final_dir, resolve_generation_module_names(args))

    if len(val_dataset) > 0:
        final_eval_metrics = evaluate(model, val_loader, dtype)
        print({"final_" + key: value for key, value in final_eval_metrics.items()})


if __name__ == "__main__":
    main()
