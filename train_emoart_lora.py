import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

from finetune.emoart import EmoArtJsonlDataset, JanusVLDataCollator
from janus.models import MultiModalityCausalLM, VLChatProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Janus on EmoArt-5k.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "JANUS_MODEL_PATH",
            "/root/autodl-tmp/hf_cache/hub/models--deepseek-ai--Janus-Pro-1B/snapshots/960ab33191f61342a4c60ae74d8dc356a39fafcb",
        ),
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def move_batch_to_device(batch, device, dtype):
    batch["input_ids"] = batch["input_ids"].to(device)
    batch["attention_mask"] = batch["attention_mask"].to(device)
    batch["images_seq_mask"] = batch["images_seq_mask"].to(device)
    batch["images_emb_mask"] = batch["images_emb_mask"].to(device)
    batch["pixel_values"] = batch["pixel_values"].to(device=device, dtype=dtype)
    batch["labels"] = batch["labels"].to(device)
    return batch


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
    else:
        model.language_model = get_peft_model(model.language_model, lora_config)

    if args.gradient_checkpointing:
        model.language_model.gradient_checkpointing_enable()
        model.language_model.enable_input_require_grads()
        model.language_model.config.use_cache = False

    model.language_model.print_trainable_parameters()
    return model, processor


def evaluate(model, dataloader, dtype):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device, dtype)
            inputs_embeds = model.prepare_inputs_embeds(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                images_seq_mask=batch["images_seq_mask"],
                images_emb_mask=batch["images_emb_mask"],
            )
            outputs = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
                return_dict=True,
            )
            total_loss += outputs.loss.item()
            total_steps += 1

    model.train()
    return total_loss / max(total_steps, 1)


def save_checkpoint(model, processor, optimizer, scheduler, output_dir, step, args):
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.language_model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(os.path.join(checkpoint_dir, "processor"))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
    with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump({"global_step": step, "args": vars(args)}, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    dtype = get_dtype(args.dtype)
    train_dataset = EmoArtJsonlDataset(args.train_manifest)
    val_dataset = EmoArtJsonlDataset(args.val_manifest)

    model, processor = build_model(args, dtype)
    collator = JanusVLDataCollator(processor=processor, max_seq_len=args.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
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
    warmup_steps = max(int(planned_steps * args.warmup_ratio), 1)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_steps,
    )

    global_step = 0
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

    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and torch.cuda.is_available()))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    with open(log_path, "a", encoding="utf-8") as log_file:
        for epoch in range(math.ceil(args.num_epochs)):
            for step, batch in enumerate(train_loader, start=1):
                step_start = time.time()
                batch = move_batch_to_device(batch, device, dtype)
                with (
                    torch.autocast(device_type="cuda", dtype=dtype)
                    if torch.cuda.is_available()
                    else nullcontext()
                ):
                    inputs_embeds = model.prepare_inputs_embeds(
                        input_ids=batch["input_ids"],
                        pixel_values=batch["pixel_values"],
                        images_seq_mask=batch["images_seq_mask"],
                        images_emb_mask=batch["images_emb_mask"],
                    )
                    outputs = model.language_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        use_cache=False,
                        return_dict=True,
                    )
                    loss = outputs.loss / args.gradient_accumulation_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step % args.gradient_accumulation_steps != 0:
                    continue

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
                    "loss": float(loss.item() * args.gradient_accumulation_steps),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "step_time_sec": round(time.time() - step_start, 4),
                    "max_memory_gb": round(max_memory_gb, 3),
                }
                print(log_record)
                log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                log_file.flush()

                if global_step % args.logging_steps == 0 and len(val_dataset) > 0:
                    pass

                if global_step % args.eval_steps == 0 and len(val_dataset) > 0:
                    eval_loss = evaluate(model, val_loader, dtype)
                    eval_record = {"global_step": global_step, "eval_loss": eval_loss}
                    print(eval_record)
                    log_file.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
                    log_file.flush()

                if global_step % args.save_steps == 0:
                    save_checkpoint(model, processor, optimizer, scheduler, args.output_dir, global_step, args)

                if args.max_steps and global_step >= args.max_steps:
                    break

            if args.max_steps and global_step >= args.max_steps:
                break

    final_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(final_dir, exist_ok=True)
    model.language_model.save_pretrained(final_dir)
    processor.save_pretrained(os.path.join(final_dir, "processor"))

    if len(val_dataset) > 0:
        final_eval_loss = evaluate(model, val_loader, dtype)
        print({"final_eval_loss": final_eval_loss})


if __name__ == "__main__":
    main()
