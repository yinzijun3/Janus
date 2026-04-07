# Retention-Aware EmoArt V1

This is the first run on the new mainline:

- keep the stronger data-side base from `frameclean_v2`
- explicitly preserve base-model behavior during training
- stop relying on "train less and hope it breaks less" as the only safeguard

## Goal

The question for this run is:

> Can we keep the EmoArt task gains while reducing the regression of Janus base behavior?

This run is not trying to beat `Exp A` on pure task-fit metrics alone. It is trying to improve the trade-off between:

- EmoArt task alignment
- visual quality stability
- retention of base distribution

## Training Base

Task training manifest:

- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v2.jsonl`

Validation manifest:

- `/root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/val.jsonl`

We keep `frameclean_v2` as the base because it was the strongest data-side branch after visual review, even though it still did not replace `Exp A`.

## Retention Set

Prompt-only retention manifest:

- `/root/autodl-tmp/emoart_gen_runs/retention_prompts_v1.jsonl`

Summary:

- `/root/autodl-tmp/emoart_gen_runs/retention_prompts_v1_summary.json`

Source:

- `/root/autodl-tmp/repos/Janus/regression_manifests/generation_v1.jsonl`

This set contains 16 generic prompts repeated 16 times, for 256 prompt-only rows. The categories intentionally stay outside the EmoArt training distribution:

- product photo
- wildlife photo
- portrait photo
- interior design
- medical diagram
- landscape photo
- vehicle photo
- graphic design
- anime / illustration

## Retention Loss

The retention objective is a KL distillation loss against the frozen base Janus model.

At each optimizer update:

1. sample one prompt-only retention batch
2. use the frozen teacher to generate a short image-token prefix
3. force the student adapter to match the teacher distribution on that prefix

The training objective becomes:

- EmoArt generation loss
- plus `0.2 * retention_kl_loss`

This is a direct retention constraint. It is stronger and more explicit than only shrinking LoRA scope or lowering learning rate.

## Fixed Training Setup

This run keeps the conservative setup from `frameclean_v2` and changes only the retention objective:

- `image_preprocess_mode = crop`
- `generation_module_mode = frozen`
- LoRA target modules:
  - `q_proj k_proj v_proj o_proj`
- `per_device_train_batch_size = 8`
- `per_device_eval_batch_size = 8`
- `gradient_accumulation_steps = 2`
- `num_epochs = 2`
- `learning_rate = 1e-4`
- `generation_learning_rate = 1e-4`
- `prompt_template = default`
- `art_texture_mode = off`

Retention-specific settings:

- `retention_loss_weight = 0.2`
- `retention_batch_size = 8`
- `retention_token_prefix = 64`
- `retention_temperature = 1.0`
- `retention_cfg_weight = 5.0`
- `retention_sample_strategy = greedy`

## Output Names

- train output:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_retention_v1_bs8`
- 8-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_retention_v1_bs8_8`
- 8-sample triptych packet:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_retention_v1_bs8_8/triptych_packet`
- 32-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_retention_v1_bs8_32`
- 32-sample triptych packet:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_retention_v1_bs8_32/triptych_packet`

## Train

```bash
/root/autodl-tmp/repos/Janus/run_emoart_retention_v1.sh train
```

## Quick Compare

```bash
/root/autodl-tmp/repos/Janus/run_emoart_retention_v1.sh compare8
```

## Full Compare

```bash
/root/autodl-tmp/repos/Janus/run_emoart_retention_v1.sh compare32
```

## Triptych Packet

```bash
/root/autodl-tmp/repos/Janus/run_emoart_retention_v1.sh packet8
```

```bash
/root/autodl-tmp/repos/Janus/run_emoart_retention_v1.sh packet32
```

## What To Look For

This run is only interesting if it improves one or more of these without triggering new hard failures:

- less global style collapse than `Exp A`
- less border / scroll / panel leakage than old branches
- less open-domain regression than `Exp A`
- fewer painterly failures caused by average-style shrinkage

The main comparison anchor is not just `Exp A`. It is:

- `Exp A`
- `frameclean_v2`
- retention-aware `v1`

If retention-aware `v1` keeps task fit close to `frameclean_v2` while improving generic retention or reducing visual over-specialization, then this becomes the first real candidate for a new mainline.
