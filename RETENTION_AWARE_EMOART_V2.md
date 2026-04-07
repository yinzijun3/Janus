# Retention-Aware EmoArt V2

This is the second retention-aware run after `retention_v1` showed a real but misaligned signal:

- retention regularization did affect generation behavior
- but the first retention set was too small and too biased toward generic photo / product prompts
- the KL constraint was also too hard, which flattened artistic structure and introduced banding-like failure modes

## Goal

The question for this run is:

> Can we preserve more of Janus base behavior without dragging the adapter toward an overly smooth, photo-biased average solution?

`retention_v2` keeps the same EmoArt task base and changes two things:

1. a much broader, mixed retention prompt set
2. a softer retention constraint

## Training Base

Task training manifest:

- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v2.jsonl`

Validation manifest:

- `/root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/val.jsonl`

We still use `frameclean_v2` as the task-training anchor because it was the strongest side branch after visual review.

## Retention Set

Prompt-only retention manifest:

- `/root/autodl-tmp/emoart_gen_runs/retention_prompts_v2.jsonl`

Summary:

- `/root/autodl-tmp/emoart_gen_runs/retention_prompts_v2_summary.json`

Source prompt file:

- `/root/autodl-tmp/repos/Janus/regression_manifests/retention_v2_source.jsonl`

This version is intentionally much broader than `v1`.

It mixes:

- open-domain photo prompts
- product / commercial prompts
- diagram / drafting prompts
- illustration / graphic-design prompts
- generic art prompts that are not tied to EmoArt metadata

The goal is to preserve both:

- open-domain controllability
- base artistic generation diversity

instead of preserving mostly clean photo-like outputs.

## Retention Loss

The retention objective is still KL distillation against a frozen base Janus teacher.

At each optimizer update:

1. sample one prompt-only retention batch
2. use the frozen teacher to generate a short image-token prefix
3. force the student adapter to match the teacher distribution on that prefix

The training objective becomes:

- EmoArt generation loss
- plus `0.05 * retention_kl_loss`

Compared with `v1`, the retention constraint is softer:

- `retention_loss_weight: 0.2 -> 0.05`
- `retention_token_prefix: 64 -> 16`

This should reduce the tendency to flatten large-scale artistic structure.

## Fixed Training Setup

This run keeps the same efficient `48G` setup:

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

- `retention_batch_size = 8`
- `retention_loss_weight = 0.05`
- `retention_token_prefix = 16`
- `retention_temperature = 1.0`
- `retention_cfg_weight = 5.0`
- `retention_sample_strategy = greedy`

## Output Names

- train output:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_retention_v2_bs8`
- 8-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_retention_v2_bs8_8`
- 8-sample triptych packet:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_retention_v2_bs8_8/triptych_packet`
- 32-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_retention_v2_bs8_32`
- 32-sample triptych packet:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_retention_v2_bs8_32/triptych_packet`

## Build Retention Manifest

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate januspro
export PYTHONPATH=/root/miniconda3/lib/python3.10/site-packages:${PYTHONPATH:-}

python /root/autodl-tmp/repos/Janus/build_emoart_retention_prompt_manifest.py \
  --input-manifest /root/autodl-tmp/repos/Janus/regression_manifests/retention_v2_source.jsonl \
  --output-manifest /root/autodl-tmp/emoart_gen_runs/retention_prompts_v2.jsonl \
  --summary-path /root/autodl-tmp/emoart_gen_runs/retention_prompts_v2_summary.json \
  --repeat-factor 8
```

## Train

```bash
/root/autodl-tmp/repos/Janus/run_emoart_retention_v2.sh train
```

## Quick Compare

```bash
/root/autodl-tmp/repos/Janus/run_emoart_retention_v2.sh compare8
```

## Full Compare

```bash
/root/autodl-tmp/repos/Janus/run_emoart_retention_v2.sh compare32
```

## Triptych Packet

```bash
/root/autodl-tmp/repos/Janus/run_emoart_retention_v2.sh packet8
```

```bash
/root/autodl-tmp/repos/Janus/run_emoart_retention_v2.sh packet32
```

## What To Look For

This run is only worth keeping if it improves one or more of these without reintroducing hard visual failures:

- less banding / panelization than `retention_v1`
- less open-domain regression than `Exp A`
- less style collapse than `Exp A`
- task-fit close to or better than `frameclean_v2`

The main comparison anchor remains:

- `Exp A`
- `frameclean_v2`
- `retention_v2`

If `retention_v2` preserves more generic and artistic behavior without breaking task quality, it becomes the first retention-aware branch worth taking to a full `32` sample evaluation.
