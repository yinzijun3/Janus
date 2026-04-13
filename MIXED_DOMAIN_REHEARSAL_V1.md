# Mixed-Domain Rehearsal V1

This is the next mainline after `retention_v2`:

- stop relying on prompt-only teacher KL as the sole retention mechanism
- keep the same mixed generic + generic-art prompt source
- replace the retention KL branch with a stronger rehearsal loss

## Goal

The question for this run is:

> Can mixed-domain pseudo-target rehearsal preserve more of Janus base behavior than prompt-only KL, without introducing the same banding and panelization failures?

## Core Idea

For a batch of mixed-domain prompts:

1. a frozen base Janus teacher generates a short image-token sequence
2. the student adapter is trained with cross-entropy on those teacher-generated tokens

This is still teacher-guided, but it is no longer "match the teacher distribution with KL only".
It is a stronger rehearsal signal:

- the student rehearses actual base-model generation trajectories
- on generic and generic-art prompts

## Training Base

Task training manifest:

- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v2.jsonl`

Validation manifest:

- `/root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/val.jsonl`

## Rehearsal Set

Rehearsal manifest:

- `/root/autodl-tmp/emoart_gen_runs/rehearsal_prompts_v1.jsonl`

This is built from:

- `/root/autodl-tmp/repos/Janus/regression_manifests/retention_v2_source.jsonl`

using the same prompt-expansion utility.

The source prompt pool is intentionally mixed:

- open-domain photo prompts
- product / commercial prompts
- diagram / drafting prompts
- generic art prompts
- illustration prompts
- graphic-design prompts

## Loss

The total training objective becomes:

- EmoArt task generation loss
- plus `0.10 * rehearsal_ce_loss`

Where `rehearsal_ce_loss` is:

- teacher generates `32` image tokens
- student is trained to predict those exact tokens

This is more like replay / rehearsal than the softer KL-only retention branch.

## Fixed Training Setup

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

Rehearsal-specific settings:

- `rehearsal_batch_size = 8`
- `rehearsal_loss_weight = 0.10`
- `rehearsal_token_count = 32`
- `rehearsal_cfg_weight = 5.0`
- `rehearsal_sample_strategy = greedy`

## Output Names

- train output:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_mixed_domain_rehearsal_v1_bs8`
- 8-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_mixed_domain_rehearsal_v1_bs8_8`
- 8-sample triptych packet:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_mixed_domain_rehearsal_v1_bs8_8/triptych_packet`
- 32-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_mixed_domain_rehearsal_v1_bs8_32`
- 32-sample triptych packet:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_mixed_domain_rehearsal_v1_bs8_32/triptych_packet`

## Build Rehearsal Manifest

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate januspro
export PYTHONPATH=/root/miniconda3/lib/python3.10/site-packages:${PYTHONPATH:-}

python /root/autodl-tmp/repos/Janus/build_emoart_retention_prompt_manifest.py \
  --input-manifest /root/autodl-tmp/repos/Janus/regression_manifests/retention_v2_source.jsonl \
  --output-manifest /root/autodl-tmp/emoart_gen_runs/rehearsal_prompts_v1.jsonl \
  --summary-path /root/autodl-tmp/emoart_gen_runs/rehearsal_prompts_v1_summary.json \
  --repeat-factor 8
```

## Train

```bash
/root/autodl-tmp/repos/Janus/run_emoart_mixed_domain_rehearsal_v1.sh train
```

## Quick Compare

```bash
/root/autodl-tmp/repos/Janus/run_emoart_mixed_domain_rehearsal_v1.sh compare8
```

## Full Compare

```bash
/root/autodl-tmp/repos/Janus/run_emoart_mixed_domain_rehearsal_v1.sh compare32
```

## What To Look For

This branch is only worth keeping if it:

- avoids the banding / panelization seen in retention branches
- keeps or improves task fit versus `frameclean_v2`
- improves the visual acceptability of difficult samples
- gives a better trade-off than prompt-only KL
