# Conservative EmoArt V1

This run is the first low-risk recovery attempt for the current failure mode:

- over-smoothing
- simplified brushwork
- border / framed-output bias
- excessive East Asian ink-painting prior leakage
- weaker abstract / oil / complex-line rendering

The conservative adjustments are:

- `image_preprocess_mode = crop`
- `generation_module_mode = head_only`
- `generation_learning_rate = 1e-4`
- no repeated texture clause injection when the base prompt already contains `painting medium:` or `surface detail:`

## Output Names

- train output:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_conservative_crop_headonly_v1`
- 32-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_conservative_crop_headonly_32`
- 8-sample quick compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_conservative_crop_headonly_8`

## Train

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate januspro && export PYTHONPATH=/root/miniconda3/lib/python3.10/site-packages:$PYTHONPATH && \
python /root/autodl-tmp/repos/Janus/train_emoart_gen_lora.py \
  --train-data /root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/train.jsonl \
  --val-data /root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/val.jsonl \
  --output-dir /root/autodl-tmp/emoart_gen_runs/out_gen_expA_conservative_crop_headonly_v1 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --generation-module-mode head_only \
  --learning-rate 1e-4 \
  --generation-learning-rate 1e-4 \
  --scheduler-type cosine \
  --gradient-accumulation-steps 16 \
  --num-epochs 3 \
  --image-preprocess-mode crop \
  --dtype bf16 \
  --prompt-template default \
  --art-texture-mode off \
  --art-texture-fields all \
  --art-texture-prob 0.0
```

## Quick Compare

Use this first if you want a cheap sanity check before the full 32-sample review.

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate januspro && export PYTHONPATH=/root/miniconda3/lib/python3.10/site-packages:$PYTHONPATH && \
python /root/autodl-tmp/repos/Janus/compare_emoart_gen.py \
  --manifest-path /root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/val.jsonl \
  --adapter-path /root/autodl-tmp/emoart_gen_runs/out_gen_expA_conservative_crop_headonly_v1/final_adapter \
  --output-dir /root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_conservative_crop_headonly_8 \
  --num-samples 8 \
  --seed 42 \
  --dtype bf16 \
  --image-preprocess-mode crop \
  --sample-strategy greedy \
  --prompt-template default \
  --art-texture-mode off \
  --art-texture-fields all
```

## Full 32-Sample Compare

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate januspro && export PYTHONPATH=/root/miniconda3/lib/python3.10/site-packages:$PYTHONPATH && \
python /root/autodl-tmp/repos/Janus/compare_emoart_gen.py \
  --manifest-path /root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/val.jsonl \
  --adapter-path /root/autodl-tmp/emoart_gen_runs/out_gen_expA_conservative_crop_headonly_v1/final_adapter \
  --output-dir /root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_conservative_crop_headonly_32 \
  --num-samples 32 \
  --seed 42 \
  --dtype bf16 \
  --image-preprocess-mode crop \
  --sample-strategy greedy \
  --prompt-template default \
  --art-texture-mode off \
  --art-texture-fields all
```

## What To Look For

Relative to the current Exp A mainline, this run is successful only if the following improve visibly:

- less border / panel / framed composition bias
- less China/Korea/ink-wash carryover on non-ink styles
- stronger abstract structure retention
- less smoothed oil-paint output
- better complex contour and line behavior

It is not enough for metrics to look similar. The visual review matters more here.

## Suggested Human Review Targets

After the 32-sample compare, inspect at least these styles first:

- `Realism`
- `Post-Impressionism`
- `Baroque`
- `High Renaissance`
- `Dada`
- `Art Informel`
- `Color Field Painting`
- `Regionalism`
- `Ink and wash painting`
- `Korea`

The goal is to check both sides:

- whether the overfit East Asian prior is reduced
- whether difficult non-ink styles recover detail and line complexity
