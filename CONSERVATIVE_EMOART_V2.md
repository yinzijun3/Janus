# Conservative EmoArt V2

This run is a stricter recovery attempt after `v1` showed that:

- `crop` helped slightly with border bias
- but `head_only` still left strong template compression
- line complexity and non-ink style diversity were still not recovered

So `v2` moves one step further toward preserving the base generator prior.

## Main Changes vs V1

- keep `image_preprocess_mode = crop`
- freeze all generation modules: `generation_module_mode = frozen`
- shrink LoRA target modules to attention only:
  - `q_proj k_proj v_proj o_proj`
- shorten training:
  - `num_epochs = 2`
- keep prompt-side duplicate texture clause filtering from the current codebase

## Expected Effect

This run is intentionally biased toward:

- less over-specialization
- less style collapse
- less East Asian scroll / panel leakage
- better preservation of line structure and oil/abstract diversity

The risk is that EmoArt task gains may become smaller than Exp A.

## Output Names

- train output:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_conservative_crop_loraonly_attn_v2`
- 8-sample quick compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_conservative_crop_loraonly_attn_v2_8`
- 8-sample triptych packet:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_conservative_crop_loraonly_attn_v2_8/triptych_packet`
- 32-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_conservative_crop_loraonly_attn_v2_32`
- 32-sample triptych packet:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_conservative_crop_loraonly_attn_v2_32/triptych_packet`

## Train

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate januspro && export PYTHONPATH=/root/miniconda3/lib/python3.10/site-packages:$PYTHONPATH && \
python /root/autodl-tmp/repos/Janus/train_emoart_gen_lora.py \
  --train-data /root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/train.jsonl \
  --val-data /root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/val.jsonl \
  --output-dir /root/autodl-tmp/emoart_gen_runs/out_gen_expA_conservative_crop_loraonly_attn_v2 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --target-modules q_proj k_proj v_proj o_proj \
  --generation-module-mode frozen \
  --learning-rate 1e-4 \
  --generation-learning-rate 1e-4 \
  --scheduler-type cosine \
  --gradient-accumulation-steps 16 \
  --num-epochs 2 \
  --image-preprocess-mode crop \
  --dtype bf16 \
  --prompt-template default \
  --art-texture-mode off \
  --art-texture-fields all \
  --art-texture-prob 0.0
```

## Quick Compare

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate januspro && export PYTHONPATH=/root/miniconda3/lib/python3.10/site-packages:$PYTHONPATH && \
python /root/autodl-tmp/repos/Janus/compare_emoart_gen.py \
  --manifest-path /root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/val.jsonl \
  --adapter-path /root/autodl-tmp/emoart_gen_runs/out_gen_expA_conservative_crop_loraonly_attn_v2/final_adapter \
  --output-dir /root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_conservative_crop_loraonly_attn_v2_8 \
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
  --adapter-path /root/autodl-tmp/emoart_gen_runs/out_gen_expA_conservative_crop_loraonly_attn_v2/final_adapter \
  --output-dir /root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_conservative_crop_loraonly_attn_v2_32 \
  --num-samples 32 \
  --seed 42 \
  --dtype bf16 \
  --image-preprocess-mode crop \
  --sample-strategy greedy \
  --prompt-template default \
  --art-texture-mode off \
  --art-texture-fields all
```

## Auto Triptych Packet

```bash
/root/autodl-tmp/repos/Janus/run_emoart_conservative_v2.sh packet8
```

```bash
/root/autodl-tmp/repos/Janus/run_emoart_conservative_v2.sh packet32
```

## Pass Criteria

This run is only worth keeping if the visual review shows:

- less frame / border / panel bias than Exp A and `v1`
- less East Asian landscape leakage on non-East-Asian styles
- better complex line retention on `Art Nouveau`, `Dada`, `Abstract*`, `Art Informel`
- better painterly diversity on oil / portrait / ornate historical styles
- no obvious collapse into the same simplified low-detail solution

The visual review matters more than aggregate metric movement here.
