# Style Reweight EmoArt V1

This run is the next step after `frameclean_v2`.

`frameclean_v2` already showed that data-side intervention is real:

- it recovered much of the semantic gap vs conservative `v2`
- but the remaining failures were no longer random
- they concentrated in a smaller group of styles

So `style_reweight_v1` starts from the cleaned `frameclean_v2` manifest and adds lightweight style-aware weighting.

## Base Manifest

- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v2.jsonl`

## Reweight Logic

Upsample styles that still tended to fail or lose complex line / painterly diversity:

- `Art Nouveau (Modern)`: `1.5`
- `Pointillism`: `1.5`
- `Regionalism`: `1.4`
- `High Renaissance`: `1.3`
- `Color Field Painting`: `1.4`
- `Naturalism`: `1.3`
- `Orientalism`: `1.2`

Downsample styles that still looked like strong default-mode candidates after frame cleaning:

- `China_images`: `0.6`
- `Islamic_image`: `0.6`
- `Sōsaku hanga`: `0.7`
- `Shin-hanga`: `0.75`
- `Ukiyo-e`: `0.75`
- `Op Art`: `0.7`
- `Constructivism`: `0.7`
- `Concretism`: `0.7`
- `Korea`: `0.8`

## Generated Manifest

- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_style_reweight_v1.jsonl`
- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_style_reweight_v1_summary.json`

## Fixed Training Setup

Same conservative setup as `v2`, `frameclean_v1`, and `frameclean_v2`:

- `image_preprocess_mode = crop`
- `generation_module_mode = frozen`
- LoRA target modules:
  - `q_proj k_proj v_proj o_proj`
- `num_epochs = 2`
- `learning_rate = 1e-4`
- `generation_learning_rate = 1e-4`
- `prompt_template = default`
- `art_texture_mode = off`

## Output Names

- train output:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_style_reweight_v1`
- 8-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_style_reweight_v1_8`
- 32-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_style_reweight_v1_32`

## Build Manifest

```bash
python /root/autodl-tmp/repos/Janus/build_emoart_style_reweight_manifest.py \
  --input-manifest /root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v2.jsonl \
  --output-manifest /root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_style_reweight_v1.jsonl \
  --summary-path /root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_style_reweight_v1_summary.json \
  --preset v1
```

## Train

```bash
/root/autodl-tmp/repos/Janus/run_emoart_style_reweight_v1.sh train
```

## Compare

```bash
/root/autodl-tmp/repos/Janus/run_emoart_style_reweight_v1.sh compare8
```

```bash
/root/autodl-tmp/repos/Janus/run_emoart_style_reweight_v1.sh compare32
```
