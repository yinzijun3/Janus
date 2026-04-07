# Failure Style Focus V1

This is a narrow continuation experiment built on top of `frameclean_v2`.

`style_reweight_v1` told us a useful negative result:

- global style reweighting was too blunt
- it gave away too much semantic stability in exchange for local texture wins
- the remaining failures are concentrated, not broad

So `failure_style_focus_v1` switches from global reweighting to a short continuation pass on a compact failure-style subset.

## Base Adapter

- `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_frameclean_v2/final_adapter`

## Input Manifest

- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v2.jsonl`

## Focus Styles

These are the styles that still looked consistently weak after `frameclean_v2`:

- `Art Nouveau (Modern)`
- `Pointillism`
- `Regionalism`
- `High Renaissance`
- `Color Field Painting`
- `Orientalism`

## Anchor Policy

To reduce overfitting during continuation, the manifest also keeps a small anchor slice from the rest of the cleaned set:

- `8` anchor samples per non-focus style
- exclude high bias families that we already know can become default modes:
  - `China_images`
  - `Islamic_image`
  - `Sōsaku hanga`
  - `Shin-hanga`
  - `Ukiyo-e`
  - `Op Art`
  - `Constructivism`
  - `Concretism`
  - `Korea`

## Focus Weights

Within the focus subset, the continuation manifest modestly repeats the hardest styles:

- `Art Nouveau (Modern)`: `1.35`
- `Pointillism`: `1.3`
- `Regionalism`: `1.25`
- `High Renaissance`: `1.2`
- `Color Field Painting`: `1.2`
- `Orientalism`: `1.15`

## Training Setup

Conservative continuation on top of `frameclean_v2`:

- resume from base adapter:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_frameclean_v2/final_adapter`
- `image_preprocess_mode = crop`
- `generation_module_mode = frozen`
- LoRA target modules:
  - `q_proj k_proj v_proj o_proj`
- `learning_rate = 5e-5`
- `generation_learning_rate = 5e-5`
- `num_epochs = 0.75`
- `prompt_template = default`
- `art_texture_mode = off`

## Output Names

- train output:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_failure_style_focus_v1`
- 8-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_failure_style_focus_v1_8`
- 32-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_failure_style_focus_v1_32`

## Build Manifest

```bash
python /root/autodl-tmp/repos/Janus/build_emoart_failure_style_focus_manifest.py \
  --input-manifest /root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v2.jsonl \
  --output-manifest /root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_failure_style_focus_v1.jsonl \
  --summary-path /root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_failure_style_focus_v1_summary.json \
  --preset v1
```

## Train

```bash
/root/autodl-tmp/repos/Janus/run_emoart_failure_style_focus_v1.sh train
```

## Compare

```bash
/root/autodl-tmp/repos/Janus/run_emoart_failure_style_focus_v1.sh compare8
```

```bash
/root/autodl-tmp/repos/Janus/run_emoart_failure_style_focus_v1.sh compare32
```
