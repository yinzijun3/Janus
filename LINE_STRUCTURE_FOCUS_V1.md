# Line Structure Focus V1

This is a narrower continuation than `failure_style_focus_v1`.

The previous focused continuation gave mixed but useful evidence:

- it could rescue a few residual failure styles
- but mixing line-structure failures with East Asian residuals created interference
- the next step is to isolate the structural / decorative failure family

## Base Adapter

- `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_frameclean_v2/final_adapter`

## Input Manifest

- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v2.jsonl`

## Focus Styles

- `Art Nouveau (Modern)`
- `Pointillism`
- `Regionalism`
- `High Renaissance`
- `Color Field Painting`

These are the styles that still looked most tied to line complexity, decorative structure, or difficult spatial organization after `frameclean_v2`.

## Excluded Residuals

This run intentionally does **not** try to solve East Asian residual failures at the same time:

- `China_images`
- `Korea`
- plus the high-bias families already excluded from anchors

It also leaves out `Orientalism` here, so this run stays focused on line/structure failure rather than broad landscape bias.

## Anchor Policy

- `6` anchor samples per non-focus style
- exclude high-bias default-mode families:
  - `China_images`
  - `Islamic_image`
  - `Sōsaku hanga`
  - `Shin-hanga`
  - `Ukiyo-e`
  - `Op Art`
  - `Constructivism`
  - `Concretism`
  - `Korea`
  - `Orientalism`

## Focus Weights

- `Art Nouveau (Modern)`: `1.45`
- `Pointillism`: `1.35`
- `Regionalism`: `1.3`
- `High Renaissance`: `1.2`
- `Color Field Painting`: `1.2`

## Training Setup

Conservative continuation on top of `frameclean_v2`:

- resume from:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_frameclean_v2/final_adapter`
- `image_preprocess_mode = crop`
- `generation_module_mode = frozen`
- LoRA target modules:
  - `q_proj k_proj v_proj o_proj`
- `learning_rate = 4e-5`
- `generation_learning_rate = 4e-5`
- `num_epochs = 0.75`
- `prompt_template = default`
- `art_texture_mode = off`

## Output Names

- train output:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_line_structure_focus_v1`
- 8-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_line_structure_focus_v1_8`
- 32-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_line_structure_focus_v1_32`
