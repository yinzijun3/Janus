# Frameclean EmoArt V2

This run is the stronger follow-up after `frameclean_v1` showed a real signal:

- border / frame cleaning improved texture proxy metrics
- but `frameclean_v1` was still too weak to reliably beat conservative `v2` on task fit

So `frameclean_v2` keeps the same conservative training setup and increases only the cleaning strength.

## Data Change

Train manifest:

- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v2.jsonl`

This manifest removes samples with:

- `frame_risk_score >= 0.55`

Audit summary:

- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v2_summary.json`

Current removal size:

- original train set: `4979`
- removed: `236`
- kept: `4743`

This is still a single-variable data experiment. It is stronger than `v1`, but it is not yet a full style rebalance.

## Fixed Training Setup

Same as conservative `v2` and `frameclean_v1`:

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
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_frameclean_v2`
- 8-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_frameclean_v2_8`
- 8-sample triptych packet:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_frameclean_v2_8/triptych_packet`
- 32-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_frameclean_v2_32`
- 32-sample triptych packet:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_frameclean_v2_32/triptych_packet`

## Train

```bash
/root/autodl-tmp/repos/Janus/run_emoart_frameclean_v2.sh train
```

## Quick Compare

```bash
/root/autodl-tmp/repos/Janus/run_emoart_frameclean_v2.sh compare8
```

## Full Compare

```bash
/root/autodl-tmp/repos/Janus/run_emoart_frameclean_v2.sh compare32
```

## Triptych Packet

```bash
/root/autodl-tmp/repos/Janus/run_emoart_frameclean_v2.sh packet8
```

```bash
/root/autodl-tmp/repos/Janus/run_emoart_frameclean_v2.sh packet32
```

## What To Look For

This run is only worth keeping if it improves the visual failure modes that `frameclean_v1` only partially fixed:

- further reduce border / panel / scroll residue
- preserve or improve line complexity on `Art Nouveau`, `Orientalism`, `Naturalism`
- avoid giving back the gains on `Rococo`, `Shin-hanga`, `Naïve Art`

If `frameclean_v2` reduces border leakage but starts to hurt semantics too much, that is the signal to stop pure threshold cleaning and switch to style-aware reweighting instead.
