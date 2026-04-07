# Frameclean EmoArt V1

This run is the first data-side validation after the training-only conservative line plateaued.

The goal is intentionally narrow:

- keep the conservative `v2` training setup unchanged
- change only the training manifest
- remove the most obvious frame / border / scroll-risk samples from training

## Data Change

Train manifest:

- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v1.jsonl`

This manifest removes samples with:

- `frame_risk_score >= 0.70`

Audit summary:

- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v1_summary.json`

Current removal size:

- original train set: `4979`
- removed: `104`
- kept: `4875`

This is intentionally conservative. It should be read as a single-variable test for border / panel / scroll bias, not a full data rebalance.

## Fixed Training Setup

Same as conservative `v2`:

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
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_expA_frameclean_v1`
- 8-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_frameclean_v1_8`
- 8-sample triptych packet:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_frameclean_v1_8/triptych_packet`
- 32-sample compare:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_frameclean_v1_32`
- 32-sample triptych packet:
  - `/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_frameclean_v1_32/triptych_packet`

## Train

```bash
/root/autodl-tmp/repos/Janus/run_emoart_frameclean_v1.sh train
```

## Quick Compare

```bash
/root/autodl-tmp/repos/Janus/run_emoart_frameclean_v1.sh compare8
```

## Full Compare

```bash
/root/autodl-tmp/repos/Janus/run_emoart_frameclean_v1.sh compare32
```

## Triptych Packet

```bash
/root/autodl-tmp/repos/Janus/run_emoart_frameclean_v1.sh packet8
```

```bash
/root/autodl-tmp/repos/Janus/run_emoart_frameclean_v1.sh packet32
```

## What To Look For

This run is only promising if we see visual gains on the specific failure modes we already identified:

- less border / frame / scroll residue on non-scroll styles
- less East Asian panel leakage on unrelated prompts
- less repeated white-margin composition
- no further collapse in complex-line and ornate styles

If border bias drops but abstraction / line complexity still collapses, that means frame cleaning is only a partial fix and the next step should be style-aware filtering or reweighting.
