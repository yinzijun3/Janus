# EmoArt Experiment System

This document describes the refactored experiment layout for EmoArt generation work.

## Goals

- keep historical one-off scripts intact
- stop mixing new runs with unrelated legacy outputs
- support both `Janus-Pro-1B` and `Janus-Pro-7B`
- treat multi-expert LoRA as the new mainline without rewriting the trainer
- isolate stage-1 family training and stage-2 texture refinement per expert

## Repo Layout

- `emoart_lab/`
  - thin orchestration layer for experiment preparation and run launching
  - builds per-expert manifests
  - materializes isolated run specs
  - launches existing training/eval scripts with frozen snapshots of config
- `configs/emoart/`
  - project-level path config
  - track-level model/expert/stage config
- `run_emoart_lab.sh`
  - shell entrypoint for prepare/list/launch commands

The training core remains in the existing scripts:

- `train_emoart_gen_lora.py`
- `compare_emoart_gen.py`
- `build_emoart_compare_triptych_packet.py`

## Output Layout

New organized runs live under:

`/root/autodl-tmp/emoart_gen_runs/organized/<project_name>/<track_name>/`

Each prepared track contains:

- `configs/`
  - frozen config snapshots used to prepare the track
- `data/family_manifests/<expert>/train.jsonl`
  - narrow expert train manifest
- `data/family_manifests/<expert>/val.jsonl`
  - narrow expert val manifest
- `data/texture_refine/<expert>/train.jsonl`
  - intersection of that expert's train set with the texture-rich subset
- `runs/<expert>/train_stage1_family/artifacts/`
  - stage-1 training outputs for that expert only
- `runs/<expert>/train_stage2_texture_refine/artifacts/`
  - stage-2 refinement outputs for that expert only
- `runs/<expert>/compare_.../artifacts/`
  - evaluation outputs for one specific manifest/sample-count combination

This means a single run directory always corresponds to one expert and one stage.

## Current Mainline Track

The first refactored mainline track is:

- `configs/emoart/project_organized_v1.json`
- `configs/emoart/tracks/januspro7b_multi_expert_mainline_v1.json`

It is built around:

- model: `deepseek-ai/Janus-Pro-7B`
- experts:
  - `decorative_linework`
  - `classical_structure`
- stages:
  - `stage1_family`
  - `stage2_texture_refine`

## Commands

Prepare the track:

```bash
/root/autodl-tmp/repos/Janus/run_emoart_lab.sh prepare \
  --project-config /root/autodl-tmp/repos/Janus/configs/emoart/project_organized_v1.json \
  --track-config /root/autodl-tmp/repos/Janus/configs/emoart/tracks/januspro7b_multi_expert_mainline_v1.json
```

List materialized runs:

```bash
/root/autodl-tmp/repos/Janus/run_emoart_lab.sh list-runs \
  --track-dir /root/autodl-tmp/emoart_gen_runs/organized/emoart_style_experts/januspro7b_multi_expert_mainline_v1
```

Launch one run in the background:

```bash
/root/autodl-tmp/repos/Janus/run_emoart_lab.sh launch \
  --track-dir /root/autodl-tmp/emoart_gen_runs/organized/emoart_style_experts/januspro7b_multi_expert_mainline_v1 \
  --expert decorative_linework \
  --run-name train_stage1_family \
  --background
```
