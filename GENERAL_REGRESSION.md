# General Regression Guide

This project now has two lightweight regression scripts for checking whether an EmoArt adapter changes Janus-Pro's original abilities outside the EmoArt task.

## 1. Image Understanding Regression

Script:

`/root/autodl-tmp/repos/Janus/general_understanding_regression.py`

Manifest template:

`/root/autodl-tmp/repos/Janus/regression_manifests/understanding_template.jsonl`

Starter manifest with broader coverage:

`/root/autodl-tmp/repos/Janus/regression_manifests/understanding_v1.jsonl`

Expanded starter manifest with fuller coverage:

`/root/autodl-tmp/repos/Janus/regression_manifests/understanding_v2.jsonl`

Each JSONL row supports:

- `id`: sample id
- `category`: optional grouping label
- `image_path`: absolute image path
- `prompt`: question shown to the model
- `reference`: optional reference answer string
- `expected_keywords`: optional keyword list that should appear
- `forbidden_keywords`: optional keyword list that should stay absent

Typical run:

```bash
python general_understanding_regression.py \
  --adapter-path /root/autodl-tmp/emoart_gen_runs/out_gen_expA_full/final_adapter \
  --manifest-path /root/autodl-tmp/repos/Janus/regression_manifests/understanding_template.jsonl \
  --output-dir /root/autodl-tmp/emoart_gen_runs/general_understanding_regression_expA
```

Outputs:

- `results.jsonl`: per-sample answers from base and adapter
- `summary.json`: aggregate keyword and optional ROUGE summary
- `report.md`: readable comparison report

## 2. Open-World Generation Regression

Script:

`/root/autodl-tmp/repos/Janus/general_generation_regression.py`

Manifest template:

`/root/autodl-tmp/repos/Janus/regression_manifests/generation_template.jsonl`

Starter manifest with multiple non-EmoArt prompt categories:

`/root/autodl-tmp/repos/Janus/regression_manifests/generation_v1.jsonl`

Expanded starter manifest with broader open-domain stress coverage:

`/root/autodl-tmp/repos/Janus/regression_manifests/generation_v2.jsonl`

Each JSONL row supports:

- `id`: sample id
- `category`: optional grouping label
- `prompt`: open-world generation prompt

Typical run:

```bash
python general_generation_regression.py \
  --adapter-path /root/autodl-tmp/emoart_gen_runs/out_gen_expA_full/final_adapter \
  --manifest-path /root/autodl-tmp/repos/Janus/regression_manifests/generation_template.jsonl \
  --output-dir /root/autodl-tmp/emoart_gen_runs/general_generation_regression_expA
```

Outputs:

- `base/`: images from original Janus-Pro
- `adapter/`: images from Janus-Pro plus EmoArt adapter
- `sheets/`: side-by-side review sheets
- `results.jsonl`: paths and optional CLIPScore values
- `summary.json`: aggregate summary
- `report.md`: review index
- `manual_review_sheet.md`: row-by-row human judgment sheet

## 3. How To Use This In Practice

Start small:

1. Add 20 to 50 image-understanding samples that are not from EmoArt.
2. Add 20 to 50 open-world generation prompts that are not art-specific.
3. Run both regressions on base vs Exp A adapter.
4. Check whether the adapter introduces clear style pollution, factual drops, or instruction-following regressions.

If you want to start immediately without drafting manifests from scratch, begin with:

1. `understanding_v1.jsonl`
2. `generation_v1.jsonl`

If you want a more decision-grade first pass without building your own benchmark yet, use:

1. `understanding_v2.jsonl`
2. `generation_v2.jsonl`

Suggested categories:

- understanding: OCR, objects, layout, chart reading, human actions, fine-grained details
- generation: product photo, realistic portrait, architecture, animals, simple illustrations, graphic design

## 4. Decision Rule

Treat the adapter as safe only if:

- EmoArt task quality improves clearly
- general understanding does not show obvious drops
- general generation does not become uniformly more painterly or less controllable
