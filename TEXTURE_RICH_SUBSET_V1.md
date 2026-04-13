# Texture-Rich Subset V1

This script builds a first-pass `texture-rich` training subset for a second-stage detail refinement run.

## Goal

Select samples that are more likely to carry useful painterly detail signals:

- visible brushstrokes
- layered paint texture
- dotted or dense mark making
- irregular painterly edges
- richer local material variation

while excluding high frame-risk samples that are likely to reinforce border / scroll / panel artifacts.

## Inputs

Recommended input:

- `/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v2.jsonl`

This keeps the strongest cleaned global training anchor while further focusing for detail-oriented second-stage refinement.

## Scoring Logic

Each record receives a `texture_rich_score` based on:

- strong `texture_tags`
- supportive `medium_tags`
- brushstroke and line-quality phrases that imply roughness, density, stippling, layering, or irregularity
- small style boost for the line-structure family that still suffers from detail collapse
- penalty for higher `frame_risk_score`

This is deliberately heuristic and transparent. It is not meant to be a final data curation method, only a strong first-pass filter.

## Output

The resulting manifest writes:

- original record fields
- `texture_rich_score`
- `texture_rich_reasons`
- `frame_risk_score`

## Example

```bash
python /root/autodl-tmp/repos/Janus/build_emoart_texture_rich_subset_v1.py \
  --input-manifest /root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_frameclean_v2.jsonl \
  --output-manifest /root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_texture_rich_v1.jsonl \
  --summary-path /root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_texture_rich_v1_summary.json
```

## Intended Use

This subset is intended for a short second-stage refinement run on top of a style-family expert or another already-correct style-direction branch.

It should not replace the main training set.
