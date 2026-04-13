# Texture-Statistics Brush Head v1

## Motivation

The rebuilt-data `D1 -> D6` line showed that a safer curriculum exists, but the
current 512-channel `SpatialHFBrushHead` is nearing saturation. Free residual
editing can still leak into portrait structure when pushed, while more cautious
settings remain too weak by eye.

## Core change

`TextureStatisticsBrushHead` replaces free local residual prediction with a
constrained two-stage route:

1. predict explicit texture statistics on the decoder feature map
2. render a constrained residual from a fixed directional / band-pass basis

## Predicted statistics

For a decoder feature map `[B, 768, 24, 24]`, the head predicts:

- orientation logits `[B, 8, 24, 24]`
- scale logits `[B, 3, 24, 24]`
- density map `[B, 1, 24, 24]`
- pigment map `[B, 1, 24, 24]`
- band logits `[B, 3, 24, 24]`
- gate map `[B, 1, 24, 24]`

## Constrained renderer

The renderer uses a fixed orientation bank plus learned low-cost mixing:

- oriented derivative filters
- scale weighting
- band-pass weighting
- density/pigment modulation

The final residual is high-pass filtered before it is added back:

`feature_out = feature_in + gate * R_texture_hp * residual_scale`

## Supervision

The statistics branch is not left unsupervised. `finetune/brush_proxy_targets.py`
constructs per-sample proxy targets from the target image:

- dominant orientation bins from Sobel gradients
- three-band texture energy targets
- density target from local gradient magnitude and local complexity
- pigment target from local variance / contrast

## Losses

- `flow`
- `low_frequency_consistency`
- `texture_stats_orientation`
- `texture_stats_band`
- `texture_stats_density`
- `texture_stats_pigment`
- `basis_usage_entropy`
- optional legacy `laplacian`, `sobel`, `fft_high_frequency` as secondary aids

## Planned experiments

- `T1_texture_statistics_head_adapt_v1`
- `T2_texture_statistics_head_texture_v1`
- `T3_texture_statistics_head_recovery_v1`

## Expected failure modes

- may improve organized texture without yet producing truly painterly strokes
- may remain weaker than `E8 0.75` even if safer than `E11`
- if strong texture returns but structure leaks, move to V2 stroke-field branch
