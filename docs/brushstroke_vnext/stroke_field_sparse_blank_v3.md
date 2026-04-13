# Stroke Field Sparse Blank Suppression v3

## Motivation

`S5` showed that objectness supervision and sparse top-k prototype routing move
the V2 branch beyond `S4`, but the renderer can still activate in low-content
regions and invent new local forms, especially on sparse `ink wash` prompts.

The next smallest useful change should therefore target two things directly:

1. make each local stroke decision more singular and elongated;
2. make blank/low-support regions more expensive to activate.

## Design

`S6` keeps the stroke-field family and rebuilt texture-balanced curriculum, but
adds one new supervision term and one stricter renderer regime.

### 1. Blank-region suppression

`brush_proxy_targets.py` now derives `blank_region_target` from the local
support signal. A location is treated as blank when both gradient-density and
pigment variation are weak. Training adds:

- `stroke_blank_suppression_loss`

This loss penalizes predicted objectness inside low-support regions rather than
only asking objectness to match a generic occupancy target.

### 2. Sparser primitive routing

`S6` sets:

- `topk_prototypes = 1`
- `renderer_kernel_size = 13`

The goal is to reduce soft averaging between multiple local prototypes and make
each activated mark look more like a single elongated primitive rather than a
small blended texture patch.

## Expected behavior

Relative to `S5`, the desired tradeoff is:

- equal or stronger local brushstroke visibility on `impressionist`,
  `expressionist`, and `ukiyoe`
- no worse `baroque` / `portrait` safety
- reduced hallucinated form insertion in low-content `ink wash` regions

If `S6` stays safe but weakens too much, the next change should focus on richer
elongated primitives rather than removing blank suppression. If `S6` still adds
new forms in blank regions, the next branch should move from proxy-guided
patches toward a more explicit stroke rasterizer.
