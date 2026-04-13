# Support-Locked Primitive Renderer v1

## Summary

`P1_support_locked_primitive_renderer_v1` closes the S7/S8 occupancy-gated branch and replaces the semi-implicit
`objectness_blob * carrier` compositor with a sparse anchor renderer.

The main design goal is simple: sparse low-content regions should no longer be able to grow large new structures from a
dense activation map. A stroke must now be attached to an explicit anchor, and that anchor must lie inside a
support-backed region.

## Motivation

`S7` and `S8` showed the same split outcome:

- portrait and baroque prompts remained safe
- pair-difference and detail proxies kept improving
- `ink_wash_mountain` still hallucinated a tree-like structure in the upper blank region

This means support gating is directionally useful, but not sufficient while the renderer still behaves like a soft
occupancy field multiplied by a feature carrier. The next architecture step should therefore change the rendering rule,
not just the gate strength.

## Design

Input:

- decoder feature map `[B, 768, 24, 24]`

Predicted fields:

- `occupancy_logits [B, 1, 24, 24]`
- `theta [B, 1, 24, 24]`
- `length [B, 1, 24, 24]`
- `width [B, 1, 24, 24]`
- `alpha [B, 1, 24, 24]`
- `prototype_logits [B, P, 24, 24]`

Renderer steps:

1. Build a binary `support_mask` from carrier high-pass evidence and local density.
2. Apply `3x3` NMS to `occupancy_logits`.
3. Keep only anchors that survive NMS and lie inside `support_mask`.
4. Keep at most `top_k_anchors = 24` anchors per image.
5. Render one elongated primitive per anchor.
6. Clip each primitive footprint to the local support patch before compositing.

This means the renderer is sparse by construction. It does not synthesize a free per-pixel objectness blob and then
hope auxiliary losses will suppress mistakes later.

## Losses

Retained:

- flow
- low-frequency consistency
- stroke theta / length / width / alpha
- stroke prototype
- basis usage entropy

New occupancy losses:

- `stroke_occupancy_bce`
- `stroke_anchor_outside_support`
- `stroke_anchor_sparsity`

Removed from the P1 mainline:

- `stroke_blank_suppression`
- `stroke_support_ceiling`

Those older losses stay in the codebase for historical branches, but P1 does not mix them with the new sparse-anchor
occupancy objective.

## Acceptance Target

P1 is successful if it does all three:

1. reduces the `ink_wash` blank-region hallucination relative to `S7`
2. preserves `classical_portrait` and `baroque_drama` safety
3. keeps pair-difference in the same broad range as the late V2 runs (`>= 7.5`)

If P1 still fails `ink_wash`, the next step should be a stronger primitive-support rule or a more explicit stroke
object renderer, not another occupancy-threshold sweep.
