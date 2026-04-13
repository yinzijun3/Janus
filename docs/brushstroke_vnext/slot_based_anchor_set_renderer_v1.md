# Slot-Based Anchor-Set Renderer v1

Date: 2026-04-13

## Summary

`CS1` shows that support topology matters, but it also shows that dense occupancy is still too permissive an abstraction.
Even after connected-component-aware support gating, the canonical `ink_wash` upper sparse region still moves too much,
and portrait-sensitive crop movement does not improve over `P1`.

The next branch should therefore stop predicting occupancy densely and instead predict a **small slot-based anchor set**.
Each slot should answer a discrete question:

1. is this anchor valid?
2. where is it?
3. what primitive does it render?
4. which support region owns it?
5. is it allowed under subject-safe exclusion?

This shifts the renderer away from "per-pixel objectness that later gets sparsified" toward "a small set of explicit stroke
objects that either exist or do not exist."

## Why This Next

The current evidence already narrows the decision:

- `T2` is a stable organized-texture baseline, but not a clear stroke-object regime
- `S7/S8` show that better occupancy regularization alone does not solve sparse-region hallucination
- `P1` shows sparse primitives are trainable
- `CS1` shows connected-support gating alone still does not fix the sparse-region problem

So the next useful change is not another exclusion tweak on a dense occupancy map. It is to remove the dense occupancy map
from the core renderer decision entirely.

## Proposed Architecture

### Inputs

- `decoder_feature_map`: `[B, 768, 24, 24]`
- `support_evidence`: derived from current support pipeline
- optional `subject_exclusion_hint`: from existing manifest fields

### Slot Outputs

For a fixed slot count `K` (recommended first pass: `K=16`), predict:

- `valid_logit`: `[B, K]`
- `center_xy`: `[B, K, 2]` in normalized `24x24` coordinates
- `theta`: `[B, K]`
- `length`: `[B, K]`
- `width`: `[B, K]`
- `alpha`: `[B, K]`
- `prototype_logits`: `[B, K, P]`
- `support_region_id` or soft support assignment: `[B, K, R]` or nearest-component assignment

### Rendering Rule

1. Build connected components from the binary support mask.
2. Predict `K` candidate anchors.
3. Keep anchors with `valid_logit > threshold`.
4. Snap each anchor to a support component:
   - if no component owns the anchor, invalidate it
   - if anchor lands inside a subject-safe exclusion region, invalidate it
5. Render one elongated primitive per remaining valid anchor.
6. Composite rendered primitives into the local residual path.

This turns the main branch decision into "which stroke objects exist?" instead of "which pixels have objectness?"

## Subject-Safe Exclusion

Unlike `CS1`, subject-safe exclusion should be active in the first slot-based run.
However, it should operate at the **slot level**, not as another dense per-pixel mask multiplier.

Recommended first version:

- build an exclusion map from:
  - `head_sensitive`
  - `style_portrait_dominant`
  - `portrait_likelihood`
  - upper-center ellipse prior
  - low-frequency saliency
- if an anchor center falls in exclusion:
  - penalize `valid_logit`
  - or force that slot invalid

## Losses

Keep:

- `flow`
- `low_frequency_consistency`
- `stroke_theta`
- `stroke_length`
- `stroke_width`
- `stroke_alpha`
- `stroke_prototype`
- `basis_usage_entropy`

Replace dense occupancy losses with slot losses:

- `slot_valid_bce`
- `slot_outside_support`
- `slot_inside_exclusion`
- `slot_component_conflict`
- `slot_count_sparsity`

Optional:

- `slot_center_snap`: encourages valid anchors to land near support peaks

## First Experiment Plan

### A1: `slot_based_anchor_set_renderer_v1`

Goal:
- beat `P1/CS1` on `ink_wash` upper sparse-region discipline
- remain at least as safe as `P1` on portrait/baroque

Init:
- warm start from `CS1 checkpoint-1125` only for compatible backbone weights
- new slot-specific heads start fresh

Data:
- `train_texture_balanced_v2`

Steps:
- `1125`

Success gates:

- `ink_wash_upper_sparse` crop proxy lower than `P1` and `CS1`
- no obvious portrait/head regression versus `P1`
- global pair diff stays materially above `T2`
- canonical baseline hash still matches

### A2: `slot_based_anchor_set_renderer_portrait_recovery_v1`

Open only if:
- A1 improves `ink_wash`
- but portrait safety is still slightly worse than desired

Data:
- `train_portrait_structure_v1`

Steps:
- `300`

## Operational Guardrails

- use numbered checkpoints only as authoritative eval/continuation sources
- fixed-baseline eval required
- zoom-crop export required
- branch decision must use:
  - authoritative summary
  - crop proxy
  - compare cards

## Decision Rule

- If A1 lowers `ink_upper_sparse` movement and keeps portrait/baroque safe, continue the slot-based family.
- If A1 still fails `ink_wash`, the next branch should move one level more explicit again:
  - a support-owned anchor list with stronger discrete assignment
  - or a hierarchical stroke program with even fewer, more explicit primitives
