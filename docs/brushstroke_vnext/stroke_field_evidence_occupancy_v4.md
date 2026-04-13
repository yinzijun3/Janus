# Stroke-Field Evidence Occupancy V4

## Motivation

`S6` showed that soft blank-region suppression is directionally useful but not
enough. The branch became stronger while staying portrait/baroque-safe, yet
`ink_wash` still acquired invented sparse upper-page structure.

The failure is not best described as "missing one more regularizer." It is a
renderer-rule failure: once `objectness` activates, the current pseudo-renderer
can still organize plausible elongated marks in regions with weak underlying
support.

## Core Idea

Move sparse-region control from a purely loss-side hint into the runtime
renderer itself.

The next variant should require stroke occupancy to be backed by feature-space
support evidence before a mark can be rasterized. In practice this means:

1. Build a local support map from the current feature map using high-pass
   carrier energy plus density cues.
2. Convert that support map into a detached support gate.
3. Dilate the gate slightly so real marks can extend locally.
4. Multiply `objectness_blob` by that support gate before rendering.

This makes the branch pay a stronger architectural price for turning on
stroke-like marks in visually blank regions.

## Training Change

Add a `stroke_support_ceiling_loss`:

- target: a dilated support map derived from the target image
- form: penalize `relu(sigmoid(objectness) - support_target)`

This differs from the earlier blank suppression term:

- blank suppression says "do not activate in obviously blank places"
- support ceiling says "do not exceed where support plausibly exists"

The new loss is more direct and matches the runtime support-gate logic.

## Proposed Experiment

`S7_stroke_field_evidence_occupancy_v1`

Starting point:
- initialize from `S6 checkpoint-1200`

Key changes:
- enable renderer-side support gating
- use detached feature support evidence
- keep `topk_prototypes=1`
- keep longer primitive support
- add `stroke_support_ceiling_loss`

Success criterion:
- preserve `S6`-level or better portrait/baroque safety
- preserve or improve `S6` global strength
- reduce sparse-region hallucination on `ink_wash_mountain`

Failure criterion:
- if `ink_wash` still invents upper-page structure while metrics continue to
  rise, the next move should stop treating the renderer as a soft compositing
  device and move toward evidence-backed primitive rasterization or explicit
  blob-connectivity rules.
