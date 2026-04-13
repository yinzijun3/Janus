# Stroke-Field Branch v2

## Positioning

This branch is the next upgrade path if `TextureStatisticsBrushHead` still
cannot open a clearly stronger brushstroke regime.

## Goal

Represent brushstroke edits as explicit stroke-like fields instead of a generic
residual or texture-statistics-only field.

## Current implementation scope

This round only adds a smokeable skeleton:

- `StrokeFieldBrushHead`
- `StrokeFieldPseudoRenderer`

No external differentiable renderer is introduced. The renderer stays
in-repo and intentionally lightweight.

## Predicted fields

For decoder feature maps, the probe head predicts:

- `theta`
- `length`
- `width`
- `curvature`
- `alpha`
- `prototype_logits`

## Renderer

The pseudo-renderer composes a stroke-like residual from a small prototype bank
and the predicted local stroke fields. This is only meant to verify runtime
integration, checkpointing, and future training interfaces.

## Upgrade trigger

V2 becomes the main experiment path if V1:

- remains clearly weaker than the target visual strength of `E8 0.75`
- or regains portrait/baroque leakage when pushed harder
- or only produces "better organized texture" instead of visible brush logic
