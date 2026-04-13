# Stroke-Field Supervision v1

## Why this step exists

`S2` and `S3` showed that the V2 stroke-field branch can train stably and stay
portrait-safe, but the renderer still tends to turn stroke parameters into
organized directional texture rather than clearly legible stroke objects.

The main missing piece is supervision. The branch predicts `theta`, `length`,
`width`, `alpha`, and `prototype_logits`, but until now only the downstream
flow/high-frequency objectives constrained those fields.

## Design

The first supervision pass stays fully in-repo and uses fixed proxy targets
derived from the target image:

- axial dominant orientation from Sobel gradients
- structure-tensor coherence to derive preferred stroke length and width
- density / local variance to derive stroke occupancy (`alpha`)
- nearest prototype assignment from orientation/length/width anchors

These targets are cheap, local, and compatible with the existing
`brush_proxy_targets.py` path.

## Losses

New optional loss terms:

- `stroke_theta`
- `stroke_length`
- `stroke_width`
- `stroke_alpha`
- `stroke_prototype`

`basis_usage_entropy` is also enabled for the stroke head so prototype selection
does not stay unnecessarily diffuse.

## Intended outcome

This step is meant to answer a narrow question:

Can the current V2 branch become visibly more stroke-like if the predicted field
is explicitly supervised toward brush proxies, or does it still collapse back to
generic texture modulation?
