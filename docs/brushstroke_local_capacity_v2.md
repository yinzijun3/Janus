# Brushstroke Local Capacity V2

Date: 2026-04-11

## Objective

Match roughly the visible effect size of E8 at `lora_scale=0.25` without inheriting full LoRA's semantic drift at higher strengths.

This design assumes:
- reference cross-attention is no longer the primary route;
- a single decoder-token adapter is too weak;
- the next architecture should stay local to the generation path and preserve composition by default.

## Proposed Direction

Use a two-stage local capacity path on the image-generation side:

1. Token-stage local adapter:
   - keep the current decoder-token adapter path after `vision_gen_dec_aligner`;
   - allow larger residual capacity than v1, but do not rely on it alone.

2. Feature-map-stage local residual block:
   - add a new brush module on the reshaped decoder feature map (`[batch, 768, 24, 24]`) immediately before `vision_gen_dec_model`;
   - this block should use 2D convolutions / residual bottlenecks so it acts directly on local image structure rather than only token embeddings;
   - gate it independently from the token adapter.

The default v2 route should be no-reference:
- no brush reference encoder;
- no decoder cross-attention injector;
- no semantic style tokens entering the decoder path.

## Minimal Implementation Plan

- Add `BrushFeatureMapAdapter` under `janus/janusflow/models/brush/`.
- Build it in `finetune/janusflow_art_runtime.py` from config as an optional module, parallel to `BrushAdapter`.
- Apply it after decoder tokens are reshaped into the feature map and before `vision_gen_dec_model(...)`.
- Expose config switches:
  - `brush.feature_map_adapter.enabled`
  - `brush.feature_map_adapter.hidden_channels`
  - `brush.feature_map_adapter.kernel_size`
  - `brush.feature_map_adapter.residual_init`
  - `brush.feature_map_adapter.residual_scale`
  - `brush.feature_map_adapter.dropout`
- Keep `brush.adapter.*` intact so token-stage and feature-map-stage adapters can be ablated independently.

## Training/Eval Policy

- Start from no-reference runs only.
- Use a 450-step screen on the texture-rich subset.
- Keep `condition_unconditioned: false`.
- Compare against E7 and E8 (`lora_scale=0.25`) using the same fixed prompt set.
- Add zoom-crop exports to evaluation before judging success, but require full-card review to remain composition-safe.

## Success Bar

- Stronger visible material/brushstroke change than E7.
- Closer to E8 `lora_scale=0.25` effect size than to E7.
- No portrait identity breakage.
- No expressionist/ink/ukiyo-e scene rewrites of the kind seen in high-scale LoRA or reference-conditioning runs.

## E9 Readout

E9 completed with the hybrid token-stage + feature-map-stage setup.

What improved:
- much safer than the old reference-conditioned branch;
- slightly richer local material signal than E7;
- no sign that the feature-map adapter itself destabilizes training.

What still failed:
- the full-card effect is still weaker than E8 `0.25`;
- some cards still read as mild repainting rather than decisive new brushstroke structure;
- baroque / portrait cards still show face-head distortion, which suggests the token-stage route may still carry too much global leverage.

## Next Assumption

The next v2 ablation should be feature-map dominant:

- disable the token-stage adapter entirely;
- keep the feature-map adapter and raise its hidden size / gate / kernel modestly;
- keep the run no-reference and compare directly against E7, E8 `0.25`, and E9.

If that still does not reach the target, the next architecture should move beyond plain residual adapters toward a stronger
spatially gated local head rather than returning to reference cross-attention.
