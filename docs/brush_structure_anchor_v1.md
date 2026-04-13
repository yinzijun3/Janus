# Brush Structure Anchor V1

## Summary

`E12` and `E14` show that locality alone is not enough. Feature-map gating and later output-space injection both keep many prompts relatively safe, but neither prevents the recurrent `baroque_drama` head/face distortion. The next experiment should therefore add an explicit structure-preservation mechanism rather than another same-family strength change.

## Proposed mechanism

`E15` introduces a structure-anchored output velocity head:

- keep the no-reference policy;
- keep the edit in output-velocity space, as in `E14`;
- compute a low-frequency structure-energy map from the incoming base velocity;
- convert that map into a protection mask;
- attenuate the learned texture residual inside protected regions.

Concretely:

1. Predict a gated high-frequency residual `gate * delta_hp`.
2. Compute `structure_energy = blur(abs(pred_velocity).mean(channel))`.
3. Normalize per image and map to a soft protection mask.
4. Apply
   `pred_velocity_out = pred_velocity + gate * (1 - anchor_strength * protect_mask) * delta_hp * residual_scale`.

This keeps background and material zones editable while reducing the edit budget in regions already carrying strong low-frequency structure.

## Why this is the next smallest useful step

- It preserves the clean E14 comparison: same insertion point, same trainable base modules, one new mechanism.
- It does not require external face detectors or new dependencies.
- It is explainable in paper terms: an explicit structure-preservation prior is added to the local texture head.

## Acceptance criteria

- Must reduce `baroque_drama` head/face deformation relative to `E14`.
- Must keep at least E12-level visible texture gain on `impressionist`, `expressionist`, and `ink`.
- If it remains conservative everywhere, the next change should not be another anchor-strength sweep alone; it should add a prompt-aware or region-aware masking source.

## E16 follow-up

`E15` shows that self-derived low-frequency structure masking is not enough by itself. `E16` therefore adds a simple but explicit spatial prior on top of the same anchored output head:

- preserve the self-derived structure mask from `E15`;
- add an upper-center Gaussian protection prior;
- combine both masks before applying the residual attenuation.

This is intentionally narrow. It tests whether the missing piece is a stronger subject-region prior rather than another change in insertion point or residual strength.
