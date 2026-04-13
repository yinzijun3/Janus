# Slot-Based Anchor-Set Renderer v2

Date: 2026-04-13

## Summary

`A1_slot_based_anchor_set_renderer_v1` is the first post-`CS1` branch that improves the target
`ink_upper_sparse` crop proxy instead of making it worse. That is a real positive signal.

However, the first long run also shows an equally important limit:

- `basis_usage_entropy_loss` flattens near the maximum-entropy regime
- `stroke_prototype_loss` also flattens early
- slot sparsity settles into a narrow conservative band by `checkpoint-225`

So the next step should not be `resume A1 unchanged`. The next step should be a **sharper slot branch**
that tries to break the early uniform-routing equilibrium.

## Why v2

`A1` tells us three things:

1. removing dense occupancy was useful
2. slot-level exclusion is at least not obviously harming the target portrait-sensitive crop
3. the current slot family is still too happy to spread probability mass broadly and use prototypes too uniformly

So the right next move is not a new renderer family yet. It is one more slot-family branch that explicitly pressures:

- fewer active slots
- fewer total slots competing per image
- less uniform prototype usage
- stronger rejection of exclusion-region anchors

## Proposed Changes vs A1

### Slot pressure

- reduce `slot_count`: `16 -> 12`
- reduce `prototype_count`: `16 -> 12`
- increase `valid_anchor_threshold`: `0.35 -> 0.45`

This should make it harder for the branch to settle into an "everyone is a little bit valid" regime.

### Support / exclusion pressure

- raise `support_threshold`: `0.28 -> 0.30`
- raise `support_mix`: `0.80 -> 0.85`
- strengthen exclusion:
  - `exclusion_logit_bias`: `-6.0 -> -8.0`
  - `exclusion_dilation`: `3 -> 5`

This keeps the branch closer to support evidence while making portrait/head-sensitive exclusion less negotiable.

### Routing pressure

- increase `stroke_prototype` weight
- increase `basis_usage_entropy` weight
- increase `slot_anchor_outside_support`
- increase `slot_anchor_inside_exclusion`
- increase `slot_component_conflict`
- increase `slot_count_sparsity`

The specific intent is to stop the branch from converging to high-entropy prototype routing with broad mild slot activation.

## Proposed Experiment

### A2: `slot_based_anchor_set_renderer_v2`

Init:
- `A1 checkpoint-225`

Data:
- `train_texture_balanced_v2`

Steps:
- `900`

Rationale:
- `A1` already showed its long-run trend by `225`
- a full `1125`-step repeat is not the right first use of budget
- `900` is enough to see whether sharper slot routing actually changes the regime

## Success Gates

- `ink_upper_sparse` improves again relative to `A1`
- `classical_face_window` does not regress past `CS1`
- `baroque_face_candle_violin` does not worsen materially
- global movement stays above `T2`
- compare cards look more like explicit stroke-object placement and less like broad organized texture

## Failure Interpretation

If `A2` still saturates early into uniform routing, the next branch should not be `A3` with more weight sweeps.
At that point the better next move is a **more explicit anchor-set / slot-program family**, where anchor ownership is
harder and prototype selection is more discrete by design rather than mainly encouraged by loss pressure.
