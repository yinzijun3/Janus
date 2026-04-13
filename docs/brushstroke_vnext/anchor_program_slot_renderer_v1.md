# Anchor-Program Slot Renderer v1

Date: 2026-04-13

## Why This Next

`A1` and `A2` together give a pretty clean result:

- removing dense occupancy was useful
- but the current soft slot formulation still collapses into near-uniform routing
- stronger thresholds and stronger losses do not fix that collapse

So the next useful change should not be another slot-weight sweep. It should make slot ownership more explicit by design.

## Core Idea

Replace "soft slot attention over support" with a small **anchor program**:

1. predict a fixed small set of slot proposals
2. assign each slot to at most one support-owned anchor site
3. choose one prototype per active slot
4. render one primitive per active slot

In other words, a slot should no longer be "a soft probability cloud over many possible anchors." It should become
"one concrete anchor decision plus one concrete primitive choice."

## Design Sketch

### Inputs

- `decoder_feature_map [B, 768, 24, 24]`
- support evidence from the current support builder
- existing subject-safe exclusion prior

### Program outputs

For `K` small slots, predict:

- `slot_on_logit [B, K]`
- `anchor_logits [B, K, H, W]`
- `prototype_logits [B, K, P]`
- `theta [B, K]`
- `length [B, K]`
- `width [B, K]`
- `alpha [B, K]`

### Harder ownership rule

- anchor logits are masked by support and exclusion before normalization
- each active slot uses a much sharper anchor selection rule
- each slot is expected to commit to one anchor site rather than distribute mass broadly
- optional future step: top-1 / straight-through anchor selection

## Why It Is Different From A1/A2

`A1/A2` still let a slot behave like a soft region selector.
This next family should make a slot behave more like a discrete brushstroke instruction.

That is the missing piece suggested by current evidence.

## Success Criteria

- improves `ink_upper_sparse` again versus `A1`
- does not worsen portrait-sensitive crop movement
- avoids immediate maximum-entropy prototype routing
- visually starts to look more like sparse stroke-object placement and less like organized local texture
