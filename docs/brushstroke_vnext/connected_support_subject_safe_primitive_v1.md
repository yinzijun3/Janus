# Connected-Support Subject-Safe Primitive v1

## Summary

This note closes the old occupancy-gated stroke-field family and defines the
next primitive-renderer branch.

`P1_support_locked_primitive_renderer_v1` showed that sparse primitive
rendering is trainable, but it still allowed two unacceptable behaviors:

- `ink_wash_mountain` could still grow a large unsupported upper-page tree form
- `classical_portrait` could still drift in the dominant head/face region

The next branch therefore keeps sparse primitives, but tightens both:

1. support topology
2. subject-safe exclusion

## CS1: Connected-Support Primitive

The first branch upgrade is component-aware anchor selection.

Changes relative to `P1`:

- build connected components on the binary `24x24` support mask
- drop components with `area < 4`
- allow at most `1` anchor per normal component
- allow at most `2` anchors for large components (`area >= 20`)
- cap total anchors to `16`
- clip each primitive to its owning component mask instead of a generic local
  support patch

This is intended to stop isolated support fragments from behaving like valid
stroke sites.

## CS2: Connected-Support + Subject-Safe Exclusion

The second branch upgrade is explicit portrait/head-safe exclusion.

For head-sensitive samples, build an exclusion map from:

- an upper-center ellipse prior
- low-frequency subject saliency from the decoder feature map
- a small dilation pass

Inside that exclusion map:

- anchor logits receive a strong negative bias
- anchor selection is masked out
- training adds explicit penalties for anchors or occupancy mass that still
  overlap the exclusion region

This keeps the primitive family focused on painterly structure rather than face
overwriting.

## Acceptance Gates

The branch is only useful if it improves the actual failure modes:

- `ink_wash_mountain` upper sparse region must stop growing a new tree form
- `classical_portrait` and `baroque_drama` must remain at least as safe as the
  current V2 safety floor
- movement/detail should stay in the late V2 range rather than collapsing back
  toward `T2`
