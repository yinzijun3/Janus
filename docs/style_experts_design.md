# Style Experts Design

## Why style experts may help

JanusFlow-Art is expected to serve art generation where stylistic variance is large and often structurally different across families such as classical academic painting, expressive painterly work, graphic flat design, and ink-based traditions. A single shared conditioning pathway can learn broad correlations, but narrower expert subspaces may help preserve:

- family-specific texture statistics
- family-specific edge handling
- family-specific color organization
- composition priors tied to historical media and schools

Experts are most likely to help once the baseline dual-condition model already works and the main failure mode becomes cross-style interference rather than basic instability.

Before attributing weak deltas to limited adapter capacity, baseline-vs-expert comparisons must be run with the
generation stack in module eval mode. Otherwise train-time adapter dropout can mask or blur the true effect size.
The 2026-04-10 larger-LoRA probe also showed the opposite failure mode: stronger adapters can quickly improve visible
texture while rewriting composition. Future expert work should include an inference strength gate or residual scale
sweep before comparing expert capacity.
For the brushstroke phase, experts should be decoder-local residual branches behind the brush adapter router. LLM MoE is
out of scope until decoder-local experts show clear gains without composition drift.
The 2026-04-10 adapter and high-frequency screens did not yet show enough visible brushstroke gain to justify expanding
expert capacity first. A follow-up audit found CFG common-mode cancellation and missing reference prompts in E3 review,
so decoder experts should wait until the corrected E1/E2/E3 runs have been reviewed with explicit `E1_`, `E2_`, `E3_`
output directories.
E3D later completed as a stronger-adapter, fine-token-only reference screen. Its metrics moved in the desired detail
direction, but the expert path should still stay deferred until manual E3D review confirms local brushstroke gains without
reference-driven composition drift.
The later E8-E16 sequence sharpened that constraint further: visible brushstroke strength is possible, but the remaining
failure mode is not obviously "missing expert capacity" alone. Before revisiting experts, the project should validate the
full-data rebuild and balanced continuation manifests so expert comparisons are not confounded by a texture-biased subset
that under-covers portrait- and baroque-sensitive recovery cases.

## Lightest viable implementation

The lightest expert design should not introduce a full MoE transformer first.

Recommended first implementation:

- keep one shared JanusFlow backbone
- keep one shared style encoder
- add a small bank of expert adapters or expert residual blocks only on the generation side
- route style-conditioned features into one or a soft mixture of experts

This keeps the understanding path untouched and limits extra memory to a narrow part of the generation branch.

## Best layer placement

Experts should sit where style-specific image synthesis decisions are strongest and least likely to damage prompt semantics.

Recommended placement order:

1. decoder-token path before `vision_gen_dec_model`
2. latent-token path immediately before the language model image-token segment
3. optionally generation-side aligners if extra capacity is needed

Experts should not be added to the understanding encoder in the first implementation.

## Router input

The router should primarily consume:

- global style embedding from the style encoder
- optional period and medium embeddings
- optional pooled prompt representation

The first router should not depend on generated-image feedback loops or unstable online metrics. A stable label-and-style driven router is easier to train and debug.

## Stability and memory risks

Main risks:

- extra experts increase memory usage quickly on the generation branch
- sparse routing can become unstable when style labels are imbalanced
- experts may collapse to one dominant route
- expert-specific conditioning can overfit texture while drifting from prompt semantics

Recommended safeguards:

- begin with a very small expert count
- keep a shared backbone and only specialize narrow residual paths
- use route entropy monitoring
- keep prompt adherence metrics in the evaluation loop
- gate experts off by config so the baseline path always remains available
