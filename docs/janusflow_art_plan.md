# JanusFlow-Art Implementation Plan

## Summary

JanusFlow-Art is implemented as a dedicated training, sampling, and evaluation stack built on the real JanusFlow generation branch. The existing Janus-Pro and EmoArt experiment codepaths remain intact. This branch is intended to be the long-term research foundation for art-focused flow generation rather than a temporary smoke-test scaffold.

The first implementation round establishes:

- a YAML-first configuration system for JanusFlow-Art
- a true rectified-flow trainer around `vision_gen_enc_model -> language_model -> vision_gen_dec_model -> SDXL VAE`
- a unified art-data parser and slot-based prompt builder
- pluggable style conditioning with global and local tokens
- configurable art-domain alignment losses
- JanusFlow-specific sampling, comparison, and blind-review export

## Architecture

### Core runtime

- `train_janusflow_art.py`
  - training entrypoint
  - loads YAML config
  - builds JanusFlow base model, VAE, style modules, optimizer, scheduler, and checkpoint manager
- `sample_janusflow_art.py`
  - deterministic prompt and seed sampling
  - inference pipelines must run with module eval mode so LoRA dropout is disabled
  - checkpoint restore must reuse the saved PEFT adapter instead of stacking a fresh wrapper
  - restored LoRA adapters can be strength-scaled at inference with `freeze.language_lora.inference_scale`
  - exports single images, grids, and rendered prompt logs
- `eval_janusflow_art.py`
  - compares base JanusFlow against a tuned checkpoint
  - keeps adapter comparisons in eval mode so tuned deltas reflect learned weights rather than train-time dropout
  - uses the restored checkpoint adapter directly so base vs tuned comparisons are not distorted by nested PEFT wrappers
  - supports `--lora-scale` sweeps for quality-preserving adapter strength selection
  - exports metrics, grids, blind-review bundles, and a markdown report

### Shared modules

- `finetune/janusflow_art_config.py`
  - YAML loading and CLI override support
- `finetune/janusflow_art_data.py`
  - unified art record parsing
  - dataset and collator implementations
  - label vocab construction
- `finetune/janusflow_art_prompting.py`
  - slot-based prompt builder
  - conservative and strong-style templates
- `finetune/janusflow_art_losses.py`
  - flow loss
  - art alignment loss
  - style classification loss
  - texture auxiliary loss
- `finetune/janusflow_art_runtime.py`
  - JanusFlow-Art pipeline assembly
  - checkpoint save/load helpers
  - prompt encoding, latent encoding, denoiser forward pass, and sampling helpers

### Style conditioning

- `janus/janusflow/models/style_encoder/label_style_encoder.py`
  - label-only style encoding from style, period, and medium IDs
- `janus/janusflow/models/style_encoder/reference_style_encoder.py`
  - reference-image style encoding from optional style-image tensors
- `janus/janusflow/models/conditioning/style_injector.py`
  - earlier global-style injection into `z_emb` and `t_emb`
  - later local-texture injection into decoder tokens before `vision_gen_dec_model`

## Phase Mapping

### Phase 0

- keep this plan updated
- track environment assumptions in `docs/assumptions.md`

### Phase 1 baseline

- baseline config in `configs/janusflow_art/baseline.yaml`
- understanding path frozen by default
- generation-side aligners and late-layer LoRA trainable by default
- sampling exports include rendered prompts and aligned grids
- quality-conscious larger-LoRA probe in `configs/janusflow_art/lora_quality_probe.yaml`
- the probe freezes generation aligners, expands attention-only LoRA to the last 16 layers, and keeps auxiliary style losses off until the style targets are better validated
- the first 450-step probe made visual changes obvious, but full-strength LoRA often rewrote composition; review favored an inference scale around `0.25` for a better quality/texture tradeoff
- brushstroke-focused experiments now move trainable capacity to decoder-local modules under `configs/janusflow_art/brushstroke/`
- the brushstroke v1 path freezes LLM/base generation modules by default and trains only config-enabled brush adapter/reference/injector modules
- 2026-04-10 brushstroke stage status: `brush_adapter_v1` and `brush_adapter_hf_loss_v1` finished 450-step screening runs. Both preserved composition, but the visible brushstroke gain was modest; the high-frequency loss raised Laplacian/HF metrics slightly without a clear visual win. The next screening run is `brush_multiscale_ref_v1` on the texture-rich manifest.
- 2026-04-10 evaluator correction: base-vs-tuned review now constructs the baseline pipeline with JanusFlow-Art LoRA, conditioning, and brush modules disabled. Fixed E1/E2/E3 baseline PNGs match by hash across experiments.
- 2026-04-10 `brush_multiscale_ref_v1_450_fixed` completed with 4,017,857 trainable parameters. Metrics improved slightly over E1/E2, but manual review still found the effect conservative, so `brush_ablation_no_ref` is the next attribution run before longer training or decoder experts.
- 2026-04-10 follow-up audit found the conservative effect was partly a logic/evaluation issue, not just capacity: brush modules were being applied to both CFG halves, which can suppress local texture deltas in the final guidance combination, and E3 fixed-prompt review had zero reference images. Brush configs now use explicit `E1_`/`E2_`/`E3_` output directories, condition only the CFG conditioned branch by default, and evaluate reference-based experiments with `prompts_brush_reference.jsonl`.

### Phase 2 dual condition

- label-only and reference-style encoders both available
- global and local conditioning can be independently toggled

### Phase 3 art alignment

- flow loss remains the main objective
- optional `art_align`, `style_cls`, and `texture_aux` losses

### Phase 4 data and prompting

- unified field protocol:
  - `prompt`
  - `style_label`
  - `period_label`
  - `medium_label`
  - `reference_style_image`
  - `texture_metadata`
- missing-field tolerant parsing
- rendered prompts logged for training and inference
- 2026-04-11 data rebuild direction: the default brushstroke research path should move off the legacy
  `train_texture_rich_v1` subset and onto a full official rebuild rooted in `annotation.official.json`
  with three manifest tiers:
  `train_full_balanced_v1`, `train_texture_balanced_v2`, and `train_portrait_structure_v1`
- the rebuild must keep explicit `rendered_prompt` fields so structure-safe and texture-push prompt variants
  can be selected per sample without being rewrapped by a generic template
- 2026-04-11 implementation status: the rebuild is now materialized at
  `/root/autodl-tmp/data/emoart_5k/gen_full_official_rebuild_v1` with
  `5532` master records, `560` validation records, `4972` main-train records,
  `1568` texture-continuation records, and `440` portrait-recovery records
- the first full-balanced and texture-balanced JanusFlow-Art smoke runs now pass with zero missing-image records after a
  loader fix that preserves non-breaking spaces inside official filenames
- 2026-04-11 D1 full-balanced run completed: the rebuilt stage-1 data path produces a clearly safer `SpatialHFBrushHead`
  base than the earlier texture-first line, but the visual effect remains intentionally conservative
- 2026-04-11 D2 texture-balanced continuation completed from the D1 checkpoint: metrics and manual review both show a
  modest but real increase in visible texture without returning to the earlier `baroque/portrait` collapse pattern
- current implication: portrait recovery is not the next bottleneck; the next useful experiment should strengthen local
  capacity or continuation pressure on top of the rebuilt D1/D2 curriculum rather than immediately invoking D3
- 2026-04-11 evaluator correction: the baseline builder now also disables `spatial_hf_head`, `output_velocity_head`,
  and `anchored_output_velocity_head`; earlier comparison bundles for those newer head families should not be treated as
  true-base references unless they were rerun after the fix
- 2026-04-11 correction sweep completed: E12, E14, E15, and E16 now have authoritative `evaluation_fixed_baseline`
  bundles; the correction changes baseline statistics but does not change the main architectural ranking
- 2026-04-11 D4 completed as a stronger texture-balanced continuation from D2 (`residual_scale=1.8`,
  slightly stronger Laplacian / HF losses). It produced a small but clean gain over D2 while remaining portrait-safe.
- 2026-04-11 D5 completed as a gate-relaxed continuation from D4 (`residual_scale=2.1`, lower gate regularization,
  stronger HF penalties). It nudged pair-difference upward but showed clear diminishing returns: visual effect remained
  close to D4 and texture proxies did not keep improving monotonically.
- 2026-04-11 D6 completed as a short portrait-structure recovery continuation from D5 on
  `train_portrait_structure_v1.jsonl`. It slightly improved the D5 tradeoff and is currently the best checkpoint inside
  the rebuilt-data 512-channel spatial-HF family, but it still does not reach the target visible strength of E8 `0.75`.
- current implication: the D4-D6 line validates the three-stage rebuilt-data curriculum, but also shows that the current
  512-channel `SpatialHFBrushHead` family is nearing saturation. The next smallest useful experiment should reallocate
  local capacity or change the local editing architecture rather than applying another minor D4/D5-style pressure sweep.
- 2026-04-11 VNext implementation has now started. The immediate mainline is
  `configs/janusflow_art/brushstroke/vnext/texture_statistics_head_adapt_v1.yaml`,
  which replaces free local residual prediction with a supervised
  texture-statistics head and a constrained basis renderer. Parallel V2 work is
  limited to a smokeable `stroke_field_probe_v1` skeleton plus design docs under
  `docs/brushstroke_vnext/`.
- 2026-04-11 VNext bootstrap status:
  - `T1_texture_statistics_head_adapt_v1` smoke passed from a D6 weight-only warm start
  - fixed-baseline smoke eval also passed, and canonical baseline hash verification matched the known conservative base
  - `S1_stroke_field_probe_v1` smoke passed after a pseudo-renderer shape fix
  - `T1_texture_statistics_head_adapt_v1` full 300-step run and fixed-baseline eval are now complete; the run is
    structure-safe and directionally positive, but still modest by eye, so `T2_texture_statistics_head_texture_v1`
    is the next mainline experiment
  - `T2_texture_statistics_head_texture_v1` full 1500-step run and fixed-baseline eval are now complete; the run is
    stronger than T1 and remains portrait/baroque-safe under spot review, but still falls short of a clear new
    brushstroke-strength regime, so `T3` is no longer the default next step
  - `S2_stroke_field_adapt_v1` full 300-step run and fixed-baseline eval are now complete; the stroke-field branch trains
    cleanly and remains structure-safe, but its current pseudo-renderer does not yet open a clearly stronger visual
    regime than T2
  - `S3_stroke_field_texture_push_v1` full 1200-step run and fixed-baseline eval are now complete after a renderer
    upgrade; it is the strongest V2 result so far and remains portrait/baroque-safe, but it still reads more like
    organized directional texture than explicit stroke objects
  - `S4_stroke_field_supervised_push_v1` full 1200-step run and fixed-baseline eval are now complete after adding
    stroke-field proxy supervision; it improves on S3 numerically and visually while remaining portrait/baroque-safe,
    but still does not fully cross into an unmistakable stroke-object regime
  - `S5_stroke_field_objectness_push_v1` full 1200-step run and fixed-baseline eval are now complete after adding
    objectness supervision and sparse top-k prototype routing; it is the strongest V2 result so far, but the branch can
    now over-activate sparse low-content regions, so the next V2 step should improve stroke-object composition rather
    than simply push harder
  - `S6_stroke_field_sparse_blank_suppression_v1` full 1200-step run and fixed-baseline eval are now complete after
    adding explicit blank-region suppression, single-prototype routing, and longer renderer support; it is stronger than
    S5 and remains portrait/baroque-safe, but it still does not resolve sparse-region hallucination in `ink_wash`
  - `S7_stroke_field_evidence_occupancy_v1` full 1200-step run and authoritative fixed-baseline eval from
    `checkpoint-1125` are now complete after adding evidence-backed occupancy gating and `stroke_support_ceiling_loss`;
    it is numerically the strongest V2 run so far and remains portrait/baroque-safe, but it still fails the key
    `ink_wash` sparse-region test because the upper blank region can still grow a new tree form
  - S7 also surfaced a new operational guardrail: a stale `final_checkpoint` alias can survive after a long run even
    when later numbered checkpoints are valid. Future V2 eval/continuation should treat numbered checkpoints as
    authoritative unless `final_checkpoint` is explicitly audited by timestamp/hash
  - `S8_stroke_field_hard_evidence_occupancy_v1` full run and authoritative fixed-baseline eval from `checkpoint-1125`
    are now complete; it is a formal `Partial` result rather than a pass because the old-family branch remains
    portrait/baroque-safe and numerically strong, but still does not eliminate the `ink_wash` sparse-region hallucinated
    tree form
  - `S8` also clarified a second operational nuance: its `final_checkpoint/art_state.pt` is current and hash-matches
    `checkpoint-1125`, but `final_checkpoint/checkpoint_summary.json` still stores `step: null`, so numbered checkpoints
    remain the authoritative continuation/evaluation source even when the final alias is not stale
  - `P1_support_locked_primitive_renderer_v1` full 1125-step run and authoritative fixed-baseline eval from
    `checkpoint-1125` are now complete after introducing a sparse support-locked primitive renderer; the run trains
    cleanly and reaches the strongest V2 movement/detail metrics so far, but it still fails the `ink_wash` sparse-region
    discipline test and also introduces visible `classical_portrait` head/face drift on the canonical compare card
  - `CS1_connected_support_primitive_renderer_v1` full 1125-step run and authoritative fixed-baseline eval from
    `checkpoint-1125` are now complete after adding connected-component-aware support gating to the primitive family;
    the run trains cleanly and improves global movement metrics over P1, but programmatic crop proxies show that
    `ink_upper_sparse` and portrait-sensitive crop movement do not improve over P1
  - current implication: the old `S5 -> S8` occupancy-gated family is closed, and the `P1 -> CS1` dense-occupancy
    primitive family is also not the right place for another full continuation. The next useful branch should keep
    primitive-object rendering but replace dense occupancy with a slot-based anchor-set renderer rather than reopen
    another same-family gate/loss or exclusion sweep
  - `A1_slot_based_anchor_set_renderer_v1` implementation is now integrated and smoke-validated:
    `py_compile`, 1-step train launch, backward pass, checkpoint save, fixed-baseline smoke eval, and zoom-crop export all pass
  - the A1 bootstrap also fixed two concrete issues before the branch was promoted:
    dynamic full-resolution coordinate generation for subject exclusion and removal of an in-place `torch.maximum`
    autograd conflict inside the slot renderer
  - the first A1 long run was then stopped early by design after `checkpoint-225`, because validation loss, prototype loss,
    and basis-usage entropy all showed early flattening rather than continued separation
  - A1's authoritative `checkpoint-225` eval does improve the key `ink_upper_sparse` crop proxy versus both `P1` and `CS1`,
    which makes the slot family more promising than the connected-support dense-occupancy family on the exact failure mode we care about
  - current implication: do not resume A1-v1 unchanged to `1125`; instead open a refined slot branch with stronger discrete
    ownership / prototype pressure
  - `A2_slot_based_anchor_set_renderer_v2` was then launched as a sharpened slot refinement from `A1 checkpoint-225`,
    but it was intentionally stopped at `global_step=30` because `stroke_prototype_loss` and `basis_usage_entropy_loss`
    stayed pinned at the same maximum-entropy regime from the first step onward
  - current implication: the slot family still looks directionally better than dense occupancy on the target sparse-region
    proxy, but the next useful change is no longer another threshold / weight sweep. It is a more explicit slot-ownership
    design

### Phase 5 evaluation

- same prompt and same seed alignment for base vs tuned
- module eval mode required for trustworthy base vs tuned comparisons
- prompt-adherence proxy, style-consistency proxy, and sharpness proxy
- blind-review packet export
- canonical baseline hash verification is now part of the evaluation contract
- baseline audits must distinguish two failure classes:
  1. baseline-weight drift: the baseline branch accidentally loads experiment weights
  2. prompt-template drift: the baseline remains official JanusFlow, but the rendered prompt changed
- current `D*` / `T*` runs pass the first audit class; any comparison against `E3*` compare cards must normalize the
  second before drawing conclusions
- execution audit checklist for all VNext runs lives at
  `docs/brushstroke_vnext/execution_audit_checklist.md`
- current VNext status:
  - `T1` validated the statistics-constrained representation as a safe foothold
  - `T2` scaled that representation meaningfully under `train_texture_balanced_v2`
  - `S2` validated that V2 can train end to end, but the current pseudo-renderer still behaves more like a cautious
    texture head than an explicit stroke generator
  - `S3` showed that a stronger directional renderer can move the branch beyond `S2/T2` while staying safe, but the
    gain is still more "directional texture organization" than a genuinely new brushstroke regime
  - `S4` showed that explicit stroke-field proxy supervision is directionally useful, which justified the objectness
    upgrade
  - `S5` confirms that the V2 ceiling is now more architectural than configurational: sparse objectness gating improves
    strength, but a semi-implicit renderer still composes edits more like localized texture patches than explicit
    coherent brush objects
  - `S6` confirms that a soft blank-region penalty alone is not enough to fix that failure mode; the branch gets
    stronger again, but sparse-region activation still needs a more explicit occupancy / primitive rule
  - `S8` confirms that even a harder support-linked occupancy rule is not enough to fully suppress unsupported sparse
    marks in `ink_wash`; that entire occupancy-gated family should now be treated as mapped rather than open-ended
  - `P1` confirms that sparse primitive rendering is trainable and can push detail metrics farther than the old family,
    but it also shows that sparse anchors alone do not solve support topology or subject safety. The next renderer needs
    connected-support discipline and stronger portrait/head exclusion rather than another generic pressure increase
  - `CS1` confirms that connected-support discipline alone is still not enough when the renderer predicts occupancy
    densely. Because the targeted `ink_upper_sparse` crop proxy gets worse rather than better versus P1, the next
    renderer should predict a small slot-based anchor set with explicit validity and support membership instead of
    another per-pixel occupancy field
  - `A1` confirms that this slot-based move is directionally right, but the first configuration saturates too early into
    near-uniform routing. The next useful experiment should therefore be a refined slot branch rather than a blind
    continuation of A1-v1
  - `A2` closes the simplest slot-refinement hypothesis: fewer slots, fewer prototypes, stricter gating, stronger
    exclusion, and higher routing-pressure losses still collapse immediately into near-uniform routing
  - therefore the next slot-family branch should change the ownership mechanism itself rather than only pushing the same
    scalar knobs harder

### Phase 6 experts

- design document only in `docs/style_experts_design.md`

## Minimal Commands

```bash
python download_janusflow_art_weights.py --target all --output-root /root/autodl-tmp/model_cache/janusflow_art

python train_janusflow_art.py --config configs/janusflow_art/baseline.yaml
python sample_janusflow_art.py --config configs/janusflow_art/baseline.yaml --checkpoint /path/to/final_checkpoint
python eval_janusflow_art.py --config configs/janusflow_art/baseline.yaml --checkpoint /path/to/final_checkpoint

# Optional local-weight overrides when Hub access is unstable:
python train_janusflow_art.py \
  --config configs/janusflow_art/baseline.yaml \
  --model-path /path/to/JanusFlow-1.3B \
  --vae-path /path/to/sdxl-vae

python sample_janusflow_art.py \
  --config configs/janusflow_art/baseline.yaml \
  --checkpoint /path/to/final_checkpoint \
  --model-path /path/to/JanusFlow-1.3B \
  --vae-path /path/to/sdxl-vae

# Quality-conscious larger-LoRA visibility probe:
python train_janusflow_art.py --config configs/janusflow_art/lora_quality_probe.yaml
python sample_janusflow_art.py \
  --config configs/janusflow_art/lora_quality_probe.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/out_janusflow_art_lora_quality_probe/final_checkpoint \
  --lora-scale 0.25

python eval_janusflow_art.py \
  --config configs/janusflow_art/lora_quality_probe.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/out_janusflow_art_lora_quality_probe/final_checkpoint \
  --output-dir /root/autodl-tmp/emoart_gen_runs/out_janusflow_art_lora_quality_probe_scale025_eval \
  --lora-scale 0.25

# Decoder-local brushstroke experiments:
python train_janusflow_art.py --config configs/janusflow_art/brushstroke/brush_adapter_v1.yaml --max-steps 1 --skip-final-eval
python train_janusflow_art.py --config configs/janusflow_art/brushstroke/brush_adapter_v1.yaml
python eval_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_adapter_v1.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E1_brush_adapter_v1_cfgcond/final_checkpoint

python train_janusflow_art.py --config configs/janusflow_art/brushstroke/brush_multiscale_ref_v1.yaml
python eval_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_multiscale_ref_v1.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3_brush_multiscale_ref_v1_cfgcond_refeval/final_checkpoint

# Full-data rebuild and balanced data-stage smoke tests:
python build_emoart_full_rebuild_v1.py
python train_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_spatial_hf_head_full_balanced_v1.yaml \
  --max-steps 1 \
  --skip-final-eval
python train_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_spatial_hf_head_texture_balanced_v2.yaml \
  --max-steps 1 \
  --skip-final-eval

# Brushstroke VNext smokes:
python train_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/vnext/texture_statistics_head_adapt_v1.yaml \
  --init-from-checkpoint /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D6_brush_spatial_hf_head_portrait_structure_recovery_after_gate_relax_v1/final_checkpoint \
  --max-steps 1 \
  --skip-final-eval

python train_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/vnext/stroke_field_probe_v1.yaml \
  --max-steps 1 \
  --skip-final-eval
```

## Acceptance Checks

- JanusFlow imports resolve
- YAML config loads
- trainer builds model graph and checkpoint structure
- sampler exports images, grids, and prompt logs
- evaluator exports summary, report, and blind-review bundle
