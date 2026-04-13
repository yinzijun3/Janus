# JanusFlow Brushstroke Handoff

Date: 2026-04-12

## Current State

This branch is testing a decoder-local brushstroke enhancement route for JanusFlow-Art. The main goal is visible brushstroke/material/texture improvement while preserving prompt semantics and composition.

Important: do not judge success from automatic metrics alone. The user explicitly confirmed that current experiment attempts remain near-baseline by eye and do not show obvious brushstroke differences. Manual compare-card review is the primary gate.

There is no active training or evaluation process at handoff time. E3D/E4/E5/E7/E9/E10/E11/E12/E14/E15/E16 have completed their 450-step screens and fixed-prompt evaluations, and have been manually reviewed. E8 has completed its LoRA visible-effect probe plus a 4-scale evaluation sweep. The rebuilt-data curriculum line has now also progressed through D1, D2, D4, D5, and D6, all with completed training, fixed-baseline evaluation, and manual compare-card review. E6 is staged but intentionally paused because it is expected to behave like another reference-conditioning variant. E13 remains prepared but lower-priority because E12 and E14 already show that stronger or later local editing alone does not yet solve the baroque failure mode.

The full-data rebuild is now also implemented and materialized at `/root/autodl-tmp/data/emoart_5k/gen_full_official_rebuild_v1`. That rebuild is the new default data path for the next brushstroke experiments unless a run explicitly needs the legacy texture-rich subset.

VNext implementation is now in progress. `T1`, `T2`, `S2`, `S3`, `S4`, `S5`, `S6`, `S7`, `S8`, `P1`, and `CS1` have all completed full training,
authoritative fixed-baseline evaluation, and compare-card export. `A1_slot_based_anchor_set_renderer_v1` has now also passed
implementation audit, 1-step smoke training, fixed-baseline smoke evaluation, and a diagnostic partial full run with
authoritative evaluation from `checkpoint-225`. `A2_slot_based_anchor_set_renderer_v2` is now scaffolded, has passed
its 1-step training smoke from `A1 checkpoint-225`, and has already produced a short early-stop diagnostic run showing
immediate uniform-routing saturation before the first numbered checkpoint. The new mainline began with
`TextureStatisticsBrushHead`, and the current parallel V2 line has now progressed from the earlier `StrokeFieldBrushHead` family into
sparse primitive rendering, then a connected-support primitive variant, and now a slot-based anchor-set renderer. New configs live under
`configs/janusflow_art/brushstroke/vnext/`, and the execution audit checklist lives at
`docs/brushstroke_vnext/execution_audit_checklist.md`.

## Documentation Rule

From this point onward, every experiment entry in this handoff should be written in a fixed paper-style format:

1. `Method`: what was changed relative to the previous accepted baseline or diagnostic run, why that change was introduced, and what part of the model was expected to move.
2. `Results`: training status, checkpoint/eval paths, key metrics, and whether the run was a smoke, 450-step screen, or full evaluation.
3. `Analysis`: human visual judgment first, then interpretation of what the run says about the architecture.

This document is the canonical experiment ledger. Future runs should not be summarized as one-line notes only; the goal is that an engineer can resume work or write the method/results section of a paper directly from this file.

## High-Confidence Conclusions

1. Adapter-only is too conservative.
   - E1 and E2 completed 450-step screening runs.
   - Manual review found near-identical images with only minor edge/color changes.
   - E3A no-reference ablation confirmed this: same reference prompt/seed set, but average visible effect remained tiny.

2. The reference branch is active but too global.
   - E3 and E3B produced clearly larger changes than E1/E2/E3A.
   - Manual review showed pose/layout/object-placement shifts, not isolated brushstroke transfer.
   - This is not acceptable quality improvement.

3. HF loss is not the main cause of drift.
   - E3B disabled Laplacian/Sobel losses but kept the reference branch.
   - It still showed similar reference-driven composition changes.

4. The current unresolved problem is scope control.
   - No-reference path: too weak.
   - Full reference path: too global.
   - E3D reduced reference scale and used fine tokens only, but manual review still showed composition drift on expressionist/portrait/ink/ukiyo-e.
   - E4 added high-pass reference conditioning; drift is reduced but still present.
   - E5 removed reference conditioning and lightly unfreezed `vision_gen_dec_aligner`; it preserved composition but remained too subtle.
   - Next work should either localize reference conditioning even more (spatial gate / explicit high-pass residual routing) or increase adapter-side texture effects without letting reference tokens rewrite structure.

5. Current visible-effect verdict: all brushstroke attempts are below the user-visible bar.
   - E1/E2/E3A/E5 are composition-safe but effectively near-baseline in brushstroke/material detail.
   - E3/E3B/E3D/E4 create larger numeric image differences, but the visible changes are mixed with pose/layout/object drift instead of clean brushstroke transfer.
   - Therefore the issue is not just that automatic metrics are weak; the current trained changes do not reliably land in the perceptual brushstroke channel.

6. The likely cause is a combination of conservative config and insufficient leverage over the generation path.
   - The decoder-local adapter path uses small residual gates and keeps the base generation decoder frozen, so it may be too weak to change visible material statistics.
   - E5 only unfreezes `vision_gen_dec_aligner` at `1e-5`; this is intentionally safe and likely too small to move visible texture in a 450-step screen.
   - Flow MSE alone mainly rewards reconstructing the target latent trajectory, not specifically brushstroke salience. Laplacian/Sobel auxiliary losses slightly move sharpness metrics but did not create convincing painterly strokes.
   - Reference-token cross-attention is not pure texture transfer. Even with fine tokens/high-pass input, it can still carry layout/semantic cues and perturb composition.
   - Training/evaluation resolution and full-card review can hide subtle texture gains, but the repeated near-baseline manual result means this is not only an evaluation presentation problem.

7. E8 shows that visible brushstroke/material change is achievable on this data/model, and that `0.75` provides a useful target-strength reference even though it is not yet a safe deployment setting.
   - At `lora_scale=0.15`, LoRA is visible but still relatively conservative.
   - At `lora_scale=0.25`, LoRA is a balanced tradeoff and is useful as a composition-safe reference point.
   - At `lora_scale=0.75`, the brushstroke gain is the clearest by eye and better matches the desired "obvious enhancement" regime, but it introduces head blur and semantic drift.
   - Therefore the project does not need more generic style power in the abstract; it needs a local architecture that can approach the perceptual strength of E8 `0.75` while preserving the structural safety that E10 approximates.

8. E9 validates the v2 direction, but not yet the target effect size.
   - The new token-plus-feature-map local path is safer than reference conditioning and slightly richer than E7.
   - Several portrait / impressionist / ukiyo-e / ink cards keep composition close to baseline while gaining some local material density.
   - However, baroque cards still show repeated face/head distortion, and many cards still read more like mild repainting than decisive new brushstroke structure.
   - E9 is therefore an incremental win, not yet a solution.

9. E10 further narrows the failure source: feature-map-only is safer, but weaker.
   - Removing the token adapter reduces the face/head drift seen in E9, especially on baroque and portrait cards.
   - This supports the idea that token-stage local edits still leak semantic structure even without explicit reference conditioning.
   - However, the first feature-map-only setting drops visible effect size too far; many cards move back toward near-baseline.
   - The next sensible step is not to re-enable the token adapter, but to push the feature-map-only branch harder.

10. E11 shows the remaining issue is locality control inside the feature-map path itself.
   - A stronger feature-map-only run increases pair difference and visible paint/material handling substantially over E10.
   - Impressionist / expressionist / ink benefit, so the branch is not fundamentally too weak.
   - But baroque face/head distortion returns, which means raw strength without stronger locality constraints still leaks into structure.
   - The next step should therefore be a new image-local architecture with explicit spatial/frequency gating or a later output residual head.

11. E12 partially validates the new gated-HF architecture family, but not yet the full target.
   - `SpatialHFBrushHead` is more controlled than the raw feature-map residual used in E11.
   - E12 raises texture energy over E10 and keeps most prompts safer than E11.
   - However, the average effect size is still well below the E8 `0.75` target-strength reference, and `baroque` remains structurally unsafe.
   - This suggests the gate/high-pass design is directionally correct but still not sufficient as the only safeguard.

12. E14 weakens the case for "later injection alone" as the main fix.
   - Moving the local edit from decoder feature maps to the final velocity output does not remove the baroque head/face failure mode.
   - E14 keeps prompt/style CLIP stable and increases high-frequency metrics, but visually it still behaves like a modest repaint on stable prompts and a destructive reshape on baroque.
   - This means the core problem is not only where the edit is inserted. The remaining issue is missing structural protection or too-loose brush supervision when the model does decide to move.

13. E15 shows that a self-derived structure anchor helps explainability more than it helps the hardest failure case.
   - The structure-protection mask remains active through late training, and `protected_gate_l1_loss` does not collapse.
   - The run slightly increases pair difference over E12/E14 while keeping most non-portrait prompts stable.
   - However, `baroque_drama` still deforms visibly, so low-frequency self-protection from the base velocity is not sufficient on its own.
   - The next architecture should add a stronger spatial prior, not just another scalar-strength change.

14. E16 closes the current anchored-output sub-family.
   - Adding an upper-center subject prior on top of E15 does not materially change the result.
   - `baroque_drama` remains visibly broken, while other prompts remain in the same conservative regime.
   - This means the current output-head-only family has likely exhausted its easy gains.
   - The next step should move to a different mask source or a different local brush mechanism, not another minor variant of the same output head.

15. The next architecture family is implemented and partially screened.
   - `SpatialHFBrushHead` adds a spatial gate plus high-pass-only residual path on decoder feature maps.
   - `OutputVelocityTextureHead` provides a later output-space fallback when feature-map editing still disturbs structure.
   - Training now supports `low_frequency_consistency` against the no-brush base path, `gate_l1` / `gate_tv` regularization, and `protected_gate_l1` for anchored heads.
   - E12, E14, E15, and E16 have now all completed empirical validation, so the next work item should be a more explicit subject-mask source or a different local brush architecture rather than another same-family strength tweak by default.

16. Data distribution was a real part of the failure pattern, not just architecture weakness.
   - The old `train_texture_rich_v1` subset was a second-stage heuristic slice, not a balanced master training set.
   - It over-compressed the task toward visible texture and under-expressed the portrait/head-sensitive recovery cases that keep failing in `baroque`, `high renaissance`, and related prompts.
   - The official dataset also contains one naturally short style (`Gongbi`, 32 images), so future balanced continuation manifests cannot assume every style has 100 unique training items.
   - The new rebuild now separates three jobs cleanly: `train_full_balanced_v1` for semantics and structure, `train_texture_balanced_v2` for visible texture pressure, and `train_portrait_structure_v1` for portrait/head recovery.

17. Prompt routing is now part of the data design, not only the model config.
   - The rebuild stores three explicit prompt variants per sample: `prompt_v2_content_safe`, `prompt_v2_texture_push`, and `prompt_v2_portrait_safe`.
   - JanusFlow-Art data loading now honors `rendered_prompt` exactly when a manifest provides it, instead of rewrapping the sample through a generic prompt template.
   - This is important because the new continuation manifests intentionally route head-sensitive samples to a more conservative prompt than non-head-sensitive texture samples.

18. A silent path-normalization bug was found and fixed while validating the rebuilt manifests.
   - The old image-path resolver normalized arbitrary strings with `normalize_text`, which collapses non-breaking spaces.
   - Several official filenames contain non-breaking spaces, especially under classical/renaissance styles.
   - The first smoke logs therefore reported false missing-image counts even though the rebuilt manifests were correct.
   - `resolve_optional_image_path` now preserves the raw path string instead of compacting internal whitespace.

19. The rebuilt-data spatial-HF line is now meaningfully mapped, and D6 is the best point inside that family so far.
   - D4 showed that the D1 -> D2 curriculum can absorb moderately stronger local texture pressure without immediately reintroducing portrait/baroque collapse.
   - D5 showed that simply relaxing gate regularization and pushing the same 512-channel spatial-HF head harder does not buy a new visible-effect regime; returns are already flattening.
   - D6 showed that portrait-structure recovery can slightly improve the D5 tradeoff rather than merely suppressing it, which validates the three-stage data curriculum as useful rather than decorative.
   - However, D6 still does not approach the target visible strength implied by E8 `0.75`, and `expressionist` / `ukiyoe` remain conservative.
   - Therefore the current bottleneck is no longer "whether the rebuilt data curriculum helps." It does. The remaining bottleneck is that the current 512-channel spatial-HF head is nearing saturation.

20. VNext should treat brushstroke as an explicitly represented object.
   - The next mainline no longer asks the model to discover painterly behavior inside a generic free residual.
   - `TextureStatisticsBrushHead` predicts orientation, scale, density, pigment, and multi-band energy maps and renders a constrained texture residual from those statistics.
   - This route is supervised with proxy targets derived from the target image, so the statistics branch cannot silently collapse into another unconstrained latent.
   - A parallel V2 stroke-field probe is implemented only as a smokeable skeleton for now; it is the fallback if the statistics route still cannot open a stronger visual regime.

21. S6 shows that the current blank-region suppression is directionally reasonable but still too weak and too indirect.
   - S6 keeps the V2 objectness route and adds an explicit blank-region penalty derived from image-local support proxies, while also making routing more singular (`topk_prototypes=1`) and the stroke primitive longer (`renderer_kernel_size=13`).
   - Numerically, S6 is stronger than S5: pair difference, Laplacian variance, and high-frequency energy all increase again, while prompt CLIP slightly improves instead of collapsing.
   - However, the key failure mode remains visible on `ink_wash_mountain`: sparse upper-page regions still acquire invented tree/branch structure rather than only refined existing marks.
   - This means the branch is no longer mainly blocked by conservative optimization, and it is not enough to say "just add a blank penalty." The renderer still composes edits like semi-implicit elongated texture objects, so sparse-region activation rules need to be more explicit and more tightly tied to existing structure.

22. S8 closes the old occupancy-gated stroke-field family without solving its main failure mode.
   - S8 upgrades S7 with a harder evidence-backed occupancy rule, but the authoritative evaluation from `checkpoint-1125`
     still fails the key `ink_wash_mountain` test.
   - Portrait and baroque safety remain acceptable, pair difference stays above S7, and canonical baseline verification still
     matches the official JanusFlow base hash.
   - However, the sparse upper region in `ink_wash_mountain` still grows a large new tree-like structure, so S8 is only a
     `Partial` result rather than a pass.
   - Therefore `S8` is the final formal validation point for that family, and further `S9/S10`-style gate/loss sweeps should
     not be opened.

23. P1 validates that sparse primitive rendering is trainable, but it does not yet solve the project bottleneck.
   - `P1_support_locked_primitive_renderer_v1` replaces the semi-implicit blob compositor with a support-locked sparse anchor
     renderer that rasterizes one elongated primitive per selected anchor.
   - The branch trains cleanly, survives a full 1125-step run, and reaches the strongest movement/detail metrics so far in the
     V2 line.
   - But the authoritative evaluation still shows the canonical `ink_wash` hallucinated upper-tree failure, and
     `classical_portrait` develops visible head/face drift on the canonical seed-42 compare card.
   - This means the project has moved past "can sparse primitive training work at all?" and into a narrower problem:
     primitives now need stronger support topology and stronger subject exclusion, not just stronger objectness.

24. The next useful architecture should no longer be another soft occupancy variant.
   - The old `S5 -> S8` family already tested objectness sparsity, blank suppression, evidence gating, and harder occupancy.
   - P1 tested a new renderer family and showed that sparse anchors alone do not prevent unsupported marks or portrait drift.
   - The next architecture should therefore combine connected-support constraints with a stronger subject-safe exclusion rule,
     for example by rasterizing only on connected support components and explicitly suppressing anchors inside dominant
     portrait/head regions.

25. CS1 closes the connected-support-per-pixel primitive branch as a no-go for direct continuation.
   - `CS1_connected_support_primitive_renderer_v1` trained cleanly from `P1 checkpoint-1125`, preserved the canonical
     baseline hash, and produced a full authoritative eval bundle from `checkpoint-1125`.
   - However, the targeted crop proxy moved in the wrong direction for the branch objective:
     - `ink_upper_sparse` mean crop diff increased from `9.5697` (P1) to `9.7279` (CS1)
     - `classical_face_window` mean crop diff increased from `7.8791` (P1) to `7.9939` (CS1)
   - So connected-support gating alone did not reduce sparse-region activation and did not improve portrait stability.
   - Because CS1 fails the only gate that justified opening `CS2`, the next useful branch should not be `CS2`. It should
     be a slot-based anchor-set renderer that stops predicting dense occupancy entirely and instead predicts a small set of
     valid anchors with explicit support membership and subject-safe exclusion.

26. A1 validates that the slot-based anchor-set branch is operationally ready for a full run.
   - `A1_slot_based_anchor_set_renderer_v1` now has a dedicated slot-based renderer and brush head integrated into the
     existing JanusFlow-Art runtime, checkpoint, loss, and eval stack.
   - The first smoke attempt exposed a real subject-exclusion grid-shape bug plus an autograd conflict from in-place
     `torch.maximum` writes inside the slot renderer. Both were fixed before promotion.
   - The corrected branch now passes `py_compile`, 1-step smoke training, checkpoint save, fixed-baseline smoke eval, and
     zoom-crop export while preserving the canonical official-base hash.
   - Therefore the next smallest useful step is no longer more implementation. It is the first authoritative full run of A1.

27. A1 is more promising than CS1 on the target sparse-region proxy, but the first version saturates too early to justify an unchanged long continuation.
   - The first partial A1 long run was intentionally stopped after the branch showed early saturation rather than continued separation.
   - By `checkpoint-225`, `basis_usage_entropy_loss` had flattened near `ln(16)`, `stroke_prototype_loss` had also flattened, and validation loss had stopped improving cleanly.
   - Even so, the authoritative `checkpoint-225` eval improved the key `ink_upper_sparse` crop proxy relative to both P1 and CS1, while also slightly improving the `classical_face_window` proxy relative to CS1.
   - The right conclusion is therefore not "A1 failed." It is that removing dense occupancy was useful, but the current slot formulation still routes too uniformly and needs stronger discrete ownership / prototype pressure before another full long run.

28. A2 shows that simply tightening slot thresholds and routing penalties is not enough to escape the same saturation regime.
   - `A2_slot_based_anchor_set_renderer_v2` reduced `slot_count` / `prototype_count`, raised `valid_anchor_threshold`, strengthened exclusion, and increased slot/prototype routing-pressure losses.
   - The branch still launched cleanly from `A1 checkpoint-225`, so this is not an implementation failure.
   - But within the first `30` steps, `basis_usage_entropy_loss` stayed pinned near `ln(12)` and `stroke_prototype_loss` stayed pinned near the same flat value at every step, while slot sparsity barely moved.
   - Therefore the next useful change should not be another same-family weight sweep. The slot family now likely needs a more explicit ownership mechanism or a more discrete anchor program, not just stronger scalar pressure.

## Key Fixes Already Made

- Baseline evaluation now disables LoRA/style/brush modules instead of accidentally using active experiment modules.
- Brush conditioning now defaults to conditioned-CFG rows only: `brush.condition_unconditioned: false`.
- Reference-backed eval prompts were added at `configs/janusflow_art/prompts_brush_reference.jsonl`.
- E3/E3A/E3B reference evaluations now report `reference_path_count: 24`.
- Brush checkpoint restore no longer misclassifies non-LoRA `language_model/` folders as PEFT adapters unless `adapter_config.json` exists.
- Experiment output dirs are now explicitly labeled `E1_`, `E2_`, `E3_`, `E3A_`, `E3B_`, `E3D_`.
- Checkpoint resume now restores `global_step` from `checkpoint_summary.json` or a `checkpoint-N` directory name before continuing training, so `--resume-from-checkpoint checkpoint-150` stops at the configured 450 planned steps instead of running 450 additional steps.
- Added an optional high-pass residual path for reference images in `finetune/janusflow_art_runtime.py`, gated by `brush.reference_encoder.high_pass.*` in config.

## Important Files

- Main plan: `docs/janusflow_brushstroke_plan.md`
- This handoff: `docs/janusflow_brushstroke_handoff.md`
- Brush modules: `janus/janusflow/models/brush/`
- Runtime wiring: `finetune/janusflow_art_runtime.py`
- Losses: `finetune/janusflow_art_losses.py`
- Train entry: `train_janusflow_art.py`
- Eval entry: `eval_janusflow_art.py`
- Sample entry: `sample_janusflow_art.py`
- Configs: `configs/janusflow_art/brushstroke/`
- VNext configs: `configs/janusflow_art/brushstroke/vnext/`
- VNext docs: `docs/brushstroke_vnext/`
- Baseline prompts: `configs/janusflow_art/prompts_baseline.jsonl`
- Reference prompts: `configs/janusflow_art/prompts_brush_reference.jsonl`

## Completed Experiment Summary

| ID | Config | Output | Manual decision |
| --- | --- | --- | --- |
| E1 | `brush_adapter_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E1_brush_adapter_v1_cfgcond` | Too conservative; do not expand. |
| E2 | `brush_adapter_hf_loss_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E2_brush_adapter_hf_loss_v1_cfgcond` | Still too subtle; HF losses did not create reliable brushstroke gains. |
| E3 | `brush_multiscale_ref_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3_brush_multiscale_ref_v1_cfgcond_refeval` | Reference branch active, but too global; do not expand directly. |
| E3A | `brush_ablation_no_ref.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3A_brush_ablation_no_ref_cfgcond_refeval` | Confirms no-reference path is visually weak. |
| E3B | `brush_ablation_no_hf_loss.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3B_brush_ablation_no_hf_loss_cfgcond_refeval` | Drift persists without HF loss; reference scope is the main issue. |
| E3D | `brush_multiscale_ref_local_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3D_brush_ref_fine_local_v1_cfgcond_refeval` | Still shows composition drift on multiple styles. |
| E4 | `brush_highpass_ref_fine_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E4_brush_highpass_ref_fine_v1_cfgcond_refeval` | Drift reduced but still present; not acceptable. |
| E5 | `brush_adapter_dec_aligner_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E5_brush_adapter_dec_aligner_v1_cfgcond_refeval` | Composition preserved, but too subtle. |
| E7 | `brush_adapter_dec_aligner_aggressive_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E7_brush_adapter_dec_aligner_aggressive_v1_cfgcond_refeval` | Stronger than E5 without drift, but still below the target visual brushstroke bar. |
| E8 | `brush_lora_visible_probe_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E8_brush_lora_visible_probe_v1` | `0.75` gives the desired visible strength but drifts; use it as a target-strength reference, not as a safe setting. |
| E9 | `brush_local_capacity_v2_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E9_brush_local_capacity_v2_v1_cfgcond_refeval` | Best local architecture so far before E12, but still weaker than high-scale E8 and not fully structure-safe on baroque/portrait. |
| E10 | `brush_feature_map_only_aggressive_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E10_brush_feature_map_only_aggressive_v1_cfgcond_refeval` | Safer than E9, but too weak; keep feature-map-only and push harder. |
| E11 | `brush_feature_map_only_stronger_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E11_brush_feature_map_only_stronger_v1_cfgcond_refeval` | Stronger than E10, but baroque structural drift returns; stop scalar tuning of this module family. |
| E12 | `brush_spatial_hf_head_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E12_brush_spatial_hf_head_v1_cfgcond_refeval` | Better controlled than E11 and slightly stronger than E10, but still below target strength and still unsafe on baroque. |
| E13 | `brush_spatial_hf_head_stronger_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E13_brush_spatial_hf_head_stronger_v1_cfgcond_refeval` | Prepared, but now lower priority because E12 is already unsafe on baroque and a stronger version likely worsens that tradeoff. |
| E14 | `brush_output_residual_head_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E14_brush_output_residual_head_v1_cfgcond_refeval` | Later output-space edit still fails on baroque and does not beat E12 on the safety-strength tradeoff. |
| E15 | `brush_output_structure_anchor_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E15_brush_output_structure_anchor_v1_cfgcond_refeval` | Structure-anchored output head is active, but self-derived protection is still insufficient on baroque. |
| E16 | `brush_output_structure_anchor_center_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E16_brush_output_structure_anchor_center_v1_cfgcond_refeval` | Upper-center protection prior does not materially improve E15; output-head-only masking remains insufficient on baroque. |
| D1 | `brush_spatial_hf_head_full_balanced_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D1_brush_spatial_hf_head_full_balanced_v1` | Safe rebuilt-data stage-1 base; useful initialization, but too conservative alone. |
| D2 | `brush_spatial_hf_head_texture_balanced_v2.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D2_brush_spatial_hf_head_texture_balanced_v2` | Best early rebuilt-data continuation; modest texture gain without obvious portrait collapse. |
| D4 | `brush_spatial_hf_head_texture_balanced_push_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D4_brush_spatial_hf_head_texture_balanced_push_v1` | Safe half-step above D2, but still not a new brushstroke-strength regime. |
| D5 | `brush_spatial_hf_head_texture_balanced_gate_relax_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D5_brush_spatial_hf_head_texture_balanced_gate_relax_v1` | Near diminishing returns for the current 512-channel spatial-HF family. |
| D6 | `brush_spatial_hf_head_portrait_structure_recovery_after_gate_relax_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D6_brush_spatial_hf_head_portrait_structure_recovery_after_gate_relax_v1` | Best current rebuilt-data spatial-HF checkpoint, but still below target visible strength. |
| A1 | `slot_based_anchor_set_renderer_v1.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/A1_slot_based_anchor_set_renderer_v1` | Promising slot-based diagnostic branch; improves the target `ink_upper_sparse` proxy vs P1/CS1, but saturates early and should not be resumed unchanged. |
| A2 | `slot_based_anchor_set_renderer_v2.yaml` | `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/A2_slot_based_anchor_set_renderer_v2` | Sharpened slot refinement still saturates immediately; stop scalar slot-pressure sweeps and move to a more explicit slot-ownership design. |

Run-level notes are in each output dir as `review_notes.md`.

## VNext Bootstrap: T1 / S1

### T1 Texture-Statistics Head Adapt Smoke + Fixed-Baseline Eval

#### Method

- `T1_texture_statistics_head_adapt_v1` is the first VNext experiment and the first direct implementation of the
  "brushstroke as explicit statistics" idea.
- Relative to D6, it removes the old `SpatialHFBrushHead` from the active path and enables the new
  `TextureStatisticsBrushHead`.
- The new head predicts explicit `orientation`, `scale`, `density`, `pigment`, and `band` fields from the decoder
  feature map, then renders a constrained local texture residual through a fixed directional/band-pass basis renderer.
- Unlike the earlier free-residual heads, this branch is supervised with image-derived proxy targets from
  `finetune/brush_proxy_targets.py`, so the new latent factors are tied to measurable local structure in the target
  image instead of being left implicit.
- This run also serves as an infrastructure audit for two new safety features:
  1. `training.init_from_checkpoint`, which warm-starts compatible model weights from D6 without restoring optimizer,
     scheduler, or global step
  2. canonical baseline hash verification inside the evaluator

#### Results

- Config: `configs/janusflow_art/brushstroke/vnext/texture_statistics_head_adapt_v1.yaml`
- Smoke command: passed with `--max-steps 1 --skip-final-eval`
- Init source: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D6_brush_spatial_hf_head_portrait_structure_recovery_after_gate_relax_v1/final_checkpoint`
- Smoke output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/T1_texture_statistics_head_adapt_v1`
- Smoke runtime summary: only `brush_texture_statistics_head_enabled=true`; all legacy adapter/reference/output heads disabled
- Smoke loss components were active as intended:
  - `texture_stats_orientation_loss`
  - `texture_stats_band_loss`
  - `texture_stats_density_loss`
  - `texture_stats_pigment_loss`
  - `basis_usage_entropy_loss`
- Fixed-baseline eval on the smoke checkpoint completed at:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/T1_texture_statistics_head_adapt_v1_smoke_eval/evaluation`
- Eval summary:
  - `avg_pair_pixel_abs_diff = 5.2341`
  - `avg_prompt_clip_tuned = 0.3443`
  - canonical baseline hash matched exactly:
    `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`

#### Analysis

- The key result here is not "T1 already wins visually"; a 1-step smoke checkpoint is not supposed to settle that.
- The important result is that the full VNext contract now works end to end:
  new head creation, proxy-supervised loss wiring, D6 warm-start without stale schedule state, checkpoint save/restore,
  baseline-module disabling, and canonical baseline verification.
- This means the next T1 full 300-step run can be trusted as a clean architectural measurement rather than another
  infrastructure shakeout.

### T1 Texture-Statistics Head Adapt Full Run

#### Method

- The first full VNext training run keeps the same architecture as the T1 smoke but extends it to the planned 300 steps on
  `train_full_balanced_v1`.
- Relative to D6, the important methodological change is not a scalar-strength adjustment; it is a representational change.
  Instead of learning a free local residual, the model predicts explicit local brush statistics
  (`orientation`, `scale`, `density`, `pigment`, `band energy`) and must reconstruct its edit through a constrained
  texture-basis renderer.
- The run uses `training.init_from_checkpoint` to initialize compatible backbone weights from D6 without inheriting
  optimizer, scheduler, or global-step state. This matters because T1 is a cross-architecture warm start, not a same-head
  continuation.
- The evaluator is run under the new canonical-baseline guardrail, so the resulting base-vs-tuned comparison can be
  trusted as an actual unchanged JanusFlow baseline.
- 2026-04-11 audit clarification:
  - In the current evaluator, the tuned branch is built with `checkpoint_path=args.checkpoint`, but the baseline branch
    is built with `checkpoint_path=None`. This means the baseline branch does not load `art_state.pt`, does not load
    finetuned aligner weights, and instead samples directly from the official base model at
    `/root/autodl-tmp/model_cache/janusflow_art/JanusFlow-1.3B`.
  - Therefore the fixed-baseline bundles for current `D*` and `T*` runs are using the untouched official JanusFlow
    weights under the configured prompt/seed/CFG/image-size setting.
  - The remaining visual mismatch between current compare-card baselines and earlier `E3*` baselines is primarily a
    prompt-contract mismatch. `E3*` used `data.prompt_template: strong_style`, while rebuilt-data and VNext runs use
    `data.prompt_template: conservative`, even when both point at the same `prompts_brush_reference.jsonl`.
  - For the canonical portrait sample, the `strong_style` template appends an explicit global emphasis clause
    ("prioritize tactile material presence, visible artistic handling, rich local texture..."), while the
    `conservative` template keeps only the descriptive content/brush/texture/composition cues. These are not minor
    prompt variants and should not be treated as cross-era baseline-equivalent.
  - Action item: if a future claim compares `D*` / `T*` baselines directly against `E3*`, the comparison must be
    rerun under the same prompt template and prompt file first. Cross-template baseline comparisons are invalid.

#### Results

- Config: `configs/janusflow_art/brushstroke/vnext/texture_statistics_head_adapt_v1.yaml`
- Train output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/T1_texture_statistics_head_adapt_v1`
- Final checkpoint:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/T1_texture_statistics_head_adapt_v1/final_checkpoint`
- Final train-side validation:
  - `eval_loss @ step 75 = 1.0129`
  - `eval_loss @ step 150 = 1.0223`
  - `eval_loss @ step 225 = 1.0174`
  - `final eval_loss = 1.0112`
- Fixed-baseline evaluation:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/T1_texture_statistics_head_adapt_v1/evaluation`
- Key evaluation metrics:
  - `avg_pair_pixel_abs_diff = 5.3489`
  - `avg_prompt_clip_baseline / tuned = 0.3449 / 0.3457`
  - `avg_laplacian_baseline / tuned = 897.32 / 968.77`
  - `avg_hf_energy_baseline / tuned = 3577.78 / 3645.18`
- Canonical baseline hash matched:
  `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`
- Manual compare-card spot review covered:
  - `classical_portrait_oil_ref_high_renaissance`
  - `baroque_drama_ref_baroque`
  - `impressionist_garden_ref_impressionism`
  - `expressionist_interior_ref_expressionism`
  - `ukiyoe_evening_ref_ukiyoe`
  - `ink_wash_mountain_ref_ink_wash`

#### Analysis

- T1 is the first run in this project where the new local representation itself looks directionally right even when the
  effect size is still limited.
- Portrait and baroque structure remain intact in the inspected cards. This is a real win over the earlier "push harder
  and break heads" pattern. In particular, `baroque_drama` is visually very safe.
- `impressionist_garden`, `ink_wash_mountain`, and `ukiyoe_evening` show the clearest gain, but the gain is still modest.
  The visual impression is not "new obvious brushstroke language"; it is closer to "more organized local texture and
  surface handling."
- `expressionist_interior` moves more strongly than most prompts, but even there the change reads as improved material
  organization rather than a dramatic new paint regime.
- So T1 should be marked as a positive architectural foothold, not a target-reaching result. It validates that the
  statistics-constrained route can stay safe while raising local texture energy, but it has not yet crossed the user’s
  visible-strength threshold. The next useful step is T2, not a redesign or rollback.

### T2 Texture-Statistics Head Texture Continuation

#### Method

- T2 keeps the `TextureStatisticsBrushHead` architecture but moves from the safe full-balanced adaptation stage into the
  stronger `train_texture_balanced_v2` curriculum. The purpose is to test whether the same representation can scale from
  "organized local texture" into a clearly stronger visible brushstroke regime without reintroducing the older
  portrait/baroque breakage.
- Relative to T1, the head is widened from `basis_channels=96` to `basis_channels=128`, the residual scale is increased
  to `1.8`, and the run is initialized from `T1 checkpoint-300` through `training.init_from_checkpoint` rather than a
  strict optimizer/global-step resume. This is important because T2 is a compatible warm start with a widened local
  renderer, not a same-shape checkpoint continuation.
- Before the long run, the execution audit caught an important loader issue: a strict restore would fail on the widened
  basis tensors. The runtime was updated so cross-architecture warm starts can partially restore compatible weights while
  logging mismatched tensors explicitly, and the T2 config was corrected to use `init_from_checkpoint` instead of
  `checkpoint.resume_from`.
- The evaluator again ran under the fixed-baseline contract: baseline branch built from the official JanusFlow base model,
  all art/brush modules disabled, and canonical baseline hash verification enabled.

#### Results

- Config: `configs/janusflow_art/brushstroke/vnext/texture_statistics_head_texture_v1.yaml`
- Init source:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/T1_texture_statistics_head_adapt_v1/checkpoint-300`
- Train output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/T2_texture_statistics_head_texture_v1`
- Final checkpoint:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/T2_texture_statistics_head_texture_v1/final_checkpoint`
- Execution-audit status:
  - config audit: passed
  - warm-start audit: passed after switching to partial compatible init loading
  - smoke/entry audit: passed before the long run
  - fixed-baseline eval audit: passed
- Train-side validation:
  - `eval_loss @ step 525 = 1.0219`
  - `eval_loss @ step 825 = 1.0331`
  - `eval_loss @ step 1050 = 1.0000`
  - `eval_loss @ step 1425 = 1.0220`
  - `eval_loss @ step 1500 = 1.0114`
  - final checkpoint summary `eval_loss = 1.0063`
- Fixed-baseline evaluation:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/T2_texture_statistics_head_texture_v1/evaluation`
- Key evaluation metrics:
  - `avg_pair_pixel_abs_diff = 5.9056`
  - `avg_prompt_clip_baseline / tuned = 0.3449 / 0.3466`
  - `avg_style_clip_baseline / tuned = 0.2808 / 0.2791`
  - `avg_laplacian_baseline / tuned = 897.32 / 1054.31`
  - `avg_hf_energy_baseline / tuned = 3577.78 / 3771.15`
- Canonical baseline hash matched:
  `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`
- Manual compare-card spot review covered the fixed seed-42 set:
  - `classical_portrait_oil_ref_high_renaissance`
  - `baroque_drama_ref_baroque`
  - `impressionist_garden_ref_impressionism`
  - `expressionist_interior_ref_expressionism`
  - `ukiyoe_evening_ref_ukiyoe`
  - `ink_wash_mountain_ref_ink_wash`

#### Analysis

- T2 is a real step up from T1. The statistics-constrained route does scale under the stronger texture curriculum, and it
  does so without falling back into the old "push harder, break the head" failure mode.
- The cleanest visible gains are in `impressionist_garden` and `ink_wash_mountain`: foliage, path edges, mist contours,
  and tree silhouettes are more articulated, and the texture organization reads as more intentional rather than merely
  sharper. `ukiyoe_evening` also improves, though more modestly, through clearer line/print-surface separation.
- `classical_portrait` and `baroque_drama` remain structurally safe by eye in the reviewed seed-42 cards. This matters
  more than the raw sharpness gain because it means the stronger texture curriculum did not reopen the old portrait-safe
  failure mode.
- `expressionist_interior` changes, but not enough to count as a new brushstroke-strength regime. More broadly, T2 still
  reads as "stronger organized texture and material articulation" rather than "the model is now explicitly painting with
  convincing brush logic."
- Numerically, T2 nearly reaches the planned `avg_pair_pixel_abs_diff >= 6.0` gate (`5.9056`) while improving prompt CLIP
  instead of hurting it. That is encouraging, but the visual outcome is still short of the E8 `0.75` reference strength.
- The practical conclusion is that T2 validates V1 as a meaningful upgrade over D6/T1, but it also suggests the current
  bottleneck has moved again: portrait/baroque recovery is no longer the main blocker, so T3 is lower-priority unless a
  later audit reveals hidden structure regression. The next major information gain is likely to come from promoting the
  V2 stroke-field line rather than from another recovery-style continuation on top of an already-safe T2.

### S1 Stroke-Field Probe Smoke

#### Method

- `S1_stroke_field_probe_v1` is a V2 skeleton run, not a quality experiment.
- It enables the new `StrokeFieldBrushHead` and `StrokeFieldPseudoRenderer`, which predict stroke-like fields
  (`theta`, `length`, `width`, `curvature`, `alpha`, `prototype_logits`) and compose a lightweight probe residual.
- The goal of S1 is to validate interfaces, checkpointing, and runtime compatibility without committing to a long run.

#### Results

- Config: `configs/janusflow_art/brushstroke/vnext/stroke_field_probe_v1.yaml`
- First smoke attempt exposed a shape bug inside the pseudo-renderer; the bug was fixed immediately in
  `janus/janusflow/models/brush/stroke_field_pseudo_renderer.py`
- Second smoke passed with `--max-steps 1 --skip-final-eval`
- Smoke output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S1_stroke_field_probe_v1`
- Runtime summary confirmed `brush_stroke_field_head_enabled=true` and all legacy brush heads disabled

#### Analysis

- S1 should be interpreted as a successful interface probe, not as evidence that the stroke-field idea already works
  visually.
- What matters is that V2 now has a valid code skeleton inside the same training/checkpoint/runtime stack as V1.
- This gives us a clean fallback path if T1/T2/T3 show that statistics-constrained texture still does not open a new
  brushstroke-strength regime.

### S2 Stroke-Field Adapt Run

#### Method

- S2 is the first real training run for the V2 stroke-field branch. It promotes the earlier smoke-only
  `StrokeFieldBrushHead` into a 300-step adaptation experiment on `train_full_balanced_v1`.
- The run uses the stroke-field parameterization already prepared in S1: the head predicts `theta`, `length`, `width`,
  `curvature`, `alpha`, and `prototype_logits`, and a lightweight pseudo-renderer converts those fields into a local
  probe residual on the decoder feature map.
- To give the branch a fair starting point, S2 initializes from the completed T2 checkpoint through
  `training.init_from_checkpoint`. This warm-start keeps the stronger aligner state learned by the texture-statistics
  branch while allowing the incompatible stroke-field tensors to start fresh.
- Relative to S1, S2 widens the stroke hidden size to `512`, increases `prototype_count` to `12`, sets
  `residual_scale=0.40`, and keeps the same no-reference conservative prompt contract plus low-frequency consistency.
- As with T1/T2, the execution audit required: config audit, warm-start audit, 1-step smoke, fixed-baseline smoke eval,
  canonical baseline hash check, full train, full fixed-baseline eval, and manual compare-card review.

#### Results

- Config: `configs/janusflow_art/brushstroke/vnext/stroke_field_adapt_v1.yaml`
- Init source:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/T2_texture_statistics_head_texture_v1/final_checkpoint`
- Smoke command: passed with `--max-steps 1 --skip-final-eval`
- Fixed-baseline smoke eval output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S2_stroke_field_adapt_v1_smoke_eval/evaluation`
- Smoke canonical baseline hash matched exactly:
  `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`
- Full train output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S2_stroke_field_adapt_v1`
- Final checkpoint:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S2_stroke_field_adapt_v1/final_checkpoint`
- Train-side validation:
  - `eval_loss @ step 75 = 0.8424`
  - `eval_loss @ step 150 = 0.8419`
  - `eval_loss @ step 300 = 0.8468`
  - final checkpoint summary `eval_loss = 0.8527`
- Fixed-baseline evaluation:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S2_stroke_field_adapt_v1/evaluation`
- Key evaluation metrics:
  - `avg_pair_pixel_abs_diff = 5.9869`
  - `avg_prompt_clip_baseline / tuned = 0.3449 / 0.3459`
  - `avg_style_clip_baseline / tuned = 0.2808 / 0.2795`
  - `avg_laplacian_baseline / tuned = 897.32 / 1036.24`
  - `avg_hf_energy_baseline / tuned = 3577.78 / 3742.31`
- Canonical baseline hash matched:
  `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`
- Manual compare-card spot review covered the same fixed seed-42 set:
  - `classical_portrait_oil_ref_high_renaissance`
  - `baroque_drama_ref_baroque`
  - `impressionist_garden_ref_impressionism`
  - `expressionist_interior_ref_expressionism`
  - `ukiyoe_evening_ref_ukiyoe`
  - `ink_wash_mountain_ref_ink_wash`

#### Analysis

- S2 is encouraging in one narrow but important sense: the stroke-field branch is not just a smokeable interface anymore.
  It trains cleanly, keeps prompt CLIP healthy, respects the baseline guardrail, and does not immediately introduce
  portrait/baroque structural failures.
- By eye, however, S2 does **not** yet produce a clearly different regime from T2. The reviewed cards remain safe, and
  the local texture/articulation gains on `impressionist_garden`, `ukiyoe_evening`, and `ink_wash_mountain` are real but
  still modest.
- `classical_portrait` and `baroque_drama` stay structurally intact, which is good. But the branch is still not
  generating the kind of explicit directional stroke logic or materially different local mark-making that would justify
  calling V2 already "broken through."
- Numerically, S2 lands very close to T2 (`avg_pair_pixel_abs_diff 5.9869` vs `5.9056`), but the visual effect is not
  obviously stronger. This suggests that the current pseudo-renderer is too weak or too generic; it is behaving more like
  another constrained local texture head than like a true stroke object model.
- The practical conclusion is that V2 is promising enough to keep, but not yet good enough to replace T2. The next useful
  move is not another same-form adaptation run; it is a renderer upgrade or a stronger stroke-object supervision signal
  inside the V2 branch.

### S3 Stroke-Field Texture Push Run

#### Method

- S3 is the first follow-up run that directly attacks the main weakness exposed by S2: the old pseudo-renderer trained
  stably, but it behaved too much like another cautious texture head instead of a stroke-object generator.
- Before training S3, the V2 renderer was substantially upgraded in code rather than merely retuned in YAML. The new
  `stroke_field_pseudo_renderer.py` now:
  - projects the decoder feature map into a carrier map before rendering;
  - builds an analytic oriented anisotropic prototype bank instead of using a near-scalar prototype response;
  - combines per-pixel `theta`, `length`, `width`, and `prototype_logits` into a compatibility-weighted stroke response;
  - modulates that response with `alpha`, `density`, direction vectors, and curvature;
  - mixes the result back through a learned `post_mix` layer and enforces a high-pass residual before adding it to the
    decoder feature map.
- In other words, S3 is not simply "S2 but longer." It is the first V2 run where the renderer itself has enough internal
  structure to represent oriented local mark-making rather than only generic residual emphasis.
- The training stage also becomes more demanding than S2: instead of the safe full-balanced stage, S3 switches to
  `train_texture_balanced_v2.jsonl`, uses `max_steps=1200`, increases `prototype_count` from `12` to `16`, sets
  `renderer_kernel_size=11`, `render_channels=12`, and raises `residual_scale` to `0.48`.
- The run initializes from the completed S2 checkpoint through `training.init_from_checkpoint`, so compatible backbone and
  aligner weights are inherited while newly shaped renderer tensors start fresh. This is an intentional weight-only warm
  start, not an optimizer/global-step resume.
- The same execution audit contract was applied again: config audit, init-source audit, 1-step smoke, fixed-baseline
  smoke eval, canonical baseline hash verification, full train, full fixed-baseline eval, manual compare-card review, and
  post-run handoff/review-note updates.

#### Results

- Config: `configs/janusflow_art/brushstroke/vnext/stroke_field_texture_push_v1.yaml`
- Init source:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S2_stroke_field_adapt_v1/final_checkpoint`
- Smoke command: passed with `--max-steps 1 --skip-final-eval`
- Fixed-baseline smoke eval output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S3_stroke_field_texture_push_v1_smoke_eval/evaluation`
- Smoke canonical baseline hash matched exactly:
  `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`
- Full train output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S3_stroke_field_texture_push_v1`
- Final checkpoint:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S3_stroke_field_texture_push_v1/final_checkpoint`
- Train-side validation checkpoints:
  - `eval_loss @ step 75 = 0.8691`
  - `eval_loss @ step 525 = 0.8582`
  - `eval_loss @ step 1050 = 0.8634`
  - `eval_loss @ step 1200 = 0.8622`
- Fixed-baseline evaluation:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S3_stroke_field_texture_push_v1/evaluation`
- Key evaluation metrics:
  - `avg_pair_pixel_abs_diff = 6.7126`
  - `avg_prompt_clip_baseline / tuned = 0.3449 / 0.3443`
  - `avg_style_clip_baseline / tuned = 0.2808 / 0.2788`
  - `avg_laplacian_baseline / tuned = 897.32 / 1079.87`
  - `avg_hf_energy_baseline / tuned = 3577.78 / 3802.97`
- Canonical baseline hash matched:
  `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`
- Manual compare-card spot review again covered the fixed seed-42 set:
  - `classical_portrait_oil_ref_high_renaissance`
  - `baroque_drama_ref_baroque`
  - `impressionist_garden_ref_impressionism`
  - `expressionist_interior_ref_expressionism`
  - `ukiyoe_evening_ref_ukiyoe`
  - `ink_wash_mountain_ref_ink_wash`

#### Analysis

- S3 is the strongest V2 stroke-field result so far in raw movement terms. It clearly surpasses S2 and T2 numerically,
  especially on pair difference, Laplacian variance, and high-frequency energy, while still passing the canonical
  baseline guardrail and remaining structurally safe on the reviewed portrait/baroque cards.
- The visual judgment is more mixed, and it is exactly the kind of mixed result that matters for the central research
  question:
  - `classical_portrait` gains slightly more surface articulation around the garment, window edge, and face-shadow
    transitions, but it still reads as a careful texture enrichment rather than an obviously new painterly mark regime.
  - `baroque_drama` remains notably stable. This is an important win because earlier stronger local families tended to
    fail here first.
  - `impressionist_garden` shows the clearest gain: foliage breakup, path texture, and local paint agitation are more
    visible than in T2/S2.
  - `expressionist_interior` becomes more active at edges and color transitions, but the result still reads more like
    controlled directional texture than fully explicit stroke objects.
  - `ukiyoe_evening` gains additional line and paper-surface articulation, but the improvement is still moderate.
  - `ink_wash_mountain` gets denser tree-edge breakup and slightly more mountain-edge articulation, yet it still does not
    produce a decisive "new brush language" moment.
- This makes S3 an informative result rather than a final success. The upgraded renderer does help; the branch is no
  longer stuck at the exact S2/T2 level. But even after the renderer upgrade, the branch still tends to translate stroke
  parameters into organized high-frequency texture emphasis rather than into clearly legible stroke objects.
- That distinction matters for diagnosis. At this point the main bottleneck no longer looks like "we simply trained too
  conservatively." S3 already uses the stronger texture-balanced curriculum, a longer run, a more explicit renderer, and
  enough residual scale to move the image while keeping `baroque` safe. Yet the resulting visual gain is still easier to
  describe as directional local texture than as obvious brushstroke reconstruction.
- Therefore the present limiting factor is more architectural than purely configurational:
  - configuration still matters for safety and for whether the signal moves at all;
  - but the harder ceiling now comes from the representation and renderer being too indirect.
- The next meaningful V2 step should therefore not be "just run even longer" or "just raise the residual scale again."
  It should introduce stronger stroke-object supervision or a renderer that rasterizes elongated local marks more
  explicitly, while preserving the current baseline-audit and portrait-safety contract.

### S4 Stroke-Field Supervised Push Run

#### Method

- S4 keeps the same V2 renderer family as S3, but adds the first explicit stroke-field proxy supervision path. This run
  was introduced because S3 showed a clear ceiling: stronger directional rendering did increase visible movement, but the
  branch still mostly behaved like organized texture enhancement instead of explicit stroke-object generation.
- The new supervision path is implemented entirely in-repo. `brush_proxy_targets.py` now derives stroke-oriented proxy
  targets from the target image:
  - axial dominant orientation from Sobel gradients;
  - structure-tensor coherence to derive preferred local stroke length and width;
  - local density / variance to derive `alpha` occupancy targets;
  - nearest prototype assignment from the renderer's own orientation/length/width anchors.
- Training then adds five new optional loss terms on the stroke-field outputs:
  - `stroke_theta`
  - `stroke_length`
  - `stroke_width`
  - `stroke_alpha`
  - `stroke_prototype`
- `basis_usage_entropy` is also enabled for the stroke head, so the branch is lightly encouraged to make more decisive
  local prototype choices instead of staying overly diffuse.
- Architecturally, S4 still uses the same no-reference V2 branch: stroke-field head on the decoder feature map, no token
  adapter, no reference cross-attention, no output-space residual head. The point of S4 is to isolate representation
  supervision rather than to mix another architectural family into the diagnosis.
- Operationally, this run is also important because it validates a pause/resume workflow inside VNext. The initial long
  run was intentionally paused mid-training, and the completed result resumed from
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S4_stroke_field_supervised_push_v1/checkpoint-525`
  using `--resume-from-checkpoint`. This avoids the known `final_checkpoint` step-restoration ambiguity and confirms that
  schedule-sensitive continuation still behaves correctly in the new branch.

#### Results

- Config: `configs/janusflow_art/brushstroke/vnext/stroke_field_supervised_push_v1.yaml`
- Design note:
  `docs/brushstroke_vnext/stroke_field_supervision_v1.md`
- Init source:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S3_stroke_field_texture_push_v1/checkpoint-1200`
- Resume source for the completed long run:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S4_stroke_field_supervised_push_v1/checkpoint-525`
- Smoke command: passed with `--max-steps 1 --skip-final-eval`
- Fixed-baseline smoke eval output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S4_stroke_field_supervised_push_v1_smoke_eval/evaluation/evaluation`
- Smoke canonical baseline hash matched exactly:
  `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`
- Full train output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S4_stroke_field_supervised_push_v1`
- Final checkpoint:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S4_stroke_field_supervised_push_v1/final_checkpoint`
- Train-side validation checkpoints:
  - `eval_loss @ step 225 = 0.9954`
  - `eval_loss @ step 750 = 0.9905`
  - `eval_loss @ step 1125 = 0.9781`
  - `eval_loss @ step 1200 = 0.9893`
  - final checkpoint summary `eval_loss = 0.9740`
- Fixed-baseline evaluation:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S4_stroke_field_supervised_push_v1/evaluation`
- Key evaluation metrics:
  - `avg_pair_pixel_abs_diff = 7.0761`
  - `avg_prompt_clip_baseline / tuned = 0.3449 / 0.3440`
  - `avg_style_clip_baseline / tuned = 0.2808 / 0.2780`
  - `avg_laplacian_baseline / tuned = 897.32 / 1095.72`
  - `avg_hf_energy_baseline / tuned = 3577.78 / 3819.78`
- Canonical baseline hash matched:
  `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`
- Manual compare-card spot review again covered the fixed seed-42 set:
  - `classical_portrait_oil_ref_high_renaissance`
  - `baroque_drama_ref_baroque`
  - `impressionist_garden_ref_impressionism`
  - `expressionist_interior_ref_expressionism`
  - `ukiyoe_evening_ref_ukiyoe`
  - `ink_wash_mountain_ref_ink_wash`

#### Analysis

- S4 is the strongest V2 result so far numerically and also the first one where the new supervision appears to buy a
  visible gain rather than only cleaner optimization statistics.
- Compared with S3, S4 increases pair difference, Laplacian variance, and high-frequency energy again while preserving
  the same fixed baseline and keeping `portrait` / `baroque` in the safe regime.
- Visual review suggests the gain is real but still bounded:
  - `classical_portrait` shows slightly stronger textile/window articulation and more local brush breakup while keeping
    face and pose stable.
  - `baroque_drama` remains safe, which is crucial. The branch still does not trade strength for the older head/face
    failure mode.
  - `impressionist_garden` again benefits the most; foliage and path handling look more actively repainted than in S3.
  - `expressionist_interior` gets stronger directional edge agitation, but the change still reads more as guided local
    texture than as clearly segmented stroke objects.
  - `ukiyoe_evening` gains a bit more line/paper articulation, though the improvement remains moderate.
  - `ink_wash_mountain` shows stronger tree-edge breakup and mountain-edge activity, but still does not fully become a
    new brush-language regime.
- The important diagnostic outcome is that explicit stroke-field supervision does help. This is not just another
  "longer training" result. The branch now moves farther than S3 without reopening portrait/baroque regression, which
  means the missing ingredient was not only optimization pressure.
- At the same time, S4 still does not fully solve the core perception gap. Even with renderer upgrade plus proxy
  supervision, the branch still tends to produce more organized and directional local texture rather than unmistakably
  object-like brush marks.
- So the post-S4 diagnosis is narrower and more useful:
  - the project is now beyond the stage where "config too conservative" is the dominant explanation;
  - the representation and supervision are moving in the right direction;
  - but the next ceiling now likely sits inside the renderer/object model itself, not just in training length or loss
    weight.
- S4 should therefore become the new V2 baseline to beat. The next meaningful improvement should target a more explicit
  stroke rasterization/composition mechanism or stronger prototype occupancy/objectness constraints, not another generic
  high-frequency sweep.

### S5 Stroke-Field Objectness Push Run

#### Method

- S5 keeps the S4 branch family but tightens the representation around sparse stroke occupancy. The motivation is very
  specific: S4 proved that explicit proxy supervision helps, but the result still looked more like organized directional
  texture than a set of more object-like painterly marks.
- Two architectural changes were introduced on top of the S4 renderer:
  - a new `objectness_head` predicts a spatial occupancy logit for where a stroke-like mark should exist at all;
  - prototype routing is no longer fully dense. The renderer now keeps only the top-`k` prototype weights per location
    (`topk_prototypes=2`) and renormalizes them before rendering.
- The pseudo-renderer was also made more selective:
  - `objectness` is converted to a local occupancy map and lightly blurred into an `objectness_blob`;
  - the final stroke gate becomes `alpha * objectness_blob * density_term`, so low-confidence regions are discouraged
    from receiving diffuse texture residue;
  - the final rendered residual is multiplied by `objectness_blob`, making the branch explicitly pay for turning on a
    stroke-like edit region.
- The training target path was extended accordingly. `brush_proxy_targets.py` now derives an `objectness_target` from
  the existing stroke-alpha occupancy heuristic, and training adds a new `stroke_objectness` loss on top of the S4
  `stroke_theta / stroke_length / stroke_width / stroke_alpha / stroke_prototype` supervision bundle.
- This is not a generic "stronger loss" rerun. S5 is a representation change inside V2: the branch is explicitly asked
  to decide both *what* local stroke prototype to render and *whether* a localized mark should exist there at all.
- Code-execution audit was completed before the long run:
  - config review: passed; `prompt_template=conservative`, no token adapter, no reference cross-attention, no output
    residual heads;
  - checkpoint review: passed; initialized from
    `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S4_stroke_field_supervised_push_v1/checkpoint-1200`
    through the weight-only init path rather than schedule resume;
  - smoke review: passed;
  - baseline review: passed with canonical hash verification before full training.

#### Results

- Config:
  `configs/janusflow_art/brushstroke/vnext/stroke_field_objectness_push_v1.yaml`
- Design note:
  `docs/brushstroke_vnext/stroke_field_objectness_v2.md`
- Init source:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S4_stroke_field_supervised_push_v1/checkpoint-1200`
- Smoke command: passed with `--max-steps 1 --skip-final-eval`
- Fixed-baseline smoke eval output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S5_stroke_field_objectness_push_v1_smoke_eval/evaluation/evaluation`
- Smoke canonical baseline hash matched exactly:
  `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`
- Full train output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S5_stroke_field_objectness_push_v1`
- Numbered completion checkpoint used for evaluation:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S5_stroke_field_objectness_push_v1/checkpoint-1200`
- `final_checkpoint` weight hash matches `checkpoint-1200` for `art_state.pt`, but
  `final_checkpoint/checkpoint_summary.json` still stores `step: null`, so future continuation should resume from the
  numbered checkpoint rather than the final alias.
- Train-side validation checkpoints:
  - `eval_loss @ step 375 = 0.9795`
  - `eval_loss @ step 600 = 0.9810`
  - `eval_loss @ step 975 = 0.9889`
  - `eval_loss @ step 1050 = 0.9726`
  - `eval_loss @ step 1125 = 0.9867`
  - `eval_loss @ step 1200 = 0.9718`
- Fixed-baseline evaluation:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S5_stroke_field_objectness_push_v1/evaluation`
- Review notes:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S5_stroke_field_objectness_push_v1/review_notes.md`
- Key evaluation metrics:
  - `avg_pair_pixel_abs_diff = 7.5529`
  - `avg_prompt_clip_baseline / tuned = 0.3449 / 0.3436`
  - `avg_style_clip_baseline / tuned = 0.2808 / 0.2784`
  - `avg_laplacian_baseline / tuned = 897.32 / 1150.09`
  - `avg_hf_energy_baseline / tuned = 3577.78 / 3892.21`
- Canonical baseline hash matched:
  `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`
- Manual compare-card review again focused first on the fixed seed-42 set:
  - `classical_portrait_oil_ref_high_renaissance`
  - `baroque_drama_ref_baroque`
  - `impressionist_garden_ref_impressionism`
  - `expressionist_interior_ref_expressionism`
  - `ukiyoe_evening_ref_ukiyoe`
  - `ink_wash_mountain_ref_ink_wash`

#### Analysis

- S5 is the strongest V2 result so far numerically. Relative to S4, it pushes pair difference, Laplacian variance, and
  high-frequency energy all upward again, so the objectness/top-`k` change is not a cosmetic refactor.
- The visual outcome is mixed in a useful way:
  - `classical_portrait` remains composition-safe and face-safe; the gain is modest but cleaner sleeve/window surface
    articulation is visible.
  - `baroque_drama` stays in the safe regime and does not reintroduce the older head/face collapse, which is a real
    success for a stronger V2 run.
  - `impressionist_garden` is the clearest positive case; foliage and path breakup are denser and more decisively
    repainted than in S4.
  - `expressionist_interior` becomes more segmented and stroke-like locally, but still not to the level of unmistakable
    explicit brush objects.
  - `ukiyoe_evening` gains stronger line/surface articulation, though it still reads as enhanced printed structure more
    than a qualitatively new renderer regime.
  - `ink_wash_mountain` exposes the main new failure tendency: the branch can now over-activate sparse regions and add a
    new upper-tree mass / extra structure, which reads as mild composition rewriting rather than only brush refinement.
- This is the most informative part of S5. The experiment shows that the current bottleneck is not simply "training too
  conservative." The branch is already strong enough to move the image substantially while preserving portrait/baroque
  safety most of the time.
- The remaining limitation is more architectural than optimizer-related:
  - `objectness` and sparse prototype routing help make edits more localized;
  - but the current pseudo-renderer still composes those localized edits as texture-like raster patches rather than as
    clearly coherent stroke objects;
  - once pushed harder, the branch starts hallucinating plausible local forms in sparse low-content regions instead of
    only enriching the existing mark structure.
- Therefore the post-S5 diagnosis is sharper than post-S4:
  - configuration is no longer the main limiter;
  - the V2 representation is improving and worth keeping;
  - but future gains should come from stronger stroke-object composition rules, blank-region suppression, or more
    explicit rasterization primitives rather than another generic increase in training length or loss weight.
- S5 becomes the new V2 baseline to beat, but it does **not** yet solve the central perceptual requirement. By eye, the
  branch is stronger and more structured than S4, yet it still falls short of a fully obvious "one-stroke-at-a-time"
  painterly regime.

### S6 Stroke-Field Sparse Blank Suppression Run

#### Method

- S6 is the direct follow-up to S5 and targets the clearest new failure mode exposed there: sparse low-content regions,
  especially `ink_wash` sky/empty-paper areas, could still be activated by the stroke-field branch and receive invented
  new tree/branch structure.
- The architecture remains inside the same V2 family rather than switching renderer class. This makes S6 a controlled
  diagnosis of whether better occupancy control is enough before a larger renderer redesign.
- Three concrete changes were introduced on top of S5:
  - prototype routing is made singular with `topk_prototypes=1` instead of `2`, so each spatial location must commit to
    a single dominant stroke template rather than blending multiple local prototypes;
  - the renderer support is elongated with `renderer_kernel_size=13`, so the surviving prototype can express a longer
    stroke primitive instead of a shorter texture patch;
  - training adds an explicit `stroke_blank_suppression` loss with weight `0.08`.
- The new suppression target is derived fully in-repo in `finetune/brush_proxy_targets.py`. It computes
  `support_target = max(density, pigment)`, converts low-support areas into a soft `blank_region_target`, and lightly
  blurs that map before training. The loss then penalizes `sigmoid(objectness)` where the target image implies
  low-support blank space.
- This means S6 is not merely "push S5 harder." It is a controlled attempt to make the V2 branch both more singular and
  more selective: fewer mixed prototypes, longer marks, and a direct cost on turning objectness on in regions that
  should remain visually quiet.
- Code-execution audit was completed before the long run:
  - config review: passed; `prompt_template=conservative`, no token adapter, no reference cross-attention, no output
    residual heads, and the baseline-disable logic still covers all learned art modules;
  - checkpoint review: passed; initialized from
    `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S5_stroke_field_objectness_push_v1/checkpoint-1200`
    via `init_from_checkpoint`, not schedule resume;
  - smoke review: passed;
  - fixed-baseline smoke eval: passed with canonical baseline hash verification.

#### Results

- Config:
  `configs/janusflow_art/brushstroke/vnext/stroke_field_sparse_blank_suppression_v1.yaml`
- Design note:
  `docs/brushstroke_vnext/stroke_field_sparse_blank_v3.md`
- Init source:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S5_stroke_field_objectness_push_v1/checkpoint-1200`
- Smoke command: passed with `--max-steps 1 --skip-final-eval`
- Fixed-baseline smoke eval output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S6_stroke_field_sparse_blank_suppression_v1_smoke_eval/evaluation/evaluation`
- Smoke canonical baseline hash matched exactly:
  `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`
- Full train output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S6_stroke_field_sparse_blank_suppression_v1`
- Numbered completion checkpoint used for evaluation:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S6_stroke_field_sparse_blank_suppression_v1/checkpoint-1200`
- `final_checkpoint` was written successfully, but `final_checkpoint/checkpoint_summary.json` again stores `step: null`,
  so future continuation must still use the numbered checkpoint rather than the final alias.
- Train-side validation checkpoints:
  - `eval_loss @ step 75 = 0.9795`
  - `eval_loss @ step 300 = 0.9820`
  - `eval_loss @ step 525 = 0.9656`
  - `eval_loss @ step 750 = 0.9871`
  - `eval_loss @ step 825 = 0.9872`
  - `eval_loss @ step 1200 = 0.9686`
- Operational note: after artifacts were written, the training tail remained resident and briefly blocked evaluation by
  holding GPU memory. The fixed-baseline eval was rerun after explicitly stopping the hung trainer session. This does not
  invalidate the experiment; it is an execution detail that should be remembered for future long runs.
- Fixed-baseline evaluation:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S6_stroke_field_sparse_blank_suppression_v1/evaluation`
- Review notes:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S6_stroke_field_sparse_blank_suppression_v1/review_notes.md`
- Key evaluation metrics:
  - `avg_pair_pixel_abs_diff = 7.8861`
  - `avg_prompt_clip_baseline / tuned = 0.3449 / 0.3451`
  - `avg_style_clip_baseline / tuned = 0.2808 / 0.2774`
  - `avg_laplacian_baseline / tuned = 897.32 / 1162.76`
  - `avg_hf_energy_baseline / tuned = 3577.78 / 3909.09`
- Canonical baseline hash matched:
  `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2`
- Manual compare-card review again focused first on the fixed seed-42 set:
  - `classical_portrait_oil_ref_high_renaissance`
  - `baroque_drama_ref_baroque`
  - `impressionist_garden_ref_impressionism`
  - `expressionist_interior_ref_expressionism`
  - `ukiyoe_evening_ref_ukiyoe`
  - `ink_wash_mountain_ref_ink_wash`

#### Analysis

- S6 is stronger than S5 by the aggregate metrics that have tracked real movement throughout the V2 line. The branch did
  not become overly timid after adding blank suppression; pair difference, Laplacian variance, and high-frequency energy
  all rise again.
- The result is also not a prompt-adherence collapse. Average prompt CLIP improves slightly over S5, while the canonical
  baseline hash still matches exactly. This matters because it rules out the easy but misleading explanation that S6
  "looks different only because eval drifted."
- The per-prompt picture is more informative than the averages:
  - `classical_portrait`: slightly stronger local material handling than S5, but still face-safe and composition-safe.
  - `baroque_drama`: remains in the safe regime; the branch still does not reintroduce the older head/face collapse.
  - `impressionist_garden`: gains a small but real increase in path/foliage repaint density over S5.
  - `expressionist_interior`: becomes even more segmented and active locally, but still reads more like strong
    directional repaint than unmistakable explicit stroke objects.
  - `ukiyoe_evening`: gets clearer line/surface articulation and a modest gain in stylized structure.
  - `ink_wash_mountain`: this is the key verdict. The upper sparse region still acquires invented tree/branch structure,
    and the failure is not meaningfully removed by the new blank suppression.
- That makes S6 a useful success/failure hybrid:
  - success, because it proves the branch can be pushed further without sacrificing portrait/baroque safety;
  - failure, because the targeted fix does not actually solve the sparse-region hallucination it was designed to fix.
- The most likely reason is architectural. The new blank penalty is soft, image-derived, and applied to objectness only.
  But the renderer still composes edits through elongated semi-implicit primitives, so once a location is activated the
  branch can still produce coherent new sparse forms instead of merely refining pre-existing marks.
- Therefore the post-S6 diagnosis is stronger than post-S5:
  - the main bottleneck is not "training too conservative";
  - the main bottleneck is also not merely "missing one more regularizer";
  - the main bottleneck is that the current pseudo-renderer still lacks a sufficiently explicit occupancy rule tying
    stroke activation to existing visual support.
- S6 becomes the new V2 baseline to beat because it is stronger than S5 while still keeping `portrait` / `baroque`
  stable. But it also closes the door on another same-family penalty sweep as the default next move. The next meaningful
  improvement should change the renderer/object model itself, especially for sparse-region activation.

## Data Rebuild R1 (Full Official Rebuild)

### Method

- R1 replaces the legacy texture-rich subset as the default source of truth for brushstroke training data. The rebuild starts from `/root/autodl-tmp/data/emoart_5k/annotation.official.json`, which is complete and internally consistent with the local image directory.
- The rebuild script is `build_emoart_full_rebuild_v1.py`. For each official sample it resolves the real image path, verifies the image file, extracts structured text fields from `description`, computes `frame_risk_score` from image border statistics, computes a text-and-image `texture_score` without any style-name bonus, and tags portrait/head-sensitive samples by lightweight text heuristics.
- Three prompt variants are materialized for every sample:
  - `prompt_v2_content_safe`: default balanced training prompt
  - `prompt_v2_texture_push`: stronger brushwork/surface cue prompt for non-head-sensitive continuation samples
  - `prompt_v2_portrait_safe`: conservative prompt with explicit structure-preservation language for head-sensitive continuation samples
- The split policy is no longer random global 10%. Instead, validation is style-balanced (`10` per style where available), the main train manifest is the remainder, `train_texture_balanced_v2` selects `28` samples per style with explicit head-sensitive retention, and `train_portrait_structure_v1` selects `20` head-sensitive samples from portrait-dominant styles.
- One implementation detail mattered in practice: because `Gongbi` has only `32` official images, the continuation texture manifest repeats some `Gongbi` rows to preserve the intended per-style training weight of `28`.

### Results

- Script: `/root/autodl-tmp/repos/Janus/build_emoart_full_rebuild_v1.py`
- Output root: `/root/autodl-tmp/data/emoart_5k/gen_full_official_rebuild_v1`
- Master manifest: `5532` valid records, `0` bad records
- Validation split: `560` records, all `56` styles represented with exactly `10` samples each
- Main train split: `4972` records
- Texture continuation split: `1568` records, all `56` styles represented with exactly `28` rows each
- Portrait recovery split: `440` records across `22` portrait-dominant styles, all rows `head_sensitive=true`
- Subject summary:
  - master portrait candidates: `3331`
  - master head-sensitive records: `2540`
  - portrait-dominant styles: `22`
- Data card and summaries were written to:
  - `/root/autodl-tmp/data/emoart_5k/gen_full_official_rebuild_v1/style_summary.json`
  - `/root/autodl-tmp/data/emoart_5k/gen_full_official_rebuild_v1/subject_summary.json`
  - `/root/autodl-tmp/data/emoart_5k/gen_full_official_rebuild_v1/data_card.md`
- Smoke verification:
  - `brush_spatial_hf_head_full_balanced_v1.yaml` 1-step smoke passed with `train_size=4972`, `val_size=560`, `train_missing_image_records=0`, `val_missing_image_records=0`
  - `brush_spatial_hf_head_texture_balanced_v2.yaml` 1-step smoke passed with `train_size=1568`, `val_size=560`, `train_missing_image_records=0`, `val_missing_image_records=0`

### Analysis

- This rebuild converts the current brushstroke work from a texture-biased subset experiment into a staged data program. That matters because the later architecture failures are not random; they repeatedly break on portrait/head-sensitive styles where the old subset under-served structure-safe recovery.
- The rebuilt manifests now encode a concrete training curriculum that matches the empirical failure pattern:
  1. `train_full_balanced_v1` for broad semantics and composition retention
  2. `train_texture_balanced_v2` for visible material/brushstroke pressure
  3. `train_portrait_structure_v1` for targeted recovery if head drift persists
- The path-normalization bug is also an important outcome. Without fixing it, the new manifests would have looked partially broken during smoke tests, and later training analysis would have mixed real data issues with a loader artifact.
- The next experiments should therefore use the new balanced manifests before introducing another major brush architecture. If the baroque/portrait failure remains after this cleaner data curriculum, confidence increases that the remaining limitation is truly architectural rather than a data-sampling artifact.

## Early-Phase Screens: E1 / E2 / E3 / E3A / E3B

### E1 Completed Run (Adapter-Only Baseline)

Method:
- E1 is the clean no-reference baseline for the brush route. The run adds only a decoder-token `BrushAdapter` after `vision_gen_dec_aligner`, keeps the LLM, VAE, and generation decoder frozen, and applies the adapter only to the conditioned CFG branch.
- The architectural hypothesis was conservative: if a small local residual on decoder tokens is enough, brushstroke/material detail should improve without perturbing pose or composition.
- In practice the adapter remained very small throughout training: effective residual gate stayed around `0.05005`, with only `1.09M` trainable parameters.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_adapter_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E1_brush_adapter_v1_cfgcond`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E1_brush_adapter_v1_cfgcond/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E1_brush_adapter_v1_cfgcond/evaluation`
- Final eval loss: `0.6830`
- Average pair pixel absolute difference: `1.5669`
- Average Laplacian variance baseline / tuned: `890.9734 / 892.9926`
- Average high-frequency energy baseline / tuned: `3576.7521 / 3584.9158`
- Average prompt CLIP baseline / tuned: `0.3175 / 0.3170`

Analysis:
- E1 is composition-safe and quality-preserving, but the effect size is too small to matter perceptually.
- Manual inspection shows only small edge, color, and local-contrast shifts; there is no convincing increase in painterly brushstroke or surface material.
- This run establishes the central failure mode of the adapter-only route: it is safe, but it does not have enough leverage over perceptual texture.

### E2 Completed Run (Adapter + HF Auxiliary Losses)

Method:
- E2 keeps the same adapter-only topology as E1 and adds Laplacian and Sobel auxiliary losses in latent space.
- The intended mechanism was to bias the model toward stronger local edge energy and high-frequency detail without enlarging the trainable module footprint.
- Importantly, E2 still has no reference branch and no decoder cross-attention injection; the only change from E1 is the optimization target.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_adapter_hf_loss_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E2_brush_adapter_hf_loss_v1_cfgcond`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E2_brush_adapter_hf_loss_v1_cfgcond/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E2_brush_adapter_hf_loss_v1_cfgcond/evaluation`
- Final eval loss: `0.8589`
- Average pair pixel absolute difference: `1.6324`
- Average Laplacian variance baseline / tuned: `890.9734 / 891.1010`
- Average high-frequency energy baseline / tuned: `3576.7521 / 3575.2295`
- Average prompt CLIP baseline / tuned: `0.3175 / 0.3179`

Analysis:
- E2 is only marginally different from E1 by both metrics and manual review.
- The HF auxiliary losses do not translate into obvious painterly texture; the run still reads as a near-copy baseline with slight edge/transition perturbations.
- This is important diagnostically: simply telling the loss to value edges does not force the model to synthesize visible brushstroke structure.

### E3 Completed Run (Multiscale Reference Conditioning)

Method:
- E3 is the first run where the brush system receives explicit reference-style information through a patch reference encoder plus decoder-local cross-attention injection.
- The adapter remains active, but the main new degree of freedom is multiscale brush tokens entering the decoder path. The hypothesis was that explicit local style tokens could provide texture cues that the no-reference adapter route could not discover alone.
- Audit fixes were already active: conditioned-only CFG application, true reference-backed prompt records, and a clean base-model baseline pipeline.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_multiscale_ref_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3_brush_multiscale_ref_v1_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3_brush_multiscale_ref_v1_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3_brush_multiscale_ref_v1_cfgcond_refeval/evaluation`
- Final eval loss: `0.8693`
- Trainable parameters: `4.02M`
- Average pair pixel absolute difference: `11.6109`
- Average Laplacian variance baseline / tuned: `890.9734 / 994.1039`
- Average high-frequency energy baseline / tuned: `3576.7521 / 3621.3108`
- Average prompt CLIP baseline / tuned: `0.3175 / 0.3128`
- `reference_path_count`: `24`

Analysis:
- E3 is the first run that is unambiguously active by eye. It changes texture, local contrast, and edge breakup much more strongly than E1/E2.
- However, the effect is not local enough: expressionist, portrait, ink, ukiyo-e, and baroque cards show pose/layout/object-placement changes rather than isolated brushstroke transfer.
- This run proves that explicit style information can generate strong visual movement, but it also proves that the current reference-conditioning mechanism is semantically entangled.

### E3A Completed Run (No-Reference Ablation)

Method:
- E3A keeps the E3 training/evaluation schedule and prompt bundle but removes the reference encoder and reference injector.
- The purpose is not to improve quality directly, but to identify whether the visible movement in E3 came from the adapter/high-frequency losses or from the reference pathway itself.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_ablation_no_ref.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3A_brush_ablation_no_ref_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3A_brush_ablation_no_ref_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3A_brush_ablation_no_ref_cfgcond_refeval/evaluation`
- Final eval loss: `0.8693`
- Average pair pixel absolute difference: `1.6026`
- Average Laplacian variance baseline / tuned: `890.9734 / 891.0231`
- Average high-frequency energy baseline / tuned: `3576.7521 / 3585.4845`
- Average prompt CLIP baseline / tuned: `0.3175 / 0.3178`

Analysis:
- E3A collapses back to the E1/E2 regime: composition-safe, but visually weak.
- This ablation cleanly isolates the source of E3's large effect size. The visible movement is overwhelmingly caused by the reference branch, not by the adapter alone.
- That conclusion is foundational for all later design work: if the project wants stronger texture without global drift, the missing piece is not "more loss weight" but a better-localized source of leverage.

### E3B Completed Run (Reference Branch Without HF Losses)

Method:
- E3B keeps the reference encoder and decoder-local reference injector from E3, but disables Laplacian and Sobel auxiliary losses.
- The question here is whether the global drift seen in E3 was a side effect of high-frequency optimization pressure, or whether it was intrinsic to the reference-conditioning route.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_ablation_no_hf_loss.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3B_brush_ablation_no_hf_loss_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3B_brush_ablation_no_hf_loss_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3B_brush_ablation_no_hf_loss_cfgcond_refeval/evaluation`
- Final eval loss: `0.6923`
- Average pair pixel absolute difference: `11.8640`
- Average Laplacian variance baseline / tuned: `890.9734 / 1038.8388`
- Average high-frequency energy baseline / tuned: `3576.7521 / 3662.8757`
- Average prompt CLIP baseline / tuned: `0.3175 / 0.3133`
- `reference_path_count`: `24`

Analysis:
- E3B still drifts. Removing the HF losses does not restore locality or composition retention.
- In other words, the problem is not that Laplacian/Sobel "made the model too sharp"; the problem is that reference cross-attention itself transmits too much semantic/layout information.
- This result justifies the later shift away from direct reference-conditioning as the main route.

## E3D Completed Run

E3D was introduced to test a less conservative adapter with more local reference injection:

- Config: `configs/janusflow_art/brushstroke/brush_multiscale_ref_local_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3D_brush_ref_fine_local_v1_cfgcond_refeval`
- Status: resumed from `checkpoint-150`, completed at 450 steps, and evaluated.
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3D_brush_ref_fine_local_v1_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3D_brush_ref_fine_local_v1_cfgcond_refeval/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3D_brush_ref_fine_local_v1_cfgcond_refeval/review_notes.md`

E3D changes:

- Adapter bottleneck: `192 -> 256`
- Adapter residual init: `0.05 -> 0.10`
- Adapter residual scale: `1.0 -> 1.25`
- Effective gate at checkpoint-150: about `0.1251`
- Reference injector: fine tokens only, `use_mid_tokens: false`
- Cross-attention scale: `0.35 -> 0.16`
- HF auxiliary losses: disabled
- Trainable params: about `4.12M`

Training/evaluation metrics:

- Train eval loss at step 300: `0.6919`
- Train eval loss at step 450: `0.6985`
- Final eval loss saved in `final_summary.json`: `0.6802`
- Average pair pixel absolute difference: `9.8600`
- Average Laplacian variance baseline / tuned: `890.9734 / 990.6650`
- Average high-frequency energy baseline / tuned: `3576.7521 / 3631.2237`
- Average prompt CLIP baseline / tuned: `0.3175 / 0.3124`
- Average style CLIP baseline / tuned: `0.2825 / 0.2817`
- `reference_path_count`: `24`

Metric-only read: E3D is less globally different than E3/E3B by average pair pixel difference, while still much stronger than E3A/no-reference. This is encouraging enough to review manually, but not enough to call it successful.

Manual review conclusion:
- Expressionist interior, portrait, ink wash, and ukiyo-e still show positional/structural drift.
- Impressionist garden is mostly preserved but not clearly improved in brushstroke detail.
- E3D is not acceptable as a production setting.

Clean rerun command:

```bash
python train_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_multiscale_ref_local_v1.yaml
```

Eval command after a completed checkpoint:

```bash
python eval_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_multiscale_ref_local_v1.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3D_brush_ref_fine_local_v1_cfgcond_refeval/final_checkpoint \
  --output-dir /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3D_brush_ref_fine_local_v1_cfgcond_refeval
```

## E4 Completed Run (High-Pass Reference)

Method:
- E4 keeps reference-conditioning active but routes the reference image through a high-pass residual preprocessing path before encoding.
- The architectural intent is to strip away low-frequency composition cues from the reference image and preserve only local texture/edge statistics.
- Relative to E3D, this is a direct attempt to make the reference branch act more like a texture donor and less like a scene-layout conditioner.

- Config: `configs/janusflow_art/brushstroke/brush_highpass_ref_fine_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E4_brush_highpass_ref_fine_v1_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E4_brush_highpass_ref_fine_v1_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E4_brush_highpass_ref_fine_v1_cfgcond_refeval/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E4_brush_highpass_ref_fine_v1_cfgcond_refeval/review_notes.md`

Results:
- Average pair pixel absolute difference: `8.7787`
- Average Laplacian variance baseline / tuned: `890.9734 / 967.0690`
- Average high-frequency energy baseline / tuned: `3576.7521 / 3598.4261`
- Average prompt CLIP baseline / tuned: `0.3175 / 0.3128`
- Average style CLIP baseline / tuned: `0.2825 / 0.2828`

Analysis:
- High-pass reference reduces drift compared to E3/E3B, but expressionist/portrait/ink/ukiyo-e still show composition shifts.
- Not acceptable as a stable local brushstroke transfer.

## E5 Completed Run (Adapter + Dec Aligner)

Method:
- E5 abandons the reference branch entirely and instead slightly unfreezes `vision_gen_dec_aligner` together with the adapter.
- The core hypothesis is that the adapter-only path may have been too weak because it could only add a tiny residual after a fully frozen decoder-token projection. Allowing the aligner to move could increase local generation leverage without introducing reference-induced drift.
- This run is intentionally conservative: aligner LR is only `1e-5`, and the rest of the generation stack remains frozen.

- Config: `configs/janusflow_art/brushstroke/brush_adapter_dec_aligner_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E5_brush_adapter_dec_aligner_v1_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E5_brush_adapter_dec_aligner_v1_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E5_brush_adapter_dec_aligner_v1_cfgcond_refeval/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E5_brush_adapter_dec_aligner_v1_cfgcond_refeval/review_notes.md`

Results:
- Average pair pixel absolute difference: `1.9380`
- Average Laplacian variance baseline / tuned: `890.9734 / 889.8027`
- Average high-frequency energy baseline / tuned: `3576.7521 / 3576.9898`
- Average prompt CLIP baseline / tuned: `0.3175 / 0.3161`
- Average style CLIP baseline / tuned: `0.2825 / 0.2820`

Analysis:
- Composition is preserved across styles, but brushstroke/texture changes are near-invisible at full-card scale.
- E5 is too conservative for the project goal.

## Pending Experiments

- E6: `configs/janusflow_art/brushstroke/brush_highpass_ref_expert_v1.yaml` (high-pass + experts = 2). Staged but paused.
- E13: `configs/janusflow_art/brushstroke/brush_spatial_hf_head_stronger_v1.yaml` (keep as an optional diagnostic only; current expectation is that it will worsen the already-unsolved E12 baroque drift).

## E8 Completed Run (LoRA Visible-Effect Probe)

Method:
- E8 is not part of the intended final architecture family; it is a visibility probe that asks how much brushstroke/material change the current data/model pair can produce at all.
- Brush modules are disabled, and the trainable path is instead a language-model LoRA on attention projections (`q/k/v/o`) over the last 16 layers.
- The critical design choice is the evaluation sweep: the same checkpoint is rendered at multiple inference strengths (`0.15`, `0.25`, `0.50`, `0.75`) so effect size and structural drift can be disentangled from raw trainability.

- Config: `configs/janusflow_art/brushstroke/brush_lora_visible_probe_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E8_brush_lora_visible_probe_v1`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E8_brush_lora_visible_probe_v1/final_checkpoint`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E8_brush_lora_visible_probe_v1/review_notes.md`

Run note:
- This first run stopped at `288` steps rather than `450` because `num_epochs: 3.0` with batch size `16` only covers `288` updates on the current subset.
- The config has been corrected to `num_epochs: 5.0` for future reruns.

Results:
- Final eval loss in `final_summary.json`: `0.6884`
- Trainable parameters: `16.78M`

Scale sweep summary:
- `0.15`: pair diff `5.3129`, Laplacian `946.1889`, HF energy `3653.6742`
- `0.25`: pair diff `8.1056`, Laplacian `988.0079`, HF energy `3697.4788`
- `0.50`: pair diff `17.3789`, Laplacian `994.5488`, HF energy `3645.2052`
- `0.75`: pair diff `22.3073`, Laplacian `1081.0996`, HF energy `3730.3046`

Analysis:
- `0.15` is visible but still somewhat conservative.
- `0.25` is the best safe tradeoff in this sweep and remains useful as a structure-preserving reference point.
- `0.75` is the strongest perceptual brushstroke regime by eye and is therefore useful as a target-strength reference, but it also introduces portrait/head blur and semantic drift.
- E8 therefore separates two goals that the brush architecture must reconcile: the visible effect size should approach `0.75`, while the structural safety should remain closer to `0.25` or E10.

## E9 Completed Run (Local Capacity V2)

Method:
- E9 is the first v2 local-capacity design. It combines the earlier token-stage adapter with a new feature-map-stage residual block applied on the reshaped decoder feature map before `vision_gen_dec_model`.
- The rationale is that token-only editing may be too abstract to change painterly texture strongly, while feature-map-space editing is closer to local image structure and may provide a more direct handle on brush/material statistics.
- The run remains no-reference, preserving the central design constraint that stronger texture should not come from explicit style-token injection.

- Config: `configs/janusflow_art/brushstroke/brush_local_capacity_v2_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E9_brush_local_capacity_v2_v1_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E9_brush_local_capacity_v2_v1_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E9_brush_local_capacity_v2_v1_cfgcond_refeval/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E9_brush_local_capacity_v2_v1_cfgcond_refeval/review_notes.md`

Results:
- Step 150 eval loss: `0.8933`
- Step 300 eval loss: `0.8688`
- Step 450 eval loss: `0.8797`
- Final checkpoint eval loss: `0.8684`
- Trainable parameters: `4.35M`

Evaluation summary:
- Average pair pixel absolute difference: `6.9286`
- Average Laplacian variance baseline / tuned: `890.9734 / 940.9259`
- Average high-frequency energy baseline / tuned: `3576.7521 / 3646.8982`
- Average prompt CLIP baseline / tuned: `0.3175 / 0.3159`
- Average style CLIP baseline / tuned: `0.2825 / 0.2816`
- `reference_path_count`: `24`

Analysis:
- E9 is safer than the old reference-conditioned runs and slightly richer than E7 on local material handling.
- Portrait, impressionist, and several ukiyo-e / ink cards keep the overall framing and subject placement close to baseline.
- The gain is still not enough to match the visible force of E8 at high scale, and many cards still read as mild repainting rather than clear new brushstroke structure.
- Baroque remains a visible failure mode: repeated face/head highlight reshaping and expression drift show that the current hybrid token + feature-map route can still leak into structure.
- Treat E9 as a useful architectural validation, not as a candidate final setting.

## E10 Completed Run (Feature-Map Only Aggressive)

Method:
- E10 removes the token-stage adapter from E9 and keeps only the feature-map local branch, together with low-level decoder aligner tuning.
- This is a pure localization test: if the token adapter was the main source of semantic leakage, then disabling it should improve identity/composition retention while preserving at least part of the local-material gain.
- The feature-map branch itself is pushed harder than in E9, but still remains a single residual family without explicit spatial gating.

- Config: `configs/janusflow_art/brushstroke/brush_feature_map_only_aggressive_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E10_brush_feature_map_only_aggressive_v1_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E10_brush_feature_map_only_aggressive_v1_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E10_brush_feature_map_only_aggressive_v1_cfgcond_refeval/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E10_brush_feature_map_only_aggressive_v1_cfgcond_refeval/review_notes.md`

Results:
- Step 450 eval loss: `0.9809`
- Trainable parameters: `3.17M`
- Feature-map effective initial gate: `0.3003`

Evaluation summary:
- Average pair pixel absolute difference: `6.3596`
- Average Laplacian variance baseline / tuned: `890.9734 / 928.0587`
- Average high-frequency energy baseline / tuned: `3576.7521 / 3615.9148`
- Average prompt CLIP baseline / tuned: `0.3175 / 0.3166`
- Average style CLIP baseline / tuned: `0.2825 / 0.2817`
- `reference_path_count`: `24`

Analysis:
- Portrait and baroque are safer than E9; the most obvious face/head drift is reduced.
- Impressionist / expressionist / ink still gain some local material density, but the overall effect is weaker than E9 and far below the visible brushstroke strength seen in E8 `0.75`.
- E10 therefore supports feature-map-only as the safer branch, but not this exact strength setting.

## E11 Completed Run (Feature-Map Only Stronger)

Method:
- E11 is the direct strength escalation of E10. It keeps the feature-map-only design but increases hidden width, kernel size, effective gate, and decoder-side learning rate.
- The purpose is diagnostic rather than final: determine whether the feature-map-only route is fundamentally too weak, or whether it can in fact reach high visible brushstroke strength if given enough capacity.

- Config: `configs/janusflow_art/brushstroke/brush_feature_map_only_stronger_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E11_brush_feature_map_only_stronger_v1_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E11_brush_feature_map_only_stronger_v1_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E11_brush_feature_map_only_stronger_v1_cfgcond_refeval/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E11_brush_feature_map_only_stronger_v1_cfgcond_refeval/review_notes.md`

Results:
- Step 450 eval loss: `1.0597`
- Final checkpoint eval loss: `1.0591`
- Trainable parameters: `3.57M`
- Feature-map effective initial gate: `0.5586`

Evaluation summary:
- Average pair pixel absolute difference: `9.4540`
- Average Laplacian variance baseline / tuned: `890.9734 / 969.8505`
- Average high-frequency energy baseline / tuned: `3576.7521 / 3662.1727`
- Average prompt CLIP baseline / tuned: `0.3175 / 0.3168`
- Average style CLIP baseline / tuned: `0.2825 / 0.2823`
- `reference_path_count`: `24`

Analysis:
- E11 is visibly stronger than E10 on impressionist / expressionist / ink and shows that the feature-map-only branch can produce meaningful local material change.
- Portrait already starts to reshape slightly, and baroque regresses into obvious face/head distortion.
- This is therefore not a target setting. It proves the branch has leverage, but that locality control is still insufficient.

## E7 Completed Run (Aggressive No-Reference Adapter)

Method:
- E7 is the maximal no-reference token-adapter diagnostic before moving into v2 architecture work.
- Relative to E5, it increases adapter bottleneck width, raises residual gate and residual scale, enables four decoder-local experts, and keeps `vision_gen_dec_aligner` trainable at a substantially higher generation-side learning rate.
- This run answers a very specific question: if the architecture stays token-local and no-reference, is the lack of visible brushstroke gain simply because earlier settings were too conservative?

- Config: `configs/janusflow_art/brushstroke/brush_adapter_dec_aligner_aggressive_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E7_brush_adapter_dec_aligner_aggressive_v1_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E7_brush_adapter_dec_aligner_aggressive_v1_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E7_brush_adapter_dec_aligner_aggressive_v1_cfgcond_refeval/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E7_brush_adapter_dec_aligner_aggressive_v1_cfgcond_refeval/review_notes.md`

Results:
- Train eval loss at step 150: `0.9758`
- Train eval loss at step 300: `0.9796`
- Train eval loss at step 450: `0.9798`
- Average pair pixel absolute difference: `7.6246`
- Average Laplacian variance baseline / tuned: `890.9734 / 949.7724`
- Average high-frequency energy baseline / tuned: `3576.7521 / 3617.4978`
- Average prompt CLIP baseline / tuned: `0.3175 / 0.3145`
- Average style CLIP baseline / tuned: `0.2825 / 0.2823`
- `reference_path_count`: `24`

Analysis:
- E7 is clearly stronger than E5 and no longer reads as a pure near-copy on every card.
- Composition remains mostly stable because there is no reference branch.
- The gain is still not strong enough: most visible changes are local contrast / edge density / material shifts rather than decisive new brushstroke structure.
- This supports the working hypothesis that the route was previously too conservative, but that a single decoder-local adapter is still not enough to hit the target visual bar.

## E12 Completed Run (Spatially Gated High-Frequency Brush Head)

Method:
- E12 introduces `SpatialHFBrushHead`, a new decoder-feature-map module that separates the texture edit into two factors: a learned spatial gate and a high-pass-only residual branch.
- The head is intended to solve the E11 problem directly: permit stronger local texture movement while suppressing low-frequency, whole-object deformation.
- Training also adds `low_frequency_consistency` against the no-brush base path plus `gate_l1` and `gate_tv` regularization, so the model is explicitly penalized if it edits broad low-frequency structure or opens the gate over the whole frame.
- The concrete implementation keeps the token adapter, reference encoder, and reference injector disabled. Only `vision_gen_dec_aligner_norm`, `vision_gen_dec_aligner`, and the new spatial HF head are trainable. This makes E12 a pure no-reference, image-local intervention.
- Architecturally, the head uses a learned one-channel spatial gate on the `24 x 24` decoder feature map and a separate texture branch whose output is explicitly high-passed as `delta - blur3x3(delta)`. The final edit is `feature_map + gate * delta_hp * residual_scale`, so low-frequency structural movement must survive both the blur subtraction and the gate regularizers.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_spatial_hf_head_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E12_brush_spatial_hf_head_v1_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E12_brush_spatial_hf_head_v1_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E12_brush_spatial_hf_head_v1_cfgcond_refeval/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E12_brush_spatial_hf_head_v1_cfgcond_refeval/review_notes.md`
- Trainable parameters: `5.94M`
- Spatial HF effective initial gate: `0.2695`
- Step-1 total loss: `0.8648`
- Step-1 components: `flow 0.5940`, `laplacian 2.4593`, `sobel 2.0396`, `fft 0.3993`, `low_frequency_consistency 1.5e-06`, `gate_l1 0.1206`, `gate_tv 0.0221`
- Step-150 eval loss: `0.9842`
- Step-300 eval loss: `0.9916`
- Step-450 eval loss: `0.9852`
- Final checkpoint eval loss in `final_summary.json`: `0.9846`
- Final-step training losses at step 450: `flow 0.6800`, `laplacian 2.6871`, `sobel 2.1756`, `fft 0.4307`, `low_frequency_consistency 2.35e-07`, `gate_l1 2.98e-04`, `gate_tv 3.36e-04`

Evaluation summary:
- Average pair pixel absolute difference: `6.6719`
- Minimum / maximum pair pixel absolute difference: `2.0746 / 11.4208`
- Average Laplacian variance baseline / tuned: `887.0809 / 940.4521`
- Average high-frequency energy baseline / tuned: `3574.1596 / 3637.0988`
- Average prompt CLIP baseline / tuned: `0.3171 / 0.3148`
- Average style CLIP baseline / tuned: `0.2821 / 0.2815`
- `reference_path_count`: `24`

Analysis:
- E12 is stronger than E10 numerically and visually, but clearly weaker than E11 and much weaker than E8 `0.75`. The gate/high-pass design did not collapse to a no-op, yet it also did not recover the desired target-strength brushstroke regime.
- Manual review across portrait, baroque, impressionist, expressionist, and ink-wash cards shows a mixed but interpretable pattern:
  - `classical_portrait`: safer than E11; facial identity and head silhouette remain largely intact, with only mild extra surface texture.
  - `baroque_drama`: still exhibits unacceptable highlight/face reshaping and head instability. The failure is reduced relative to the worst E11 cards, but it is not eliminated.
  - `impressionist_garden` and `expressionist_interior`: texture density and paint breakup are modestly richer than baseline and slightly more convincing than E10, but still not at the "obvious enhancement" level defined by E8 `0.75`.
  - `ink_wash_mountain`: local dark-light handling is slightly richer and remains compositionally stable.
- The training dynamics explain part of this behavior. `gate_l1` and `gate_tv` shrink rapidly over training, which means the gate becomes extremely sparse and conservative. That likely helps keep many prompts safer than E11, but it also limits the total visible effect size.
- Therefore E12 is best read as a partial validation of the gated-HF idea: explicit spatial/frequency control does improve the safety-strength tradeoff relative to raw feature-map scaling, but feature-map-stage editing alone still does not reliably protect baroque portraits, and the current gate regularization may be overly suppressive for target-level brushstroke enhancement.

## E14 Completed Run (Output-Space Residual Head)

Method:
- E14 introduces `OutputVelocityTextureHead`, which moves the texture edit later in the pipeline and applies it directly on the predicted latent velocity rather than earlier decoder features.
- This design is the fallback for the case where any feature-map intervention still entangles too much semantic structure. The bet is that later injection should reduce the chance of modifying pose/head geometry.
- The run keeps the same no-reference policy as E12: token adapter, reference encoder, reference injector, and feature-map adapter are all disabled. Only `vision_gen_dec_aligner_norm`, `vision_gen_dec_aligner`, and the late output residual head are trainable.
- Concretely, E14 predicts a 4-channel velocity residual, removes its low-frequency component through a blur-based high-pass path, and then gates the residual before adding it back to `pred_velocity`. This is meant to operate as a narrow texture correction rather than a semantic feature rewrite.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_output_residual_head_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E14_brush_output_residual_head_v1_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E14_brush_output_residual_head_v1_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E14_brush_output_residual_head_v1_cfgcond_refeval/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E14_brush_output_residual_head_v1_cfgcond_refeval/review_notes.md`
- Trainable parameters: `1.58M`
- Output residual scale: `0.35`
- Step-1 total loss: `0.9579`
- Step-1 components: `flow 0.6682`, `laplacian 2.6344`, `sobel 2.1703`, `fft 0.4306`, `low_frequency_consistency 1.1e-06`, `gate_l1 0.1079`, `gate_tv 0.0359`
- Step-150 eval loss: `0.9850`
- Step-300 eval loss: `0.9923`
- Step-450 eval loss: `0.9859`
- Final-step training losses at step 450: `flow 0.6976`, `laplacian 2.7140`, `sobel 2.2284`, `fft 0.4348`, `low_frequency_consistency 8.15e-07`, `gate_l1 0.0908`, `gate_tv 0.0300`

Evaluation summary:
- Average pair pixel absolute difference: `6.5432`
- Minimum / maximum pair pixel absolute difference: `2.0265 / 11.2614`
- Average Laplacian variance baseline / tuned: `887.6416 / 971.5827`
- Average high-frequency energy baseline / tuned: `3576.2996 / 3685.0406`
- Average prompt CLIP baseline / tuned: `0.3168 / 0.3158`
- Average style CLIP baseline / tuned: `0.2824 / 0.2824`
- `reference_path_count`: `24`

Analysis:
- E14 does not improve the central failure case. `baroque_drama` still shows pronounced head and facial distortion, including obvious highlight reshaping and a visibly deformed forehead/face mass.
- On stable prompts such as `classical_portrait`, `impressionist_garden`, and `ink_wash_mountain`, the run is mostly conservative. It adds some extra edge energy or local texture density, but the result still reads more like a mild repaint than a decisive brushstroke enhancement.
- `expressionist_interior` remains broadly stable, but the visual gain is still modest and does not justify the remaining baroque failure.
- Relative to E12, E14 is not clearly stronger by pair difference and is only modestly higher in Laplacian / HF metrics. By eye, that extra HF energy does not translate into clearly better painterly handling; it behaves closer to "sharper local residue" than "better brushstroke structure."
- One useful diagnostic signal is that `gate_l1` and `gate_tv` remain much larger through late training than in E12. In other words, E14 does not fail because the gate collapses shut. It fails even while the late output head remains active, which implies that output-space injection alone is still not sufficiently structure-aware.
- Therefore E14 is a negative architectural result: moving the edit later is not enough. The next architecture should introduce an explicit structural anchor or masking policy, not just another locality-preserving residual family.

## E15 Completed Run (Structure-Anchored Output Head)

Method:
- E15 keeps the output-space edit position from E14 but augments it with an explicit structure-protection mechanism.
- The new head estimates low-frequency structure energy from the incoming base velocity, converts that into a soft protection mask, and attenuates the learned high-frequency residual in protected regions.
- Relative to E14, this is the first run that tries to protect identity-bearing structure directly rather than only relying on locality from insertion point or gate sparsity.
- Training also adds `protected_gate_l1`, which penalizes gate usage inside protected regions so the model cannot ignore the structure mask without paying an explicit loss.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_output_structure_anchor_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E15_brush_output_structure_anchor_v1_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E15_brush_output_structure_anchor_v1_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E15_brush_output_structure_anchor_v1_cfgcond_refeval/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E15_brush_output_structure_anchor_v1_cfgcond_refeval/review_notes.md`
- Trainable parameters: `1.58M`
- Anchored output residual scale: `0.35`
- Step-1 total loss: `0.9017`
- Step-1 components: `flow 0.6211`, `laplacian 2.5388`, `sobel 2.0759`, `fft 0.4103`, `low_frequency_consistency 9.6e-08`, `gate_l1 0.1045`, `gate_tv 0.0332`, `protected_gate_l1 0.1030`
- Step-150 eval loss: `0.9865`
- Step-300 eval loss: `0.9908`
- Step-450 eval loss: `0.9873`
- Final checkpoint eval loss in `final_summary.json`: `0.9867`
- Final-step training losses at step 450: `flow 0.6345`, `laplacian 2.6274`, `sobel 2.1093`, `fft 0.4180`, `low_frequency_consistency 5.63e-08`, `gate_l1 0.0752`, `gate_tv 0.0248`, `protected_gate_l1 0.0742`

Evaluation summary:
- Average pair pixel absolute difference: `6.8839`
- Minimum / maximum pair pixel absolute difference: `2.1812 / 11.8697`
- Average Laplacian variance baseline / tuned: `890.9616 / 964.4624`
- Average high-frequency energy baseline / tuned: `3578.8147 / 3672.8381`
- Average prompt CLIP baseline / tuned: `0.3174 / 0.3163`
- Average style CLIP baseline / tuned: `0.2829 / 0.2817`
- `reference_path_count`: `24`

Analysis:
- E15 is more informative than it is successful. The structure anchor stays active throughout training, and `protected_gate_l1` remains nontrivial, so the run really is using its protection mechanism.
- Compared with E14, the run is slightly more active by pair difference and still keeps portrait / impressionist / expressionist / ink prompts broadly stable.
- However, the core acceptance failure remains. `baroque_drama` still shows visible forehead/head deformation and is not safe enough by eye.
- This means that a protection mask derived only from self low-frequency velocity energy is not a strong enough prior for identity preservation. The model still finds ways to reshape the main subject while satisfying the current anchor.
- The next iteration should therefore add a stronger spatial prior on top of the structure anchor, for example a center-biased portrait protection map or another prompt-agnostic subject prior, before trying more strength scaling.

## E16 Completed Run (Structure Anchor + Center Prior)

Method:
- E16 is the direct follow-up to E15. It keeps the same anchored output head, the same self-derived low-frequency structure mask, and the same `protected_gate_l1` objective.
- The only intended change is the addition of an upper-center Gaussian protection prior before residual attenuation.
- This design is deliberately narrow. It asks whether the remaining baroque failure is mostly due to insufficient subject-region bias rather than insufficient structure masking in the abstract.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_output_structure_anchor_center_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E16_brush_output_structure_anchor_center_v1_cfgcond_refeval`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E16_brush_output_structure_anchor_center_v1_cfgcond_refeval/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E16_brush_output_structure_anchor_center_v1_cfgcond_refeval/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E16_brush_output_structure_anchor_center_v1_cfgcond_refeval/review_notes.md`
- Trainable parameters: `1.58M`
- Anchored output residual scale: `0.35`
- Step-150 eval loss: `0.9865`
- Step-300 eval loss: `0.9908`
- Step-450 eval loss: `0.9873`
- Final-step training losses at step 450: `flow 0.6345`, `laplacian 2.6275`, `sobel 2.1093`, `fft 0.4181`, `low_frequency_consistency 5.77e-08`, `gate_l1 0.0752`, `gate_tv 0.0248`, `protected_gate_l1 0.0741`

Evaluation summary:
- Average pair pixel absolute difference: `6.8442`
- Minimum / maximum pair pixel absolute difference: `2.1300 / 11.9991`
- Average Laplacian variance baseline / tuned: `891.5082 / 968.0089`
- Average high-frequency energy baseline / tuned: `3579.4624 / 3675.5516`
- Average prompt CLIP baseline / tuned: `0.3174 / 0.3166`
- Average style CLIP baseline / tuned: `0.2825 / 0.2821`
- `reference_path_count`: `24`

Analysis:
- E16 does not materially improve on E15. The upper-center protection prior changes the control mechanism, but the resulting images remain in the same visual regime.
- `classical_portrait`, `impressionist_garden`, `expressionist_interior`, and `ink_wash_mountain` stay broadly stable and conservative, which means the added prior does not destabilize the branch.
- However, `baroque_drama` still shows clear head and forehead deformation. The prior is therefore not strong or specific enough to protect the identity-bearing region that actually matters.
- Relative to E15, the metrics are effectively lateral: pair difference is slightly lower, HF energy slightly higher, and prompt/style CLIP stay close. By eye, these changes do not amount to a new quality regime.
- E16 should therefore be treated as closure on the current anchored-output sub-family. The next step should not be another small variant of this same head; it should use a different mask source or a different local brush architecture.

## Diagnosis: LoRA Restart vs Brush Architecture

Do not restart from LoRA as a blind replacement yet, but do use LoRA as the next control/escape route.

Current evidence:
- The no-reference brush path is too conservative. It preserves composition, but it does not visibly change brushstroke/material details.
- The reference brush path has enough leverage to change images, but its leverage is not well localized. It changes structure as well as texture.
- A previous larger attention-only LoRA quality probe showed that LoRA can make visible artistic changes, but full-strength inference may rewrite composition and weaken prompt adherence.

Interpretation:
- The current brush architecture/config is probably underpowered when reference conditioning is disabled.
- Reference cross-attention is probably too semantically entangled when enabled.
- A pure "start over with LoRA" move may recover visible style quickly, but it risks returning to global style/composition drift unless strength is controlled and evaluation includes fixed-seed baseline comparisons.

Recommended direction:
- Pause E6 for now. It still uses reference cross-attention, only with high-pass input and two experts, so it is unlikely to resolve the core problem by itself.
- E7 confirmed that stronger no-reference local capacity can move the image without major composition drift, but it still falls short of the target brushstroke effect size.
- E8 confirmed that clearly visible brushstroke/material change is achievable, and that LoRA `0.75` is a useful target-strength reference even though it is not structurally safe.
- E9 confirmed that the v2 local-capacity direction is better than a single adapter, but it still falls short of the target and still leaks some structure on baroque/portrait-like prompts.
- E10 confirmed that removing the token adapter improves safety, especially on portrait/baroque, but the first feature-map-only setting is too weak.
- E11 confirmed that stronger feature-map-only capacity can recover visible effect size, but without better locality control it again leaks into structure.
- E12 confirmed that explicit spatial/frequency gating improves the tradeoff but does not yet solve the baroque failure mode and still undershoots the target visible effect size.
- E14 confirmed that later output-space editing does not remove the baroque failure mode either, so simple placement changes are no longer the main question.
- E15 confirmed that a self-derived structure anchor is directionally reasonable but still not strong enough. The next step should therefore be a more explicitly structure-anchored local architecture with a stronger spatial prior, for example a center-biased portrait protection mask, a prompt-aware identity anchor, or a branch that predicts only texture residual statistics rather than free residual fields.
- E16 confirmed that even adding a simple center-biased subject prior does not materially improve the anchored-output family. The next step should therefore use a different mask source or move away from output-head-only tuning entirely.
- D4 confirmed that the rebuilt D1 -> D2 data curriculum can support a moderate increase in local texture pressure without losing portrait safety.
- D5 confirmed that further same-family gate relaxation is close to a plateau; the current 512-channel spatial-HF head does not unlock a new effect tier just by opening the gate more.
- D6 confirmed that portrait-structure recovery is worth keeping in the data curriculum because it can slightly improve the D5 tradeoff, but it does not remove the deeper strength ceiling of the present head family.
- S5 confirmed that stronger V2 supervision is no longer blocked mainly by conservative optimization. The branch can now
  move well beyond S4 while keeping portrait/baroque largely safe, but the remaining failure mode has shifted toward
  sparse-region hallucination and texture-like patch composition rather than simple under-training.
- S6 confirmed that adding a soft blank-region suppression loss is not enough to solve that sparse-region hallucination.
  The branch becomes stronger again while staying portrait/baroque-safe, but `ink_wash` still exposes invented sparse
  upper-page structure. That means the next gain must come from more explicit occupancy / primitive rules rather than a
  slightly stronger regularizer on the same semi-implicit renderer.
- Add zoom-crop evaluation cards before judging the next architecture iteration, but treat full-card near-baseline results as failure unless crops show clear material gains without layout drift.

## Suggested Next Step

Next smallest useful step: treat T2 as the stable VNext visual baseline, S7 as the authoritative old-family V2 baseline,
P1/CS1 as the dense-primitive branch to beat, and `A1@checkpoint-225` as the current slot-family diagnostic point;
then move to a refined slot-based anchor-set renderer rather than another per-pixel occupancy refinement inside the current primitive family.

Recommended next action:

1. Keep token adapter disabled and do not return to reference cross-attention.
2. Treat baroque / portrait identity stability and `ink_wash` sparse-region discipline as the two non-negotiable regression gates.
3. Use E8 `0.75` as the target effect-strength reference, D6 as the rebuilt-data spatial-HF safety baseline, T2 as the stable
   VNext statistics baseline, S7 as the authoritative old-family V2 baseline, and P1/CS1 as the sparse-primitive branch to beat.
4. Retire the old `S5 -> S8` occupancy-gated stroke-field family. It has now been tested with soft blank suppression, evidence
   gating, support ceilings, and hard occupancy, and it should not receive another same-family sweep.
5. Do not open `CS2` on top of the current per-pixel connected-support primitive branch. CS1 already showed that connected-support
   gating alone does not improve the sparse-region failure mode enough to justify another full same-family continuation.
6. Do not resume `A1_slot_based_anchor_set_renderer_v1` unchanged. The `checkpoint-225` diagnostic run suggests the branch is directionally
   useful, but the current slot/prototype formulation saturates too early into near-uniform routing.
7. Do not prioritize E13 or another D4/D5-style pressure sweep. Current evidence says the open question is no longer generic
   strength pressure; it is whether dense occupancy itself is the wrong abstraction.

Since E3D/E4/E5, D4/D5/D6, S5-S8, and now P1 still do not fully meet the visible brushstroke bar, the next implementation
should not simply increase learning rate or steps. Prefer one of these targeted changes:

- keep the rebuilt-data curriculum and current fixed-baseline audit path, but stop iterating on dense occupancy primitives;
- keep the slot-based anchor-set prediction problem, but refine it so slot ownership and prototype use become less uniform
  and less eager to saturate early;
- keep a subject-safe exclusion rule in that next renderer, but make it act on anchor slots rather than on another dense
  occupancy field;
- `A2_slot_based_anchor_set_renderer_v2` has already been tried as that first refinement and failed quickly into the same
  maximum-entropy slot/prototype regime; the next branch should therefore change the slot ownership mechanism itself, not
  just repeat another threshold / loss sweep;
- only revisit wider local texture heads if the next slot-based renderer still fails, because P1 and CS1 already show that generic
  strength increases are no longer the main bottleneck;
- keep reviewer-facing zoom crops in the evaluation contract, because full-card views can still hide local brushstroke
  differences even when the global compare cards remain helpful.

## Known Risks

- E1/E2-style adapter effects are too subtle for the project goal.
- E3/E3B reference conditioning can rewrite content.
- Automatic metrics can show movement even when manual visual quality is not improved.
- The E3D `train_log.jsonl` contains duplicate entries for steps 151-171 because the interrupted unsaved tail had already logged those steps before the clean resume from `checkpoint-150`; checkpoint summaries and evaluation outputs are authoritative.
- Existing git status includes many untracked JanusFlow-Art files; do not assume a clean worktree.
- Evaluation bundles created before the 2026-04-11 baseline-head fix are not trustworthy as true-base comparisons for
  configs that enable `spatial_hf_head`, `output_velocity_head`, or `anchored_output_velocity_head`. In those runs,
  the baseline branch accidentally kept the newly added head enabled with random initialization.

## D1 Completed Run (Full-Balanced Stage-1 Base)

Method:
- D1 is the first full experiment that uses the rebuilt official-data pipeline rather than the older `train_texture_rich_v1` subset.
- The purpose of D1 is not maximum brushstroke intensity. It is a curriculum stage whose job is to establish a safer no-reference base model before any texture-heavy continuation.
- The model architecture remains the `SpatialHFBrushHead` branch introduced in E12: token adapter disabled, reference encoder disabled, condition injector disabled, feature-map adapter disabled, and no output-space residual head.
- Only `vision_gen_dec_aligner_norm`, `vision_gen_dec_aligner`, and the spatial HF head are trainable. The local edit remains explicitly frequency-limited and gate-regularized through the same `low_frequency_consistency`, `gate_l1`, and `gate_tv` losses.
- The actual methodological change is therefore in the data: D1 trains on `train_full_balanced_v1.jsonl`, a rebuilt official split that favors style balance and prompt/content safety rather than texture extremity.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_spatial_hf_head_full_balanced_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D1_brush_spatial_hf_head_full_balanced_v1`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D1_brush_spatial_hf_head_full_balanced_v1/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D1_brush_spatial_hf_head_full_balanced_v1/evaluation_full_fixed_baseline/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D1_brush_spatial_hf_head_full_balanced_v1/review_notes.md`
- Trainable parameters: `5.94M`
- Spatial HF effective gate: `0.2695`
- Step-150 eval loss: `0.9820`
- Step-300 eval loss: `0.9933`
- Step-450 eval loss: `0.9723`
- Final summary eval loss: `0.9841`
- Baseline correction note: the original D1 evaluation bundle accidentally left `spatial_hf_head` enabled in the
  baseline config; the fixed-baseline rerun above is authoritative.

Evaluation summary:
- Average pair pixel absolute difference: `4.5028`
- Minimum / maximum pair pixel absolute difference: `0.9970 / 7.6406`
- Average Laplacian variance baseline / tuned: `897.3197 / 912.4242`
- Average high-frequency energy baseline / tuned: `3577.7778 / 3565.6639`
- Average prompt CLIP baseline / tuned: `0.3449 / 0.3448`
- Average style CLIP baseline / tuned: `0.2808 / 0.2801`
- `reference_path_count`: `24`

Analysis:
- D1 behaves like the intended stage-1 stabilization model. It is more conservative than the earlier texture-first runs, but that conservatism now appears useful rather than merely underpowered.
- Manual review of the fixed prompt cards indicates that `classical_portrait` and `baroque_drama` are structurally steadier than the earlier E11/E14/E15/E16 family. The branch no longer shows the characteristic severe head collapse seen in the old unstable texture routes.
- At the same time, D1 is clearly below the visual bar for brushstroke enhancement. `impressionist_garden`, `expressionist_interior`, and `ink_wash_mountain` remain close to baseline and mostly read as slight surface perturbations rather than decisive material changes.
- The correct interpretation is therefore not that D1 solves the brushstroke problem. Rather, D1 validates the rebuilt full-balanced data stage as a safer initialization point for later continuation.

## D2 Completed Run (Texture-Balanced Continuation from D1)

Method:
- D2 is the second stage of the rebuilt-data curriculum. It resumes from the D1 checkpoint and switches the training set from `train_full_balanced_v1.jsonl` to `train_texture_balanced_v2.jsonl`.
- Crucially, this is a data-stage continuation rather than a new architecture branch. The trainable modules, no-reference policy, spatial HF head, and loss family are unchanged relative to D1.
- What changes is the distribution of training pressure: texture-rich samples are upweighted in a style-balanced way, while head-sensitive samples remain present and are routed through `prompt_v2_portrait_safe` rather than the stronger `texture_push` prompt template.
- This experiment directly tests the central data hypothesis that emerged from the E8/E11/E12 failures: brushstroke strength should be increased through a staged data curriculum before introducing another more aggressive local editor.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_spatial_hf_head_texture_balanced_v2.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D2_brush_spatial_hf_head_texture_balanced_v2`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D2_brush_spatial_hf_head_texture_balanced_v2/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D2_brush_spatial_hf_head_texture_balanced_v2/evaluation_full_fixed_baseline/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D2_brush_spatial_hf_head_texture_balanced_v2/review_notes.md`
- Trainable parameters: `5.94M`
- Spatial HF effective gate: `0.2695`
- Step-150 eval loss: `0.9818`
- Step-300 eval loss: `0.9932`
- Step-450 eval loss: `0.9721`
- Baseline correction note: the original D2 evaluation bundle accidentally left `spatial_hf_head` enabled in the
  baseline config; the fixed-baseline rerun above is authoritative.

Evaluation summary:
- Average pair pixel absolute difference: `5.1358`
- Minimum / maximum pair pixel absolute difference: `1.0206 / 8.7521`
- Average Laplacian variance baseline / tuned: `897.3197 / 956.3055`
- Average high-frequency energy baseline / tuned: `3577.7778 / 3630.1808`
- Average prompt CLIP baseline / tuned: `0.3449 / 0.3442`
- Average style CLIP baseline / tuned: `0.2808 / 0.2807`
- `reference_path_count`: `24`

Analysis:
- D2 is the first rebuilt-data result that improves visible detail without reintroducing the earlier catastrophic portrait/baroque failure. Compared with D1, it raises pair difference, Laplacian variance, and high-frequency energy while keeping prompt similarity effectively unchanged.
- The manual review pattern is informative:
  - `classical_portrait` becomes slightly richer in canvas-grain and local texture while preserving pose and face structure.
  - `baroque_drama` remains close to baseline across the reviewed seeds and does not return to the old head-deformation regime.
  - `impressionist_garden` and `ink_wash_mountain` show the clearest benefit, with visibly denser paint breakup and dark-edge texture than D1.
  - `expressionist_interior` and `ukiyoe_evening` remain conservative and do not yet show a decisive brushstroke jump.
- Therefore the rebuilt data curriculum appears directionally correct: D1 gives a safer base, and D2 adds real texture pressure without immediately breaking head-sensitive prompts.
- The limitation is that D2 is still not close to the E8 `0.75` target-strength reference. It improves the tradeoff, but it does not yet enter a clearly stronger visual regime.
- The most plausible next move is not D3 portrait recovery, because D2 does not currently show strong portrait/baroque collapse. The more useful next experiment is a stronger local-capacity continuation on top of the D1/D2 data curriculum, using this safer data path rather than the old texture-only subset.

## Baseline Correction Sweep (E12 / E14 / E15 / E16)

Method:
- After the D1/D2 audit, the evaluator baseline builder was inspected and found to disable only the older brush modules.
- The newer local heads introduced later in the project, namely `spatial_hf_head`, `output_velocity_head`, and `anchored_output_velocity_head`, were not being disabled in the baseline branch.
- As a result, baseline images for E12/E14/E15/E16 were not true base-model outputs. They were sampled with the corresponding new head still instantiated at random initialization.
- After fixing `build_base_evaluation_config()` in `eval_janusflow_art.py`, all four affected runs were re-evaluated into new `evaluation_fixed_baseline/` directories.

Results:
- The corrected baseline hash for `00042_001.png` is now identical across E12, E14, E15, and E16: `57dadb67930c`, matching the earlier trustworthy baseline family used by E1/E3D/E4/E5/E7/E9.
- Corrected evaluation directories:
  - `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E12_brush_spatial_hf_head_v1_cfgcond_refeval/evaluation_fixed_baseline/evaluation`
  - `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E14_brush_output_residual_head_v1_cfgcond_refeval/evaluation_fixed_baseline/evaluation`
  - `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E15_brush_output_structure_anchor_v1_cfgcond_refeval/evaluation_fixed_baseline/evaluation`
  - `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E16_brush_output_structure_anchor_center_v1_cfgcond_refeval/evaluation_fixed_baseline/evaluation`

Corrected summaries:
- E12:
  - pair diff `6.6324`
  - Laplacian baseline / tuned `890.9734 / 940.4521`
  - HF baseline / tuned `3576.7521 / 3637.0988`
  - prompt CLIP baseline / tuned `0.3175 / 0.3148`
- E14:
  - pair diff `6.4905`
  - Laplacian baseline / tuned `890.9734 / 971.5827`
  - HF baseline / tuned `3576.7521 / 3685.0406`
  - prompt CLIP baseline / tuned `0.3175 / 0.3158`
- E15:
  - pair diff `6.8194`
  - Laplacian baseline / tuned `890.9734 / 964.4624`
  - HF baseline / tuned `3576.7521 / 3672.8381`
  - prompt CLIP baseline / tuned `0.3175 / 0.3163`
- E16:
  - pair diff `6.8490`
  - Laplacian baseline / tuned `890.9734 / 968.0089`
  - HF baseline / tuned `3576.7521 / 3675.5516`
  - prompt CLIP baseline / tuned `0.3175 / 0.3166`

Analysis:
- The correction changes the baseline numbers and makes cross-run baseline statistics internally consistent again.
- Importantly, it does not overturn the qualitative ordering that mattered for research decisions. E12 remains the safer-but-too-weak gated feature-map route; E14 remains an unsuccessful output-space fallback; E15 and E16 remain lateral anchored-output variants that do not solve the baroque failure.
- In other words, the evaluator bug polluted the baseline branch, but it does not rescue the later output-head family. The main architecture conclusions from E12/E14/E15/E16 still stand after correction.

## D4 Completed Run (Texture-Balanced Push Continuation from D2)

Method:
- D4 continues directly from the corrected D2 texture-balanced checkpoint and keeps the same rebuilt-data curriculum, no-reference policy, and `SpatialHFBrushHead` topology.
- This is a same-family pressure test rather than a new branch. The purpose is to ask whether the safer D1 -> D2 curriculum can support a modest increase in local texture pressure without reintroducing the older portrait/baroque failure mode.
- The architectural write-path is unchanged: token adapter off, reference encoder off, condition injector off, feature-map adapter off, and no output-space residual head.
- Relative to D2, the experiment increases continuation pressure in three controlled ways: longer schedule (`450 -> 900` total steps), larger `spatial_hf_head.residual_scale` (`1.5 -> 1.8`), and slightly stronger texture losses (`laplacian 0.08 -> 0.09`, `fft_high_frequency 0.03 -> 0.04`).
- Operational note: this run also surfaced an important resume detail. `final_checkpoint/checkpoint_summary.json` stores `step: null`, while numbered checkpoints preserve the actual step. Therefore all continuation runs should resume from `checkpoint-<step>` rather than `final_checkpoint` when exact global-step restoration matters.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_spatial_hf_head_texture_balanced_push_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D4_brush_spatial_hf_head_texture_balanced_push_v1`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D4_brush_spatial_hf_head_texture_balanced_push_v1/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D4_brush_spatial_hf_head_texture_balanced_push_v1/evaluation_fixed_baseline/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D4_brush_spatial_hf_head_texture_balanced_push_v1/review_notes.md`
- Trainable parameters: `5.94M`
- Spatial HF effective gate: `0.3234`
- Spatial HF residual scale: `1.8`
- Step-600 eval loss: `1.0127`
- Step-750 eval loss: `1.0246`
- Step-900 eval loss: `1.0029`
- Final summary eval loss: `1.0151`

Evaluation summary:
- Average pair pixel absolute difference: `5.1990`
- Minimum / maximum pair pixel absolute difference: `1.0518 / 8.9454`
- Average Laplacian variance baseline / tuned: `897.3197 / 966.1586`
- Average high-frequency energy baseline / tuned: `3577.7778 / 3639.0180`
- Average prompt CLIP baseline / tuned: `0.3449 / 0.3447`
- Average style CLIP baseline / tuned: `0.2808 / 0.2804`
- `reference_path_count`: `24`

Analysis:
- D4 is a real but deliberately small step beyond D2. It raises pair difference, Laplacian variance, and high-frequency energy while essentially preserving prompt similarity.
- Manual review suggests that this extra movement is mostly well-behaved:
  - `classical_portrait` shows slightly richer sleeve/window texture while face structure remains intact.
  - `baroque_drama` stays in the safe regime and does not revert to the earlier head-deformation failure.
  - `impressionist_garden` and `ink_wash_mountain` become modestly denser in local breakup and edge texture than D2.
  - `expressionist_interior` and `ukiyoe_evening` remain comparatively conservative.
- The key conclusion is not that D4 reaches the target effect size. It does not. Rather, it shows that the rebuilt-data curriculum can tolerate a moderate local-strength increase without immediately collapsing portrait safety.
- This made a stronger same-family continuation worth trying before changing architecture again.

## D5 Completed Run (Texture-Balanced Gate-Relax Continuation from D4)

Method:
- D5 resumes from D4 and deliberately tests whether the same 512-channel spatial-HF family is still under-regularized or already approaching saturation.
- The dataset remains `train_texture_balanced_v2.jsonl`, so this is not a data change. The goal is to isolate the effect of stronger local editing pressure inside the same continuation regime.
- Relative to D4, the run increases `spatial_hf_head.residual_scale` again (`1.8 -> 2.1`), pushes texture losses further (`laplacian 0.09 -> 0.10`, `fft_high_frequency 0.04 -> 0.05`), and relaxes gate regularization (`gate_l1 0.005 -> 0.003`, `gate_tv 0.01 -> 0.006`).
- The working hypothesis was that the D4 branch might still be artificially limited by sparse gating and conservative texture penalties, and that a looser gate could unlock more visible brushwork.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_spatial_hf_head_texture_balanced_gate_relax_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D5_brush_spatial_hf_head_texture_balanced_gate_relax_v1`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D5_brush_spatial_hf_head_texture_balanced_gate_relax_v1/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D5_brush_spatial_hf_head_texture_balanced_gate_relax_v1/evaluation_fixed_baseline/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D5_brush_spatial_hf_head_texture_balanced_gate_relax_v1/review_notes.md`
- Trainable parameters: `5.94M`
- Spatial HF effective gate: `0.3773`
- Spatial HF residual scale: `2.1`
- Step-1050 eval loss: `1.0437`
- Step-1200 eval loss: `1.0560`
- Step-1350 eval loss: `1.0338`
- Final summary eval loss: `1.0463`

Evaluation summary:
- Average pair pixel absolute difference: `5.2357`
- Minimum / maximum pair pixel absolute difference: `1.0141 / 8.7958`
- Average Laplacian variance baseline / tuned: `897.3197 / 958.8049`
- Average high-frequency energy baseline / tuned: `3577.7778 / 3633.4737`
- Average prompt CLIP baseline / tuned: `0.3449 / 0.3440`
- Average style CLIP baseline / tuned: `0.2808 / 0.2803`
- `reference_path_count`: `24`

Analysis:
- D5 does move slightly farther than D4 in pair-difference terms, but it does not improve the texture proxies in a clean monotonic way. Both tuned Laplacian variance and tuned high-frequency energy slip slightly relative to D4, while prompt CLIP also drops a little.
- Visual inspection supports the same reading:
  - `baroque_drama` remains broadly safe, which is good, but it does not gain a new level of visible painterly detail.
  - `classical_portrait`, `impressionist_garden`, and `ink_wash_mountain` remain close to the D4 look, with only subtle surface changes.
  - `expressionist_interior` and `ukiyoe_evening` still do not break into a meaningfully stronger visual regime.
- This means D5 is not a failure in the old catastrophic sense, but it is close to a plateau signal. Relaxing the gate and increasing high-frequency pressure inside the same 512-channel spatial-HF family does not buy a proportional increase in visible brushstroke quality.
- Therefore the project should stop assuming that “more of the same D4 pressure” will produce a qualitatively different outcome. The next useful step is either a different data stage or a different local-capacity allocation, not another small push on the same texture-balanced continuation alone.

## D6 Completed Run (Portrait-Structure Recovery After D5)

Method:
- D6 resumes from D5 but switches the data stage to `train_portrait_structure_v1.jsonl`, the rebuilt recovery split designed to overrepresent head-sensitive and portrait-dominant samples.
- The purpose is not to reduce the branch back to D1-level conservatism. Instead, D6 tests whether a short portrait-focused continuation can keep most of the D5 local-detail gain while improving the structure-safety tradeoff on the prompts that historically failed first.
- The architecture remains unchanged from D5 so that the effect can be attributed to the recovery curriculum rather than to a new brush mechanism.
- Relative to D5, D6 keeps the stronger `residual_scale=2.1` but restores more conservative regularization (`gate_l1=0.005`, `gate_tv=0.01`) and slightly increases `low_frequency_consistency` (`0.15 -> 0.20`) while reducing the auxiliary texture pressure back toward D2-style values (`laplacian=0.08`, `fft_high_frequency=0.03`).
- In other words, D6 is a short structure-aware cleanup stage run after the strongest safe texture continuation reached by D5.

Results:
- Config: `configs/janusflow_art/brushstroke/brush_spatial_hf_head_portrait_structure_recovery_after_gate_relax_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D6_brush_spatial_hf_head_portrait_structure_recovery_after_gate_relax_v1`
- Final checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D6_brush_spatial_hf_head_portrait_structure_recovery_after_gate_relax_v1/final_checkpoint`
- Evaluation: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D6_brush_spatial_hf_head_portrait_structure_recovery_after_gate_relax_v1/evaluation_fixed_baseline/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/D6_brush_spatial_hf_head_portrait_structure_recovery_after_gate_relax_v1/review_notes.md`
- Trainable parameters: `5.94M`
- Spatial HF effective gate: `0.3773`
- Spatial HF residual scale: `2.1`
- Step-1425 eval loss: `0.9753`
- Step-1500 eval loss: `0.9748`
- Final summary eval loss: `0.9758`

Evaluation summary:
- Average pair pixel absolute difference: `5.2745`
- Minimum / maximum pair pixel absolute difference: `1.0395 / 9.1604`
- Average Laplacian variance baseline / tuned: `897.3197 / 964.2402`
- Average high-frequency energy baseline / tuned: `3577.7778 / 3638.5433`
- Average prompt CLIP baseline / tuned: `0.3449 / 0.3446`
- Average style CLIP baseline / tuned: `0.2808 / 0.2805`
- `reference_path_count`: `24`

Analysis:
- D6 is the most interesting result in the D4-D6 subfamily because it does not simply retreat to a safer-but-weaker point. It slightly improves pair difference over D5, recovers most of the D4 texture-proxy gains, and also restores prompt/style similarity toward the D4 level.
- Manual review suggests that this is a modest cleanup rather than a dramatic visual shift:
  - `classical_portrait` remains stable and keeps the mild fabric/window texture gain.
  - `baroque_drama` stays safely within the non-collapse regime and looks at least as stable as D5.
  - `impressionist_garden` and `ink_wash_mountain` retain the denser local texture visible in D4/D5.
  - `expressionist_interior` and `ukiyoe_evening` are still conservative; D6 does not unlock a new brushstroke-strength regime there.
- The implication is encouraging but bounded. Portrait-structure recovery is not merely a “damage control” stage; it can slightly improve the overall tradeoff after a stronger texture continuation. However, it does not solve the deeper limitation of this 512-channel spatial-HF family.
- The best interpretation of D4-D6 together is therefore:
  - D4 shows that stronger pressure on the rebuilt texture-balanced curriculum is viable.
  - D5 shows that same-family gate relaxation is already near diminishing returns.
  - D6 shows that a short portrait-structure continuation can slightly tidy that stronger branch, but not transform it.
- Consequently, D6 should be treated as the best current checkpoint inside this rebuilt-data spatial-HF line, but not as the final answer. The next real gain will likely require a different local-capacity allocation or a stronger architecture, not another small regularization sweep on the same 512-channel head.

## S7 Completed Run (Evidence-Backed Occupancy Gating)

Method:
- S7 continues the V2 stroke-field branch from the `S6` checkpoint family and tests a stricter renderer-side occupancy rule rather than another generic strength increase.
- The central architectural change is that stroke activation is no longer allowed to depend only on objectness and sparse routing. The pseudo-renderer now computes a local `support_evidence` map from two sources: high-pass carrier magnitude and normalized density evidence. That evidence is passed through a temperature-controlled support gate and multiplied into the objectness blob before rasterization.
- In parallel, the training loss adds `stroke_support_ceiling_loss`, which penalizes objectness predictions that exceed a dilated support target derived from local density/pigment proxy signals. This is intended to make sparse-region hallucination harder by aligning stroke occupancy with observed image support instead of only with learned objectness.
- The experiment keeps the stronger V2 ingredients from S6 that were already useful: `topk_prototypes=1`, longer renderer support (`kernel_size=13`), and explicit blank suppression. Relative to S6, the new ingredients are therefore not a larger-capacity renderer but a more explicit evidence-backed occupancy rule.
- Code-execution audit was completed before running:
  - config audit: `stroke_field_head` on, legacy adapter / reference / output heads off
  - checkpoint audit: initialization from `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S6_stroke_field_sparse_blank_suppression_v1/checkpoint-1200`
  - smoke audit: `--max-steps 1 --skip-final-eval` passed
  - baseline audit: canonical baseline hash matched the official JanusFlow base hash

Results:
- Config: `configs/janusflow_art/brushstroke/vnext/stroke_field_evidence_occupancy_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S7_stroke_field_evidence_occupancy_v1`
- Smoke evaluation output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S7_stroke_field_evidence_occupancy_v1_smoke_eval/evaluation/evaluation`
- Authoritative evaluation output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S7_stroke_field_evidence_occupancy_v1/evaluation_checkpoint1125/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S7_stroke_field_evidence_occupancy_v1/review_notes.md`
- Trainable parameters: `7.51M`
- Step-300 eval loss: `0.9793`
- Step-375 eval loss: `0.9731`
- Step-450 eval loss: `0.9743`
- Step-525 eval loss: `0.9630`
- Step-750 eval loss: `0.9847`
- Step-900 eval loss: `0.9870`
- Step-975 eval loss: `0.9833`
- Step-1050 eval loss: `0.9669`
- Step-1125 eval loss: `0.9811`
- Final train step reached: `1200`

Authoritative evaluation summary (`checkpoint-1125`):
- Checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S7_stroke_field_evidence_occupancy_v1/checkpoint-1125`
- Average pair pixel absolute difference: `8.1485`
- Minimum / maximum pair pixel absolute difference: `2.2093 / 12.9904`
- Average Laplacian variance baseline / tuned: `897.3197 / 1166.6109`
- Average high-frequency energy baseline / tuned: `3577.7778 / 3913.3963`
- Average prompt CLIP baseline / tuned: `0.3449 / 0.3443`
- Average style CLIP baseline / tuned: `0.2808 / 0.2786`
- `reference_path_count`: `24`
- Canonical baseline hash: `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2` (`match=true`)

Operational checkpoint audit:
- `final_checkpoint/art_state.pt` was stale and not the true end state for this run. Its timestamp (`2026-04-12 21:13`) predates the later numbered checkpoints, and evaluating it reproduced the earlier smoke-eval outputs exactly.
- `checkpoint-1125/art_state.pt` is newer and differs from both `checkpoint-75` and the stale `final_checkpoint` alias. Therefore `checkpoint-1125` is the authoritative checkpoint for S7 evaluation and any future continuation unless a new numbered checkpoint is produced.
- This run again confirms that `final_checkpoint` cannot be assumed to be authoritative after long runs when the trainer tail hangs or exits imperfectly. Numbered checkpoints remain the only safe continuation source.

Analysis:
- Numerically, S7 is the strongest V2 result so far. It exceeds S6 on all three movement/detail proxies:
  - pair diff `7.8861 -> 8.1485`
  - tuned Laplacian `1162.76 -> 1166.61`
  - tuned high-frequency energy `3909.09 -> 3913.40`
- The portrait-safety side remains acceptable. `classical_portrait` and `baroque_drama` preserve the head silhouette and facial placement; there is no return to the earlier collapse mode where stronger local editing deforms the dominant face.
- `impressionist_garden` also benefits modestly. The tuned branch looks slightly denser and more locally organized than baseline without obvious layout drift.
- However, the central failure case remains unsolved. In `ink_wash_mountain`, the upper sparse region still acquires a large new tree form. This means the branch still treats some blank space as license to synthesize new stroke objects instead of merely refining existing support. The new evidence-backed gate made activation more selective, but not selective enough.
- `ukiyoe_evening` also remains conservative, with only light surface shifts. So the current gain is not a clean new regime of controlled brushstroke rendering across prompt families; it is a stronger version of the existing V2 behavior.
- Therefore the main conclusion is mixed but clear:
  - S7 validates that explicit support-conditioned occupancy is directionally correct.
  - S7 does **not** validate it as a sufficient fix for sparse-region hallucination.
  - The next renderer change should be more explicit than a soft support gate. Candidate directions now include hard occupancy masking from support evidence, stronger penalties on isolated objectness islands, or primitive rasterization that requires connected support rather than semi-implicit patch synthesis.

## S8 Completed Run (Hard Evidence Occupancy, Final Old-Family Validation)

Method:
- S8 is the planned final formal test inside the old occupancy-gated stroke-field family. It keeps the S7 branch structure
  but hardens the evidence-backed occupancy rule further instead of adding another generic strength sweep.
- The explicit research question is narrow: can a harder support-linked occupancy mechanism solve the `ink_wash` sparse-region
  hallucination without sacrificing the portrait/baroque safety that S7 preserved?
- This run was audited under the VNext execution checklist before and after training:
  - config audit: legacy adapter / reference / output heads disabled, stroke-field head enabled
  - checkpoint audit: authoritative evaluation fixed to `checkpoint-1125`
  - smoke audit: smoke train and smoke eval passed
  - baseline audit: canonical baseline hash matched the official JanusFlow base
- Operational audit note: unlike S7, `final_checkpoint/art_state.pt` for S8 is not stale. It matches the numbered
  `checkpoint-1125` weights by SHA256. However, `final_checkpoint/checkpoint_summary.json` still stores `step: null`, so
  numbered checkpoints remain the authoritative continuation and evaluation source.

Results:
- Config: `configs/janusflow_art/brushstroke/vnext/stroke_field_hard_evidence_occupancy_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S8_stroke_field_hard_evidence_occupancy_v1`
- Authoritative checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S8_stroke_field_hard_evidence_occupancy_v1/checkpoint-1125`
- Authoritative evaluation output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S8_stroke_field_hard_evidence_occupancy_v1/evaluation_checkpoint1125/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S8_stroke_field_hard_evidence_occupancy_v1/review_notes.md`
- Result classification: `Partial`
- Average pair pixel absolute difference: `8.3892`
- Minimum / maximum pair pixel absolute difference: `2.2189 / 13.0937`
- Average Laplacian variance baseline / tuned: `897.3197 / 1170.5930`
- Average high-frequency energy baseline / tuned: `3577.7778 / 3913.3372`
- Average prompt CLIP baseline / tuned: `0.3449 / 0.3442`
- Average style CLIP baseline / tuned: `0.2808 / 0.2785`
- `reference_path_count`: `24`
- Canonical baseline hash: `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2` (`match=true`)
- `final_checkpoint` alias audit:
  - `checkpoint-1125/art_state.pt` SHA256:
    `1ec291ce37dcf1d0a6421103c671302e7c5aecbc7e495ecab022e72fa21e21a8`
  - `final_checkpoint/art_state.pt` SHA256:
    `1ec291ce37dcf1d0a6421103c671302e7c5aecbc7e495ecab022e72fa21e21a8`
  - `final_checkpoint/checkpoint_summary.json` still records `step: null`

Analysis:
- S8 is not a failed run in the generic sense. It remains portrait/baroque-safe, keeps the strongest old-family metrics so
  far, and preserves the fixed-baseline audit contract.
- However, it fails the only question that justified continuing this family: the canonical `ink_wash_mountain` compare card
  still grows a large new upper-page tree form rather than only refining existing marks.
- The correct classification is therefore `Partial`, not `Pass`.
- This result is strong enough to close the family:
  - it proves that harder support-linked occupancy is directionally useful,
  - it disproves the idea that another small same-family gate/loss variant is likely to solve the real sparse-region issue.
- Because S8 is only `Partial`, `S7 checkpoint-1125` remains the authoritative V2 baseline/init source for future branch
  comparisons, and `S8` should not spawn `S9/S10`-style follow-ups.

## P1 Completed Run (Support-Locked Primitive Renderer)

Method:
- P1 opens a new renderer family instead of another occupancy-gated stroke-field variant.
- Relative to S7/S8, it replaces the semi-implicit blob compositor with a sparse support-locked primitive renderer:
  1. predict `occupancy_logits`, `theta`, `length`, `width`, `alpha`, and `prototype_logits`
  2. build a support mask from image-local support evidence
  3. apply `3x3` NMS to occupancy
  4. keep only support-backed top-k anchors (`top_k_anchors=24`)
  5. rasterize one elongated primitive per anchor inside support-allowed regions
- This deliberately removes the older "dense objectness field times carrier field" behavior that was letting sparse regions
  grow semi-implicit patches.
- Per the S8 decision rule, P1 initializes from `S7 checkpoint-1125` rather than S8 because S8 closed as `Partial`.
- The run passed the full execution audit contract:
  - config audit: only `brush_support_locked_primitive_head` enabled
  - checkpoint audit: `init_from_checkpoint` from `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/S7_stroke_field_evidence_occupancy_v1/checkpoint-1125`
  - smoke audit: `--max-steps 1 --skip-final-eval` passed
  - smoke fixed-baseline eval passed
  - canonical baseline hash matched the official JanusFlow base during both smoke and authoritative eval
- Operational audit note: `final_checkpoint/art_state.pt` is current for P1 and matches `checkpoint-1125` by SHA256, but
  `final_checkpoint/checkpoint_summary.json` still stores `step: null`, so the numbered checkpoint remains authoritative.

Results:
- Config: `configs/janusflow_art/brushstroke/vnext/primitive_renderer_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/P1_support_locked_primitive_renderer_v1`
- Smoke evaluation output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/P1_support_locked_primitive_renderer_v1_smoke_eval/evaluation`
- Authoritative checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/P1_support_locked_primitive_renderer_v1/checkpoint-1125`
- Authoritative evaluation output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/P1_support_locked_primitive_renderer_v1/evaluation_checkpoint1125/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/P1_support_locked_primitive_renderer_v1/review_notes.md`
- Trainable parameters: `7.51M`
- Smoke evaluation summary:
  - pair diff `8.1593`
  - prompt CLIP tuned `0.3448`
  - tuned Laplacian `1161.4536`
  - tuned HF energy `3905.8827`
- Authoritative evaluation summary (`checkpoint-1125`):
  - Average pair pixel absolute difference: `8.5591`
  - Minimum / maximum pair pixel absolute difference: `2.2827 / 12.9911`
  - Average Laplacian variance baseline / tuned: `897.3197 / 1174.2936`
  - Average high-frequency energy baseline / tuned: `3577.7778 / 3916.2858`
  - Average prompt CLIP baseline / tuned: `0.3449 / 0.3427`
  - Average style CLIP baseline / tuned: `0.2808 / 0.2775`
  - `reference_path_count`: `24`
  - Canonical baseline hash:
    `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2` (`match=true`)
- `final_checkpoint` alias audit:
  - `checkpoint-1125/art_state.pt` SHA256:
    `cccb0476fbdc89a02c93d21fd353f58518690e6dabede45528f04d0ba5855708`
  - `final_checkpoint/art_state.pt` SHA256:
    `cccb0476fbdc89a02c93d21fd353f58518690e6dabede45528f04d0ba5855708`
  - `final_checkpoint/checkpoint_summary.json` still records `step: null`

Analysis:
- P1 succeeds at one important thing: it proves that sparse support-locked primitive rendering can train stably inside the
  current JanusFlow-Art runtime and can push the V2 line to its strongest movement/detail metrics so far.
- But it does not satisfy the actual research gate:
  - `ink_wash_mountain` still grows a large unsupported upper-page tree structure
  - `classical_portrait` shows visible head/face drift on the canonical seed-42 compare card
  - prompt CLIP drops more than the target success threshold for the branch
- So P1 should not be interpreted as "the fix finally worked." It should be interpreted as a useful negative result:
  sparse anchors and elongated primitives are not enough on their own if the support topology is still too permissive and
  the renderer has no stronger subject-safe exclusion rule.
- The practical implication is that the next branch should keep the primitive-object framing but tighten two things
  together:
  1. connected-support constraints so anchors cannot light up isolated support fragments as if they were full stroke sites
  2. explicit portrait/head exclusion so dominant subject regions are harder to overwrite
- P1 is therefore a good architectural stepping stone, but not the next baseline to standardize on.

## CS1 Completed Run (Connected-Support Primitive Renderer)

Method:
- CS1 is the first direct follow-up to P1 and tests whether hardening support topology inside the same sparse-primitive family can
  solve the remaining sparse-region hallucination without adding a separate subject-exclusion branch yet.
- Relative to P1, CS1 keeps the same primitive-object representation:
  1. predict `occupancy_logits`, `theta`, `length`, `width`, `alpha`, and `prototype_logits`
  2. derive a binary support mask from local support evidence
  3. apply `3x3` NMS to occupancy
  4. rasterize elongated primitives from top-k anchors
- The architectural change is specifically in anchor selection and support topology:
  - `component_aware_enabled=true`
  - support mask is decomposed into 4-neighbor connected components at `24x24`
  - components with area `< 4` are dropped
  - small components allow at most `1` anchor
  - components with area `>= 20` allow at most `2` anchors
  - global `top_k_anchors` is reduced to `16`
  - primitive footprints are clipped to their owning component mask instead of a softer local support patch
- CS1 intentionally keeps `subject_safe_exclusion_enabled=false`. This isolates the question "is connected support alone enough
  to fix `ink_wash`?" If the answer is no, the project should not spend another full run on another dense-occupancy variant.
- The run passed the execution audit contract:
  - config audit: only `brush_support_locked_primitive_head` enabled
  - checkpoint audit: weight-only init from `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/P1_support_locked_primitive_renderer_v1/checkpoint-1125`
  - smoke audit: train smoke and fixed-baseline smoke eval both passed
  - authoritative eval used `checkpoint-1125`, not `final_checkpoint`
  - canonical baseline hash matched the official JanusFlow base

Results:
- Config: `configs/janusflow_art/brushstroke/vnext/connected_support_primitive_renderer_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/CS1_connected_support_primitive_renderer_v1`
- Smoke evaluation output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/CS1_connected_support_primitive_renderer_v1_smoke_eval/evaluation`
- Authoritative checkpoint: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/CS1_connected_support_primitive_renderer_v1/checkpoint-1125`
- Authoritative evaluation output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/CS1_connected_support_primitive_renderer_v1/evaluation_checkpoint1125/evaluation`
- Review notes: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/CS1_connected_support_primitive_renderer_v1/review_notes.md`
- Final train step reached: `1125`
- Step-1125 eval loss: `1.2916`
- Final-save eval loss: `1.2855`
- Authoritative evaluation summary (`checkpoint-1125`):
  - Average pair pixel absolute difference: `8.7457`
  - Minimum / maximum pair pixel absolute difference: `2.4327 / 13.7699`
  - Average Laplacian variance baseline / tuned: `897.3197 / 1159.5249`
  - Average high-frequency energy baseline / tuned: `3577.7778 / 3892.1451`
  - Average prompt CLIP baseline / tuned: `0.3449 / 0.3428`
  - Average style CLIP baseline / tuned: `0.2808 / 0.2773`
  - `reference_path_count`: `24`
  - Canonical baseline hash:
    `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2` (`match=true`)
- Operational checkpoint audit:
  - numbered checkpoints through `checkpoint-1125` exist and are the authoritative continuation/evaluation source
  - `final_checkpoint/checkpoint_summary.json` still stores `step: null`
  - later continuation should therefore still use `checkpoint-1125`
- Programmatic crop-diff proxy, averaged across seeds `42/43/44/45`:
  - `classical_face_window`: `T2 7.0115`, `S7 7.6926`, `P1 7.8791`, `CS1 7.9939`
  - `baroque_face_candle_violin`: `T2 3.0516`, `S7 4.5234`, `P1 4.5368`, `CS1 4.5467`
  - `ink_upper_sparse`: `T2 4.0117`, `S7 8.4646`, `P1 9.5697`, `CS1 9.7279`
  - `ink_lower_tree_band`: `T2 6.6014`, `S7 9.9618`, `P1 11.3833`, `CS1 11.5872`

Analysis:
- CS1 succeeds on engineering stability:
  - the branch trains cleanly from the P1 warm start
  - the connected-component rule is genuinely active in training (`stroke_anchor_small_component_loss` stays at `0.0`)
  - the corrected baseline contract remains intact
- But CS1 fails the only research question that justified opening it:
  - the sparse upper region of `ink_wash` does not calm down relative to P1
  - the crop proxy suggests the opposite trend, with `ink_upper_sparse` movement rising from `9.5697` to `9.7279`
  - portrait-sensitive crop movement also does not improve over P1
- Therefore CS1 does not pass the gate for opening `CS2`. If connected support alone does not improve the sparse-region failure
  mode, then adding subject-safe exclusion on top of the same dense-occupancy family is unlikely to be the highest-value next full run.
- The useful conclusion is narrower and stronger:
  1. support topology matters,
  2. but a dense occupancy map is still too permissive an abstraction,
  3. so the next branch should keep primitive-object rendering while moving to a small slot-based anchor set with explicit valid-anchor
     decisions, support membership, and subject-safe exclusion at the slot level.
- In short: CS1 is a useful negative result. It closes the connected-support-per-pixel family and points directly toward a slot-based
  anchor-set renderer as the next experiment class.

## A1 Partial Run (Slot-Based Anchor-Set Renderer v1)

Method:
- A1 opens the first slot-based primitive family after `CS1` closed the dense-occupancy primitive line.
- Relative to `P1/CS1`, A1 removes dense per-pixel occupancy prediction from the main renderer decision and instead predicts
  a fixed small slot set. Each slot predicts:
  - a valid / invalid score
  - a support-masked anchor center
  - `theta`, `length`, `width`, `alpha`
  - prototype logits
- Valid slots are then assigned to support-backed regions and rasterized as elongated primitives. Subject-safe exclusion is
  already active at the slot stage in this first run.
- This branch also served as a real implementation audit. Before promotion, smoke exposed two concrete bugs:
  1. subject-exclusion priors were reusing kernel-local coordinate buffers instead of building full `24x24` grids
  2. slot audit maps used an in-place `torch.maximum` pattern that broke autograd
- Both bugs were fixed before the branch was allowed to proceed beyond smoke.
- Unlike the original long-run plan, this first A1 training attempt was intentionally stopped early once the branch showed
  clear saturation signals rather than obvious continuing gains.

Results:
- Config: `configs/janusflow_art/brushstroke/vnext/slot_based_anchor_set_renderer_v1.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/A1_slot_based_anchor_set_renderer_v1`
- Smoke evaluation output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/A1_slot_based_anchor_set_renderer_v1_smoke_eval/evaluation`
- Stable numbered checkpoints produced before stop:
  - `checkpoint-75`
  - `checkpoint-150`
  - `checkpoint-225`
- Authoritative partial-run checkpoint:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/A1_slot_based_anchor_set_renderer_v1/checkpoint-225`
- Authoritative evaluation output:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/A1_slot_based_anchor_set_renderer_v1/evaluation_checkpoint225/evaluation`
- Review notes:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/A1_slot_based_anchor_set_renderer_v1/review_notes.md`
- Long-run early-stop rationale:
  - `eval_loss @ step 150 = 1.0776`
  - `eval_loss @ step 225 = 1.0926`
  - `basis_usage_entropy_loss` flattened near `2.7726` (`ln(16)`)
  - `stroke_prototype_loss` flattened near `2.7656`
  - slot losses entered a narrow conservative band rather than continuing to separate
- Authoritative evaluation summary (`checkpoint-225`):
  - Average pair pixel absolute difference: `8.6597`
  - Average Laplacian variance baseline / tuned: `897.3197 / 1160.4334`
  - Average high-frequency energy baseline / tuned: `3577.7778 / 3893.3706`
  - Average prompt CLIP baseline / tuned: `0.3449 / 0.3446`
  - Canonical baseline hash:
    `9fb3f09e615f4bd306f76b30209d3ed605b7b299ea75410b3759cf2b0dab00f2` (`match=true`)
- Programmatic crop-diff proxy, averaged across seeds `42/43/44/45`:
  - `classical_face_window`: `P1 7.8804`, `CS1 7.9988`, `A1 7.7967`
  - `baroque_face_candle_violin`: `P1 4.5482`, `CS1 4.5580`, `A1 4.5647`
  - `impressionist_garden_foliage_cluster`: `P1 10.6501`, `CS1 10.5566`, `A1 10.6741`
  - `ukiyoe_evening_roof_outline`: `P1 10.4720`, `CS1 10.8816`, `A1 10.6251`
  - `ink_upper_sparse`: `P1 9.5726`, `CS1 9.7300`, `A1 9.1455`
  - `ink_lower_tree_band`: `P1 11.3833`, `CS1 11.5872`, `A1 11.4508`
- Operational checkpoint audit:
  - `final_checkpoint` exists but is not the authoritative source for this interrupted run
  - `checkpoint-225` is the latest stable numbered checkpoint and the valid continuation / evaluation source for A1-v1

Analysis:
- A1 is not yet a pass, but it is the first post-CS1 branch that moves the exact target proxy in the right direction.
- The key positive signal is `ink_upper_sparse`: `9.7300 -> 9.1455` relative to CS1, and also below P1. This is the first
  meaningful indication that removing dense occupancy in favor of explicit slots is addressing the right failure mode.
- A1 also improves `classical_face_window` relative to CS1, which suggests slot-level subject-safe exclusion is at least
  not worsening the portrait-sensitive crop by default.
- However, the branch remains in roughly the same global movement regime as `P1/CS1`, and the training traces show early
  collapse toward nearly uniform prototype use and overly stable slot sparsity. That is not the profile of a branch that is
  likely to reveal a new regime just by running longer with the same settings.
- Therefore the right conclusion is:
  1. do not resume A1-v1 unchanged to `1125`
  2. keep the slot-based branch as the current most promising V2 direction
  3. open a refined slot branch that explicitly pushes sharper slot ownership and less uniform prototype routing before
     spending another full long run

## A2 Early-Stop Diagnostic (Slot-Based Anchor-Set Renderer v2)

Method:
- A2 is a direct refinement of A1 rather than a new renderer family.
- Relative to A1, it attempts to break the early uniform-routing equilibrium by:
  - reducing `slot_count` from `16` to `12`
  - reducing `prototype_count` from `16` to `12`
  - raising `valid_anchor_threshold` from `0.35` to `0.45`
  - strengthening support / exclusion pressure
  - increasing slot/prototype routing-pressure losses
- The branch uses weight-only init from `A1 checkpoint-225`.
- The intended question is very specific: can a sharper slot formulation avoid the A1 early-saturation pattern before the
  project spends another long run on the slot family?

Results:
- Config: `configs/janusflow_art/brushstroke/vnext/slot_based_anchor_set_renderer_v2.yaml`
- Output: `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/A2_slot_based_anchor_set_renderer_v2`
- Smoke train: passed
- Review notes:
  `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/A2_slot_based_anchor_set_renderer_v2/review_notes.md`
- Long-run diagnostic status:
  - launched cleanly from `A1 checkpoint-225`
  - intentionally stopped at `global_step=30`
  - no numbered checkpoint was produced before stop
  - existing `final_checkpoint` in the directory is still the smoke artifact and must not be treated as a long-run result
- Early diagnostic metrics:
  - `stroke_prototype_loss` remained pinned at `2.484375` from step `1` through step `30`
  - `basis_usage_entropy_loss` remained pinned at `2.4849067` from step `1` through step `30` (approximately `ln(12)`)
  - `slot_anchor_outside_support_loss` stayed in a very narrow band near `0.31`
  - `slot_count_sparsity_loss` stayed in a very narrow band near `0.31`
  - `slot_component_conflict_loss` stayed at `0.0`
- Operational note:
  - because this run was interrupted before the first save interval, there is no authoritative numbered long-run checkpoint
  - the only reliable artifact from this branch is the smoke checkpoint plus the step-1..30 diagnostic trace in `train_log.jsonl`

Analysis:
- A2 is useful precisely because it fails quickly and cleanly.
- The branch is not broken in the software sense; it restores, launches, trains, and logs correctly.
- But it reproduces the same core slot-family problem even more starkly than A1: the model falls immediately into a
  maximum-entropy prototype regime with nearly fixed slot sparsity.
- That means the current problem is not just "weights are too weak" or "thresholds are too soft." It is more structural:
  the slot family still lacks a mechanism that makes ownership and prototype choice genuinely discrete.
- Therefore the next useful branch should not be `A3` as another scalar sweep. It should move to a more explicit slot
  ownership design, for example a slot-program / anchor-set variant with harder assignment or smaller learned anchor sets.
