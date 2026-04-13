# JanusFlow Brushstroke Experiment Plan

## Objective

Prioritize brushstroke fidelity, canvas/material texture, painterly edges, and local high-frequency detail without letting style tuning rewrite prompt semantics or global composition.

This plan supersedes further "bigger LoRA" work for the current phase. LoRA can remain as a comparison baseline, but new trainable capacity should be decoder-local by default.

## Architecture Direction

- Add brush modules only on the generation decoder path: after `vision_gen_dec_aligner` and before `vision_gen_dec_model`.
- Keep LLM, VAE, `vision_gen_enc_model`, and `vision_gen_dec_model` frozen in first-pass experiments.
- Use `BrushAdapter` as the main local residual path: LayerNorm, bottleneck projection, depthwise 2D convolution over decoder tokens, FiLM conditioning, and a conservative learned residual gate.
- Use `PatchBrushReferenceEncoder` only when reference conditioning is enabled. It exports coarse, mid, and fine brush tokens from reference/target images.
- Use `BrushConditionInjector` for decoder-token-only cross-attention. It must not inject into LLM tokens.
- Use optional decoder experts only inside the adapter residual branch, never as LLM MoE in this phase.

## Experiment Configs

Configs live in `configs/janusflow_art/brushstroke/`.

- `brush_adapter_v1.yaml`: decoder-side adapter only, no high-frequency auxiliary loss.
- `brush_adapter_hf_loss_v1.yaml`: adapter plus Laplacian/Sobel latent high-frequency losses.
- `brush_multiscale_ref_v1.yaml`: adapter plus patch reference encoder and multi-scale local texture tokens.
- `brush_decoder_expert_v1.yaml`: multiscale reference plus four decoder-local residual experts.
- `brush_ablation_no_ref.yaml`: texture-rich data with adapter and high-frequency losses but no reference branch.
- `brush_ablation_no_hf_loss.yaml`: multiscale reference branch without high-frequency losses.
- `brush_ablation_adapter_off.yaml`: reference conditioner only, adapter disabled.
- `brush_multiscale_ref_local_v1.yaml`: stronger adapter gate with fine-token-only, lower-scale reference injection.
- `brush_highpass_ref_fine_v1.yaml`: fine-token reference with high-pass residual routing.
- `brush_adapter_dec_aligner_v1.yaml`: no reference branch; adapter plus low-LR `vision_gen_dec_aligner` unfreeze.
- `brush_highpass_ref_expert_v1.yaml`: high-pass reference plus 2 adapter experts. Staged but paused.
- `brush_adapter_dec_aligner_aggressive_v1.yaml`: no reference branch; stronger adapter, decoder experts, and higher dec-aligner LR for visible-effect diagnosis.
- `brush_lora_visible_probe_v1.yaml`: controlled LoRA visible-effect probe with brush modules disabled and inference-scale sweep.
- `brush_local_capacity_v2_v1.yaml`: hybrid token-stage plus feature-map-stage local capacity with no reference branch.
- `brush_feature_map_only_aggressive_v1.yaml`: feature-map-dominant v2 ablation with token adapter disabled and stronger image-space residuals.
- `brush_feature_map_only_stronger_v1.yaml`: stronger feature-map-only follow-up with larger hidden width, larger gate, and higher decoder-side LR.
- `brush_spatial_hf_head_v1.yaml`: spatially gated high-frequency feature-map head with low-frequency consistency anchoring.
- `brush_spatial_hf_head_stronger_v1.yaml`: stronger follow-up for the spatial HF head.
- `brush_output_residual_head_v1.yaml`: output-space velocity residual head for late texture injection.

All configs default to local JanusFlow/SDXL-VAE paths and write to `/root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/`.

## Review Protocol

Every non-smoke run must export `evaluation/compare_cards`, `blind_review`, `summary.json`, and a run-level `review_notes.md`.

Manual review must inspect at least:

- the highest pair-difference card
- the lowest pair-difference card
- a portrait prompt
- an ink/landscape prompt
- a texture-rich prompt

A run should not proceed to longer training if high-frequency metrics improve but the image looks like noise, UI/screenshot artifacts, or major composition drift.

## Progress Log

- 2026-04-10 `brush_adapter_v1`: 450-step adapter-only run completed. The route was composition-safe and quality-preserving, but the effect was weak; visible changes were mostly slight local contrast and sharpening.
- 2026-04-10 `brush_adapter_hf_loss_v1`: 450-step adapter plus Laplacian/Sobel run completed. Laplacian and high-frequency metrics rose slightly, prompt/style CLIP dipped slightly, and manual review did not find noise collapse. The visual effect was still too subtle for expansion, so the next stage is `brush_multiscale_ref_v1`.
- 2026-04-10 evaluator baseline fix: older brushstroke evaluation bundles used the active experiment config for both baseline and tuned pipelines, so baseline images could include randomly initialized brush modules and change across experiments. Fixed evaluations now disable LoRA, conditioning, and all brush modules for the baseline pipeline.
- 2026-04-10 texture-rich schedule fix: texture-rich configs now use `num_epochs: 3.0` so 450-step screening runs are not truncated at 384 steps by the smaller subset size.
- 2026-04-10 `brush_multiscale_ref_v1`: rerun completed at 450 steps in `brush_multiscale_ref_v1_450_fixed` after the schedule fix. Fixed-baseline metrics improved slightly over E1/E2, but human review still found a conservative effect size, so the next step is `brush_ablation_no_ref`.
- 2026-04-10 review-blocker audit: E1/E2/E3 were not visually separable enough. Two logic issues were found and fixed before continuing: brush modules were applied to both conditioned and unconditioned CFG rows, which can cancel local texture signals during CFG combination, and E3 evaluation prompts did not contain `reference_style_image`, so the reference encoder was inactive during fixed-prompt review. Current brush configs now write to explicit `E1_`, `E2_`, `E3_` output roots, apply brush modules only to the conditioned CFG branch by default, and use a reference-backed prompt file for reference/injector experiments.
- 2026-04-10 `E3_brush_multiscale_ref_v1_cfgcond_refeval`: corrected 450-step run completed and evaluated with `reference_path_count: 24/24`. The reference branch is clearly active: average pixel difference increased to `11.6109`, Laplacian rose from `890.9734` to `994.1039`, and high-frequency energy rose from `3576.7521` to `3621.3108`. Manual review found visible texture/edge changes, but also pose and composition drift, so E3 should not expand directly. Next runs are E3-labeled ablations: `E3A` no reference branch, `E3B` no high-frequency loss, and `E3C` adapter off.
- 2026-04-10 `E3A_brush_ablation_no_ref_cfgcond_refeval`: 450-step no-reference ablation completed and evaluated on the same reference-backed prompt/seed set. Average pixel difference fell to `1.6026`, prompt/style CLIP slightly improved, and manual review returned to near-identical adapter-only behavior. This confirms the visible E3 effect and most of its drift come from reference conditioning rather than the adapter or high-frequency loss alone. Next run: `E3B` no high-frequency loss with reference branch enabled.
- 2026-04-10 `E3B_brush_ablation_no_hf_loss_cfgcond_refeval`: 450-step no-HF ablation completed. Manual review still found reference-driven pose/composition shifts, and many samples remain too close to baseline to count as reliable brushstroke enhancement. Metrics agree but are not the primary decision: average pixel difference stayed high at `11.8640`, prompt CLIP fell from `0.3175` to `0.3133`, and reference conditioning remained active. Do not expand E3B. Skip E3C for now and run `E3D_brush_ref_fine_local_v1_cfgcond_refeval`, which uses a stronger adapter gate but fine-token-only, lower-scale reference injection.
- 2026-04-10 `E3D_brush_ref_fine_local_v1_cfgcond_refeval`: resumed from `checkpoint-150` and completed the planned 450-step screen after fixing checkpoint resume to restore `global_step`. Evaluation exported 24 reference-backed compare cards with average pair pixel difference `9.8600`, Laplacian `890.9734 -> 990.6650`, high-frequency energy `3576.7521 -> 3631.2237`, prompt CLIP `0.3175 -> 0.3124`, and `reference_path_count: 24`. Manual review still showed composition drift (expressionist/portrait/ink/ukiyo-e).
- 2026-04-10 `E4_brush_highpass_ref_fine_v1_cfgcond_refeval`: 450-step run completed with high-pass reference routing. Metrics: avg pair diff `8.7787`, Laplacian `890.9734 -> 967.0690`, HF energy `3576.7521 -> 3598.4261`. Manual review shows drift reduced but still present.
- 2026-04-10 `E5_brush_adapter_dec_aligner_v1_cfgcond_refeval`: 450-step run completed with dec aligner unfreeze and no reference branch. Metrics: avg pair diff `1.9380`, Laplacian `890.9734 -> 889.8027`, HF energy `3576.7521 -> 3576.9898`. Manual review shows composition preserved but brushstroke changes too subtle.
- 2026-04-10 `E6_brush_highpass_ref_expert_v1_cfgcond_refeval`: config prepared but paused. It is expected to behave like another high-pass reference-conditioning variant, so do not prioritize it before the visible-effect diagnostics.
- 2026-04-10 `E7_brush_adapter_dec_aligner_aggressive_v1_cfgcond_refeval`: config prepared. This is the next brush-side diagnostic: stronger no-reference adapter, 4 adapter experts, higher dec-aligner LR, and stronger high-frequency auxiliary losses.
- 2026-04-10 `E8_brush_lora_visible_probe_v1`: config prepared. This is the next LoRA-side diagnostic: brush modules disabled, texture-rich data, reference-backed evaluation prompts, and inference-scale sweep.
- 2026-04-11 `E7_brush_adapter_dec_aligner_aggressive_v1_cfgcond_refeval`: 450-step run completed. Metrics: avg pair diff `7.6246`, Laplacian `890.9734 -> 949.7724`, HF energy `3576.7521 -> 3617.4978`, prompt CLIP `0.3175 -> 0.3145`. Manual review shows stronger visible movement than E5 without major composition drift, but still not enough decisive brushstroke gain.
- 2026-04-11 `E8_brush_lora_visible_probe_v1`: first run completed at `288` steps because `num_epochs: 3.0` with batch size `16` only yields `288` updates on this subset. Final eval loss `0.6884`. Scale sweep result: `0.15` conservative, `0.25` strongest usable point, `0.50/0.75` drift too much. The config is now corrected to `num_epochs: 5.0` for future reruns, but the current run is already sufficient to set the target effect size for v2.
- 2026-04-11 `E9_brush_local_capacity_v2_v1_cfgcond_refeval`: 450-step v2 run completed with both token and feature-map local adapters active. Metrics: avg pair diff `6.9286`, Laplacian `890.9734 -> 940.9259`, HF energy `3576.7521 -> 3646.8982`, prompt CLIP `0.3175 -> 0.3159`. Manual review shows a modest local-material gain over E7 and much safer behavior than reference-conditioning, but it still does not reach the visible strength of E8 `0.25`, and baroque cards still distort face/head structure.
- 2026-04-11 `E10_brush_feature_map_only_aggressive_v1_cfgcond_refeval`: 450-step run evaluated. Metrics: avg pair diff `6.3596`, Laplacian `890.9734 -> 928.0587`, HF energy `3576.7521 -> 3615.9148`, prompt CLIP `0.3175 -> 0.3166`. Manual review shows E10 is safer than E9 on portrait/baroque structure but weaker in visible brushstroke gain, so the feature-map-only branch looks directionally right but underpowered at this setting.
- 2026-04-11 `E11_brush_feature_map_only_stronger_v1_cfgcond_refeval`: 450-step run completed. Metrics: avg pair diff `9.4540`, Laplacian `890.9734 -> 969.8505`, HF energy `3576.7521 -> 3662.1727`, prompt CLIP `0.3175 -> 0.3168`. Manual review shows stronger visible movement than E10 on impressionist / expressionist / ink, but baroque face/head distortion returns. This suggests the feature-map-only branch has enough leverage, but scaling one residual block is no longer structure-safe.
- 2026-04-11 spatial-HF implementation: added a new `SpatialHFBrushHead` that applies a spatial gate to high-pass residuals on decoder feature maps, plus a `low_frequency_consistency` loss against the base no-brush path. Also added `gate_l1` and `gate_tv` regularizers.
- 2026-04-11 output-head implementation: added `OutputVelocityTextureHead` for late output-space texture injection when feature-map editing still leaks into structure.
- 2026-04-11 `E12_brush_spatial_hf_head_v1_cfgcond_refeval`: 450-step run completed and evaluated. Metrics: avg pair diff `6.6719`, Laplacian `887.0809 -> 940.4521`, HF energy `3574.1596 -> 3637.0988`, prompt CLIP `0.3171 -> 0.3148`. Manual review shows E12 is safer than E11 and slightly stronger than E10, but still below the target visible-effect bar and still not safe on `baroque_drama`.
- 2026-04-11 `E14_brush_output_residual_head_v1_cfgcond_refeval`: 450-step run completed and evaluated. Metrics: avg pair diff `6.5432`, Laplacian `887.6416 -> 971.5827`, HF energy `3576.2996 -> 3685.0406`, prompt CLIP `0.3168 -> 0.3158`. Manual review shows later output-space injection does not solve the baroque head/face distortion problem and does not visibly beat E12 on stable prompts.
- 2026-04-11 next direction update: do not prioritize `E13` by default. E12 and E14 together indicate that neither stronger gated feature-map editing nor later output residual injection is likely to solve the remaining problem without an explicit structural anchor or masking policy.
- 2026-04-11 `E15_brush_output_structure_anchor_v1_cfgcond_refeval`: implemented a structure-anchored output head that derives a soft protection mask from low-frequency base velocity energy and penalizes gate usage inside protected regions. Smoke passed with `protected_gate_l1_loss` active, so the next step is the full 450-step screen.
- 2026-04-11 `E15_brush_output_structure_anchor_v1_cfgcond_refeval`: 450-step run completed and evaluated. Metrics: avg pair diff `6.8839`, Laplacian `890.9616 -> 964.4624`, HF energy `3578.8147 -> 3672.8381`, prompt CLIP `0.3174 -> 0.3163`. Manual review shows the self-derived structure anchor remains active and keeps most non-portrait prompts stable, but `baroque_drama` still deforms clearly, so the next iteration needs a stronger spatial prior rather than another strength-only tweak.
- 2026-04-11 `E16_brush_output_structure_anchor_center_v1_cfgcond_refeval`: implemented as the direct E15 follow-up. It keeps the same anchored output head but adds an upper-center protection prior to bias the edit away from portrait-like subject regions. Smoke passed, so the next step is the full 450-step screen.
- 2026-04-11 `E16_brush_output_structure_anchor_center_v1_cfgcond_refeval`: 450-step run completed and evaluated. Metrics: avg pair diff `6.8442`, Laplacian `891.5082 -> 968.0089`, HF energy `3579.4624 -> 3675.5516`, prompt CLIP `0.3174 -> 0.3166`. Manual review shows the added center prior does not materially improve E15; `baroque_drama` still deforms clearly, so this output-head sub-family should stop expanding as the primary line.

## Default Commands

```bash
python train_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_adapter_v1.yaml

python eval_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_adapter_v1.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E1_brush_adapter_v1_cfgcond/final_checkpoint

python train_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_multiscale_ref_v1.yaml

python eval_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_multiscale_ref_v1.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3_brush_multiscale_ref_v1_cfgcond_refeval/final_checkpoint

python eval_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_multiscale_ref_local_v1.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3D_brush_ref_fine_local_v1_cfgcond_refeval/final_checkpoint \
  --output-dir /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E3D_brush_ref_fine_local_v1_cfgcond_refeval

python train_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_adapter_dec_aligner_aggressive_v1.yaml

python eval_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_adapter_dec_aligner_aggressive_v1.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E7_brush_adapter_dec_aligner_aggressive_v1_cfgcond_refeval/final_checkpoint

python train_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_lora_visible_probe_v1.yaml

python eval_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_lora_visible_probe_v1.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E8_brush_lora_visible_probe_v1/final_checkpoint \
  --lora-scale 0.25 \
  --output-dir /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E8_brush_lora_visible_probe_v1_scale025

python train_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_local_capacity_v2_v1.yaml

python eval_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_local_capacity_v2_v1.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E9_brush_local_capacity_v2_v1_cfgcond_refeval/final_checkpoint

python train_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_feature_map_only_aggressive_v1.yaml

python eval_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_feature_map_only_aggressive_v1.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E10_brush_feature_map_only_aggressive_v1_cfgcond_refeval/final_checkpoint

python train_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_feature_map_only_stronger_v1.yaml

python eval_janusflow_art.py \
  --config configs/janusflow_art/brushstroke/brush_feature_map_only_stronger_v1.yaml \
  --checkpoint /root/autodl-tmp/emoart_gen_runs/brushstroke_experiments/E11_brush_feature_map_only_stronger_v1_cfgcond_refeval/final_checkpoint
```

Use `--max-steps 1 --skip-final-eval` for smoke checks before any full experiment.
