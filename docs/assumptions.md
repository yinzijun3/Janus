# JanusFlow-Art Assumptions

## Repository assumptions

- The current repository contains JanusFlow model definitions and demo inference code, but no mature JanusFlow training stack.
- Existing EmoArt training and evaluation infrastructure is centered on the Janus-Pro autoregressive generation path and is preserved unchanged.
- JanusFlow-Art therefore introduces a new first-class stack instead of overloading the Janus-Pro trainer.

## Model availability

- No verified local JanusFlow checkpoint snapshot was found during the initial audit.
- Default configs point to `deepseek-ai/JanusFlow-1.3B`.
- If a local mirror is later prepared, `model.model_path` should be updated in YAML rather than hardcoded in scripts.

## Prompt tag policy

- Existing JanusFlow examples mix `image_start_tag` and `image_gen_tag`.
- Runtime verification on 2026-04-08 showed that the official JanusFlow demo behavior is the safer reference:
  generation prompts should end with `image_start_tag`, and sampling then replaces that final token slot with `t_emb`.
- Using `image_gen_tag` in the new runtime produced visibly degraded outputs for both the untouched base model and tuned checkpoints:
  severe blur, material collapse, and strong color drift.
- Runtime verification on 2026-04-08 also showed that the CFG unconditional branch must keep the same prompt length and
  blank the conditional content with pad tokens plus an attention mask, mirroring the official demo.
- Earlier JanusFlow-Art checkpoints and evaluation exports produced before these two runtime fixes should be treated as
  transitional debugging artifacts rather than trustworthy baseline results.
- Runtime verification on 2026-04-08 also identified a PEFT integration hazard: if JanusFlow-Art forwards through an
  unwrapped decoder backbone instead of the wrapped `language_model`, LoRA adapters are skipped during both training and
  sampling. Forward passes must therefore call the wrapped language model directly whenever LoRA is enabled.
- Runtime verification on 2026-04-08 also identified an inference-mode hazard: constructing a pipeline with
  `training=False` is not sufficient unless the underlying `nn.Module` tree is explicitly switched to `.eval()`.
  Otherwise LoRA dropout remains active during sampling and evaluation, which can make tuned outputs look like
  weak, noisy near-copies instead of a stable adapter effect.
- Runtime verification on 2026-04-09 also identified a checkpoint-restore hazard: when a checkpoint already contains
  `language_model/`, JanusFlow-Art must load that saved PEFT adapter directly onto the raw language model rather than
  first attaching a fresh LoRA wrapper. Otherwise the runtime can end up with nested PEFT wrappers, which obscures the
  true adapter effect and makes further resume behavior unreliable.
- Runtime verification on 2026-04-09 also showed that the local `sdxl_vae.safetensors` bundle is an LDM-format VAE
  checkpoint paired with a diffusers `config.json`. Local-only JanusFlow-Art runs must therefore convert the VAE
  checkpoint keys into diffusers format before instantiating `AutoencoderKL`.
- Quality-probe review on 2026-04-10 showed that a larger attention-only LoRA can make visible artistic changes, but
  full-strength inference may rewrite composition and weaken prompt adherence. Adapter strength should be swept at
  inference before judging a checkpoint or launching another larger-LoRA run.
- Brushstroke experiments after 2026-04-10 assume that local texture control belongs closer to `vision_gen_dec_model`
  than to the LLM. First-pass brush modules therefore freeze the LLM and base generation UViT modules, train only
  decoder-local residual/conditioning modules, and use small residual gates to limit composition drift.
- The 2026-04-10 adapter-only and high-frequency-loss brush screens suggest that small decoder-local residual gates are
  composition-safe but may be too conservative for strong brushstroke gains. The next assumption to test is whether
  patch-level reference tokens add a controllable local texture signal before increasing adapter capacity or loss weight.
- Evaluation bundles created before the 2026-04-10 baseline fix should not be used for cross-experiment baseline
  comparisons. The corrected evaluator builds the baseline pipeline with JanusFlow-Art LoRA, conditioning, and brush
  modules disabled, so the baseline remains the untouched JanusFlow base under the same prompt, seed, CFG, and image size.
- 2026-04-11 baseline audit clarification: in the current corrected evaluator, the baseline branch is instantiated with
  `checkpoint_path=None`, so it does not load finetuned aligner or head weights from the experiment checkpoint. If a
  baseline still looks different from earlier `E3*` bundles, prompt-template drift (`strong_style` vs `conservative`)
  should be audited before suspecting baseline-weight drift.
- Multiscale brush-reference experiments may use the target image as the reference image during training when a manifest
  lacks `reference_style_image`; this behavior is config-gated and remains disabled for older configs.
- Brush modules should not be applied symmetrically to both conditioned and unconditioned CFG rows by default. Symmetric
  application can turn local texture changes into common-mode signals and reduce their visibility after CFG combination.
  The brush configs therefore default to `brush.condition_unconditioned: false`, while the old symmetric behavior remains
  available as a config switch.
- Reference-branch brush evaluations must use prompt JSONL records with real `reference_style_image` paths. Otherwise the
  reference encoder/injector is correctly masked to zero and the run should be interpreted as an adapter-only review.
- Training resume must restore the saved global step as well as optimizer and scheduler state. Otherwise a resume from
  `checkpoint-150` under a 450-step config would run 450 additional updates instead of stopping at the planned step 450.
- E3D/E4 objective metrics show fine-token reference conditioning is active and less globally different than the earlier
  E3/E3B reference screens, but manual review still found composition drift. High-pass routing reduces drift but does not
  fully solve it.
- E5 (adapter + low-LR `vision_gen_dec_aligner` unfreeze with no reference branch) preserved composition but produced
  near-invisible brushstroke changes at full-card scale, indicating that adapter-only texture effects remain too weak.
- E6 is intentionally paused even though its config exists, because it is still a reference cross-attention variant and is
  unlikely to resolve the current near-baseline-vs-drift failure mode by itself.
- The next diagnostic assumption is that a more aggressive no-reference local route (E7) should be tested before expanding
  reference conditioning further. If E7 remains near-baseline while a controlled LoRA probe (E8) creates visible texture,
  then the brush route likely needs a higher-leverage local generation path rather than another single adapter variant.
- E7 partially validated that assumption: stronger no-reference local capacity can move images without major composition
  drift, but the effect is still not strong enough to count as reliable brushstroke enhancement. This increases confidence
  that a single decoder-local adapter is underpowered for the project target.
- E8 further validates the direction: LoRA can produce the desired visible effect on this data/model, and the usable effect
  size is roughly around `lora_scale=0.25`. Higher scales quickly cross into semantic/structural drift. This means the
  missing ingredient is not "whether visible style change is possible", but whether a local generation-path architecture can
  reproduce that effect size without global rewrite behavior.
- E9 validates that the new v2 local-capacity path is directionally correct: adding a feature-map-stage residual block
  produces a safer local-material gain than the earlier single-adapter route, and does so without reference-conditioning.
  However, the current hybrid token + feature-map setup is still not strong enough to match E8 `0.25`, and it can still
  leak into structure on baroque/portrait-like prompts.
- E10 supports the idea that the token-stage adapter contributes to structural drift inside v2: once removed, portrait and
  baroque cards become safer. However, the first feature-map-only setting is too weak and falls back toward near-baseline
  behavior on many prompts.
- The next diagnostic assumption is therefore narrower: the project should keep the feature-map-only path, but increase its
  local capacity and decoder-side learning rate before introducing another architecture. If a stronger feature-map-only run
  still cannot approach the visible effect size of E8 `0.25`, the next change should be a new image-local architecture
  rather than a return to token adapter or reference conditioning.
- E11 partially validates and then closes that assumption: a stronger feature-map-only run can approach the needed effect
  size, but it reintroduces structural failures on baroque/portrait-like prompts. The remaining problem is therefore not
  "insufficient raw strength" alone; it is missing locality control inside the image-space branch itself.
- The next assumption to test is that the image-local route needs an explicit spatial/frequency gate or later output-space
  residual head, not just a larger single feature-map adapter.
- That next architecture is now implemented in two forms:
  `SpatialHFBrushHead` on decoder feature maps, and `OutputVelocityTextureHead` on the final velocity output.
- The training stack also now supports a low-frequency consistency loss against the frozen no-brush path, plus gate sparsity
  regularization (`gate_l1`, `gate_tv`) so local texture heads do not silently become full-frame editors.
- E12 partially validates that assumption: explicit spatial/frequency gating improves the safety-strength tradeoff over E11,
  but it still undershoots the target visible effect size and still leaves `baroque_drama` structurally unsafe.
- E14 weakens the simpler fallback assumption that moving the edit later will automatically preserve structure. Output-space
  injection still fails on the same baroque head/face case even while gate regularization remains active.
- The next working assumption should therefore be stronger: the brush route needs an explicit structure anchor or edit mask,
  not just a different residual insertion point. Candidate anchors include prompt-conditioned portrait safety masks,
  stronger low-frequency identity matching on portrait-like prompts, or residual heads that predict texture statistics
  rather than free local residual fields.
- E15 is the first direct test of that stronger assumption: it keeps output-space editing but adds a soft protection mask
  derived from low-frequency structure energy in the incoming base velocity, plus an explicit penalty on gate usage inside
  protected regions.
- E15 only partially validates that assumption. The protection mechanism remains active through training, but it still does
  not adequately protect `baroque_drama`, which suggests that self-derived low-frequency structure alone is not a strong
  enough subject prior.
- The next assumption to test is narrower: keep the output-space structure anchor, but add a stronger spatial prior on top
  of it so the head is biased away from central subject regions even when the self-derived structure mask is ambiguous.
- E16 is the first concrete test of that narrower assumption: it preserves the E15 self-derived structure mask and adds an
  upper-center protection prior to make portrait-like subject regions more expensive to edit.
- E16 does not materially improve on E15, which suggests that hand-designed output-head priors alone are not enough. The
  next useful assumption should involve a different mask source or a different local brush mechanism rather than another
  small variant of the same anchored output head.
- The next mainline improvement should therefore test data and prompt structure before another major architecture branch.
  `annotation.official.json` is now the authoritative full-dataset source, and future brushstroke experiments should
  prefer the rebuilt balanced manifests over the older `train_texture_rich_v1` subset unless an ablation explicitly
  requires the legacy data path.
- In that rebuild, `texture_score` should not contain a style-name bonus. Style balance and portrait safety are handled by
  explicit split construction, not by smuggling style priors into the texture ranking itself.
- Head-sensitive samples should be identified by lightweight text heuristics and composition cues first. External face
  detectors or captioners remain optional future upgrades rather than a prerequisite for the first rebuild.
- JanusFlow-Art data loading must respect exact pre-rendered prompt variants when a manifest provides `rendered_prompt`.
  Otherwise structure-safe and texture-push continuation manifests collapse back into the generic prompt template.
- The official full dataset is almost style-uniform, but not perfectly so: `Gongbi` has `32` images rather than `100`.
  Continuation manifests that target a fixed per-style count may therefore need explicit row repetition for rare styles.
- Official filenames may contain non-breaking spaces. Path resolution code must preserve raw internal whitespace; otherwise
  manifest validation and training startup can report false missing-image records on otherwise valid samples.
- The rebuilt data curriculum is now partially validated in practice. A full-balanced stage-1 run (D1) produces a safer
  initialization point for spatial-HF editing than the older texture-first subsets, even though it is too conservative to
  satisfy the visible-brushstroke target on its own.
- Texture-balanced continuation from that safer base (D2) increases local detail and high-frequency energy without
  reintroducing the earlier severe `baroque/portrait` structural failure. This suggests the next limiting factor is no
  longer portrait safety first, but effect strength within the safer rebuilt-data regime.
- Because D2 remains conservative rather than unstable, portrait-recovery continuation should be treated as conditional,
  not automatic. The next mainline assumption to test is that stronger local capacity on top of the D1/D2 data curriculum
  may now be viable without returning to E11-style head drift.
- In continuation training, `final_checkpoint` should not be used as the default resume source when exact schedule
  restoration matters, because `final_checkpoint/checkpoint_summary.json` stores `step: null`. Numbered checkpoints such
  as `checkpoint-450`, `checkpoint-900`, or `checkpoint-1350` correctly preserve the saved global step.
- A 2026-04-11 evaluator audit found a baseline-construction bug for the newer brush-head family: the baseline config
  disabled legacy brush modules but failed to disable `spatial_hf_head`, `output_velocity_head`, and
  `anchored_output_velocity_head`. Any evaluation bundle produced before this fix for those configs should be treated as
  having an invalid "base" branch, because the baseline still instantiated a randomly initialized local head.
- D4 validates that the rebuilt texture-balanced curriculum can tolerate moderately stronger spatial-HF editing without
  returning to catastrophic portrait/baroque collapse.
- D5 weakens the assumption that the same 512-channel spatial-HF branch still has large untapped headroom. Relaxing the
  gate and increasing texture pressure further does not buy a proportionally stronger visible result.
- D6 partially validates a more nuanced assumption: a short portrait-structure recovery continuation can slightly improve
  the D5 tradeoff instead of merely suppressing it. However, the recovered branch still remains in the same overall
  strength regime, so the main bottleneck is now local-capacity allocation rather than data curriculum alone.
- The current best checkpoint in the rebuilt-data spatial-HF family is therefore not the most aggressively pushed one,
  but the strongest branch after a short portrait-structure cleanup. This suggests the three-stage curriculum is useful,
  while the 512-channel spatial-HF head itself is approaching saturation.
- The next architecture should not keep treating brushstroke as an unconstrained
  residual. The new VNext mainline assumes that visible painterly change is
  more likely if the model predicts explicit texture statistics and those
  statistics are supervised with fixed proxy targets.
- Those proxy targets must be cheap and fully local to the repo. The first VNext
  round therefore uses fixed Sobel / blur / local-variance operators on the
  ground-truth image rather than external detectors or captioners.
- Cross-architecture warm starts must use model-weight initialization only. The
  new `training.init_from_checkpoint` path exists specifically so VNext heads
  can initialize from D6-compatible weights without inheriting optimizer state
  or a stale global step.
- Evaluation should fail fast if the baseline branch drifts. Canonical baseline
  hash verification is now treated as a required guardrail, not an optional
  debugging convenience.
- The first VNext smoke eval validated that guardrail in practice: the T1
  evaluator reproduced the expected conservative baseline hash exactly for
  `classical_portrait_oil_ref_high_renaissance`, seed `42`.
- The first full texture-statistics run (T1) suggests that constraining the edit
  through explicit local statistics is safer than the older free-residual
  families, but the first-stage full-balanced setting still reads as a modest
  texture-organization gain rather than a clearly stronger brushstroke regime.
- T2 answered that question positively: the statistics representation *can* be
  scaled safely under the stronger texture curriculum, and the reviewed
  portrait/baroque cards remain stable.
- The remaining assumption is now narrower: the current statistics route may be
  able to organize texture and material better than D6, but it may still be too
  indirect to produce the next visible brushstroke regime on its own. This
  shifts priority toward the V2 stroke-field branch before spending more time on
  portrait-recovery continuation.
- S2 partially confirmed that shift: a first stroke-field run can train cleanly
  and remain structure-safe, but the current pseudo-renderer is still too weak
  to create a clearly different painterly regime from T2. The next V2 work
  should therefore upgrade renderer expressiveness or stroke-object supervision,
  not merely rerun the same skeleton longer.
- S3 refined that diagnosis rather than overturning it. A more explicit
  directional renderer and a longer texture-balanced continuation can increase
  image movement while keeping `portrait` / `baroque` safe, but the result
  still reads more like organized directional texture than explicit stroke
  objects. This means the main bottleneck is now more architectural than purely
  configurational.
- Therefore future V2 work should prioritize renderer/object supervision over
  another simple increase in steps, residual scale, or generic high-frequency
  pressure.
- S4 strengthened that conclusion. Adding explicit stroke-field proxy
  supervision improves the branch beyond S3 without reopening
  `portrait`/`baroque` failures, which means the missing signal is not just
  "more training" or "more texture pressure." The remaining ceiling is now more
  likely inside the renderer/object model than in optimizer conservatism.
- S5 sharpened that diagnosis again. Adding objectness supervision and sparse
  top-k prototype routing makes the V2 branch visibly stronger than S4, so the
  project is no longer mainly blocked by conservative training pressure.
- However, S5 also shows that the current pseudo-renderer still behaves like a
  semi-implicit texture compositor in sparse regions. When pushed harder, it
  can invent new local forms in low-content spaces such as `ink_wash`
  backgrounds instead of only refining existing marks.
- The next V2 assumption should therefore target renderer/object composition
  quality directly: stronger elongated-mark primitives, clearer stroke occupancy
  rules, and explicit suppression of objectness activation in blank regions are
  now more important than another generic loss-weight or schedule increase.
- S6 refines that assumption further. A soft blank-region suppression loss can
  coexist with a stronger V2 branch and does not automatically collapse the
  effect size, but it is still too weak and too indirect to prevent sparse
  hallucinated structure in `ink_wash`.
- Therefore the next assumption is no longer "add some blank regularization."
  It is that sparse-region safety will require a more explicit occupancy rule:
  stroke activation should be tied to existing support evidence, isolated
  objectness blobs should be penalized more directly, or the renderer should
  rasterize only evidence-backed elongated primitives rather than generic
  semi-implicit patches.
- S7 partially validates that next step. Evidence-backed occupancy gating and a
  support-ceiling loss improve the V2 branch numerically while keeping
  `portrait` / `baroque` safe, so support-conditioned occupancy is better than
  pure objectness plus soft blank regularization alone.
- But S7 also falsifies the stronger assumption that this support gate is
  sufficient by itself. The `ink_wash` failure mode still survives, which means
  sparse-region suppression likely needs a harder occupancy mechanism or a more
  object-like primitive rule rather than another soft weighting term.
- S8 closes that family more decisively. A harder evidence-linked occupancy rule
  can still preserve portrait/baroque safety and high detail metrics, but it
  does not reliably stop the canonical `ink_wash` upper-page hallucinated tree
  form. Therefore the old occupancy-gated stroke-field family should now be
  considered mapped rather than still open for small same-family sweeps.
- A current `final_checkpoint/art_state.pt` is not sufficient to declare a run
  authoritative. Even when the final alias is current by hash, its
  `checkpoint_summary.json` can still store `step: null`. Numbered checkpoints
  remain the default authoritative source for continuation and evaluation.
- P1 partially validates the next architectural assumption: replacing the old
  semi-implicit blob compositor with a sparse support-locked primitive renderer
  is trainable and can push the branch to its strongest movement/detail metrics
  so far.
- But P1 also falsifies the simpler hope that sparse anchors and elongated
  primitives alone are enough. The canonical `ink_wash` hallucinated tree form
  still survives, and `classical_portrait` shows visible head/face drift on the
  canonical compare card. So the next renderer must tighten support topology and
  add a stronger subject-safe exclusion rule, not just keep the same support map
  and primitive set with more training pressure.
- CS1 tests that narrower follow-up directly and weakens it. Connected-support
  component gating is trainable and keeps the runtime/eval contract clean, but
  the targeted crop proxy shows no improvement over P1 on either the
  `ink_upper_sparse` region or the `classical_face_window` region.
- The next assumption should therefore be more structural: the remaining problem
  is not merely "support topology plus exclusion strength." It is that dense
  occupancy is still too permissive an abstraction. The next useful renderer
  should predict a small slot-based anchor set with explicit valid-anchor
  decisions, support membership, and subject-safe exclusion at the slot level.
- The first `A1_slot_based_anchor_set_renderer_v1` smoke validates that this
  slot-based branch is practical inside the current stack: it can warm-start
  from `CS1`, launch training, complete a backward pass, save checkpoints, and
  pass fixed-baseline smoke evaluation without breaking the canonical official-base hash.
- A1 also surfaces two explicit implementation assumptions for future slot-like
  renderers:
  1. subject-exclusion priors must build coordinate grids at the actual decoder
     feature-map size rather than reusing kernel-local renderer buffers
  2. dense audit maps should avoid in-place read-modify-write patterns such as
     `torch.maximum` on indexed slices, because those can silently break autograd
     even when the high-level branch design is otherwise correct
- For operational rigor, `final_checkpoint` must now be treated as an auditable
  alias rather than a guaranteed source of truth. S7 produced a stale
  `final_checkpoint/art_state.pt` that predated the later numbered checkpoints,
  and evaluating that alias exactly reproduced an earlier smoke-eval bundle.
  Future continuation and evaluation should default to numbered checkpoints
  unless timestamp/hash checks prove the final alias is current.
- The first partial long run of `A1_slot_based_anchor_set_renderer_v1` refines
  the slot-family assumption further. Removing dense occupancy in favor of
  explicit slots does improve the target `ink_upper_sparse` proxy relative to
  `P1` and `CS1`, so the slot-based move is not a dead end.
- However, A1-v1 also suggests that the current slot formulation saturates too
  early into nearly uniform prototype routing and overly stable slot sparsity.
  The next slot-family assumption should therefore be more specific:
  slots likely need stronger discrete ownership and stronger anti-uniform
  prototype pressure before another long run is worth the budget.
- A2 directly tested that narrower assumption with fewer slots, fewer prototypes,
  stricter valid-anchor gating, stronger exclusion, and higher routing-pressure
  losses. The branch still saturated immediately into the same
  maximum-entropy routing regime.
- Therefore the next slot-family assumption should move past scalar pressure:
  the model likely needs a more explicit ownership mechanism or harder slot /
  prototype assignment, not just stronger penalties on the existing soft slot
  formulation.
- Prompt-tag behavior is isolated in the new runtime so future upstream clarifications can be adopted centrally.

## Training objective

- JanusFlow-Art uses a true rectified-flow style training objective on SDXL-VAE latents.
- The implementation does not fall back to Janus-Pro VQ-token teacher forcing.
- Baseline training uses linear interpolation between Gaussian noise and encoded image latents and supervises the velocity target with flow matching.

## Conditioning assumptions

- Global style is injected before the language model through the encoded latent token stream and the time embedding.
- Local texture is injected later through decoder tokens before the JanusFlow decoder reconstructs the velocity field.
- This split is treated as the default “earlier global, later local” policy unless future ablations show a better placement.

## Evaluation assumptions

- Prompt-adherence proxy defaults to CLIP image-text similarity when CLIP dependencies are available.
- Style-consistency proxy defaults to CLIP similarity against a style-only prompt constructed from labels.
- Sharpness and detail proxies default to Laplacian variance and high-frequency energy.
- If optional metric dependencies are unavailable, evaluation still exports aligned images, reports, and blind-review bundles.

## Environment assumptions

- The active training environment now resolves JanusFlow, `transformers`, CUDA, and CLIP dependencies well enough to run
  E3D training and evaluation end to end on the local RTX 4090.
- Syntax-level smoke checks and config validation are still expected before touching shared entrypoints.
