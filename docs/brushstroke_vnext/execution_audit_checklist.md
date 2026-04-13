# Brushstroke VNext Execution Audit Checklist

Every experiment in the VNext family must clear this checklist before it is
treated as valid.

## Pre-run audit

1. Config audit
   - train and eval prompt files match
   - `rendered_prompt` routing is intentional for the chosen manifest
   - only the intended brush module family is enabled
   - no legacy reference/token/output head is accidentally left on
2. Checkpoint audit
   - same-architecture continuation uses `checkpoint-<step>` via `resume-from-checkpoint`
   - cross-architecture warm-start uses `training.init_from_checkpoint` or `--init-from-checkpoint`
   - `final_checkpoint` is never used for schedule-sensitive continuation
3. Baseline audit
   - evaluator baseline path disables all learned `*_head`, `*_adapter`, LoRA, and conditioning modules
   - canonical baseline hash is configured and verified in `summary.json`

## Run audit

1. Smoke
   - `--max-steps 1 --skip-final-eval`
   - import, forward, backward, and checkpoint save all succeed
2. Full train
   - checkpoint cadence is correct
   - `brush_runtime` summary matches the intended module family
3. Eval
   - fixed-baseline eval completed
   - canonical baseline hash matches
   - compare cards exported
4. Review
   - manual review covers `classical_portrait`, `baroque_drama`, `impressionist_garden`,
     `expressionist_interior`, `ukiyoe_evening`, and `ink_wash_mountain`
   - review notes identify both visual wins and structural failures

## Documentation audit

After every smoke, train, or eval run, update:

- `docs/janusflow_brushstroke_handoff.md`
- `docs/janusflow_art_plan.md`
- `docs/assumptions.md`
- the experiment output directory `review_notes.md`

Each experiment entry must record:

- `Method`
- `Results`
- `Analysis`
- baseline hash
- resume/init source
- smoke status
- code execution audit status
