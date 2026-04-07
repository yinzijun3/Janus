# Visual QA Rules

This project should treat visual quality as a hard gate, not a secondary note under metrics.

## Priority Order

1. Visual acceptability
2. Subject retention and structure stability
3. Style faithfulness
4. Automatic metrics

If a result fails at step 1 or 2, it is a failed sample even when PSNR / MAE / NLL improve.

## Hard Fail Conditions

A sample is `hard_fail` if any of the following happens:

- main subject disappears or is almost blank
- human body or face is severely deformed
- composition collapses into unusable geometry or empty framing
- important scene objects are replaced by unrelated abstractions

These samples are not eligible for promotion, style policy, or follow-up aggregate comparison.

## Soft Fail Conditions

A sample is `soft_fail` if:

- the image is technically viewable, but the structure is obviously flattened
- the scene turns into a repetitive template
- the output keeps only coarse semantic hints while losing the defining style
- border / scroll / panel bias dominates the image
- painterly detail collapses into smooth bands, icons, or decorative blocks

Soft fails should be counted as failures in experiment summaries.

## Borderline Conditions

A sample is `borderline` if:

- the image is usable at first glance
- the subject is still present
- but style faithfulness, detail richness, or composition quality are clearly weak
- or the image shows persistent frame / scroll / print-like bias

Borderline samples can support exploratory analysis, but they do not justify promoting a branch.

## Acceptable Conditions

A sample is `acceptable` only if:

- the subject is intact
- no strong deformation or collapse is visible
- the composition is coherent
- the output remains reasonably faithful to the target style family
- the image quality would not be rejected in a manual side-by-side review

## Reporting Rule

Every future experiment review should include:

- `acceptable_count`
- `borderline_count`
- `soft_fail_count`
- `hard_fail_count`
- a shortlist of hard fails with image links

Automatic metrics may still be reported, but they must never overrule hard or soft visual failures.
