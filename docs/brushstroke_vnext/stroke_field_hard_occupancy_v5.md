# Stroke-Field Hard Occupancy v5

## Motivation

`S7` showed that support-conditioned occupancy is directionally correct but not
sufficient. The branch became stronger while staying portrait-safe, yet
`ink_wash_mountain` still grew a large new tree in an upper blank region. That
failure means a soft support gate can still leak semi-implicit objectness into
areas that should remain empty.

## Change

`v5` adds a harder occupancy rule inside the renderer:

- compute the same mixed support evidence from carrier high-pass magnitude and
  normalized density
- threshold that evidence into a binary support mask
- apply a small morphological opening to remove isolated support specks
- dilate the cleaned support mask slightly
- multiply the runtime support gate by this hard support mask before it gates
  objectness

This turns support conditioning from a soft preference into a stricter
"supported regions only" rule while still preserving the existing soft gate as
  a graded weighting term inside the supported region.

## Expected effect

- preserve `portrait` / `baroque` safety
- keep roughly `S7`-level local texture strength
- reduce or eliminate the large isolated tree synthesized in the upper blank
  region of `ink_wash_mountain`

## Risk

If the hard mask is too strict, the branch may collapse back toward `S6`/`D6`
strength, especially on `ukiyoe_evening` and `impressionist_garden`. The
experiment should therefore be interpreted primarily through the tradeoff:
better sparse-region discipline without a large collapse in visible texture.
