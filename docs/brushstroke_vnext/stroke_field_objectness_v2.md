# Stroke-Field Objectness v2

## Why this step exists

`S4` confirmed that explicit stroke proxy supervision helps, but the branch
still reads more like organized directional texture than explicit stroke
objects.

The remaining weakness is not only "what prototype should this location look
like?" but also "should there be a stroke object here at all, and should a
small number of prototypes dominate locally?"

## Design

This step adds two objectness-oriented constraints:

1. `objectness` prediction
   - a dedicated map predicts whether a stroke object should occupy a location
   - it gates the rendered residual separately from `alpha`

2. sparse prototype routing
   - the renderer keeps only the top-k prototype weights per location and
     renormalizes them
   - this is a lightweight objectness prior: fewer local prototypes should mean
     more legible local marks

## Supervision

The proxy target pipeline now also emits `objectness_target`, derived from the
stroke occupancy proxy but sharpened into a more selective target than the raw
`alpha` map.

New optional loss:

- `stroke_objectness`

## Intended outcome

If `S4` made the branch better at learning stroke-like statistics, `S5` is
meant to test whether explicit occupancy plus sparse prototype selection can
make the renderer behave more like localized marks rather than continuous
texture modulation.
