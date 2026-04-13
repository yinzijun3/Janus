# Dual-Route JanusFlow v3

## Status

Design note only. Not implemented in this round.

## Motivation

Current brush modules still operate like local edit attachments on a single
generation path. If content and brush/style remain entangled inside the same
internal route, later masking and gating can only partially recover structure.

## Proposed decomposition

- content route: identity, structure, composition, low-frequency stability
- brush route: painterly texture, material feel, stroke behavior

## Fusion policy

- allow interaction only in selected high-resolution visual layers
- preserve stronger content memory for low-frequency structure
- let the brush route focus on mid/high-frequency appearance

## Candidate losses

- content consistency
- brush/style alignment
- route disentanglement
- selective fusion regularization

## Why not implement now

This is a larger architectural bet. Before taking it, we first need to know
whether brush representation is already the main bottleneck. V1 and the V2
probe answer that question more directly.
