# mlir-aie fork pins

These pins are the immutable reference for every Phase 2 task that
touches the IRON fork (~12 tasks across Waves 1b–5 of
`bio-on-xdna-phase2-plan.md`). T1.2 builds the wheel from the SHA the
outer repo's submodule currently points at; every Wave 2 IRON
primitive (T2.1–T2.7) modifies the fork and re-pins the outer-repo
submodule reference. Do not silently advance these pins; advance them
inside a Wave 2 task with clear motivation.

## Canonical pin location

A self-pinning file cannot reference its own commit SHA (chicken /
egg). The **canonical source of truth for the fork pin is the outer
repo's submodule reference**:

```
# in /home/$USER/genetics
git submodule status third_party/mlir-aie
# leading SHA = the pinned fork commit (this file's containing tree).
```

This file documents the *upstream divergence point* (immutable for
the lifetime of this fork generation) plus the metadata + rationale
needed to rebase the fork against `Xilinx/mlir-aie` later.

## Source

- Fork: https://github.com/opensensor/mlir-aie
- Fork branch: main
- Date pinned (initial T0.1): 2026-04-25

## Upstream divergence

- Upstream: https://github.com/Xilinx/mlir-aie
- Divergence point: 979629649ea679b70043dc9150a00e0f29b72aad
- Date of divergence (upstream commit date): 2026-04-23
- Divergence-point commit subject: "Replace hardcoded register values
  in AIEInsertTraceFlows fully with database lookup (#3035)"

The fork was created fresh on 2026-04-25 as a mirror of
`Xilinx/mlir-aie@main` with no opensensor-side commits prior to T0.1.
T0.1's PINS.md commit is the first fork-side advance over upstream;
every subsequent Phase 2 fork commit (Waves 2–3) advances HEAD
further. The divergence point above stays anchored for the lifetime
of this fork generation; it is the SHA a future un-fork rebase
targets when it walks `Xilinx/mlir-aie`'s history.

## Why this fork

Phase 2's IRON primitive additions (CascadeFifo, PacketFifo,
AccumFifo, SparseFifo, MemtileAggregator, Worker.fn_args extension,
diagnostics) ride on this fork until upstream PRs merge. Per
`PRDs/PRD-bio-on-xdna-phase2.md` §4.1's "default to upstreamable"
policy, every fork commit targets the upstream coding conventions;
submodule un-fork is the eventual goal once the corresponding
`Xilinx/mlir-aie` PRs land.

## Verification

Verify the divergence point (idempotent, safe to re-run):

```
cd third_party/mlir-aie
git remote add upstream https://github.com/Xilinx/mlir-aie 2>/dev/null || true
git fetch --no-tags upstream main
git merge-base HEAD upstream/main
# expected (Phase 2 lifetime): 979629649ea679b70043dc9150a00e0f29b72aad
```

Verify the outer-repo pin (whichever SHA the submodule reference
currently points to):

```
git -C /home/$USER/genetics submodule status third_party/mlir-aie
git -C /home/$USER/genetics/third_party/mlir-aie rev-parse HEAD
# the two SHAs must match.
```
