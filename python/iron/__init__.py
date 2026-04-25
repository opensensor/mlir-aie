# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""IRON: High-level Python API for programming AMD Ryzen AI NPUs.

Provides the primary abstractions for describing NPU designs:

- :class:`Buffer` -- named memory region shared between Workers and the Runtime
- :class:`ObjectFifo` -- synchronized dataflow channel between program components
- :class:`Worker` -- a task running on an AIE compute core
- :class:`Runtime` -- host-side orchestration of data movement and worker execution
- :class:`Program` -- top-level container that compiles a design to MLIR
- :class:`Kernel` / :class:`ExternalFunction` -- pre-compiled or C++ kernel functions
- :class:`WorkerRuntimeBarrier` -- synchronization primitive between workers and runtime
- Tensor utilities (:func:`arange`, :func:`zeros`, :func:`ones`, etc.) for NPU-accessible buffers

Wave 2 primitive reservation slots (T1.2; replaced one-by-one by later tasks):

- :class:`CascadeFifo` -- T2.1; first-class cascade-stream ObjectFifo subclass.
- :class:`PacketFifo` -- T2.2; variable-rate ObjectFifo + pktMerge / TLAST / OoO BD.
- :class:`AccumFifo` -- T2.3; FP32 accumulator inter-tile state passing.
- :class:`SparseFifo` -- T2.5; on-the-fly N:M sparsity decompression on S2MM.
- :class:`MemtileAggregator` -- T2.6; memtile-mediated fan-in helper.
"""

# T1.2: build provenance tag — substituted at build time by
# tools/actions/build-mlir-aie-fork.sh (sed-replaces __BUILD_SHA__ with the
# submodule's HEAD SHA). Consumers import this and assert it matches the
# fork SHA they expect, closing the "wheel-from-anywhere" provenance gap.
# See tests/test_iron_fork_provenance.py in the outer repo.
_BUILD_PROVENANCE = "fork:opensensor/mlir-aie@__BUILD_SHA__:T1.2"

from .buffer import Buffer
from .kernel import ExternalFunction, Kernel
from .program import Program
from .worker import Worker, WorkerRuntimeBarrier
from .runtime import Runtime
from .dataflow import ObjectFifo
from .sparse import SparseFifo
from .dtype import str_to_dtype, dtype_to_str
from aie.utils.jit import jit
from aie.utils import (
    tensor,
    ones,
    zeros,
    randint,
    rand,
    arange,
    zeros_like,
    set_tensor_class,
    get_current_device,
)


# ---------------------------------------------------------------------------
# T1.2: pre-staged Wave 2 primitive export stubs.
#
# Each stub raises NotImplementedError until its owning Wave 2 task replaces
# it with a real implementation. The reservation slots exist so that Wave 2
# tasks (T2.1 .. T2.6) executing in parallel never collide on this file:
# each task swaps in ONE class definition + one `from .module import Class`
# line at its reserved slot, never editing another task's slot.
#
# Per the Phase 2 plan's "Wave 2 fork-task serialization rule": adding the
# real class is one substitution; adding the import alongside it is the
# other. Conflicts on this file are mechanical (different lines).
# ---------------------------------------------------------------------------


class CascadeFifo:
    """T2.1 reservation slot -- CascadeFifo (cascade-stream ObjectFifo subclass).

    Will lower to ``aie.put_cascade`` / ``aie.get_cascade`` MLIR ops between
    vertically-adjacent CoreTiles. Phase 1's ``bionpu/iron_extensions/
    cascade_stream.py`` is the wrapper-level proof-of-concept being promoted.

    Raises :class:`NotImplementedError` until T2.1 lands.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CascadeFifo: T2.1 not yet landed (T1.2 reservation slot)"
        )


class PacketFifo:
    """T2.2 reservation slot -- variable-rate ObjectFifo with pktMerge / TLAST.

    Will expose packet-header routing (per-packet fan-out target), pktMerge
    N:1, finish-on-TLAST (variable-length stream termination), and
    out-of-order BD processing on memtile.

    Raises :class:`NotImplementedError` until T2.2 lands.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PacketFifo: T2.2 not yet landed (T1.2 reservation slot)"
        )


class AccumFifo:
    """T2.3 reservation slot -- FP32 accumulator inter-tile state passing.

    Persists 512-bit BM (accumulator) state across timesteps within a tile
    and across tiles via cascade-stream BM transfer. Closes G-T6.4-100.

    Raises :class:`NotImplementedError` until T2.3 lands.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AccumFifo: T2.3 not yet landed (T1.2 reservation slot)"
        )


class MemtileAggregator:
    """T2.6 reservation slot -- memtile-mediated fan-in helper.

    Encapsulates 4-into-1 memtile fan-in via S2MM ch 0..3, memtile-side
    layout reorganization (5D address generation), and fan-out via memtile
    MM2S. Promotes Phase 1's hand-rolled T5.3-memtile retrofit topology.

    Raises :class:`NotImplementedError` until T2.6 lands.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MemtileAggregator: T2.6 not yet landed (T1.2 reservation slot)"
        )


__all__ = [
    # Build-set provenance tag (T1.2).
    "_BUILD_PROVENANCE",
    # Existing IRON primitives (Phase 1 + earlier).
    "Buffer",
    "ExternalFunction",
    "Kernel",
    "Program",
    "Worker",
    "WorkerRuntimeBarrier",
    "Runtime",
    "ObjectFifo",
    "str_to_dtype",
    "dtype_to_str",
    "jit",
    "tensor",
    "ones",
    "zeros",
    "randint",
    "rand",
    "arange",
    "zeros_like",
    "set_tensor_class",
    "get_current_device",
    # Wave 2 reservation slots (T1.2 stubs, replaced by T2.1..T2.6).
    "CascadeFifo",
    "PacketFifo",
    "AccumFifo",
    "SparseFifo",
    "MemtileAggregator",
]
