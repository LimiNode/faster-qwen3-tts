"""Opt-in attention backend selection for CUDA-graph experiments."""

import os
from contextlib import nullcontext


def sdpa_kernel_context():
    """Return a context forcing the efficient SDPA kernel when requested.

    The sealed CMP runtime has an efficient FP16 kernel but no compiled Flash
    Attention kernel.  Keep this diagnostic-only because support depends on
    tensor shape, dtype, and the installed PyTorch build.
    """

    if os.environ.get("QTB_FASTER_FORCE_SDPA_EFFICIENT") != "1":
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return nullcontext()
    return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
